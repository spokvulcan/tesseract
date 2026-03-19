//
//  BackgroundAgentFactory.swift
//  tesseract
//

import Foundation

/// Creates isolated `Agent` instances for background scheduled task execution.
///
/// Replicates `AgentFactory.makeAgent` steps 1–8 with full bootstrap parity.
/// Each call to `createAgent` produces a fresh `Agent` with its own compaction
/// state but shares the same `AgentEngine` and `ToolRegistry`.
@MainActor
final class BackgroundAgentFactory {

    private let agentEngine: AgentEngine
    private let sessionStore: BackgroundSessionStore
    private let settingsManager: SettingsManager
    private let ensureModelLoaded: @MainActor () async throws -> Void

    // Cached at init — after running the same bootstrap as AgentFactory steps 1-4
    private let cachedSkills: [SkillMetadata]
    private let cachedLoadedContext: ContextLoader.LoadedContext
    private let cachedTools: [AgentToolDefinition]
    private let agentRoot: URL

    // MARK: - Init

    init(
        agentEngine: AgentEngine,
        toolRegistry: ToolRegistry,
        extensionHost: ExtensionHost,
        sessionStore: BackgroundSessionStore,
        packageRegistry: PackageRegistry,
        settingsManager: SettingsManager,
        ensureModelLoaded: @escaping @MainActor () async throws -> Void
    ) {
        self.agentEngine = agentEngine
        self.sessionStore = sessionStore
        self.settingsManager = settingsManager
        self.ensureModelLoaded = ensureModelLoaded
        self.agentRoot = PathSandbox.defaultRoot

        // 1. Bootstrap packages (same as AgentFactory step 1) — idempotent
        PackageBootstrap.bootstrap(
            packageRegistry: packageRegistry,
            extensionHost: extensionHost,
            agentRoot: agentRoot
        )

        // 2. Refresh extension tools after package bootstrap (AgentFactory step 2)
        toolRegistry.refreshExtensionTools(from: extensionHost)

        // Cache tools snapshot — after bootstrap + refresh
        self.cachedTools = toolRegistry.allTools

        // Cache skills (AgentFactory step 3)
        let skillsDir = agentRoot.appendingPathComponent("skills")
        let cachedSkillPaths = PackageBootstrap.cachedSkillPaths(
            from: packageRegistry, agentRoot: agentRoot
        )
        self.cachedSkills = SkillRegistry.discover(
            locations: [skillsDir],
            packageSkillFiles: cachedSkillPaths
        )

        // Cache context files (AgentFactory step 4)
        let contextLoader = ContextLoader(agentRoot: agentRoot)
        self.cachedLoadedContext = contextLoader.load(
            packageContextFiles: packageRegistry.allContextFilePaths,
            packagePromptAppends: packageRegistry.allPromptAppendPaths,
            packageSystemOverrides: []
        )
    }

    // MARK: - Agent Creation

    /// Create a fully configured `Agent` for the given scheduled task,
    /// returning it alongside the loaded session for later persistence.
    private func createAgent(for task: ScheduledTask) async -> (Agent, BackgroundSession) {
        let selectedModelID = settingsManager.selectedAgentModelID

        // 1. Load or create session
        let session = await sessionStore.loadOrCreate(
            sessionId: task.sessionId,
            taskId: task.id,
            taskName: task.name,
            sessionType: .cron
        )

        // 2. Assemble system prompt with cached context + fresh date/time + task preamble
        let basePrompt = SystemPromptAssembler.assemble(
            defaultPrompt: SystemPromptAssembler.defaultCorePrompt,
            loadedContext: cachedLoadedContext,
            skills: cachedSkills,
            tools: cachedTools,
            dateTime: Date(),
            agentRoot: agentRoot.path
        )
        let systemPrompt = basePrompt + "\n\n"
            + SystemPromptAssembler.backgroundPreamble(for: task)

        // 3. Build compaction transform — fresh ContextManager per run
        let generateParameters = AgentGenerateParameters.forModel(selectedModelID)
        let contextManager = ContextManager(settings: .standard)
        let engine = agentEngine

        let compactionTransform = makeCompactionTransform(
            contextManager: contextManager,
            contextWindow: 120_000,
            summarize: { prompt in
                let (stream, continuation) = AsyncThrowingStream.makeStream(of: AgentGeneration.self)
                let task = Task { @MainActor in
                    do {
                        let s = try engine.generate(prompt: prompt, parameters: generateParameters)
                        for try await gen in s {
                            continuation.yield(gen)
                        }
                        continuation.finish()
                    } catch {
                        continuation.finish(throwing: error)
                    }
                }
                continuation.onTermination = { _ in task.cancel() }
                var result = ""
                for try await gen in stream {
                    if case .text(let chunk) = gen {
                        result += chunk
                    }
                }
                return result
            }
        )

        // 4. Build generate closure — same MainActor-hopping pattern as AgentFactory
        let generate: LLMGenerateFunction = { systemPrompt, messages, tools, _ in
            let (stream, continuation) = AsyncThrowingStream.makeStream(of: AgentGeneration.self)
            let task = Task { @MainActor in
                do {
                    let engineStream = try engine.generate(
                        systemPrompt: systemPrompt,
                        messages: messages,
                        tools: tools,
                        parameters: generateParameters
                    )
                    for try await generation in engineStream {
                        continuation.yield(generation)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
            return stream
        }

        // 5. Create agent config — no steering/followUp queues for background
        let config = AgentLoopConfig(
            model: AgentModelRef(id: selectedModelID),
            convertToLlm: { msgs in msgs.compactMap { $0.toLLMMessage() } },
            contextTransform: compactionTransform,
            getSteeringMessages: nil,
            getFollowUpMessages: nil
        )

        // 6. Instantiate agent
        let agent = Agent(
            config: config,
            systemPrompt: systemPrompt,
            tools: cachedTools,
            generate: generate
        )

        // 7. Restore messages if session has prior history
        if !session.messages.isEmpty {
            do {
                let restored = try SyncMessageCodec.decodeAll(session.messages)
                agent.loadMessages(restored)
            } catch {
                Log.agent.error("Failed to restore background session messages for task \(task.id): \(error)")
            }
        }

        return (agent, session)
    }

    // MARK: - Execute & Persist

    /// Execute a scheduled task with a background agent and persist the transcript.
    /// This replaces the stub `executeTask` closure in `DependencyContainer`.
    func executeAndPersist(task: ScheduledTask) async -> TaskRunResult {
        // Ensure the LLM is loaded before attempting generation
        do {
            try await ensureModelLoaded()
        } catch {
            let message = "Model not available: \(error.localizedDescription)"
            Log.agent.error("Background task '\(task.name)' skipped: \(message)")
            return .error(message: message)
        }

        let (agent, session) = await createAgent(for: task)

        // Record message count before prompting so we can detect new output
        let messageCountBefore = agent.context.messages.count

        // Prompt the agent with the task instruction
        agent.prompt(UserMessage.create(task.prompt))
        await agent.waitForIdle()

        // Extract result — only consider messages produced during THIS run.
        // Require non-empty content: a tool-call-only assistant turn (empty content)
        // indicates the model failed mid-turn before producing a final summary.
        let newMessages = agent.context.messages.dropFirst(messageCountBefore)
        let result: TaskRunResult
        if let lastAssistant = newMessages.last(where: {
            guard let a = $0 as? AssistantMessage else { return false }
            return !a.content.isEmpty
        }) as? AssistantMessage {
            let summary = String(lastAssistant.content.prefix(500))
            result = .success(summary: summary)
        } else {
            // No new assistant text — generation failed or produced no output
            result = .error(message: "No response generated")
        }

        // Persist full session transcript
        var updated = session
        do {
            updated.messages = try SyncMessageCodec.encodeAll(
                agent.context.messages.map { $0 as any AgentMessageProtocol }
            )
        } catch {
            Log.agent.error("Failed to encode background session messages for task \(task.id): \(error)")
        }
        updated.lastRunAt = Date()
        await sessionStore.save(updated)

        Log.agent.info("Background task '\(task.name)' completed: \(result.displaySummary)")
        return result
    }
}
