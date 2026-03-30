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
    private let arbiter: InferenceArbiter

    // Cached at init — after running the same bootstrap as AgentFactory steps 1-4
    private let cachedSkills: [SkillMetadata]
    private let cachedLoadedContext: ContextLoader.LoadedContext
    private let cachedTools: [AgentToolDefinition]
    private let agentRoot: URL

    /// In-flight agent and session during `executeAndPersist`. Set at execution
    /// start, cleared on completion. Used by `persistInFlightSession()` to flush
    /// the accumulated transcript on shutdown before the app terminates.
    private var inFlightAgent: Agent?
    private var inFlightSession: BackgroundSession?

    // MARK: - Init

    init(
        agentEngine: AgentEngine,
        toolRegistry: ToolRegistry,
        extensionHost: ExtensionHost,
        sessionStore: BackgroundSessionStore,
        packageRegistry: PackageRegistry,
        settingsManager: SettingsManager,
        arbiter: InferenceArbiter
    ) {
        self.agentEngine = agentEngine
        self.sessionStore = sessionStore
        self.settingsManager = settingsManager
        self.arbiter = arbiter
        self.agentRoot = PathSandbox.defaultRoot

        // 1. Bootstrap packages (same as AgentFactory step 1) — idempotent
        PackageBootstrap.bootstrap(
            packageRegistry: packageRegistry,
            extensionHost: extensionHost,
            agentRoot: agentRoot,
            settingsManager: settingsManager
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
    private func createAgent(for task: ScheduledTask, sessionType: SessionType = .cron) async -> (Agent, BackgroundSession) {
        let selectedModelID = settingsManager.selectedAgentModelID

        // 1. Load or create session — populate metadata from task before save
        var session = await sessionStore.loadOrCreate(sessionId: task.sessionId)
        session.taskId = task.id
        session.displayName = task.name
        session.sessionType = sessionType

        // 2. Assemble system prompt with cached context + fresh date/time + task preamble
        let basePrompt = SystemPromptAssembler.assemble(
            defaultPrompt: SystemPromptAssembler.defaultCorePrompt,
            loadedContext: cachedLoadedContext,
            skills: cachedSkills,
            tools: cachedTools,
            dateTime: Date(),
            agentRoot: agentRoot.path
        )
        let preamble: String
        switch sessionType {
        case .heartbeat:
            preamble = SystemPromptAssembler.heartbeatPreamble()
        case .cron:
            preamble = SystemPromptAssembler.backgroundPreamble(for: task)
        }
        let systemPrompt = basePrompt + "\n\n" + preamble

        // 3. Build compaction transform — fresh ContextManager per run
        let generateParameters = AgentGenerateParameters.forModel(selectedModelID)
        let contextManager = ContextManager(settings: .standard)
        let engine = agentEngine

        let compactionTransform = makeCompactionTransform(
            contextManager: contextManager,
            contextWindow: 120_000,
            summarize: makeSummarizeClosure(
                engine: engine,
                parametersProvider: { generateParameters }
            )
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
    func executeAndPersist(task: ScheduledTask, sessionType: SessionType = .cron) async -> TaskRunResult {
        defer {
            inFlightAgent = nil
            inFlightSession = nil
        }
        do {
            // Deferred lease: waits until no foreground work is active, then acquires.
            // Re-waits if foreground work arrives while waiting, so background tasks
            // never wedge between consecutive user turns.
            return try await arbiter.withDeferredGPU(.llm) {
                let (agent, session) = await self.createAgent(for: task, sessionType: sessionType)

                // Track in-flight state for shutdown persistence
                self.inFlightAgent = agent
                self.inFlightSession = session

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
                await self.sessionStore.save(updated)

                Log.agent.info("Background task '\(task.name)' completed: \(result.displaySummary)")
                return result
            }
        } catch {
            let message = "Model not available: \(error.localizedDescription)"
            Log.agent.error("Background task '\(task.name)' skipped: \(message)")
            return .error(message: message)
        }
    }

    // MARK: - Shutdown Persistence

    /// Flush the in-flight background session transcript to disk.
    /// Called from the shutdown path to preserve accumulated messages
    /// before the app terminates mid-execution.
    func persistInFlightSession() async {
        guard let agent = inFlightAgent, var session = inFlightSession else { return }
        do {
            session.messages = try SyncMessageCodec.encodeAll(
                agent.context.messages.map { $0 as any AgentMessageProtocol }
            )
            session.lastRunAt = Date()
            await sessionStore.save(session)
            Log.agent.info("Persisted in-flight background session '\(session.displayName ?? "unknown")'")
        } catch {
            Log.agent.error("Failed to persist in-flight background session: \(error)")
        }
    }
}
