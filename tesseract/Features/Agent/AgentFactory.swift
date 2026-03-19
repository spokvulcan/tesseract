//
//  AgentFactory.swift
//  tesseract
//

import Foundation

/// Extracts the multi-step agent bootstrap from DependencyContainer so the
/// container stays focused on wiring dependencies rather than orchestration.
enum AgentFactory {

    /// Bootstrap packages, discover tools/skills/context, assemble the system
    /// prompt, wire compaction, and return a fully configured `Agent`.
    @MainActor
    static func makeAgent(
        engine: AgentEngine,
        packageRegistry: PackageRegistry,
        extensionHost: ExtensionHost,
        toolRegistry: ToolRegistry,
        contextManager: ContextManager,
        selectedModelID: String
    ) -> Agent {
        let agentRoot = PathSandbox.defaultRoot

        // 1. Bootstrap packages (discover, seed data, register extensions)
        PackageBootstrap.bootstrap(
            packageRegistry: packageRegistry,
            extensionHost: extensionHost,
            agentRoot: agentRoot
        )

        // 2. Refresh extension tools after package bootstrap
        toolRegistry.refreshExtensionTools(from: extensionHost)
        let tools = toolRegistry.allTools

        // 3. Discover skills from agent root + packages (sandbox-local paths)
        let skillsDir = agentRoot.appendingPathComponent("skills")
        let cachedSkillPaths = PackageBootstrap.cachedSkillPaths(
            from: packageRegistry, agentRoot: agentRoot
        )
        let skills = SkillRegistry.discover(
            locations: [skillsDir],
            packageSkillFiles: cachedSkillPaths
        )

        // 4. Load context files (AGENTS.md, CLAUDE.md, APPEND_SYSTEM.md, etc.)
        let contextLoader = ContextLoader(agentRoot: agentRoot)
        let loadedContext = contextLoader.load(
            packageContextFiles: packageRegistry.allContextFilePaths,
            packagePromptAppends: packageRegistry.allPromptAppendPaths,
            packageSystemOverrides: []
        )

        // 5. Assemble system prompt with full context
        let systemPrompt = SystemPromptAssembler.assemble(
            defaultPrompt: SystemPromptAssembler.defaultCorePrompt,
            loadedContext: loadedContext,
            skills: skills,
            tools: tools,
            dateTime: Date(),
            agentRoot: agentRoot.path
        )

        // 6. Build compaction transform
        let generateParameters = AgentGenerateParameters.forModel(selectedModelID)

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

        // 7. Create agent config
        let config = AgentLoopConfig(
            model: AgentModelRef(id: selectedModelID),
            convertToLlm: { msgs in msgs.compactMap { $0.toLLMMessage() } },
            contextTransform: compactionTransform,
            getSteeringMessages: nil,
            getFollowUpMessages: nil
        )

        // 8. Create Agent
        return Agent(
            config: config,
            systemPrompt: systemPrompt,
            tools: tools,
            generate: { systemPrompt, messages, tools, _ in
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
        )
    }
}
