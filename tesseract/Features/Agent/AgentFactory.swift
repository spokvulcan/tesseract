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
        inferenceService: ServerInferenceService,
        fallbackEngine: any LegacyInternalInferenceEngine,
        packageRegistry: PackageRegistry,
        extensionHost: ExtensionHost,
        toolRegistry: ToolRegistry,
        contextManager: ContextManager,
        selectedModelID: String,
        settingsManager: SettingsManager,
        rollbackEnabled: @escaping @MainActor @Sendable () -> Bool
    ) -> Agent {
        let agentRoot = PathSandbox.defaultRoot

        // 1. Bootstrap packages (discover, seed data, register extensions)
        PackageBootstrap.bootstrap(
            packageRegistry: packageRegistry,
            extensionHost: extensionHost,
            agentRoot: agentRoot,
            settingsManager: settingsManager
        )

        // 2. Refresh extension tools after package bootstrap
        toolRegistry.refreshExtensionTools(from: extensionHost)

        // 3. Discover skills from agent root + packages (sandbox-local paths)
        let skillsDir = agentRoot.appendingPathComponent("skills")
        let cachedSkillPaths = PackageBootstrap.cachedSkillPaths(
            from: packageRegistry, agentRoot: agentRoot
        )
        let skills = SkillRegistry.discover(
            locations: [skillsDir],
            packageSkillFiles: cachedSkillPaths
        )

        // 3b. Register the use_skill tool (needs discovered skills)
        let skillTool = createSkillTool(skills: skills)
        toolRegistry.appendBuiltInTool(skillTool)

        // 4. Get all tools (now includes use_skill)
        let tools = toolRegistry.allTools

        // 5. Load context files (AGENTS.md, CLAUDE.md, APPEND_SYSTEM.md, etc.)
        let contextLoader = ContextLoader(agentRoot: agentRoot)
        let loadedContext = contextLoader.load(
            packageContextFiles: packageRegistry.allContextFilePaths,
            packagePromptAppends: packageRegistry.allPromptAppendPaths,
            packageSystemOverrides: []
        )

        // 6. Assemble system prompt with full context
        let systemPrompt = SystemPromptAssembler.assemble(
            defaultPrompt: SystemPromptAssembler.defaultCorePrompt,
            loadedContext: loadedContext,
            skills: skills,
            tools: tools,
            dateTime: Date(),
            agentRoot: agentRoot.path
        )

        // 7. Build compaction transform
        let triAttention = settingsManager.makeTriAttentionConfig()
        let parametersProvider: @MainActor @Sendable () -> AgentGenerateParameters = {
            [selectedModelID, triAttention] in
            var parameters = AgentGenerateParameters.forModel(selectedModelID)
            parameters.triAttention = triAttention
            return parameters
        }

        let compactionTransform = makeCompactionTransform(
            contextManager: contextManager,
            contextWindow: 262_144,
            summarize: makeSummarizeClosure(
                inferenceService: inferenceService,
                fallbackEngine: fallbackEngine,
                rollbackEnabled: rollbackEnabled,
                parametersProvider: parametersProvider
            )
        )

        // 8. Create agent config
        let config = AgentLoopConfig(
            model: AgentModelRef(id: selectedModelID),
            convertToLlm: { msgs in msgs.compactMap { $0.toLLMMessage() } },
            contextTransform: compactionTransform,
            getSteeringMessages: nil,
            getFollowUpMessages: nil
        )

        // 9. Create Agent
        return Agent(
            config: config,
            systemPrompt: systemPrompt,
            tools: tools,
            generate: makeServerInferenceGenerateClosure(
                inferenceService: inferenceService,
                fallbackEngine: fallbackEngine,
                rollbackEnabled: rollbackEnabled,
                parametersProvider: parametersProvider
            )
        )
    }
}
