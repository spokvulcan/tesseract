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
    ///
    /// `gating` names the consumer this agent serves (ADR-0048): the tool set
    /// is resolved through `ActiveToolSet` once here, and the system prompt is
    /// assembled from that *resolved* set's facts — never from the raw
    /// registry — so the prompt cannot orient the model toward tools the
    /// consumer will not carry.
    @MainActor
    static func makeAgent(
        inferenceService: ServerInferenceService,
        packageRegistry: PackageRegistry,
        extensionHost: ExtensionHost,
        toolRegistry: ToolRegistry,
        contextManager: ContextManager,
        settingsManager: SettingsManager,
        gating: ToolGating,
        mcpToolsExtension: MCPToolsExtension? = nil
    ) -> Agent {
        let agentRoot = PathSandbox.defaultRoot

        // 1. Bootstrap packages (discover, seed data, register extensions)
        PackageBootstrap.bootstrap(
            packageRegistry: packageRegistry,
            extensionHost: extensionHost,
            agentRoot: agentRoot,
            settingsManager: settingsManager,
            mcpToolsExtension: mcpToolsExtension
        )

        // 2. Refresh extension tools after package bootstrap
        toolRegistry.refreshExtensionTools(from: extensionHost)

        // 3. Discover skills from agent root + packages (sandbox-local paths) —
        // the same discovery the Command Palette and Skill Pill row run, so the
        // model's use_skill listing can never drift from the user surfaces.
        let skills = PackageBootstrap.discoverAgentSkills(packageRegistry: packageRegistry)

        // 3b. Register the use_skill tool (needs discovered skills)
        let skillTool = createSkillTool(skills: skills)
        toolRegistry.appendBuiltInTool(skillTool)

        // 4. Resolve the live tool set for this consumer (now includes
        // use_skill). One resolve feeds both the agent's callable set and the
        // prompt facts below — the ADR-0048 invariant.
        let tools = ActiveToolSet.resolve(from: toolRegistry.allTools, gating: gating)
        let facts = ActiveToolSet.promptFacts(for: tools)

        // 5. Load context files (AGENTS.md, CLAUDE.md, APPEND_SYSTEM.md, etc.)
        let contextLoader = ContextLoader(agentRoot: agentRoot)
        let loadedContext = contextLoader.load(
            packageContextFiles: packageRegistry.allContextFilePaths,
            packagePromptAppends: packageRegistry.allPromptAppendPaths,
            packageSystemOverrides: []
        )

        // 6. Assemble system prompt with full context, from the resolved facts
        let systemPrompt = SystemPromptAssembler.assemble(
            defaultPrompt: SystemPromptAssembler.defaultCorePrompt,
            loadedContext: loadedContext,
            skills: skills,
            facts: facts,
            dateTime: Date(),
            agentRoot: agentRoot.path
        )

        // 7. Build compaction transform. The provider live-reads settings so a
        // runtime change to the selected model takes effect on the very next
        // inference call without rebuilding the agent.
        let parametersProvider: @MainActor @Sendable () -> AgentGenerateParameters = {
            [settingsManager] in
            settingsManager.makeAgentGenerateParameters()
        }

        let compactionTransform = makeCompactionTransform(
            contextManager: contextManager,
            contextWindow: 262_144,
            summarize: makeSummarizeClosure(
                inferenceService: inferenceService,
                parametersProvider: parametersProvider
            )
        )

        // 8. Create agent config
        let config = AgentLoopConfig(
            model: AgentModelRef(id: settingsManager.selectedAgentModelID),
            convertToLlm: { msgs in msgs.compactMap { $0.toLLMMessage() } },
            contextTransform: compactionTransform,
            getSteeringMessages: nil,
            getFollowUpMessages: nil
        )

        // 9. Create Agent, seeded with the resolved set, and wire the prompt
        // reassembler: when a later resolve changes the prompt facts (Web
        // Access flip, browser tools landing after a late MCP connect), the
        // agent rebuilds its prompt from the same captured assembly inputs.
        let agent = Agent(
            config: config,
            systemPrompt: systemPrompt,
            tools: tools,
            generate: makeServerInferenceGenerateClosure(
                inferenceService: inferenceService,
                parametersProvider: parametersProvider
            )
        )
        agent.setSystemPromptReassembler(initialFacts: facts) { facts in
            SystemPromptAssembler.assemble(
                defaultPrompt: SystemPromptAssembler.defaultCorePrompt,
                loadedContext: loadedContext,
                skills: skills,
                facts: facts,
                dateTime: Date(),
                agentRoot: agentRoot.path
            )
        }
        return agent
    }
}
