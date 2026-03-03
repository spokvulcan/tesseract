import Foundation

/// Legacy facade that delegates to ``SystemPromptAssembler``.
///
/// The old three-tier prompt logic (minimal/condensed/default) is removed.
/// The Pi-style assembler now owns prompt construction. This wrapper keeps
/// existing callers (``AgentCoordinator``, ``BenchmarkRunner``) compiling
/// until they are rewritten in later tasks.
///
/// - No memory injection in core prompt (memories are read via the `read` tool)
/// - No model-specific tiers (the assembler uses a single generic core prompt)
/// - Package APPEND_SYSTEM content is appended when available
/// - Skills are listed when a `read` tool is available
enum SystemPromptBuilder {

    /// Assembles the system prompt using the new Pi-style assembler.
    ///
    /// Parameters preserved for backward compatibility with existing callers.
    /// `memories` and `conversationSummaries` are intentionally ignored —
    /// the new architecture reads memories via the `read` tool, and conversation
    /// summaries are handled by compaction.
    static func build(
        modelID: String? = nil,
        instructions: String? = nil,
        memories: [String]? = nil,
        conversationSummaries: [String]? = nil,
        voiceMode: Bool = false
    ) -> String {
        // Use the Pi-style assembler with empty context (no packages/skills wired yet).
        // The full wiring happens in Task 6.5 (DependencyContainer).
        SystemPromptAssembler.assemble(
            defaultPrompt: SystemPromptAssembler.defaultCorePrompt,
            loadedContext: ContextLoader.LoadedContext(
                contextFiles: [],
                systemOverride: nil,
                systemAppend: nil
            ),
            skills: [],
            tools: [],
            dateTime: Date(),
            agentRoot: PathSandbox.defaultRoot.path
        )
    }
}
