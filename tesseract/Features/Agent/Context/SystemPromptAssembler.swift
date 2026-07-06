import Foundation

// MARK: - SystemPromptAssembler

/// Pi-style system prompt assembly. Combines a base prompt with context files,
/// skills, overrides, and runtime metadata into the final system prompt string.
///
/// Used by `Agent` and `BenchmarkRunner` for system prompt construction.
nonisolated enum SystemPromptAssembler: Sendable {

    // MARK: - Default Prompt

    static let defaultCorePrompt = """
        You are an assistant running fully on this Mac inside Tesseract. You act through the tools available in this session.

        Tool rules:
        - Discover files with ls before assuming paths
        - Read a file before editing it
        - edit replaces exact text — old_text must match the file exactly
        - write creates files; pass overwrite: true only to replace a file you have already read
        - Keep replies brief; refer to files by their paths
        """

    // MARK: - Private

    /// Shared date formatter (thread-safe after initialization).
    private static let dateFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateStyle = .full
        f.timeStyle = .short
        return f
    }()

    // MARK: - Assembly

    /// Assemble the complete system prompt from all sources.
    ///
    /// Assembly order (spec J.3):
    /// 1. Base prompt (SYSTEM.md override or default)
    /// 2. APPEND_SYSTEM.md content
    /// 3. Skills listing (only when a `use_skill` tool is available)
    /// 4. Context files as "# Project Context" sections
    /// 5. Date/time
    /// 6. Working directory
    static func assemble(
        defaultPrompt: String = defaultCorePrompt,
        loadedContext: ContextLoader.LoadedContext,
        skills: [SkillMetadata],
        tools: [AgentToolDefinition],
        dateTime: Date = Date(),
        agentRoot: String
    ) -> String {
        var sections: [String] = []

        // 1. Base prompt
        sections.append(loadedContext.systemOverride ?? defaultPrompt)

        // 2. Append system
        if let append = loadedContext.systemAppend {
            sections.append(append)
        }

        // 3. Skills listing (only if the model has the use_skill tool)
        let hasSkillTool = tools.contains { $0.name == skillToolName }
        if hasSkillTool {
            let skillsListing = SkillRegistry.formatForPrompt(skills)
            if !skillsListing.isEmpty {
                sections.append(skillsListing)
            }
        }

        // 4. Context files
        for (path, content) in loadedContext.contextFiles {
            let filename = (path as NSString).lastPathComponent
            sections.append("# Project Context: \(filename)\n\n\(content)")
        }

        // 5. Date/time
        sections.append("Current date and time: \(dateFormatter.string(from: dateTime))")

        // 6. Working directory
        sections.append("Current working directory: \(agentRoot)")

        return sections.joined(separator: "\n\n")
    }
}
