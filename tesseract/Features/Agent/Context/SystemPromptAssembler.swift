import Foundation

// MARK: - SystemPromptAssembler

/// Pi-style system prompt assembly. Combines a base prompt with context files,
/// skills, overrides, and runtime metadata into the final system prompt string.
///
/// Used by `Agent` and `BenchmarkRunner` for system prompt construction.
nonisolated enum SystemPromptAssembler: Sendable {

    // MARK: - Default Prompt

    static let defaultCorePrompt = """
        You are an expert local assistant operating inside Tesseract, a tool-calling agent harness.
        You help users by reading, editing, and writing files, and by using other tools provided by the current package or project.

        Guidelines:
        - Use ls to discover project structure
        - Always read a file before editing it
        - Use edit for targeted changes — old_text must match the file exactly
        - Use write only for creating new files or complete rewrites
        - Be concise in responses
        - Reference file paths clearly
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
    /// 3. Context files as "# Project Context" sections
    /// 4. Skills listing (only when a `read` tool is available)
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

        // 3. Context files
        for (path, content) in loadedContext.contextFiles {
            let filename = (path as NSString).lastPathComponent
            sections.append("# Project Context: \(filename)\n\n\(content)")
        }

        // 4. Skills listing (only if the model has a read tool to load them)
        let hasReadTool = tools.contains { $0.name == "read" }
        if hasReadTool {
            let skillsListing = SkillRegistry.formatForPrompt(skills)
            if !skillsListing.isEmpty {
                sections.append(skillsListing)
            }
        }

        // 5. Date/time
        sections.append("Current date and time: \(dateFormatter.string(from: dateTime))")

        // 6. Working directory
        sections.append("Current working directory: \(agentRoot)")

        return sections.joined(separator: "\n\n")
    }
}
