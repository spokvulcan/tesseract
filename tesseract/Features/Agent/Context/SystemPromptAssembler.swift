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
        You help users by reading files, editing files, writing files, and using other tools provided by the current package or project.

        Available tools:
        - read: Read file contents
        - write: Create or overwrite files
        - edit: Make surgical edits to files (find exact text and replace)
        - list: List files and directories

        In addition to the tools above, you may have access to other custom tools depending on the current package or project.

        Guidelines:
        - Use list to discover files and directories
        - Use read to examine files before editing and writing
        - Use edit for precise changes (old text must match exactly)
        - Use write only if read the file first and it is empty or not exists, otherwise use edit.
        - Be concise in your responses
        - Show file paths clearly when working with files
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
            let skillsXML = SkillRegistry.formatForPrompt(skills)
            if !skillsXML.isEmpty {
                sections.append(skillsXML)
            }
        }

        // 5. Date/time
        sections.append("Current date and time: \(dateFormatter.string(from: dateTime))")

        // 6. Working directory
        sections.append("Current working directory: \(agentRoot)")

        return sections.joined(separator: "\n\n")
    }
}
