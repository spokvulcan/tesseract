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

    /// A short web-orientation block, injected only when the turn carries browser
    /// tools (ADR-0028). It gives a small model the search → read → interact order
    /// up front — so it doesn't flail — and the rule to cite pages, not snippets.
    static let webOrientation = """
        Web access: when you need current or external information, work through the browser tools in order —
        - browser.search to find candidate pages (returns titles, URLs, and snippets),
        - browser.fetch (or navigate + read_page) to read a page's actual content,
        - navigate / page_map / click / type to interact with pages that are gated or dynamic.
        Search snippets are navigation hints, not facts: open and read a page before relying on it, and cite the pages you read.
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
    /// `facts` must be derived from the *resolved* tool set (the one the agent
    /// can actually call) via `ActiveToolSet.promptFacts(for:)` — never from
    /// the raw registry — so the orientation sections and the callable set
    /// cannot diverge (ADR-0048).
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
        facts: PromptToolFacts,
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
        if facts.hasSkillTool {
            let skillsListing = SkillRegistry.formatForPrompt(skills)
            if !skillsListing.isEmpty {
                sections.append(skillsListing)
            }
        }

        // 3.5 Web orientation — only when the turn carries browser tools
        // (ADR-0028). The fact rides `browserToolNames` membership over the
        // resolved set, so "is a browser tool" means exactly one thing: a user
        // MCP server that happens to sanitize into the `browser` namespace
        // can't trip the block the way a bare `browser.` prefix check would.
        if facts.carriesBrowserTools {
            sections.append(webOrientation)
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

    /// Convenience over a concrete tool array — derives the facts from the
    /// given tools. Callers on the agent path must pass the *resolved* set;
    /// `BenchmarkRunner` and tests pass the exact set their run carries.
    static func assemble(
        defaultPrompt: String = defaultCorePrompt,
        loadedContext: ContextLoader.LoadedContext,
        skills: [SkillMetadata],
        tools: [AgentToolDefinition],
        dateTime: Date = Date(),
        agentRoot: String
    ) -> String {
        assemble(
            defaultPrompt: defaultPrompt,
            loadedContext: loadedContext,
            skills: skills,
            facts: ActiveToolSet.promptFacts(for: tools),
            dateTime: dateTime,
            agentRoot: agentRoot
        )
    }
}
