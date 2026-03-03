import Foundation

// MARK: - ContextLoader

/// Discovers and loads context files (`AGENTS.md`, `CLAUDE.md`) and prompt
/// override files (`SYSTEM.md`, `APPEND_SYSTEM.md`) from sandbox-accessible
/// locations.
///
/// No ancestor directory walking (sandbox constraint). Files that don't exist
/// or are empty are silently skipped.
nonisolated struct ContextLoader: Sendable {

    /// Root of the agent data directory.
    /// Typically `~/Library/Application Support/tesse-ract/agent/`.
    let agentRoot: URL

    // MARK: - LoadedContext

    /// Result of scanning all known locations for context and override files.
    struct LoadedContext: Sendable {
        /// Context files in load order (path, content). Each file appears at most once.
        let contextFiles: [(path: String, content: String)]
        /// If a `SYSTEM.md` was found, its content replaces the default system prompt.
        let systemOverride: String?
        /// If an `APPEND_SYSTEM.md` was found, its content is appended to the prompt.
        let systemAppend: String?
    }

    // MARK: - Public API

    /// Load all context and override files from known locations.
    ///
    /// - Parameters:
    ///   - packageContextFiles: Individual context file URLs from packages
    ///     (resolved by `PackageRegistry`).
    ///   - packagePromptAppends: Individual `APPEND_SYSTEM.md` file URLs from packages.
    ///   - packageSystemOverrides: Individual `SYSTEM.md` file URLs from packages.
    ///   - extensionPaths: Directory URLs provided by extension `resources_discover` events.
    /// - Returns: Aggregated context with deduplication.
    func load(
        packageContextFiles: [URL] = [],
        packagePromptAppends: [URL] = [],
        packageSystemOverrides: [URL] = [],
        extensionPaths: [URL] = []
    ) -> LoadedContext {
        var contextFiles: [(path: String, content: String)] = []
        var seenPaths = Set<String>()

        var systemOverride: String?
        var systemAppend: String?

        // --- Context files (load order = precedence) ---

        // 1. Global agent directory: AGENTS.md or CLAUDE.md
        if let (path, content) = loadContextFile(from: agentRoot) {
            if seenPaths.insert(path).inserted {
                contextFiles.append((path, content))
            }
        }

        // 2. Package-provided context (individual file URLs)
        for url in packageContextFiles {
            let path = url.path
            if seenPaths.insert(path).inserted {
                if let content = readNonEmpty(url) {
                    contextFiles.append((path, content))
                }
            }
        }

        // 3. Extension-provided context (directory URLs — scan for AGENTS.md/CLAUDE.md)
        for url in extensionPaths {
            if let (path, content) = loadContextFile(from: url) {
                if seenPaths.insert(path).inserted {
                    contextFiles.append((path, content))
                }
            }
        }

        // --- Prompt overrides (first found wins) ---

        // Check agent root directory
        if systemOverride == nil {
            systemOverride = readNonEmpty(agentRoot.appendingPathComponent("SYSTEM.md"))
        }
        if systemAppend == nil {
            systemAppend = readNonEmpty(agentRoot.appendingPathComponent("APPEND_SYSTEM.md"))
        }

        // Check package-provided overrides (first non-empty wins)
        for url in packageSystemOverrides {
            if systemOverride != nil { break }
            systemOverride = readNonEmpty(url)
        }
        for url in packagePromptAppends {
            if systemAppend != nil { break }
            systemAppend = readNonEmpty(url)
        }

        // Check extension-provided directories
        for location in extensionPaths {
            if systemOverride == nil {
                systemOverride = readNonEmpty(location.appendingPathComponent("SYSTEM.md"))
            }
            if systemAppend == nil {
                systemAppend = readNonEmpty(location.appendingPathComponent("APPEND_SYSTEM.md"))
            }
            if systemOverride != nil, systemAppend != nil { break }
        }

        return LoadedContext(
            contextFiles: contextFiles,
            systemOverride: systemOverride,
            systemAppend: systemAppend
        )
    }

    // MARK: - Private

    /// Try to load a context file (`AGENTS.md` or `CLAUDE.md`) from a directory.
    /// Returns the resolved absolute path and content, or nil if neither exists.
    private func loadContextFile(from directory: URL) -> (path: String, content: String)? {
        // Check AGENTS.md first (matches Pi convention)
        let agentsURL = directory.appendingPathComponent("AGENTS.md")
        if let content = readNonEmpty(agentsURL) {
            return (agentsURL.path, content)
        }

        // Fall back to CLAUDE.md
        let claudeURL = directory.appendingPathComponent("CLAUDE.md")
        if let content = readNonEmpty(claudeURL) {
            return (claudeURL.path, content)
        }

        return nil
    }

    /// Read a file's contents, returning nil if the file doesn't exist or is empty.
    private func readNonEmpty(_ url: URL) -> String? {
        guard let data = try? Data(contentsOf: url),
              let text = String(data: data, encoding: .utf8),
              !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            return nil
        }
        return text
    }
}
