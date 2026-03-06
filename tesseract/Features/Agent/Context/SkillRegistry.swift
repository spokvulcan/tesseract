import Foundation

// MARK: - SkillMetadata

/// Parsed metadata from a skill file's YAML frontmatter.
nonisolated struct SkillMetadata: Sendable {
    /// Lowercase, hyphenated skill name (e.g. "memory-management").
    let name: String
    /// Short description (max 1024 chars) shown in the system prompt.
    let description: String
    /// Absolute path so the model can use the `read` tool to load the skill.
    let filePath: String
    /// When true, the skill is available but not listed in the prompt.
    let disableModelInvocation: Bool
}

// MARK: - SkillRegistry

/// Discovers skill files from known locations, parses YAML frontmatter, and
/// generates the XML listing injected into the system prompt.
nonisolated enum SkillRegistry: Sendable {

    // MARK: - Discovery

    /// Discover all skills from the given locations (in load order).
    ///
    /// - Parameters:
    ///   - locations: Skill directories to scan (e.g. `agentRoot/skills/`).
    ///   - packageSkillFiles: Individual skill file URLs from packages
    ///     (resolved by `PackageRegistry`, e.g. `package/skills/memory/SKILL.md`).
    /// - Returns: Deduplicated skills in discovery order.
    static func discover(
        locations: [URL],
        packageSkillFiles: [URL] = []
    ) -> [SkillMetadata] {
        var results: [SkillMetadata] = []
        var seenNames = Set<String>()

        // 1. Scan skill directories (user-global, extension-provided)
        for directory in locations {
            let skills = scanDirectory(directory)
            for skill in skills {
                if seenNames.insert(skill.name).inserted {
                    results.append(skill)
                }
            }
        }

        // 2. Parse individual package skill files
        for fileURL in packageSkillFiles {
            let fallbackName = fileURL.deletingLastPathComponent().lastPathComponent
            if let skill = parseSkillFile(fileURL, fallbackName: fallbackName) {
                if seenNames.insert(skill.name).inserted {
                    results.append(skill)
                }
            }
        }

        return results
    }

    // MARK: - Prompt Formatting

    /// Format discovered skills as XML for the system prompt.
    /// Only includes skills where `disableModelInvocation` is false.
    static func formatForPrompt(_ skills: [SkillMetadata]) -> String {
        let eligible = skills.filter { !$0.disableModelInvocation }
        guard !eligible.isEmpty else { return "" }

        var xml = """
            The following skills provide specialized instructions for specific tasks.
            Never guess how specific workflow or task should be done.
            Always read skill before do anything related to skill description. 
            Use the read tool to read a skill's file when the task matches its description.
            When a skill file references a relative path, resolve it against the skill directory
            (parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.
            Skills are .md files not tools, always use read tool to load the skill.
            <available_skills>
            """

        for skill in eligible {
            xml += "\n  <skill>"
            xml += "\n    <name>\(escapeXML(skill.name))</name>"
            xml += "\n    <description>\(escapeXML(skill.description))</description>"
            xml += "\n    <location>\(escapeXML(skill.filePath))</location>"
            xml += "\n  </skill>"
        }

        xml += "\n</available_skills>"
        return xml
    }

    // MARK: - Private — Directory Scanning

    /// Maximum depth for recursive skill directory scanning.
    private static let maxScanDepth = 5

    /// Scan a single directory for skills:
    /// 1. Root `.md` files with valid frontmatter
    /// 2. `SKILL.md` in subdirectories (recursive, depth-limited)
    private static func scanDirectory(_ directory: URL) -> [SkillMetadata] {
        let fm = FileManager.default
        var results: [SkillMetadata] = []

        guard let entries = try? fm.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }

        for entry in entries {
            let isDirectory = (try? entry.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory ?? false

            if isDirectory {
                // Look for SKILL.md in this subdirectory
                let skillFile = entry.appendingPathComponent("SKILL.md")
                if let skill = parseSkillFile(skillFile, fallbackName: entry.lastPathComponent) {
                    results.append(skill)
                }
                // Recurse into nested subdirectories
                results.append(contentsOf: scanSubdirectories(entry, depth: 1))
            } else if entry.pathExtension.lowercased() == "md" {
                // Root .md files with valid frontmatter
                if let skill = parseSkillFile(entry, fallbackName: entry.deletingPathExtension().lastPathComponent) {
                    results.append(skill)
                }
            }
        }

        return results
    }

    /// Recursively scan subdirectories for `SKILL.md` files, with depth limit.
    private static func scanSubdirectories(_ directory: URL, depth: Int) -> [SkillMetadata] {
        guard depth < maxScanDepth else { return [] }

        let fm = FileManager.default
        var results: [SkillMetadata] = []

        guard let entries = try? fm.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }

        for entry in entries {
            let isDirectory = (try? entry.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory ?? false
            guard isDirectory else { continue }

            let skillFile = entry.appendingPathComponent("SKILL.md")
            if let skill = parseSkillFile(skillFile, fallbackName: entry.lastPathComponent) {
                results.append(skill)
            }
            results.append(contentsOf: scanSubdirectories(entry, depth: depth + 1))
        }

        return results
    }

    // MARK: - Private — Frontmatter Parsing

    /// Parse a skill file's YAML frontmatter.
    /// Returns nil if the file doesn't exist, has no frontmatter, or lacks a description.
    private static func parseSkillFile(_ url: URL, fallbackName: String) -> SkillMetadata? {
        guard let data = try? Data(contentsOf: url),
              let text = String(data: data, encoding: .utf8),
              !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            return nil
        }

        guard let frontmatter = extractFrontmatter(text) else { return nil }

        // Description is required
        guard let description = frontmatter["description"],
              !description.isEmpty
        else {
            return nil
        }

        let name = frontmatter["name"] ?? fallbackName
        let disableInvocation = frontmatter["disable-model-invocation"]
            .map { $0.lowercased() == "true" } ?? false

        return SkillMetadata(
            name: name.lowercased(),
            description: String(description.prefix(1024)),
            filePath: url.path,
            disableModelInvocation: disableInvocation
        )
    }

    /// Extract key-value pairs from YAML frontmatter (between `---` delimiters).
    /// Handles simple `key: value` pairs only (no nested YAML).
    private static func extractFrontmatter(_ text: String) -> [String: String]? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("---") else { return nil }

        // Find the closing ---
        let afterOpening = trimmed.index(trimmed.startIndex, offsetBy: 3)
        let rest = trimmed[afterOpening...]

        guard let closingRange = rest.range(of: "\n---") else { return nil }

        let frontmatterBlock = rest[..<closingRange.lowerBound]
        var result: [String: String] = [:]

        for line in frontmatterBlock.split(separator: "\n", omittingEmptySubsequences: true) {
            let trimmedLine = line.trimmingCharacters(in: .whitespaces)
            guard let colonIndex = trimmedLine.firstIndex(of: ":") else { continue }

            let key = trimmedLine[..<colonIndex].trimmingCharacters(in: .whitespaces)
            let value = trimmedLine[trimmedLine.index(after: colonIndex)...]
                .trimmingCharacters(in: .whitespaces)

            // Strip surrounding quotes if present
            let unquoted = stripQuotes(value)
            if !key.isEmpty {
                result[key] = unquoted
            }
        }

        return result.isEmpty ? nil : result
    }

    /// Remove surrounding single or double quotes from a string.
    private static func stripQuotes(_ value: String) -> String {
        if (value.hasPrefix("\"") && value.hasSuffix("\""))
            || (value.hasPrefix("'") && value.hasSuffix("'"))
        {
            return String(value.dropFirst().dropLast())
        }
        return value
    }

    /// Escape XML special characters in text content.
    private static func escapeXML(_ text: String) -> String {
        text.replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;")
    }
}
