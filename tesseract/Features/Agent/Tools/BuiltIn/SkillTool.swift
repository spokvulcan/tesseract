import Foundation

// MARK: - Constants

/// Tool name used in SystemPromptAssembler gate check and tool definition.
nonisolated(unsafe) let skillToolName = "use_skill"

// MARK: - SkillTool Factory

/// Creates the `use_skill` tool that gives the agent first-class access to skills.
///
/// Three modes:
/// - **List** (no `name`): returns names + descriptions of available skills.
/// - **Load** (`name` only): reads the skill's SKILL.md, strips frontmatter,
///   and returns the body plus a listing of linked files.
/// - **Linked file** (`name` + `file_path`): reads a specific reference/template
///   file within the skill's directory.
nonisolated func createSkillTool(skills: [SkillMetadata]) -> AgentToolDefinition {
    AgentToolDefinition(
        name: skillToolName,
        label: "Use Skill",
        description: "Load a skill's instructions, or list all available skills. "
            + "Call with no arguments to see available skills. "
            + "Call with a name to load that skill's full instructions.",
        parameterSchema: JSONSchema(
            type: "object",
            properties: [
                "name": PropertySchema(
                    type: "string",
                    description: "Skill name to load. Omit to list all available skills."
                ),
                "file_path": PropertySchema(
                    type: "string",
                    description: "Relative path to a linked file within the skill directory "
                        + "(e.g. 'references/api.md'). Only valid when name is provided."
                ),
            ],
            required: []
        ),
        execute: { _, argsJSON, _, _ in
            let name = ToolArgExtractor.string(argsJSON, key: "name")
            let filePath = ToolArgExtractor.string(argsJSON, key: "file_path")

            // ── List mode ──────────────────────────────────────────────
            guard let name else {
                return listSkills(skills)
            }

            // ── Find skill ─────────────────────────────────────────────
            // Disabled skills are still loadable by name — only hidden from listings.
            // This lets users invoke them via slash commands that mention the name.
            let query = name.lowercased()
            guard let skill = skills.first(where: { $0.name == query }) else {
                let available = skills
                    .filter { !$0.disableModelInvocation }
                    .map(\.name)
                    .joined(separator: ", ")
                return .error(
                    "Unknown skill '\(name)'. Available skills: \(available)"
                )
            }

            // ── Linked file mode ───────────────────────────────────────
            if let filePath {
                return loadLinkedFile(skill: skill, relativePath: filePath)
            }

            // ── Load mode ──────────────────────────────────────────────
            return loadSkill(skill)
        }
    )
}

// MARK: - Private Helpers

private nonisolated func listSkills(_ skills: [SkillMetadata]) -> AgentToolResult {
    let eligible = skills.filter { !$0.disableModelInvocation }
    guard !eligible.isEmpty else {
        return .text("No skills available.")
    }

    var output = "Available skills:\n"
    for skill in eligible {
        output += "\n- \(skill.name): \(skill.description)"
    }
    output += "\n\nCall use_skill(name: \"skill-name\") to load full instructions."
    return .text(output)
}

private nonisolated func loadSkill(_ skill: SkillMetadata) -> AgentToolResult {
    let fileURL = URL(fileURLWithPath: skill.filePath)

    guard let data = try? Data(contentsOf: fileURL),
          let fullText = String(data: data, encoding: .utf8)
    else {
        return .error("Failed to read skill file: \(skill.filePath)")
    }

    let body = SkillRegistry.bodyContent(of: fullText)
    let skillDir = fileURL.deletingLastPathComponent()

    var output = "# Skill: \(skill.name)\n"
    output += "Location: \(skill.filePath)\n\n"
    output += body

    let linked = enumerateLinkedFiles(in: skillDir, skillFile: fileURL)
    if !linked.isEmpty {
        output += "\n\n---\nLinked files:"
        for file in linked {
            output += "\n- \(file)"
        }
        output += "\n\nCall use_skill(name: \"\(skill.name)\", file_path: \"<path>\") to load."
    }

    return .text(output)
}

private nonisolated func loadLinkedFile(
    skill: SkillMetadata,
    relativePath: String
) -> AgentToolResult {
    let skillDir = URL(fileURLWithPath: skill.filePath).deletingLastPathComponent()

    // Use PathSandbox for proper traversal + symlink protection
    let sandbox = PathSandbox(root: skillDir)
    let resolved: URL
    do {
        resolved = try sandbox.resolve(relativePath)
    } catch {
        return .error("Path '\(relativePath)' is outside the skill directory.")
    }

    guard let data = try? Data(contentsOf: resolved),
          let content = String(data: data, encoding: .utf8)
    else {
        let available = enumerateLinkedFiles(in: skillDir,
                                             skillFile: URL(fileURLWithPath: skill.filePath))
        if available.isEmpty {
            return .error("File not found: \(relativePath)")
        }
        var msg = "File not found: \(relativePath)\n\nAvailable linked files:"
        for file in available {
            msg += "\n- \(file)"
        }
        return .error(msg)
    }

    return .text("# \(skill.name) / \(relativePath)\n\n\(content)")
}

/// Scans a skill directory for linked files (references, templates, scripts, etc.).
/// Returns relative paths sorted alphabetically. Scans one level of subdirectories.
private nonisolated func enumerateLinkedFiles(in directory: URL, skillFile: URL) -> [String] {
    let fm = FileManager.default
    var results: [String] = []

    guard let entries = try? fm.contentsOfDirectory(
        at: directory,
        includingPropertiesForKeys: [.isDirectoryKey],
        options: [.skipsHiddenFiles]
    ) else {
        return []
    }

    for entry in entries {
        if entry.standardizedFileURL == skillFile.standardizedFileURL { continue }

        let isDir = (try? entry.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory ?? false

        if isDir {
            let subName = entry.lastPathComponent
            if let subEntries = try? fm.contentsOfDirectory(
                at: entry,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
            ) {
                for subFile in subEntries {
                    let subIsDir = (try? subFile.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory ?? false
                    if !subIsDir {
                        results.append("\(subName)/\(subFile.lastPathComponent)")
                    }
                }
            }
        } else {
            results.append(entry.lastPathComponent)
        }
    }

    return results.sorted()
}
