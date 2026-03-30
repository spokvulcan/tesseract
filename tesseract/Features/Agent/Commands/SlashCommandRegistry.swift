import Foundation

// MARK: - SlashCommandRegistry

/// Combines built-in, skill, and extension commands into a searchable registry.
/// Rebuilt when skills or extensions change.
@MainActor
final class SlashCommandRegistry {

    private(set) var commands: [SlashCommand] = []

    // MARK: - Built-in Commands

    private static let builtInCommands: [SlashCommand] = [
        SlashCommand(
            name: "compact",
            description: "Compact conversation context",
            source: .builtIn,
            argumentHint: "[instructions]"
        ),
        SlashCommand(
            name: "new",
            description: "Start a new conversation",
            source: .builtIn,
            argumentHint: nil
        ),
        SlashCommand(
            name: "clear",
            description: "Clear conversation (alias for /new)",
            source: .builtIn,
            argumentHint: nil
        ),
    ]

    // MARK: - Rebuild

    /// Rebuild the merged command list from all sources.
    func rebuild(skills: [SkillMetadata], extensionHost: ExtensionHost? = nil) {
        var result = Self.builtInCommands
        var seenNames = Set(result.map(\.name))

        for skill in skills {
            guard seenNames.insert(skill.name).inserted else { continue }
            result.append(SlashCommand(
                name: skill.name,
                description: skill.description,
                source: .skill(filePath: skill.filePath),
                argumentHint: nil
            ))
        }

        if let extensionHost {
            for ext in extensionHost.registeredExtensions {
                for (_, cmd) in ext.commands {
                    guard seenNames.insert(cmd.name).inserted else { continue }
                    result.append(SlashCommand(
                        name: cmd.name,
                        description: cmd.description,
                        source: .extension(extensionPath: ext.path),
                        argumentHint: nil
                    ))
                }
            }
        }

        commands = result
    }

    // MARK: - Lookup

    /// Look up a command by exact name.
    func command(named name: String) -> SlashCommand? {
        commands.first { $0.name == name }
    }

    /// Filter commands matching a prefix (substring on name + description).
    /// Returns all commands when prefix is empty.
    func filter(prefix: String) -> [SlashCommand] {
        guard !prefix.isEmpty else { return commands }
        let lowered = prefix.lowercased()
        return commands.filter { cmd in
            cmd.name.lowercased().contains(lowered)
                || cmd.description.lowercased().contains(lowered)
        }.sorted { lhs, rhs in
            let lhsPrefix = lhs.name.lowercased().hasPrefix(lowered)
            let rhsPrefix = rhs.name.lowercased().hasPrefix(lowered)
            if lhsPrefix != rhsPrefix { return lhsPrefix }
            return lhs.name < rhs.name
        }
    }
}
