import Foundation

// MARK: - SlashCommandSource

/// Where a slash command originates.
enum SlashCommandSource: Sendable, Hashable {
    case builtIn
    case skill(filePath: String)
    case `extension`(extensionPath: String)
}

// MARK: - SlashCommand

/// A slash command known to the system.
struct SlashCommand: Sendable, Identifiable, Hashable {
    var id: String { name }
    let name: String
    let description: String
    let source: SlashCommandSource
    let argumentHint: String?

    var displayName: String {
        if let hint = argumentHint {
            return "/\(name) \(hint)"
        }
        return "/\(name)"
    }
}

// MARK: - SlashCommandParseResult

/// Result of parsing user input for a slash command.
enum SlashCommandParseResult: Sendable {
    /// Input is not a command.
    case notACommand
    /// User is still typing the command name.
    case partial(prefix: String)
    /// Fully matched command with optional arguments.
    case matched(command: SlashCommand, arguments: String)
    /// Looks like a command but no match found.
    case unknown(name: String)
}
