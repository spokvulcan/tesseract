import Foundation

// MARK: - SlashCommandParser

/// Stateless parser that extracts slash commands from input text.
enum SlashCommandParser {

    /// Parse input text against the registry.
    /// Only recognizes commands at the very start of input.
    static func parse(
        _ input: String,
        registry: SlashCommandRegistry
    ) -> SlashCommandParseResult {
        let trimmed = input.trimmingCharacters(in: .whitespaces)
        guard trimmed.hasPrefix("/") else { return .notACommand }

        let afterSlash = String(trimmed.dropFirst())
        guard !afterSlash.isEmpty else { return .partial(prefix: "") }

        // Split into command name and arguments at first space
        let parts = afterSlash.split(separator: " ", maxSplits: 1)
        let commandName = String(parts[0]).lowercased()
        let arguments = parts.count > 1
            ? String(parts[1]).trimmingCharacters(in: .whitespaces)
            : ""

        let hasTrailingContent = afterSlash.contains(" ")

        if let command = registry.command(named: commandName) {
            return .matched(command: command, arguments: arguments)
        }

        // No exact match — if no space yet, treat as partial typing
        if !hasTrailingContent {
            return .partial(prefix: commandName)
        }

        return .unknown(name: commandName)
    }

    /// Extract the prefix being typed for autocomplete filtering.
    /// Returns nil if input doesn't start with "/" or user is past the command name.
    static func autocompletePrefix(_ input: String) -> String? {
        let trimmed = input.trimmingCharacters(in: .whitespaces)
        guard trimmed.hasPrefix("/") else { return nil }
        let afterSlash = String(trimmed.dropFirst())
        // Only complete the command name, not arguments
        if afterSlash.contains(" ") { return nil }
        return afterSlash.lowercased()
    }
}
