import Foundation

/// Flat, pre-computed row model for the agent chat list.
/// All strings are ready for display — no JSON, no Date formatting, no protocol conversion in the render path.
struct ChatRow: Identifiable, Equatable, Sendable {
    let id: String
    let kind: Kind

    enum Kind: Equatable, Sendable {
        case user(UserRow)
        case assistantText(AssistantTextRow)
        case thinking(ThinkingRow)
        case toolCall(ToolCallRow)
        case toolText(ToolTextRow)
        case system(SystemRow)
        case turnHeader(TurnHeaderRow)
        case streamingText(StreamingTextRow)
        case streamingIndicator
    }
}

struct UserRow: Equatable, Sendable {
    let content: String
    let timestamp: String
    let messageID: UUID
}

struct AssistantTextRow: Equatable, Sendable {
    let content: String
    let timestamp: String
    let messageID: UUID
    let hasStepsAbove: Bool
}

struct ThinkingRow: Equatable, Sendable {
    let content: String
    let isLast: Bool
}

struct ToolCallRow: Equatable, Sendable {
    let displayTitle: String
    let iconName: String
    let argumentsFormatted: String
    let resultContent: String?
    let isError: Bool
    let isLast: Bool
    let isDetailExpanded: Bool

    func togglingDetail() -> ToolCallRow {
        ToolCallRow(displayTitle: displayTitle, iconName: iconName,
                    argumentsFormatted: argumentsFormatted, resultContent: resultContent,
                    isError: isError, isLast: isLast, isDetailExpanded: !isDetailExpanded)
    }
}

struct ToolTextRow: Equatable, Sendable {
    let content: String
    let isLast: Bool
}

struct SystemRow: Equatable, Sendable {
    let content: String
}

struct TurnHeaderRow: Equatable, Sendable {
    let stepCount: Int
    let isGenerating: Bool
    let turnID: UUID
    let isExpanded: Bool
}

struct StreamingTextRow: Equatable, Sendable {
    let content: String
}
