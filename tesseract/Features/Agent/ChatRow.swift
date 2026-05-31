import Foundation

/// Flat, pre-computed row model for the agent chat list.
/// All strings are ready for display — no JSON, no Date formatting, no protocol conversion in the render path.
///
/// `nonisolated` value type: built by the `nonisolated` Chat Transcript
/// projection and consumed by the MainActor coordinator and SwiftUI views alike.
nonisolated struct ChatRow: Identifiable, Equatable, Sendable {
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

    /// Returns a copy with `isLast` stamped on step-row kinds (thinking, toolCall, toolText).
    func withIsLast(_ isLast: Bool) -> ChatRow {
        switch kind {
        case .thinking(let d):
            ChatRow(id: id, kind: .thinking(ThinkingRow(content: d.content, isLast: isLast)))
        case .toolCall(let d):
            ChatRow(id: id, kind: .toolCall(ToolCallRow(
                displayTitle: d.displayTitle, iconName: d.iconName,
                argumentsFormatted: d.argumentsFormatted, resultContent: d.resultContent,
                isError: d.isError, isLast: isLast,
                isDetailExpanded: d.isDetailExpanded, filePath: d.filePath)))
        case .toolText(let d):
            ChatRow(id: id, kind: .toolText(ToolTextRow(content: d.content, isLast: isLast)))
        default:
            self
        }
    }
}

nonisolated struct UserRow: Equatable, Sendable {
    let content: String
    let images: [ImageAttachment]
    let timestamp: String
    let messageID: UUID
}

nonisolated struct AssistantTextRow: Equatable, Sendable {
    let content: String
    let timestamp: String
    let messageID: UUID
    let hasStepsAbove: Bool
}

nonisolated struct ThinkingRow: Equatable, Sendable {
    let content: String
    let isLast: Bool
}

nonisolated struct ToolCallRow: Equatable, Sendable {
    let displayTitle: String
    let iconName: String
    let argumentsFormatted: String
    let resultContent: String?
    let isError: Bool
    let isLast: Bool
    let isDetailExpanded: Bool
    let filePath: String?

    func togglingDetail() -> ToolCallRow {
        ToolCallRow(displayTitle: displayTitle, iconName: iconName,
                    argumentsFormatted: argumentsFormatted, resultContent: resultContent,
                    isError: isError, isLast: isLast, isDetailExpanded: !isDetailExpanded,
                    filePath: filePath)
    }
}

nonisolated struct ToolTextRow: Equatable, Sendable {
    let content: String
    let isLast: Bool
}

nonisolated struct SystemRow: Equatable, Sendable {
    let content: String
}

nonisolated struct TurnHeaderRow: Equatable, Sendable {
    let stepCount: Int
    let isGenerating: Bool
    let turnID: UUID
    let isExpanded: Bool
}

nonisolated struct StreamingTextRow: Equatable, Sendable {
    let content: String
}
