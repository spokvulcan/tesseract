import Foundation

/// Inserted by context compaction to replace superseded history with a summary.
/// Renders as a `<summary>` XML block in the LLM context so the model treats it
/// as condensed prior knowledge rather than a user turn.
nonisolated struct CompactionSummaryMessage: CustomAgentMessage, PersistableMessage, Sendable,
    Codable, Equatable, Identifiable
{
    static let persistenceTag = "compaction_summary"
    let customType = "compaction_summary"

    let id: UUID
    let summary: String
    let tokensBefore: Int
    let timestamp: Date

    init(id: UUID = UUID(), summary: String, tokensBefore: Int, timestamp: Date = Date()) {
        self.id = id
        self.summary = summary
        self.tokensBefore = tokensBefore
        self.timestamp = timestamp
    }

    func toLLMMessage() -> LLMMessage? {
        .user(content: "<summary>\n\(summary)\n</summary>")
    }

    /// The one-line system-row text shown for this compaction marker. The single
    /// source for the wording, reused by the Chat Transcript projection and the
    /// `AgentChatMessage` conversion so the two render paths can't drift.
    var displayText: String {
        "[Context compacted — \(tokensBefore) tokens summarized]"
    }
}
