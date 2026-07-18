import Foundation

/// The turn-class vocabulary (#327 §2, ADR-0040 §8): which kind of turn opened
/// a conversation. `interactive` is the owner typing; the rest are the
/// Companion's own turns. Raw values are the persisted tag — the store and the
/// index keep plain strings so pre-tag files load unchanged.
nonisolated enum TurnOrigin: String, Codable, Sendable {
    /// The owner's own typed (or spoken) chat — never badged.
    case interactive
    /// A rhythm beat's turn — the fired batch carried a rhythm-class wake.
    case beat
    /// A booked wake fired on time (promise, follow-up, re-summons).
    case wake
    /// Unoccasioned cognition (ADR-0040 §7).
    case ambient
    /// Overdue wakes triaged late (past the catch-up grace).
    case catchup
    /// Reserved (#327 §2): sleep passes ride `internalCompletion`, not the
    /// turn machinery, so nothing emits this yet — the tag waits for them.
    case sleep
}

struct AgentConversation: Identifiable, Sendable {
    let id: UUID
    var messages: [any AgentMessageProtocol & Sendable]
    let createdAt: Date
    var updatedAt: Date
    /// Which turn class opened this conversation (#327's one-interface tag).
    var origin: TurnOrigin

    /// Derive title from first user message content.
    var title: String {
        for msg in messages {
            if let user = msg.asUser {
                let text = user.content.prefix(80)
                return text.isEmpty ? "New Conversation" : String(text)
            }
        }
        return "New Conversation"
    }

    var messageCount: Int { messages.count }

    init(
        id: UUID = UUID(),
        messages: [any AgentMessageProtocol & Sendable] = [],
        createdAt: Date = Date(),
        updatedAt: Date = Date(),
        origin: TurnOrigin = .interactive
    ) {
        self.id = id
        self.messages = messages
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.origin = origin
    }
}

/// Lightweight summary for the conversation index (avoids loading full message history).
struct AgentConversationSummary: Identifiable, Codable, Sendable {
    let id: UUID
    var title: String
    let createdAt: Date
    var updatedAt: Date
    var messageCount: Int
    /// Raw string, optional, so a pre-tag index decodes unchanged; nil (or an
    /// unknown tag) reads as interactive.
    var origin: String?

    /// The typed view of the raw tag.
    var turnOrigin: TurnOrigin { origin.flatMap(TurnOrigin.init(rawValue:)) ?? .interactive }

    init(from conversation: AgentConversation) {
        self.id = conversation.id
        self.title = conversation.title
        self.createdAt = conversation.createdAt
        self.updatedAt = conversation.updatedAt
        self.messageCount = conversation.messageCount
        self.origin = conversation.origin.rawValue
    }

    init(
        id: UUID, title: String, createdAt: Date, updatedAt: Date, messageCount: Int,
        origin: String? = nil
    ) {
        self.id = id
        self.title = title
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.messageCount = messageCount
        self.origin = origin
    }
}
