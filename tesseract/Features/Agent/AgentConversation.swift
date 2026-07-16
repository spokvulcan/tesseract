import Foundation

struct AgentConversation: Identifiable, Sendable {
    let id: UUID
    var messages: [any AgentMessageProtocol & Sendable]
    let createdAt: Date
    var updatedAt: Date
    /// Which turn class opened this conversation (#327's one-interface tag):
    /// `interactive` (the owner), or the Companion's `wake | ambient | catchup
    /// | sleep`. Optional-with-default so pre-tag files load unchanged — the
    /// store wipes on version bumps, and the owner's history outranks a
    /// required field.
    var origin: String

    /// Derive title from first user message content.
    var title: String {
        for msg in messages {
            if let core = msg as? CoreMessage, case .user(let user) = core {
                let text = user.content.prefix(80)
                return text.isEmpty ? "New Conversation" : String(text)
            }
            // Also check bare UserMessage (not wrapped in CoreMessage)
            if let user = msg as? UserMessage {
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
        origin: String = "interactive"
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
    /// Optional so a pre-tag index decodes unchanged; nil reads as interactive.
    var origin: String?

    init(from conversation: AgentConversation) {
        self.id = conversation.id
        self.title = conversation.title
        self.createdAt = conversation.createdAt
        self.updatedAt = conversation.updatedAt
        self.messageCount = conversation.messageCount
        self.origin = conversation.origin
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
