import Foundation

struct AgentConversation: Identifiable, Sendable {
    let id: UUID
    var messages: [any AgentMessageProtocol & Sendable]
    let createdAt: Date
    var updatedAt: Date

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
        updatedAt: Date = Date()
    ) {
        self.id = id
        self.messages = messages
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }
}

/// Lightweight summary for the conversation index (avoids loading full message history).
struct AgentConversationSummary: Identifiable, Codable, Sendable {
    let id: UUID
    var title: String
    let createdAt: Date
    var updatedAt: Date
    var messageCount: Int

    init(from conversation: AgentConversation) {
        self.id = conversation.id
        self.title = conversation.title
        self.createdAt = conversation.createdAt
        self.updatedAt = conversation.updatedAt
        self.messageCount = conversation.messageCount
    }

    init(id: UUID, title: String, createdAt: Date, updatedAt: Date, messageCount: Int) {
        self.id = id
        self.title = title
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.messageCount = messageCount
    }
}
