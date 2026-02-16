import Foundation

struct AgentConversation: Identifiable, Codable, Sendable {
    let id: UUID
    var messages: [AgentChatMessage]
    let createdAt: Date
    var updatedAt: Date

    var title: String {
        messages.first(where: { $0.role == .user })?.content.prefix(80).description ?? "New Conversation"
    }

    var messageCount: Int { messages.count }

    init(id: UUID = UUID(), messages: [AgentChatMessage] = [], createdAt: Date = Date(), updatedAt: Date = Date()) {
        self.id = id
        self.messages = messages
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }
}

/// Lightweight summary for the conversation index (avoids loading full message history).
struct AgentConversationSummary: Identifiable, Codable, Sendable {
    let id: UUID
    let title: String
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
}
