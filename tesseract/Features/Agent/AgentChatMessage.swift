import Foundation

/// Minimal domain message type for the agent inference layer.
///
/// Decoupled from ``MLXLMCommon/Chat.Message`` to stay fully `Sendable`
/// (text-only — no image/video baggage) and to avoid requiring `MLXLMCommon`
/// imports throughout app code. See ``AgentChatFormatter`` for the bridge.
struct AgentChatMessage: Sendable, Codable, Identifiable {
    enum Role: String, Sendable, Codable {
        case system
        case user
        case assistant
        case tool
    }

    let id: UUID
    let timestamp: Date
    let role: Role
    let content: String
    let thinking: String?

    static func system(_ content: String) -> Self {
        Self(id: UUID(), timestamp: Date(), role: .system, content: content, thinking: nil)
    }

    static func user(_ content: String) -> Self {
        Self(id: UUID(), timestamp: Date(), role: .user, content: content, thinking: nil)
    }

    static func assistant(_ content: String, thinking: String? = nil) -> Self {
        Self(id: UUID(), timestamp: Date(), role: .assistant, content: content, thinking: thinking)
    }

    static func tool(_ content: String) -> Self {
        Self(id: UUID(), timestamp: Date(), role: .tool, content: content, thinking: nil)
    }
}
