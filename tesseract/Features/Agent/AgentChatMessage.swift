import Foundation
import MLXLMCommon

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
    let toolCalls: [ToolCall]

    // Explicit init with default toolCalls for convenience.
    init(id: UUID, timestamp: Date, role: Role, content: String, thinking: String?, toolCalls: [ToolCall] = []) {
        self.id = id
        self.timestamp = timestamp
        self.role = role
        self.content = content
        self.thinking = thinking
        self.toolCalls = toolCalls
    }

    // Custom Codable for backward compatibility (toolCalls absent in older data).
    enum CodingKeys: String, CodingKey {
        case id, timestamp, role, content, thinking, toolCalls
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(UUID.self, forKey: .id)
        timestamp = try c.decode(Date.self, forKey: .timestamp)
        role = try c.decode(Role.self, forKey: .role)
        content = try c.decode(String.self, forKey: .content)
        thinking = try c.decodeIfPresent(String.self, forKey: .thinking)
        toolCalls = try c.decodeIfPresent([ToolCall].self, forKey: .toolCalls) ?? []
    }

    static func system(_ content: String) -> Self {
        Self(id: UUID(), timestamp: Date(), role: .system, content: content, thinking: nil)
    }

    static func user(_ content: String) -> Self {
        Self(id: UUID(), timestamp: Date(), role: .user, content: content, thinking: nil)
    }

    static func assistant(_ content: String, thinking: String? = nil, toolCalls: [ToolCall] = []) -> Self {
        Self(id: UUID(), timestamp: Date(), role: .assistant, content: content, thinking: thinking, toolCalls: toolCalls)
    }

    static func tool(_ content: String) -> Self {
        Self(id: UUID(), timestamp: Date(), role: .tool, content: content, thinking: nil)
    }

    // MARK: - Observation Masking

    private static let maskedToolContent = "[Tool result omitted — re-run tool if needed]"

    /// Applies observation masking: replaces old tool results with a short placeholder
    /// to reduce context size while preserving conversation structure.
    ///
    /// JetBrains research shows this outperforms LLM summarization — the model keeps its
    /// action history (user + assistant messages) intact but sheds stale tool outputs.
    ///
    /// - Parameters:
    ///   - messages: The full message list (excluding system prompt).
    ///   - preserveRecent: Number of most-recent messages whose tool results stay intact.
    static func withObservationMasking(
        _ messages: [AgentChatMessage],
        preserveRecent: Int = 20
    ) -> [AgentChatMessage] {
        guard messages.count > preserveRecent else { return messages }
        let maskBefore = messages.count - preserveRecent
        return messages.enumerated().map { i, msg in
            if i < maskBefore && msg.role == .tool && msg.content != maskedToolContent {
                return .tool(maskedToolContent)
            }
            return msg
        }
    }
}
