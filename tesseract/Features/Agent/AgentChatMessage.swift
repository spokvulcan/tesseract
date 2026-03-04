import Foundation
import MLXLMCommon

/// UI display message type for the agent chat view.
///
/// Kept as a flat, `Codable` struct for view rendering. The coordinator converts
/// protocol-based messages (`CoreMessage`, `AssistantMessage`, etc.) into this
/// type via `init(from:)`. Not used as a primary storage format — the new
/// architecture persists `AgentMessageProtocol` types via tagged JSON.
struct AgentChatMessage: AgentMessageProtocol, Sendable, Codable, Identifiable {
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

    // MARK: - AgentMessageProtocol

    nonisolated func toLLMMessage() -> LLMMessage? {
        switch role {
        case .system: .system(content: content)
        case .user: .user(content: content)
        case .assistant: .assistant(content: content, toolCalls: nil)
        case .tool: .toolResult(toolCallId: "", content: content)
        }
    }

    // MARK: - Convert from Protocol Messages

    /// Creates an `AgentChatMessage` from any `AgentMessageProtocol` for legacy UI display.
    init(from message: any AgentMessageProtocol) {
        let now = Date()
        switch message {
        case let core as CoreMessage:
            switch core {
            case .user(let user):
                self.init(id: user.id, timestamp: user.timestamp, role: .user, content: user.content, thinking: nil)
            case .assistant(let asst):
                self.init(id: asst.id, timestamp: asst.timestamp, role: .assistant, content: asst.content, thinking: asst.thinking, toolCalls: Self.convertToolCalls(asst.toolCalls))
            case .toolResult(let tr):
                self.init(id: tr.id, timestamp: tr.timestamp, role: .tool, content: tr.content.textContent, thinking: nil)
            }
        case let user as UserMessage:
            self.init(id: user.id, timestamp: user.timestamp, role: .user, content: user.content, thinking: nil)
        case let asst as AssistantMessage:
            self.init(id: asst.id, timestamp: asst.timestamp, role: .assistant, content: asst.content, thinking: asst.thinking, toolCalls: Self.convertToolCalls(asst.toolCalls))
        case let tr as ToolResultMessage:
            self.init(id: tr.id, timestamp: tr.timestamp, role: .tool, content: tr.content.textContent, thinking: nil)
        case let compaction as CompactionSummaryMessage:
            self.init(id: UUID(), timestamp: compaction.timestamp, role: .system,
                      content: "[Context compacted — \(compaction.tokensBefore) tokens summarized]", thinking: nil)
        case let chat as AgentChatMessage:
            self = chat
        default:
            // Unknown message type — render as system note
            self.init(id: UUID(), timestamp: now, role: .system, content: "[Unknown message]", thinking: nil)
        }
    }

    // MARK: - ToolCallInfo → ToolCall Conversion

    /// Converts new-architecture `ToolCallInfo` to legacy `ToolCall` (MLXLMCommon).
    private static func convertToolCalls(_ calls: [ToolCallInfo]) -> [ToolCall] {
        calls.map { info in
            let args: [String: JSONValue]
            if !info.argumentsJSON.isEmpty,
               let data = info.argumentsJSON.data(using: .utf8),
               let parsed = try? JSONDecoder().decode([String: JSONValue].self, from: data)
            {
                args = parsed
            } else {
                args = [:]
            }
            return ToolCall(function: .init(name: info.name, arguments: args))
        }
    }

}
