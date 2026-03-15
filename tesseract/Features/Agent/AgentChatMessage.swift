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
    let toolCallId: String?
    let isError: Bool

    // Explicit init with default toolCalls for convenience.
    init(id: UUID, timestamp: Date, role: Role, content: String, thinking: String?, toolCalls: [ToolCall] = [], toolCallId: String? = nil, isError: Bool = false) {
        self.id = id
        self.timestamp = timestamp
        self.role = role
        self.content = content
        self.thinking = thinking
        self.toolCalls = toolCalls
        self.toolCallId = toolCallId
        self.isError = isError
    }

    static func system(_ content: String) -> Self {
        Self(id: UUID(), timestamp: Date(), role: .system, content: content, thinking: nil, toolCallId: nil)
    }

    static func user(_ content: String) -> Self {
        Self(id: UUID(), timestamp: Date(), role: .user, content: content, thinking: nil, toolCallId: nil)
    }

    static func assistant(_ content: String, thinking: String? = nil, toolCalls: [ToolCall] = []) -> Self {
        Self(id: UUID(), timestamp: Date(), role: .assistant, content: content, thinking: thinking, toolCalls: toolCalls, toolCallId: nil)
    }

    static func tool(_ content: String, toolCallId: String, isError: Bool = false) -> Self {
        Self(id: UUID(), timestamp: Date(), role: .tool, content: content, thinking: nil, toolCallId: toolCallId, isError: isError)
    }

    // MARK: - AgentMessageProtocol

    nonisolated func toLLMMessage() -> LLMMessage? {
        switch role {
        case .system: .system(content: content)
        case .user: .user(content: content)
        case .assistant: .assistant(content: content, toolCalls: nil)
        case .tool: .toolResult(toolCallId: toolCallId ?? "", content: content)
        }
    }

    // MARK: - Convert from Protocol Messages

    /// Creates an `AgentChatMessage` from any `AgentMessageProtocol`.
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
                self.init(id: tr.id, timestamp: tr.timestamp, role: .tool, content: tr.content.textContent, thinking: nil, toolCallId: tr.toolCallId, isError: tr.isError)
            }
        case let user as UserMessage:
            self.init(id: user.id, timestamp: user.timestamp, role: .user, content: user.content, thinking: nil)
        case let asst as AssistantMessage:
            self.init(id: asst.id, timestamp: asst.timestamp, role: .assistant, content: asst.content, thinking: asst.thinking, toolCalls: Self.convertToolCalls(asst.toolCalls))
        case let tr as ToolResultMessage:
            self.init(id: tr.id, timestamp: tr.timestamp, role: .tool, content: tr.content.textContent, thinking: nil, toolCallId: tr.toolCallId, isError: tr.isError)
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

    /// Converts new-architecture `ToolCallInfo` to `ToolCall` (MLXLMCommon).
    private static func convertToolCalls(_ calls: [ToolCallInfo]) -> [ToolCall] {
        calls.map { info in
            buildToolCall(name: info.name, arguments: info.parsedArguments)
        }
    }

    private static func buildToolCall(name: String, arguments: [String: JSONValue]) -> ToolCall {
        let payload = ToolCallPayload(function: .init(name: name, arguments: arguments))
        if let data = try? JSONEncoder().encode(payload),
           let toolCall = try? JSONDecoder().decode(ToolCall.self, from: data) {
            return toolCall
        }

        assertionFailure("Failed to rebuild ToolCall with normalized arguments")
        return ToolCall(function: .init(name: name, arguments: [:]))
    }

    private struct ToolCallPayload: Codable {
        let function: FunctionPayload
    }

    private struct FunctionPayload: Codable {
        let name: String
        let arguments: [String: JSONValue]
    }

}
