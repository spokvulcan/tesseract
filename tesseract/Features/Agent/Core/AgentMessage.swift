import Foundation

// MARK: - ToolCallInfo

/// Parsed tool call extracted from an assistant response.
nonisolated struct ToolCallInfo: Sendable, Codable, Hashable, Identifiable {
    let id: String
    let name: String
    let argumentsJSON: String
}

// MARK: - LLMMessage

/// Low-level message representation for the LLM context window.
/// Transient — not persisted. Built from higher-level message types via `toLLMMessage()`.
nonisolated enum LLMMessage: Sendable, Equatable {
    case system(content: String)
    case user(content: String)
    case assistant(content: String, toolCalls: [ToolCallInfo]?)
    case toolResult(toolCallId: String, content: String)
}

// MARK: - AgentMessageProtocol

/// Common interface for all messages flowing through the agent pipeline.
protocol AgentMessageProtocol: Sendable {
    /// Convert to LLM context representation. Returns nil for messages
    /// that should not appear in the LLM context (e.g. UI-only messages).
    nonisolated func toLLMMessage() -> LLMMessage?
}

// MARK: - UserMessage

/// A message from the user (text input or transcribed voice).
nonisolated struct UserMessage: AgentMessageProtocol, Codable, Equatable, Identifiable, Sendable {
    let id: UUID
    let content: String
    let timestamp: Date

    init(id: UUID = UUID(), content: String, timestamp: Date = Date()) {
        self.id = id
        self.content = content
        self.timestamp = timestamp
    }

    func toLLMMessage() -> LLMMessage? {
        .user(content: content)
    }
}

// MARK: - AssistantMessage

/// A response from the LLM, optionally containing thinking and tool calls.
nonisolated struct AssistantMessage: AgentMessageProtocol, Codable, Equatable, Identifiable, Sendable {
    let id: UUID
    let content: String
    let thinking: String?
    let toolCalls: [ToolCallInfo]
    let timestamp: Date

    init(
        id: UUID = UUID(),
        content: String,
        thinking: String? = nil,
        toolCalls: [ToolCallInfo] = [],
        timestamp: Date = Date()
    ) {
        self.id = id
        self.content = content
        self.thinking = thinking
        self.toolCalls = toolCalls
        self.timestamp = timestamp
    }

    func toLLMMessage() -> LLMMessage? {
        .assistant(content: content, toolCalls: toolCalls.isEmpty ? nil : toolCalls)
    }
}

// MARK: - ToolResultMessage

/// The result of executing a tool call.
nonisolated struct ToolResultMessage: AgentMessageProtocol, Codable, Equatable, Identifiable, Sendable {
    let id: UUID
    let toolCallId: String
    let toolName: String
    let content: [ContentBlock]
    let isError: Bool
    let timestamp: Date

    init(
        id: UUID = UUID(),
        toolCallId: String,
        toolName: String,
        content: [ContentBlock],
        isError: Bool = false,
        timestamp: Date = Date()
    ) {
        self.id = id
        self.toolCallId = toolCallId
        self.toolName = toolName
        self.content = content
        self.isError = isError
        self.timestamp = timestamp
    }

    func toLLMMessage() -> LLMMessage? {
        let textContent = content.compactMap { block -> String? in
            if case .text(let text) = block { return text }
            return nil
        }.joined(separator: "\n")
        return .toolResult(toolCallId: toolCallId, content: textContent)
    }
}

// MARK: - CoreMessage

/// Unified wrapper for the three core message types in a conversation.
nonisolated enum CoreMessage: AgentMessageProtocol, Sendable, Equatable, Identifiable {
    case user(UserMessage)
    case assistant(AssistantMessage)
    case toolResult(ToolResultMessage)

    var id: UUID {
        switch self {
        case .user(let msg): return msg.id
        case .assistant(let msg): return msg.id
        case .toolResult(let msg): return msg.id
        }
    }

    func toLLMMessage() -> LLMMessage? {
        switch self {
        case .user(let msg): return msg.toLLMMessage()
        case .assistant(let msg): return msg.toLLMMessage()
        case .toolResult(let msg): return msg.toLLMMessage()
        }
    }
}

// MARK: - CustomAgentMessage

/// Protocol for plugin/extension messages that carry a type discriminator.
protocol CustomAgentMessage: AgentMessageProtocol {
    var customType: String { get }
}

// MARK: - AgentMessage typealias

/// Existential type for any message in the agent pipeline.
typealias AgentMessage = any AgentMessageProtocol
