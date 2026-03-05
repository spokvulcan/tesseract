import Foundation

// MARK: - ContextTransformReason

/// Why a context transformation was applied.
enum ContextTransformReason: Sendable, Equatable {
    case compaction
    case extensionTransform(String)
}

// MARK: - ContextTransformResult

/// Outcome of a context transformation pass.
struct ContextTransformResult: Sendable {
    let messages: [any AgentMessageProtocol & Sendable]
    let didMutate: Bool
    let reason: ContextTransformReason
}

// MARK: - ToolCallDelta

/// Incremental streaming update for a single tool call being parsed.
struct ToolCallDelta: Sendable {
    let toolCallId: String
    let name: String?
    let argumentsDelta: String?
}

// MARK: - AssistantStreamDelta

/// A single streaming chunk from the LLM response.
struct AssistantStreamDelta: Sendable {
    let textDelta: String?
    let thinkingDelta: String?
    let toolCallDelta: ToolCallDelta?
}

// MARK: - AgentEvent

/// Events emitted by the agent loop for observers (UI, logging, extensions).
enum AgentEvent: Sendable {
    // -- Agent lifecycle --
    case agentStart
    case agentEnd(messages: [any AgentMessageProtocol & Sendable])

    // -- Turn lifecycle --
    case turnStart
    case turnEnd(
        message: AssistantMessage,
        toolResults: [ToolResultMessage],
        contextMessages: [any AgentMessageProtocol & Sendable]
    )

    // -- Context transform lifecycle --
    case contextTransformStart(reason: ContextTransformReason)
    case contextTransformEnd(
        reason: ContextTransformReason,
        didMutate: Bool,
        messages: [any AgentMessageProtocol & Sendable]?
    )

    // -- Message lifecycle --
    case messageStart(message: any AgentMessageProtocol & Sendable)
    case messageUpdate(message: AssistantMessage, streamDelta: AssistantStreamDelta)
    case messageEnd(message: any AgentMessageProtocol & Sendable)

    // -- Tool execution --
    case toolExecutionStart(toolCallId: String, toolName: String, argsJSON: String)
    case toolExecutionUpdate(toolCallId: String, toolName: String, result: AgentToolResult)
    case toolExecutionEnd(toolCallId: String, toolName: String, result: AgentToolResult, isError: Bool)
}
