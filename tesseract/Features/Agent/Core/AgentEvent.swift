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

// MARK: - AssistantMessageEvent

/// The assistant-message stream protocol — pi-ai's `AssistantMessageEvent`,
/// verbatim (ADR-0024): `start`, then part-scoped `*_start / *_delta / *_end`
/// events addressed by `contentIndex` into the partial message's `content`,
/// terminated by `done` (stop / length / toolUse) or `error` (error /
/// aborted). Every event carries the partial `AssistantMessage` so a consumer
/// can always resync its fold from the snapshot.
nonisolated enum AssistantMessageEvent: Sendable {
    case start(partial: AssistantMessage)
    case textStart(contentIndex: Int, partial: AssistantMessage)
    case textDelta(contentIndex: Int, delta: String, partial: AssistantMessage)
    case textEnd(contentIndex: Int, content: String, partial: AssistantMessage)
    case thinkingStart(contentIndex: Int, partial: AssistantMessage)
    case thinkingDelta(contentIndex: Int, delta: String, partial: AssistantMessage)
    case thinkingEnd(contentIndex: Int, content: String, partial: AssistantMessage)
    case toolcallStart(contentIndex: Int, partial: AssistantMessage)
    case toolcallDelta(contentIndex: Int, delta: String, partial: AssistantMessage)
    case toolcallEnd(contentIndex: Int, toolCall: ToolCallPart, partial: AssistantMessage)
    /// Terminal success. `reason` is `.stop`, `.length`, or `.toolUse`.
    case done(reason: StopReason, message: AssistantMessage)
    /// Terminal failure. `reason` is `.error` or `.aborted`; the message
    /// preserves the partial content produced before the failure.
    case error(reason: StopReason, error: AssistantMessage)

    /// The message snapshot this event carries, regardless of case.
    var partial: AssistantMessage {
        switch self {
        case .start(let partial),
            .textStart(_, let partial),
            .textDelta(_, _, let partial),
            .textEnd(_, _, let partial),
            .thinkingStart(_, let partial),
            .thinkingDelta(_, _, let partial),
            .thinkingEnd(_, _, let partial),
            .toolcallStart(_, let partial),
            .toolcallDelta(_, _, let partial),
            .toolcallEnd(_, _, let partial):
            return partial
        case .done(_, let message):
            return message
        case .error(_, let error):
            return error
        }
    }
}

// MARK: - AgentEvent

/// Events emitted by the agent loop for observers (UI, logging, extensions).
enum AgentEvent: Sendable {
    // -- Agent lifecycle --
    case agentStart
    case agentEnd(messages: [any AgentMessageProtocol & Sendable])

    // -- Failure --
    /// A generation failure the user must see (e.g. a vision-tower rejection).
    /// Emitted alongside the terminal turn/agent-end so observers can surface the
    /// message. The in-app path routes it to the shared error banner; the HTTP
    /// path surfaces the same failure separately as a thrown error on its
    /// `AgentGeneration` stream (it does not observe this event).
    case generationError(message: String)

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
    /// Streaming update for the in-flight assistant message: the pi-ai stream
    /// event plus the partial message it carries (pi-mono's `message_update`).
    case messageUpdate(message: AssistantMessage, event: AssistantMessageEvent)
    case messageEnd(message: any AgentMessageProtocol & Sendable)
    case malformedToolCall(raw: String)

    // -- Tool execution --
    case toolExecutionStart(toolCallId: String, toolName: String, argsJSON: String)
    case toolExecutionUpdate(toolCallId: String, toolName: String, result: AgentToolResult)
    case toolExecutionEnd(
        toolCallId: String, toolName: String, result: AgentToolResult, isError: Bool)
}
