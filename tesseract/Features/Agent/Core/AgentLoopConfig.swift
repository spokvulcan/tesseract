import Foundation

// MARK: - AgentModelRef

/// Placeholder for the model identifier. Connected to AgentEngine/LLMActor in Epic 2.
struct AgentModelRef: Sendable {
    let id: String
}

// MARK: - ContextTransformConfig

/// Pairs a declared reason with the transform closure so the loop can emit
/// `contextTransformStart(reason:)` before awaiting the async transform.
struct ContextTransformConfig: Sendable {
    let reason: ContextTransformReason
    let transform: @Sendable (
        _ messages: [any AgentMessageProtocol],
        _ signal: CancellationToken?
    ) async -> ContextTransformResult
}

// MARK: - AgentLoopConfig

/// Configuration for the agent double-loop. Injected once at run start.
struct AgentLoopConfig: Sendable {
    /// Model to use for generation.
    let model: AgentModelRef

    /// Converts the authoritative message array to LLM-ready messages at the call boundary.
    let convertToLlm: @Sendable ([any AgentMessageProtocol]) -> [LLMMessage]

    /// Optional context transform (compaction, pruning, extension injection).
    /// Runs before `convertToLlm` each turn. If `didMutate` is true, the loop
    /// writes the result back into the authoritative context.
    let contextTransform: ContextTransformConfig?

    /// Optional: returns steering messages (user interrupts mid-run).
    let getSteeringMessages: (@Sendable () async -> [any AgentMessageProtocol])?

    /// Optional: returns follow-up messages (queued for after the agent finishes).
    let getFollowUpMessages: (@Sendable () async -> [any AgentMessageProtocol])?
}
