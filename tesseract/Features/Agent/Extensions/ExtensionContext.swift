import Foundation

// MARK: - ExtensionContext

/// Context passed to extension handlers when events fire.
/// Provides read-only access to agent state and limited control operations.
/// A concrete implementation is created during Epic 6 integration.
protocol ExtensionContext: Sendable {
    /// Working directory / agent root.
    var cwd: String { get }

    /// Current model (if loaded).
    var model: AgentModelRef? { get }

    /// Whether the agent is currently idle (not generating).
    func isIdle() -> Bool

    /// Abort the current generation.
    func abort()

    /// Get the current system prompt.
    func getSystemPrompt() -> String

    /// Get context window usage statistics.
    func getContextUsage() -> ContextUsage?

    /// Trigger context compaction.
    func compact(options: CompactOptions?)
}

// MARK: - ContextUsage

/// Token usage statistics for the current context window.
nonisolated struct ContextUsage: Sendable {
    let estimatedTokens: Int
    let contextWindow: Int
    let percentUsed: Double
}

// MARK: - CompactOptions

/// Options for triggering context compaction.
nonisolated struct CompactOptions: Sendable {
    /// Compact even if under the normal threshold.
    let force: Bool
}
