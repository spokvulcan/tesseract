import Foundation
import Observation

// MARK: - AgentPhase

/// Lightweight run phase — drives UI indicators for what the agent is doing right now.
enum AgentPhase: Sendable, Equatable {
    case idle
    case transformingContext(ContextTransformReason)
    case streaming
    case executingTool(String) // tool name
}

// MARK: - AgentState

/// Observable state container for the agent. SwiftUI views read properties directly
/// via the Observation framework — no `@Published` wrappers needed.
@MainActor
@Observable
final class AgentState {
    var systemPrompt: String = ""
    var model: AgentModelRef?
    var tools: [AgentToolDefinition] = []
    var messages: [any AgentMessageProtocol] = []
    var phase: AgentPhase = .idle
    var streamMessage: AssistantMessage?
    var pendingToolCalls: Set<String> = []
    var error: String?

    /// Convenience — true when the agent is doing anything (not idle).
    var isStreaming: Bool { phase != .idle }
}
