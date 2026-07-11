import Foundation
import Observation

// MARK: - AgentState

/// Observable state container for the agent: the committed message log plus
/// the busy bit the idle guards read. SwiftUI views read properties directly
/// via the Observation framework — no `@Published` wrappers needed.
///
/// Run-presentation detail (live stream, phase, pending tool calls) is NOT
/// here: the **Chat Session**'s event fold owns it (ADR-0024). This state
/// once mirrored that detail — a second fold nobody read.
@MainActor
@Observable
final class AgentState {
    var systemPrompt: String = ""
    var tools: [AgentToolDefinition] = []
    var messages: [any AgentMessageProtocol] = []

    /// True from run begin (or standalone compaction) until the envelope
    /// settles — owned by `Agent.beginRun`/`finishRun`, not the event fold.
    var isBusy: Bool = false
}
