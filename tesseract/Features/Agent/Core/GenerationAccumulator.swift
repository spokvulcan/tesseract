import Foundation
import MLXLMCommon

/// The single home for folding an `AgentGeneration` event stream into the
/// accumulated content of one assistant turn. A pure value: no side effects,
/// no control flow, no output type. Each consumer keeps its own `for await`
/// loop and its own **Generation Projection** from this state to its output
/// shape; only the shared accumulation transitions live here.
///
/// See `CONTEXT.md` → Language → *Generation accumulation*.
nonisolated struct GenerationAccumulator: Sendable {
    /// Accumulated assistant text for the turn.
    private(set) var text: String = ""

    /// `nil` = no `<think>` block was ever opened; `""` = one opened but has
    /// produced no content yet. Do not collapse the optionality.
    private(set) var thinking: String?

    /// Finalized tool calls in arrival order, as their raw event payloads.
    /// Consumers project these into their own shape (e.g. assigning stable ids).
    private(set) var toolCalls: [ToolCall] = []

    /// Raw text of any `<tool_call>` blocks whose JSON failed to parse.
    private(set) var malformedToolCallRaw: String = ""

    /// Character count of the safe prefix recorded when the thinking-loop
    /// safeguard fired (`nil` if it never fired). Lets a consumer emit its
    /// vendor sidecar without re-deriving the length.
    private(set) var safeguardSafePrefixChars: Int?

    /// Whether the thinking-loop safeguard intervened on this turn.
    var safeguardTriggered: Bool { safeguardSafePrefixChars != nil }

    /// Folds one generation event into the accumulated turn state.
    mutating func ingest(_ event: AgentGeneration) {
        switch event {
        case .text(let chunk):
            text += chunk
        case .thinkStart:
            thinking = thinking ?? ""
        case .thinking(let chunk):
            thinking = (thinking ?? "") + chunk
        case .thinkEnd:
            break
        case .thinkReclassify:
            // `<think>` never closed: reclassify buffered thinking as text by
            // appending it AFTER any pre-think text, then clear the buffer.
            text += (thinking ?? "")
            thinking = nil
        case .thinkTruncate(let safePrefix):
            // Safeguard fired: the safe prefix becomes the canonical reasoning,
            // discarding whatever was buffered up to the trigger.
            thinking = safePrefix
            safeguardSafePrefixChars = safePrefix.count
        case .toolCall(let call):
            toolCalls.append(call)
        case .malformedToolCall(let raw):
            malformedToolCallRaw += raw
        case .toolCallDelta, .info:
            break  // caller concerns (live UI deltas, completion metrics)
        }
    }
}
