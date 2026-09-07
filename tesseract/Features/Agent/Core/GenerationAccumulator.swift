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

    /// The single home of the malformed→text fallback predicate: true when the
    /// turn produced no text and no successful tool calls but did capture a
    /// malformed `<tool_call>` buffer — the one case where a **Generation
    /// Projection** surfaces that raw buffer as the message content instead of
    /// dropping the turn as contentless. A derived `Bool` query (not an output
    /// shape), so it honors this value's "no output type" rule. Consumed by both
    /// `AssistantMessageProjection.finalize` and `CompletionProjection`, so the
    /// rule has one definition rather than two mirrored copies.
    var surfacesMalformedBuffer: Bool {
        toolCalls.isEmpty && text.isEmpty && !malformedToolCallRaw.isEmpty
    }

    /// Folds one generation event into the accumulated turn state.
    mutating func ingest(_ event: AgentGeneration) {
        switch event {
        case .text(let chunk):
            text += chunk
        case .thinkStart:
            thinking = thinking ?? ""
        case .thinking(let chunk):
            // In-place append through the optional — `(thinking ?? "") + chunk`
            // rebuilt the whole accumulated string on every token.
            if thinking == nil { thinking = "" }
            thinking? += chunk
        case .thinkEnd:
            break
        case .thinkReclassify:
            // `<think>` never closed: reclassify buffered thinking as text by
            // appending it AFTER any pre-think text, then clear the buffer
            // (the streamed mirror follows — reclassified thinking is text).
            text += (thinking ?? "")
            thinking = nil
        case .toolCall(let call):
            toolCalls.append(call)
        case .malformedToolCall(let raw):
            malformedToolCallRaw += raw
        case .toolCallDelta, .info:
            break  // caller concerns (live UI deltas, completion metrics)
        }
    }
}
