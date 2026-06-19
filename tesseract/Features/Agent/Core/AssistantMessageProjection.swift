import Foundation
import MLXLMCommon

/// The agent path's concrete **Generation Projection** — a sibling to the
/// server's `CompletionProjection`. Where `CompletionProjection` maps a
/// *terminal* `GenerationAccumulator` to one HTTP response, this projection maps
/// the *per-event* accumulator state to what the streaming agent driver emits,
/// and the terminal state to the committed `AssistantMessage`.
///
/// A pure value with no `emit` and no logging: every side effect (the
/// `messageUpdate`/`malformedToolCall` emits, the warning log, the
/// surfaced-fallback info log) stays in the driver, exactly as
/// `CompletionProjection`'s effects stay in `CompletionHandler`. The driver folds
/// the shared `GenerationAccumulator` and hands its state in on each call; this
/// projection owns only the one thing the accumulator deliberately does not —
/// the turn's tool-call *identity* (`ToolCallInfo` with stable ids), which the
/// accumulator's raw `[ToolCall]` lacks.
///
/// `snapshot` returns the raw turn (used per-event and on cancel/error);
/// `finalize` applies the malformed→text fallback (terminal end-of-turn only).
/// The fallback predicate itself lives once on the accumulator
/// (`surfacesMalformedBuffer`), shared with `CompletionProjection`.
///
/// See `CONTEXT.md` → Language → *Generation Projection*.
nonisolated struct AssistantMessageProjection: Sendable {
    /// The turn's tool-call identity, owned here in arrival order. Each
    /// `.toolCall` event mints a `ToolCallInfo` with a fresh UUID; later snapshots
    /// keep earlier ids stable.
    private(set) var toolCalls: [ToolCallInfo] = []

    /// What the driver should do for one event, named by intent so the driver
    /// switches on the projection's decision rather than re-inspecting the raw
    /// event.
    enum Step: Sendable {
        /// Emit a `messageUpdate` carrying this raw snapshot and delta. Covers
        /// `.text`, `.thinking`, `.thinkReclassify`, `.thinkTruncate`, `.toolCall`.
        case update(AssistantMessage, AssistantStreamDelta)
        /// A `<tool_call>` block failed to parse: the driver logs the warning and
        /// emits the distinct `.malformedToolCall` event. No `messageUpdate`.
        case malformed(raw: String)
        /// Nothing to emit. Covers `.thinkStart`, `.thinkEnd`, `.toolCallDelta`,
        /// `.info` — caller concerns the projection does not surface.
        case silent
    }

    /// Project one already-folded event into the driver's next action. Call with
    /// the accumulator *after* `ingest(event)` so snapshots reflect the new state.
    /// Mutating because `.toolCall` assigns identity into `toolCalls`.
    mutating func step(_ event: AgentGeneration, _ accumulator: GenerationAccumulator) -> Step {
        switch event {
        case .text(let text):
            return .update(
                snapshot(accumulator),
                AssistantStreamDelta(textDelta: text, thinkingDelta: nil, toolCallDelta: nil)
            )

        case .thinking(let text):
            return .update(
                snapshot(accumulator),
                AssistantStreamDelta(textDelta: nil, thinkingDelta: text, toolCallDelta: nil)
            )

        case .thinkReclassify, .thinkTruncate:
            // The accumulator owns the content rule (reclassify appends buffered
            // thinking after any pre-think text; truncate resets to the safe
            // prefix). The projection just re-snapshots with an empty delta.
            return .update(
                snapshot(accumulator),
                AssistantStreamDelta(textDelta: nil, thinkingDelta: nil, toolCallDelta: nil)
            )

        case .toolCall(let call):
            let info = ToolCallInfo(
                id: UUID().uuidString,
                name: call.function.name,
                argumentsJSON: ToolArgumentNormalizer.encode(call.function.arguments)
            )
            toolCalls.append(info)
            return .update(
                snapshot(accumulator),
                AssistantStreamDelta(
                    textDelta: nil, thinkingDelta: nil,
                    toolCallDelta: ToolCallDelta(
                        toolCallId: info.id, name: info.name, argumentsDelta: nil
                    )
                )
            )

        case .malformedToolCall(let raw):
            return .malformed(raw: raw)

        case .thinkStart, .thinkEnd, .toolCallDelta, .info:
            return .silent
        }
    }

    /// The turn so far with **raw** content — accumulated text as-is, no fallback.
    /// Used for every per-event update and for the cancel/error paths, which must
    /// preserve the partial message exactly as produced.
    func snapshot(_ accumulator: GenerationAccumulator) -> AssistantMessage {
        AssistantMessage.fromStream(
            content: accumulator.text,
            thinking: accumulator.thinking,
            toolCalls: toolCalls
        )
    }

    /// The committed terminal message, with the malformed→text fallback applied:
    /// when `surfacesMalformedBuffer`, the dropped `<tool_call>` buffer becomes
    /// the content so the turn is persisted instead of dropped as contentless.
    /// Terminal end-of-turn only — never cancel/error, which use `snapshot`.
    func finalize(_ accumulator: GenerationAccumulator) -> AssistantMessage {
        AssistantMessage.fromStream(
            content: accumulator.surfacesMalformedBuffer
                ? accumulator.malformedToolCallRaw
                : accumulator.text,
            thinking: accumulator.thinking,
            toolCalls: toolCalls
        )
    }
}
