import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// Exercises the agent path's concrete **Generation Projection** through its only
/// interface: fold an event into a `GenerationAccumulator`, hand the folded state
/// to `step`, and assert the returned `Step`, plus `snapshot`/`finalize`. Never
/// reaches past the public surface — the projection's interface *is* the test
/// surface. Sibling to `CompletionProjectionTests` and `GenerationAccumulatorTests`.
struct AssistantMessageProjectionTests {

    /// Drive one event the way the driver does: `ingest` then `step` against the
    /// post-ingest accumulator. Returns the resulting `Step` and leaves `acc`/`proj`
    /// advanced so a sequence can be built up across calls.
    private func step(
        _ event: AgentGeneration,
        _ acc: inout GenerationAccumulator,
        _ proj: inout AssistantMessageProjection
    ) -> AssistantMessageProjection.Step {
        acc.ingest(event)
        return proj.step(event, acc)
    }

    // MARK: - Per-event Step coverage

    /// `.text` → `.update` carrying the new chunk as the text delta and a snapshot
    /// whose content is the raw accumulated text.
    @Test func textYieldsUpdateWithTextDelta() {
        var acc = GenerationAccumulator()
        var proj = AssistantMessageProjection()

        guard case .update(let message, let delta) = step(.text("Hello"), &acc, &proj) else {
            Issue.record("expected .update for .text")
            return
        }
        #expect(message.content == "Hello")
        #expect(delta.textDelta == "Hello")
        #expect(delta.thinkingDelta == nil)
        #expect(delta.toolCallDelta == nil)
    }

    /// `.thinking` → `.update` carrying the chunk as the thinking delta.
    @Test func thinkingYieldsUpdateWithThinkingDelta() {
        var acc = GenerationAccumulator()
        var proj = AssistantMessageProjection()
        _ = step(.thinkStart, &acc, &proj)

        guard case .update(let message, let delta) = step(.thinking("R"), &acc, &proj) else {
            Issue.record("expected .update for .thinking")
            return
        }
        #expect(message.thinking == "R")
        #expect(delta.thinkingDelta == "R")
        #expect(delta.textDelta == nil)
        #expect(delta.toolCallDelta == nil)
    }

    /// `.thinkReclassify` → `.update` with an empty delta, and the snapshot reflects
    /// the accumulator's append-after-text rule (buffered thinking lands after the
    /// pre-think text, thinking cleared).
    @Test func reclassifyYieldsUpdateWithEmptyDeltaAndAppendedContent() {
        var acc = GenerationAccumulator()
        var proj = AssistantMessageProjection()
        _ = step(.text("pre"), &acc, &proj)
        _ = step(.thinkStart, &acc, &proj)
        _ = step(.thinking("R"), &acc, &proj)

        guard case .update(let message, let delta) = step(.thinkReclassify, &acc, &proj) else {
            Issue.record("expected .update for .thinkReclassify")
            return
        }
        #expect(message.content == "preR")
        #expect(message.thinking == nil)
        #expect(delta.textDelta == nil)
        #expect(delta.thinkingDelta == nil)
        #expect(delta.toolCallDelta == nil)
    }

    /// `.thinkTruncate` → `.update` with an empty delta and the snapshot's thinking
    /// reset to the safe prefix.
    @Test func truncateYieldsUpdateWithSafePrefixThinking() {
        var acc = GenerationAccumulator()
        var proj = AssistantMessageProjection()
        _ = step(.thinkStart, &acc, &proj)
        _ = step(.thinking("garbage"), &acc, &proj)

        guard
            case .update(let message, let delta) =
                step(.thinkTruncate(safePrefix: "clean"), &acc, &proj)
        else {
            Issue.record("expected .update for .thinkTruncate")
            return
        }
        #expect(message.thinking == "clean")
        #expect(delta.textDelta == nil)
        #expect(delta.thinkingDelta == nil)
        #expect(delta.toolCallDelta == nil)
    }

    /// `.toolCall` → `.update` with identity assigned: the snapshot gains a
    /// `ToolCallInfo`, the delta's `toolCallId` equals that newest id, and the
    /// arguments are encoded through the normalizer.
    @Test func toolCallYieldsUpdateWithAssignedIdentity() throws {
        var acc = GenerationAccumulator()
        var proj = AssistantMessageProjection()
        let call = ToolCall(function: .init(name: "search", arguments: ["q": .string("hi")]))

        guard case .update(let message, let delta) = step(.toolCall(call), &acc, &proj) else {
            Issue.record("expected .update for .toolCall")
            return
        }
        let info = try #require(message.toolCalls.last)
        #expect(message.toolCalls.count == 1)
        #expect(info.name == "search")
        #expect(info.argumentsJSON == ToolArgumentNormalizer.encode(["q": .string("hi")]))
        // The streamed delta's id is exactly the snapshot's newest tool-call id.
        #expect(delta.toolCallDelta?.toolCallId == info.id)
        #expect(delta.toolCallDelta?.name == "search")
        #expect(delta.textDelta == nil)
        #expect(delta.thinkingDelta == nil)
    }

    /// `.malformedToolCall` → `.malformed(raw:)`, raw text preserved verbatim. No
    /// snapshot is emitted for this arm (the driver emits the distinct event).
    @Test func malformedToolCallYieldsMalformed() {
        var acc = GenerationAccumulator()
        var proj = AssistantMessageProjection()

        guard case .malformed(let raw) = step(.malformedToolCall("{bad"), &acc, &proj) else {
            Issue.record("expected .malformed for .malformedToolCall")
            return
        }
        #expect(raw == "{bad")
    }

    /// The four caller-concern events surface nothing.
    @Test func thinkStartEndDeltaInfoAreSilent() {
        var acc = GenerationAccumulator()
        var proj = AssistantMessageProjection()

        for event: AgentGeneration in [
            .thinkStart, .thinkEnd,
            .toolCallDelta(name: "f", argumentsDelta: "{partial"),
            .info(GenerationFixtures.info()),
        ] {
            guard case .silent = step(event, &acc, &proj) else {
                Issue.record("expected .silent for \(event)")
                return
            }
        }
    }

    // MARK: - Structural tool-call identity

    /// Across two `.toolCall` events: ids are distinct, the array accumulates in
    /// arrival order, and the first id stays stable in the later snapshot.
    @Test func toolCallIdentityIsDistinctStableAndAccumulating() {
        var acc = GenerationAccumulator()
        var proj = AssistantMessageProjection()
        let first = ToolCall(function: .init(name: "first", arguments: [:]))
        let second = ToolCall(function: .init(name: "second", arguments: [:]))

        guard case .update(let firstSnapshot, _) = step(.toolCall(first), &acc, &proj) else {
            Issue.record("expected .update for first .toolCall")
            return
        }
        let firstID = firstSnapshot.toolCalls.last?.id

        guard
            case .update(let secondSnapshot, let secondDelta) =
                step(.toolCall(second), &acc, &proj)
        else {
            Issue.record("expected .update for second .toolCall")
            return
        }

        // Accumulates in order.
        #expect(secondSnapshot.toolCalls.map(\.name) == ["first", "second"])
        // First id stable across the later snapshot.
        #expect(secondSnapshot.toolCalls.first?.id == firstID)
        // Ids distinct, and the newest delta points at the newest id.
        #expect(secondSnapshot.toolCalls.first?.id != secondSnapshot.toolCalls.last?.id)
        #expect(secondDelta.toolCallDelta?.toolCallId == secondSnapshot.toolCalls.last?.id)
    }

    // MARK: - snapshot vs finalize

    /// `snapshot` uses RAW text: a malformed-only turn snapshots as empty content
    /// (no fallback), which is what the cancel/error paths must preserve.
    @Test func snapshotUsesRawTextWithoutFallback() {
        var acc = GenerationAccumulator()
        var proj = AssistantMessageProjection()
        _ = step(.malformedToolCall("<tool_call>raw</tool_call>"), &acc, &proj)

        #expect(proj.snapshot(acc).content.isEmpty)
    }

    /// `finalize` applies the fallback exactly when `surfacesMalformedBuffer`: an
    /// otherwise-empty turn with a captured buffer surfaces the raw buffer.
    @Test func finalizeSurfacesMalformedBufferWhenTurnOtherwiseEmpty() {
        var acc = GenerationAccumulator()
        var proj = AssistantMessageProjection()
        _ = step(.malformedToolCall("<tool_call>raw</tool_call>"), &acc, &proj)

        #expect(acc.surfacesMalformedBuffer)
        #expect(proj.finalize(acc).content == "<tool_call>raw</tool_call>")
    }

    /// `finalize` leaves real text untouched even when a malformed fragment was
    /// also captured (the predicate is false once text is non-empty).
    @Test func finalizeKeepsRealTextOverMalformedBuffer() {
        var acc = GenerationAccumulator()
        var proj = AssistantMessageProjection()
        _ = step(.text("answer"), &acc, &proj)
        _ = step(.malformedToolCall("junk"), &acc, &proj)

        #expect(!acc.surfacesMalformedBuffer)
        #expect(proj.finalize(acc).content == "answer")
    }
}
