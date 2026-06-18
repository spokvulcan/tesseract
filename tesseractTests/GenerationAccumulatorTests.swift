import Testing
import MLXLMCommon
@testable import Tesseract_Agent

/// Exercises the Generation Accumulator through its only interface: feed an
/// `AgentGeneration` sequence via `ingest(_:)` and assert the accumulated state.
/// Never reaches into private storage — the interface is the test surface.
struct GenerationAccumulatorTests {

    /// Headline regression. When a `<think>` block never closes, buffered
    /// thinking must be appended *after* the text the model already produced
    /// — not prepended (the agent double-loop's historical ordering bug).
    @Test func reclassifyAppendsBufferedThinkingAfterPretext() {
        var acc = GenerationAccumulator()
        acc.ingest(.text("pre"))
        acc.ingest(.thinkStart)
        acc.ingest(.thinking("R"))
        acc.ingest(.thinkReclassify)

        #expect(acc.text == "preR")
        #expect(acc.thinking == nil)
    }

    /// The exact sequence both producers emit on a thinking-safeguard
    /// intervention: truncate to the safe prefix (discarding buffered garbage),
    /// inject the safeguard message, then end the block. The injection lands
    /// AFTER the safe prefix, and the safe-prefix length is recorded.
    @Test func safeguardSequenceTruncatesThenAppendsInjection() {
        var acc = GenerationAccumulator()
        acc.ingest(.thinkStart)
        acc.ingest(.thinking("loop loop loop"))  // pre-safeguard garbage
        acc.ingest(.thinkTruncate(safePrefix: "Step 1. "))
        acc.ingest(.thinking("(safeguard)"))
        acc.ingest(.thinkEnd)

        #expect(acc.thinking == "Step 1. (safeguard)")
        #expect(acc.safeguardSafePrefixChars == "Step 1. ".count)
    }

    /// `.thinkTruncate` on its own replaces thinking with the safe prefix,
    /// records its length, and flips the derived safeguard flag.
    @Test func truncateAloneResetsThinkingAndFlagsSafeguard() {
        var acc = GenerationAccumulator()
        acc.ingest(.thinkStart)
        acc.ingest(.thinking("garbage"))
        #expect(!acc.safeguardTriggered)

        acc.ingest(.thinkTruncate(safePrefix: "clean"))

        #expect(acc.thinking == "clean")
        #expect(acc.safeguardSafePrefixChars == 5)
        #expect(acc.safeguardTriggered)
    }

    /// `nil` (never opened) is distinct from `""` (opened, empty), and a
    /// closed block keeps its content rather than being nilled.
    @Test func thinkingIsNilUntilStartEmptyOnOpenAndKeptAcrossEnd() {
        var acc = GenerationAccumulator()
        #expect(acc.thinking == nil)
        acc.ingest(.thinkStart)
        #expect(acc.thinking?.isEmpty == true)
        acc.ingest(.thinkEnd)
        #expect(acc.thinking?.isEmpty == true)
    }

    /// Malformed tool-call fragments accumulate raw, and finalized tool calls
    /// append in arrival order as their raw event payloads.
    @Test func malformedAccumulatesRawAndToolCallsAppendInOrder() {
        var acc = GenerationAccumulator()
        acc.ingest(.malformedToolCall("{bad"))
        acc.ingest(.malformedToolCall("json}"))
        acc.ingest(.toolCall(GenerationFixtures.toolCall(name: "first")))
        acc.ingest(.toolCall(GenerationFixtures.toolCall(name: "second")))

        #expect(acc.malformedToolCallRaw == "{badjson}")
        #expect(acc.toolCalls.map { $0.function.name } == ["first", "second"])
    }

    /// The single home of the malformed→text fallback predicate, asserted at all
    /// four corners. True only when the turn produced no text and no successful
    /// tool calls but did capture a malformed `<tool_call>` buffer; any of those
    /// conditions failing makes it false. Both **Generation Projection**s consume
    /// this one query.
    @Test func surfacesMalformedBufferOnlyWhenOtherwiseEmptyWithBuffer() {
        // Otherwise-empty turn with a captured buffer → true.
        var bufferOnly = GenerationAccumulator()
        bufferOnly.ingest(.malformedToolCall("<tool_call>raw</tool_call>"))
        #expect(bufferOnly.surfacesMalformedBuffer)

        // Non-empty text → false (real content wins).
        var withText = GenerationAccumulator()
        withText.ingest(.text("answer"))
        withText.ingest(.malformedToolCall("junk"))
        #expect(!withText.surfacesMalformedBuffer)

        // A successful tool call present → false.
        var withToolCall = GenerationAccumulator()
        withToolCall.ingest(.toolCall(GenerationFixtures.toolCall(name: "read")))
        withToolCall.ingest(.malformedToolCall("junk"))
        #expect(!withToolCall.surfacesMalformedBuffer)

        // Empty buffer → false (nothing to surface).
        let empty = GenerationAccumulator()
        #expect(!empty.surfacesMalformedBuffer)
    }

    /// In-flight tool-call deltas and completion metrics are caller concerns;
    /// they leave accumulated turn content untouched.
    @Test func toolCallDeltaAndInfoAreInert() {
        var acc = GenerationAccumulator()
        acc.ingest(.text("hi"))
        acc.ingest(.toolCallDelta(name: "f", argumentsDelta: "{partial"))
        acc.ingest(.info(GenerationFixtures.info()))

        #expect(acc.text == "hi")
        #expect(acc.thinking == nil)
        #expect(acc.toolCalls.isEmpty)
        #expect(acc.malformedToolCallRaw.isEmpty)
        #expect(acc.safeguardSafePrefixChars == nil)
        #expect(!acc.safeguardTriggered)
    }

}
