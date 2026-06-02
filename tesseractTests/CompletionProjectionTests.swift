import Foundation
import MLXLMCommon
import Testing
@testable import Tesseract_Agent

/// Exercises the server's **Generation Projection** through its only interface:
/// build a terminal `GenerationAccumulator` from a synthetic `AgentGeneration`
/// sequence, then assert the projected value. Never reaches past the public
/// surface, never opens a socket or hops the MainActor — the projection's
/// interface *is* the test surface. Mirrors `GenerationAccumulatorTests`.
struct CompletionProjectionTests {

    // MARK: - finish_reason

    /// Tracer bullet: a turn that produced a tool call finishes as `.tool_calls`.
    @Test func toolCallsYieldToolCallsFinishReason() {
        var acc = GenerationAccumulator()
        acc.ingest(.toolCall(GenerationFixtures.toolCall(name: "bash")))

        let projection = CompletionProjection(
            accumulator: acc,
            info: GenerationFixtures.info(generationTokenCount: 5),
            maxTokens: 256,
            completionID: "chatcmpl-tracer"
        )

        #expect(projection.finishReason == .tool_calls)
        #expect(projection.toolCalls.map { $0.function.name } == ["bash"])
        #expect(projection.diagnostic.classification == .normal)
    }

    /// A generation that reached its max-tokens budget without tool calls
    /// finishes as `.length`, and the diagnostic carries the metrics the log
    /// records verbatim.
    @Test func lengthFinishReasonWhenGenerationHitsMaxTokens() {
        var acc = GenerationAccumulator()
        acc.ingest(.text("Truncated"))

        let projection = CompletionProjection(
            accumulator: acc,
            info: GenerationFixtures.info(generationTokenCount: 32, stopReason: .length),
            maxTokens: 32,
            completionID: "chatcmpl-len"
        )

        #expect(projection.finishReason == .length)
        #expect(projection.textContent == "Truncated")
        #expect(projection.diagnostic.classification == .normal)
        #expect(projection.diagnostic.generationTokenCount == 32)
        #expect(projection.diagnostic.maxTokens == 32)
        #expect(projection.diagnostic.stopReason == "length(maxTokens)")
    }

    /// A plain text stop is `.stop` and an ordinary `.normal` diagnostic, with no
    /// fallback surfaced.
    @Test func plainTextStopClassifiesNormal() {
        var acc = GenerationAccumulator()
        acc.ingest(.text("Hello"))

        let projection = CompletionProjection(
            accumulator: acc,
            info: GenerationFixtures.info(generationTokenCount: 5),
            maxTokens: 256,
            completionID: "chatcmpl-stop"
        )

        #expect(projection.finishReason == .stop)
        #expect(projection.diagnostic.classification == .normal)
        #expect(projection.malformedFallbackSurfaced == false)
    }

    // MARK: - Diagnostic classification

    /// An empty `.stop` turn that nonetheless produced reasoning is the
    /// empty-payload-with-reasoning case — the jundot/omlx#825 symptom.
    @Test func emptyStopWithReasoningClassifiesEmptyPayload() {
        var acc = GenerationAccumulator()
        acc.ingest(.thinkStart)
        acc.ingest(.thinking("I considered it."))
        acc.ingest(.thinkEnd)

        let projection = CompletionProjection(
            accumulator: acc,
            info: GenerationFixtures.info(generationTokenCount: 5),
            maxTokens: 256,
            completionID: "chatcmpl-empty"
        )

        #expect(projection.finishReason == .stop)
        #expect(projection.textContent.isEmpty)
        #expect(projection.thinkingContent == "I considered it.")
        #expect(projection.diagnostic.classification == .emptyPayloadWithReasoning)
    }

    /// Headline regression — the exact drift this projection fixes. A turn that
    /// stopped with empty text, no tool calls, and a dropped malformed
    /// `<tool_call>` buffer must (a) classify as `.malformedToolCallDropped` (a
    /// warning), and (b) surface the raw buffer as the text content. The old
    /// non-streaming path classified this *after* the fallback had already
    /// repopulated the text, so it logged plain `info` and never carried the
    /// malformed branch or `malformedLen` field — this case made executable.
    @Test func emptyStopWithDroppedMalformedToolCallSurfacesBufferAndWarns() {
        let rawBuffer = #"{"name":"bash","arguments":{"command":"#
        var acc = GenerationAccumulator()
        acc.ingest(.malformedToolCall(rawBuffer))

        let projection = CompletionProjection(
            accumulator: acc,
            info: GenerationFixtures.info(generationTokenCount: 7),
            maxTokens: 256,
            completionID: "chatcmpl-malformed"
        )

        #expect(projection.finishReason == .stop)
        #expect(projection.malformedFallbackSurfaced == true)
        #expect(projection.textContent == rawBuffer)
        #expect(projection.diagnostic.classification == .malformedToolCallDropped)
        #expect(projection.diagnostic.malformedLen == rawBuffer.count)
        // textLen records the RAW pre-fallback text (empty), proving the
        // diagnostic classified before the fallback moved the buffer into the
        // content — the precise ordering the old non-streaming path got wrong.
        #expect(projection.diagnostic.textLen == 0)
    }

    /// When a malformed buffer is dropped but the turn DID produce text, the
    /// diagnostic still flags the dropped call, yet the fallback does not fire —
    /// the existing text stands and is never overwritten.
    @Test func malformedWithTextFlagsDroppedCallButKeepsText() {
        var acc = GenerationAccumulator()
        acc.ingest(.text("Here is the answer."))
        acc.ingest(.malformedToolCall("{broken"))

        let projection = CompletionProjection(
            accumulator: acc,
            info: GenerationFixtures.info(generationTokenCount: 9),
            maxTokens: 256,
            completionID: "chatcmpl-mixed"
        )

        #expect(projection.textContent == "Here is the answer.")
        #expect(projection.malformedFallbackSurfaced == false)
        #expect(projection.diagnostic.classification == .malformedToolCallDropped)
    }

    /// Precedence: when an empty `.stop` turn carries BOTH reasoning and a
    /// dropped malformed buffer, empty-payload-with-reasoning wins the
    /// classification (matching the streaming path's historical `if / else if`),
    /// while the fallback independently still surfaces the buffer as text.
    @Test func emptyPayloadWithReasoningOutranksMalformedDropped() {
        var acc = GenerationAccumulator()
        acc.ingest(.thinkStart)
        acc.ingest(.thinking("Reasoned but emitted nothing."))
        acc.ingest(.thinkEnd)
        acc.ingest(.malformedToolCall("{broken"))

        let projection = CompletionProjection(
            accumulator: acc,
            info: GenerationFixtures.info(generationTokenCount: 4),
            maxTokens: 256,
            completionID: "chatcmpl-both"
        )

        #expect(projection.diagnostic.classification == .emptyPayloadWithReasoning)
        #expect(projection.malformedFallbackSurfaced == true)
        #expect(projection.textContent == "{broken")
    }

    // MARK: - thinking nil-vs-empty boundary

    /// `thinking == nil` (never opened) and `thinking == ""` (opened, empty) both
    /// project to an empty `thinkingContent`; the optional is preserved up to
    /// this boundary and collapsed no earlier.
    @Test func thinkingNilAndEmptyBothProjectToEmptyContent() {
        var never = GenerationAccumulator()
        never.ingest(.text("a"))
        var opened = GenerationAccumulator()
        opened.ingest(.thinkStart)   // thinking == ""
        opened.ingest(.text("a"))

        let info = GenerationFixtures.info(generationTokenCount: 1)
        let p1 = CompletionProjection(accumulator: never, info: info, maxTokens: 256, completionID: "n")
        let p2 = CompletionProjection(accumulator: opened, info: info, maxTokens: 256, completionID: "o")

        #expect(p1.thinkingContent == "")
        #expect(p2.thinkingContent == "")
    }

    // MARK: - Safeguard sidecar

    /// The safeguard sidecar is present with the recorded safe-prefix length when
    /// the thinking-loop safeguard fired, and absent when it did not.
    @Test func safeguardReportPresentOnlyWhenSafeguardFired() {
        var fired = GenerationAccumulator()
        fired.ingest(.thinkStart)
        fired.ingest(.thinking("loop loop"))
        fired.ingest(.thinkTruncate(safePrefix: "Step 1."))

        var quiet = GenerationAccumulator()
        quiet.ingest(.text("done"))

        let info = GenerationFixtures.info(generationTokenCount: 3)
        let firedP = CompletionProjection(accumulator: fired, info: info, maxTokens: 256, completionID: "f")
        let quietP = CompletionProjection(accumulator: quiet, info: info, maxTokens: 256, completionID: "q")

        #expect(firedP.safeguardReport?.safe_prefix_chars == "Step 1.".count)
        #expect(quietP.safeguardReport == nil)
    }

    // MARK: - Missing metrics

    /// With no terminal `.info` and no max-tokens, the turn stops and the
    /// diagnostic records the sentinel fields the log prints (`0` / `-1` /
    /// `"nil"`).
    @Test func nilInfoProducesStopWithSentinelDiagnosticFields() {
        var acc = GenerationAccumulator()
        acc.ingest(.text("hi"))

        let projection = CompletionProjection(
            accumulator: acc,
            info: nil,
            maxTokens: nil,
            completionID: "chatcmpl-noinfo"
        )

        #expect(projection.finishReason == .stop)
        #expect(projection.info == nil)
        #expect(projection.diagnostic.generationTokenCount == 0)
        #expect(projection.diagnostic.maxTokens == -1)
        #expect(projection.diagnostic.stopReason == "nil")
    }

    // MARK: - Rendered line & severity

    /// The malformed-dropped case renders a `warning` line carrying the
    /// `malformedLen=` field and the MALFORMED suffix — the exact log artifact
    /// that drifted between the two paths in #46, now pinned at the render layer
    /// (not just the classification).
    @Test func diagnosticRendersMalformedLineAtWarningWithMalformedLen() {
        var acc = GenerationAccumulator()
        acc.ingest(.malformedToolCall(#"{"name":"bash""#))
        let diagnostic = CompletionProjection(
            accumulator: acc,
            info: GenerationFixtures.info(generationTokenCount: 7),
            maxTokens: 256,
            completionID: "chatcmpl-render-malformed"
        ).diagnostic

        #expect(diagnostic.severity == .warning)
        let line = diagnostic.renderedLine(label: "non-streaming")
        #expect(line.contains("HTTP non-streaming finish_reason decision"))
        #expect(line.contains("malformedLen=\(diagnostic.malformedLen)"))
        #expect(line.hasSuffix("— MALFORMED TOOL CALL DROPPED"))
    }

    /// An empty `.stop` with reasoning renders the EMPTY PAYLOAD warning line.
    @Test func diagnosticRendersEmptyPayloadLineAtWarning() {
        var acc = GenerationAccumulator()
        acc.ingest(.thinkStart)
        acc.ingest(.thinking("Reasoned."))
        acc.ingest(.thinkEnd)
        let diagnostic = CompletionProjection(
            accumulator: acc,
            info: GenerationFixtures.info(generationTokenCount: 3),
            maxTokens: 256,
            completionID: "chatcmpl-render-empty"
        ).diagnostic

        #expect(diagnostic.severity == .warning)
        #expect(
            diagnostic.renderedLine(label: "streaming")
                .hasSuffix("— EMPTY PAYLOAD WITH REASONING")
        )
    }

    /// A normal turn renders an `info` line with no classification suffix, and
    /// the `label` distinguishes the path.
    @Test func diagnosticRendersNormalLineAtInfoWithoutSuffix() {
        var acc = GenerationAccumulator()
        acc.ingest(.text("Hello"))
        let diagnostic = CompletionProjection(
            accumulator: acc,
            info: GenerationFixtures.info(generationTokenCount: 5),
            maxTokens: 256,
            completionID: "chatcmpl-render-normal"
        ).diagnostic

        #expect(diagnostic.severity == .info)
        let line = diagnostic.renderedLine(label: "streaming")
        #expect(line.contains("HTTP streaming finish_reason decision"))
        #expect(line.contains("malformedLen=0"))
        #expect(!line.contains("MALFORMED TOOL CALL DROPPED"))
        #expect(!line.contains("EMPTY PAYLOAD WITH REASONING"))
    }
}
