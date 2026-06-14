import Foundation
import MLXLMCommon
import Testing
import os

@testable import Tesseract_Agent

// MARK: - Scripted token stream

/// One UTF-8 byte per token, matching `FakeChatMLTokenizer`'s byte-level scheme.
private func tokens(for text: String) -> [Int] {
    Array(text.utf8).map(Int.init)
}

private func makeConfiguration() -> ModelConfiguration {
    ModelConfiguration(id: "test/loop", toolCallFormat: .json)
}

/// A finished upstream raw-token stream: the tokens followed by the
/// authoritative `.info` — exactly what `generateTokenTask` produces.
private func scriptedTokenStream(
    script: [Int],
    stopReason: GenerateStopReason,
    promptTime: TimeInterval = 0
) -> AsyncStream<TokenGeneration> {
    AsyncStream { continuation in
        for token in script {
            continuation.yield(.token(token))
        }
        continuation.yield(
            .info(
                GenerateCompletionInfo(
                    promptTokenCount: 7,
                    generationTokenCount: script.count,
                    promptTime: promptTime,
                    generationTime: 0.1,
                    stopReason: stopReason
                )))
        continuation.finish()
    }
}

private func collectEvents(
    script: [Int],
    stopReason: GenerateStopReason = .stop,
    promptTime: TimeInterval = 0
) async -> [RawGeneration] {
    let (stream, task) = TokenGenerationLoop.events(
        from: scriptedTokenStream(
            script: script, stopReason: stopReason, promptTime: promptTime),
        generationTask: nil,
        promptTokenCount: 7,
        modelConfiguration: makeConfiguration(),
        tokenizer: FakeChatMLTokenizer()
    )
    var events: [RawGeneration] = []
    for await event in stream { events.append(event) }
    await task.value
    return events
}

private extension [RawGeneration] {
    var joinedChunks: String {
        compactMap { if case .chunk(let t) = $0 { return t } else { return nil } }.joined()
    }
    var joinedDeltas: String {
        compactMap { if case .toolCallBufferDelta(let d) = $0 { return d } else { return nil } }
            .joined()
    }
    var toolCallNames: [String] {
        compactMap { if case .toolCall(let c) = $0 { return c.function.name } else { return nil } }
    }
    var completionInfo: GenerateCompletionInfo? {
        compactMap { if case .info(let i) = $0 { return i } else { return nil } }.first
    }
}

// MARK: - TokenGenerationLoop

struct TokenGenerationLoopTests {

    @Test func plainTextStreamsChunksAndPassesInfoThrough() async {
        let events = await collectEvents(
            script: tokens(for: "Hello world"),
            promptTime: 0.5
        )

        #expect(events.joinedChunks == "Hello world")
        #expect(events.joinedDeltas.isEmpty)
        #expect(events.toolCallNames.isEmpty)

        // Upstream's `.info` is authoritative and passes through untouched.
        let info = events.completionInfo
        #expect(info?.stopReason == .stop)
        #expect(info?.generationTokenCount == tokens(for: "Hello world").count)
        #expect(info?.promptTokenCount == 7)
        #expect(info?.promptTime == 0.5)
        // `.info` is the terminal event.
        guard case .info = events.last else {
            Issue.record("expected `.info` terminal event, got \(String(describing: events.last))")
            return
        }
    }

    @Test func taggedToolCallIsFilteredFromChunksAndEmittedOnce() async {
        let body = #"{"name": "read", "arguments": {"file_path": "/x"}}"#
        let events = await collectEvents(
            script: tokens(for: "Sure. <tool_call>\(body)</tool_call>")
        )

        #expect(events.joinedChunks == "Sure. ")
        #expect(events.toolCallNames == ["read"])

        // Delta stream carries the buffered block as it fills in: the first
        // delta includes the open tag, and the chunk completing the close tag
        // emits nothing (the authoritative `.toolCall` covers it). With the
        // byte-level tokenizer everything up to the final `>` streams out.
        #expect(events.joinedDeltas == "<tool_call>\(body)</tool_call")

        // The `.toolCall` arrives after all of its deltas.
        let lastDeltaIndex = events.lastIndex {
            if case .toolCallBufferDelta = $0 { return true } else { return false }
        }
        let callIndex = events.firstIndex {
            if case .toolCall = $0 { return true } else { return false }
        }
        #expect(lastDeltaIndex != nil && callIndex != nil)
        if let lastDeltaIndex, let callIndex {
            #expect(lastDeltaIndex < callIndex)
        }
    }

    @Test func unclosedParseableToolCallIsRecoveredAtEOS() async {
        // Generation ends (length) mid tool call, before the close tag: the
        // EOS recovery parses the buffered block and still emits the call.
        let events = await collectEvents(
            script: tokens(for: #"<tool_call>{"name": "read", "arguments": {}}"#),
            stopReason: .length
        )

        #expect(events.toolCallNames == ["read"])
        #expect(events.joinedChunks.isEmpty)
        #expect(events.completionInfo?.stopReason == .length)
    }

    @Test func unclosedMalformedToolCallBufferIsNotReemittedAsText() async {
        // The delta stream already carried the buffered bytes; re-emitting the
        // residual as a `.chunk` would duplicate them. The consumer
        // (GenerationStreamLoop) owns malformed-tool-call surfacing from the
        // concatenated deltas.
        let events = await collectEvents(
            script: tokens(for: "<tool_call>not json at all"),
            stopReason: .length
        )

        #expect(events.toolCallNames.isEmpty)
        #expect(events.joinedChunks.isEmpty)
        #expect(events.joinedDeltas.hasPrefix("<tool_call>"))
        #expect(events.completionInfo?.stopReason == .length)
    }

    @Test func partialStartTagResidualIsEmittedAsText() async {
        // Cut mid start tag: the tracker never emitted a delta for the
        // buffered bytes, so the EOS residual must come back as a regular
        // chunk — suppressing it would silently drop trailing text
        // (#67 review).
        let events = await collectEvents(
            script: tokens(for: "Compare: 1 <tool"),
            stopReason: .length
        )

        #expect(events.joinedChunks == "Compare: 1 <tool")
        #expect(events.joinedDeltas.isEmpty)
        #expect(events.toolCallNames.isEmpty)
    }

    @Test func bareJSONToolCallEmitsNoDeltasAndParses() async {
        // The processor's bare-JSON fallback parses the call mid-stream; the
        // tracker mirrors that state and stays silent (no spurious deltas
        // from the `<` inside the JSON, no suppressed text after it).
        let body = #"{"name": "read", "arguments": {"file_path": "/x<y"}}"#
        let events = await collectEvents(script: tokens(for: "Use \(body) ok"))

        #expect(events.toolCallNames == ["read"])
        #expect(events.joinedDeltas.isEmpty)
        #expect(events.joinedChunks == "Use  ok")
    }

    @Test func unclosedBareJSONResidualIsEmittedAsTextWithoutDeltas() async {
        // A bare-JSON buffer containing a literal start tag inside a string
        // value: the old tracker misread the `<` as a tag start, emitted the
        // processor-buffered bytes as spurious deltas, and then suppressed
        // the EOS residual — losing the text from the chunk stream
        // (#67 review).
        let buffered = #"{"a": "<tool_call>x""#
        let events = await collectEvents(
            script: tokens(for: buffered),
            stopReason: .length
        )

        #expect(events.joinedDeltas.isEmpty)
        #expect(events.toolCallNames.isEmpty)
        #expect(events.joinedChunks == buffered)
    }

    @Test func cancellingTheTaskCancelsUpstreamAndFinishesTheStream() async {
        // The production contract (`RawGenerationHandle.cancel()`): cancelling
        // the returned task must propagate to the upstream generation task and
        // still end the stream with a terminal `.info`.
        let (tokens, tokensContinuation) = AsyncStream<TokenGeneration>.makeStream()
        let upstreamCancelled = OSAllocatedUnfairLock(initialState: false)
        // Stand-in for upstream's generation task: runs until cancelled, then
        // finishes its stream — the upstream loop reacts to cancellation the
        // same way.
        let generationTask = Task {
            while !Task.isCancelled {
                await Task.yield()
            }
            upstreamCancelled.withLock { $0 = true }
            tokensContinuation.finish()
        }
        // A few tokens are already buffered so the consumer sees a chunk
        // before it cancels.
        for token in [Int](repeating: Int(UInt8(ascii: "a")), count: 4) {
            tokensContinuation.yield(.token(token))
        }

        let (stream, task) = TokenGenerationLoop.events(
            from: tokens,
            generationTask: generationTask,
            promptTokenCount: 1,
            modelConfiguration: makeConfiguration(),
            tokenizer: FakeChatMLTokenizer()
        )

        var sawChunk = false
        var info: GenerateCompletionInfo?
        for await event in stream {
            switch event {
            case .chunk:
                if !sawChunk {
                    sawChunk = true
                    task.cancel()
                }
            case .info(let i):
                info = i
            default:
                break
            }
        }
        await task.value

        #expect(sawChunk)
        #expect(upstreamCancelled.withLock { $0 })
        // No upstream `.info` arrived, so the mapping synthesizes a
        // cancellation one — the terminal event contract holds.
        #expect(info?.stopReason == .cancelled)
    }
}

// MARK: - ToolCallDeltaTracker

struct ToolCallDeltaTrackerTests {

    private func makeTracker() -> ToolCallDeltaTracker {
        ToolCallDeltaTracker(format: .json)
    }

    @Test func plainTextProducesNoDeltas() {
        var tracker = makeTracker()
        #expect(tracker.observe("hello ") == nil)
        #expect(tracker.observe("world") == nil)
        #expect(tracker.isMidToolCall == false)
    }

    @Test func firstDeltaIncludesOpenTagOnceConfirmed() {
        var tracker = makeTracker()
        // Partial start tag: ambiguous, nothing surfaces yet.
        #expect(tracker.observe("<tool") == nil)
        #expect(tracker.isMidToolCall)
        #expect(tracker.deltasCarriedBuffer == false)
        // Tag confirmed: the whole buffer (open tag included) is the delta.
        #expect(tracker.observe(#"_call>{"a":"#) == #"<tool_call>{"a":"#)
        #expect(tracker.deltasCarriedBuffer)
    }

    @Test func closeTagChunkEmitsNoDelta() {
        var tracker = makeTracker()
        #expect(tracker.observe("<tool_call>{}") == "<tool_call>{}")
        #expect(tracker.observe("</tool_call>") == nil)
        #expect(tracker.isMidToolCall == false)
    }

    @Test func falsePositiveStartIsFlushedWithoutDelta() {
        var tracker = makeTracker()
        #expect(tracker.observe("a < b") == nil)
        #expect(tracker.isMidToolCall == false)
        #expect(tracker.observe("still normal text") == nil)
    }

    @Test func trailingTextAfterCloseIsRescannedForNextCall() {
        var tracker = makeTracker()
        let delta = tracker.observe(#"<tool_call>{}</tool_call><tool_call>{"b""#)
        #expect(delta == #"<tool_call>{"b""#)
        #expect(tracker.isMidToolCall)
    }

    @Test func incrementalCollectingEmitsOnlyNewBytes() {
        var tracker = makeTracker()
        #expect(tracker.observe("<tool_call>") == "<tool_call>")
        #expect(tracker.observe(#"{"name""#) == #"{"name""#)
        #expect(tracker.observe(": 1}") == ": 1}")
        #expect(tracker.observe("</tool_call>") == nil)
    }

    @Test func bareJSONCollectionProducesNoDeltas() {
        var tracker = makeTracker()
        // The `<` inside the string value must not be misread as a tag start
        // — the processor is collecting this as bare JSON (#67 review).
        #expect(tracker.observe(#"{"a": "<tool_call>x"}"#) == nil)
        #expect(tracker.isMidToolCall == false)
    }

    @Test func unclosedBareJSONIsMidCallButCarriedNoDeltas() {
        var tracker = makeTracker()
        #expect(tracker.observe(#"{"a": "<tool"#) == nil)
        #expect(tracker.isMidToolCall)
        #expect(tracker.deltasCarriedBuffer == false)
    }

    @Test func invalidJSONPrefixFallsBackToTaggedCollection() {
        var tracker = makeTracker()
        // `{x` cannot begin a JSON object, so the processor prefers tagged
        // parsing from the later `<` — mirror that.
        #expect(tracker.observe(#"{x <tool_call>{"a""#) == #"<tool_call>{"a""#)
        #expect(tracker.deltasCarriedBuffer)
    }

    @Test func completedJSONObjectRescansTrailingForTaggedCall() {
        var tracker = makeTracker()
        // Bare-JSON object completes (the processor parses or flushes it),
        // then a tagged call starts in the same chunk.
        let delta = tracker.observe(#"{"a": 1} <tool_call>{"b""#)
        #expect(delta == #"<tool_call>{"b""#)
        #expect(tracker.isMidToolCall)
    }

    @Test func bareJSONSpanningChunksResolvesAndRescans() {
        var tracker = makeTracker()
        #expect(tracker.observe("{") == nil)
        #expect(tracker.isMidToolCall)
        #expect(tracker.observe(#""a": "<tool_call>x""#) == nil)
        // Object closes; trailing tagged call is picked up.
        #expect(tracker.observe("}<tool_call>") == "<tool_call>")
        #expect(tracker.deltasCarriedBuffer)
    }
}
