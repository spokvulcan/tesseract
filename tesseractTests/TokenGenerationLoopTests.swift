import Foundation
import MLXLMCommon
import Testing
import os

@testable import Tesseract_Agent

// MARK: - Scripted token source

/// A deterministic `TokenIteratorProtocol` that replays a fixed token script —
/// lets the tests drive `TokenGenerationLoop` without a model or GPU. The
/// `consumed` counter is shared so a test can observe how far the loop got
/// after the iterator itself has been consumed by the loop.
private struct ScriptedTokenIterator: TokenIteratorProtocol {
    private let script: [Int]
    private var index = 0
    let maxTokens: Int?
    private(set) var tokenCount = 0
    let promptPrefillTime: TimeInterval
    let consumed: OSAllocatedUnfairLock<Int>

    init(
        script: [Int], maxTokens: Int? = nil, promptPrefillTime: TimeInterval = 0,
        consumed: OSAllocatedUnfairLock<Int> = .init(initialState: 0)
    ) {
        self.script = script
        self.maxTokens = maxTokens
        self.promptPrefillTime = promptPrefillTime
        self.consumed = consumed
    }

    mutating func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens { return nil }
        guard index < script.count else { return nil }
        let token = script[index]
        index += 1
        tokenCount += 1
        consumed.withLock { $0 += 1 }
        return token
    }
}

/// One UTF-8 byte per token, matching `FakeChatMLTokenizer`'s byte-level scheme.
private func tokens(for text: String) -> [Int] {
    Array(text.utf8).map(Int.init)
}

private let eosToken = 0

private func makeConfiguration() -> ModelConfiguration {
    var configuration = ModelConfiguration(id: "test/loop", toolCallFormat: .json)
    configuration.eosTokenIds = [eosToken]
    return configuration
}

private func collectEvents(
    script: [Int], maxTokens: Int? = nil, promptPrefillTime: TimeInterval = 0
) async -> [RawGeneration] {
    let (stream, task) = TokenGenerationLoop.start(
        promptTokenCount: 7,
        modelConfiguration: makeConfiguration(),
        tokenizer: FakeChatMLTokenizer(),
        iterator: ScriptedTokenIterator(
            script: script, maxTokens: maxTokens, promptPrefillTime: promptPrefillTime)
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

    @Test func plainTextStreamsChunksAndStopsOnEOSToken() async {
        let events = await collectEvents(
            script: tokens(for: "Hello world") + [eosToken],
            promptPrefillTime: 0.5
        )

        #expect(events.joinedChunks == "Hello world")
        #expect(events.joinedDeltas.isEmpty)
        #expect(events.toolCallNames.isEmpty)

        let info = events.completionInfo
        #expect(info?.stopReason == .stop)
        #expect(info?.generationTokenCount == tokens(for: "Hello world").count)
        #expect(info?.promptTokenCount == 7)
        // promptTime folds in the iterator's chunked-prefill time.
        #expect((info?.promptTime ?? 0) >= 0.5)
        // `.info` is the terminal event.
        guard case .info = events.last else {
            Issue.record("expected `.info` terminal event, got \(String(describing: events.last))")
            return
        }
    }

    @Test func taggedToolCallIsFilteredFromChunksAndEmittedOnce() async {
        let body = #"{"name": "read", "arguments": {"file_path": "/x"}}"#
        let events = await collectEvents(
            script: tokens(for: "Sure. <tool_call>\(body)</tool_call>") + [eosToken]
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
        let script = tokens(for: #"<tool_call>{"name": "read", "arguments": {}}"#)
        let events = await collectEvents(script: script, maxTokens: script.count)

        #expect(events.toolCallNames == ["read"])
        #expect(events.joinedChunks.isEmpty)
        #expect(events.completionInfo?.stopReason == .length)
    }

    @Test func unclosedMalformedToolCallBufferIsNotReemittedAsText() async {
        // The delta stream already carried the buffered bytes; re-emitting the
        // residual as a `.chunk` would duplicate them. The consumer
        // (GenerationStreamLoop) owns malformed-tool-call surfacing from the
        // concatenated deltas.
        let script = tokens(for: "<tool_call>not json at all")
        let events = await collectEvents(script: script, maxTokens: script.count)

        #expect(events.toolCallNames.isEmpty)
        #expect(events.joinedChunks.isEmpty)
        #expect(events.joinedDeltas.hasPrefix("<tool_call>"))
        #expect(events.completionInfo?.stopReason == .length)
    }

    @Test func cancellingTheTaskStopsTheLoopEarly() async {
        // The production contract (`RawGenerationHandle.cancel()`): cancelling
        // the returned task stops the loop mid-script. The script alternates in
        // newlines so the byte-level detokenizer keeps resegmenting (stays
        // linear) and the loop spins long enough for the cancel to land mid-run.
        let consumed = OSAllocatedUnfairLock<Int>(initialState: 0)
        let scriptLength = 200_000
        let (stream, task) = TokenGenerationLoop.start(
            promptTokenCount: 1,
            modelConfiguration: makeConfiguration(),
            tokenizer: FakeChatMLTokenizer(),
            iterator: ScriptedTokenIterator(
                script: Array(repeating: tokens(for: "a\n"), count: scriptLength / 2)
                    .flatMap { $0 },
                consumed: consumed)
        )

        var sawChunk = false
        for await event in stream {
            if case .chunk = event, !sawChunk {
                sawChunk = true
                task.cancel()
            }
        }
        await task.value

        #expect(sawChunk)
        // The loop must have stopped well before draining the whole script.
        #expect(consumed.withLock { $0 } < scriptLength)
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
        // Tag confirmed: the whole buffer (open tag included) is the delta.
        #expect(tracker.observe(#"_call>{"a":"#) == #"<tool_call>{"a":"#)
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
}
