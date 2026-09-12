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
    promptTime: TimeInterval = 0,
    tokenizer: any Tokenizer = FakeChatMLTokenizer(),
    modelConfiguration: ModelConfiguration = makeConfiguration()
) async -> [RawGeneration] {
    let (stream, task) = TokenGenerationLoop.events(
        from: scriptedTokenStream(
            script: script, stopReason: stopReason, promptTime: promptTime),
        generationTask: nil,
        promptTokenCount: 7,
        modelConfiguration: modelConfiguration,
        tokenizer: tokenizer
    )
    var events: [RawGeneration] = []
    for await event in stream { events.append(event) }
    await task.value
    return events
}

private extension [RawGeneration] {
    var chunks: [String] {
        compactMap { if case .chunk(let t) = $0 { return t } else { return nil } }
    }
    var joinedChunks: String {
        chunks.joined()
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
        // The first block's body deltas out in full (open tag included, close
        // tag excluded), then the trailing tagged call is picked up.
        let delta = tracker.observe(#"<tool_call>{}</tool_call><tool_call>{"b""#)
        #expect(delta == #"<tool_call>{}<tool_call>{"b""#)
        #expect(tracker.isMidToolCall)
    }

    @Test func closeTagChunkDeltasItsBodyBytesButNotTheTag() {
        var tracker = makeTracker()
        #expect(tracker.observe(#"<tool_call>{"a":"#) == #"<tool_call>{"a":"#)
        // The chunk carrying the close tag still deltas the body bytes that
        // precede it — the Argument Transcoder needs the complete body on
        // this channel — while the tag itself never surfaces.
        #expect(tracker.observe(#" 1}</tool_call>"#) == " 1}")
        #expect(tracker.isMidToolCall == false)
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

    // MARK: - Live detokenizer streaming parity & delivery latency

    private static func naiveDetokenizedText(script: [Int], tokenizer: any Tokenizer) -> String {
        var detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
        var text = ""
        for token in script {
            detokenizer.append(token: token)
            if let chunk = detokenizer.next() {
                text += chunk
            }
        }
        return text
    }

    @Test func streamedChunksMatchNaiveDetokenizerAcrossFidelityShapes() async {
        let tokenizer = FakeChatMLTokenizer()
        let shapes: [(name: String, text: String)] = [
            ("prose", "The quick brown fox jumps over the lazy dog.\n\nSecond paragraph here with some numbers: 12345."),
            ("cjk and emoji", "Hello 😀 🏳️‍🌈 日本語テキスト 中文测试 🇺🇸 and more text"),
            ("code", "```swift\nfunc greet(name: String) -> String {\n    return \"Hello, \\(name)!\"\n}\n```"),
            ("long newline-free run", String(repeating: "abcdefghijklmnopqrstuvwxyz0123456789", count: 25)),
        ]

        for shape in shapes {
            let script = tokens(for: shape.text)
            let events = await collectEvents(script: script, tokenizer: tokenizer)
            let naive = Self.naiveDetokenizedText(script: script, tokenizer: tokenizer)
            #expect(
                events.joinedChunks == naive,
                "shape '\(shape.name)' mismatch: got \(events.joinedChunks), expected \(naive)"
            )
        }

        // Ill-formed UTF-8 bytes shape
        let illFormedTokens = [0xE0, 0x80, 0x80, 0xED, 0xA0, 0x80, 0xC3, 0xFF]
        let illEvents = await collectEvents(script: illFormedTokens, tokenizer: tokenizer)
        let illNaive = Self.naiveDetokenizedText(script: illFormedTokens, tokenizer: tokenizer)
        #expect(illEvents.joinedChunks == illNaive)

        // Tagged tool call shape: live loop extracts the tool call and text
        let toolCallText = "Let me check.\n<tool_call>{\"name\": \"test_tool\", \"arguments\": {\"param\": 123}}</tool_call>\nDone."
        let tcScript = tokens(for: toolCallText)
        let tcEvents = await collectEvents(script: tcScript, tokenizer: tokenizer)
        #expect(tcEvents.toolCallNames == ["test_tool"])
        #expect(tcEvents.joinedChunks == "Let me check.\n\nDone.")
    }

    @Test func deliveryLatencyEmitsChunksImmediatelyForNewlineFreeRun() async {
        // Assert that chunks are emitted incrementally during token iteration,
        // rather than held until end-of-stream or a newline.
        let (inStream, inContinuation) = AsyncStream<TokenGeneration>.makeStream()
        let (outStream, task) = TokenGenerationLoop.events(
            from: inStream,
            generationTask: nil,
            promptTokenCount: 7,
            modelConfiguration: makeConfiguration(),
            tokenizer: FakeChatMLTokenizer()
        )

        var iterator = outStream.makeAsyncIterator()

        // Feed first token of a newline-free string: "a"
        inContinuation.yield(.token(Int(Character("a").asciiValue!)))

        // Next event on outStream should arrive immediately without waiting for stream finish
        let firstEvent = await iterator.next()
        guard case .chunk(let text) = firstEvent else {
            Issue.record("expected .chunk on first token, got \(String(describing: firstEvent))")
            return
        }
        #expect(text == "a")

        // Feed second token: "b"
        inContinuation.yield(.token(Int(Character("b").asciiValue!)))
        let secondEvent = await iterator.next()
        guard case .chunk(let text2) = secondEvent else {
            Issue.record("expected .chunk on second token, got \(String(describing: secondEvent))")
            return
        }
        #expect(text2 == "b")

        // Close stream
        inContinuation.yield(.info(GenerateCompletionInfo(
            promptTokenCount: 7,
            generationTokenCount: 2,
            promptTime: 0,
            generationTime: 0.1,
            stopReason: .stop
        )))
        inContinuation.finish()

        let thirdEvent = await iterator.next()
        guard case .info = thirdEvent else {
            Issue.record("expected .info terminal event, got \(String(describing: thirdEvent))")
            return
        }
        await task.value
    }

    @Test func deliveryLatencyHoldsIncompleteScalarUntilCompleted() async {
        // A multi-byte scalar (4-byte 😀 = 0xF0, 0x9F, 0x98, 0x80)
        let (inStream, inContinuation) = AsyncStream<TokenGeneration>.makeStream()
        let (outStream, task) = TokenGenerationLoop.events(
            from: inStream,
            generationTask: nil,
            promptTokenCount: 7,
            modelConfiguration: makeConfiguration(),
            tokenizer: FakeChatMLTokenizer()
        )

        var iterator = outStream.makeAsyncIterator()

        // Yield first 3 bytes of emoji: incomplete scalar, must not emit
        inContinuation.yield(.token(0xF0))
        inContinuation.yield(.token(0x9F))
        inContinuation.yield(.token(0x98))

        // Yield 4th byte: completes the emoji
        inContinuation.yield(.token(0x80))

        let completedEvent = await iterator.next()
        guard case .chunk(let text) = completedEvent else {
            Issue.record("expected completed emoji chunk, got \(String(describing: completedEvent))")
            return
        }
        #expect(text == "😀")

        inContinuation.yield(.info(GenerateCompletionInfo(
            promptTokenCount: 7,
            generationTokenCount: 4,
            promptTime: 0,
            generationTime: 0.1,
            stopReason: .stop
        )))
        inContinuation.finish()
        await task.value
    }

    @Test func windowPathTokenizerStreamsThroughLiveLoop() async {
        let tokenizer = TestWindowPathTokenizer()
        let text = "First line here.\nSecond line with some words."
        let script = tokenizer.encode(text: text, addSpecialTokens: false)
        let events = await collectEvents(script: script, tokenizer: tokenizer)
        #expect(events.joinedChunks == text)
    }

    @Test func verificationFallbackTokenizerStreamsThroughLiveLoop() async {
        let tokenizer = TestCollapsingTokenizer()
        let text = "a  b\ncd ef\ngh"
        let script = tokenizer.encode(text: text, addSpecialTokens: false)
        let events = await collectEvents(script: script, tokenizer: tokenizer)
        // CollapsingTokenizer collapses doubled spaces to single space
        #expect(events.joinedChunks == "a b\ncd ef\ngh")
    }
}

// MARK: - Test tokenizers for Window Path & Fallbacks

private struct TestWindowPathTokenizer: Tokenizer {
    enum Rule {
        case farLookbehind
        case countCase
    }
    var rule: Rule?
    private let bytes = FakeChatMLTokenizer()

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        bytes.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        let text = bytes.decode(tokenIds: tokenIds, skipSpecialTokens: skipSpecialTokens)
        switch rule {
        case nil:
            return text
        case .farLookbehind:
            return text.replacingOccurrences(of: "Q", with: text.contains("P") ? "p" : "q")
        case .countCase:
            guard tokenIds.count % 3 == 1, let last = text.last else { return text }
            return String(text.dropLast()) + String(last).uppercased()
        }
    }

    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        []
    }
}

private struct TestCollapsingTokenizer: Tokenizer {
    private let bytes = FakeChatMLTokenizer()

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        bytes.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        bytes.decode(tokenIds: tokenIds, skipSpecialTokens: skipSpecialTokens)
            .replacingOccurrences(of: "  ", with: " ")
    }

    func convertTokenToId(_ token: String) -> Int? { bytes.convertTokenToId(token) }
    func convertIdToToken(_ id: Int) -> String? { bytes.convertIdToToken(id) }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        []
    }
}
