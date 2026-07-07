import Foundation
import MLXLMCommon
import Testing
import os

@testable import Tesseract_Agent

// MARK: - Test helpers

/// Records the `AgentGeneration` events the loop pushes to its sink. Non-Sendable
/// on purpose: the loop drives the sink inline on the same (nonisolated) task the
/// test awaits `run` on, so there is no isolation boundary to cross.
private final class SinkRecorder {
    private(set) var events: [AgentGeneration] = []
    var sink: GenerationStreamLoop.Sink { { event in self.events.append(event) } }
}

/// Thread-safe event recorder for tests that read the sink from a sibling task
/// while the loop runs on another (the cancel tests). `Sendable` via a lock.
private final class LockedRecorder: Sendable {
    private let store = OSAllocatedUnfairLock<[AgentGeneration]>(initialState: [])
    var sink: GenerationStreamLoop.Sink { { event in self.store.withLock { $0.append(event) } } }
    func snapshot() -> [AgentGeneration] { store.withLock { $0 } }
}

/// A controllable raw handle whose stream stays open until cancelled, recording
/// how many times `cancel` / `waitForCompletion` were invoked.
private actor StreamProbe {
    private var cancelCalls = 0
    private var waitCalls = 0
    private var continuation: AsyncStream<RawGeneration>.Continuation?

    /// Build a handle; `initial` events are yielded up front and the stream is
    /// left open (it only finishes on `cancel`).
    func makeHandle(initial: [RawGeneration] = []) -> GenerationStreamLoop.RawGenerationHandle {
        let (stream, continuation) = AsyncStream<RawGeneration>.makeStream()
        self.continuation = continuation
        for event in initial { continuation.yield(event) }
        return GenerationStreamLoop.RawGenerationHandle(
            stream: stream,
            cancel: { Task { await self.cancel() } },
            waitForCompletion: { await self.wait() }
        )
    }

    private func cancel() {
        cancelCalls += 1
        continuation?.finish()
    }

    private func wait() { waitCalls += 1 }

    func cancelCount() -> Int { cancelCalls }
    func waitCount() -> Int { waitCalls }
}

private func waitUntil(
    timeout: Duration = .seconds(1),
    _ condition: @escaping @Sendable () async -> Bool
) async -> Bool {
    let clock = ContinuousClock()
    let deadline = clock.now.advanced(by: timeout)
    while clock.now < deadline {
        if await condition() { return true }
        try? await Task.sleep(for: .milliseconds(5))
    }
    return await condition()
}

/// Build a finished `RawGenerationHandle` from a fixed event list. `cancel` /
/// `waitForCompletion` default to no-ops; pass a probe's closures to observe them.
private func cannedHandle(
    _ events: [RawGeneration],
    cancel: @escaping @Sendable () -> Void = {},
    waitForCompletion: @escaping @Sendable () async -> Void = {}
) -> GenerationStreamLoop.RawGenerationHandle {
    let (stream, continuation) = AsyncStream<RawGeneration>.makeStream()
    for event in events { continuation.yield(event) }
    continuation.finish()
    return GenerationStreamLoop.RawGenerationHandle(
        stream: stream,
        cancel: cancel,
        waitForCompletion: waitForCompletion
    )
}

private func toolCallGen(name: String, arguments: [String: any Sendable] = [:]) -> RawGeneration {
    .toolCall(ToolCall(function: .init(name: name, arguments: arguments)))
}

private func info(
    prompt: Int = 10,
    generated: Int = 3,
    promptTime: TimeInterval = 0.1,
    generateTime: TimeInterval = 0.2,
    stopReason: GenerateStopReason = .stop
) -> RawGeneration {
    .info(
        GenerateCompletionInfo(
            promptTokenCount: prompt,
            generationTokenCount: generated,
            promptTime: promptTime,
            generationTime: generateTime,
            stopReason: stopReason
        ))
}

private extension AgentGeneration {
    var asText: String? { if case .text(let t) = self { return t } else { return nil } }
    var asToolCallDelta: String? {
        if case .toolCallDelta(_, let d) = self { return d } else { return nil }
    }
    var isMalformedToolCall: Bool {
        if case .malformedToolCall = self { return true } else { return false }
    }
    var isToolCall: Bool {
        if case .toolCall = self { return true } else { return false }
    }
    var isInfo: Bool {
        if case .info = self { return true } else { return false }
    }
    var asThinkTruncate: String? {
        if case .thinkTruncate(let s) = self { return s } else { return nil }
    }
    var asThinking: String? {
        if case .thinking(let s) = self { return s } else { return nil }
    }
    var isThinkEnd: Bool {
        if case .thinkEnd = self { return true } else { return false }
    }
}

/// Config that trips the line-repeat signal after three identical lines, with no
/// grace period — used to drive an intervention hermetically.
private let trippingSafeguard = ThinkingRepetitionDetector.Config(
    minLineLength: 10, maxLineRepeats: 3, minCharsBeforeIntervention: 0
)
private let interventionPrelude = "Reasoning about the user's constraints first.\n"
private let interventionRepeated = "Now I loop on the same thought forever.\n"

/// Stream that trips `trippingSafeguard`: one legit line then the loop line ×3.
private let trippingThinkingEvents: [RawGeneration] = [
    .chunk(interventionPrelude),
    .chunk(interventionRepeated),
    .chunk(interventionRepeated),
    .chunk(interventionRepeated),
]

// MARK: - Tests

nonisolated struct GenerationStreamLoopTests {

    @Test
    func plainTextStreamForwardsTextCapturesInfoAndEndsNaturally() async throws {
        let recorder = SinkRecorder()
        let loop = GenerationStreamLoop(
            initial: cannedHandle([.chunk("hello world"), info(generated: 7)]),
            startsInsideThinkBlock: false,
            safeguard: .init(enabled: false)
        )

        let outcome = try await loop.run(continuation: nil, sink: recorder.sink)

        // `.text` is forwarded to the sink; `.info` is captured, never sunk.
        #expect(recorder.events.compactMap(\.asText) == ["hello world"])
        #expect(
            !recorder.events.contains { if case .info = $0 { return true } else { return false } })

        #expect(outcome.completionInfo?.generationTokenCount == 7)
        #expect(outcome.intervened == false)
        #expect(outcome.cancelled == false)
    }

    @Test
    func libraryParsedToolCallSuppressesAppLevelToolEventsButForwardsDeltas() async throws {
        let recorder = SinkRecorder()
        let loop = GenerationStreamLoop(
            initial: cannedHandle([
                // Vendor parses a tool call — flips libraryParsedToolCalls.
                toolCallGen(name: "vendorcall"),
                // Vendor buffer delta is forwarded as a progressive UI event.
                .toolCallBufferDelta("abc"),
                // App parser would surface this complete `<tool_call>` as a
                // `.toolCall(appcall)` plus a body delta; the vendor already owns
                // tool calls, so both must be suppressed.
                .chunk("<tool_call>\n{\"name\":\"appcall\",\"arguments\":{}}</tool_call>"),
                info(),
            ]),
            startsInsideThinkBlock: false,
            safeguard: .init(enabled: false)
        )

        let outcome = try await loop.run(continuation: nil, sink: recorder.sink)

        // The vendor delta is forwarded; the app parser's body delta is not.
        #expect(recorder.events.compactMap(\.asToolCallDelta) == ["abc"])
        // The vendor tool call passes through; the app-parsed one is suppressed.
        let toolCallNames = recorder.events.compactMap { e -> String? in
            if case .toolCall(let call) = e { return call.function.name } else { return nil }
        }
        #expect(toolCallNames == ["vendorcall"])
        #expect(!recorder.events.contains { $0.isMalformedToolCall })
        #expect(outcome.diagnostics.libraryParsedToolCalls == true)
    }

    @Test
    func vendorBufferDeltasLockTheToolNameAtTheProducer() async throws {
        let recorder = SinkRecorder()
        let loop = GenerationStreamLoop(
            initial: cannedHandle([
                // Name literal split across deltas: nil until it closes.
                .toolCallBufferDelta(#"{"name": "rea"#),
                .toolCallBufferDelta(#"d", "arguments": {}}"#),
                // Parsed call consumes the buffer — the lock resets.
                toolCallGen(name: "read"),
                .toolCallBufferDelta(#"{"name": "wri"#),
                info(),
            ]),
            startsInsideThinkBlock: false,
            safeguard: .init(enabled: false)
        )

        _ = try await loop.run(continuation: nil, sink: recorder.sink)

        var deltaNames: [String?] = []
        for event in recorder.events {
            if case .toolCallDelta(let name, _) = event { deltaNames.append(name) }
        }
        #expect(deltaNames == [nil, "read", nil])
    }

    @Test
    func bufferedToolCallWithNoCloseAtEOSSurfacesOneWrappedMalformedToolCall() async throws {
        let recorder = SinkRecorder()
        let loop = GenerationStreamLoop(
            initial: cannedHandle([
                // Vendor buffered a `<tool_call>` body but the model hit EOS
                // before the close tag — no `.toolCall`, no `.info`.
                .toolCallBufferDelta("<tool_call>\n<read>\n<file_path>/x</file_path>")
            ]),
            startsInsideThinkBlock: false,
            safeguard: .init(enabled: false)
        )

        _ = try await loop.run(continuation: nil, sink: recorder.sink)

        let malformed = recorder.events.compactMap { e -> String? in
            if case .malformedToolCall(let raw) = e { return raw } else { return nil }
        }
        #expect(malformed.count == 1)
        #expect(malformed.first?.hasPrefix("<tool_call>") == true)
        #expect(malformed.first?.hasSuffix("</tool_call>") == true)
        #expect(malformed.first?.contains("/x") == true)
    }

    @Test
    func successfulToolCallResetsBufferSoNoMalformedSurfacesAtEOS() async throws {
        let recorder = SinkRecorder()
        let loop = GenerationStreamLoop(
            initial: cannedHandle([
                // A buffered call that DOES close successfully resets the accum...
                .toolCallBufferDelta("<tool_call>\n<read>\n<file_path>/x</file_path>"),
                toolCallGen(name: "read", arguments: ["file_path": "/x"]),
                // ...so EOS sees an empty dropped buffer — nothing malformed.
                info(),
            ]),
            startsInsideThinkBlock: false,
            safeguard: .init(enabled: false)
        )

        _ = try await loop.run(continuation: nil, sink: recorder.sink)

        #expect(!recorder.events.contains { $0.isMalformedToolCall })
        #expect(recorder.events.contains { $0.isToolCall })
    }

    @Test
    func thinkingLoopTripsInterventionTripleInOrderWithoutContinuation() async throws {
        let recorder = SinkRecorder()
        let loop = GenerationStreamLoop(
            initial: cannedHandle(trippingThinkingEvents),
            startsInsideThinkBlock: true,
            safeguard: trippingSafeguard
        )

        let outcome = try await loop.run(continuation: nil, sink: recorder.sink)

        // The intervention replaces the degenerate thinking with the triple:
        // .thinkTruncate(safe) → .thinking(injection) → .thinkEnd, in order.
        guard let i = recorder.events.firstIndex(where: { $0.asThinkTruncate != nil }) else {
            Issue.record("expected a .thinkTruncate")
            return
        }
        #expect(recorder.events[i].asThinkTruncate == interventionPrelude)
        #expect(recorder.events[i + 1].asThinking == trippingSafeguard.injectionMessage)
        #expect(recorder.events[i + 2].isThinkEnd)
        #expect(outcome.intervened == true)
        #expect(outcome.cancelled == false)
    }

    @Test
    func noContinuationInterventionDoesNotFlushBufferedThinkingAfterThinkEnd() async throws {
        let recorder = SinkRecorder()
        let trailingReasoning = "and some trailing reasoning the safeguard truncated"
        let loop = GenerationStreamLoop(
            initial: cannedHandle([
                .chunk(interventionPrelude),
                .chunk(interventionRepeated),
                // The third identical line trips the safeguard; the un-terminated
                // tail after it stays buffered in the (un-reinit) parser at the
                // moment of intervention.
                .chunk(interventionRepeated + trailingReasoning),
            ]),
            startsInsideThinkBlock: true,
            safeguard: trippingSafeguard
        )

        _ = try await loop.run(continuation: nil, sink: recorder.sink)

        // The truncation triple's `.thinkEnd` closes the think block. A finalize()
        // flush would forward the still-buffered `trailingReasoning` as
        // `.thinkReclassify` + `.text` AFTER that `.thinkEnd`, re-polluting the
        // reasoning the safeguard just truncated. Nothing text-bearing may follow.
        guard let endIdx = recorder.events.firstIndex(where: { $0.isThinkEnd }) else {
            Issue.record("expected a .thinkEnd from the truncation triple")
            return
        }
        let afterThinkEnd = recorder.events[(endIdx + 1)...]
        #expect(!afterThinkEnd.contains { $0.asText != nil || $0.asThinking != nil })
        #expect(!recorder.events.contains { $0.asText == trailingReasoning })
    }

    @Test
    func interventionSwapsToContinuationWithSafePrefixAndClassifiesPostThinkAsText() async throws {
        let recorder = SinkRecorder()
        let recordedPrefix = OSAllocatedUnfairLock<String?>(initialState: nil)
        let starter: GenerationStreamLoop.ContinuationStarter = { safePrefix in
            recordedPrefix.withLock { $0 = safePrefix }
            return cannedHandle([.chunk("the answer"), info(generated: 5)])
        }
        let loop = GenerationStreamLoop(
            initial: cannedHandle(trippingThinkingEvents),
            startsInsideThinkBlock: true,
            safeguard: trippingSafeguard
        )

        let outcome = try await loop.run(continuation: starter, sink: recorder.sink)

        // The starter is invoked with the safe prefix captured before the loop.
        #expect(recordedPrefix.withLock { $0 } == interventionPrelude)
        // The continuation picks up after `</think>`, so its output is text.
        #expect(recorder.events.compactMap(\.asText).contains("the answer"))
        #expect(outcome.intervened == true)
        #expect(outcome.cancelled == false)
        // The terminal `.info` comes from the continuation stream.
        #expect(outcome.completionInfo?.generationTokenCount == 5)
    }

    @Test
    func continuationFailureEndsGracefullyWithTruncatedResponse() async throws {
        struct StarterError: Error {}
        let recorder = SinkRecorder()
        let starter: GenerationStreamLoop.ContinuationStarter = { _ in throw StarterError() }
        let loop = GenerationStreamLoop(
            initial: cannedHandle(trippingThinkingEvents),
            startsInsideThinkBlock: true,
            safeguard: trippingSafeguard
        )

        // A throwing starter must not propagate out of `run`.
        let outcome = try await loop.run(continuation: starter, sink: recorder.sink)

        #expect(recorder.events.contains { $0.asThinkTruncate != nil })
        #expect(outcome.intervened == true)
        #expect(outcome.cancelled == false)
    }

    @Test
    func cancelAfterSwapCancelsContinuationHandleOnceAndReportsCancelled() async throws {
        let recorder = LockedRecorder()
        let probe = StreamProbe()
        let starter: GenerationStreamLoop.ContinuationStarter = { _ in
            // Open stream with one item so we can observe the post-swap consume.
            await probe.makeHandle(initial: [.chunk("continuing")])
        }
        let loop = GenerationStreamLoop(
            initial: cannedHandle(trippingThinkingEvents),
            startsInsideThinkBlock: true,
            safeguard: trippingSafeguard
        )

        let runTask = Task { [recorder] in
            try await loop.run(continuation: starter, sink: recorder.sink)
        }

        // Once the continuation output appears, the new handle is installed and
        // being consumed — we are unambiguously after the swap.
        #expect(await waitUntil { recorder.snapshot().compactMap(\.asText).contains("continuing") })

        loop.cancelCurrent()

        let outcome = try await runTask.value
        #expect(outcome.cancelled == true)
        // The continuation handle — not the original — is the one cancelled/waited.
        #expect(await waitUntil { await probe.cancelCount() == 1 })
        #expect(await probe.waitCount() == 1)
    }

    @Test
    func cancelBeforeRunCancelsInitialHandleWaitsOnceAndReportsCancelled() async throws {
        let recorder = LockedRecorder()
        let probe = StreamProbe()
        let initial = await probe.makeHandle(initial: [.chunk("never consumed")])
        let loop = GenerationStreamLoop(
            initial: initial,
            startsInsideThinkBlock: false,
            safeguard: .init(enabled: false)
        )

        // Cancel BEFORE `run` is awaited — the "available pre-`run`" contract.
        loop.cancelCurrent()

        let outcome = try await loop.run(continuation: nil, sink: recorder.sink)

        #expect(outcome.cancelled == true)
        #expect(await waitUntil { await probe.cancelCount() == 1 })
        #expect(await probe.waitCount() == 1)
        // The stream was never consumed, so nothing reached the sink.
        #expect(recorder.snapshot().isEmpty)
    }

    @Test
    func silentCloseWithoutInfoPopulatesDiagnosticsPostFinalize() async throws {
        let recorder = SinkRecorder()
        let loop = GenerationStreamLoop(
            // Trailing `<` is held back by the parser as a possible partial tag;
            // it is only flushed by finalize(). No `.info` ⇒ silent close.
            initial: cannedHandle([.chunk("answer<")]),
            startsInsideThinkBlock: false,
            safeguard: .init(enabled: false)
        )

        let outcome = try await loop.run(continuation: nil, sink: recorder.sink)

        #expect(outcome.completionInfo == nil)
        #expect(outcome.diagnostics.rawChunksJoined == "answer<")
        #expect(outcome.diagnostics.libraryParsedToolCalls == false)
        // Snapshot is taken AFTER finalize() flushed the buffer — bufferLen == 0.
        // (A pre-finalize snapshot would report the held-back "<".)
        #expect(outcome.diagnostics.finalizeState.bufferLen == 0)
    }

    @Test
    func diagnosticsPopulatedEvenWhenStreamEndsWithInfo() async throws {
        let recorder = SinkRecorder()
        let loop = GenerationStreamLoop(
            initial: cannedHandle([.chunk("done"), info()]),
            startsInsideThinkBlock: false,
            safeguard: .init(enabled: false)
        )

        let outcome = try await loop.run(continuation: nil, sink: recorder.sink)

        #expect(outcome.completionInfo != nil)
        #expect(outcome.diagnostics.rawChunksJoined == "done")
    }

    // MARK: - RawGenerationHandle normalization

    @Test
    func handleFromServerRawStartMapsCancelWaitAndStreamOneToOne() async {
        let cancelCalls = OSAllocatedUnfairLock<Int>(initialState: 0)
        let waitCalls = OSAllocatedUnfairLock<Int>(initialState: 0)
        let (stream, continuation) = AsyncStream<RawGeneration>.makeStream()
        continuation.yield(.chunk("hi"))
        continuation.finish()
        let start = HTTPServerRawGenerationStart(
            stream: stream,
            cancel: { cancelCalls.withLock { $0 += 1 } },
            waitForCompletion: { waitCalls.withLock { $0 += 1 } }
        )

        let handle = GenerationStreamLoop.RawGenerationHandle(start)

        handle.cancel()
        await handle.waitForCompletion()
        #expect(cancelCalls.withLock { $0 } == 1)
        #expect(waitCalls.withLock { $0 } == 1)

        var texts: [String] = []
        for await item in handle.stream {
            if case .chunk(let text) = item { texts.append(text) }
        }
        #expect(texts == ["hi"])
    }

    @Test
    func handleFromStreamAndCompletionDrivesCompletionTask() async {
        // A completion task that only finishes once cancelled.
        let completion = Task<Void, Never> {
            while !Task.isCancelled { try? await Task.sleep(for: .milliseconds(5)) }
        }
        let (stream, continuation) = AsyncStream<RawGeneration>.makeStream()
        continuation.finish()

        let handle = GenerationStreamLoop.RawGenerationHandle(
            stream: stream,
            completion: completion
        )

        handle.cancel()
        await handle.waitForCompletion()
        #expect(completion.isCancelled)
    }
}
