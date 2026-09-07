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
    var asThinking: String? {
        if case .thinking(let s) = self { return s } else { return nil }
    }
    var isThinkEnd: Bool {
        if case .thinkEnd = self { return true } else { return false }
    }
}

// MARK: - Tests

nonisolated struct GenerationStreamLoopTests {

    @Test
    func reasoningRunsToItsNaturalEndWithoutIntervention() async throws {
        // Both old length thresholds, plus a repeated reasoning passage. Neither
        // is permission to replace the model's reasoning or discard its answer.
        let longReasoning = (0..<9_000).map { "step\($0)\n" }.joined()
        let repeated = String(repeating: "Let me verify this constraint once more.\n", count: 250)
        let recorder = SinkRecorder()
        let cancelled = OSAllocatedUnfairLock(initialState: false)
        let loop = GenerationStreamLoop(
            initial: cannedHandle(
                [
                    .chunk(longReasoning), .chunk(repeated),
                    .chunk("</think>Final answer."), info(generated: 25_000),
                ], cancel: { cancelled.withLock { $0 = true } }),
            startsInsideThinkBlock: true)
        let outcome = try await loop.run(sink: recorder.sink)
        #expect(recorder.events.compactMap(\.asThinking).joined() == longReasoning + repeated)
        #expect(recorder.events.compactMap(\.asText).joined() == "Final answer.")
        #expect(!cancelled.withLock { $0 })
        #expect(outcome.completionInfo?.generationTokenCount == 25_000)
        #expect(!outcome.cancelled)
    }

    @Test
    func plainTextStreamForwardsTextCapturesInfoAndEndsNaturally() async throws {
        let recorder = SinkRecorder()
        let loop = GenerationStreamLoop(
            initial: cannedHandle([.chunk("hello world"), info(generated: 7)]),
            startsInsideThinkBlock: false
        )

        let outcome = try await loop.run(sink: recorder.sink)

        // `.text` is forwarded to the sink; `.info` is captured, never sunk.
        #expect(recorder.events.compactMap(\.asText) == ["hello world"])
        #expect(
            !recorder.events.contains { if case .info = $0 { return true } else { return false } })

        #expect(outcome.completionInfo?.generationTokenCount == 7)
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
            startsInsideThinkBlock: false
        )

        let outcome = try await loop.run(sink: recorder.sink)

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
            startsInsideThinkBlock: false
        )

        _ = try await loop.run(sink: recorder.sink)

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
            startsInsideThinkBlock: false
        )

        _ = try await loop.run(sink: recorder.sink)

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
            startsInsideThinkBlock: false
        )

        _ = try await loop.run(sink: recorder.sink)

        #expect(!recorder.events.contains { $0.isMalformedToolCall })
        #expect(recorder.events.contains { $0.isToolCall })
    }

    @Test
    func externalCancelDuringThinkingStopsTheOriginalHandleOnce() async throws {
        let probe = StreamProbe()
        let recorder = LockedRecorder()
        let loop = GenerationStreamLoop(
            initial: await probe.makeHandle(initial: [.chunk("Still reasoning.\n")]),
            startsInsideThinkBlock: true)
        let task = Task { try await loop.run(sink: recorder.sink) }
        let received = await waitUntil {
            recorder.snapshot().contains { $0.asThinking != nil }
        }
        #expect(received)
        loop.cancelCurrent()
        loop.cancelCurrent()
        let outcome = try await task.value
        #expect(outcome.cancelled)
        #expect(await probe.cancelCount() == 1)
        #expect(await probe.waitCount() == 1)
        #expect(recorder.snapshot().compactMap(\.asText).isEmpty)
    }

    @Test
    func cancelBeforeRunCancelsInitialHandleWaitsOnceAndReportsCancelled() async throws {
        let recorder = LockedRecorder()
        let probe = StreamProbe()
        let initial = await probe.makeHandle(initial: [.chunk("never consumed")])
        let loop = GenerationStreamLoop(
            initial: initial,
            startsInsideThinkBlock: false
        )

        // Cancel BEFORE `run` is awaited — the "available pre-`run`" contract.
        loop.cancelCurrent()

        let outcome = try await loop.run(sink: recorder.sink)

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
            startsInsideThinkBlock: false
        )

        let outcome = try await loop.run(sink: recorder.sink)

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
            startsInsideThinkBlock: false
        )

        let outcome = try await loop.run(sink: recorder.sink)

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
