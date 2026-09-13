import Foundation
import MLXLMCommon
import Testing
import os

@testable import Tesseract_Agent

struct LiveTokenGenerationLoopTests {
    @Test(.timeLimit(.minutes(1)))
    func splitUnicodeEmitsOnCompletionAndAnIncompleteFinalScalarStaysWithheld() async throws {
        let tokenizer = try await ByteLevelTokenizerFixture.load()
        let (tokens, input) = AsyncStream<TokenGeneration>.makeStream()
        let (stream, task) = TokenGenerationLoop.events(
            from: tokens, generationTask: nil, promptTokenCount: 7,
            modelConfiguration: ModelConfiguration(id: "test/live"), tokenizer: tokenizer)
        defer { input.finish(); task.cancel() }
        var output = stream.makeAsyncIterator()
        for token in [0xE2, 0x82, 0xAC] { input.yield(.token(token)) }
        if case .chunk(let text) = await output.next() {
            #expect(Array(text.utf8) == [0xE2, 0x82, 0xAC])
        } else {
            Issue.record("Expected the complete scalar before generating more tokens or EOS")
        }
        for token in [0xF0, 0x9F] { input.yield(.token(token)) }
        input.yield(
            .info(
                GenerateCompletionInfo(
                    promptTokenCount: 7, generationTokenCount: 5, promptTime: 0,
                    generationTime: 0, stopReason: .length)))
        input.finish()
        if case .info(let info) = await output.next() {
            #expect(info.generationTokenCount == 5)
            #expect(info.stopReason == .length)
        } else {
            Issue.record("An incomplete final scalar must not emit replacement text before info")
        }
        #expect(await output.next() == nil)
        await task.value
    }

    @Test(.timeLimit(.minutes(1)))
    func abandoningTheConsumerCancelsUpstreamAndWaitsForItsCleanup() async throws {
        let tokenizer = try await ByteLevelTokenizerFixture.load()
        let (tokens, input) = AsyncStream<TokenGeneration>.makeStream()
        let (parked, parking) = AsyncStream<Void>.makeStream()
        let firstChunk = StreamTestGate()
        let cleanupStarted = StreamTestGate()
        let allowCleanup = StreamTestGate()
        let state = OSAllocatedUnfairLock(
            initialState: (cancelled: false, cleanedUp: false, completed: false))
        let upstream = Task {
            for await _ in parked {}
            state.withLock { $0.cancelled = Task.isCancelled }
            await cleanupStarted.release()
            await allowCleanup.wait()
            input.finish()
            state.withLock { $0.cleanedUp = true }
        }
        let (stream, mapper) = TokenGenerationLoop.events(
            from: tokens, generationTask: upstream, promptTokenCount: 7,
            modelConfiguration: ModelConfiguration(id: "test/live"), tokenizer: tokenizer)
        let completion = Task {
            await mapper.value
            #expect(state.withLock { $0.cleanedUp }, "Mapper finished before upstream cleanup")
            state.withLock { $0.completed = true }
        }
        let consumer = Task {
            for await event in stream {
                if case .chunk = event { await firstChunk.release() }
            }
        }
        input.yield(.token(97))
        await firstChunk.wait()
        // Cancel only the consuming task. The stream's termination callback
        // must cancel the mapper and its upstream task without a direct call.
        consumer.cancel()
        await cleanupStarted.wait()
        #expect(state.withLock { $0.cancelled })
        #expect(!state.withLock { $0.completed })
        await allowCleanup.release()
        await completion.value
        await consumer.value
        parking.finish()
    }

    @Test(.timeLimit(.minutes(1)))
    func naturalStreamAndMapperCompletionWaitForUpstreamCleanup() async throws {
        let tokenizer = try await ByteLevelTokenizerFixture.load()
        let (tokens, input) = AsyncStream<TokenGeneration>.makeStream()
        let allowCleanup = StreamTestGate()
        let receivedInfo = StreamTestGate()
        let state = OSAllocatedUnfairLock(
            initialState: (cleanedUp: false, mapperDone: false, streamDone: false))
        let upstream = Task {
            await allowCleanup.wait()
            state.withLock { $0.cleanedUp = true }
        }
        let (stream, mapper) = TokenGenerationLoop.events(
            from: tokens, generationTask: upstream, promptTokenCount: 7,
            modelConfiguration: ModelConfiguration(id: "test/live"), tokenizer: tokenizer)
        let completion = Task {
            await mapper.value
            #expect(state.withLock { $0.cleanedUp }, "Mapper finished before upstream cleanup")
            state.withLock { $0.mapperDone = true }
        }
        let consumer = Task {
            for await event in stream {
                if case .info = event { await receivedInfo.release() }
            }
            #expect(
                state.withLock { $0.cleanedUp }, "Output stream finished before upstream cleanup")
            state.withLock { $0.streamDone = true }
        }
        input.yield(
            .info(
                GenerateCompletionInfo(
                    promptTokenCount: 7, generationTokenCount: 0, promptTime: 0,
                    generationTime: 0, stopReason: .stop)))
        input.finish()
        await receivedInfo.wait()
        #expect(!state.withLock { $0.mapperDone || $0.streamDone })
        await allowCleanup.release()
        await completion.value
        await consumer.value
    }

    @Test(.timeLimit(.minutes(1)))
    func textAndToolArgumentsArriveBeforeTheProducerAdvances() async throws {
        let tokenizer = DecodeWorkTokenizer(try await ByteLevelTokenizerFixture.load())
        let (tokens, input) = AsyncStream<TokenGeneration>.makeStream()
        let (stream, task) = TokenGenerationLoop.events(
            from: tokens, generationTask: nil, promptTokenCount: 7,
            modelConfiguration: ModelConfiguration(id: "test/live", toolCallFormat: .json),
            tokenizer: tokenizer)
        defer {
            input.finish()
            task.cancel()
        }
        var output = stream.makeAsyncIterator()

        for byte in "hello".utf8 {
            input.yield(.token(Int(byte)))
            guard case .chunk(let text) = await output.next() else {
                Issue.record("Expected immediate text before generating another token")
                return
            }
            #expect(Array(text.utf8) == [byte])
        }
        for byte in "<tool_call>".utf8 { input.yield(.token(Int(byte))) }
        guard case .toolCallBufferDelta(let opening) = await output.next() else {
            Issue.record("Expected the opening tool-call delta before its body is generated")
            return
        }
        #expect(opening == "<tool_call>")
        for byte in #"{"name":"write","arguments":{"text":"streaming"#.utf8 {
            input.yield(.token(Int(byte)))
            guard case .toolCallBufferDelta(let delta) = await output.next() else {
                Issue.record("Expected an argument delta before generating another token")
                return
            }
            #expect(Array(delta.utf8) == [byte])
        }

        // Cancellation preserves already-delivered arguments and synthesizes
        // the terminal info without waiting for a newline, close tag, or EOS.
        task.cancel()
        var completions: [GenerateCompletionInfo] = []
        while let event = await output.next() {
            if case .info(let info) = event { completions.append(info) }
            if case .chunk = event { Issue.record("Duplicated malformed tool-call residual") }
        }
        await task.value
        #expect(completions.count == 1)
        #expect(completions.first?.stopReason == .cancelled)
        #expect(tokenizer.decodedTokens == 0, "Live mapping must use the recognized byte path")
    }
}

/// An explicit, cancellation-independent upstream cleanup barrier. Tests
/// release it themselves; no sleeps or executor polling decide the ordering.
private actor StreamTestGate {
    private var open = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func wait() async {
        if open { return }
        await withCheckedContinuation { waiters.append($0) }
    }

    func release() {
        open = true
        let pending = waiters
        waiters = []
        for waiter in pending { waiter.resume() }
    }
}
