import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// The engine-stepped raw token loop (PRD #173): `rawTokenTask` semantics —
/// combined stop-token set, prompt/generation timing split, authoritative
/// `.info` — with every `next()` running inside a granted Batch Engine
/// decode step, so pool lanes interleave at token granularity.
@MainActor
@Suite(.serialized) struct TokenGenerationLoopSteppedTests {

    /// An engine whose lease is a pass-through — these tests pin loop
    /// semantics, not lease choreography (BatchEngineTests owns that).
    private func makeEngine() -> BatchEngine {
        BatchEngine(
            leaseRunner: { _, _, body in await body() },
            leaseWaiters: LeaseWaiterSignal(),
            demandSatisfied: { _ in true },
            laneBudget: {
                BatchLaneBudget(
                    headroomBytes: 64 << 30, evictableCacheBytes: 0,
                    perLaneBytes: 4 << 30)
            }
        )
    }

    nonisolated private func makeConfiguration(stopToken: Int? = nil) -> ModelConfiguration {
        var configuration = ModelConfiguration(id: "test/stepped-loop", toolCallFormat: .json)
        if let stopToken { configuration.eosTokenIds = [stopToken] }
        return configuration
    }

    /// Scripted `next()` closure: yields the script then nil.
    private final class ScriptedIterator: @unchecked Sendable {
        private var script: [Int]
        private(set) var served = 0
        let maxTokens: Int?

        init(_ script: [Int], maxTokens: Int? = nil) {
            self.script = script
            self.maxTokens = maxTokens
        }

        func next() -> Int? {
            guard !script.isEmpty else { return nil }
            served += 1
            return script.removeFirst()
        }
    }

    private func runLoop(
        script: ScriptedIterator,
        stopToken: Int? = nil,
        submitFirst: Bool = true
    ) async throws -> (tokens: [Int], info: GenerateCompletionInfo?) {
        let engine = makeEngine()
        let laneID = UUID()
        _ = try await engine.submit(
            BatchSubmission(
                requestID: laneID,
                demand: BatchModelDemand(modelIDOverride: nil, vision: .fromSettings),
                mode: .pooled
            ))
        let (stream, task) = TokenGenerationLoop.steppedRawTokenTask(
            promptTokenCount: 7,
            modelConfiguration: makeConfiguration(stopToken: stopToken),
            tokenizer: FakeChatMLTokenizer(),
            engine: engine,
            laneID: laneID,
            nextToken: { script.next() },
            iteratorStats: { (script.served, script.maxTokens, 0) }
        )
        var tokens: [Int] = []
        var info: GenerateCompletionInfo?
        for await event in stream {
            switch event {
            case .token(let token): tokens.append(token)
            case .info(let completion): info = completion
            }
        }
        await task.value
        await engine.laneFinished(laneID)
        return (tokens, info)
    }

    @Test func streamsTokensAndStopsAtStopToken() async throws {
        let script = ScriptedIterator([65, 66, 67, 99, 68])
        let (tokens, info) = try await runLoop(script: script, stopToken: 99)

        #expect(tokens == [65, 66, 67])
        let completion = try #require(info)
        #expect(completion.generationTokenCount == 3)
        if case .stop = completion.stopReason {
        } else {
            Issue.record("expected .stop, got \(completion.stopReason)")
        }
        // The stop token was pulled but never yielded; nothing past it ran.
        #expect(script.served == 4)
    }

    @Test func exhaustedIteratorAtMaxTokensReportsLength() async throws {
        let script = ScriptedIterator([65, 66], maxTokens: 2)
        let (tokens, info) = try await runLoop(script: script)

        #expect(tokens == [65, 66])
        let completion = try #require(info)
        if case .length = completion.stopReason {
        } else {
            Issue.record("expected .length, got \(completion.stopReason)")
        }
    }

    @Test func lanesInterleaveTokenByToken() async throws {
        // Two stepped loops on one engine: their served tokens interleave —
        // neither stream waits for the other to finish (the PRD's "streams
        // do not freeze" story at the loop level).
        let engine = makeEngine()
        let order = OSAllocatedUnfairLockBox()

        @Sendable func drive(_ label: String, count: Int) async throws {
            let laneID = UUID()
            _ = try await engine.submit(
                BatchSubmission(
                    requestID: laneID,
                    demand: BatchModelDemand(modelIDOverride: nil, vision: .fromSettings),
                    mode: .pooled
                ))
            let script = ScriptedIterator(Array(1...count))
            let (stream, task) = TokenGenerationLoop.steppedRawTokenTask(
                promptTokenCount: 1,
                modelConfiguration: makeConfiguration(),
                tokenizer: FakeChatMLTokenizer(),
                engine: engine,
                laneID: laneID,
                nextToken: {
                    order.append(label)
                    return script.next()
                },
                iteratorStats: { (script.served, nil, 0) }
            )
            for await _ in stream {}
            await task.value
            await engine.laneFinished(laneID)
        }

        async let first: Void = drive("a", count: 25)
        async let second: Void = drive("b", count: 25)
        _ = try await (first, second)

        let sequence = order.values
        // Both lanes' steps appear, and they interleave: some "b" lands
        // strictly between two "a"s.
        let firstB = try #require(sequence.firstIndex(of: "b"))
        let lastA = try #require(sequence.lastIndex(of: "a"))
        #expect(firstB < lastA)
    }

    @Test func cancellingTheTaskEndsWithCancelledInfo() async throws {
        let engine = makeEngine()
        let laneID = UUID()
        _ = try await engine.submit(
            BatchSubmission(
                requestID: laneID,
                demand: BatchModelDemand(modelIDOverride: nil, vision: .fromSettings),
                mode: .pooled
            ))
        let script = ScriptedIterator(Array(1...100_000))
        let (stream, task) = TokenGenerationLoop.steppedRawTokenTask(
            promptTokenCount: 1,
            modelConfiguration: makeConfiguration(),
            tokenizer: FakeChatMLTokenizer(),
            engine: engine,
            laneID: laneID,
            nextToken: { script.next() },
            iteratorStats: { (script.served, nil, 0) }
        )
        var received = 0
        var info: GenerateCompletionInfo?
        for await event in stream {
            switch event {
            case .token:
                received += 1
                if received == 5 { task.cancel() }
            case .info(let completion):
                info = completion
            }
        }
        await task.value
        await engine.laneFinished(laneID)

        let completion = try #require(info)
        if case .cancelled = completion.stopReason {
        } else {
            Issue.record("expected .cancelled, got \(completion.stopReason)")
        }
        #expect(script.served < 100_000)
    }
}

/// Ordered append-only label list, lock-protected (steps run off-main).
private final class OSAllocatedUnfairLockBox: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [String] = []

    func append(_ label: String) {
        lock.lock()
        storage.append(label)
        lock.unlock()
    }

    var values: [String] {
        lock.lock()
        defer { lock.unlock() }
        return storage
    }
}
