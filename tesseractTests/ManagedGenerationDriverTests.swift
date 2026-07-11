//
//  ManagedGenerationDriverTests.swift
//  tesseractTests
//
//  The **Managed Generation Driver** at its own seam: scripted raw handles
//  drive the shared envelope — the outcome tail (terminal `.info` re-yield
//  through the sink, no tail on cancel) and the start-handle contract
//  (`cancel` fires the live-handle bridge and the task; stream termination
//  triggers the same cancel). The spine internals stay covered by
//  `GenerationStreamLoopTests`; these tests pin what the two callers
//  previously hand-copied.
//

import Foundation
import MLXLMCommon
import Testing
import os

@testable import Tesseract_Agent

// MARK: - Fixtures

private func cannedHandle(
    _ events: [RawGeneration],
    cancel: @escaping @Sendable () -> Void = {}
) -> GenerationStreamLoop.RawGenerationHandle {
    let (stream, continuation) = AsyncStream<RawGeneration>.makeStream()
    for event in events { continuation.yield(event) }
    continuation.finish()
    return GenerationStreamLoop.RawGenerationHandle(
        stream: stream, cancel: cancel, waitForCompletion: {})
}

private func infoEvent(generated: Int = 3) -> RawGeneration {
    .info(
        GenerateCompletionInfo(
            promptTokenCount: 10,
            generationTokenCount: generated,
            promptTime: 0.1,
            generationTime: 0.2,
            stopReason: .stop
        ))
}

@MainActor
private func makeDriver() -> ManagedGenerationDriver {
    ManagedGenerationDriver(
        parameters: .default,
        startsInsideThinkBlock: false,
        logContext: "test_driver=1"
    )
}

@MainActor
struct ManagedGenerationDriverTests {

    // MARK: - Outcome tail

    /// A natural finish re-yields the terminal `.info` through the sink —
    /// downstream consumers read completion metrics from the stream.
    @Test func naturalFinishReYieldsTheTerminalInfoThroughTheSink() async throws {
        let driver = makeDriver()
        var events: [AgentGeneration] = []

        let outcome = try await driver.run(
            initial: cannedHandle([.chunk("hello"), infoEvent(generated: 7)]),
            cancelBridge: LateBoundCancel(),
            continuationStarter: nil
        ) { events.append($0) }

        #expect(outcome.completionInfo?.generationTokenCount == 7)
        let infos = events.compactMap { event -> AgentGeneration.Info? in
            if case .info(let info) = event { return info }
            return nil
        }
        #expect(infos.count == 1)
        #expect(infos.first?.generationTokenCount == 7)
        #expect(events.contains { if case .text = $0 { return true } else { return false } })
    }

    /// A cancelled run gets no tail: the sink never sees `.info`.
    @Test func cancelledOutcomeGetsNoInfoReYield() async throws {
        let driver = makeDriver()
        let bridge = LateBoundCancel()
        let recorder = OSAllocatedUnfairLock<[AgentGeneration]>(initialState: [])

        // A stream that stays open until cancelled through the bridge.
        let (stream, continuation) = AsyncStream<RawGeneration>.makeStream()
        continuation.yield(.chunk("partial"))
        let handle = GenerationStreamLoop.RawGenerationHandle(
            stream: stream,
            cancel: { continuation.finish() },
            waitForCompletion: {}
        )

        let runTask = Task {
            try await driver.run(
                initial: handle,
                cancelBridge: bridge,
                continuationStarter: nil
            ) { event in recorder.withLock { $0.append(event) } }
        }
        // Wait for the first chunk to prove the loop is consuming, then
        // cancel the live handle through the bridge — the driver must
        // report a cancelled outcome with no `.info` re-yield.
        let deadline = ContinuousClock.now + .seconds(3)
        while recorder.withLock({ $0.isEmpty }), ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(5))
        }
        bridge()
        let outcome = try await runTask.value

        #expect(outcome.cancelled)
        #expect(outcome.completionInfo == nil)
        let sawInfo = recorder.withLock { events in
            events.contains { if case .info = $0 { return true } else { return false } }
        }
        #expect(!sawInfo)
    }

    // MARK: - Start-handle contract

    /// `cancel` fires the live-handle bridge *and* the driving task; the
    /// contract is idempotent.
    @Test func startCancelFiresBridgeAndTask() async {
        let bridgeFired = OSAllocatedUnfairLock(initialState: 0)
        let bridge = LateBoundCancel()
        bridge.fill { bridgeFired.withLock { $0 += 1 } }

        let (stream, continuation) = AsyncThrowingStream<AgentGeneration, Error>.makeStream()
        let parked = Task { while !Task.isCancelled { await Task.yield() } }
        let task = Task { await parked.value }

        let start = ManagedGenerationDriver.makeStart(
            stream: stream,
            continuation: continuation,
            cachedTokenCount: 42,
            cancelBridge: bridge,
            task: task
        )

        #expect(start.cachedTokenCount == 42)
        start.cancel()
        parked.cancel()
        await start.waitForCompletion()
        #expect(bridgeFired.withLock { $0 } >= 1)
        #expect(task.isCancelled)
    }

    /// Stream termination (the consumer finishing or going away) triggers the
    /// same cancel — the wiring the two callers previously duplicated.
    @Test func streamTerminationTriggersTheCancelContract() async {
        let bridgeFired = OSAllocatedUnfairLock(initialState: 0)
        let bridge = LateBoundCancel()
        bridge.fill { bridgeFired.withLock { $0 += 1 } }

        let (stream, continuation) = AsyncThrowingStream<AgentGeneration, Error>.makeStream()
        let task = Task {}

        _ = ManagedGenerationDriver.makeStart(
            stream: stream,
            continuation: continuation,
            cachedTokenCount: 0,
            cancelBridge: bridge,
            task: task
        )
        continuation.finish()
        do {
            for try await _ in stream {}
        } catch {}

        let deadline = ContinuousClock.now + .seconds(3)
        while bridgeFired.withLock({ $0 }) == 0, ContinuousClock.now < deadline {
            try? await Task.sleep(for: .milliseconds(5))
        }
        #expect(bridgeFired.withLock { $0 } >= 1)
    }
}
