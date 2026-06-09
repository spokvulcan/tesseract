import Foundation
import Testing
@testable import Tesseract_Agent

/// The unload drain contract of the **Server Completion** module —
/// ADR-0006's in-actor backstop. `LLMActor.unloadModel` must cancel-and-await
/// the active completion before teardown, concurrent drains must all wait
/// for the completion to finish (not skip past a slot another drain is
/// already awaiting), and the natural-finish clear must drop only its own
/// slot.
@MainActor
struct ServerCompletionDrainTests {

    @Test
    func unloadCancelsAndAwaitsTheActiveCompletion() async {
        let actor = LLMActor()
        let probe = ServerCompletionProbe()
        await actor.registerServerCompletionForTesting(
            await probe.makeHandle(), id: UUID()
        )

        let unloadFinished = AsyncFlag()
        let unloadTask = Task {
            await actor.unloadModel()
            await unloadFinished.set()
        }

        #expect(await waitUntil { await probe.waitersEnteredCount() >= 1 })
        #expect(await waitUntil { await probe.cancelCount() >= 1 })

        // The drain must not return while the completion is still running.
        try? await Task.sleep(for: .milliseconds(50))
        #expect(await unloadFinished.get() == false)

        await probe.finishAllWaiters()
        await unloadTask.value
        #expect(await unloadFinished.get())
    }

    @Test
    func concurrentUnloadsBothAwaitTheActiveCompletion() async {
        let actor = LLMActor()
        let probe = ServerCompletionProbe()
        await actor.registerServerCompletionForTesting(
            await probe.makeHandle(), id: UUID()
        )

        let firstFinished = AsyncFlag()
        let secondFinished = AsyncFlag()
        let firstUnload = Task {
            await actor.unloadModel()
            await firstFinished.set()
        }
        let secondUnload = Task {
            await actor.unloadModel()
            await secondFinished.set()
        }

        #expect(await waitUntil { await probe.waitersEnteredCount() >= 2 })

        // Neither drain may return while the completion is still running —
        // the second must not slip past a slot the first is already awaiting.
        try? await Task.sleep(for: .milliseconds(50))
        #expect(await firstFinished.get() == false)
        #expect(await secondFinished.get() == false)

        await probe.finishAllWaiters()
        await firstUnload.value
        await secondUnload.value
        #expect(await firstFinished.get())
        #expect(await secondFinished.get())
    }

    @Test
    func naturalFinishClearsTheSlotSoUnloadHasNothingToDrain() async {
        let actor = LLMActor()
        let probe = ServerCompletionProbe()
        let requestID = UUID()
        await actor.registerServerCompletionForTesting(
            await probe.makeHandle(), id: requestID
        )

        await actor.clearFinishedServerCompletion(requestID)

        let unloadFinished = AsyncFlag()
        Task {
            await actor.unloadModel()
            await unloadFinished.set()
        }
        #expect(await waitUntil { await unloadFinished.get() })
        #expect(await probe.cancelCount() == 0)
        #expect(await probe.waitersEnteredCount() == 0)
    }

    @Test
    func staleFinishDoesNotDropANewerSlot() async {
        let actor = LLMActor()
        let probe = ServerCompletionProbe()
        await actor.registerServerCompletionForTesting(
            await probe.makeHandle(), id: UUID()
        )

        // A finisher for some other (older) request must not clear the slot.
        await actor.clearFinishedServerCompletion(UUID())

        let unloadFinished = AsyncFlag()
        let unloadTask = Task {
            await actor.unloadModel()
            await unloadFinished.set()
        }

        #expect(await waitUntil { await probe.cancelCount() >= 1 })
        await probe.finishAllWaiters()
        await unloadTask.value
        #expect(await unloadFinished.get())
    }
}

/// Controllable fake for the registered completion handle: counts cancels,
/// parks `waitForCompletion` callers until the test releases them.
private actor ServerCompletionProbe {
    private var cancelCalls = 0
    private var waitersEntered = 0
    private var parkedWaiters: [CheckedContinuation<Void, Never>] = []
    private var finished = false

    func makeHandle() -> HTTPServerGenerationStart {
        let (stream, _) = AsyncThrowingStream<AgentGeneration, Error>.makeStream()
        return HTTPServerGenerationStart(
            stream: stream,
            cachedTokenCount: 0,
            cancel: { Task { await self.recordCancel() } },
            waitForCompletion: { await self.waitForCompletion() }
        )
    }

    func recordCancel() {
        cancelCalls += 1
    }

    func waitForCompletion() async {
        waitersEntered += 1
        guard !finished else { return }
        await withCheckedContinuation { continuation in
            parkedWaiters.append(continuation)
        }
    }

    func finishAllWaiters() {
        finished = true
        let waiters = parkedWaiters
        parkedWaiters = []
        for waiter in waiters {
            waiter.resume()
        }
    }

    func cancelCount() -> Int { cancelCalls }
    func waitersEnteredCount() -> Int { waitersEntered }
}

private actor AsyncFlag {
    private var value = false

    func set() {
        value = true
    }

    func get() -> Bool {
        value
    }
}

private func waitUntil(
    timeout: Duration = .seconds(1),
    condition: @escaping @Sendable () async -> Bool
) async -> Bool {
    let clock = ContinuousClock()
    let deadline = clock.now.advanced(by: timeout)

    while clock.now < deadline {
        if await condition() {
            return true
        }
        try? await Task.sleep(for: .milliseconds(10))
    }

    return await condition()
}
