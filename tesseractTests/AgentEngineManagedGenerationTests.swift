import Foundation
import MLXLMCommon
import Testing
@testable import Tesseract_Agent

@MainActor
struct AgentEngineManagedGenerationTests {

    @Test
    func cancelGenerationAndWaitWaitsForUnderlyingRawGeneration() async {
        let engine = AgentEngine()
        let probe = RawGenerationProbe()
        let finished = AsyncFlag()

        _ = engine.wrapManagedGeneration {
            await probe.makeStart()
        }

        #expect(await waitUntil { await probe.wasStarted() })

        let waitTask = Task {
            await engine.cancelGenerationAndWait()
            await finished.set()
        }

        #expect(await waitUntil { await probe.didEnterWait() })

        try? await Task.sleep(for: .milliseconds(50))
        #expect(await finished.get() == false)

        await probe.allowWaitToFinish()
        await waitTask.value

        #expect(await finished.get())
        #expect(await probe.cancelCount() == 1)
        #expect(engine.isGenerating == false)
    }
}

private actor RawGenerationProbe {
    private var started = false
    private var waitEntered = false
    private var waitAllowed = false
    private var cancelCalls = 0
    private var waitContinuation: CheckedContinuation<Void, Never>?
    private var streamContinuation: AsyncStream<Generation>.Continuation?

    func makeStart() -> HTTPServerRawGenerationStart {
        started = true
        let (stream, continuation) = AsyncStream<Generation>.makeStream()
        streamContinuation = continuation
        return HTTPServerRawGenerationStart(
            stream: stream,
            cancel: { Task { await self.cancel() } },
            waitForCompletion: { await self.waitForCompletion() }
        )
    }

    func cancel() {
        cancelCalls += 1
        streamContinuation?.finish()
    }

    func waitForCompletion() async {
        waitEntered = true
        guard !waitAllowed else { return }
        await withCheckedContinuation { continuation in
            waitContinuation = continuation
        }
    }

    func allowWaitToFinish() {
        waitAllowed = true
        waitContinuation?.resume()
        waitContinuation = nil
    }

    func wasStarted() -> Bool { started }
    func didEnterWait() -> Bool { waitEntered }
    func cancelCount() -> Int { cancelCalls }
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
