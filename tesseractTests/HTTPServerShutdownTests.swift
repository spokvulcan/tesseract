import Foundation
import Testing
@testable import Tesseract_Agent

@MainActor
struct HTTPServerShutdownTests {

    @Test
    func stopAndDrainWaitsForConnectionTasksToFinish() async {
        let server = HTTPServer()
        let probe = ConnectionDrainProbe()
        let stopFinished = AsyncFlag()

        let task = Task {
            do {
                try await Task.sleep(for: .seconds(10))
            } catch {
                await probe.markCancelled()
                await probe.waitForTransportCancellation()
                await probe.blockUntilReleased()
            }
        }
        let accepted = server.registerConnectionTaskForTesting(
            task,
            cancelTransport: { Task { await probe.markTransportCancelled() } }
        )
        #expect(accepted)

        let stopTask = Task {
            await server.stopAndDrain()
            await stopFinished.set()
        }

        await probe.waitForCancellation()
        #expect(await waitUntil { await probe.transportWasCancelled() })
        try? await Task.sleep(for: .milliseconds(50))
        #expect(await stopFinished.get() == false)

        await probe.release()
        await stopTask.value

        #expect(await stopFinished.get())
    }

    @Test
    func stopAndDrainRejectsLateConnectionTasks() async {
        let server = HTTPServer()
        let activeProbe = ConnectionDrainProbe()
        let stopFinished = AsyncFlag()

        let activeTask = Task {
            do {
                try await Task.sleep(for: .seconds(10))
            } catch {
                await activeProbe.markCancelled()
                await activeProbe.blockUntilReleased()
            }
        }
        let acceptedActive = server.registerConnectionTaskForTesting(activeTask)
        #expect(acceptedActive)

        let stopTask = Task {
            await server.stopAndDrain()
            await stopFinished.set()
        }

        await activeProbe.waitForCancellation()

        let lateCancelled = AsyncFlag()
        let lateTransportCancelled = AsyncFlag()
        let lateTask = Task {
            do {
                try await Task.sleep(for: .seconds(10))
            } catch {
                await lateCancelled.set()
            }
        }

        let accepted = server.registerConnectionTaskForTesting(
            lateTask,
            cancelTransport: { Task { await lateTransportCancelled.set() } }
        )

        #expect(accepted == false)
        #expect(await waitUntil { await lateCancelled.get() })
        #expect(await waitUntil { await lateTransportCancelled.get() })
        #expect(await stopFinished.get() == false)

        await activeProbe.release()
        await stopTask.value
        await lateTask.value

        #expect(await stopFinished.get())
    }
}

private actor ConnectionDrainProbe {
    private var cancelled = false
    private var transportCancelled = false
    private var cancellationContinuation: CheckedContinuation<Void, Never>?
    private var transportCancellationContinuation: CheckedContinuation<Void, Never>?
    private var releaseContinuation: CheckedContinuation<Void, Never>?

    func markCancelled() {
        cancelled = true
        cancellationContinuation?.resume()
        cancellationContinuation = nil
    }

    func markTransportCancelled() {
        transportCancelled = true
        transportCancellationContinuation?.resume()
        transportCancellationContinuation = nil
    }

    func waitForCancellation() async {
        guard !cancelled else { return }
        await withCheckedContinuation { continuation in
            cancellationContinuation = continuation
        }
    }

    func waitForTransportCancellation() async {
        guard !transportCancelled else { return }
        await withCheckedContinuation { continuation in
            transportCancellationContinuation = continuation
        }
    }

    func transportWasCancelled() -> Bool {
        transportCancelled
    }

    func blockUntilReleased() async {
        await withCheckedContinuation { continuation in
            releaseContinuation = continuation
        }
    }

    func release() {
        releaseContinuation?.resume()
        releaseContinuation = nil
    }
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
