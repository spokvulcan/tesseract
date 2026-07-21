import Foundation
import Testing

@testable import Tesseract_Agent

// MARK: - Scripted transport helpers (no sockets)

/// A one-shot, cancellation-aware latch: `wait()` suspends until `trip()` is
/// called (or the awaiting task is cancelled). Mirrors the production
/// `HTTPConnectionLifecycle.waitForDisconnect` contract so the driver's
/// disconnect/never-fires probes behave exactly as they do over a real socket.
private actor AsyncLatch {
    private var tripped = false
    private var waiters: [UUID: CheckedContinuation<Void, Never>] = [:]

    func trip() {
        guard !tripped else { return }
        tripped = true
        let current = waiters.values
        waiters.removeAll()
        for waiter in current { waiter.resume() }
    }

    func wait() async {
        if tripped { return }
        let id = UUID()
        await withTaskCancellationHandler {
            await withCheckedContinuation { continuation in
                if tripped || Task.isCancelled {
                    continuation.resume()
                } else {
                    waiters[id] = continuation
                }
            }
        } onCancel: {
            Task { await self.resumeCancelled(id) }
        }
    }

    private func resumeCancelled(_ id: UUID) {
        waiters.removeValue(forKey: id)?.resume()
    }
}

/// Counts keepalive writes the scripted transport emitted.
private actor Counter {
    private(set) var value = 0
    func increment() { value += 1 }
}

/// A terminal drive outcome that carries no accumulated content — the shape a
/// clean completion hands back to the handler.
private func completedOutcome() -> CompletionHandler.StreamingOutcome {
    .completed(GenerationAccumulator(), nil, wireStreamedToolCalls: false)
}

/// A never-tripping, cancellation-aware disconnect probe — the connection stays
/// up for the whole run, so this watcher only ever ends via `cancelAll()`.
private func connectionStaysUp() -> AsyncLatch {
    AsyncLatch()
}

// MARK: - Tests

/// Drives `StreamLifecycleDriver.run` — the transport-lifecycle race behind
/// `runStreamingCompletion` — with scripted transport closures, asserting the
/// first-finisher-wins outcome, the cancel bridge, and keepalive cadence
/// without opening a socket.
struct StreamLifecycleDriverTests {

    /// A disconnect mid-drive wins the race, bridges the cancel into
    /// generation, reports `.connectionState`, and the drive observes its own
    /// cancellation (it does not linger).
    @Test func disconnectMidDriveCancelsDrivePromptly() async {
        let cancelBridged = LeaseAcquiredSignal()
        let driveObservedCancel = LeaseAcquiredSignal()
        let driveRunning = AsyncLatch()
        let connection = AsyncLatch()

        // Drop the connection once the drive is actually running.
        let dropper = Task {
            await driveRunning.wait()
            await connection.trip()
        }

        let outcome = await StreamLifecycleDriver.run(
            transport: .init(
                waitForDisconnect: { await connection.wait() },
                idleFor: { _ in false },
                sendKeepalive: { true }
            ),
            keepaliveCadence: .milliseconds(5),
            onTransportCancel: { cancelBridged.set() },
            drive: {
                await driveRunning.trip()
                while !Task.isCancelled { try? await Task.sleep(for: .milliseconds(2)) }
                driveObservedCancel.set()
                return .cancelled
            }
        )
        await dropper.value

        guard case .disconnected(let source) = outcome else {
            Issue.record("expected .disconnected, got \(outcome)")
            return
        }
        #expect(source == .connectionState)
        #expect(cancelBridged.isSet)
        #expect(driveObservedCancel.isSet)
    }

    /// While the drive is quiet and the stream stays idle, the prober emits
    /// keepalives at the cadence; once the drive ends the whole race is torn
    /// down and no further keepalives are written.
    @Test func idleDriveEmitsKeepalivesThenStopsWhenDriveEnds() async {
        let keepalives = Counter()
        let finishDrive = AsyncLatch()
        let connection = connectionStaysUp()

        // End the drive only after a few keepalives have actually gone out —
        // deterministic, not time-based.
        let releaser = Task {
            while await keepalives.value < 3 { try? await Task.sleep(for: .milliseconds(1)) }
            await finishDrive.trip()
        }

        let outcome = await StreamLifecycleDriver.run(
            transport: .init(
                waitForDisconnect: { await connection.wait() },
                idleFor: { _ in true },
                sendKeepalive: {
                    await keepalives.increment()
                    return true
                }
            ),
            keepaliveCadence: .milliseconds(3),
            onTransportCancel: {},
            drive: {
                await finishDrive.wait()
                return completedOutcome()
            }
        )
        await releaser.value

        guard case .completed = outcome else {
            Issue.record("expected .completed, got \(outcome)")
            return
        }

        let atTeardown = await keepalives.value
        #expect(atTeardown >= 3, "keepalives fire while the idle drive is quiet")

        // The prober is torn down with the race: no keepalive after the drive ends.
        try? await Task.sleep(for: .milliseconds(30))
        #expect(await keepalives.value == atTeardown)
    }

    /// A drive that completes immediately wins before the first cadence tick, so
    /// neither watcher ever fires — no keepalives, no transport cancel.
    @Test func normalCompletionTearsDownBothWatchers() async {
        let keepalives = Counter()
        let cancelBridged = LeaseAcquiredSignal()
        let connection = connectionStaysUp()

        let outcome = await StreamLifecycleDriver.run(
            transport: .init(
                waitForDisconnect: { await connection.wait() },
                idleFor: { _ in true },
                sendKeepalive: {
                    await keepalives.increment()
                    return true
                }
            ),
            keepaliveCadence: .milliseconds(10),
            onTransportCancel: { cancelBridged.set() },
            drive: { completedOutcome() }
        )

        guard case .completed = outcome else {
            Issue.record("expected .completed, got \(outcome)")
            return
        }

        // Give the (now cancelled) watchers a window to misbehave.
        try? await Task.sleep(for: .milliseconds(40))
        #expect(await keepalives.value == 0, "a clean completion never keepalives")
        #expect(!cancelBridged.isSet, "a clean completion never cancels the transport")
    }

    /// A drive error passes through the race unchanged — the driver adds no
    /// interpretation to the drive's own terminal outcome.
    @Test func driveErrorPropagatesUnchanged() async {
        let connection = connectionStaysUp()

        let outcome = await StreamLifecycleDriver.run(
            transport: .init(
                waitForDisconnect: { await connection.wait() },
                idleFor: { _ in false },
                sendKeepalive: { true }
            ),
            keepaliveCadence: .milliseconds(5),
            onTransportCancel: {},
            drive: { .failed("boom") }
        )

        guard case .failed(let message) = outcome else {
            Issue.record("expected .failed, got \(outcome)")
            return
        }
        #expect(message == "boom")
    }

    /// Cancelling the whole run cancels every watcher and the drive, and the
    /// race resolves to `.cancelled` (it does not hang on the suspended probes).
    @Test func runCancellationPropagates() async {
        let connection = connectionStaysUp()

        let runner = Task {
            await StreamLifecycleDriver.run(
                transport: .init(
                    waitForDisconnect: { await connection.wait() },
                    idleFor: { _ in false },
                    sendKeepalive: { true }
                ),
                keepaliveCadence: .milliseconds(50),
                onTransportCancel: {},
                drive: {
                    while !Task.isCancelled { try? await Task.sleep(for: .milliseconds(2)) }
                    return .cancelled
                }
            )
        }

        // Let the race spin up, then cancel the enclosing task.
        try? await Task.sleep(for: .milliseconds(10))
        runner.cancel()
        let outcome = await runner.value

        guard case .cancelled = outcome else {
            Issue.record("expected .cancelled, got \(outcome)")
            return
        }
    }
}
