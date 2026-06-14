import Testing

@testable import Tesseract_Agent

/// Tests for the **GPU Lease Queue** — the pure, zero-dependency mutual-exclusion
/// lease carved out of the `InferenceArbiter` (issue #58). Driven directly through
/// the single `withExclusive` operation, the way `OperationGuard` is tested as a
/// value: no engines, no models, no peer.
///
/// Determinism: every test runs on the MainActor and sequences concurrent callers
/// with held continuations (`Gate`) and explicit `Task.yield()` handoffs — never
/// wall-clock sleeps. All assertions target the observable contract of
/// `withExclusive` (ordering, throwing, exclusion, `isLeased`), never the private
/// waiter array.
@MainActor
struct GPULeaseQueueTests {

    /// A continuation-backed latch: bodies park on `wait()` until the test calls
    /// `open()`, letting a test hold the lease exactly as long as it needs.
    @MainActor
    private final class Gate {
        private var isOpen = false
        private var parked: [CheckedContinuation<Void, Never>] = []

        func wait() async {
            if isOpen { return }
            await withCheckedContinuation { parked.append($0) }
        }

        func open() {
            isOpen = true
            let resuming = parked
            parked.removeAll()
            for continuation in resuming { continuation.resume() }
        }
    }

    /// Records body entry/exit marks so ordering and overlap are assertable
    /// without sharing a mutable local across tasks.
    @MainActor
    private final class EventLog {
        private(set) var events: [String] = []
        func add(_ event: String) { events.append(event) }
    }

    /// Give freshly-created tasks room to run up to their next suspension point.
    /// Bounded yields, no wall-clock time.
    private func settle(_ rounds: Int = 10) async {
        for _ in 0..<rounds { await Task.yield() }
    }

    /// The tracer: an uncontended `withExclusive` runs its body, returns its value,
    /// and leaves the lease free.
    @Test
    func uncontendedCallRunsBodyAndReleases() async throws {
        let queue = GPULeaseQueue()

        let value = try await queue.withExclusive { 42 }

        #expect(value == 42)
        #expect(!queue.isLeased)
    }

    /// The mutual-exclusion guarantee itself: a second caller arriving while the
    /// lease is held must not enter its body until the holder's body has exited.
    @Test
    func contendedBodiesNeverOverlap() async throws {
        let queue = GPULeaseQueue()
        let log = EventLog()
        let gate = Gate()

        let holder = Task {
            try await queue.withExclusive {
                log.add("A-in")
                await gate.wait()
                log.add("A-out")
            }
        }
        await settle()

        let contender = Task {
            try await queue.withExclusive {
                log.add("B-in")
                log.add("B-out")
            }
        }
        await settle()

        // The holder is parked inside its body; the contender must still be queued.
        #expect(log.events == ["A-in"])

        gate.open()
        try await holder.value
        try await contender.value

        #expect(log.events == ["A-in", "A-out", "B-in", "B-out"])
    }

    /// Queued waiters acquire the lease in arrival order — a queue-bypass
    /// regression (any later arrival overtaking an earlier one) breaks this.
    @Test
    func queuedWaitersAcquireInFIFOOrder() async throws {
        let queue = GPULeaseQueue()
        let log = EventLog()
        let gate = Gate()

        let holder = Task {
            try await queue.withExclusive { await gate.wait() }
        }
        await settle()

        var contenders: [Task<Void, any Error>] = []
        for name in ["B", "C", "D"] {
            contenders.append(
                Task {
                    try await queue.withExclusive { log.add(name) }
                })
            // Let each contender reach its queue slot before the next arrives,
            // so arrival order is pinned.
            await settle()
        }

        gate.open()
        try await holder.value
        for contender in contenders { try await contender.value }

        #expect(log.events == ["B", "C", "D"])
    }

    /// The atomic handoff: on release with waiters queued, the lease passes to the
    /// next waiter directly — there is no instant where the queue looks free. A
    /// release that clears `isLeased` and *then* wakes the waiter opens a one-job
    /// window where a third caller sees a free queue and barges past it; this test
    /// schedules a barger into exactly that window and demands FIFO survives.
    @Test
    func handoffLeavesNoWindowForABargingCaller() async throws {
        let queue = GPULeaseQueue()
        let log = EventLog()
        let holderGate = Gate()
        let bargeSignal = Gate()

        let holder = Task {
            try await queue.withExclusive {
                log.add("A")
                await holderGate.wait()
            }
        }
        await settle()

        let queued = Task {
            try await queue.withExclusive { log.add("B") }
        }
        await settle()

        // Parked before its withExclusive call; released into the handoff window.
        let barger = Task {
            await bargeSignal.wait()
            try await queue.withExclusive { log.add("C") }
        }
        await settle()

        // Order matters: the holder's release job runs first, then the barger's
        // withExclusive lands between the handoff and the queued waiter's resume.
        holderGate.open()
        bargeSignal.open()

        try await holder.value
        try await queued.value
        try await barger.value

        #expect(log.events == ["A", "B", "C"])
    }

    /// A waiter cancelled while queued throws `CancellationError` without ever
    /// acquiring the lease — its body never runs — and the queue stays functional
    /// for later callers.
    @Test
    func cancelWhileQueuedThrowsWithoutAcquiring() async throws {
        let queue = GPULeaseQueue()
        let log = EventLog()
        let gate = Gate()

        let holder = Task {
            try await queue.withExclusive { await gate.wait() }
        }
        await settle()

        let cancelled = Task {
            try await queue.withExclusive { log.add("X") }
        }
        await settle()

        cancelled.cancel()
        gate.open()

        try await holder.value
        let outcome = await cancelled.result
        #expect(throws: CancellationError.self) { try outcome.get() }
        #expect(log.events.isEmpty)

        // The queue is not wedged: a fresh caller acquires and releases normally.
        let value = try await queue.withExclusive { 7 }
        #expect(value == 7)
        #expect(!queue.isLeased)
    }

    /// The handoff race: a waiter whose task is cancelled *after* the releasing
    /// holder has resumed it, but *before* it claims the lease, must not inherit
    /// it — its body never runs and it throws `CancellationError`. And the lease
    /// must pass onward, not be orphaned: a bystander queued behind the cancelled
    /// waiter still acquires, and the queue drains to free. (The orphaning half is
    /// the wedge this test exists to lock out: a pre-claim cancellation check that
    /// merely throws strands `isLeased` true forever, stalling all inference.)
    ///
    /// Job ordering aims the cancel into the window — `gate.open()` enqueues the
    /// holder's release job, then the cancel runs as its own MainActor job behind
    /// it, after the handoff has resumed the victim but before the victim's own
    /// job claims. Scheduler ordering is not contractual, so the scenario sweeps
    /// ten iterations: every interleaving must uphold both invariants (body never
    /// runs + lease passes onward), and an implementation with the window open
    /// cannot win the race ten times in a row.
    @Test
    func cancelDuringHandoffCannotInheritTheLease() async throws {
        for iteration in 0..<10 {
            let queue = GPULeaseQueue()
            let log = EventLog()
            let gate = Gate()

            let holder = Task {
                try await queue.withExclusive { await gate.wait() }
            }
            await settle()

            let victim = Task {
                try await queue.withExclusive { log.add("X") }
            }
            await settle()

            let bystander = Task {
                try await queue.withExclusive { log.add("B") }
            }
            await settle()

            gate.open()
            // Enqueued behind the holder's release job: by the time this runs,
            // the handoff has resumed the victim but the victim's job has not run.
            let canceller = Task { victim.cancel() }

            try await holder.value
            await canceller.value
            let outcome = await victim.result
            #expect(throws: CancellationError.self) { try outcome.get() }

            // Hang-proof wedge probe: if the lease was orphaned, the bystander
            // never runs; assert via the log after settling, then cancel as
            // cleanup so a regression fails fast instead of deadlocking the suite.
            await settle()
            #expect(log.events == ["B"], "iteration \(iteration)")
            #expect(!queue.isLeased, "iteration \(iteration)")
            bystander.cancel()
            _ = await bystander.result
        }
    }

    /// A throwing body can never wedge the queue: the error propagates to the
    /// caller, the lease is released, and a queued waiter still acquires.
    @Test
    func bodyThrowReleasesTheLeaseToTheNextWaiter() async throws {
        struct BodyError: Error {}
        let queue = GPULeaseQueue()
        let log = EventLog()
        let gate = Gate()

        let thrower = Task {
            try await queue.withExclusive {
                await gate.wait()
                throw BodyError()
            }
        }
        await settle()

        let waiter = Task {
            try await queue.withExclusive { log.add("W") }
        }
        await settle()

        gate.open()
        let outcome = await thrower.result
        #expect(throws: BodyError.self) { try outcome.get() }

        try await waiter.value
        #expect(log.events == ["W"])
        #expect(!queue.isLeased)
    }
}
