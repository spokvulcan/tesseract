import Dispatch
import Foundation
import Synchronization
import Testing

// MARK: - ParksPoolThreadsTrait

/// Marks a test case that parks a cooperative-pool thread on a
/// ``ParkingGate``: a toy model held mid-forward (``ForwardGate``), the SSD
/// writer held inside a payload's materialize step (``BlockingMaterializer``).
/// At most ``PoolLane/parking``'s width of such cases run at once, and the
/// gates a case leaves shut open when it ends.
///
/// The parked thread waits for the test, and the test needs the pool to reach
/// its `open()`: an actor hop, a `Task.sleep` wake-up. Once as many gates park
/// as the pool has threads, nothing can run that would open them. The CI
/// runner's pool has three threads and a run starts 64 cases at once; there
/// every run hung until the job's two-hour timeout.
///
/// Apply it to the test, or to a suite whose every test parks:
/// `@Test(.parksPoolThreads)`. A gate made outside it records an issue. On a
/// one-thread pool (`LIBDISPATCH_COOPERATIVE_POOL_STRICT=1`) the case is
/// cancelled, as its first park would hang the run.
nonisolated struct ParksPoolThreadsTrait: SuiteTrait, TestTrait, TestScoping {
    var isRecursive: Bool { true }

    func scopeProvider(for test: Test, testCase: Test.Case?) -> Self? {
        testCase == nil ? nil : self
    }

    func provideScope(
        for test: Test, testCase: Test.Case?,
        performing function: @concurrent @Sendable () async throws -> Void
    ) async throws {
        if ProcessInfo.processInfo.environment["LIBDISPATCH_COOPERATIVE_POOL_STRICT"] == "1" {
            try Test.cancel("Parks a pool thread, and this pool has one")
        }
        await PoolLane.parking.enter()
        defer { PoolLane.parking.leave() }
        let gates = ParkingGate.Registry()
        defer { gates.openAll() }
        try await ParkingGate.$registry.withValue(gates) { try await function() }
    }
}

extension Trait where Self == ParksPoolThreadsTrait {
    /// The case parks a pool thread on a ``ParkingGate``; see
    /// ``ParksPoolThreadsTrait``.
    static var parksPoolThreads: Self { Self() }
}

// MARK: - ParkingGate

/// Holds the thread that calls ``park()`` until the test calls ``open()``.
/// Synchronous code, such as a model's forward or a payload's materialize
/// step, can only be held this way, and the thread it runs on is a pool
/// thread, kept for the whole wait. The case that makes a gate takes
/// `@Test(.parksPoolThreads)`.
///
/// `open()` latches: every later park passes, and an `open()` before the
/// thread arrives is remembered. A park still waiting after ``waitBackstop``
/// records an issue where the gate was made and lets the thread go, so a test
/// that never opens its gate fails instead of hanging the run.
nonisolated final class ParkingGate: Sendable {
    /// The gates the current ``ParksPoolThreadsTrait`` case has made; nil
    /// outside one.
    @TaskLocal static var registry: Registry?

    private let opened = DispatchGroup()
    private let isOpen = Mutex(false)
    private let sourceLocation: SourceLocation

    init(sourceLocation: SourceLocation = #_sourceLocation) {
        self.sourceLocation = sourceLocation
        opened.enter()
        if let registry = Self.registry {
            registry.add(self)
        } else {
            Issue.record(
                "A ParkingGate parks a pool thread: mark the test @Test(.parksPoolThreads)",
                sourceLocation: sourceLocation)
        }
    }

    func park() {
        let backstop = DispatchTimeInterval.seconds(Int(waitBackstop.components.seconds))
        if opened.wait(timeout: .now() + backstop) == .timedOut {
            Issue.record(
                "A thread parked here for \(waitBackstop) without the gate opening",
                sourceLocation: sourceLocation)
        }
    }

    func open() {
        let wasOpen = isOpen.withLock { isOpen in
            defer { isOpen = true }
            return isOpen
        }
        if !wasOpen { opened.leave() }
    }

    /// Opens a case's gates when it ends, so no thread it parked outlives its
    /// place in the lane.
    nonisolated final class Registry: Sendable {
        private let gates = Mutex<[ParkingGate]>([])

        func add(_ gate: ParkingGate) {
            gates.withLock { $0.append(gate) }
        }

        func openAll() {
            let made = gates.withLock { gates in
                defer { gates.removeAll() }
                return gates
            }
            for gate in made { gate.open() }
        }
    }
}
