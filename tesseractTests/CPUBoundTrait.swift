import Foundation
import Synchronization
import Testing

// MARK: - CPUBoundTrait

/// Marks a suite whose test cases run seconds of synchronous CPU work, such
/// as loading a real tokenizer in a Debug build or scanning the app sources.
/// At most ``CPUBoundLane/shared``'s width of such cases run at once in the
/// test process.
///
/// Every suite in a parallel run shares one test host and so one Swift
/// cooperative thread pool, a thread per core. A case that computes without
/// suspending keeps its pool thread until it returns. When the real-tokenizer
/// suites all started together they took every pool thread for about 40 s, and
/// everything else waited for one: actor jobs, `Task.sleep` wake-ups (which
/// resume through the global executor even for main-actor code), the HTTP
/// fixtures' connection tasks. Tests with 3–10 s deadlines failed in a dozen
/// unrelated suites. A case waiting for the lane is suspended and holds no
/// thread.
///
/// Apply it to the suite: `@Suite(.cpuBound)`. It applies to each test case,
/// so a parameterized test's cases take turns too.
nonisolated struct CPUBoundTrait: SuiteTrait, TestTrait, TestScoping {
    var isRecursive: Bool { true }

    func scopeProvider(for test: Test, testCase: Test.Case?) -> Self? {
        testCase == nil ? nil : self
    }

    func provideScope(
        for test: Test, testCase: Test.Case?,
        performing function: @concurrent @Sendable () async throws -> Void
    ) async throws {
        await CPUBoundLane.shared.enter()
        defer { CPUBoundLane.shared.leave() }
        try await function()
    }
}

extension Trait where Self == CPUBoundTrait {
    /// The case runs long synchronous CPU work; see ``CPUBoundTrait``.
    static var cpuBound: Self { Self() }
}

// MARK: - CPUBoundLane

/// A FIFO async semaphore: `enter()` suspends while `width` holders are
/// inside, `leave()` admits the next waiter.
nonisolated final class CPUBoundLane: Sendable {
    /// A quarter of the cores, leaving the rest of the pool to the other suites.
    static let shared = CPUBoundLane(
        width: max(1, ProcessInfo.processInfo.activeProcessorCount / 4))

    private struct State {
        var free: Int
        var waiters: [CheckedContinuation<Void, Never>] = []
    }

    private let state: Mutex<State>

    init(width: Int) {
        state = Mutex(State(free: width))
    }

    func enter() async {
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            let admitted = state.withLock { state in
                guard state.free > 0 else {
                    state.waiters.append(continuation)
                    return false
                }
                state.free -= 1
                return true
            }
            if admitted { continuation.resume() }
        }
    }

    func leave() {
        let next = state.withLock { state -> CheckedContinuation<Void, Never>? in
            guard !state.waiters.isEmpty else {
                state.free += 1
                return nil
            }
            return state.waiters.removeFirst()
        }
        next?.resume()
    }
}
