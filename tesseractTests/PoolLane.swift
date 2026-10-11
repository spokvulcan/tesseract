import Foundation
import Synchronization

/// A FIFO async semaphore over the test host's cooperative thread pool:
/// `enter()` suspends while `width` cases are inside, `leave()` admits the
/// next waiter. A case waiting for its lane is suspended and holds no thread.
///
/// Each lane takes a quarter of the cores, so the two together leave the rest
/// of the pool to the other suites, and to the code that opens the gates the
/// parking cases wait on.
nonisolated final class PoolLane: Sendable {
    /// ``CPUBoundTrait``'s cases, which compute for seconds on their thread.
    static let cpuBound = PoolLane(width: quarterOfTheCores)

    /// ``ParksPoolThreadsTrait``'s cases, which park a thread on a
    /// ``ParkingGate`` until the test opens it.
    static let parking = PoolLane(width: quarterOfTheCores)

    private static var quarterOfTheCores: Int {
        max(1, ProcessInfo.processInfo.activeProcessorCount / 4)
    }

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
