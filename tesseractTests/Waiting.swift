import Observation

/// How long a test waits for something that must happen before calling it a
/// hang. A backstop, not a latency budget: a correct run never comes near it,
/// but two full runs and an agent's focused suites can share this Mac's cores,
/// and then one hop between an actor and the main actor can take seconds. A
/// budget short enough to notice fails tests whose code is fine.
let waitBackstop: Duration = .seconds(60)

/// Suspends until `condition` holds, re-evaluating it whenever the
/// `@Observable` state it reads changes. Nothing polls, so `condition` must
/// read only observable state; for anything else, use `waitUntil`.
///
/// There is no deadline: a condition that never holds is a hang, which the
/// suite's `.timeLimit` ends by cancelling the test. Returns whether the
/// condition holds then, so `#expect(await observe(until: …))` names the
/// condition that never came true.
@MainActor
@discardableResult
func observe(until condition: @escaping @MainActor @Sendable () -> Bool) async -> Bool {
    for await holds in Observations(condition) where holds { return true }
    return condition()
}

/// Polls `condition` every 10 ms until it holds, for state that isn't
/// `@Observable` (a writer's callbacks, a file on disk). Returns the final
/// value for the caller's `#expect` once `timeout` passes or the test is
/// cancelled.
@MainActor
func waitUntil(
    timeout: Duration = waitBackstop,
    _ condition: @MainActor () -> Bool
) async -> Bool {
    let deadline = ContinuousClock.now + timeout
    while ContinuousClock.now < deadline, !Task.isCancelled {
        if condition() { return true }
        try? await Task.sleep(for: .milliseconds(10))
    }
    return condition()
}
