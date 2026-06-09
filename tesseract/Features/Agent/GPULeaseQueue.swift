//
//  GPULeaseQueue.swift
//  tesseract
//
//  The **GPU Lease Queue** — the pure mutual-exclusion lease carved out of the
//  `InferenceArbiter` (issue #58). It owns only the lease protocol: the FIFO
//  waiter queue, the `isLeased` flag, the atomic handoff, and the cancellation
//  rules. It is slot-agnostic — it knows nothing of `ModelSlot`, engines, or
//  models — so it constructs with `()` and is unit-tested directly, the way
//  `OperationGuard` is tested as a value.
//

import Foundation

/// Serializes access to one exclusive resource (the GPU) behind a single scoped
/// operation, `withExclusive`. Only one body runs at a time; contended callers
/// queue FIFO.
@MainActor
final class GPULeaseQueue {

    /// Whether a `withExclusive` body currently holds the lease. Read-only to
    /// callers; tests use it to pin the atomic-handoff contract.
    private(set) var isLeased = false

    /// FIFO queue of contended callers waiting for the lease. Entries are keyed
    /// by UUID so a cancelled waiter can be removed without disturbing the order.
    private var waiters: [(id: UUID, continuation: CheckedContinuation<Void, any Error>)] = []

    /// Run `body` while exclusively holding the lease, releasing on exit —
    /// including on throw. Contended callers wait in FIFO order; arriving while
    /// waiters are queued also queues (no queue-bypass).
    ///
    /// Cancellation:
    ///   - While waiting in the queue: the waiter is removed and
    ///     `CancellationError` is thrown without ever acquiring the lease.
    ///   - During the handoff race (resumed by the releasing holder, cancelled
    ///     before its own job claims): the pre-claim `Task.checkCancellation()`
    ///     throws — the body never runs — and the lease the handoff carried is
    ///     released onward (next waiter, or cleared), never orphaned.
    ///   - Once the body runs: cancellation propagates normally through it and
    ///     the lease is released via `defer`.
    func withExclusive<T: Sendable>(_ body: () async throws -> T) async throws -> T {
        if isLeased || !waiters.isEmpty {
            let waiterID = UUID()
            try await withTaskCancellationHandler {
                try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
                    if Task.isCancelled {
                        continuation.resume(throwing: CancellationError())
                        return
                    }
                    waiters.append((id: waiterID, continuation: continuation))
                }
            } onCancel: {
                // Runs concurrently — MainActor hop to safely mutate waiters.
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    if let idx = self.waiters.firstIndex(where: { $0.id == waiterID }) {
                        let removed = self.waiters.remove(at: idx)
                        removed.continuation.resume(throwing: CancellationError())
                    }
                    // If already removed by the handoff (which removes before
                    // resuming), firstIndex returns nil — no double-resume.
                }
            }
        }
        isLeased = true
        // A waiter resumed by the handoff already owns the lease (`isLeased`
        // stayed true on its behalf). If cancellation won the race, it must not
        // run the body — and must pass the lease on rather than strand it.
        do {
            try Task.checkCancellation()
        } catch {
            release()
            throw error
        }
        defer { release() }
        return try await body()
    }

    /// Atomic handoff: if waiters are queued, keep `isLeased` true and resume the
    /// next waiter directly — the lease changes hands without an instant where the
    /// queue looks free, so a third caller can never barge between holder and
    /// waiter. Only when the queue drains does `isLeased` clear.
    private func release() {
        if waiters.isEmpty {
            isLeased = false
            Log.general.info("GPULeaseQueue: lease released, queue drained")
        } else {
            Log.general.debug("GPULeaseQueue: handing off lease, \(self.waiters.count - 1) still queued")
            waiters.removeFirst().continuation.resume()
        }
    }
}
