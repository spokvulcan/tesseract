//
//  LLMGate.swift
//  tesseract
//
//  The **LLM Gate** — one language-model generation at a time. The loaded
//  model is one container with one prefix cache, and the server's cache
//  claim, leaf handoff and active-generation slot are single-tenant, so chat
//  turns, the Companion's moments, HTTP requests, `/compact`, model reloads
//  and offloads take turns here, FIFO. Nothing else does: speech, dictation,
//  the proofreader and the embedder run beside the LLM (ADR-0081 — it was
//  the GPU Lease Queue, which made the voice wait for whole generations).
//  MLX keeps concurrent evaluation safe with its own process-wide lock.
//
//  It owns only the gate protocol: the FIFO waiter queue, the `isHeld`
//  flag, the atomic handoff, and the cancellation rules — no models, no
//  engines — so it constructs with `()` and is unit-tested directly.
//

import Foundation

/// Serializes LLM work behind a single scoped operation, `withExclusive`. Only
/// one body runs at a time; contended callers queue FIFO.
@MainActor
final class LLMGate {

    /// Whether a `withExclusive` body is running now. Read-only to callers;
    /// tests use it to pin the atomic-handoff contract.
    private(set) var isHeld = false

    /// FIFO queue of contended callers waiting for the gate. Entries are keyed
    /// by UUID so a cancelled waiter can be removed without disturbing the order.
    private var waiters: [(id: UUID, continuation: CheckedContinuation<Void, any Error>)] = []

    /// Run `body` while exclusively holding the gate, releasing on exit —
    /// including on throw. Contended callers wait in FIFO order; arriving while
    /// waiters are queued also queues (no queue-bypass).
    ///
    /// Cancellation:
    ///   - While waiting in the queue: the waiter is removed and
    ///     `CancellationError` is thrown without ever acquiring the gate.
    ///   - During the handoff race (resumed by the releasing holder, cancelled
    ///     before its own job claims): the pre-claim `Task.checkCancellation()`
    ///     throws — the body never runs — and the gate the handoff carried is
    ///     released onward (next waiter, or cleared), never orphaned.
    ///   - Once the body runs: cancellation propagates normally through it and
    ///     the gate is released via `defer`.
    func withExclusive<T: Sendable>(_ body: () async throws -> T) async throws -> T {
        if isHeld || !waiters.isEmpty {
            let waiterID = UUID()
            try await withTaskCancellationHandler {
                try await withCheckedThrowingContinuation {
                    (continuation: CheckedContinuation<Void, any Error>) in
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
        isHeld = true
        // A waiter resumed by the handoff already owns the gate (`isHeld`
        // stayed true on its behalf). If cancellation won the race, it must not
        // run the body — and must pass the gate on rather than strand it.
        do {
            try Task.checkCancellation()
        } catch {
            release()
            throw error
        }
        defer { release() }
        return try await body()
    }

    /// Atomic handoff: if waiters are queued, keep `isHeld` true and resume the
    /// next waiter directly — the gate changes hands without an instant where the
    /// queue looks free, so a third caller can never barge between holder and
    /// waiter. Only when the queue drains does `isHeld` clear.
    private func release() {
        if waiters.isEmpty {
            isHeld = false
            Log.general.info("LLMGate: released, queue drained")
        } else {
            Log.general.debug(
                "LLMGate: handing off, \(self.waiters.count - 1) still queued")
            waiters.removeFirst().continuation.resume()
        }
    }
}
