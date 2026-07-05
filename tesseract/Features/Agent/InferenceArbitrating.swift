//
//  InferenceArbitrating.swift
//  tesseract
//
//  The narrow seam the lease-acquiring consumers (`AgentRunController`,
//  `SpeechCoordinator`, `AgentCoordinator`) depend on — one member, the scoped
//  lease. Two adapters make it real (ADR-0001): the production
//  `InferenceArbiter`, and the in-memory peer in `tesseractTests` that records
//  lease calls and runs the body without loading a model.
//
//  Deliberately single-member: `reloadLLMIfNeeded` and the read-only model
//  state stay on the concrete facade — their only consumers already hold it.
//  Both adapters are `@MainActor final class`es (not actors), so this is a
//  plain `@MainActor` protocol — no `nonisolated` escape hatch is needed
//  (contrast the actor-backed speech model ports of ADR-0003).
//

import os

/// Lock-protected waiter visibility (PRD #173, ADR-0022 **Boundary Yield**):
/// the one fact the **Batch Engine** needs off-MainActor — "is a
/// slot-preserving consumer (TTS) waiting on the lease right now?" — readable
/// at step/chunk boundaries without an actor hop. Deliberately counts *only*
/// slot-preserving waiters: yielding to a consumer that could change the
/// loaded model (`reloadLLMIfNeeded`, a different-model completion) would
/// reload under paused lanes — that case is an **Admission Freeze**, never a
/// yield. A count, not a bool, so overlapping waiters can't blank each other.
nonisolated final class LeaseWaiterSignal: Sendable {
    private let count = OSAllocatedUnfairLock(initialState: 0)

    var hasWaiters: Bool {
        count.withLock { $0 > 0 }
    }

    /// Mutated by `InferenceArbiter` around a slot-preserving consumer's wait
    /// (and by the in-memory test peer directly).
    func increment() {
        count.withLock { $0 += 1 }
    }

    func decrement() {
        count.withLock { $0 = max(0, $0 - 1) }
    }
}

/// How a lease chooses the `.llm` slot's vision mode (ADR-0008).
nonisolated enum LLMVisionRequirement: Sendable, Equatable {
    /// Chat UI and background agents: load vision when the user's global
    /// "Use vision models when available" opt-out is on *and* the model is
    /// capable. Opting out forces the text-only container.
    case fromSettings
    /// HTTP server path: vision whenever the target model is capable, so a
    /// generated client config that advertises image input is always honored —
    /// the global opt-out cannot silently break a configured client.
    case visionIfCapable

    /// Resolve to a concrete "load the vision container?" decision. Pure so the
    /// policy is unit-tested without the arbiter: `.fromSettings` honors the
    /// global opt-out *and* capability; `.visionIfCapable` ignores the opt-out
    /// (ADR-0008) and follows capability alone.
    nonisolated func wantsVision(useVisionWhenAvailable: Bool, isVisionCapable: Bool) -> Bool {
        switch self {
        case .fromSettings: useVisionWhenAvailable && isVisionCapable
        case .visionIfCapable: isVisionCapable
        }
    }
}

/// Scoped exclusive GPU access with the required model loaded: waits FIFO-fair
/// for the lease, ensures `slot`'s model is resident, runs `body`, releases on
/// exit — including on throw.
@MainActor
protocol InferenceArbitrating {
    func withExclusiveGPU<T: Sendable>(
        _ slot: ModelSlot,
        llmModelIDOverride: String?,
        llmVision: LLMVisionRequirement,
        body: () async throws -> T
    ) async throws -> T

    /// Waiter visibility for the **Batch Engine**'s Boundary Yield (PRD #173):
    /// whether a slot-preserving consumer (TTS) is queued on the lease,
    /// readable off-MainActor at step/chunk boundaries. The one widening
    /// ADR-0022 called for.
    nonisolated var leaseWaiters: LeaseWaiterSignal { get }
}

extension InferenceArbitrating {
    /// Protocol requirements cannot carry default arguments; this restores the
    /// common two-argument call shape (`withExclusiveGPU(.llm) { … }`).
    func withExclusiveGPU<T: Sendable>(
        _ slot: ModelSlot,
        body: () async throws -> T
    ) async throws -> T {
        try await withExclusiveGPU(
            slot,
            llmModelIDOverride: nil,
            llmVision: .fromSettings,
            body: body
        )
    }
}
