import Foundation

/// One iteration's view of the **Batch Engine**'s world (PRD #173,
/// ADR-0022): the waiting queue, the live **Lane**s, and whether any
/// slot-preserving consumer is waiting on the GPU lease. Deliberately
/// clock-free (`waitedSeconds`, not timestamps) and identity-thin, so the
/// scheduling core stays a pure value function.
nonisolated struct BatchEngineSnapshot: Sendable, Equatable {

    /// One waiting completion, not yet a lane.
    struct QueueEntry: Sendable, Equatable {
        let requestID: UUID
        /// Strict arrival index — the FIFO key.
        let arrivalOrder: Int
        /// How long this entry has waited; at the aging threshold the entry
        /// leaves cache-aware ordering and is served strictly FIFO.
        let waitedSeconds: TimeInterval
        /// Longest radix-tree prefix match at enqueue (SGLang cache-aware
        /// ordering); 0 when no probe was possible.
        let matchedPrefixLength: Int
        /// A request the pool must drain for before it runs solo — an
        /// image-bearing request, or one whose generation runs the
        /// monolithic single-flight path (**Admission Freeze**).
        let requiresExclusivePool: Bool
        /// Whether the currently loaded model state satisfies this entry's
        /// demand (model + vision, ADR-0008 upgrade-never-downgrade). False
        /// freezes admissions and, once the pool is empty, asks the engine
        /// to re-acquire the lease with this entry's demand so the arbiter
        /// reloads.
        let demandSatisfiedByLoadedModel: Bool
    }

    /// Where a lane is in its life: each phase names the kind of engine step
    /// it needs next.
    enum LanePhase: Sendable, Equatable {
        /// Admitted, awaiting its startup step (tokenize, resolve, restore).
        case starting
        /// Chunked prefill in progress.
        case prefilling
        /// Streaming decode steps.
        case decoding
        /// Decode ended; leaf capture / teardown work remains.
        case finishing
    }

    /// One live lane, as the scheduler sees it.
    struct Lane: Sendable, Equatable {
        let requestID: UUID
        /// Strict admission index — FIFO fairness key among lanes.
        let admissionOrder: Int
        let phase: LanePhase
        /// Whether the lane's drive has a step waiting for a grant right now.
        let hasPendingStep: Bool
        /// The lane's solo prefill chunk size (its request's
        /// `prefillStepSize`); the policy shrinks it while siblings decode.
        let preferredPrefillChunk: Int
        /// An **Admission Freeze** resident: runs alone, blocks admissions
        /// until drained (different model or image-bearing request).
        let isExclusive: Bool
    }

    /// The waiting queue in arrival order.
    let queue: [QueueEntry]
    /// Live lanes in admission order.
    let lanes: [Lane]
    /// `min(hard cap, headroom lanes)` — computed by the engine from the
    /// Active-Inference Reserve arithmetic.
    let laneCapacity: Int
    /// A slot-preserving consumer (TTS) is waiting on the GPU lease —
    /// **Boundary Yield** beats every other action.
    let leaseHasWaiters: Bool
    /// Whether the previous grant was prefill-class (startup / prefill chunk /
    /// finish) — the alternation bit that bounds a decode lane's stall at one
    /// chunk.
    let lastGrantWasPrefillClass: Bool
}

/// What the engine does next. One action per iteration keeps the core
/// exhaustively testable as plain values.
nonisolated enum BatchEngineDecision: Sendable, Equatable {
    /// Release the lease to a waiting slot-preserving consumer at this
    /// boundary; lanes pause as plain data (**Boundary Yield**).
    case yieldLease
    /// Turn the ordered queue's head into a lane.
    case admit(UUID)
    /// Pool is empty: run the drain-requiring head solo (**Admission
    /// Freeze**'s second half — the freeze itself is the *absence* of
    /// `admit` while the head requires an exclusive pool).
    case admitExclusive(UUID)
    /// Pool is empty and the head's model demand is not satisfied by the
    /// loaded state: release the lease and re-acquire with the head's
    /// demand so the arbiter reloads under it.
    case switchPool(UUID)
    /// Execute one prefill-class step for one lane.
    case grant(UUID, BatchEngineGrant)
    /// One decode step for every decoding lane with a pending step.
    case decodeRound([UUID])
    /// Nothing to do — park until a submission, step request, or waiter.
    case idle
}

/// The prefill-class step kinds a single lane can be granted.
nonisolated enum BatchEngineGrant: Sendable, Equatable {
    case startup
    case prefillChunk(tokens: Int)
    case finish
}

/// The pure scheduling core: queue ordering (longest-prefix-match aged to
/// strict FIFO) and the per-iteration decision. Policy only — no clocks, no
/// actors, no GPU.
nonisolated enum BatchEnginePolicy {

    /// After this wait an entry stops competing on prefix match and is served
    /// in strict arrival order (ADR-0022: "aged to strict FIFO after ~10 s").
    static let agingThresholdSeconds: TimeInterval = 10

    /// The lane hard cap — a constant, not a Setting (ADR-0022); the
    /// decode-shape bench may override it to 8.
    static let hardLaneCap = 4

    /// How many lanes the machine's measured headroom funds: each lane costs
    /// one Active-Inference Reserve; claimable room is the ceiling's damped
    /// headroom fraction plus the cache bytes an eviction drain could demote
    /// (ADR-0018 arithmetic). Floored at one lane — a big model serializes to
    /// N=1 by arithmetic, never by swap.
    static func admittableLanes(
        headroomBytes: Int,
        evictableCacheBytes: Int,
        perLaneBytes: Int,
        hardCap: Int = hardLaneCap
    ) -> Int {
        guard perLaneBytes > 0 else { return max(1, hardCap) }
        let claimable = Int(
            Double(max(headroomBytes, 0)) * DynamicCeilingPolicy.headroomFraction)
        let funded = (claimable + max(evictableCacheBytes, 0)) / perLaneBytes
        return max(1, min(hardCap, funded))
    }

    /// **Lane Admission** order: entries that have aged past the threshold
    /// first, in strict arrival order; then fresh entries by longest prefix
    /// match, ties broken by arrival.
    static func orderedQueue(
        _ entries: [BatchEngineSnapshot.QueueEntry]
    ) -> [BatchEngineSnapshot.QueueEntry] {
        let aged =
            entries
            .filter { $0.waitedSeconds >= agingThresholdSeconds }
            .sorted { $0.arrivalOrder < $1.arrivalOrder }
        let fresh =
            entries
            .filter { $0.waitedSeconds < agingThresholdSeconds }
            .sorted {
                if $0.matchedPrefixLength != $1.matchedPrefixLength {
                    return $0.matchedPrefixLength > $1.matchedPrefixLength
                }
                return $0.arrivalOrder < $1.arrivalOrder
            }
        return aged + fresh
    }

    /// The per-iteration decision. Precedence: boundary yield, then
    /// admission (bookkeeping — no GPU), then one prefill-class grant or a
    /// decode round, alternating so a decode lane's stall is bounded by one
    /// chunk.
    static func decide(_ snapshot: BatchEngineSnapshot) -> BatchEngineDecision {
        if snapshot.leaseHasWaiters {
            return .yieldLease
        }

        // Lane Admission / Admission Freeze.
        let hasExclusiveLane = snapshot.lanes.contains(where: \.isExclusive)
        if let head = orderedQueue(snapshot.queue).first, !hasExclusiveLane {
            if !head.demandSatisfiedByLoadedModel {
                if snapshot.lanes.isEmpty {
                    return .switchPool(head.requestID)
                }
                // Freeze: the pool drains by attrition before the reload.
            } else if head.requiresExclusivePool {
                if snapshot.lanes.isEmpty {
                    return .admitExclusive(head.requestID)
                }
                // Freeze: no admissions — not even compatible later entries —
                // while the pool drains by attrition. Fall through to step.
            } else if snapshot.lanes.count < snapshot.laneCapacity {
                return .admit(head.requestID)
            }
        }

        return stepDecision(snapshot)
    }

    /// Prefill chunk size while any lane is decoding — bounds a decode
    /// lane's stall at roughly a second on the largest model (ADR-0022).
    static let decodingPrefillChunk = 512

    /// The stepping half: choose among the lanes' pending step requests.
    /// Prefill-class work (finish > startup > prefill chunk, admission
    /// order) alternates with decode rounds; a decode round steps every
    /// decoding lane with a pending step, admission order.
    private static func stepDecision(
        _ snapshot: BatchEngineSnapshot
    ) -> BatchEngineDecision {
        let pending = snapshot.lanes.filter(\.hasPendingStep)
        let decodeLanes =
            pending
            .filter { $0.phase == .decoding }
            .sorted { $0.admissionOrder < $1.admissionOrder }
        let prefillClass =
            pending
            .filter { $0.phase != .decoding }
            .sorted {
                // Finish first — it frees a lane; then admission order,
                // which also serializes prefills ("one prefill at a time,
                // admission order").
                let leftFinishing = $0.phase == .finishing
                let rightFinishing = $1.phase == .finishing
                if leftFinishing != rightFinishing { return leftFinishing }
                return $0.admissionOrder < $1.admissionOrder
            }

        if snapshot.lastGrantWasPrefillClass, !decodeLanes.isEmpty {
            return .decodeRound(decodeLanes.map(\.requestID))
        }
        if let next = prefillClass.first {
            switch next.phase {
            case .finishing:
                return .grant(next.requestID, .finish)
            case .starting:
                return .grant(next.requestID, .startup)
            case .prefilling:
                let anyDecoding = snapshot.lanes.contains { $0.phase == .decoding }
                let tokens =
                    anyDecoding
                    ? min(Self.decodingPrefillChunk, next.preferredPrefillChunk)
                    : next.preferredPrefillChunk
                return .grant(next.requestID, .prefillChunk(tokens: tokens))
            case .decoding:
                break  // unreachable — filtered above
            }
        }
        if !decodeLanes.isEmpty {
            return .decodeRound(decodeLanes.map(\.requestID))
        }
        return .idle
    }
}
