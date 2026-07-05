import Foundation

/// What a submission needs loaded — the arbiter's `.llm` demand vocabulary.
nonisolated struct BatchModelDemand: Sendable, Equatable {
    let modelIDOverride: String?
    let vision: LLMVisionRequirement
}

/// One completion submitted to the **Batch Engine** (PRD #173, ADR-0022).
nonisolated struct BatchSubmission: Sendable {
    let requestID: UUID
    let demand: BatchModelDemand
    /// Image-bearing requests run solo — exactly today's semantics — until
    /// the overlapping-image stress run and the parked MLX buffer-lifetime
    /// investigation both clear (ADR-0022).
    let bearsImages: Bool
    /// The completion's generation runs the monolithic single-flight path
    /// (managed arm, standard route) rather than engine-stepped lanes; it
    /// occupies the pool exclusively, like an image request.
    let runsMonolithic: Bool
    /// Longest radix-tree prefix match probed at submit; 0 when no probe was
    /// possible. Orders the waiting queue (SGLang cache-aware), aged to FIFO.
    let matchedPrefixLength: Int
    /// The lane's solo prefill chunk size (the request's `prefillStepSize`).
    let preferredPrefillChunk: Int
    /// The admission deadline — today's 60 s busy contract, preserved
    /// verbatim as a Lane Admission timeout (503 + `Retry-After` above).
    /// `nil` waits indefinitely — the internal Agent Run's contract, which
    /// has never had an acquisition timeout.
    let admissionTimeout: Duration?

    init(
        requestID: UUID,
        demand: BatchModelDemand,
        bearsImages: Bool,
        runsMonolithic: Bool,
        matchedPrefixLength: Int = 0,
        preferredPrefillChunk: Int = 1_024,
        admissionTimeout: Duration? = .seconds(60)
    ) {
        self.requestID = requestID
        self.demand = demand
        self.bearsImages = bearsImages
        self.runsMonolithic = runsMonolithic
        self.matchedPrefixLength = matchedPrefixLength
        self.preferredPrefillChunk = preferredPrefillChunk
        self.admissionTimeout = admissionTimeout
    }
}

/// What `submit` returns once the request became a **Lane**.
nonisolated struct BatchLaneAdmission: Sendable, Equatable {
    let laneID: UUID
    /// The lane runs alone (image-bearing / monolithic path); its generation
    /// may use today's single-flight code unchanged.
    let isExclusive: Bool
    /// Queue wait, for the lane lifecycle diagnostics.
    let queueWaitSeconds: TimeInterval
}

/// The step kinds a pool lane's drive requests from the engine.
nonisolated enum BatchStepKind: Sendable, Equatable {
    case startup
    case prefillChunk
    case decode
    case finish
}

/// What a granted step may do — for a prefill chunk, how many tokens.
nonisolated struct BatchStepGrant: Sendable, Equatable {
    let prefillChunkTokens: Int
}

/// The machine-headroom inputs for the Lane Admission arithmetic, sampled on
/// MainActor by the injected closure.
nonisolated struct BatchLaneBudget: Sendable, Equatable {
    let headroomBytes: Int
    let evictableCacheBytes: Int
    let perLaneBytes: Int
}

nonisolated enum BatchEngineError: Error {
    /// A lane requested a second step while one was already pending — a
    /// drive-loop bug, surfaced loudly.
    case overlappingStep
}

/// How the engine acquires the GPU lease: the arbiter's `withExclusiveGPU`
/// behind a closure so the engine depends on functions, not the `@MainActor`
/// protocol. The engine passes a `@Sendable` body that runs its grant loop.
typealias BatchLeaseRunner =
    @Sendable @MainActor (
        _ modelIDOverride: String?,
        _ vision: LLMVisionRequirement,
        _ body: @Sendable @escaping () async -> Void
    ) async throws -> Void

/// The **Batch Engine** (ADR-0022): owns the GPU lease whenever any lane is
/// live; completions submit instead of acquiring. Model-agnostic — lanes'
/// drive tasks bring their own step bodies (which enter the Model Session
/// themselves); the engine decides *when* each body runs, via the pure
/// `BatchEnginePolicy`, and executes granted bodies one at a time so all
/// Metal work stays serialized. Boundary Yield releases the lease to a
/// waiting slot-preserving consumer between steps; Admission Freeze drains
/// the pool for model switches and image-bearing requests.
actor BatchEngine {

    // MARK: - Dependencies

    private let leaseRunner: BatchLeaseRunner
    private let leaseWaiters: LeaseWaiterSignal
    /// Does the currently loaded model state satisfy a demand? Evaluated on
    /// MainActor against live settings + `loadedLLMState` at enqueue and
    /// after every (re)acquisition.
    private let demandSatisfied: @Sendable @MainActor (BatchModelDemand) -> Bool
    /// The Lane Admission arithmetic's inputs (ADR-0018 reserve pricing).
    private let laneBudget: @Sendable @MainActor () -> BatchLaneBudget
    private let hardLaneCap: Int

    // MARK: - State

    private struct QueuedEntry {
        let submission: BatchSubmission
        let arrivalOrder: Int
        let enqueuedAt: ContinuousClock.Instant
        var demandIsSatisfied: Bool
        let continuation: CheckedContinuation<BatchLaneAdmission, any Error>
        let timeoutTask: Task<Void, Never>?
    }

    private struct PendingStep {
        let id: UUID
        let kind: BatchStepKind
        let run: @Sendable (BatchStepGrant) async -> Void
        let cancel: @Sendable () -> Void
    }

    private struct LaneRecord {
        let submission: BatchSubmission
        let admissionOrder: Int
        var phase: BatchEngineSnapshot.LanePhase
        var pendingStep: PendingStep?
        let isExclusive: Bool
    }

    private var queue: [QueuedEntry] = []
    private var lanes: [UUID: LaneRecord] = [:]
    private var arrivalCounter = 0
    private var admissionCounter = 0
    private var lastGrantWasPrefillClass = false
    private var cachedLaneCapacity = 1
    /// The demand the pool is currently running under — re-acquisitions after
    /// a Boundary Yield must use it, never the queue head's (a reload under
    /// live lanes is the one forbidden move).
    private var poolDemand: BatchModelDemand?
    private var driverTask: Task<Void, Never>?
    private var parked: [CheckedContinuation<Void, Never>] = []
    private let clock = ContinuousClock()

    init(
        leaseRunner: @escaping BatchLeaseRunner,
        leaseWaiters: LeaseWaiterSignal,
        demandSatisfied: @escaping @Sendable @MainActor (BatchModelDemand) -> Bool,
        laneBudget: @escaping @Sendable @MainActor () -> BatchLaneBudget,
        hardLaneCap: Int = BatchEnginePolicy.hardLaneCap
    ) {
        self.leaseRunner = leaseRunner
        self.leaseWaiters = leaseWaiters
        self.demandSatisfied = demandSatisfied
        self.laneBudget = laneBudget
        self.hardLaneCap = hardLaneCap
    }

    // MARK: - Submission

    /// True when no lane is live and nothing waits — the Speculative
    /// Canonical Prefill's idle predicate (ADR-0009 stays idle-only).
    var isIdle: Bool { lanes.isEmpty && queue.isEmpty }

    /// Submit a completion; suspends until it becomes a lane. Throws
    /// `LeaseTimeoutError` when the admission deadline passes (the caller
    /// maps it to today's 503 + `Retry-After`), `CancellationError` when the
    /// submitting task is cancelled while queued.
    func submit(_ submission: BatchSubmission) async throws -> BatchLaneAdmission {
        let id = submission.requestID
        let satisfied = await demandSatisfied(submission.demand)
        await refreshLaneCapacity()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                if Task.isCancelled {
                    continuation.resume(throwing: CancellationError())
                    return
                }
                let timeoutTask = submission.admissionTimeout.map { timeout in
                    Task { [weak self] in
                        try? await Task.sleep(for: timeout)
                        guard !Task.isCancelled else { return }
                        await self?.timeoutEntry(id)
                    }
                }
                queue.append(
                    QueuedEntry(
                        submission: submission,
                        arrivalOrder: arrivalCounter,
                        enqueuedAt: clock.now,
                        demandIsSatisfied: satisfied,
                        continuation: continuation,
                        timeoutTask: timeoutTask
                    ))
                arrivalCounter += 1
                wake()
            }
        } onCancel: {
            Task { await self.cancelEntry(id) }
        }
    }

    /// A lane's drive requests one engine step; the engine runs `body` when
    /// the scheduling core grants it. The body performs its own Model Session
    /// entry; the engine only sequences. Throws `CancellationError` when the
    /// lane was torn down or the requesting task cancelled while pending.
    func step<R: Sendable>(
        lane laneID: UUID,
        kind: BatchStepKind,
        body: @escaping @Sendable (BatchStepGrant) async throws -> R
    ) async throws -> R {
        let stepID = UUID()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                if Task.isCancelled {
                    continuation.resume(throwing: CancellationError())
                    return
                }
                guard lanes[laneID] != nil else {
                    continuation.resume(throwing: CancellationError())
                    return
                }
                guard lanes[laneID]?.pendingStep == nil else {
                    continuation.resume(throwing: BatchEngineError.overlappingStep)
                    return
                }
                lanes[laneID]?.pendingStep = PendingStep(
                    id: stepID,
                    kind: kind,
                    run: { grant in
                        do {
                            continuation.resume(returning: try await body(grant))
                        } catch {
                            continuation.resume(throwing: error)
                        }
                    },
                    cancel: {
                        continuation.resume(throwing: CancellationError())
                    }
                )
                lanes[laneID]?.phase = Self.phase(for: kind)
                wake()
            }
        } onCancel: {
            Task { await self.cancelPendingStep(lane: laneID, step: stepID) }
        }
    }

    /// Lane teardown: the drive calls this on every exit path. Pending step
    /// (if any) is cancelled, capacity returns to the pool immediately
    /// (user story 16). Idempotent.
    func laneFinished(_ laneID: UUID) async {
        guard let lane = lanes.removeValue(forKey: laneID) else { return }
        lane.pendingStep?.cancel()
        await refreshLaneCapacity()
        wake()
    }

    // MARK: - Driver

    private func wake() {
        let waiters = parked
        parked = []
        for waiter in waiters { waiter.resume() }
        if driverTask == nil, !queue.isEmpty || !lanes.isEmpty {
            driverTask = Task { await self.drive() }
        }
    }

    private func parkUntilWork() async {
        await withCheckedContinuation { parked.append($0) }
    }

    private var hasWork: Bool { !queue.isEmpty || !lanes.isEmpty }

    private func drive() async {
        while hasWork {
            guard let target = acquisitionTarget() else { break }
            do {
                try await leaseRunner(target.modelIDOverride, target.vision) {
                    await self.runLeaseSession(target: target)
                }
            } catch {
                failTarget(target, error: error)
            }
        }
        driverTask = nil
        // A submission that raced the loop exit restarts the driver.
        if hasWork { wake() }
    }

    /// The demand for the next acquisition: live lanes pin the pool's demand
    /// (never reload under them); an empty pool follows the ordered head.
    private func acquisitionTarget() -> BatchModelDemand? {
        if !lanes.isEmpty, let poolDemand { return poolDemand }
        let ordered = BatchEnginePolicy.orderedQueue(snapshotQueue())
        guard let head = ordered.first,
            let entry = queue.first(where: { $0.submission.requestID == head.requestID })
        else { return poolDemand }
        return entry.submission.demand
    }

    /// One lease tenure: refresh compatibility against the (possibly just
    /// reloaded) model state, then grant steps until the pool drains, a
    /// waiter needs the lease, or the head needs a different model.
    private func runLeaseSession(target: BatchModelDemand) async {
        poolDemand = target
        for index in queue.indices {
            queue[index].demandIsSatisfied =
                await demandSatisfied(queue[index].submission.demand)
        }
        await refreshLaneCapacity()

        // A session that has run lanes releases the lease when the pool
        // fully drains — a natural tenure boundary, so a queued consumer the
        // engine never yields to (`reloadLLMIfNeeded`, a direct lease user)
        // gets its FIFO turn between batches instead of starving behind
        // back-to-back completions. A session resumed under paused lanes
        // owes that release from the start.
        var owesDrainRelease = !lanes.isEmpty

        while true {
            if lanes.isEmpty {
                if queue.isEmpty { return }
                if owesDrainRelease { return }
            }
            let decision = BatchEnginePolicy.decide(makeSnapshot())
            switch decision {
            case .yieldLease:
                return
            case .switchPool:
                // Head needs a reload; pool is empty by construction. Exit
                // the session — the driver re-acquires with the head's
                // demand and the arbiter reloads under that lease.
                return
            case .admit(let id):
                admitEntry(id, exclusive: false)
                owesDrainRelease = true
            case .admitExclusive(let id):
                admitEntry(id, exclusive: true)
                owesDrainRelease = true
            case .grant(let id, let grant):
                let tokens: Int =
                    if case .prefillChunk(let t) = grant { t } else { 0 }
                await executeGrant(
                    id,
                    grant: BatchStepGrant(prefillChunkTokens: tokens),
                    prefillClass: true
                )
            case .decodeRound(let ids):
                for id in ids {
                    await executeGrant(
                        id,
                        grant: BatchStepGrant(prefillChunkTokens: 0),
                        prefillClass: false
                    )
                }
            case .idle:
                await parkUntilWork()
            }
        }
    }

    private func executeGrant(
        _ laneID: UUID, grant: BatchStepGrant, prefillClass: Bool
    ) async {
        guard let pending = lanes[laneID]?.pendingStep else { return }
        lanes[laneID]?.pendingStep = nil
        lastGrantWasPrefillClass = prefillClass
        await pending.run(grant)
    }

    // MARK: - Admission bookkeeping

    private func admitEntry(_ requestID: UUID, exclusive: Bool) {
        guard let index = queue.firstIndex(where: { $0.submission.requestID == requestID })
        else { return }
        let entry = queue.remove(at: index)
        entry.timeoutTask?.cancel()
        let waited = entry.enqueuedAt.duration(to: clock.now)
        lanes[requestID] = LaneRecord(
            submission: entry.submission,
            admissionOrder: admissionCounter,
            phase: .starting,
            pendingStep: nil,
            isExclusive: exclusive
        )
        admissionCounter += 1
        entry.continuation.resume(
            returning: BatchLaneAdmission(
                laneID: requestID,
                isExclusive: exclusive,
                queueWaitSeconds: TimeInterval(waited.components.seconds)
                    + TimeInterval(waited.components.attoseconds) / 1e18
            ))
    }

    private func timeoutEntry(_ requestID: UUID) {
        guard let index = queue.firstIndex(where: { $0.submission.requestID == requestID })
        else { return }
        let entry = queue.remove(at: index)
        entry.continuation.resume(throwing: LeaseTimeoutError())
    }

    private func cancelEntry(_ requestID: UUID) {
        guard let index = queue.firstIndex(where: { $0.submission.requestID == requestID })
        else { return }
        let entry = queue.remove(at: index)
        entry.timeoutTask?.cancel()
        entry.continuation.resume(throwing: CancellationError())
    }

    /// Lease acquisition for `target` failed (`ensureLoaded` threw — model
    /// deleted while queued, load error): fail the entries that demanded it;
    /// others retry on the next driver iteration. Live lanes cannot exist
    /// here in the normal path (acquisition for a fresh target only happens
    /// with an empty pool), but a re-acquisition failure after a yield tears
    /// the paused lanes down rather than stranding them.
    private func failTarget(_ target: BatchModelDemand, error: any Error) {
        let matching = queue.enumerated().filter {
            $0.element.submission.demand == target
        }
        for (_, entry) in matching.reversed() {
            entry.timeoutTask?.cancel()
            entry.continuation.resume(throwing: error)
        }
        queue.removeAll { $0.submission.demand == target }
        for (laneID, lane) in lanes {
            lane.pendingStep?.cancel()
            lanes[laneID]?.pendingStep = nil
        }
    }

    private func cancelPendingStep(lane laneID: UUID, step stepID: UUID) {
        guard let pending = lanes[laneID]?.pendingStep, pending.id == stepID
        else { return }
        lanes[laneID]?.pendingStep = nil
        pending.cancel()
        wake()
    }

    private func refreshLaneCapacity() async {
        let budget = await laneBudget()
        cachedLaneCapacity = BatchEnginePolicy.admittableLanes(
            headroomBytes: budget.headroomBytes,
            evictableCacheBytes: budget.evictableCacheBytes,
            perLaneBytes: budget.perLaneBytes,
            hardCap: hardLaneCap
        )
    }

    // MARK: - Snapshot

    private func snapshotQueue() -> [BatchEngineSnapshot.QueueEntry] {
        queue.map { entry in
            let waited = entry.enqueuedAt.duration(to: clock.now)
            return BatchEngineSnapshot.QueueEntry(
                requestID: entry.submission.requestID,
                arrivalOrder: entry.arrivalOrder,
                waitedSeconds: TimeInterval(waited.components.seconds)
                    + TimeInterval(waited.components.attoseconds) / 1e18,
                matchedPrefixLength: entry.submission.matchedPrefixLength,
                requiresExclusivePool: entry.submission.bearsImages
                    || entry.submission.runsMonolithic,
                demandSatisfiedByLoadedModel: entry.demandIsSatisfied
            )
        }
    }

    private func makeSnapshot() -> BatchEngineSnapshot {
        BatchEngineSnapshot(
            queue: snapshotQueue(),
            lanes: lanes.values
                .sorted { $0.admissionOrder < $1.admissionOrder }
                .map { lane in
                    BatchEngineSnapshot.Lane(
                        requestID: lane.submission.requestID,
                        admissionOrder: lane.admissionOrder,
                        phase: lane.phase,
                        hasPendingStep: lane.pendingStep != nil,
                        preferredPrefillChunk: lane.submission.preferredPrefillChunk,
                        isExclusive: lane.isExclusive
                    )
                },
            laneCapacity: cachedLaneCapacity,
            leaseHasWaiters: leaseWaiters.hasWaiters,
            lastGrantWasPrefillClass: lastGrantWasPrefillClass
        )
    }

    private static func phase(for kind: BatchStepKind) -> BatchEngineSnapshot.LanePhase {
        switch kind {
        case .startup: .starting
        case .prefillChunk: .prefilling
        case .decode: .decoding
        case .finish: .finishing
        }
    }
}
