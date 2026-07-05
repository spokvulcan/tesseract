//
//  SSDSnapshotStore.swift
//  tesseract
//
//  Writer-side and read-side façade for the SSD prefix-cache tier.
//  Owns the front-door admission *queue* (under its own queue lock),
//  the detached writer task that drains it, the `.safetensors` body
//  I/O (`writePayload`, `loadSync`, the Metal-affine decode), and the
//  *effects* of the ledger's decisions (file deletes, `onCommit` /
//  `onDrop`, diagnostics). It composes a **Snapshot Ledger**, which
//  owns the in-memory residency authority — the manifest, the byte
//  budget, recency, the type-protected LRU cut, the in-flight-delete
//  tombstones, and `manifest.json` / `_meta.json` durability — under a
//  separate ledger lock. See `SnapshotLedger.swift` and `CONTEXT.md`
//  ("SSD snapshot ledger") for the split-along-the-lock rationale.
//
//  **Deliberately not a Swift `actor`.** The front door must be
//  callable from a synchronous MainActor closure (the LLMActor store
//  call sites), so we use a `nonisolated final class` with an
//  `NSLock` for cross-thread safety. Converting this to a Swift
//  actor would force every caller into `await` and reintroduce a
//  memory-safety bug where a burst of mid-prefill captures can
//  accumulate GiBs of pending bytes outside the queue cap before
//  back-pressure applies.
//
//  **Callback-shaped node coupling.** The writer notifies callers
//  about commit / drop events via `@Sendable` closures set at init
//  time, keeping the store decoupled from `RadixTreeNode` and
//  `PrefixCacheManager`. Downstream composition code (tiered store)
//  wires the closures to the real radix-tree lifecycle handlers;
//  tests wire them to in-process trackers.
//
//  **Placeholder on-disk format.** The writer serializes a simple
//  length-prefixed binary container (8-byte LE header length + JSON
//  header + concatenated tensor blobs) via the neutral
//  `encodePlaceholderContainer` (in `PlaceholderContainer.swift`,
//  shared with the ledger's rebuild). The real safetensors format
//  comes in a downstream task that ships `HybridCacheSnapshot`
//  `serialize(to:metadata:)` / `deserialize(from:)` wrappers; at
//  that point this writer switches over. The header JSON carries a
//  `"format_kind": "tesseract-cache-v1"` marker so the downstream
//  reader can refuse placeholder files cleanly.
//

import Foundation
import MLX
import MLXLMCommon

// MARK: - SSD decode errors

/// Errors thrown by the store's Metal-affine `decodePlaceholderContainer`
/// while reconstructing tensor blobs from a placeholder container. The
/// header-parse errors are separate — they belong to the MLX-free codec
/// (`PlaceholderContainerError` in `PlaceholderContainer.swift`). Both
/// error types funnel into the same recoverable `.decodeFailed` miss in
/// `loadSync`, which drops the descriptor and the on-disk file.
nonisolated enum SSDLoadError: Error {
    /// A header `byte_offset` / `byte_size` pair points outside the
    /// container's blob section — truncated, corrupt, or hostile file.
    case truncatedBlob
    /// A header array's `dtype` string has no MLX dtype mapping.
    case unknownDType(String)
    /// A **Segment Chain** failed composition validation: layer counts,
    /// suffix contiguity, slot counts, dtypes, or token-axis extents
    /// disagree across segments. The whole chain is condemned.
    case segmentMismatch(String)
}

// MARK: - Front door outcome types

/// Result of attempting to enqueue a payload into the SSD writer's
/// pending queue. Synchronous so the MainActor call sites can branch
/// without `await`.
nonisolated enum TryEnqueueResult: Sendable, Equatable {
    /// Item accepted into the pending queue. The returned `SnapshotRef`
    /// is the snapshot's on-disk identity; the composition layer attaches
    /// it to the radix node as a pending-write `SnapshotState`.
    case accepted(SnapshotRef)

    /// Single payload exceeds the front door's `maxPendingBytes`
    /// cap. Never enqueued. Expected to be vanishingly rare — the
    /// cap is sized at `min(4 GiB, physicalMemory / 16)`, and
    /// realistic mid-prefill payloads are 50–600 MiB.
    case rejectedTooLargeForBudget

    /// Descriptor's `checkpointType` string did not round-trip
    /// through the wire-format extension. Descriptors come from
    /// our own call sites, so this is a programming error — we
    /// surface it rather than silently corrupt the manifest.
    case rejectedInvalidCheckpointType

    /// Descriptor's `partitionDigest` has no matching entry in
    /// `manifest.partitions`. The `SnapshotManifest` invariant
    /// requires every snapshot entry to reference a registered
    /// partition; the warm-start path drops dangling entries as
    /// defensive repair. Callers must invoke
    /// `registerPartition(_:digest:)` for each distinct partition
    /// before enqueuing any snapshot with that digest.
    case rejectedUnregisteredPartition

    /// The payload is a **Leaf Extension Admission** but its base is
    /// neither resident, queued, nor in the writer's hands — the suffix
    /// can never compose. The caller treats the leaf as RAM-only; the
    /// next turn self-heals with a full write.
    case rejectedExtensionBaseUnavailable
}

/// Why a pending or resident item was dropped. Fed into the
/// `onDrop` callback so downstream composition code can update the
/// radix node's Snapshot Ref state and log diagnostics.
nonisolated enum SSDDropReason: Sendable, Equatable {
    /// Front door dropped an older pending entry to fit the new
    /// enqueue under `maxPendingBytes`.
    case backpressureOldest

    /// Previously committed resident was evicted by a later
    /// admission's type-protected LRU cut (or by a disk-full
    /// retry) to make room for an incoming entry. The file and
    /// the manifest entry are already gone by the time this fires.
    /// The router (`TieredSnapshotStore`) clears the node's committed
    /// Snapshot Ref eagerly — eviction scoring and **Snapshot
    /// Demotion** read `ref != nil` as "backed", so a stale ref must
    /// not outlive its backing.
    case evictedByLRU

    /// Admission LRU cut could not free enough SSD budget without
    /// violating `.system` type protection; incoming was non-system.
    case systemProtectionWins

    /// Incoming descriptor's `bytes` is larger than the total SSD
    /// budget — no amount of eviction can create room. Distinct
    /// from `.diskFull` because it's a configuration problem
    /// (`budgetBytes` is under-sized for this payload), not a
    /// filesystem fullness condition.
    case exceedsBudget

    /// Writer hit `ENOSPC` / `EDQUOT` and could not recover after a
    /// single eviction-retry pass.
    case diskFull

    /// Writer hit a non-space I/O error (permissions, fsync
    /// failure, serialization error, other Foundation error).
    case writerIOError

    /// A pending **Leaf Extension Admission** lost its base before the
    /// commit-time fold (the base's own write dropped, or a hydration
    /// failure removed it) — the suffix can never compose, so the item
    /// is dropped and its file (if written) deleted.
    case extensionBaseLost

    /// `loadSync` could not materialize a resident back into RAM
    /// (file missing, fingerprint mismatch, placeholder decode
    /// error). The descriptor was removed from the manifest and the
    /// file deleted before the callback fired.
    case hydrationFailure
}

/// Payload of the writer's commit callback. Beyond the committed
/// snapshot's ID it carries the two facts only the writer's durable
/// commit can produce, so the tree-side router never has to guess:
/// - `consumedBaseID`: the base entry a **Leaf Extension Admission**'s
///   fold consumed (`nil` for full writes). The router discards the
///   base's now-stale tree ref *here* — not at admission — so a writer
///   drop leaves the base reachable (the transfer degrades to
///   `preserved`).
/// - `chainBytesOnDisk`: the committed entry's whole-chain byte count
///   (`PersistedSnapshotDescriptor.totalBytes`), refreshed onto the
///   live ref so in-session telemetry matches what a warm start would
///   restore.
nonisolated struct SSDCommitInfo: Sendable, Equatable {
    let snapshotID: String
    let consumedBaseID: String?
    let chainBytesOnDisk: Int
}

// MARK: - SSDSnapshotStore

// Evolving MVP mid-refactor (see CLAUDE.md); structural limit kept lenient — splitting deferred.
// swiftlint:disable:next type_body_length
nonisolated final class SSDSnapshotStore: @unchecked Sendable, SnapshotHydrating {

    // MARK: - Configuration (immutable after init)

    let rootURL: URL
    let budgetBytes: Int
    let maxPendingBytes: Int

    // MARK: - Snapshot Ledger (residency + manifest durability authority)

    /// The in-memory authority over the SSD-resident set: its byte
    /// budget, recency, the type-protected LRU cut, the in-flight-delete
    /// tombstones, and `manifest.json`/`_meta.json` durability. The store
    /// composes it in `init` and performs the *effects* of its decisions
    /// (file deletes, `onCommit`/`onDrop`, diagnostics) outside any lock.
    /// See `CONTEXT.md` ("SSD snapshot ledger").
    private let ledger: SnapshotLedger

    // MARK: - Queue-lock-protected mutable state

    /// Guards the writer's pending queue only. Distinct from the
    /// ledger's lock — the two never nest because the writer is
    /// single-threaded and releases each lock between steps.
    private let queueLock = NSLock()
    private var pending: [PendingWrite] = []
    private var pendingBytes: Int = 0
    /// The snapshot ID the writer is currently processing — popped from
    /// `pending` but not yet committed or dropped. An extension whose
    /// base is in the writer's hands is still valid (FIFO settles the
    /// base first), so the front-door base check must see this window.
    private var inFlightSnapshotID: String?
    /// Continuations waiting for the writer to finish its current
    /// `drainPending` iteration. `flushAsync()` registers one per
    /// call and the writer resumes them all after each drain cycle
    /// so unload paths can observe pending writes as durable.
    private var drainWaiters: [CheckedContinuation<Void, Never>] = []

    // MARK: - Writer wakeup + task

    private let wakeupStream: AsyncStream<Void>
    private let wakeupContinuation: AsyncStream<Void>.Continuation
    private var writerTask: Task<Void, Never>?

    // MARK: - Callbacks (immutable after init)

    private let onCommit: @Sendable (SSDCommitInfo) -> Void
    private let onDrop: @Sendable (String, SSDDropReason) -> Void
    private let writerDrainPreludeForTesting: (@Sendable () async -> Void)?

    // MARK: - Deferred-class write scheduling (PRD #150)

    /// Shared busy signal with the inference path. While a hydration
    /// read or prefill is in flight, deferrable pending items wait —
    /// bounded by `maxDeferredHoldup`. `nil` (tests, replay caches)
    /// means no gating: every item processes immediately.
    private let activityGate: StorageActivityGate?

    /// Longest a deferrable item waits out a busy gate before it is
    /// written anyway. Liveness bound, not policy: the PRD wants
    /// overlap *avoided when possible*, never writes that starve.
    static let maxDeferredHoldup: Duration = .seconds(30)

    /// Delay before the writer re-checks a gate-blocked deferred item.
    static let deferredRecheckInterval: Duration = .milliseconds(500)

    /// Set for the duration of `flushAsync`: unload/benchmark drains
    /// must write everything out regardless of gate state.
    private var forceDrainDeferred = false
    /// Guards against stacking re-wake tasks while a deferred item
    /// waits out the gate.
    private var deferredRewakeScheduled = false

    // MARK: - Public API

    init(
        config: SSDPrefixCacheConfig,
        manifestDebounce: Duration = .milliseconds(500),
        activityGate: StorageActivityGate? = nil,
        onCommit: @escaping @Sendable (SSDCommitInfo) -> Void = { _ in },
        onDrop: @escaping @Sendable (String, SSDDropReason) -> Void = { _, _ in },
        writerDrainPreludeForTesting: (@Sendable () async -> Void)? = nil
    ) {
        self.rootURL = config.rootURL
        self.budgetBytes = config.budgetBytes
        self.maxPendingBytes = config.maxPendingBytes
        self.activityGate = activityGate
        self.ledger = SnapshotLedger(
            rootURL: config.rootURL,
            budgetBytes: config.budgetBytes,
            manifestDebounce: manifestDebounce,
            budgetCapBytes: config.budgetCapBytes,
            // Dynamic SSD budget (ADR-0018): production configs measure
            // free disk; test/replay configs stay on the static bootstrap.
            freeDiskBytesProvider: config.measuresFreeDisk
                ? { SSDBudgetPolicy.measuredFreeDiskBytes(rootURL: $0) }
                : nil
        )
        self.onCommit = onCommit
        self.onDrop = onDrop
        self.writerDrainPreludeForTesting = writerDrainPreludeForTesting

        let (stream, continuation) = AsyncStream<Void>.makeStream(
            bufferingPolicy: .bufferingNewest(1))
        self.wakeupStream = stream
        self.wakeupContinuation = continuation
        self.writerTask = nil

        Log.agent.info(
            "SSDSnapshotStore init root=\(config.rootURL.path) "
                + "budgetBytes=\(config.budgetBytes) "
                + "maxPendingBytes=\(config.maxPendingBytes)"
        )

        // Task is started after self is fully initialized; it
        // captures self weakly so the class can deinit while the
        // wakeup stream is idle.
        self.writerTask = Task.detached { [weak self] in
            await self?.writerLoop()
        }
    }

    deinit {
        wakeupContinuation.finish()
        writerTask?.cancel()
    }

    /// The **Survival Gate** pre-check, forwarded to the **Snapshot
    /// Ledger** (which owns the cut this simulates). Pure read — see
    /// `SnapshotLedger.survivesAdmissionCut`. `nonisolated` like
    /// `tryEnqueue`, so the MainActor front doors call it without a
    /// hop.
    func survivesAdmissionCut(
        tokenOffset: Int,
        totalBytes: Int,
        checkpointType: HybridCacheSnapshot.CheckpointType,
        lastAccessAt: TimeInterval,
        scoring config: EvictionConfiguration
    ) -> Bool {
        ledger.survivesAdmissionCut(
            tokenOffset: tokenOffset,
            totalBytes: totalBytes,
            checkpointType: checkpointType,
            lastAccessAt: lastAccessAt,
            scoring: config
        )
    }

    /// Synchronous front-door admission. Acquires the lock, enforces
    /// the `maxPendingBytes` byte budget via drop-oldest-pending,
    /// pushes the item onto the writer's pending queue, and wakes
    /// the writer. Never suspends and never crosses an actor
    /// boundary — safe to call from inside a MainActor closure.
    ///
    /// `refreshRecencyAtCommit: false` marks a **Snapshot Demotion**:
    /// the writer's commit preserves the descriptor's own (stale)
    /// `lastAccessAt` instead of re-stamping it.
    ///
    /// `scoringConfig` rides the queue to the ledger's admission cut —
    /// see `PendingWrite.scoringConfig`. The default (α = 0) keeps the
    /// cut at plain type-protected LRU.
    ///
    /// `mandatory: true` marks a guarantee-class write — the end-of-turn
    /// leaf whose SSD copy the **Leaf Home Guarantee** (ADR-0019)
    /// promises. It is exempt from the `maxPendingBytes` size rejection
    /// and is never a back-pressure victim; the pending total may
    /// transiently exceed the cap by at most the guarantee payloads in
    /// flight. Its remaining rejection paths are hard errors, logged at
    /// error level with `mandatory=true` on the `ssdAdmit` event.
    /// `deferrable: true` marks a deferred-class write (a hit-count
    /// promotion, PRD #150): the writer holds it while the
    /// `StorageActivityGate` reports hydration/prefill in flight,
    /// bounded by `maxDeferredHoldup`. Mutually exclusive with
    /// `mandatory` by construction — the guarantee write is never
    /// deferred (ADR-0019).
    func tryEnqueue(
        payload: SnapshotPayload,
        descriptor: PersistedSnapshotDescriptor,
        refreshRecencyAtCommit: Bool = true,
        scoringConfig: EvictionConfiguration = EvictionConfiguration(),
        condemnedResidentIDs: Set<String> = [],
        mandatory: Bool = false,
        deferrable: Bool = false
    ) -> TryEnqueueResult {
        // Parse the wire-format checkpoint type before taking the lock;
        // no sense holding the lock for a parse that can fail.
        guard
            let checkpointType = HybridCacheSnapshot.CheckpointType(
                wireString: descriptor.checkpointType
            )
        else {
            PrefixCacheDiagnostics.logSystem(
                PrefixCacheDiagnostics.SSDAdmitEvent(
                    id: descriptor.snapshotID,
                    bytes: descriptor.bytes,
                    outcome: .droppedInvalidCheckpointType,
                    mandatory: mandatory
                ),
                level: mandatory ? .error : .info
            )
            return .rejectedInvalidCheckpointType
        }

        let payloadBytes = payload.totalBytes

        // A single payload larger than the cap cannot be queued at all
        // for an opportunistic write; no amount of back-pressure
        // eviction can create room. A guarantee-class write is exempt
        // (ADR-0019): bouncing the end-of-turn leaf off a RAM-sized
        // pending cap was defect 3 of the 2026-07-04 incident — it
        // enqueues regardless, and the writer drains the transient
        // overshoot.
        if payloadBytes > maxPendingBytes {
            guard mandatory else {
                PrefixCacheDiagnostics.logSystem(
                    PrefixCacheDiagnostics.SSDAdmitEvent(
                        id: descriptor.snapshotID,
                        bytes: payloadBytes,
                        outcome: .droppedTooLargeForBudget
                    ))
                return .rejectedTooLargeForBudget
            }
            Log.agent.warning(
                "SSD guarantee-class write \(descriptor.snapshotID.prefix(8)) exceeds "
                    + "maxPendingBytes (\(payloadBytes) > \(maxPendingBytes)) — "
                    + "enqueuing anyway (Leaf Home Guarantee, ADR-0019)"
            )
        }

        // Enforce the manifest invariant that every snapshots entry
        // must reference a registered partition. The writer cannot
        // autoregister because `PartitionMeta` carries load-bearing
        // fields (modelFingerprint, kvBits, etc.) that only the caller
        // knows at model-load time. Reject here so the caller gets
        // immediate feedback rather than persisting a dangling manifest
        // entry. The ledger check runs before the queue lock — the two
        // locks never nest.
        guard ledger.hasPartition(digest: descriptor.partitionDigest) else {
            PrefixCacheDiagnostics.logSystem(
                PrefixCacheDiagnostics.SSDAdmitEvent(
                    id: descriptor.snapshotID,
                    bytes: payloadBytes,
                    outcome: .droppedUnregisteredPartition,
                    mandatory: mandatory
                ),
                level: mandatory ? .error : .info
            )
            return .rejectedUnregisteredPartition
        }

        // Leaf extension: the base must still be reachable — resident
        // in the manifest, queued, or in the writer's hands (FIFO
        // settles it before this item). `beginExtensionTransfer`
        // shields a resident base from the LRU cut for the pending
        // window; every terminal writer path releases the shield. The
        // queue-lock check runs first and releases before the ledger
        // call — the two locks never nest.
        let extendingBaseID = payload.extending?.baseSnapshotID
        if let extendingBaseID {
            queueLock.lock()
            let baseIsQueuedOrInFlight =
                inFlightSnapshotID == extendingBaseID
                || pending.contains { $0.descriptor.snapshotID == extendingBaseID }
            // An extension's fold needs its base committed first (FIFO).
            // A gate-held deferrable base would let this extension be
            // popped past it and drop with `extensionBaseLost` — so the
            // extension's arrival promotes the queued base to immediate.
            if let baseIndex = pending.firstIndex(where: {
                $0.descriptor.snapshotID == extendingBaseID && $0.deferrable
            }) {
                pending[baseIndex].deferrable = false
            }
            queueLock.unlock()
            guard
                ledger.beginExtensionTransfer(
                    baseID: extendingBaseID,
                    baseIsQueuedOrInFlight: baseIsQueuedOrInFlight
                )
            else {
                PrefixCacheDiagnostics.logSystem(
                    PrefixCacheDiagnostics.SSDAdmitEvent(
                        id: descriptor.snapshotID,
                        bytes: payloadBytes,
                        outcome: .droppedExtensionBaseUnavailable
                    ))
                return .rejectedExtensionBaseUnavailable
            }
        }

        enqueueApplyingBackPressure(
            PendingWrite(
                payload: payload,
                descriptor: descriptor,
                extendingBaseID: extendingBaseID,
                refreshRecencyAtCommit: refreshRecencyAtCommit,
                scoringConfig: scoringConfig,
                condemnedResidentIDs: condemnedResidentIDs,
                mandatory: mandatory,
                deferrable: deferrable && !mandatory,
                enqueuedAt: .now
            ))

        wakeupContinuation.yield()

        return .accepted(
            SnapshotRef(
                snapshotID: descriptor.snapshotID,
                partitionDigest: descriptor.partitionDigest,
                tokenOffset: descriptor.tokenOffset,
                checkpointType: checkpointType,
                bytesOnDisk: descriptor.bytes
            )
        )
    }

    /// Append `item` to the pending queue, first dropping the oldest
    /// NON-mandatory pending entries until it fits under the cap.
    /// Guarantee-class items are never back-pressure victims
    /// (ADR-0019) — when only they remain the queue transiently
    /// overshoots the cap instead, bounded by the guarantee payloads in
    /// flight. Dropped IDs are recorded under the lock and their
    /// callbacks fire AFTER releasing it, so a callback body can take
    /// other locks without risking a deadlock.
    private func enqueueApplyingBackPressure(_ item: PendingWrite) {
        var droppedItems: [(id: String, bytes: Int, extendingBaseID: String?)] = []
        let payloadBytes = item.payload.totalBytes

        queueLock.lock()

        // Two victim passes: deferrable items first — a dropped
        // promotion write costs nothing (the RAM body stays, demotion
        // covers eviction), a dropped write-through costs its node the
        // pending backing — then the remaining non-mandatory entries.
        for victimsArePureRedundancy in [true, false] {
            var scanIndex = 0
            while pendingBytes + payloadBytes > maxPendingBytes,
                scanIndex < pending.count
            {
                let candidate = pending[scanIndex]
                guard !candidate.mandatory,
                    candidate.deferrable == victimsArePureRedundancy
                else {
                    scanIndex += 1
                    continue
                }
                let oldest = pending.remove(at: scanIndex)
                pendingBytes -= oldest.payload.totalBytes
                droppedItems.append(
                    (
                        id: oldest.descriptor.snapshotID,
                        bytes: oldest.payload.totalBytes,
                        extendingBaseID: oldest.extendingBaseID
                    ))
            }
        }

        pending.append(item)
        pendingBytes += payloadBytes

        queueLock.unlock()

        for dropped in droppedItems {
            if let baseID = dropped.extendingBaseID {
                ledger.releaseExtensionTransfer(baseID: baseID)
            }
            // Both events fire per bumped item: the admission outcome
            // (`droppedByteBudget` is the terminal verdict for that
            // earlier `tryEnqueue` call), and the lifecycle callback
            // (`storageRefDropCallback`, the stable telemetry name, so
            // the radix node sees its pending ref cleared).
            PrefixCacheDiagnostics.logSystem(
                PrefixCacheDiagnostics.SSDAdmitEvent(
                    id: dropped.id,
                    bytes: dropped.bytes,
                    outcome: .droppedByteBudget
                ))
            PrefixCacheDiagnostics.logSystem(
                PrefixCacheDiagnostics.SnapshotRefDropCallbackEvent(
                    id: dropped.id,
                    reason: .backpressureOldest
                )
            )
            onDrop(dropped.id, .backpressureOldest)
        }
    }

    nonisolated func diagnosticsSnapshot() -> PromptCacheSSDSnapshot {
        // Two independent critical sections by design: the ledger's
        // residency totals and the queue depth live under separate locks
        // that intentionally never nest. A writer committing between the
        // two samples can yield a frame mixing pre-/post-commit
        // `currentBytes` and `pendingBytes`. That is acceptable — this is
        // purely observational telemetry, never a control input — so the
        // returned pair is *not* a consistent cross-lock snapshot.
        let residency = ledger.residencyStats()
        queueLock.lock()
        let queuedBytes = pendingBytes
        let queuedCount = pending.count
        queueLock.unlock()

        let budget = ledger.budgetContext()
        return PromptCacheSSDSnapshot(
            enabled: true,
            rootPath: rootURL.path,
            // The budget currently in force (measured; ADR-0018), not
            // the bootstrap constant this store was constructed with.
            budgetBytes: budget.budgetBytes,
            currentBytes: residency.currentBytes,
            pendingBytes: queuedBytes,
            maxPendingBytes: maxPendingBytes,
            pendingCount: queuedCount,
            snapshotCount: residency.snapshotCount,
            partitionCount: residency.partitionCount,
            budgetFloorBytes: budget.floorBytes,
            freeDiskBytes: budget.freeDiskBytes,
            budgetFloorBound: budget.floorBound
        )
    }

    /// Block until the writer's pending queue is fully drained and
    /// the in-memory manifest has been persisted to disk. Used by
    /// `LLMActor.unloadModel` / benchmark restart scenarios to
    /// guarantee that in-flight writes survive a `deinit`: without
    /// this, any pending descriptors and any dirty manifest get
    /// dropped when the store's detached writer task is cancelled.
    ///
    /// Idempotent; safe to call even if the queue is already empty
    /// — the writer is woken once and runs a no-op `drainPending`
    /// before the continuation resumes.
    func flushAsync() async {
        // A flush drains *everything*: deferred-class items stop
        // waiting out the activity gate — unload durability outranks
        // bandwidth scheduling.
        setForceDrainDeferred(true)
        defer { setForceDrainDeferred(false) }

        // Register before yielding so the writer can never skip our
        // signal. New admissions landing between the wakeup and the
        // resume pop off a second drain pass — covered by the loop.
        repeat {
            await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
                registerDrainWaiter(cont)
                wakeupContinuation.yield()
            }
        } while hasPendingWrites()

        ledger.persistNow()
    }

    private func setForceDrainDeferred(_ value: Bool) {
        queueLock.lock()
        forceDrainDeferred = value
        queueLock.unlock()
    }

    private func registerDrainWaiter(
        _ cont: CheckedContinuation<Void, Never>
    ) {
        queueLock.lock()
        defer { queueLock.unlock() }
        drainWaiters.append(cont)
    }

    private func hasPendingWrites() -> Bool {
        queueLock.lock()
        defer { queueLock.unlock() }
        return !pending.isEmpty
    }

    /// Bump the descriptor's `lastAccessAt` so the writer's
    /// admission-time LRU cut sees fresh recency data. A lookup
    /// landing on a committed ref calls this so that hot entries
    /// never look stale to the SSD-tier eviction policy. No-op when
    /// the ID is not in the manifest (the entry was evicted between
    /// the lookup and the bump).
    func recordHit(id: String) {
        ledger.recordHit(id: id)
    }

    /// Upsert a `PartitionMeta` entry into the manifest. Must be
    /// called at least once for each distinct `partitionDigest`
    /// before any descriptor with that digest is enqueued: the
    /// `SnapshotManifest` invariant requires every entry in
    /// `manifest.snapshots` to reference a partition present in
    /// `manifest.partitions`, and the warm-start path drops
    /// descriptors whose partition entry is missing as a
    /// defensive repair. Omitting this leaves persisted manifests
    /// unrecoverable.
    ///
    /// Idempotent — repeat calls with the same digest simply
    /// overwrite the stored metadata, which is the correct
    /// behavior when a partition's fingerprint or session
    /// affinity changes between model loads.
    func registerPartition(_ meta: PartitionMeta, digest: String) {
        ledger.registerPartition(meta, digest: digest)
    }

    /// Remove a snapshot from the SSD tier immediately. Handles all three
    /// states:
    /// - pending queue entry not yet popped
    /// - committed resident in the manifest
    /// - in-flight writer item already popped from the queue
    ///
    /// The committed and in-flight cases hand off to
    /// `ledger.removeOrTombstone`: a committed resident is returned for
    /// file deletion; an in-flight write is tombstoned so the writer's
    /// `consumeTombstone` pre-write skip and `commit` self-veto drop it.
    func deleteSnapshot(snapshotID: String) {
        queueLock.lock()
        if let pendingIndex = pending.firstIndex(where: { $0.descriptor.snapshotID == snapshotID })
        {
            let removed = pending.remove(at: pendingIndex)
            pendingBytes -= removed.payload.totalBytes
            if pendingBytes < 0 { pendingBytes = 0 }
            queueLock.unlock()
            if let baseID = removed.extendingBaseID {
                ledger.releaseExtensionTransfer(baseID: baseID)
            }
            return
        }
        queueLock.unlock()

        // Not in the pending queue: hand off to the ledger, which
        // atomically removes a committed resident (returning it for file
        // deletion) or tombstones an in-flight write so its later
        // `commit` self-vetoes. The queue lock is released first — the
        // two locks never nest. A chain head's removal deletes every
        // segment file it owns.
        if let resident = ledger.removeOrTombstone(id: snapshotID) {
            deleteChainFiles(resident.fileURLs)
            logSSDDelete(
                id: resident.snapshotID, bytes: resident.bytes, reason: .superseded
            )
        }
    }

    /// One-call `ssdDelete` emission. Every writer path that deletes a
    /// snapshot file goes through here so no deletion can miss the
    /// endurance ledger.
    private func logSSDDelete(
        id: String, bytes: Int, reason: PrefixCacheDiagnostics.SSDDeleteReason
    ) {
        PrefixCacheDiagnostics.logSystem(
            PrefixCacheDiagnostics.SSDDeleteEvent(id: id, bytes: bytes, reason: reason)
        )
    }

    /// Whether `snapshotID` is a base currently shielded by a pending
    /// **Leaf Extension Admission**. Forwards to the ledger's authoritative
    /// shield set — the same one the LRU cut already excludes.
    func isTransferringBase(_ snapshotID: String) -> Bool {
        ledger.isTransferringBase(snapshotID)
    }

    // MARK: - Writer loop

    private func writerLoop() async {
        for await _ in wakeupStream {
            await writerDrainPreludeForTesting?()
            await drainPending()
            resumeDrainWaiters()
        }
        // Stream closed (deinit). Fail any straggling waiters by
        // resuming them so `flushAsync` callers do not leak.
        resumeDrainWaiters()
    }

    private func resumeDrainWaiters() {
        queueLock.lock()
        let waiters = drainWaiters
        drainWaiters.removeAll(keepingCapacity: false)
        queueLock.unlock()
        for waiter in waiters {
            waiter.resume()
        }
    }

    /// Drain every pending item currently in the queue. Each
    /// iteration takes the lock only long enough to pop one item,
    /// processes it outside the lock, and releases the bytes under
    /// the lock afterward. Holding the lock during I/O would block
    /// the front door indefinitely.
    private func drainPending() async {
        while let item = popNextPending() {
            await processPendingItem(item)
        }
    }

    private func popNextPending() -> PendingWrite? {
        queueLock.lock()
        defer { queueLock.unlock() }
        guard !pending.isEmpty else {
            inFlightSnapshotID = nil
            return nil
        }
        // FIFO over the processable set: non-deferrable items always;
        // deferrable items when the gate is idle (or absent), the item
        // has waited out `maxDeferredHoldup`, or a flush forces the
        // drain. A blocked deferrable head does not block later
        // non-deferrable items — write-order between unrelated
        // snapshots carries no invariant (extension bases are promoted
        // to non-deferrable at the extension's enqueue, so base-before-
        // suffix is preserved the FIFO way).
        let gateBusy = activityGate?.isBusy ?? false
        let now: ContinuousClock.Instant = .now
        for index in pending.indices {
            let item = pending[index]
            if item.deferrable, !forceDrainDeferred, gateBusy,
                now - item.enqueuedAt < Self.maxDeferredHoldup
            {
                continue
            }
            let selected = pending.remove(at: index)
            inFlightSnapshotID = selected.descriptor.snapshotID
            return selected
        }
        // Only gate-blocked deferrable items remain — leave them queued
        // and let a delayed re-wake retry once the gate quiets down.
        inFlightSnapshotID = nil
        scheduleDeferredRewakeLocked()
        return nil
    }

    /// Schedule a one-shot delayed wakeup so gate-blocked deferrable
    /// items are re-checked without busy-spinning. Must be called with
    /// `queueLock` held.
    private func scheduleDeferredRewakeLocked() {
        guard !deferredRewakeScheduled else { return }
        deferredRewakeScheduled = true
        Task.detached { [weak self] in
            try? await Task.sleep(for: Self.deferredRecheckInterval)
            self?.clearDeferredRewakeAndWake()
        }
    }

    /// Synchronous tail of the delayed re-wake: `NSLock` cannot be
    /// taken inside an async context, so the rewake task calls out
    /// to this nonisolated-sync hop instead.
    private func clearDeferredRewakeAndWake() {
        queueLock.lock()
        deferredRewakeScheduled = false
        queueLock.unlock()
        wakeupContinuation.yield()
    }

    // Evolving MVP mid-refactor (see CLAUDE.md); structural limit kept lenient — splitting deferred.
    // swiftlint:disable:next function_body_length
    private func processPendingItem(_ item: PendingWrite) async {
        if ledger.consumeTombstone(id: item.descriptor.snapshotID) {
            if let baseID = item.extendingBaseID {
                ledger.releaseExtensionTransfer(baseID: baseID)
            }
            releasePendingBytes(item.payload.totalBytes)
            return
        }

        // 1. Admission LRU cut. The ledger returns the admit/drop
        //    decision and a list of committed residents it evicted to
        //    make room, under its own lock. File deletion and `onDrop`
        //    callbacks happen here, outside that lock.
        let (admission, evicted) = ledger.admit(
            item.descriptor,
            scoring: item.scoringConfig,
            condemned: item.condemnedResidentIDs
        )
        // Each evicted resident gets its own admission-cut event
        // before the file delete + drop callback fire. The pair
        // (`ssdEvictAtAdmission` + `storageRefDropCallback(...,
        // .evictedByLRU)`) lets operators correlate write-time
        // pressure with the lifecycle transitions on the radix
        // tree.
        for resident in evicted {
            PrefixCacheDiagnostics.logSystem(
                PrefixCacheDiagnostics.SSDEvictAtAdmissionEvent(
                    victimID: resident.snapshotID,
                    incomingID: item.descriptor.snapshotID
                )
            )
        }
        finalizeEvictions(evicted)

        // Writer-drop helper local to this item: every terminal drop
        // path must release the base's LRU shield exactly once.
        func dropItem(reason: SSDDropReason) {
            if let baseID = item.extendingBaseID {
                ledger.releaseExtensionTransfer(baseID: baseID)
            }
            releasePendingBytes(item.payload.totalBytes)
            emitWriterDrop(
                id: item.descriptor.snapshotID,
                bytes: item.payload.totalBytes,
                reason: reason,
                mandatory: item.mandatory
            )
        }

        switch admission {
        case .admit:
            break
        case .drop(let reason):
            dropItem(reason: reason)
            return
        }

        // 1b. Leaf extension: fold the base's chain into the descriptor
        //     *before* the write so the embedded per-file header carries
        //     the true chain. The base entry stays authoritative in the
        //     manifest until the commit below — a crash inside this
        //     window warm-starts at the base offset. FIFO has already
        //     settled the base (committed or dropped); a missing base
        //     here means its own write failed, so the suffix is dropped.
        let descriptorToWrite: PersistedSnapshotDescriptor
        if let baseID = item.extendingBaseID {
            guard
                let folded = ledger.prepareFoldedDescriptor(
                    item.descriptor,
                    baseID: baseID
                )
            else {
                dropItem(reason: .extensionBaseLost)
                return
            }
            descriptorToWrite = folded
        } else {
            descriptorToWrite = item.descriptor
        }

        // 2. Write the payload atomically (temp + fsync + rename).
        //    On disk-full, run a single eviction-retry pass — its
        //    victim is also cleaned up outside the lock. Any other
        //    error drops the incoming and fires the drop callback.
        do {
            try writePayload(item.payload, descriptor: descriptorToWrite)
        } catch WriteError.diskFull {
            if let retryVictim = ledger.retryAfterDiskFull(descriptorToWrite) {
                PrefixCacheDiagnostics.logSystem(
                    PrefixCacheDiagnostics.SSDEvictAtAdmissionEvent(
                        victimID: retryVictim.snapshotID,
                        incomingID: item.descriptor.snapshotID
                    )
                )
                finalizeEvictions([retryVictim])
                do {
                    try writePayload(item.payload, descriptor: descriptorToWrite)
                } catch {
                    Log.agent.error(
                        "SSD writer diskFull retry failed for \(item.descriptor.snapshotID): "
                            + "\(String(describing: error))"
                    )
                    dropItem(reason: .diskFull)
                    return
                }
            } else {
                Log.agent.error(
                    "SSD writer diskFull and no eviction victim available for "
                        + "\(item.descriptor.snapshotID)"
                )
                dropItem(reason: .diskFull)
                return
            }
        } catch {
            Log.agent.error(
                "SSD writer I/O failure for \(item.descriptor.snapshotID): "
                    + "\(String(describing: error))"
            )
            dropItem(reason: .writerIOError)
            return
        }

        if ledger.consumeTombstone(id: item.descriptor.snapshotID) {
            if let baseID = item.extendingBaseID {
                ledger.releaseExtensionTransfer(baseID: baseID)
            }
            try? FileManager.default.removeItem(at: fileURL(for: item.descriptor))
            logSSDDelete(
                id: item.descriptor.snapshotID,
                bytes: item.descriptor.bytes,
                reason: .tombstoneVeto
            )
            releasePendingBytes(item.payload.totalBytes)
            return
        }

        // 3. Write succeeded: register the descriptor in the ledger,
        //    release the pending byte budget, and fire the commit
        //    callback. For an extension the commit consumes the base
        //    entry atomically (the chain fold). A `false` here is the
        //    tombstone self-veto or a lost base — drop the own file
        //    (inherited files still belong to the surviving base entry,
        //    or were already deleted with it).
        guard
            ledger.commit(
                descriptorToWrite,
                consumingBase: item.extendingBaseID,
                refreshRecency: item.refreshRecencyAtCommit
            )
        else {
            try? FileManager.default.removeItem(at: fileURL(for: item.descriptor))
            logSSDDelete(
                id: item.descriptor.snapshotID,
                bytes: item.descriptor.bytes,
                reason: .tombstoneVeto
            )
            releasePendingBytes(item.payload.totalBytes)
            return
        }
        releasePendingBytes(item.payload.totalBytes)
        PrefixCacheDiagnostics.logSystem(
            PrefixCacheDiagnostics.SSDAdmitEvent(
                id: item.descriptor.snapshotID,
                bytes: item.descriptor.bytes,
                outcome: .accepted,
                writeClass: item.mandatory
                    ? "guarantee" : (item.deferrable ? "deferred" : "writeThrough")
            ))
        if let baseID = item.extendingBaseID {
            PrefixCacheDiagnostics.logSystem(
                PrefixCacheDiagnostics.LeafExtensionCommitEvent(
                    id: item.descriptor.snapshotID,
                    baseID: baseID,
                    suffixBytes: descriptorToWrite.bytes,
                    chainBytes: descriptorToWrite.totalBytes,
                    chainSegments: descriptorToWrite.inheritedSegments.count + 1
                )
            )
        }
        PrefixCacheDiagnostics.logSystem(
            PrefixCacheDiagnostics.SnapshotRefCommitEvent(id: item.descriptor.snapshotID)
        )
        onCommit(
            SSDCommitInfo(
                snapshotID: item.descriptor.snapshotID,
                consumedBaseID: item.extendingBaseID,
                chainBytesOnDisk: descriptorToWrite.totalBytes
            ))
    }

    /// Centralized writer-drop emission: terminal `ssdAdmit` outcome
    /// for the failed item, then the `storageRefDropCallback` telemetry
    /// lifecycle event, then the actual `onDrop` invocation.
    /// `processPendingItem`'s drop branches all funnel through here
    /// so the event ordering stays in lockstep with the callback
    /// firing order — tests can assert "admit drop precedes
    /// callback drop" without a per-branch check.
    private func emitWriterDrop(
        id: String,
        bytes: Int,
        reason: SSDDropReason,
        mandatory: Bool = false
    ) {
        let outcome: PrefixCacheDiagnostics.SSDAdmitOutcome
        switch reason {
        case .systemProtectionWins: outcome = .droppedSystemProtectionWins
        case .exceedsBudget: outcome = .droppedExceedsBudget
        case .diskFull: outcome = .droppedDiskFull
        case .writerIOError: outcome = .droppedWriterIOError
        case .extensionBaseLost: outcome = .droppedExtensionBaseLost
        case .backpressureOldest, .evictedByLRU, .hydrationFailure:
            // These reasons never originate from `processPendingItem`
            // — front-door, eviction loop, and hydration paths fire
            // their own events directly. Mapping them here would be
            // a logic error; surface it loudly.
            assertionFailure(
                "emitWriterDrop called with non-writer reason \(reason)"
            )
            outcome = .droppedWriterIOError
        }
        // A dropped guarantee-class write is a hard error (ADR-0019):
        // the Leaf Home Guarantee expected this copy to land. Legal only
        // for disk-full / I/O error / explicit invalidation — anything
        // else surfacing here at error level is a defect to triage.
        PrefixCacheDiagnostics.logSystem(
            PrefixCacheDiagnostics.SSDAdmitEvent(
                id: id,
                bytes: bytes,
                outcome: outcome,
                mandatory: mandatory
            ),
            level: mandatory ? .error : .info
        )
        PrefixCacheDiagnostics.logSystem(
            PrefixCacheDiagnostics.SnapshotRefDropCallbackEvent(
                id: id,
                reason: reason
            )
        )
        onDrop(id, reason)
    }

    /// Delete the on-disk files and fire `onDrop(.evictedByLRU)`
    /// for every resident that was removed from the manifest by
    /// the admission cut. Runs OUTSIDE the front-door lock so the
    /// synchronous `tryEnqueue` path never pays for filesystem I/O.
    private func finalizeEvictions(_ evicted: [EvictedResident]) {
        for resident in evicted {
            for url in resident.fileURLs {
                deleteResidentFile(url)
            }
            logSSDDelete(
                id: resident.snapshotID, bytes: resident.bytes, reason: .evicted
            )
            PrefixCacheDiagnostics.logSystem(
                PrefixCacheDiagnostics.SnapshotRefDropCallbackEvent(
                    id: resident.snapshotID,
                    reason: .evictedByLRU
                )
            )
            onDrop(resident.snapshotID, .evictedByLRU)
        }
    }

    private func deleteResidentFile(_ url: URL) {
        do {
            try FileManager.default.removeItem(at: url)
        } catch {
            Log.agent.warning(
                "SSD eviction failed to remove \(url.path): \(String(describing: error))"
            )
        }
    }

    /// Best-effort removal of a chain's on-disk segment files
    /// (explicit delete, hydration-failure sweep). Silent (`try?`) by
    /// design — a file already gone means a racing path beat us to it;
    /// contrast `deleteResidentFile`, whose eviction caller wants the
    /// warning.
    private func deleteChainFiles(_ urls: [URL]) {
        for url in urls {
            try? FileManager.default.removeItem(at: url)
        }
    }

    // MARK: - File write

    private enum WriteError: Error {
        case diskFull
        case ioError(underlying: Error)
    }

    /// Serialize the payload into the placeholder binary format,
    /// write it to `{snapshotID}.safetensors.tmp`, fsync, and
    /// rename atomically to `{snapshotID}.safetensors`. Any
    /// ENOSPC/EDQUOT error is surfaced as `.diskFull` so the caller
    /// can retry after eviction.
    private func writePayload(
        _ payload: SnapshotPayload,
        descriptor: PersistedSnapshotDescriptor
    ) throws {
        let finalURL = fileURL(for: descriptor)
        let tempURL = finalURL.appendingPathExtension("tmp")

        // Ensure directory tree exists before attempting the write.
        do {
            try FileManager.default.createDirectory(
                at: finalURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
        } catch {
            throw classifyWriteError(error)
        }

        let data: Data
        do {
            data = try encodePlaceholderContainer(payload: payload, descriptor: descriptor)
        } catch {
            throw WriteError.ioError(underlying: error)
        }

        // Remove any stale temp file from a previous aborted run
        // before we create the new handle, so the write is an
        // unambiguous fresh creation.
        try? FileManager.default.removeItem(at: tempURL)

        guard
            FileManager.default.createFile(atPath: tempURL.path, contents: nil)
        else {
            throw WriteError.ioError(
                underlying: NSError(
                    domain: NSCocoaErrorDomain,
                    code: NSFileWriteUnknownError,
                    userInfo: [NSLocalizedDescriptionKey: "createFile failed for \(tempURL.path)"]
                )
            )
        }

        let handle: FileHandle
        do {
            handle = try FileHandle(forWritingTo: tempURL)
        } catch {
            throw classifyWriteError(error)
        }

        do {
            try handle.write(contentsOf: data)
            try handle.synchronize()
            try handle.close()
        } catch {
            try? handle.close()
            try? FileManager.default.removeItem(at: tempURL)
            throw classifyWriteError(error)
        }

        // Atomic rename: on APFS this either completely succeeds
        // (temp vanishes, final appears) or completely fails (temp
        // intact, final unchanged). Crash-during-write leaves at
        // most a stale `.tmp` file, which warm start sweeps.
        do {
            if FileManager.default.fileExists(atPath: finalURL.path) {
                try FileManager.default.removeItem(at: finalURL)
            }
            try FileManager.default.moveItem(at: tempURL, to: finalURL)
        } catch {
            try? FileManager.default.removeItem(at: tempURL)
            throw classifyWriteError(error)
        }
    }

    /// Map Foundation errors to our `WriteError`. ENOSPC / EDQUOT
    /// become `.diskFull`; everything else is `.ioError`.
    private func classifyWriteError(_ error: Error) -> WriteError {
        let nsError = error as NSError

        if nsError.domain == NSPOSIXErrorDomain {
            switch nsError.code {
            case Int(ENOSPC), Int(EDQUOT):
                return .diskFull
            default:
                return .ioError(underlying: error)
            }
        }

        if nsError.domain == NSCocoaErrorDomain,
            nsError.code == NSFileWriteOutOfSpaceError
        {
            return .diskFull
        }

        return .ioError(underlying: error)
    }

    // MARK: - Pending byte budget

    /// Release `bytes` from the front door's pending byte counter.
    /// Called once per processed item regardless of outcome
    /// (commit, drop, retry failure).
    private func releasePendingBytes(_ bytes: Int) {
        queueLock.lock()
        defer { queueLock.unlock() }
        pendingBytes -= bytes
        if pendingBytes < 0 {
            // Indicates a bookkeeping bug — every processed item
            // must have come through `tryEnqueue` which already
            // added `bytes` to `pendingBytes`. Crash in debug so
            // the root cause surfaces; clamp in production so the
            // writer stays usable.
            assertionFailure("SSDSnapshotStore pendingBytes went negative")
            Log.agent.fault("SSDSnapshotStore pendingBytes went negative; clamping")
            pendingBytes = 0
        }
    }

    // MARK: - File path derivation

    /// Canonical on-disk URL for a snapshot file. Both the writer's
    /// descriptor-based path and the reader's `SnapshotRef`
    /// path must go through this single helper so the sharding rule
    /// cannot drift between write and read.
    private nonisolated func fileURL(
        snapshotID: String,
        partitionDigest: String
    ) -> URL {
        let relative = PersistedSnapshotDescriptor.relativeFilePath(
            snapshotID: snapshotID,
            partitionDigest: partitionDigest
        )
        return rootURL.appendingPathComponent(relative)
    }

    private func fileURL(for descriptor: PersistedSnapshotDescriptor) -> URL {
        fileURL(
            snapshotID: descriptor.snapshotID,
            partitionDigest: descriptor.partitionDigest
        )
    }

    // MARK: - Pending write (private value type)

    private struct PendingWrite: Sendable {
        let payload: SnapshotPayload
        let descriptor: PersistedSnapshotDescriptor
        /// Non-nil for a **Leaf Extension Admission**: the base whose
        /// chain the commit folds and whose LRU shield every terminal
        /// path must release.
        let extendingBaseID: String?
        /// `false` for a **Snapshot Demotion**: the commit must keep
        /// the descriptor's stale `lastAccessAt` instead of re-stamping
        /// it, so demoted bodies never look hot to the SSD LRU.
        let refreshRecencyAtCommit: Bool
        /// The **Eviction Configuration** snapshot taken at enqueue time
        /// (MainActor) — the shared α, FLOP profile, and measured
        /// estimates the ledger's terminal-loss cut scores with. Carried
        /// by value because the cut runs on the writer thread, where the
        /// manager's live configuration is unreachable; millisecond
        /// staleness is irrelevant to a damped α.
        let scoringConfig: EvictionConfiguration
        /// Residents this write supersedes (enqueue-before-delete,
        /// ADR-0019): their deletion is deferred to this write's commit,
        /// so the ledger's admission cut consumes them first — a
        /// near-full tier must not evict an unrelated resident to make
        /// room for a write whose own ancestors are already doomed.
        let condemnedResidentIDs: Set<String>
        /// Guarantee-class marker (the Leaf Home Guarantee write,
        /// ADR-0019): never a back-pressure victim, and any writer-side
        /// drop is a hard error logged at error level.
        let mandatory: Bool
        /// Deferred-class marker (PRD #150): the writer holds this item
        /// while the `StorageActivityGate` is busy (bounded by
        /// `maxDeferredHoldup`), and back-pressure victimizes it first.
        /// `var`: an extension enqueue promotes its queued base to
        /// immediate by clearing the flag.
        var deferrable: Bool = false
        /// Enqueue instant, for the deferred-holdup liveness bound.
        var enqueuedAt: ContinuousClock.Instant = .now
    }
}

// MARK: - Warm start + lazy hydration

extension SSDSnapshotStore {

    /// Load the SSD manifest and return the partitions that survive
    /// fingerprint validation against `expectedFingerprint`. Thin
    /// delegator to the **Snapshot Ledger**, which owns the manifest
    /// load, the corrupt-manifest directory-walk rebuild, the
    /// schema-mismatch backup + wipe, and the
    /// fingerprint/schema/checkpoint-type restore filtering. Called once
    /// at model load from `PrefixCacheManager.warmStart`; `nonisolated`
    /// so the manager invokes it without a hop.
    nonisolated func warmStartLoad(
        expectedFingerprint: String
    ) -> WarmStartOutcome {
        ledger.seedFromWarmStart(expectedFingerprint: expectedFingerprint)
    }

    /// Synchronously materialize an SSD-resident snapshot back into
    /// RAM. Must be called inside `container.perform` on LLMActor —
    /// the reconstructed MLXArrays are touched by the Metal command
    /// queue as soon as the caller evaluates them, and a background
    /// queue would deadlock per the oMLX regression.
    ///
    /// On any failure (partition not in manifest, fingerprint
    /// mismatch, file missing, decode error) the descriptor is
    /// removed from the manifest, the on-disk file is deleted, and
    /// `onDrop` fires with `.hydrationFailure`. Subsequent lookups
    /// on the same path therefore miss cleanly rather than
    /// re-attempting hydration on a broken file.
    ///
    /// Returns `nil` on any failure; `HybridCacheSnapshot` on success.
    nonisolated func loadSync(
        snapshotRef: SnapshotRef,
        expectedFingerprint: String,
        interruption: (@Sendable () -> Bool)? = nil
    ) -> HybridCacheSnapshot? {
        // Mark the hydration window for the deferred-write scheduler
        // (PRD #150): large-block reads and writes on one NVMe device
        // contend, so deferrable writer items wait this out.
        activityGate?.hydrationDidBegin()
        defer { activityGate?.hydrationDidEnd() }

        // Fingerprint gate: compare the partition's persisted
        // fingerprint against the caller's expected value. Mismatch
        // or missing partition is terminal — drop and miss. Read
        // through the ledger; a lock is not a MainActor hop, so this
        // off-main read still honours ADR-0001.
        let partitionFingerprint = ledger.partitionFingerprint(
            digest: snapshotRef.partitionDigest
        )

        guard let partitionFingerprint else {
            return failLoad(
                snapshotRef,
                reason: .partitionNotInManifest,
                "partition not in manifest digest=\(snapshotRef.partitionDigest)"
            )
        }
        guard partitionFingerprint == expectedFingerprint else {
            return failLoad(
                snapshotRef,
                reason: .fingerprintMismatch,
                "fingerprint mismatch partition=\(partitionFingerprint.prefix(8)) "
                    + "expected=\(expectedFingerprint.prefix(8))"
            )
        }

        // Resolve the resident's **Segment Chain** — one URL for an
        // ordinary full snapshot, several for an extension-built leaf.
        // A missing entry means the resident was evicted between the
        // lookup and this hydration.
        guard let chainURLs = ledger.chainForHydration(id: snapshotRef.snapshotID) else {
            return failLoad(
                snapshotRef,
                reason: .notResident,
                "chain lookup failed — resident evicted before hydration"
            )
        }

        // Read every segment file; let `Data(contentsOf:)` surface
        // missing / permission / IO errors via one catch site.
        // `.mappedIfSafe` lets the kernel page in on demand so
        // ~200 MiB snapshots do not spike peak RSS during hydration.
        // The interruption poll between segments is the yield point a
        // preempted background hydration exits through (PRD #149 item
        // 7) — an interrupted return leaves the backing untouched.
        var segmentData: [Data] = []
        segmentData.reserveCapacity(chainURLs.count)
        for url in chainURLs {
            if interruption?() == true { return nil }
            do {
                segmentData.append(try Data(contentsOf: url, options: .mappedIfSafe))
            } catch {
                return failLoad(
                    snapshotRef,
                    reason: .readFailed,
                    "read failed error=\(error)"
                )
            }
        }
        if interruption?() == true { return nil }

        // Decode and compose the chain, reconstructing MLXArrays
        // from raw payload bytes inside the caller's Metal-affine
        // context.
        do {
            return try decodeSegmentChain(
                segmentData,
                tokenOffset: snapshotRef.tokenOffset,
                checkpointType: snapshotRef.checkpointType
            )
        } catch {
            return failLoad(
                snapshotRef,
                reason: .decodeFailed,
                "decode failed error=\(error)"
            )
        }
    }

    /// Synchronously materialize a **Chain-Prefix Restore** point
    /// (ADR-0012): compose only the owning chain's leading segments
    /// covering `[0..point.boundaryOffset]` into a snapshot at the
    /// boundary. Same Metal-affinity contract as `loadSync` — must run
    /// inside `container.perform`. Strictly less work than a full-chain
    /// hydration: fewer files read, fewer arrays materialized, and the
    /// non-sliceable layers take the last *included* segment's copy —
    /// the recurrent state exactly as of the boundary, written at that
    /// historical leaf's own admission.
    ///
    /// Failure split, deliberate:
    /// - Owner gone / boundary off the segment grid / fingerprint or
    ///   partition gate: the *point* is stale, the chain (if any) may be
    ///   fine — return `nil` without touching the owner. The caller
    ///   clears the restore point; the next lookup degrades shallower.
    /// - Read / decode failure: the leading segments are shared with the
    ///   full chain, so the owner's own hydration would fail the same
    ///   way — condemn the whole chain (`dropHydrationFailure`), which
    ///   fires `onDrop(.hydrationFailure)` and eagerly clears the
    ///   owner's tree ref *and* its dependent restore points.
    nonisolated func loadSyncPrefix(
        point: ChainPrefixRestorePoint,
        expectedFingerprint: String,
        interruption: (@Sendable () -> Bool)? = nil
    ) -> HybridCacheSnapshot? {
        // Same hydration mark as `loadSync` (PRD #150).
        activityGate?.hydrationDidBegin()
        defer { activityGate?.hydrationDidEnd() }

        func missPoint(
            _ reason: PrefixCacheDiagnostics.SSDMissReason,
            _ message: @autoclosure () -> String
        ) -> HybridCacheSnapshot? {
            Log.agent.error(
                "SSDSnapshotStore.loadSyncPrefix: \(message()) "
                    + "owner=\(point.ownerSnapshotID.prefix(8)) boundary=\(point.boundaryOffset)"
            )
            PrefixCacheDiagnostics.logSystem(
                PrefixCacheDiagnostics.SSDMissEvent(
                    id: point.ownerSnapshotID,
                    reason: reason
                ))
            return nil
        }

        let partitionFingerprint = ledger.partitionFingerprint(
            digest: point.partitionDigest
        )
        guard let partitionFingerprint else {
            return missPoint(
                .partitionNotInManifest,
                "partition not in manifest digest=\(point.partitionDigest)"
            )
        }
        guard partitionFingerprint == expectedFingerprint else {
            return missPoint(
                .fingerprintMismatch,
                "fingerprint mismatch partition=\(partitionFingerprint.prefix(8)) "
                    + "expected=\(expectedFingerprint.prefix(8))"
            )
        }

        guard
            let prefixURLs = ledger.chainPrefixForHydration(
                ownerID: point.ownerSnapshotID,
                boundaryOffset: point.boundaryOffset
            )
        else {
            return missPoint(
                .notResident,
                "owner gone or boundary off the segment grid"
            )
        }

        var segmentData: [Data] = []
        segmentData.reserveCapacity(prefixURLs.count)
        for url in prefixURLs {
            // Yield point (PRD #149 item 7): an interrupted return
            // leaves the chain untouched — no condemn, no miss event.
            if interruption?() == true { return nil }
            do {
                segmentData.append(try Data(contentsOf: url, options: .mappedIfSafe))
            } catch {
                Log.agent.error(
                    "SSDSnapshotStore.loadSyncPrefix: read failed error=\(error) "
                        + "owner=\(point.ownerSnapshotID.prefix(8)) — condemning the chain"
                )
                PrefixCacheDiagnostics.logSystem(
                    PrefixCacheDiagnostics.SSDMissEvent(
                        id: point.ownerSnapshotID, reason: .readFailed
                    ))
                dropHydrationFailure(id: point.ownerSnapshotID)
                return nil
            }
        }

        if interruption?() == true { return nil }

        do {
            return try decodeSegmentChain(
                segmentData,
                tokenOffset: point.boundaryOffset,
                checkpointType: point.checkpointType
            )
        } catch {
            Log.agent.error(
                "SSDSnapshotStore.loadSyncPrefix: decode failed error=\(error) "
                    + "owner=\(point.ownerSnapshotID.prefix(8)) — condemning the chain"
            )
            PrefixCacheDiagnostics.logSystem(
                PrefixCacheDiagnostics.SSDMissEvent(
                    id: point.ownerSnapshotID, reason: .decodeFailed
                ))
            dropHydrationFailure(id: point.ownerSnapshotID)
            return nil
        }
    }

    /// Shared terminal branch for every `loadSync` failure path.
    /// Logs the `message`, emits an `ssdMiss(id:reason:)` diagnostic
    /// event so operators can correlate the hydration miss with the
    /// node id that just got cleared, drops the descriptor + on-disk
    /// file + fires `onDrop(.hydrationFailure)`, and returns `nil`
    /// so the caller can `return` the result directly.
    private nonisolated func failLoad(
        _ ref: SnapshotRef,
        reason: PrefixCacheDiagnostics.SSDMissReason,
        _ message: @autoclosure () -> String
    ) -> HybridCacheSnapshot? {
        Log.agent.error(
            "SSDSnapshotStore.loadSync: \(message()) "
                + "id=\(ref.snapshotID.prefix(8))"
        )
        PrefixCacheDiagnostics.logSystem(
            PrefixCacheDiagnostics.SSDMissEvent(
                id: ref.snapshotID,
                reason: reason
            ))
        dropHydrationFailure(id: ref.snapshotID)
        return nil
    }

    /// Remove the descriptor from the manifest, delete every on-disk
    /// segment file of its chain, and fire `onDrop` with
    /// `.hydrationFailure`. Called from every `loadSync` error path so
    /// the node's Snapshot Ref gets cleared and subsequent lookups miss
    /// cleanly.
    private nonisolated func dropHydrationFailure(id: String) {
        if let evicted = ledger.remove(id: id) {
            deleteChainFiles(evicted.fileURLs)
            logSSDDelete(
                id: evicted.snapshotID, bytes: evicted.bytes, reason: .hydrationFailure
            )
        }
        PrefixCacheDiagnostics.logSystem(
            PrefixCacheDiagnostics.SnapshotRefDropCallbackEvent(
                id: id,
                reason: .hydrationFailure
            )
        )
        onDrop(id, .hydrationFailure)
    }

    /// Decode a **Segment Chain** (ordered shallow→deep, own file last)
    /// into one composed `HybridCacheSnapshot`. Must run inside
    /// `container.perform` — constructing MLXArrays from `Data` is
    /// Metal-affine.
    ///
    /// Composition is per layer, shallow→deep: a whole-state layer
    /// entry *resets* the fold (last whole copy wins — the recurrent /
    /// rotating / chunked classes), a suffix entry *appends* its token
    /// range. Only contributing segments are materialized, and layers
    /// compose one at a time, so peak transient RAM stays around one
    /// snapshot plus one layer. Any cross-segment disagreement (layer
    /// counts, suffix contiguity, slot counts, dtypes, token-axis
    /// extents) throws `SSDLoadError.segmentMismatch`, condemning the
    /// chain via the caller's failure path.
    private nonisolated func decodeSegmentChain(
        _ segments: [Data],
        tokenOffset: Int,
        checkpointType: HybridCacheSnapshot.CheckpointType
    ) throws -> HybridCacheSnapshot {
        guard !segments.isEmpty else {
            throw SSDLoadError.segmentMismatch("empty chain")
        }

        var parsed: [(header: PlaceholderContainerHeader, blobsStart: Int, data: Data)] = []
        parsed.reserveCapacity(segments.count)
        for data in segments {
            let (header, blobsStart) = try PlaceholderContainerHeader.parse(from: data)
            parsed.append((header, blobsStart, data))
        }

        let layerCount = parsed[0].header.layers.count
        guard parsed.allSatisfy({ $0.header.layers.count == layerCount }) else {
            throw SSDLoadError.segmentMismatch("layer count varies across segments")
        }

        var totalBytes = 0
        var snapshotLayers: [HybridCacheSnapshot.LayerState] = []
        snapshotLayers.reserveCapacity(layerCount)

        for layerIndex in 0..<layerCount {
            // Pass 1 — headers only: which segments contribute, with
            // contiguity validated against each segment's capture offset.
            var contributors: [Int] = []
            var covered = 0
            for (segmentIndex, segment) in parsed.enumerated() {
                let layerHeader = segment.header.layers[layerIndex]
                let segmentOffset = segment.header.descriptor.tokenOffset
                if let suffixBase = layerHeader.suffixBaseOffset {
                    guard !contributors.isEmpty, suffixBase == covered else {
                        throw SSDLoadError.segmentMismatch(
                            "suffix base \(suffixBase) does not continue covered "
                                + "offset \(covered) at layer \(layerIndex)"
                        )
                    }
                    let expectedTokens = segmentOffset - suffixBase
                    guard
                        layerHeader.arrays.allSatisfy({
                            $0.shape.count >= 3 && $0.shape[$0.shape.count - 2] == expectedTokens
                        })
                    else {
                        throw SSDLoadError.segmentMismatch(
                            "suffix extent mismatch at layer \(layerIndex)"
                        )
                    }
                    contributors.append(segmentIndex)
                } else {
                    contributors = [segmentIndex]
                }
                covered = segmentOffset
            }

            let lastLayerHeader = parsed[contributors.last!].header.layers[layerIndex]
            guard
                contributors.allSatisfy({
                    let layer = parsed[$0].header.layers[layerIndex]
                    return layer.className == lastLayerHeader.className
                        && layer.arrays.count == lastLayerHeader.arrays.count
                })
            else {
                throw SSDLoadError.segmentMismatch(
                    "class or array-slot mismatch at layer \(layerIndex)"
                )
            }

            // Pass 2 — materialize contributors and concatenate suffix
            // pieces along the token axis (dim −2).
            var pieces: [[MLXArray]] = []
            pieces.reserveCapacity(contributors.count)
            for segmentIndex in contributors {
                let segment = parsed[segmentIndex]
                let arrays = try materializeLayerArrays(
                    segment.header.layers[layerIndex],
                    blobsStart: segment.blobsStart,
                    data: segment.data,
                    totalBytes: &totalBytes
                )
                pieces.append(arrays)
            }
            let slotCount = pieces[0].count
            let merged: [MLXArray] = (0..<slotCount).map { slot in
                let slotPieces = pieces.map { $0[slot] }
                return slotPieces.count == 1
                    ? slotPieces[0]
                    : concatenated(slotPieces, axis: -2)
            }

            snapshotLayers.append(
                HybridCacheSnapshot.LayerState(
                    className: lastLayerHeader.className,
                    state: merged,
                    metaState: lastLayerHeader.metaState,
                    offset: lastLayerHeader.offset
                ))
        }

        return HybridCacheSnapshot(
            tokenOffset: tokenOffset,
            layers: snapshotLayers,
            checkpointType: checkpointType,
            memoryBytes: totalBytes,
            createdAt: .now
        )
    }

    /// Materialize one layer's blob slices into MLXArrays. The header
    /// carries per-array `byte_offset` / `byte_size` pairs relative to
    /// the blob section; we slice those bytes out and feed them to
    /// `MLXArray(_:_:dtype:)` — a convenience initializer that owns a
    /// fresh backing allocation without going through a safetensors
    /// round-trip.
    private nonisolated func materializeLayerArrays(
        _ layerHeader: PlaceholderContainerHeader.Layer,
        blobsStart: Int,
        data: Data,
        totalBytes: inout Int
    ) throws -> [MLXArray] {
        var stateArrays: [MLXArray] = []
        stateArrays.reserveCapacity(layerHeader.arrays.count)
        for arrayHeader in layerHeader.arrays {
            guard let dtype = ServerCompletion.dtypeFromWireString(arrayHeader.dtype) else {
                throw SSDLoadError.unknownDType(arrayHeader.dtype)
            }
            // `byteOffset` / `byteSize` are `Int`s JSON-decoded from
            // the untrusted header — guard non-negativity and use
            // checked adds so a corrupt or hostile pair throws the
            // recoverable `.truncatedBlob` miss instead of trapping
            // the `Int` arithmetic or the `Data` subscript below.
            guard arrayHeader.byteOffset >= 0, arrayHeader.byteSize >= 0 else {
                throw SSDLoadError.truncatedBlob
            }
            let (sliceStart, startOverflow) =
                blobsStart.addingReportingOverflow(arrayHeader.byteOffset)
            let (sliceEnd, endOverflow) =
                sliceStart.addingReportingOverflow(arrayHeader.byteSize)
            guard !startOverflow, !endOverflow, sliceEnd <= data.count else {
                throw SSDLoadError.truncatedBlob
            }
            let blob = data[sliceStart..<sliceEnd]
            stateArrays.append(MLXArray(blob, arrayHeader.shape, dtype: dtype))
            totalBytes += arrayHeader.byteSize
        }
        return stateArrays
    }

}

// MARK: - Testing hooks

extension SSDSnapshotStore {

    /// Synchronous accessor for the current SSD byte count. Delegates to
    /// the ledger, which owns the byte total.
    nonisolated func currentSSDBytesForTesting() -> Int {
        ledger.currentSSDBytesForTesting()
    }

    /// Snapshot of the manifest's descriptor IDs, sorted by
    /// `lastAccessAt` ascending. Exposed for tests that verify the LRU
    /// ordering after a sequence of commits.
    nonisolated func residentIDsByRecencyForTesting() -> [String] {
        ledger.residentIDsByRecencyForTesting()
    }

    /// Synchronous peek at the pending queue depth. Tests use this to
    /// assert back-pressure eviction happened immediately rather than
    /// deferred to the writer. Queue-side state, so it stays on the store.
    nonisolated func pendingCountForTesting() -> Int {
        queueLock.lock()
        defer { queueLock.unlock() }
        return pending.count
    }

    /// Force-flush the debounced manifest persist. Tests call this to
    /// observe on-disk manifest state without waiting on the debounce
    /// window.
    nonisolated func flushManifestForTesting() {
        ledger.persistNow()
    }

    /// Inject a descriptor into the manifest without going through the
    /// writer loop. Tests use this to drive `recordHit` / `loadSync`
    /// scenarios deterministically. The partition must already be
    /// registered so the manifest invariant holds.
    nonisolated func seedDescriptorForTesting(_ descriptor: PersistedSnapshotDescriptor) {
        ledger.seedDescriptorForTesting(descriptor)
    }

    /// Read a descriptor's current `lastAccessAt` without mutating
    /// anything. Used by the `recordHit` regression test.
    nonisolated func lastAccessAtForTesting(id: String) -> Double {
        ledger.lastAccessAtForTesting(id: id)
    }

    /// Read a resident's full descriptor (chain fields included). Used
    /// by the leaf-extension fold tests.
    nonisolated func residentDescriptorForTesting(id: String) -> PersistedSnapshotDescriptor? {
        ledger.residentDescriptorForTesting(id: id)
    }

    /// The set of base IDs currently shielded by a pending extension
    /// transfer. Empty when no extension is in flight — the leaf-extension
    /// tests assert every terminal writer path releases its shield.
    nonisolated func transferringBaseIDsForTesting() -> Set<String> {
        ledger.transferringBaseIDsForTesting()
    }
}

// MARK: - Placeholder on-disk format
//
// The placeholder-container header type, its header-only parse, and the
// `encodePlaceholderContainer` writer live in the neutral
// `PlaceholderContainer.swift` so the **Snapshot Ledger**'s rebuild and
// the store's writer/`loadSync` share one MLX-free codec without a
// store↔ledger dependency cycle. The Metal-affine
// `decodePlaceholderContainer` (above) stays here, inside
// `container.perform`.
