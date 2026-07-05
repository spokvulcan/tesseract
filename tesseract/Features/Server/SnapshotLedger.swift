//
//  SnapshotLedger.swift
//  tesseract
//
//  The **Snapshot Ledger** — the in-memory authority over the SSD
//  prefix-cache tier: which snapshots are resident, their byte budget,
//  their recency, and the durability of that record. Carved out of
//  `SSDSnapshotStore` so the on-disk schema's whole life has one home:
//  load `manifest.json`, rebuild-on-corruption from a directory walk,
//  fingerprint / schema-version / checkpoint-type restore filtering,
//  the type-protected LRU admission cut, the recency bump, the
//  in-flight-delete tombstone protocol, and the debounced
//  `manifest.json` + `_meta.json` persist.
//
//  **The split from the store is along the lock.** `SSDSnapshotStore`
//  keeps the *queue lock* (`pending`, `pendingBytes`, `drainWaiters`)
//  and the `.safetensors` body I/O; the Ledger keeps the *ledger lock*
//  (`manifest`, `currentSSDBytes`, the persist dirty-flag/task, the
//  tombstones). The two locks never nest — the writer is
//  single-threaded and releases the lock between every step, and the
//  one cross-lock method (`SSDSnapshotStore.deleteSnapshot`) runs them
//  sequentially. See `CONTEXT.md` ("SSD snapshot ledger").
//
//  **The Ledger returns what changed; the store performs the effects.**
//  `admit`, `retryAfterDiskFull`, `removeOrTombstone`, and `remove`
//  return the residents that left the manifest; the store deletes their
//  files and fires `onCommit`/`onDrop` + diagnostics *outside* any lock,
//  so a slow file delete never blocks a concurrent front-door
//  `tryEnqueue`. The type-protected LRU policy lives **inside** `admit`,
//  run atomically under the ledger lock — a multi-candidate cut must not
//  tear.
//
//  **Not a Swift `actor`.** It is reached from the off-MainActor
//  `SSDSnapshotStore.loadSync` (the ADR-0001 path) for the
//  `partitionFingerprint(digest:)` gate and the failure-path `remove`,
//  so it stays a `nonisolated` lock-based class. A lock is not a
//  MainActor hop, so this does not reopen ADR-0001.
//

import Foundation
import MLXLMCommon

// MARK: - Warm-start outcome

/// Partitioned view of a successfully loaded (or rebuilt) manifest.
/// Returned by `SnapshotLedger.seedFromWarmStart` so
/// `PrefixCacheManager.warmStart` can iterate the valid descriptors
/// without taking the ledger lock.
nonisolated struct WarmStartOutcome: Sendable {
    struct Partition: Sendable {
        let digest: String
        let meta: PartitionMeta
        let descriptors: [PersistedSnapshotDescriptor]
    }

    /// Why warm start reclaimed a partition. Rendered into the
    /// `ssdPartitionInvalidated` diagnostics event and, from there,
    /// the cache panel's notable-events feed — the 2026-07-04 lesson
    /// that a legitimate invalidation must never be silent.
    enum PartitionInvalidationReason: String, Sendable {
        /// The partition's persisted `modelFingerprint` no longer
        /// matches the loaded model (model re-downloaded / changed —
        /// or, under the single-model tier, a different model loaded).
        case fingerprintChanged
        /// Stale-partition GC (PRD #150): unused past
        /// `SSDStalePartitionPolicy.maxUnusedAge`.
        case staleUnused
        /// The `PartitionMeta` carries a stale schema version inside a
        /// current-version manifest — a hand-edited or partially
        /// upgraded file.
        case schemaStale
    }

    /// One reclaimed partition, with everything the diagnostics event
    /// (and the panel copy "Cache for <model> was reset") needs.
    struct InvalidatedPartition: Sendable {
        let digest: String
        let modelID: String
        /// Chain-total bytes of the descriptors that left the manifest
        /// with this partition — what the reclaim returned to the budget.
        let bytes: Int
        let reason: PartitionInvalidationReason
    }

    let validPartitions: [Partition]
    let invalidated: [InvalidatedPartition]

    var invalidatedPartitionDigests: [String] { invalidated.map(\.digest) }

    static let empty = WarmStartOutcome(
        validPartitions: [],
        invalidated: []
    )
}

// MARK: - Evicted resident

/// A committed resident the ledger removed from the in-memory manifest
/// but whose on-disk files have not yet been deleted and whose `onDrop`
/// callback has not yet fired. The store finalizes these effects
/// (file deletes + callback + diagnostics) *outside* any lock. Carrying
/// the `fileURLs` here lets the store delete without recomputing the
/// sharded paths — one URL per **Snapshot Segment** of the resident's
/// chain (a single element for ordinary full snapshots).
nonisolated struct EvictedResident: Sendable {
    let snapshotID: String
    let fileURLs: [URL]
    /// Chain-total bytes the removal freed (`descriptor.totalBytes`) —
    /// the endurance ledger's bytes-deleted input (PRD #150).
    let bytes: Int
}

// MARK: - Admission decision

/// The ledger's admit verdict for one incoming descriptor. `.drop`
/// carries the `SSDDropReason` the store maps to the terminal `ssdAdmit`
/// diagnostic + `onDrop` callback.
nonisolated enum AdmissionDecision: Sendable, Equatable {
    case admit
    case drop(SSDDropReason)
}

// MARK: - SnapshotLedger

nonisolated final class SnapshotLedger: @unchecked Sendable {

    // MARK: - Configuration (immutable after init)

    /// Root of the SSD tier on disk. The Ledger owns `manifest.json`
    /// and the per-partition `_meta.json` sidecars under it; the store
    /// owns the `.safetensors` bodies under the same root.
    let rootURL: URL

    /// Bootstrap SSD byte budget (ADR-0018): in force before the first
    /// free-disk measurement, and the floor of the measured value. When
    /// no `freeDiskBytesProvider` is injected this *is* the budget for
    /// the ledger's lifetime — the pre-dynamic behavior every test
    /// fixture pins.
    let budgetBytes: Int

    /// User cap on the dynamic budget (nil = "Automatic"). Caps the
    /// measured value, never raises it (ADR-0018: caps, never floors).
    private let budgetCapBytes: Int?

    /// Free-disk probe for the volume under `rootURL`; `nil` disables
    /// dynamic budgeting. Injected so tests can script disk sizes.
    private let freeDiskBytesProvider: ((URL) -> Int?)?

    /// Minimum spacing between free-disk measurements. Consulted at the
    /// admission cut — the only place the budget matters — so an idle
    /// tier measures nothing.
    static let budgetReevaluationMinimumInterval: Duration = .seconds(60)

    /// Minimum idle time before the in-memory manifest is persisted.
    /// Injected so tests can shorten it; production uses 500 ms.
    private let manifestDebounce: Duration

    // MARK: - Lock-protected mutable state

    private let lock = NSLock()
    private var manifest: SnapshotManifest = .empty()
    private var currentSSDBytes: Int = 0
    /// The budget currently in force (ADR-0018): starts at the
    /// (cap-clamped) bootstrap, re-derived from measured free disk by
    /// `reevaluateBudgetIfDueLocked`.
    private var dynamicBudgetBytes: Int
    private var lastBudgetReevaluation: ContinuousClock.Instant?
    /// Last measured free-disk bytes (`nil` until the first probe, and
    /// forever when no provider is injected). Panel context only —
    /// never a control input.
    private var lastMeasuredFreeDiskBytes: Int?
    /// Whether the last measurement was floor-bound — free space, not
    /// policy, holding the budget up. Distinct from `budget == floor`:
    /// a user cap below the floor must not read as a full disk.
    private var lastMeasurementFloorBound = false
    private var manifestDirty: Bool = false
    private var manifestPersistTask: Task<Void, Never>?
    /// Manifest file writes currently in flight. Claimed under `lock`
    /// (together with clearing `manifestDirty`), performed outside it. A
    /// `persistNow` caller that observes a clean flag waits for this to
    /// reach zero, so "persist now" always means the manifest is on disk
    /// when the call returns — without it, a caller racing the debounced
    /// persist task returns while the file write is still in flight.
    private var manifestWritesInFlight = 0
    private let manifestWriteCondition = NSCondition()
    /// Snapshot IDs the store deleted while a write may already be in
    /// flight. `consumeTombstone` (the writer's pre-write skip) and
    /// `commit` (the self-veto) both check this set so a superseded
    /// leaf never lands in the manifest after its node was cleaned up.
    private var deletedInFlightSnapshotIDs: Set<String> = []

    /// Base snapshot IDs a pending **Leaf Extension Admission** will
    /// fold at commit. While an ID is in this set the LRU cut must not
    /// pick it as a victim — the cut would orphan the in-flight suffix.
    /// Inserted by `beginExtensionTransfer` (the front door), removed by
    /// `commit(_:consumingBase:)` or `releaseExtensionTransfer` (every
    /// writer drop / tombstone path).
    private var transferringBaseIDs: Set<String> = []

    init(
        rootURL: URL,
        budgetBytes: Int,
        manifestDebounce: Duration,
        budgetCapBytes: Int? = nil,
        freeDiskBytesProvider: ((URL) -> Int?)? = nil
    ) {
        self.rootURL = rootURL
        self.budgetBytes = budgetBytes
        self.budgetCapBytes = budgetCapBytes
        self.freeDiskBytesProvider = freeDiskBytesProvider
        self.dynamicBudgetBytes = applyBudgetCap(budgetBytes, cap: budgetCapBytes)
        self.manifestDebounce = manifestDebounce
    }

    /// The budget currently in force — the bootstrap until the first
    /// measurement, then the measured value. Exposed for diagnostics.
    func currentBudgetBytes() -> Int {
        lock.lock()
        defer { lock.unlock() }
        return dynamicBudgetBytes
    }

    /// Panel-facing budget context (PRD #150): the budget in force, the
    /// floor it degrades to, the last measured free-disk bytes (`nil`
    /// when dynamic budgeting is off or unmeasured), and whether that
    /// measurement was floor-bound. The panel's "nearly-full disk
    /// degraded to the floor" copy reads off exactly this.
    func budgetContext() -> (
        budgetBytes: Int, floorBytes: Int, freeDiskBytes: Int?, floorBound: Bool
    ) {
        lock.lock()
        defer { lock.unlock() }
        return (
            dynamicBudgetBytes, budgetBytes, lastMeasuredFreeDiskBytes,
            lastMeasurementFloorBound
        )
    }

    /// Re-derive the budget from measured free disk space, throttled
    /// (ADR-0018). Must be called with `lock` held — it sits at the top
    /// of the admission cut, the one consumer of the budget. A failed
    /// probe keeps the current value.
    private func reevaluateBudgetIfDueLocked() {
        guard let freeDiskBytesProvider else { return }
        let now: ContinuousClock.Instant = .now
        if let last = lastBudgetReevaluation,
            now - last < Self.budgetReevaluationMinimumInterval
        {
            return
        }
        lastBudgetReevaluation = now
        guard let freeDiskBytes = freeDiskBytesProvider(rootURL) else { return }
        lastMeasuredFreeDiskBytes = freeDiskBytes
        lastMeasurementFloorBound = SSDBudgetPolicy.isFloorBound(
            freeDiskBytes: freeDiskBytes,
            currentTierBytes: currentSSDBytes,
            floorBytes: budgetBytes
        )
        let next = SSDBudgetPolicy.budgetBytes(
            freeDiskBytes: freeDiskBytes,
            currentTierBytes: currentSSDBytes,
            floorBytes: budgetBytes,
            capBytes: budgetCapBytes
        )
        guard next != dynamicBudgetBytes else { return }
        let previous = dynamicBudgetBytes
        dynamicBudgetBytes = next
        PrefixCacheDiagnostics.logSystem(
            PrefixCacheDiagnostics.SSDBudgetChangeEvent(
                previousBytes: previous,
                currentBytes: next,
                freeDiskBytes: freeDiskBytes,
                tierBytes: currentSSDBytes,
                capBytes: budgetCapBytes
            ))
    }

    deinit {
        manifestPersistTask?.cancel()
    }

    // MARK: - Descriptor schema factory

    /// Mint a `PersistedSnapshotDescriptor` for a freshly captured
    /// snapshot. The ledger owns the on-disk schema — the rebuild walk,
    /// the manifest persist, and the `relativeFilePath` sharding all live
    /// here — so descriptor construction lives here too rather than on
    /// `PrefixCacheManager`. Callers (the `TieredSnapshotStore` enqueue
    /// front door) hand domain inputs and receive a fully-shaped
    /// descriptor: a fresh `snapshotID`, the sharded `fileRelativePath`
    /// derived from it, and the current `schemaVersion` stamp.
    ///
    /// `createdAt` is stamped to now. `lastAccessAt` defaults to now and
    /// `commit` re-stamps it when the write lands so the first LRU cut sees
    /// fresh recency — except for a **Snapshot Demotion**, which passes the
    /// node's real (stale) `lastAccessAt` here and suppresses the commit
    /// re-stamp, so demoted bodies never look hot to the SSD LRU.
    /// `nonisolated static` — pure construction, no ledger state,
    /// callable from the MainActor front door without a hop.
    nonisolated static func makeDescriptor(
        partitionKey: CachePartitionKey,
        pathFromRoot: [Int],
        snapshot: HybridCacheSnapshot,
        payloadBytes: Int,
        segmentBaseOffset: Int = 0,
        lastAccessAt: TimeInterval? = nil
    ) -> PersistedSnapshotDescriptor {
        let snapshotID = UUID().uuidString
        let partitionDigest = partitionKey.partitionDigest
        let now = Date().timeIntervalSinceReferenceDate
        return PersistedSnapshotDescriptor(
            snapshotID: snapshotID,
            partitionDigest: partitionDigest,
            pathFromRoot: pathFromRoot,
            tokenOffset: snapshot.tokenOffset,
            checkpointType: snapshot.checkpointType.wireString,
            bytes: payloadBytes,
            segmentBaseOffset: segmentBaseOffset,
            createdAt: now,
            lastAccessAt: lastAccessAt ?? now,
            fileRelativePath: PersistedSnapshotDescriptor.relativeFilePath(
                snapshotID: snapshotID,
                partitionDigest: partitionDigest
            ),
            schemaVersion: SnapshotManifestSchema.currentVersion
        )
    }

    // MARK: - Leaf extension transfer protocol

    /// Front-door validation for a **Leaf Extension Admission**: shield
    /// the base from the LRU cut for the pending window. Returns `false`
    /// when the base is neither resident nor — per the caller's own
    /// queue check — queued/in-flight, in which case the extension must
    /// be rejected (its suffix payload cannot compose without the base).
    /// The caller (`SSDSnapshotStore.tryEnqueue`) checks its pending
    /// queue under the queue lock *before* this call; the two locks
    /// never nest.
    func beginExtensionTransfer(
        baseID: String,
        baseIsQueuedOrInFlight: Bool
    ) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard manifest.snapshots[baseID] != nil || baseIsQueuedOrInFlight else {
            return false
        }
        transferringBaseIDs.insert(baseID)
        return true
    }

    /// Release the LRU shield on `baseID` without folding — every
    /// writer path that terminates a pending extension short of commit
    /// (back-pressure drop, tombstone, write failure, fold failure)
    /// must call this exactly once.
    func releaseExtensionTransfer(baseID: String) {
        lock.lock()
        defer { lock.unlock() }
        transferringBaseIDs.remove(baseID)
    }

    /// Whether `baseID` is currently shielded by a pending **Leaf
    /// Extension Admission** — its fold has neither committed nor dropped.
    /// The LRU cut already excludes this set; the manager's explicit
    /// supersession-delete path consults it through the same lock so it
    /// never reclaims a base a still-in-flight fold elsewhere on the tree
    /// will consume.
    func isTransferringBase(_ baseID: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return transferringBaseIDs.contains(baseID)
    }

    /// Fold the base's **Segment Chain** into a pending extension's
    /// descriptor: inherited = base's inherited + base's own file. Run
    /// by the writer *before* `writePayload` so the embedded per-file
    /// header carries the true chain. Pure read — the base entry stays
    /// authoritative in the manifest until `commit(_:consumingBase:)`,
    /// which is what makes a crash inside the window warm-start at the
    /// base offset instead of losing the conversation. Returns `nil`
    /// when the base is gone or its offset disagrees with the slice
    /// boundary (the suffix cannot compose) — the caller drops the item.
    func prepareFoldedDescriptor(
        _ descriptor: PersistedSnapshotDescriptor,
        baseID: String
    ) -> PersistedSnapshotDescriptor? {
        lock.lock()
        defer { lock.unlock() }
        guard let base = manifest.snapshots[baseID],
            base.tokenOffset == descriptor.segmentBaseOffset
        else { return nil }
        return PersistedSnapshotDescriptor(
            snapshotID: descriptor.snapshotID,
            partitionDigest: descriptor.partitionDigest,
            pathFromRoot: descriptor.pathFromRoot,
            tokenOffset: descriptor.tokenOffset,
            checkpointType: descriptor.checkpointType,
            bytes: descriptor.bytes,
            segmentBaseOffset: descriptor.segmentBaseOffset,
            inheritedSegments: base.inheritedSegments + [base.ownSegment],
            createdAt: descriptor.createdAt,
            lastAccessAt: descriptor.lastAccessAt,
            fileRelativePath: descriptor.fileRelativePath,
            schemaVersion: descriptor.schemaVersion
        )
    }

    // MARK: - Partition registry

    /// True when `digest` has a registered partition. The store's
    /// `tryEnqueue` front door uses this to enforce the manifest
    /// invariant — every snapshot entry must reference a registered
    /// partition — before mutating its queue.
    func hasPartition(digest: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return manifest.partitions[digest] != nil
    }

    /// The persisted `modelFingerprint` for `digest`, or `nil` when the
    /// partition is not registered. The off-MainActor `loadSync` read
    /// uses this for its hydration gate (ADR-0001).
    func partitionFingerprint(digest: String) -> String? {
        lock.lock()
        defer { lock.unlock() }
        return manifest.partitions[digest]?.modelFingerprint
    }

    /// Upsert a `PartitionMeta` entry into the manifest and write its
    /// `_meta.json` sidecar. Must be called at least once for each
    /// distinct `digest` before any descriptor with that digest is
    /// committed: the `SnapshotManifest` invariant requires every entry
    /// in `manifest.snapshots` to reference a partition present in
    /// `manifest.partitions`, and the warm-start path drops descriptors
    /// whose partition entry is missing as a defensive repair.
    ///
    /// Idempotent — repeat calls with the same digest overwrite the
    /// stored metadata, which is correct when a partition's fingerprint
    /// or session affinity changes between model loads.
    ///
    /// **Use stamping (stale-partition GC, PRD #150).** A value-equal
    /// re-registration (warm start replaying the persisted meta) is a
    /// no-op — a warm start alone is not "use". A same-identity call
    /// with fresh timestamps (the admission path minting a new meta)
    /// preserves the stored `createdAt` and bumps `lastUsedAt`,
    /// throttled to `SSDStalePartitionPolicy.lastUsedRefreshInterval`
    /// so per-admission calls don't turn into a sidecar write storm.
    func registerPartition(_ meta: PartitionMeta, digest: String) {
        lock.lock()
        let existing = manifest.partitions[digest]
        if existing == meta {
            lock.unlock()
            return
        }

        let now = Date().timeIntervalSinceReferenceDate
        let stored: PartitionMeta
        if let existing, existing.sameIdentity(as: meta) {
            if let last = existing.lastUsedAt,
                now - last < SSDStalePartitionPolicy.lastUsedRefreshInterval
            {
                lock.unlock()
                return
            }
            var refreshed = existing
            refreshed.lastUsedAt = now
            stored = refreshed
        } else {
            var fresh = meta
            if fresh.lastUsedAt == nil {
                fresh.lastUsedAt = now
            }
            stored = fresh
        }
        manifest.partitions[digest] = stored
        manifestDirty = true
        scheduleManifestPersistLocked()
        lock.unlock()

        // Persist a per-partition `_meta.json` sidecar so the
        // directory-walk rebuild can validate partition fingerprints
        // without relying on the top-level `manifest.json`. The file is
        // idempotent — repeat writes for the same digest overwrite cleanly.
        writePartitionMetaFile(stored, digest: digest)
    }

    /// Serialize a `PartitionMeta` to `partitions/{digest}/_meta.json`.
    /// Best-effort: failures log at error level but do not abort the
    /// caller. The only consumer that depends on the file is the
    /// corrupt-manifest rebuild path, which already falls back to
    /// "invalidated partition" when the file is missing.
    private func writePartitionMetaFile(
        _ meta: PartitionMeta,
        digest: String
    ) {
        let dir =
            rootURL
            .appendingPathComponent("partitions")
            .appendingPathComponent(digest)
        let url = dir.appendingPathComponent("_meta.json")
        do {
            try FileManager.default.createDirectory(
                at: dir,
                withIntermediateDirectories: true
            )
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            let data = try encoder.encode(meta)
            try data.write(to: url, options: .atomic)
        } catch {
            Log.agent.error(
                "SnapshotLedger.writePartitionMetaFile failed "
                    + "digest=\(digest) error=\(String(describing: error))"
            )
        }
    }

    // MARK: - Recency

    /// Bump the descriptor's `lastAccessAt` so the admission-time LRU
    /// cut sees fresh recency data. A lookup landing on a committed ref
    /// calls this so hot entries never look stale to the eviction
    /// policy. No-op when the ID is not in the manifest (the entry was
    /// evicted between the lookup and the bump).
    func recordHit(id: String) {
        lock.lock()
        defer { lock.unlock() }
        guard var descriptor = manifest.snapshots[id] else { return }
        let now = Date().timeIntervalSinceReferenceDate
        descriptor.lastAccessAt = now
        manifest.snapshots[id] = descriptor
        // A hit is "use" for the stale-partition GC clock (PRD #150):
        // a read-heavy partition must not age out just because nothing
        // wrote to it. Throttled like the register-path bump; persisted
        // by the same debounced manifest write this hit schedules.
        if var meta = manifest.partitions[descriptor.partitionDigest],
            now - (meta.lastUsedAt ?? 0) >= SSDStalePartitionPolicy.lastUsedRefreshInterval
        {
            meta.lastUsedAt = now
            manifest.partitions[descriptor.partitionDigest] = meta
        }
        scheduleManifestPersistLocked()
    }

    // MARK: - Admission cut (type-protected, terminal-loss utility)

    /// The type-protected admission cut, scored by terminal-loss
    /// utility (ADR-0011). An SSD eviction is always a terminal loss —
    /// the next hit pays a full re-prefill — so non-`.system` victims
    /// fall in ascending `norm(recency) + α · norm(re-prefill seconds /
    /// chain bytes)` order under the one shared α carried in `scoring`.
    /// At `α = 0` (the default) the order is exactly the previous
    /// type-protected LRU. The asymmetric protection rule is unchanged:
    ///
    /// 1. Evict non-`.system` residents in utility order until the
    ///    incoming entry fits.
    /// 2. If non-system is exhausted and budget is still too tight:
    ///    - `.system` incoming → fall through to evict oldest
    ///      `.system` residents (lateral move, recency-ordered;
    ///      protection preserved across the set).
    ///    - non-system incoming → drop the incoming. Never evict
    ///      `.system` residents to make room for lower-value types —
    ///      hard protection, regardless of score.
    ///
    /// Returns the admit/drop decision plus the list of residents that
    /// were removed from the in-memory manifest, atomically under the
    /// ledger lock so a concurrent `recordHit` or hydration-failure
    /// `remove` cannot tear a multi-candidate cut. The store deletes the
    /// evicted files and fires `onDrop(.evictedByLRU)` **outside** the
    /// lock.
    func admit(
        _ descriptor: PersistedSnapshotDescriptor,
        scoring config: EvictionConfiguration = EvictionConfiguration(),
        condemned: Set<String> = []
    ) -> (decision: AdmissionDecision, evicted: [EvictedResident]) {
        var evicted: [EvictedResident] = []

        lock.lock()
        defer { lock.unlock() }

        // The budget matters exactly here — re-derive it from measured
        // free disk first (throttled; ADR-0018).
        reevaluateBudgetIfDueLocked()

        // For a pending extension this is the own-file bytes only —
        // the inherited chain is still counted under the (shielded)
        // base entry, so the net post-fold growth is exactly this.
        let spaceNeeded = descriptor.totalBytes
        if currentSSDBytes + spaceNeeded <= dynamicBudgetBytes {
            return (.admit, evicted)
        }

        // A payload larger than the whole budget can NEVER fit — reject
        // it before any eviction runs. Without this guard the cut would
        // destroy residents (condemned and eligible alike) to make room
        // that still isn't enough, then drop the incoming anyway.
        if spaceNeeded > dynamicBudgetBytes {
            return (.drop(.exceedsBudget), evicted)
        }

        // Pass 0 — residents this write supersedes (enqueue-before-delete,
        // ADR-0019: their deletion is deferred to this write's commit, so
        // they still occupy budget here). They are doomed either way, so
        // a near-full cut must consume them before touching an unrelated
        // resident. Shielded transfer bases never ride in `condemned` —
        // the manager's supersession walk excludes them.
        if !condemned.isEmpty {
            evictUnderLock(
                order: manifest.snapshots.values
                    .filter { condemned.contains($0.snapshotID) }
                    .sorted {
                        ($0.lastAccessAt, $0.snapshotID) < ($1.lastAccessAt, $1.snapshotID)
                    },
                until: spaceNeeded,
                into: &evicted
            )
            if currentSSDBytes + spaceNeeded <= dynamicBudgetBytes {
                return (.admit, evicted)
            }
        }

        // Unrecognized wire strings are treated as non-system so they
        // participate in normal utility eviction and never bypass system
        // protection. Front-door validation rejects these already, so
        // this branch is only reached in tests.
        let incomingType =
            HybridCacheSnapshot.CheckpointType(
                wireString: descriptor.checkpointType
            ) ?? .leaf

        // Pass 1 — evict non-system residents in terminal-loss utility
        // order under the lock.
        evictUnderLock(
            order: terminalLossOrderedResidentsLocked(
                matching: { $0 != .system },
                config: config
            ),
            until: spaceNeeded,
            into: &evicted
        )

        if currentSSDBytes + spaceNeeded <= dynamicBudgetBytes {
            return (.admit, evicted)
        }

        // Pass 2 — non-system eligible set exhausted.
        switch incomingType {
        case .system:
            // Lateral move: evict oldest system residents until fit.
            // Recency-only on purpose — among same-type survival
            // prefixes the utility refinement buys nothing, and the
            // LRU order is the long-standing pinned behavior.
            evictUnderLock(
                order: sortedEligibleResidentsLocked(matching: { $0 == .system }),
                until: spaceNeeded,
                into: &evicted
            )
            if currentSSDBytes + spaceNeeded <= dynamicBudgetBytes {
                return (.admit, evicted)
            }
            // Every resident is gone and the incoming still doesn't fit
            // — the single payload is larger than the total budget. A
            // configuration problem, not filesystem fullness.
            return (.drop(.exceedsBudget), evicted)

        case .leaf, .branchPoint:
            // Non-system incoming, non-system eligible set empty. System
            // protection kicks in: drop the incoming rather than
            // destroying any `.system` resident.
            return (.drop(.systemProtectionWins), evicted)
        }
    }

    /// Eviction-retry handle called when the store's `writePayload`
    /// throws `diskFull`. Evicts the single oldest eligible resident —
    /// non-system by default, falling through to any victim when the
    /// incoming is itself a `.system` entry. Returns the evicted
    /// resident so the store can delete the file and fire the drop
    /// callback outside the lock; `nil` when the eligible set is empty.
    func retryAfterDiskFull(
        _ descriptor: PersistedSnapshotDescriptor
    ) -> EvictedResident? {
        lock.lock()
        defer { lock.unlock() }

        let incomingType =
            HybridCacheSnapshot.CheckpointType(
                wireString: descriptor.checkpointType
            ) ?? .leaf
        let predicate: (HybridCacheSnapshot.CheckpointType) -> Bool =
            incomingType == .system ? { _ in true } : { $0 != .system }

        guard let victim = sortedEligibleResidentsLocked(matching: predicate).first else {
            return nil
        }
        return removeResidentUnderLock(snapshotID: victim.snapshotID)
    }

    /// Walk `order` (worst victim first), evicting each entry and
    /// freeing its SSD bytes until the requested room is available (or
    /// the order is exhausted). Every removed resident is appended to
    /// `evicted`. Must be called with `lock` held.
    private func evictUnderLock(
        order: [PersistedSnapshotDescriptor],
        until spaceNeeded: Int,
        into evicted: inout [EvictedResident]
    ) {
        for victim in order {
            if currentSSDBytes + spaceNeeded <= dynamicBudgetBytes {
                return
            }
            if let resident = removeResidentUnderLock(snapshotID: victim.snapshotID) {
                evicted.append(resident)
            }
        }
    }

    /// Eligible residents in eviction order under terminal-loss
    /// utility. Must be called with `lock` held.
    private func terminalLossOrderedResidentsLocked(
        matching predicate: (HybridCacheSnapshot.CheckpointType) -> Bool,
        config: EvictionConfiguration
    ) -> [PersistedSnapshotDescriptor] {
        terminalLossOrder(
            sortedEligibleResidentsLocked(matching: predicate),
            config: config
        )
    }

    /// Order descriptors worst-victim-first under terminal-loss
    /// utility: ascending `norm(1/age) + α · norm(re-prefill seconds /
    /// chain bytes)`. Re-prefill spans the *whole* chain — an SSD loss
    /// re-prefills `[0, tokenOffset]` from scratch — and bytes are the
    /// chain total (`totalBytes`), never per-segment values; both
    /// inputs are already persisted (schema v8 carries `tokenOffset`,
    /// `bytes`, and the inherited segments), so the cut needs no
    /// manifest schema bump. The `α = 0` fast path returns the plain
    /// LRU order — byte-identical to the pre-ADR-0011 cut. Pure
    /// derivation over its inputs; shared by the cut and the
    /// **Survival Gate** simulation.
    private func terminalLossOrder(
        _ candidates: [PersistedSnapshotDescriptor],
        config: EvictionConfiguration
    ) -> [PersistedSnapshotDescriptor] {
        let lru = candidates.sorted {
            ($0.lastAccessAt, $0.snapshotID) < ($1.lastAccessAt, $1.snapshotID)
        }
        guard config.alpha != 0, lru.count > 1 else { return lru }

        let now = Date().timeIntervalSinceReferenceDate
        let rawRecencies = lru.map {
            EvictionPolicy.recencyWeight(ageSeconds: now - $0.lastAccessAt)
        }
        let terms = EvictionPolicy.blendedTerms(
            rawRecencies: rawRecencies, alpha: config.alpha
        ) {
            lru.map { resident -> Double in
                guard resident.totalBytes > 0 else { return 0 }
                let rePrefillSeconds =
                    EvictionPolicy.parentRelativeFlops(
                        nodeOffset: resident.tokenOffset,
                        parentOffset: 0,
                        profile: config.flopProfile
                    ) / config.estimates.prefillFlopsPerSecond
                return rePrefillSeconds / Double(resident.totalBytes)
            }
        }

        return zip(lru, terms)
            .map { resident, terms in
                (resident: resident, utility: terms.utility)
            }
            .sorted {
                ($0.utility, $0.resident.lastAccessAt, $0.resident.snapshotID)
                    < ($1.utility, $1.resident.lastAccessAt, $1.resident.snapshotID)
            }
            .map(\.resident)
    }

    // MARK: - Survival Gate (PRD #82 slice #90)

    /// The **Survival Gate** pre-check: would an incoming chain survive
    /// the eviction its own admission triggers? Simulates the cut over
    /// residents ∪ {incoming} — freeing room worst-utility-first — and
    /// answers `false` exactly when the simulation picks the incoming
    /// itself before enough room exists, i.e. when writing it would
    /// only churn the SSD. Inert while the ledger is unfilled (an
    /// unfilled ledger admits everything, keeping cold-start behavior
    /// unchanged), and a `.system` incoming always passes — the cut's
    /// lateral move owns that case and the survival prefix must never
    /// gate itself out.
    ///
    /// One atomic read under the ledger lock. Pure — no effects; the
    /// caller (the **Snapshot Admission** front doors in
    /// `PrefixCacheManager`) decides what a `false` means: a demotion
    /// terminal-drops, a non-end-of-turn leaf degrades to RAM-only with
    /// supersession *preserve*, a checkpoint skips its write-through.
    func survivesAdmissionCut(
        tokenOffset: Int,
        totalBytes: Int,
        checkpointType: HybridCacheSnapshot.CheckpointType,
        lastAccessAt: TimeInterval,
        scoring config: EvictionConfiguration
    ) -> Bool {
        lock.lock()
        defer { lock.unlock() }

        // Same budget the cut it simulates would see (ADR-0018).
        reevaluateBudgetIfDueLocked()

        if currentSSDBytes + totalBytes <= dynamicBudgetBytes { return true }
        if checkpointType == .system { return true }

        // Stand-in descriptor carrying exactly the fields scoring
        // reads; the sentinel UUID cannot collide with a resident.
        let incoming = PersistedSnapshotDescriptor(
            snapshotID: UUID().uuidString,
            partitionDigest: "",
            pathFromRoot: [],
            tokenOffset: tokenOffset,
            checkpointType: checkpointType.wireString,
            bytes: totalBytes,
            createdAt: lastAccessAt,
            lastAccessAt: lastAccessAt,
            fileRelativePath: "",
            schemaVersion: SnapshotManifestSchema.currentVersion
        )

        let pool = terminalLossOrder(
            sortedEligibleResidentsLocked(matching: { $0 != .system }) + [incoming],
            config: config
        )
        var simulatedBytes = currentSSDBytes
        for victim in pool {
            if simulatedBytes + totalBytes <= dynamicBudgetBytes { return true }
            if victim.snapshotID == incoming.snapshotID { return false }
            simulatedBytes -= victim.totalBytes
        }
        return simulatedBytes + totalBytes <= dynamicBudgetBytes
    }

    /// The subset of manifest descriptors whose parsed checkpoint type
    /// satisfies `predicate`, sorted by `lastAccessAt` ascending (oldest
    /// first). Descriptors whose wire checkpoint type fails to parse are
    /// dropped — they shouldn't exist in practice and are not eviction
    /// candidates. Bases shielded by a pending extension transfer are
    /// excluded — evicting one would orphan the in-flight suffix. Must
    /// be called with `lock` held.
    private func sortedEligibleResidentsLocked(
        matching predicate: (HybridCacheSnapshot.CheckpointType) -> Bool
    ) -> [PersistedSnapshotDescriptor] {
        manifest.snapshots.values
            .filter { descriptor in
                guard !transferringBaseIDs.contains(descriptor.snapshotID) else {
                    return false
                }
                guard
                    let checkpointType = HybridCacheSnapshot.CheckpointType(
                        wireString: descriptor.checkpointType
                    )
                else { return false }
                return predicate(checkpointType)
            }
            .sorted { ($0.lastAccessAt, $0.snapshotID) < ($1.lastAccessAt, $1.snapshotID) }
    }

    /// Drop the manifest entry, decrement the SSD byte count by the
    /// chain total, and return an `EvictedResident` carrying every
    /// segment file URL so the store can delete them outside the lock.
    /// Returns `nil` when the snapshotID is not in the manifest.
    /// Schedules the debounced persist so eviction-only paths still
    /// write the updated manifest without waiting for an unrelated
    /// subsequent mutation. Must be called with `lock` held.
    private func removeResidentUnderLock(snapshotID: String) -> EvictedResident? {
        guard let descriptor = manifest.snapshots.removeValue(forKey: snapshotID) else {
            return nil
        }
        currentSSDBytes -= descriptor.totalBytes
        manifestDirty = true
        scheduleManifestPersistLocked()
        return EvictedResident(
            snapshotID: descriptor.snapshotID,
            fileURLs: chainFileURLs(for: descriptor),
            bytes: descriptor.totalBytes
        )
    }

    // MARK: - Commit + in-flight-delete tombstone protocol

    /// Insert a freshly written descriptor as a committed resident.
    /// Returns `false` — vetoing the commit — when a prior
    /// `removeOrTombstone` tombstoned this ID while the write was in
    /// flight, or when `consumingBase` names a base that is no longer
    /// resident (a concurrent hydration failure removed it; the suffix
    /// cannot compose); the store then deletes the orphaned own file.
    /// On success the entry's `lastAccessAt` is stamped to commit time
    /// so the first LRU cut sees fresh recency — unless `refreshRecency`
    /// is `false` (a **Snapshot Demotion**), in which case the
    /// descriptor's own `lastAccessAt` is preserved: demoted bodies are
    /// the *least* valuable, and re-stamping them would make every
    /// pressure event invert the SSD tier's recency signal. Only
    /// hydrations and extensions refresh (see `CONTEXT.md`).
    ///
    /// `consumingBase` is the **Leaf Extension Admission** fold: the
    /// base entry leaves the manifest (its files stay — they are the
    /// new entry's inherited segments) and its LRU shield is released,
    /// atomically with the insert, so no instant exists where the chain
    /// is double-counted or unowned.
    func commit(
        _ descriptor: PersistedSnapshotDescriptor,
        consumingBase: String? = nil,
        refreshRecency: Bool = true
    ) -> Bool {
        lock.lock()
        defer { lock.unlock() }

        if deletedInFlightSnapshotIDs.remove(descriptor.snapshotID) != nil {
            if let consumingBase {
                transferringBaseIDs.remove(consumingBase)
            }
            return false
        }

        if let consumingBase {
            transferringBaseIDs.remove(consumingBase)
            guard let base = manifest.snapshots.removeValue(forKey: consumingBase) else {
                return false
            }
            currentSSDBytes -= base.totalBytes
        }

        var fresh = descriptor
        if refreshRecency {
            fresh.lastAccessAt = Date().timeIntervalSinceReferenceDate
        }

        // Idempotent byte accounting: if this ID is already resident,
        // subtract its prior bytes before adding the fresh entry's, so a
        // re-commit replaces rather than double-counts (mirrors
        // `seedDescriptorForTesting`). The sole production caller commits
        // each fresh UUID once, so this is defensive against future
        // re-commit paths, not a live correction.
        if let prior = manifest.snapshots[descriptor.snapshotID] {
            currentSSDBytes -= prior.totalBytes
        }
        manifest.snapshots[descriptor.snapshotID] = fresh
        currentSSDBytes += descriptor.totalBytes
        manifestDirty = true
        scheduleManifestPersistLocked()
        return true
    }

    /// Atomically "resident → remove + return, else tombstone." The
    /// store's `deleteSnapshot` calls this after finding the ID absent
    /// from its pending queue: if the snapshot already committed, the
    /// resident is returned for file deletion; if a write is still in
    /// flight, a tombstone is recorded so the later `commit` self-vetoes.
    func removeOrTombstone(id: String) -> EvictedResident? {
        lock.lock()
        defer { lock.unlock() }
        if let resident = removeResidentUnderLock(snapshotID: id) {
            return resident
        }
        deletedInFlightSnapshotIDs.insert(id)
        return nil
    }

    /// Remove a committed resident from the manifest, returning it for
    /// file deletion. Used by the store's `loadSync` failure path
    /// (`dropHydrationFailure`). Returns `nil` when the ID is absent.
    func remove(id: String) -> EvictedResident? {
        lock.lock()
        defer { lock.unlock() }
        return removeResidentUnderLock(snapshotID: id)
    }

    /// The writer's pre-write skip: consume a tombstone for `id` if one
    /// exists, returning whether the in-flight write should be skipped.
    func consumeTombstone(id: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return deletedInFlightSnapshotIDs.remove(id) != nil
    }

    // MARK: - Diagnostics

    /// The residency totals the store folds into its
    /// `PromptCacheSSDSnapshot` alongside its own queue depth.
    func residencyStats() -> (currentBytes: Int, snapshotCount: Int, partitionCount: Int) {
        lock.lock()
        defer { lock.unlock() }
        return (currentSSDBytes, manifest.snapshots.count, manifest.partitions.count)
    }

    // MARK: - Manifest persistence (debounced)

    /// Schedule a manifest persist after the configured debounce window.
    /// Multiple rapid updates coalesce into a single write. Must be
    /// called with `lock` held.
    private func scheduleManifestPersistLocked() {
        manifestDirty = true
        manifestPersistTask?.cancel()
        let debounce = manifestDebounce
        manifestPersistTask = Task.detached { [weak self] in
            do {
                try await Task.sleep(for: debounce)
            } catch {
                return
            }
            self?.persistNow()
        }
    }

    /// Force the debounced persist now if the manifest is dirty. Called
    /// by the scheduled task after the debounce window, and synchronously
    /// by the store's `flushAsync` (after draining the writer) and the
    /// manifest-flush test hook.
    func persistNow() {
        lock.lock()
        guard manifestDirty else {
            lock.unlock()
            // A concurrent persist may have claimed the dirty flag and
            // still be mid-write. Wait it out so a return from this call
            // always means the claimed manifest state is on disk.
            manifestWriteCondition.lock()
            while manifestWritesInFlight > 0 {
                manifestWriteCondition.wait()
            }
            manifestWriteCondition.unlock()
            return
        }
        let snapshot = manifest
        manifestDirty = false
        // Claimed under `lock`, so a clean-flag observer above can never
        // miss the write this claim is about to perform.
        manifestWriteCondition.lock()
        manifestWritesInFlight += 1
        manifestWriteCondition.unlock()
        lock.unlock()

        let manifestURL = self.manifestURL
        var writeFailed = false

        do {
            try FileManager.default.createDirectory(
                at: rootURL,
                withIntermediateDirectories: true
            )
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            let data = try encoder.encode(snapshot)
            // `.atomic` writes to a sibling temp file and uses
            // `rename(2)` to replace the target. Either the old manifest
            // stays or the new one appears; there is never a window with
            // neither file on disk.
            try data.write(to: manifestURL, options: .atomic)
        } catch {
            Log.agent.error(
                "SnapshotLedger manifest persist failed: \(String(describing: error))"
            )
            writeFailed = true
        }

        manifestWriteCondition.lock()
        manifestWritesInFlight -= 1
        manifestWriteCondition.broadcast()
        manifestWriteCondition.unlock()

        if writeFailed {
            // Mark dirty again so the next operation reschedules a
            // persist. Pure opportunistic retry; no backoff needed for a
            // local filesystem.
            lock.lock()
            manifestDirty = true
            lock.unlock()
        }
    }

    /// Authoritative on-disk manifest URL.
    private var manifestURL: URL {
        rootURL.appendingPathComponent("manifest.json")
    }

    // MARK: - File path derivation

    /// Canonical on-disk URL for a snapshot file, read straight from the
    /// descriptor's persisted `fileRelativePath` rather than recomputing
    /// the shard layout. The path is stamped once at write time (via
    /// `PersistedSnapshotDescriptor.relativeFilePath`) and carried in the
    /// manifest, so the `EvictedResident.fileURLs` the ledger hands back
    /// always name the files the store wrote — and the sharding rule has
    /// a single source of truth instead of a recomputation that could
    /// silently diverge from the stored field.
    private func fileURL(for descriptor: PersistedSnapshotDescriptor) -> URL {
        rootURL.appendingPathComponent(descriptor.fileRelativePath)
    }

    /// Every segment file of the descriptor's chain, shallow→deep, own
    /// file last — the order `loadSync` composes in and the set a
    /// removal deletes. Must not require the lock (pure derivation from
    /// the descriptor value).
    private func chainFileURLs(for descriptor: PersistedSnapshotDescriptor) -> [URL] {
        descriptor.chainFileRelativePaths.map(rootURL.appendingPathComponent)
    }

    /// Read-path accessor for `loadSync`: the resident's ordered chain
    /// file URLs (base-first, own file last — the order hydration
    /// composes in). `nil` when the ID is not resident (evicted between
    /// the lookup and the hydration).
    func chainForHydration(id: String) -> [URL]? {
        lock.lock()
        defer { lock.unlock() }
        guard let descriptor = manifest.snapshots[id] else { return nil }
        return chainFileURLs(for: descriptor)
    }

    /// Read-path accessor for `loadSyncPrefix` (**Chain-Prefix Restore**,
    /// ADR-0012): the leading inherited segments of `ownerID`'s chain
    /// covering `[0..boundaryOffset]`, in compose order. Restore points
    /// sit only on the segment grid — `boundaryOffset` must be exactly a
    /// consumed leaf's capture offset, so the last leading segment must
    /// end at it. `nil` when the owner left the manifest (evicted between
    /// the lookup and the hydration) or the boundary is off the grid
    /// (a stale point against a re-shaped chain) — both degrade to a
    /// clean miss, never a wrong-extent compose.
    func chainPrefixForHydration(ownerID: String, boundaryOffset: Int) -> [URL]? {
        lock.lock()
        defer { lock.unlock() }
        guard let descriptor = manifest.snapshots[ownerID] else { return nil }
        let leading = descriptor.inheritedSegments.prefix { $0.tokenOffset <= boundaryOffset }
        guard leading.last?.tokenOffset == boundaryOffset else { return nil }
        return leading.map { rootURL.appendingPathComponent($0.fileRelativePath) }
    }
}

// MARK: - Warm start (manifest load + corrupt-manifest rebuild)

extension SnapshotLedger {

    /// Read `manifest.json` from disk, validate each partition's
    /// `modelFingerprint` against the loaded model, and seed the
    /// in-memory manifest + `currentSSDBytes` with only the valid
    /// subset. On a corrupt `manifest.json` the corrupt file is archived
    /// and the manifest is rebuilt from the on-disk snapshot headers. On
    /// a schema mismatch the manifest is backed up and the tier starts
    /// fresh. Invalidated partitions get their directories asynchronously
    /// deleted.
    ///
    /// Called once at model load via `SSDSnapshotStore.warmStartLoad`.
    /// `nonisolated` so the manager invokes it without a hop — the work
    /// is synchronous file I/O plus a lock-protected manifest swap.
    nonisolated func seedFromWarmStart(
        expectedFingerprint: String
    ) -> WarmStartOutcome {
        let manifestURL = self.manifestURL

        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            return .empty
        }

        let data: Data
        do {
            data = try Data(contentsOf: manifestURL)
        } catch {
            Log.agent.error(
                "SnapshotLedger.seedFromWarmStart read failed: \(String(describing: error))"
            )
            return .empty
        }

        let loaded: SnapshotManifest
        do {
            loaded = try JSONDecoder().decode(SnapshotManifest.self, from: data)
        } catch {
            Log.agent.error(
                "SnapshotLedger.seedFromWarmStart decode failed: \(String(describing: error))"
            )
            // Rename the corrupt manifest for forensics, then rebuild
            // from the on-disk snapshot headers. Each file's header
            // carries the full `PersistedSnapshotDescriptor`, and
            // `partitions/{digest}/_meta.json` carries the partition's
            // fingerprint, so the rebuild reconstructs everything the
            // normal path would produce.
            let ts = Int(Date().timeIntervalSince1970)
            let corruptURL = rootURL.appendingPathComponent("manifest.corrupt.\(ts).json")
            try? FileManager.default.moveItem(at: manifestURL, to: corruptURL)
            return rebuildManifestFromDirectoryWalk(
                expectedFingerprint: expectedFingerprint
            )
        }

        guard loaded.isSchemaCompatible else {
            Log.agent.info(
                "SnapshotLedger.seedFromWarmStart schema mismatch "
                    + "loaded=\(loaded.schemaVersion) "
                    + "current=\(SnapshotManifestSchema.currentVersion); starting fresh"
            )
            let backupURL = rootURL.appendingPathComponent(
                "manifest.v\(loaded.schemaVersion).bak"
            )
            try? FileManager.default.removeItem(at: backupURL)
            if FileManager.default.fileExists(atPath: manifestURL.path) {
                try? FileManager.default.moveItem(at: manifestURL, to: backupURL)
            }
            let partitionsDir = rootURL.appendingPathComponent("partitions")
            if FileManager.default.fileExists(atPath: partitionsDir.path) {
                Task.detached {
                    try? FileManager.default.removeItem(at: partitionsDir)
                }
            }
            return .empty
        }

        return commitRestoredManifest(
            loaded,
            expectedFingerprint: expectedFingerprint,
            persistManifestAfter: false,
            source: "manifest.json"
        )
    }

    /// Rebuild the manifest by walking `partitions/*/` after a corrupt
    /// `manifest.json`. Reads `_meta.json` per partition for the
    /// fingerprint, then parses each `.safetensors` file's header
    /// (sidestepping the tensor payload) to recover every descriptor
    /// field needed for the radix tree + LRU budget.
    ///
    /// Files whose header cannot be decoded are deleted so the next
    /// admission can reuse their name space. The resulting manifest is
    /// persisted via the debounced write path so subsequent restarts do
    /// not retrace the walk.
    private nonisolated func rebuildManifestFromDirectoryWalk(
        expectedFingerprint: String
    ) -> WarmStartOutcome {
        let partitionsDir = rootURL.appendingPathComponent("partitions")
        guard FileManager.default.fileExists(atPath: partitionsDir.path) else {
            Log.agent.info(
                "SnapshotLedger rebuild: no `partitions/` directory, starting fresh"
            )
            return .empty
        }

        let partitionNames: [String]
        do {
            partitionNames = try FileManager.default.contentsOfDirectory(
                atPath: partitionsDir.path
            )
        } catch {
            Log.agent.error(
                "SnapshotLedger rebuild: partitions/ listing failed: \(String(describing: error))"
            )
            return .empty
        }

        var rebuilt = SnapshotManifest.empty()
        var orphanedFiles: [URL] = []
        /// Every `.safetensors` file seen, keyed by root-relative path
        /// — including files whose header failed to decode, so chain
        /// integrity checks see them as present and a head referencing
        /// one survives the rebuild (its hydration fails lazily and
        /// cleans up the whole chain then).
        var walkedFiles: [String: URL] = [:]
        /// Files whose embedded descriptor decoded cleanly.
        var candidates: [PersistedSnapshotDescriptor] = []

        for digest in partitionNames {
            let partitionDir = partitionsDir.appendingPathComponent(digest)
            let metaURL = partitionDir.appendingPathComponent("_meta.json")
            guard let metaData = try? Data(contentsOf: metaURL),
                let meta = try? JSONDecoder().decode(PartitionMeta.self, from: metaData)
            else {
                Log.agent.error(
                    "SnapshotLedger rebuild: partition \(digest) missing or "
                        + "unreadable _meta.json"
                )
                // Skip the partition entirely — the normal fingerprint
                // mismatch path in `commitRestoredManifest` handles
                // directory cleanup for digests not in `rebuilt.partitions`.
                continue
            }
            rebuilt.partitions[digest] = meta

            let snapshotsDir = partitionDir.appendingPathComponent("snapshots")
            guard
                let shardNames = try? FileManager.default.contentsOfDirectory(
                    atPath: snapshotsDir.path
                )
            else { continue }

            for shard in shardNames {
                let shardDir = snapshotsDir.appendingPathComponent(shard)
                guard
                    let fileNames = try? FileManager.default.contentsOfDirectory(
                        atPath: shardDir.path
                    )
                else { continue }
                for name in fileNames where name.hasSuffix(".safetensors") {
                    let fileURL = shardDir.appendingPathComponent(name)
                    let relativePath = "partitions/\(digest)/snapshots/\(shard)/\(name)"
                    walkedFiles[relativePath] = fileURL
                    guard let descriptor = extractDescriptorFromFile(fileURL),
                        descriptor.partitionDigest == digest
                    else { continue }
                    candidates.append(descriptor)
                }
            }
        }

        // Chain-head resolution. A file's embedded descriptor carries the
        // full chain known at write time and the chain below a segment
        // never changes, so the deepest descriptor of each chain — the
        // **head**: one no other candidate lists as inherited — describes
        // the whole chain accurately. A crash between an extension's file
        // write and the manifest persist leaves both the base's and the
        // extension's descriptors on disk; the base loses its entry here
        // (its file survives as the extension's inherited segment), which
        // is exactly the post-crash state the commit-time fold would have
        // produced.
        let inheritedPaths = Set(
            candidates.flatMap { $0.inheritedSegments.map(\.fileRelativePath) }
        )
        var keptPaths = Set<String>()
        for descriptor in candidates
        where !inheritedPaths.contains(descriptor.fileRelativePath) {
            guard
                descriptor.inheritedSegments.allSatisfy({
                    walkedFiles[$0.fileRelativePath] != nil
                })
            else {
                // Broken chain — a missing inherited file means the head
                // can never compose. Drop the entry; its own file falls
                // out via the keep-set sweep below.
                continue
            }
            rebuilt.snapshots[descriptor.snapshotID] = descriptor
            keptPaths.formUnion(descriptor.chainFileRelativePaths)
        }
        for (relativePath, url) in walkedFiles where !keptPaths.contains(relativePath) {
            orphanedFiles.append(url)
        }

        Log.agent.info(
            "SnapshotLedger rebuild: recovered "
                + "partitions=\(rebuilt.partitions.count) "
                + "descriptors=\(rebuilt.snapshots.count) "
                + "orphanedFiles=\(orphanedFiles.count)"
        )

        // Delete orphaned files off the hot path.
        if !orphanedFiles.isEmpty {
            Task.detached {
                for url in orphanedFiles {
                    try? FileManager.default.removeItem(at: url)
                }
            }
        }

        return commitRestoredManifest(
            rebuilt,
            expectedFingerprint: expectedFingerprint,
            persistManifestAfter: true,
            source: "rebuild"
        )
    }

    /// Shared commit step for both the normal JSON path and the
    /// directory-walk rebuild. Applies the fingerprint filter, drops
    /// descriptors whose wire-format checkpoint type no longer
    /// round-trips (so their bytes do not leak into the SSD budget),
    /// seeds `currentSSDBytes`, schedules async cleanup of invalidated
    /// partition directories + dead-descriptor files, and optionally
    /// kicks the debounced persist (used by the rebuild path to
    /// overwrite the just-renamed corrupt `manifest.json`).
    private nonisolated func commitRestoredManifest(
        _ loaded: SnapshotManifest,
        expectedFingerprint: String,
        persistManifestAfter: Bool,
        source: String
    ) -> WarmStartOutcome {
        var restored = SnapshotManifest.empty()
        var invalidated: [WarmStartOutcome.InvalidatedPartition] = []
        let now = Date().timeIntervalSinceReferenceDate

        // Chain-total bytes per partition, so each reclaim can report
        // what it returned to the budget.
        var bytesByPartition: [String: Int] = [:]
        for descriptor in loaded.snapshots.values {
            bytesByPartition[descriptor.partitionDigest, default: 0] += descriptor.totalBytes
        }
        func invalidate(
            _ digest: String,
            _ meta: PartitionMeta,
            _ reason: WarmStartOutcome.PartitionInvalidationReason
        ) {
            invalidated.append(
                WarmStartOutcome.InvalidatedPartition(
                    digest: digest,
                    modelID: meta.modelID,
                    bytes: bytesByPartition[digest] ?? 0,
                    reason: reason
                ))
        }

        // True when a legacy meta was grace-stamped below — the stamp
        // must reach disk this session or the GC clock never starts.
        var mutatedRestoredMeta = false

        // Stale-partition GC (PRD #150): staleness is measured against
        // the freshest valid partition's use stamp, not the wall clock
        // — an idle week must not reclaim the whole tier (see
        // `SSDStalePartitionPolicy`). Legacy metas without a stamp are
        // treated as fresh here and grace-stamped below.
        // Anchor on *real* stamps only: a legacy nil-stamped partition
        // is about to be grace-stamped, and letting it count as "used
        // now" would inflate the anchor and reclaim a genuinely-stamped
        // sibling at the migration launch — the wall-clock regression
        // the relative rule exists to prevent. No stamps anywhere → no
        // anchor → nothing reclaimed this launch.
        let tierMostRecentUse = loaded.partitions.values
            .filter {
                $0.schemaVersion == SnapshotManifestSchema.currentVersion
                    && $0.modelFingerprint == expectedFingerprint
            }
            .compactMap(\.lastUsedAt)
            .max()

        for (digest, meta) in loaded.partitions {
            // Stale `PartitionMeta` inside a current-version manifest
            // signals a hand-edited or partially upgraded file — drop it
            // rather than reattach under stale canonicalization.
            guard meta.schemaVersion == SnapshotManifestSchema.currentVersion else {
                invalidate(digest, meta, .schemaStale)
                continue
            }
            guard meta.modelFingerprint == expectedFingerprint else {
                invalidate(digest, meta, .fingerprintChanged)
                continue
            }
            // A legacy meta without a stamp is grace-stamped to "now" —
            // the clock starts here, it does not retroactively reclaim
            // long-lived caches.
            if let lastUsed = meta.lastUsedAt {
                if let anchor = tierMostRecentUse,
                    anchor - lastUsed > SSDStalePartitionPolicy.maxUnusedAge
                {
                    invalidate(digest, meta, .staleUnused)
                    continue
                }
                restored.partitions[digest] = meta
            } else {
                var graced = meta
                graced.lastUsedAt = now
                restored.partitions[digest] = graced
                mutatedRestoredMeta = true
            }
        }

        var descriptorsByDigest: [String: [PersistedSnapshotDescriptor]] = [:]
        var deadDescriptorFiles: [URL] = []
        for (id, desc) in loaded.snapshots {
            guard restored.partitions[desc.partitionDigest] != nil else { continue }
            // Same rationale as the `PartitionMeta` filter above.
            guard desc.schemaVersion == SnapshotManifestSchema.currentVersion else {
                deadDescriptorFiles.append(
                    contentsOf: chainFileURLs(for: desc)
                )
                continue
            }
            // Drop descriptors whose wire-format checkpoint type no
            // longer decodes — `PrefixCacheManager.warmStart` would skip
            // them silently otherwise, leaving their bytes stranded in
            // `currentSSDBytes`.
            guard
                HybridCacheSnapshot.CheckpointType(
                    wireString: desc.checkpointType
                ) != nil
            else {
                deadDescriptorFiles.append(
                    contentsOf: chainFileURLs(for: desc)
                )
                continue
            }
            restored.snapshots[id] = desc
            descriptorsByDigest[desc.partitionDigest, default: []].append(desc)
        }

        let seedBytes = restored.snapshots.values.reduce(0) { $0 + $1.totalBytes }

        // Reclaims and grace stamps must reach disk even on the normal
        // manifest-load path (which otherwise defers persistence to the
        // next mutation): an unpersisted grace stamp restarts the GC
        // clock every launch, and an unpersisted reclaim resurfaces the
        // partition's dangling descriptors on the next read.
        let persistAfter =
            persistManifestAfter || mutatedRestoredMeta || !invalidated.isEmpty

        lock.lock()
        self.manifest = restored
        self.currentSSDBytes = seedBytes
        if persistAfter {
            self.manifestDirty = true
            scheduleManifestPersistLocked()
        }
        lock.unlock()

        let validPartitions: [WarmStartOutcome.Partition] = restored.partitions.map {
            digest, meta in
            WarmStartOutcome.Partition(
                digest: digest,
                meta: meta,
                descriptors: (descriptorsByDigest[digest] ?? [])
                    .sorted { $0.snapshotID < $1.snapshotID }
            )
        }
        .sorted { $0.digest < $1.digest }

        if !invalidated.isEmpty {
            let capturedRoot = rootURL
            let capturedDigests = invalidated.map(\.digest)
            Task.detached {
                for digest in capturedDigests {
                    let dir =
                        capturedRoot
                        .appendingPathComponent("partitions")
                        .appendingPathComponent(digest)
                    try? FileManager.default.removeItem(at: dir)
                }
            }
        }

        if !deadDescriptorFiles.isEmpty {
            let urls = deadDescriptorFiles
            Task.detached {
                for url in urls {
                    try? FileManager.default.removeItem(at: url)
                }
            }
        }

        Log.agent.info(
            "SnapshotLedger.seedFromWarmStart source=\(source) "
                + "partitions=\(restored.partitions.count) "
                + "snapshots=\(restored.snapshots.count) "
                + "bytes=\(seedBytes) "
                + "invalidated=\(invalidated.count) "
                + "dead=\(deadDescriptorFiles.count)"
        )

        return WarmStartOutcome(
            validPartitions: validPartitions,
            invalidated: invalidated.sorted { $0.digest < $1.digest }
        )
    }

    /// Read only the header of a placeholder container file and return
    /// the embedded descriptor, applying the schema-version gate. Used
    /// by the rebuild path — skips the tensor payload via the neutral
    /// `PlaceholderContainerHeader.readHeaderOnly`. Returns `nil` on any
    /// read/decode failure or stale schema; the caller deletes the file.
    private nonisolated func extractDescriptorFromFile(
        _ url: URL
    ) -> PersistedSnapshotDescriptor? {
        guard let header = PlaceholderContainerHeader.readHeaderOnly(from: url) else {
            return nil
        }
        // Stale-schema files cannot be reattached safely; treat as
        // orphaned and let the caller delete.
        guard header.schemaVersion == SnapshotManifestSchema.currentVersion,
            header.descriptor.schemaVersion == SnapshotManifestSchema.currentVersion
        else {
            return nil
        }
        return header.descriptor
    }
}

// MARK: - Testing hooks

extension SnapshotLedger {

    /// Synchronous accessor for the current SSD byte count.
    nonisolated func currentSSDBytesForTesting() -> Int {
        lock.lock()
        defer { lock.unlock() }
        return currentSSDBytes
    }

    /// Manifest descriptor IDs, sorted by `lastAccessAt` ascending —
    /// the LRU recency order an eviction cut would consume.
    nonisolated func residentIDsByRecencyForTesting() -> [String] {
        lock.lock()
        defer { lock.unlock() }
        return manifest.snapshots.values
            .sorted { ($0.lastAccessAt, $0.snapshotID) < ($1.lastAccessAt, $1.snapshotID) }
            .map(\.snapshotID)
    }

    /// Inject a descriptor into the manifest without going through the
    /// writer loop. The partition must already be registered so the
    /// manifest invariant holds; the seeded descriptor updates
    /// `currentSSDBytes`.
    nonisolated func seedDescriptorForTesting(_ descriptor: PersistedSnapshotDescriptor) {
        lock.lock()
        defer { lock.unlock() }
        precondition(
            manifest.partitions[descriptor.partitionDigest] != nil,
            "seedDescriptorForTesting: partition not registered"
        )
        if let previous = manifest.snapshots[descriptor.snapshotID] {
            currentSSDBytes -= previous.totalBytes
        }
        manifest.snapshots[descriptor.snapshotID] = descriptor
        currentSSDBytes += descriptor.totalBytes
    }

    /// Read a resident descriptor without mutating anything. Used by
    /// the extension-transfer tests to assert the commit-time fold.
    nonisolated func residentDescriptorForTesting(id: String) -> PersistedSnapshotDescriptor? {
        lock.lock()
        defer { lock.unlock() }
        return manifest.snapshots[id]
    }

    /// The currently shielded extension bases. Tests assert the shield
    /// is released on every terminal writer path.
    nonisolated func transferringBaseIDsForTesting() -> Set<String> {
        lock.lock()
        defer { lock.unlock() }
        return transferringBaseIDs
    }

    /// Read a descriptor's current `lastAccessAt` without mutating
    /// anything. Used by the `recordHit` regression test.
    nonisolated func lastAccessAtForTesting(id: String) -> Double {
        lock.lock()
        defer { lock.unlock() }
        return manifest.snapshots[id]?.lastAccessAt ?? -1
    }

    /// Read a partition's in-memory meta without mutating anything.
    /// Used by the stale-partition GC tests to assert `lastUsedAt`
    /// stamping (register merge, hit bump, warm-start grace).
    nonisolated func partitionMetaForTesting(digest: String) -> PartitionMeta? {
        lock.lock()
        defer { lock.unlock() }
        return manifest.partitions[digest]
    }
}
