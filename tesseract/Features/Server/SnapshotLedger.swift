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

    let validPartitions: [Partition]
    let invalidatedPartitionDigests: [String]

    static let empty = WarmStartOutcome(
        validPartitions: [],
        invalidatedPartitionDigests: []
    )
}

// MARK: - Evicted resident

/// A committed resident the ledger removed from the in-memory manifest
/// but whose on-disk file has not yet been deleted and whose `onDrop`
/// callback has not yet fired. The store finalizes these effects
/// (file delete + callback + diagnostics) *outside* any lock. Carrying
/// the `fileURL` here lets the store delete without recomputing the
/// sharded path.
nonisolated struct EvictedResident: Sendable {
    let snapshotID: String
    let fileURL: URL
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

    /// Hard SSD byte budget the type-protected LRU cut enforces.
    let budgetBytes: Int

    /// Minimum idle time before the in-memory manifest is persisted.
    /// Injected so tests can shorten it; production uses 500 ms.
    private let manifestDebounce: Duration

    // MARK: - Lock-protected mutable state

    private let lock = NSLock()
    private var manifest: SnapshotManifest = .empty()
    private var currentSSDBytes: Int = 0
    private var manifestDirty: Bool = false
    private var manifestPersistTask: Task<Void, Never>?
    /// Snapshot IDs the store deleted while a write may already be in
    /// flight. `consumeTombstone` (the writer's pre-write skip) and
    /// `commit` (the self-veto) both check this set so a superseded
    /// leaf never lands in the manifest after its node was cleaned up.
    private var deletedInFlightSnapshotIDs: Set<String> = []

    init(rootURL: URL, budgetBytes: Int, manifestDebounce: Duration) {
        self.rootURL = rootURL
        self.budgetBytes = budgetBytes
        self.manifestDebounce = manifestDebounce
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
    /// `createdAt` / `lastAccessAt` are stamped to now; `commit` re-stamps
    /// `lastAccessAt` when the write lands so the first LRU cut sees fresh
    /// recency. `nonisolated static` — pure construction, no ledger state,
    /// callable from the MainActor front door without a hop.
    nonisolated static func makeDescriptor(
        partitionKey: CachePartitionKey,
        pathFromRoot: [Int],
        snapshot: HybridCacheSnapshot,
        payloadBytes: Int
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
            createdAt: now,
            lastAccessAt: now,
            fileRelativePath: PersistedSnapshotDescriptor.relativeFilePath(
                snapshotID: snapshotID,
                partitionDigest: partitionDigest
            ),
            schemaVersion: SnapshotManifestSchema.currentVersion
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
    func registerPartition(_ meta: PartitionMeta, digest: String) {
        lock.lock()
        if manifest.partitions[digest] == meta {
            lock.unlock()
            return
        }
        manifest.partitions[digest] = meta
        manifestDirty = true
        scheduleManifestPersistLocked()
        lock.unlock()

        // Persist a per-partition `_meta.json` sidecar so the
        // directory-walk rebuild can validate partition fingerprints
        // without relying on the top-level `manifest.json`. The file is
        // idempotent — repeat writes for the same digest overwrite cleanly.
        writePartitionMetaFile(meta, digest: digest)
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
        let dir = rootURL
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
        descriptor.lastAccessAt = Date().timeIntervalSinceReferenceDate
        manifest.snapshots[id] = descriptor
        scheduleManifestPersistLocked()
    }

    // MARK: - Admission cut (type-protected LRU)

    /// Type-protected LRU cut, mirroring the asymmetric protection rule:
    ///
    /// 1. Evict oldest non-`.system` residents one by one until the
    ///    incoming entry fits.
    /// 2. If non-system is exhausted and budget is still too tight:
    ///    - `.system` incoming → fall through to evict oldest
    ///      `.system` residents (lateral move; protection preserved
    ///      across the set).
    ///    - non-system incoming → drop the incoming. Never evict
    ///      `.system` residents to make room for lower-value types.
    ///
    /// Returns the admit/drop decision plus the list of residents that
    /// were removed from the in-memory manifest, atomically under the
    /// ledger lock so a concurrent `recordHit` or hydration-failure
    /// `remove` cannot tear a multi-candidate cut. The store deletes the
    /// evicted files and fires `onDrop(.evictedByLRU)` **outside** the
    /// lock.
    func admit(
        _ descriptor: PersistedSnapshotDescriptor
    ) -> (decision: AdmissionDecision, evicted: [EvictedResident]) {
        var evicted: [EvictedResident] = []

        lock.lock()
        defer { lock.unlock() }

        let spaceNeeded = descriptor.bytes
        if currentSSDBytes + spaceNeeded <= budgetBytes {
            return (.admit, evicted)
        }

        // Unrecognized wire strings are treated as non-system so they
        // participate in normal LRU eviction and never bypass system
        // protection. Front-door validation rejects these already, so
        // this branch is only reached in tests.
        let incomingType = HybridCacheSnapshot.CheckpointType(
            wireString: descriptor.checkpointType
        ) ?? .leaf

        // Pass 1 — evict oldest non-system residents under the lock.
        evictOldestUnderLock(
            matching: { $0 != .system },
            until: spaceNeeded,
            into: &evicted
        )

        if currentSSDBytes + spaceNeeded <= budgetBytes {
            return (.admit, evicted)
        }

        // Pass 2 — non-system eligible set exhausted.
        switch incomingType {
        case .system:
            // Lateral move: evict oldest system residents until fit.
            evictOldestUnderLock(
                matching: { $0 == .system },
                until: spaceNeeded,
                into: &evicted
            )
            if currentSSDBytes + spaceNeeded <= budgetBytes {
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

        let incomingType = HybridCacheSnapshot.CheckpointType(
            wireString: descriptor.checkpointType
        ) ?? .leaf
        let predicate: (HybridCacheSnapshot.CheckpointType) -> Bool =
            incomingType == .system ? { _ in true } : { $0 != .system }

        guard let victim = sortedEligibleResidentsLocked(matching: predicate).first else {
            return nil
        }
        return removeResidentUnderLock(snapshotID: victim.snapshotID)
    }

    /// Walk the manifest's snapshots in ascending `lastAccessAt` order,
    /// evicting each that matches the predicate and freeing its SSD
    /// bytes until the requested room is available (or the eligible set
    /// is exhausted). Every removed resident is appended to `evicted`.
    /// Must be called with `lock` held.
    private func evictOldestUnderLock(
        matching predicate: (HybridCacheSnapshot.CheckpointType) -> Bool,
        until spaceNeeded: Int,
        into evicted: inout [EvictedResident]
    ) {
        for victim in sortedEligibleResidentsLocked(matching: predicate) {
            if currentSSDBytes + spaceNeeded <= budgetBytes {
                return
            }
            if let resident = removeResidentUnderLock(snapshotID: victim.snapshotID) {
                evicted.append(resident)
            }
        }
    }

    /// The subset of manifest descriptors whose parsed checkpoint type
    /// satisfies `predicate`, sorted by `lastAccessAt` ascending (oldest
    /// first). Descriptors whose wire checkpoint type fails to parse are
    /// dropped — they shouldn't exist in practice and are not eviction
    /// candidates. Must be called with `lock` held.
    private func sortedEligibleResidentsLocked(
        matching predicate: (HybridCacheSnapshot.CheckpointType) -> Bool
    ) -> [PersistedSnapshotDescriptor] {
        manifest.snapshots.values
            .filter { descriptor in
                guard
                    let checkpointType = HybridCacheSnapshot.CheckpointType(
                        wireString: descriptor.checkpointType
                    )
                else { return false }
                return predicate(checkpointType)
            }
            .sorted { ($0.lastAccessAt, $0.snapshotID) < ($1.lastAccessAt, $1.snapshotID) }
    }

    /// Drop the manifest entry, decrement the SSD byte count, and return
    /// an `EvictedResident` carrying the file URL so the store can
    /// delete it outside the lock. Returns `nil` when the snapshotID is
    /// not in the manifest. Schedules the debounced persist so
    /// eviction-only paths still write the updated manifest without
    /// waiting for an unrelated subsequent mutation. Must be called with
    /// `lock` held.
    private func removeResidentUnderLock(snapshotID: String) -> EvictedResident? {
        guard let descriptor = manifest.snapshots.removeValue(forKey: snapshotID) else {
            return nil
        }
        currentSSDBytes -= descriptor.bytes
        manifestDirty = true
        scheduleManifestPersistLocked()
        return EvictedResident(
            snapshotID: descriptor.snapshotID,
            fileURL: fileURL(for: descriptor)
        )
    }

    // MARK: - Commit + in-flight-delete tombstone protocol

    /// Insert a freshly written descriptor as a committed resident.
    /// Returns `false` — vetoing the commit — when a prior
    /// `removeOrTombstone` tombstoned this ID while the write was in
    /// flight; the store then deletes the orphaned file. On success the
    /// entry's `lastAccessAt` is stamped to commit time so the first LRU
    /// cut sees fresh recency.
    func commit(_ descriptor: PersistedSnapshotDescriptor) -> Bool {
        lock.lock()
        defer { lock.unlock() }

        if deletedInFlightSnapshotIDs.remove(descriptor.snapshotID) != nil {
            return false
        }

        var fresh = descriptor
        fresh.lastAccessAt = Date().timeIntervalSinceReferenceDate

        // Idempotent byte accounting: if this ID is already resident,
        // subtract its prior bytes before adding the fresh entry's, so a
        // re-commit replaces rather than double-counts (mirrors
        // `seedDescriptorForTesting`). The sole production caller commits
        // each fresh UUID once, so this is defensive against future
        // re-commit paths, not a live correction.
        if let prior = manifest.snapshots[descriptor.snapshotID] {
            currentSSDBytes -= prior.bytes
        }
        manifest.snapshots[descriptor.snapshotID] = fresh
        currentSSDBytes += descriptor.bytes
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
            return
        }
        let snapshot = manifest
        manifestDirty = false
        lock.unlock()

        let manifestURL = self.manifestURL

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
    /// manifest, so an `EvictedResident.fileURL` the ledger hands back
    /// always names the file the store wrote — and the sharding rule has
    /// a single source of truth instead of a recomputation that could
    /// silently diverge from the stored field.
    private func fileURL(for descriptor: PersistedSnapshotDescriptor) -> URL {
        rootURL.appendingPathComponent(descriptor.fileRelativePath)
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
        var recoveredDescriptors = 0
        var orphanedFiles: [URL] = []

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
            guard let shardNames = try? FileManager.default.contentsOfDirectory(
                atPath: snapshotsDir.path
            ) else { continue }

            for shard in shardNames {
                let shardDir = snapshotsDir.appendingPathComponent(shard)
                guard let fileNames = try? FileManager.default.contentsOfDirectory(
                    atPath: shardDir.path
                ) else { continue }
                for name in fileNames where name.hasSuffix(".safetensors") {
                    let fileURL = shardDir.appendingPathComponent(name)
                    guard let descriptor = extractDescriptorFromFile(fileURL),
                          descriptor.partitionDigest == digest
                    else {
                        orphanedFiles.append(fileURL)
                        continue
                    }
                    rebuilt.snapshots[descriptor.snapshotID] = descriptor
                    recoveredDescriptors += 1
                }
            }
        }

        Log.agent.info(
            "SnapshotLedger rebuild: recovered "
            + "partitions=\(rebuilt.partitions.count) "
            + "descriptors=\(recoveredDescriptors) "
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
        var invalidatedDigests: [String] = []
        for (digest, meta) in loaded.partitions {
            // Stale `PartitionMeta` inside a current-version manifest
            // signals a hand-edited or partially upgraded file — drop it
            // rather than reattach under stale canonicalization.
            guard meta.schemaVersion == SnapshotManifestSchema.currentVersion else {
                invalidatedDigests.append(digest)
                continue
            }
            if meta.modelFingerprint == expectedFingerprint {
                restored.partitions[digest] = meta
            } else {
                invalidatedDigests.append(digest)
            }
        }

        var descriptorsByDigest: [String: [PersistedSnapshotDescriptor]] = [:]
        var deadDescriptorFiles: [URL] = []
        for (id, desc) in loaded.snapshots {
            guard restored.partitions[desc.partitionDigest] != nil else { continue }
            // Same rationale as the `PartitionMeta` filter above.
            guard desc.schemaVersion == SnapshotManifestSchema.currentVersion else {
                deadDescriptorFiles.append(
                    fileURL(for: desc)
                )
                continue
            }
            // Drop descriptors whose wire-format checkpoint type no
            // longer decodes — `PrefixCacheManager.warmStart` would skip
            // them silently otherwise, leaving their bytes stranded in
            // `currentSSDBytes`.
            guard HybridCacheSnapshot.CheckpointType(
                wireString: desc.checkpointType
            ) != nil else {
                deadDescriptorFiles.append(
                    fileURL(for: desc)
                )
                continue
            }
            restored.snapshots[id] = desc
            descriptorsByDigest[desc.partitionDigest, default: []].append(desc)
        }

        let seedBytes = restored.snapshots.values.reduce(0) { $0 + $1.bytes }

        lock.lock()
        self.manifest = restored
        self.currentSSDBytes = seedBytes
        if persistManifestAfter {
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

        if !invalidatedDigests.isEmpty {
            let capturedRoot = rootURL
            let capturedDigests = invalidatedDigests
            Task.detached {
                for digest in capturedDigests {
                    let dir = capturedRoot
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
            + "invalidated=\(invalidatedDigests.count) "
            + "dead=\(deadDescriptorFiles.count)"
        )

        return WarmStartOutcome(
            validPartitions: validPartitions,
            invalidatedPartitionDigests: invalidatedDigests
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
            currentSSDBytes -= previous.bytes
        }
        manifest.snapshots[descriptor.snapshotID] = descriptor
        currentSSDBytes += descriptor.bytes
    }

    /// Read a descriptor's current `lastAccessAt` without mutating
    /// anything. Used by the `recordHit` regression test.
    nonisolated func lastAccessAtForTesting(id: String) -> Double {
        lock.lock()
        defer { lock.unlock() }
        return manifest.snapshots[id]?.lastAccessAt ?? -1
    }
}
