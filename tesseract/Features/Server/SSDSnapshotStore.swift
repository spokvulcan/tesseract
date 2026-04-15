//
//  SSDSnapshotStore.swift
//  tesseract
//
//  Writer-side skeleton for the SSD prefix-cache tier. Owns the
//  front-door admission queue, the detached writer task that drains
//  it, the in-memory manifest + debounced persist, and the
//  admission-time type-protected LRU cut. Read-side (warm start +
//  hydration via `loadSync`) is a separate downstream concern and
//  deliberately NOT part of this file yet.
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
//  **Placeholder on-disk format.** This file writes a simple
//  length-prefixed binary container (8-byte LE header length + JSON
//  header + concatenated tensor blobs). The real safetensors format
//  comes in a downstream task that ships `HybridCacheSnapshot`
//  `serialize(to:metadata:)` / `deserialize(from:)` wrappers; at
//  that point this writer switches over. The header JSON carries a
//  `"format_kind": "tesseract-cache-v1"` marker so the downstream
//  reader can refuse placeholder files cleanly.
//

import Foundation
import MLX
import MLXLMCommon

// MARK: - Front door outcome types

/// Result of attempting to enqueue a payload into the SSD writer's
/// pending queue. Synchronous so the MainActor call sites can branch
/// without `await`.
nonisolated enum TryEnqueueResult: Sendable, Equatable {
    /// Item accepted into the pending queue. The returned
    /// `SnapshotStorageRef` carries `committed: false`; downstream
    /// composition code attaches it to the radix node.
    case accepted(SnapshotStorageRef)

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
}

/// Why a pending or resident item was dropped. Fed into the
/// `onDrop` callback so downstream composition code can clear the
/// radix node's `storageRef` and log diagnostics.
nonisolated enum SSDDropReason: Sendable, Equatable {
    /// Front door dropped an older pending entry to fit the new
    /// enqueue under `maxPendingBytes`.
    case backpressureOldest

    /// Previously committed resident was evicted by a later
    /// admission's type-protected LRU cut (or by a disk-full
    /// retry) to make room for an incoming entry. The file and
    /// the manifest entry are already gone by the time this fires
    /// — consumers must clear their `storageRef` so subsequent
    /// lookups don't return a stale SSD hit pointing at a deleted
    /// file.
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

    /// `loadSync` could not materialize a resident back into RAM
    /// (file missing, fingerprint mismatch, placeholder decode
    /// error). The descriptor was removed from the manifest and the
    /// file deleted before the callback fired.
    case hydrationFailure
}

// MARK: - Warm-start outcome

/// Partitioned view of a successfully loaded manifest. Returned by
/// `SSDSnapshotStore.warmStartLoad` so `PrefixCacheManager.warmStart`
/// can iterate the valid descriptors without taking the store's lock.
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

// MARK: - SSDSnapshotStore

nonisolated final class SSDSnapshotStore: @unchecked Sendable {

    // MARK: - Configuration (immutable after init)

    let rootURL: URL
    let budgetBytes: Int
    let maxPendingBytes: Int
    /// Minimum idle time before the in-memory manifest is persisted
    /// to disk. Injected so tests can set it to a short duration;
    /// production uses 500 ms per the plan.
    private let manifestDebounce: Duration

    // MARK: - Lock-protected mutable state

    private let lock = NSLock()
    private var pending: [PendingWrite] = []
    private var pendingBytes: Int = 0
    private var manifest: SnapshotManifest = .empty()
    private var currentSSDBytes: Int = 0
    private var manifestDirty: Bool = false
    private var manifestPersistTask: Task<Void, Never>?
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

    private let onCommit: @Sendable (String) -> Void
    private let onDrop: @Sendable (String, SSDDropReason) -> Void

    // MARK: - Public API

    init(
        config: SSDPrefixCacheConfig,
        manifestDebounce: Duration = .milliseconds(500),
        onCommit: @escaping @Sendable (String) -> Void = { _ in },
        onDrop: @escaping @Sendable (String, SSDDropReason) -> Void = { _, _ in }
    ) {
        self.rootURL = config.rootURL
        self.budgetBytes = config.budgetBytes
        self.maxPendingBytes = config.maxPendingBytes
        self.manifestDebounce = manifestDebounce
        self.onCommit = onCommit
        self.onDrop = onDrop

        let (stream, continuation) = AsyncStream<Void>.makeStream(bufferingPolicy: .bufferingNewest(1))
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
        manifestPersistTask?.cancel()
    }

    /// Synchronous front-door admission. Acquires the lock, enforces
    /// the `maxPendingBytes` byte budget via drop-oldest-pending,
    /// pushes the item onto the writer's pending queue, and wakes
    /// the writer. Never suspends and never crosses an actor
    /// boundary — safe to call from inside a MainActor closure.
    func tryEnqueue(
        payload: SnapshotPayload,
        descriptor: PersistedSnapshotDescriptor
    ) -> TryEnqueueResult {
        // Parse the wire-format checkpoint type before taking the lock;
        // no sense holding the lock for a parse that can fail.
        guard
            let checkpointType = HybridCacheSnapshot.CheckpointType(
                wireString: descriptor.checkpointType
            )
        else {
            PrefixCacheDiagnostics.logSystem(PrefixCacheDiagnostics.SSDAdmitEvent(
                id: descriptor.snapshotID,
                bytes: descriptor.bytes,
                outcome: .droppedInvalidCheckpointType
            ))
            return .rejectedInvalidCheckpointType
        }

        let payloadBytes = payload.totalBytes

        // A single payload larger than the cap cannot be queued at
        // all; no amount of back-pressure eviction can create room.
        if payloadBytes > maxPendingBytes {
            PrefixCacheDiagnostics.logSystem(PrefixCacheDiagnostics.SSDAdmitEvent(
                id: descriptor.snapshotID,
                bytes: payloadBytes,
                outcome: .droppedTooLargeForBudget
            ))
            return .rejectedTooLargeForBudget
        }

        var droppedItems: [(id: String, bytes: Int)] = []

        lock.lock()

        // Enforce the manifest invariant that every snapshots
        // entry must reference a partition in `manifest.partitions`.
        // The writer cannot autoregister because `PartitionMeta`
        // carries load-bearing fields (modelFingerprint, kvBits,
        // etc.) that only the caller knows at model-load time.
        // Reject here so the caller gets immediate feedback
        // rather than persisting a dangling manifest entry.
        guard manifest.partitions[descriptor.partitionDigest] != nil else {
            lock.unlock()
            PrefixCacheDiagnostics.logSystem(PrefixCacheDiagnostics.SSDAdmitEvent(
                id: descriptor.snapshotID,
                bytes: payloadBytes,
                outcome: .droppedUnregisteredPartition
            ))
            return .rejectedUnregisteredPartition
        }

        // Drop-oldest-pending until the new payload fits under the
        // cap. Record dropped IDs and fire the callbacks AFTER
        // releasing the lock so the callback body can take other
        // locks without risking a deadlock.
        while pendingBytes + payloadBytes > maxPendingBytes,
              let oldest = pending.first {
            pending.removeFirst()
            pendingBytes -= oldest.payload.totalBytes
            droppedItems.append(
                (id: oldest.descriptor.snapshotID, bytes: oldest.payload.totalBytes)
            )
        }

        pending.append(PendingWrite(payload: payload, descriptor: descriptor))
        pendingBytes += payloadBytes

        lock.unlock()

        for item in droppedItems {
            // Both events fire per bumped item: the admission outcome
            // (`droppedByteBudget` is the terminal verdict for that
            // earlier `tryEnqueue` call), and the lifecycle callback
            // (`storageRefDropCallback` so the radix node sees its
            // pending ref cleared).
            PrefixCacheDiagnostics.logSystem(PrefixCacheDiagnostics.SSDAdmitEvent(
                id: item.id,
                bytes: item.bytes,
                outcome: .droppedByteBudget
            ))
            PrefixCacheDiagnostics.logSystem(
                PrefixCacheDiagnostics.StorageRefDropCallbackEvent(
                    id: item.id,
                    reason: .backpressureOldest
                )
            )
            onDrop(item.id, .backpressureOldest)
        }

        wakeupContinuation.yield()

        return .accepted(
            SnapshotStorageRef(
                snapshotID: descriptor.snapshotID,
                partitionDigest: descriptor.partitionDigest,
                tokenOffset: descriptor.tokenOffset,
                checkpointType: checkpointType,
                bytesOnDisk: descriptor.bytes,
                lastAccessTime: .now,
                committed: false
            )
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
        // Register before yielding so the writer can never skip our
        // signal. New admissions landing between the wakeup and the
        // resume pop off a second drain pass — covered by the loop.
        repeat {
            await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
                registerDrainWaiter(cont)
                wakeupContinuation.yield()
            }
        } while hasPendingWrites()

        persistManifestIfDirty()
    }

    private func registerDrainWaiter(
        _ cont: CheckedContinuation<Void, Never>
    ) {
        lock.lock()
        defer { lock.unlock() }
        drainWaiters.append(cont)
    }

    private func hasPendingWrites() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return !pending.isEmpty
    }

    /// Bump the descriptor's `lastAccessAt` so the writer's
    /// admission-time LRU cut sees fresh recency data. A lookup
    /// landing on a committed ref calls this so that hot entries
    /// never look stale to the SSD-tier eviction policy. No-op when
    /// the ID is not in the manifest (the entry was evicted between
    /// the lookup and the bump).
    func recordHit(id: String) {
        lock.lock()
        defer { lock.unlock() }
        guard var descriptor = manifest.snapshots[id] else { return }
        descriptor.lastAccessAt = Date().timeIntervalSinceReferenceDate
        manifest.snapshots[id] = descriptor
        scheduleManifestPersistLocked()
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
        // directory-walk rebuild path (`rebuildManifestFromDirectoryWalk`)
        // can validate partition fingerprints without relying on the
        // top-level `manifest.json`. The file is idempotent — repeat
        // writes for the same digest overwrite cleanly.
        writePartitionMetaFile(meta, digest: digest)
    }

    /// Serialize a `PartitionMeta` to `partitions/{digest}/_meta.json`.
    /// Best-effort: failures log at error level but do not abort the
    /// caller. The only caller that depends on the file is the
    /// corrupt-manifest rebuild path, which already falls back to
    /// "invalidated partition" when the file is missing.
    private nonisolated func writePartitionMetaFile(
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
                "SSDSnapshotStore.writePartitionMetaFile failed "
                + "digest=\(digest) error=\(String(describing: error))"
            )
        }
    }

    // MARK: - Writer loop

    private func writerLoop() async {
        for await _ in wakeupStream {
            await drainPending()
            resumeDrainWaiters()
        }
        // Stream closed (deinit). Fail any straggling waiters by
        // resuming them so `flushAsync` callers do not leak.
        resumeDrainWaiters()
    }

    private func resumeDrainWaiters() {
        lock.lock()
        let waiters = drainWaiters
        drainWaiters.removeAll(keepingCapacity: false)
        lock.unlock()
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
        lock.lock()
        defer { lock.unlock() }
        guard !pending.isEmpty else { return nil }
        return pending.removeFirst()
    }

    private func processPendingItem(_ item: PendingWrite) async {
        // 1. Admission LRU cut. Returns the admit/drop decision
        //    and a list of committed residents that were evicted
        //    to make room. The lock is held only long enough to
        //    mutate the in-memory manifest — file deletion and
        //    `onDrop` callbacks happen after release.
        let (admission, evicted) = admitUnderBudget(item.descriptor)
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

        switch admission {
        case .admit:
            break
        case .drop(let reason):
            releasePendingBytes(item.payload.totalBytes)
            emitWriterDrop(
                id: item.descriptor.snapshotID,
                bytes: item.payload.totalBytes,
                reason: reason
            )
            return
        }

        // 2. Write the payload atomically (temp + fsync + rename).
        //    On disk-full, run a single eviction-retry pass — its
        //    victim is also cleaned up outside the lock. Any other
        //    error drops the incoming and fires the drop callback.
        do {
            try writePayload(item.payload, descriptor: item.descriptor)
        } catch WriteError.diskFull {
            if let retryVictim = retryAfterDiskFull(item.descriptor) {
                PrefixCacheDiagnostics.logSystem(
                    PrefixCacheDiagnostics.SSDEvictAtAdmissionEvent(
                        victimID: retryVictim.snapshotID,
                        incomingID: item.descriptor.snapshotID
                    )
                )
                finalizeEvictions([retryVictim])
                do {
                    try writePayload(item.payload, descriptor: item.descriptor)
                } catch {
                    Log.agent.error(
                        "SSD writer diskFull retry failed for \(item.descriptor.snapshotID): "
                        + "\(String(describing: error))"
                    )
                    releasePendingBytes(item.payload.totalBytes)
                    emitWriterDrop(
                        id: item.descriptor.snapshotID,
                        bytes: item.payload.totalBytes,
                        reason: .diskFull
                    )
                    return
                }
            } else {
                Log.agent.error(
                    "SSD writer diskFull and no eviction victim available for "
                    + "\(item.descriptor.snapshotID)"
                )
                releasePendingBytes(item.payload.totalBytes)
                emitWriterDrop(
                    id: item.descriptor.snapshotID,
                    bytes: item.payload.totalBytes,
                    reason: .diskFull
                )
                return
            }
        } catch {
            Log.agent.error(
                "SSD writer I/O failure for \(item.descriptor.snapshotID): "
                + "\(String(describing: error))"
            )
            releasePendingBytes(item.payload.totalBytes)
            emitWriterDrop(
                id: item.descriptor.snapshotID,
                bytes: item.payload.totalBytes,
                reason: .writerIOError
            )
            return
        }

        // 3. Write succeeded: register the descriptor, release the
        //    pending byte budget, and fire the commit callback.
        commitDescriptorToManifest(item.descriptor)
        releasePendingBytes(item.payload.totalBytes)
        PrefixCacheDiagnostics.logSystem(PrefixCacheDiagnostics.SSDAdmitEvent(
            id: item.descriptor.snapshotID,
            bytes: item.descriptor.bytes,
            outcome: .accepted
        ))
        PrefixCacheDiagnostics.logSystem(
            PrefixCacheDiagnostics.StorageRefCommitEvent(id: item.descriptor.snapshotID)
        )
        onCommit(item.descriptor.snapshotID)
    }

    /// Centralized writer-drop emission: terminal `ssdAdmit` outcome
    /// for the failed item, then the `storageRefDropCallback`
    /// lifecycle event, then the actual `onDrop` invocation.
    /// `processPendingItem`'s drop branches all funnel through here
    /// so the event ordering stays in lockstep with the callback
    /// firing order — tests can assert "admit drop precedes
    /// callback drop" without a per-branch check.
    private func emitWriterDrop(
        id: String,
        bytes: Int,
        reason: SSDDropReason
    ) {
        let outcome: PrefixCacheDiagnostics.SSDAdmitOutcome
        switch reason {
        case .systemProtectionWins: outcome = .droppedSystemProtectionWins
        case .exceedsBudget: outcome = .droppedExceedsBudget
        case .diskFull: outcome = .droppedDiskFull
        case .writerIOError: outcome = .droppedWriterIOError
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
        PrefixCacheDiagnostics.logSystem(PrefixCacheDiagnostics.SSDAdmitEvent(
            id: id,
            bytes: bytes,
            outcome: outcome
        ))
        PrefixCacheDiagnostics.logSystem(
            PrefixCacheDiagnostics.StorageRefDropCallbackEvent(
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
            deleteResidentFile(resident.fileURL)
            PrefixCacheDiagnostics.logSystem(
                PrefixCacheDiagnostics.StorageRefDropCallbackEvent(
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

    // MARK: - Admission cut

    private enum AdmissionDecision {
        case admit
        case drop(SSDDropReason)
    }

    /// Snapshot of a committed resident that was removed from the
    /// in-memory manifest but whose on-disk file has not yet been
    /// deleted and whose `onDrop` callback has not yet fired. The
    /// writer finalizes these outside the lock via
    /// `finalizeEvictions(_:)`.
    private struct EvictedResident {
        let snapshotID: String
        let fileURL: URL
    }

    /// Type-protected LRU cut, exactly mirroring the plan's
    /// asymmetric protection rule:
    ///
    /// 1. Evict oldest non-`.system` residents one by one until the
    ///    incoming entry fits.
    /// 2. If non-system is exhausted and budget is still too tight:
    ///    - `.system` incoming → fall through to evict oldest
    ///      `.system` residents (lateral move; protection is
    ///      preserved across the set).
    ///    - non-system incoming → drop the incoming. Never evict
    ///      `.system` residents to make room for lower-value types.
    ///
    /// Returns the admit/drop decision plus the list of residents
    /// that were removed from the in-memory manifest. The caller
    /// is responsible for deleting their files and firing the
    /// `onDrop(.evictedByLRU)` callbacks **outside** the lock.
    /// Keeping filesystem I/O out of the front-door critical
    /// section is load-bearing: a slow delete must never block a
    /// concurrent `tryEnqueue` call on the MainActor thread.
    private func admitUnderBudget(
        _ descriptor: PersistedSnapshotDescriptor
    ) -> (decision: AdmissionDecision, evicted: [EvictedResident]) {
        var evicted: [EvictedResident] = []

        lock.lock()
        defer { lock.unlock() }

        let spaceNeeded = descriptor.bytes
        if currentSSDBytes + spaceNeeded <= budgetBytes {
            return (.admit, evicted)
        }

        // Unrecognized wire strings are treated as non-system so
        // they participate in normal LRU eviction and never bypass
        // system protection. Front-door validation rejects these
        // already, so this branch is only reached in tests.
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
            // Protection is preserved across the set.
            evictOldestUnderLock(
                matching: { $0 == .system },
                until: spaceNeeded,
                into: &evicted
            )
            if currentSSDBytes + spaceNeeded <= budgetBytes {
                return (.admit, evicted)
            }
            // Every resident is gone and the incoming still doesn't
            // fit — the single payload is larger than the total
            // budget. This is a configuration problem (a payload
            // bigger than `budgetBytes`), not filesystem fullness.
            return (.drop(.exceedsBudget), evicted)

        case .lastMessageBoundary, .leaf, .branchPoint:
            // Non-system incoming, non-system eligible set empty.
            // System protection kicks in: drop the incoming rather
            // than destroying any `.system` resident.
            return (.drop(.systemProtectionWins), evicted)
        }
    }

    /// Walk the manifest's snapshots in ascending `lastAccessAt`
    /// order, evicting each one that matches the predicate and
    /// freeing its SSD bytes, until the requested amount of room
    /// is available (or the eligible set is exhausted). Every
    /// removed resident is appended to `evicted` so the caller
    /// can finalize file deletion outside the lock. Must be
    /// called with `lock` held.
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

    /// Return the subset of manifest descriptors whose parsed
    /// checkpoint type satisfies `predicate`, sorted by
    /// `lastAccessAt` ascending (oldest first). Descriptors whose
    /// wire checkpoint type fails to parse are dropped — they
    /// shouldn't exist in practice and are not eviction candidates.
    /// Must be called with `lock` held.
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
            .sorted { $0.lastAccessAt < $1.lastAccessAt }
    }

    /// Drop the manifest entry, decrement the SSD byte count, and
    /// return an `EvictedResident` carrying the file URL so the
    /// caller can delete it outside the lock. Must be called with
    /// `lock` held. Returns `nil` when the snapshotID is not in
    /// the manifest. Schedules the debounced manifest persist so
    /// eviction-only paths (admission cut ending in drop, disk-full
    /// retry dropping the incoming) still write the updated
    /// manifest to disk without waiting for an unrelated
    /// subsequent mutation.
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

    /// Eviction-retry handle called when `writePayload` throws
    /// `diskFull`. Evicts the single oldest eligible resident —
    /// non-system by default, falling through to any victim when
    /// the incoming is itself a `.system` entry. Returns the
    /// evicted resident so the caller can delete the file and
    /// fire the drop callback outside the lock; returns `nil`
    /// when the eligible set is empty.
    private func retryAfterDiskFull(
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
           nsError.code == NSFileWriteOutOfSpaceError {
            return .diskFull
        }

        return .ioError(underlying: error)
    }

    // MARK: - Manifest mutation

    private func commitDescriptorToManifest(_ descriptor: PersistedSnapshotDescriptor) {
        lock.lock()
        defer { lock.unlock() }

        // Newly committed entries get a `lastAccessAt` of "now"; the
        // front door builds the descriptor once at extraction time,
        // but the writer-side recency clock should reflect commit
        // time for the first LRU cut.
        var fresh = descriptor
        fresh.lastAccessAt = Date().timeIntervalSinceReferenceDate

        manifest.snapshots[descriptor.snapshotID] = fresh
        currentSSDBytes += descriptor.bytes
        manifestDirty = true
        scheduleManifestPersistLocked()
    }

    /// Release `bytes` from the front door's pending byte counter.
    /// Called once per processed item regardless of outcome
    /// (commit, drop, retry failure).
    private func releasePendingBytes(_ bytes: Int) {
        lock.lock()
        defer { lock.unlock() }
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

    // MARK: - Manifest persistence (debounced)

    /// Schedule a manifest persist after the configured debounce
    /// window. Multiple rapid updates coalesce into a single write.
    /// Must be called with `lock` held.
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
            self?.persistManifestIfDirty()
        }
    }

    private func persistManifestIfDirty() {
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
            // `rename(2)` to replace the target. Either the old
            // manifest stays or the new one appears; there is
            // never a window with neither file on disk.
            try data.write(to: manifestURL, options: .atomic)
        } catch {
            Log.agent.error(
                "SSDSnapshotStore manifest persist failed: \(String(describing: error))"
            )
            // Mark the manifest dirty again so the next operation
            // reschedules a persist. Pure opportunistic retry; no
            // exponential backoff needed for a local filesystem.
            lock.lock()
            manifestDirty = true
            lock.unlock()
        }
    }

    // MARK: - File path derivation

    /// Canonical on-disk URL for a snapshot file. Both the writer's
    /// descriptor-based path and the reader's `SnapshotStorageRef`
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

    /// Authoritative on-disk manifest URL. Used by the writer's
    /// debounced persist and by `warmStartLoad`.
    fileprivate var manifestURL: URL {
        rootURL.appendingPathComponent("manifest.json")
    }

    // MARK: - Pending write (private value type)

    private struct PendingWrite: Sendable {
        let payload: SnapshotPayload
        let descriptor: PersistedSnapshotDescriptor
    }
}

// MARK: - Warm start + lazy hydration

extension SSDSnapshotStore {

    /// Read `manifest.json` from disk, validate each partition's
    /// `modelFingerprint` against the currently loaded model, and
    /// seed the in-memory manifest + `currentSSDBytes` with only the
    /// valid subset. Invalidated partitions get their directories
    /// asynchronously deleted.
    ///
    /// Called once at model load from `PrefixCacheManager.warmStart`.
    /// Nonisolated so the manager can invoke it without a hop — the
    /// underlying work is synchronous file I/O plus a lock-protected
    /// manifest swap.
    ///
    /// Returns the partitioned view of the loaded manifest so the
    /// manager can iterate valid descriptors and call
    /// `restoreStorageRef` without re-reading the store's private
    /// state.
    nonisolated func warmStartLoad(
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
                "SSDSnapshotStore.warmStartLoad read failed: \(String(describing: error))"
            )
            return .empty
        }

        let loaded: SnapshotManifest
        do {
            loaded = try JSONDecoder().decode(SnapshotManifest.self, from: data)
        } catch {
            Log.agent.error(
                "SSDSnapshotStore.warmStartLoad decode failed: \(String(describing: error))"
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
                "SSDSnapshotStore.warmStartLoad schema mismatch "
                + "loaded=\(loaded.schemaVersion) "
                + "current=\(SnapshotManifestSchema.currentVersion); starting fresh"
            )
            return .empty
        }

        return commitRestoredManifest(
            loaded,
            expectedFingerprint: expectedFingerprint,
            persistManifestAfter: false,
            source: "manifest.json"
        )
    }

    /// Rebuild the manifest by walking `partitions/*/` after a
    /// corrupt `manifest.json`. Reads `_meta.json` per partition for
    /// the fingerprint, then parses each `.safetensors` file's header
    /// (sidestepping the tensor payload) to recover every descriptor
    /// field needed for the radix tree + LRU budget.
    ///
    /// Files whose header cannot be decoded are deleted so the next
    /// admission can reuse their name space. The resulting
    /// manifest is persisted via the debounced write path so
    /// subsequent restarts do not retrace the directory walk.
    private nonisolated func rebuildManifestFromDirectoryWalk(
        expectedFingerprint: String
    ) -> WarmStartOutcome {
        let partitionsDir = rootURL.appendingPathComponent("partitions")
        guard FileManager.default.fileExists(atPath: partitionsDir.path) else {
            Log.agent.info(
                "SSDSnapshotStore rebuild: no `partitions/` directory, starting fresh"
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
                "SSDSnapshotStore rebuild: partitions/ listing failed: \(String(describing: error))"
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
                    "SSDSnapshotStore rebuild: partition \(digest) missing or "
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
            "SSDSnapshotStore rebuild: recovered "
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
    /// directory-walk rebuild. Applies the fingerprint filter,
    /// drops descriptors whose wire-format checkpoint type no longer
    /// round-trips (so their bytes do not leak into the SSD budget),
    /// seeds `currentSSDBytes`, schedules async cleanup of invalidated
    /// partition directories + dead-descriptor files, and optionally
    /// kicks the debounced manifest persist (used by the rebuild path
    /// to overwrite the just-renamed corrupt `manifest.json`).
    private nonisolated func commitRestoredManifest(
        _ loaded: SnapshotManifest,
        expectedFingerprint: String,
        persistManifestAfter: Bool,
        source: String
    ) -> WarmStartOutcome {
        var restored = SnapshotManifest.empty()
        var invalidatedDigests: [String] = []
        for (digest, meta) in loaded.partitions {
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
            // Drop descriptors whose wire-format checkpoint type no
            // longer decodes — `PrefixCacheManager.warmStart` would
            // skip them silently otherwise, leaving their bytes
            // stranded in `currentSSDBytes`.
            guard HybridCacheSnapshot.CheckpointType(
                wireString: desc.checkpointType
            ) != nil else {
                deadDescriptorFiles.append(
                    fileURL(snapshotID: desc.snapshotID, partitionDigest: desc.partitionDigest)
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
            "SSDSnapshotStore.warmStartLoad source=\(source) "
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

    /// Read only the header of a placeholder container file and
    /// return the embedded descriptor. Used by the rebuild path —
    /// skips the tensor payload so the walk is fast even with
    /// hundreds of snapshots. Returns `nil` on any read or decode
    /// failure; caller deletes the file.
    private nonisolated func extractDescriptorFromFile(
        _ url: URL
    ) -> PersistedSnapshotDescriptor? {
        let handle: FileHandle
        do {
            handle = try FileHandle(forReadingFrom: url)
        } catch {
            return nil
        }
        defer { try? handle.close() }

        guard let lengthData = try? handle.read(upToCount: 8),
              lengthData.count == 8
        else { return nil }
        let headerLength = lengthData.withUnsafeBytes {
            $0.load(as: UInt64.self).littleEndian
        }
        guard headerLength <= UInt64(Int.max),
              let headerData = try? handle.read(upToCount: Int(headerLength)),
              headerData.count == Int(headerLength)
        else { return nil }
        guard let header = try? JSONDecoder().decode(
            PlaceholderContainerHeader.self,
            from: headerData
        ) else { return nil }
        return header.descriptor
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
        storageRef: SnapshotStorageRef,
        expectedFingerprint: String
    ) -> HybridCacheSnapshot? {
        // Fingerprint gate: compare the partition's persisted
        // fingerprint against the caller's expected value. Mismatch
        // or missing partition is terminal — drop and miss.
        lock.lock()
        let partitionFingerprint = manifest
            .partitions[storageRef.partitionDigest]?
            .modelFingerprint
        lock.unlock()

        guard let partitionFingerprint else {
            return failLoad(
                storageRef,
                reason: .partitionNotInManifest,
                "partition not in manifest digest=\(storageRef.partitionDigest)"
            )
        }
        guard partitionFingerprint == expectedFingerprint else {
            return failLoad(
                storageRef,
                reason: .fingerprintMismatch,
                "fingerprint mismatch partition=\(partitionFingerprint.prefix(8)) "
                + "expected=\(expectedFingerprint.prefix(8))"
            )
        }

        // Read the file directly; let `Data(contentsOf:)` surface
        // missing / permission / IO errors via one catch site.
        // `.mappedIfSafe` lets the kernel page in on demand so
        // ~200 MiB snapshots do not spike peak RSS during hydration.
        let url = fileURL(forStorageRef: storageRef)
        let fileData: Data
        do {
            fileData = try Data(contentsOf: url, options: .mappedIfSafe)
        } catch {
            return failLoad(
                storageRef,
                reason: .readFailed,
                "read failed error=\(error)"
            )
        }

        // Decode the placeholder container, reconstructing MLXArrays
        // from raw payload bytes inside the caller's Metal-affine
        // context.
        do {
            return try decodePlaceholderContainer(
                fileData,
                tokenOffset: storageRef.tokenOffset,
                checkpointType: storageRef.checkpointType
            )
        } catch {
            return failLoad(
                storageRef,
                reason: .decodeFailed,
                "decode failed error=\(error)"
            )
        }
    }

    /// Shared terminal branch for every `loadSync` failure path.
    /// Logs the `message`, emits an `ssdMiss(id:reason:)` diagnostic
    /// event so operators can correlate the hydration miss with the
    /// node id that just got cleared, drops the descriptor + on-disk
    /// file + fires `onDrop(.hydrationFailure)`, and returns `nil`
    /// so the caller can `return` the result directly.
    private nonisolated func failLoad(
        _ ref: SnapshotStorageRef,
        reason: PrefixCacheDiagnostics.SSDMissReason,
        _ message: @autoclosure () -> String
    ) -> HybridCacheSnapshot? {
        Log.agent.error(
            "SSDSnapshotStore.loadSync: \(message()) "
            + "id=\(ref.snapshotID.prefix(8))"
        )
        PrefixCacheDiagnostics.logSystem(PrefixCacheDiagnostics.SSDMissEvent(
            id: ref.snapshotID,
            reason: reason
        ))
        dropHydrationFailure(id: ref.snapshotID)
        return nil
    }

    private nonisolated func fileURL(
        forStorageRef ref: SnapshotStorageRef
    ) -> URL {
        fileURL(
            snapshotID: ref.snapshotID,
            partitionDigest: ref.partitionDigest
        )
    }

    /// Remove the descriptor from the manifest, delete the on-disk
    /// file, and fire `onDrop` with `.hydrationFailure`. Called from
    /// every `loadSync` error path so the node's storageRef gets
    /// cleared and subsequent lookups miss cleanly.
    private nonisolated func dropHydrationFailure(id: String) {
        lock.lock()
        let evicted = removeResidentUnderLock(snapshotID: id)
        lock.unlock()

        if let evicted {
            try? FileManager.default.removeItem(at: evicted.fileURL)
        }
        PrefixCacheDiagnostics.logSystem(
            PrefixCacheDiagnostics.StorageRefDropCallbackEvent(
                id: id,
                reason: .hydrationFailure
            )
        )
        onDrop(id, .hydrationFailure)
    }

    /// Decode a placeholder-container file into a `HybridCacheSnapshot`.
    /// Must run inside `container.perform` — constructing MLXArrays
    /// from `Data` is Metal-affine.
    ///
    /// The placeholder header carries per-array `byte_offset` /
    /// `byte_size` pairs relative to the blob section that follows
    /// the header. We slice those bytes out and feed them to
    /// `MLXArray(_:_:dtype:)` — a convenience initializer that
    /// owns a fresh backing allocation without going through a
    /// safetensors round-trip.
    private nonisolated func decodePlaceholderContainer(
        _ data: Data,
        tokenOffset: Int,
        checkpointType: HybridCacheSnapshot.CheckpointType
    ) throws -> HybridCacheSnapshot {
        let (header, blobsStart) = try parseContainerHeader(data)

        var totalBytes = 0
        var snapshotLayers: [HybridCacheSnapshot.LayerState] = []
        snapshotLayers.reserveCapacity(header.layers.count)

        for layerHeader in header.layers {
            var stateArrays: [MLXArray] = []
            stateArrays.reserveCapacity(layerHeader.arrays.count)
            for arrayHeader in layerHeader.arrays {
                guard let dtype = LLMActor.dtypeFromWireString(arrayHeader.dtype) else {
                    throw SSDLoadError.unknownDType(arrayHeader.dtype)
                }
                let sliceStart = blobsStart + arrayHeader.byteOffset
                let sliceEnd = sliceStart + arrayHeader.byteSize
                guard sliceEnd <= data.count else {
                    throw SSDLoadError.truncatedBlob
                }
                let blob = data[sliceStart..<sliceEnd]
                stateArrays.append(MLXArray(blob, arrayHeader.shape, dtype: dtype))
                totalBytes += arrayHeader.byteSize
            }

            snapshotLayers.append(HybridCacheSnapshot.LayerState(
                className: layerHeader.className,
                state: stateArrays,
                metaState: layerHeader.metaState,
                offset: layerHeader.offset
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

    /// Parse the 8-byte length prefix + JSON header without touching
    /// the tensor payload. Returned `blobsStart` points at the first
    /// byte after the header (same as `data.startIndex + 8 + headerLength`
    /// when `data` is a fresh `Data`). Shared by `decodePlaceholderContainer`
    /// (full hydration) and `extractDescriptorFromFile(url:)` (rebuild).
    private nonisolated func parseContainerHeader(
        _ data: Data
    ) throws -> (header: PlaceholderContainerHeader, blobsStart: Int) {
        guard data.count >= 8 else { throw SSDLoadError.truncatedHeader }
        let headerLength = data.prefix(8).withUnsafeBytes {
            $0.load(as: UInt64.self).littleEndian
        }
        let headerEnd = 8 + Int(headerLength)
        guard headerEnd <= data.count else { throw SSDLoadError.truncatedHeader }
        let headerData = data[8..<headerEnd]
        let header: PlaceholderContainerHeader
        do {
            header = try JSONDecoder().decode(
                PlaceholderContainerHeader.self,
                from: headerData
            )
        } catch {
            throw SSDLoadError.invalidHeader(String(describing: error))
        }
        return (header, headerEnd)
    }

}

// MARK: - SSDLoadError

/// Errors thrown by the placeholder-container decoder. All variants
/// map to a terminal `loadSync` failure that drops the descriptor
/// and the on-disk file before reporting a miss.
nonisolated enum SSDLoadError: Error {
    case truncatedHeader
    case truncatedBlob
    case invalidHeader(String)
    case unknownDType(String)
}

// MARK: - Testing hooks

extension SSDSnapshotStore {

    /// Synchronous accessor for the current SSD byte count. Exposed
    /// for tests that need to observe admission decisions. Acquires
    /// the same lock that the writer uses, so calling this in a
    /// hot loop would serialize against the writer; tests only call
    /// it between quiescent checkpoints.
    nonisolated func currentSSDBytesForTesting() -> Int {
        lock.lock()
        defer { lock.unlock() }
        return currentSSDBytes
    }

    /// Snapshot of the manifest's descriptor IDs, sorted by
    /// `lastAccessAt` ascending. Exposed for tests that need to
    /// verify the LRU ordering after a sequence of commits.
    nonisolated func residentIDsByRecencyForTesting() -> [String] {
        lock.lock()
        defer { lock.unlock() }
        return manifest.snapshots.values
            .sorted { $0.lastAccessAt < $1.lastAccessAt }
            .map(\.snapshotID)
    }

    /// Synchronous peek at the pending queue depth. Tests use this
    /// to assert back-pressure eviction happened immediately rather
    /// than deferred to the writer.
    nonisolated func pendingCountForTesting() -> Int {
        lock.lock()
        defer { lock.unlock() }
        return pending.count
    }

    /// Force-flush the debounced manifest persist. Tests call this
    /// to observe on-disk manifest state without waiting on the
    /// debounce window.
    nonisolated func flushManifestForTesting() {
        persistManifestIfDirty()
    }

    /// Inject a descriptor into the manifest without going through
    /// the writer loop. Tests use this to drive `recordHit` /
    /// `loadSync` scenarios deterministically — the writer path has
    /// its own callback timing to wait on, which is overkill for
    /// tests that just want a known manifest state.
    ///
    /// The partition must already be registered via
    /// `registerPartition(_:digest:)` so the manifest invariant
    /// holds; the seeded descriptor updates `currentSSDBytes`.
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
    /// anything. Used by the `recordHit` regression test to observe
    /// the bump without racing the writer's debounced persist.
    nonisolated func lastAccessAtForTesting(id: String) -> Double {
        lock.lock()
        defer { lock.unlock() }
        return manifest.snapshots[id]?.lastAccessAt ?? -1
    }
}

// MARK: - Placeholder on-disk format

/// Codable header for the placeholder container. Pinned with the
/// full `PersistedSnapshotDescriptor` so a directory walk can
/// rebuild the authoritative manifest after a `manifest.json`
/// corruption: every descriptor field needed to reconstruct the
/// radix-tree shape + LRU bookkeeping survives in each file.
///
/// `PartitionMeta` is deliberately NOT duplicated per file — it
/// lives in `partitions/{digest}/_meta.json` so the rebuild can
/// validate the partition's fingerprint without paying per-file
/// duplication. See
/// `SSDSnapshotStore.writePartitionMetaFile(_:digest:)` and
/// `rebuildManifestFromDirectoryWalk(_:)`.
private nonisolated struct PlaceholderContainerHeader: Codable, Sendable {
    let formatKind: String
    let schemaVersion: Int
    let descriptor: PersistedSnapshotDescriptor
    let layers: [Layer]

    nonisolated struct Layer: Codable, Sendable {
        let className: String
        let metaState: [String]
        let offset: Int
        let arrays: [ArrayEntry]

        enum CodingKeys: String, CodingKey {
            case className = "class_name"
            case metaState = "meta_state"
            case offset
            case arrays
        }
    }

    nonisolated struct ArrayEntry: Codable, Sendable {
        let dtype: String
        let shape: [Int]
        let byteOffset: Int
        let byteSize: Int

        enum CodingKeys: String, CodingKey {
            case dtype, shape
            case byteOffset = "byte_offset"
            case byteSize = "byte_size"
        }
    }

    enum CodingKeys: String, CodingKey {
        case formatKind = "format_kind"
        case schemaVersion = "schema_version"
        case descriptor
        case layers
    }
}

/// Serialize a payload into a single byte blob following the
/// placeholder container format:
///
/// ```
/// [8 bytes little-endian UInt64: header JSON length]
/// [header JSON bytes]
/// [concatenated array data blobs]
/// ```
///
/// The header carries the full descriptor so warm start can rebuild
/// the manifest from a directory walk when `manifest.json` is corrupt.
private nonisolated func encodePlaceholderContainer(
    payload: SnapshotPayload,
    descriptor: PersistedSnapshotDescriptor
) throws -> Data {
    var layerHeaders: [PlaceholderContainerHeader.Layer] = []
    layerHeaders.reserveCapacity(payload.layers.count)
    var blobs: [Data] = []
    var runningByteOffset = 0

    for layer in payload.layers {
        var arrayEntries: [PlaceholderContainerHeader.ArrayEntry] = []
        arrayEntries.reserveCapacity(layer.state.count)
        for array in layer.state {
            arrayEntries.append(.init(
                dtype: array.dtype,
                shape: array.shape,
                byteOffset: runningByteOffset,
                byteSize: array.data.count
            ))
            runningByteOffset += array.data.count
            blobs.append(array.data)
        }
        layerHeaders.append(.init(
            className: layer.className,
            metaState: layer.metaState,
            offset: layer.offset,
            arrays: arrayEntries
        ))
    }

    let header = PlaceholderContainerHeader(
        formatKind: "tesseract-cache-v1",
        schemaVersion: SnapshotManifestSchema.currentVersion,
        descriptor: descriptor,
        layers: layerHeaders
    )

    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    let headerData = try encoder.encode(header)

    var out = Data(capacity: 8 + headerData.count + runningByteOffset)
    var headerLength = UInt64(headerData.count).littleEndian
    withUnsafeBytes(of: &headerLength) { out.append(contentsOf: $0) }
    out.append(headerData)
    for blob in blobs {
        out.append(blob)
    }
    return out
}
