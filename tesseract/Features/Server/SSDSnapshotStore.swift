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
            return .rejectedInvalidCheckpointType
        }

        let payloadBytes = payload.totalBytes

        // A single payload larger than the cap cannot be queued at
        // all; no amount of back-pressure eviction can create room.
        if payloadBytes > maxPendingBytes {
            return .rejectedTooLargeForBudget
        }

        var droppedIDs: [String] = []

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
            droppedIDs.append(oldest.descriptor.snapshotID)
        }

        pending.append(PendingWrite(payload: payload, descriptor: descriptor))
        pendingBytes += payloadBytes

        lock.unlock()

        for id in droppedIDs {
            onDrop(id, .backpressureOldest)
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
        defer { lock.unlock() }
        manifest.partitions[digest] = meta
        manifestDirty = true
        scheduleManifestPersistLocked()
    }

    // MARK: - Writer loop

    private func writerLoop() async {
        for await _ in wakeupStream {
            await drainPending()
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
        finalizeEvictions(evicted)

        switch admission {
        case .admit:
            break
        case .drop(let reason):
            releasePendingBytes(item.payload.totalBytes)
            onDrop(item.descriptor.snapshotID, reason)
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
                finalizeEvictions([retryVictim])
                do {
                    try writePayload(item.payload, descriptor: item.descriptor)
                } catch {
                    Log.agent.error(
                        "SSD writer diskFull retry failed for \(item.descriptor.snapshotID): "
                        + "\(String(describing: error))"
                    )
                    releasePendingBytes(item.payload.totalBytes)
                    onDrop(item.descriptor.snapshotID, .diskFull)
                    return
                }
            } else {
                Log.agent.error(
                    "SSD writer diskFull and no eviction victim available for "
                    + "\(item.descriptor.snapshotID)"
                )
                releasePendingBytes(item.payload.totalBytes)
                onDrop(item.descriptor.snapshotID, .diskFull)
                return
            }
        } catch {
            Log.agent.error(
                "SSD writer I/O failure for \(item.descriptor.snapshotID): "
                + "\(String(describing: error))"
            )
            releasePendingBytes(item.payload.totalBytes)
            onDrop(item.descriptor.snapshotID, .writerIOError)
            return
        }

        // 3. Write succeeded: register the descriptor, release the
        //    pending byte budget, and fire the commit callback.
        commitDescriptorToManifest(item.descriptor)
        releasePendingBytes(item.payload.totalBytes)
        onCommit(item.descriptor.snapshotID)
    }

    /// Delete the on-disk files and fire `onDrop(.evictedByLRU)`
    /// for every resident that was removed from the manifest by
    /// the admission cut. Runs OUTSIDE the front-door lock so the
    /// synchronous `tryEnqueue` path never pays for filesystem I/O.
    private func finalizeEvictions(_ evicted: [EvictedResident]) {
        for resident in evicted {
            deleteResidentFile(resident.fileURL)
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

        let manifestURL = rootURL.appendingPathComponent("manifest.json")

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

    private func fileURL(for descriptor: PersistedSnapshotDescriptor) -> URL {
        let relative = PersistedSnapshotDescriptor.relativeFilePath(
            snapshotID: descriptor.snapshotID,
            partitionDigest: descriptor.partitionDigest
        )
        return rootURL.appendingPathComponent(relative)
    }

    // MARK: - Pending write (private value type)

    private struct PendingWrite: Sendable {
        let payload: SnapshotPayload
        let descriptor: PersistedSnapshotDescriptor
    }
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
}

// MARK: - Placeholder on-disk format

/// Serialize a payload into a single byte blob following the
/// placeholder container format. Format:
///
/// ```
/// [8 bytes little-endian UInt64: header JSON length]
/// [header JSON bytes]
/// [concatenated array data blobs]
/// ```
///
/// The header JSON has `"format_kind": "tesseract-cache-v1"` so the
/// downstream real-safetensors reader can refuse these files
/// cleanly when it ships.
private nonisolated func encodePlaceholderContainer(
    payload: SnapshotPayload,
    descriptor: PersistedSnapshotDescriptor
) throws -> Data {
    // Build the header structure + a parallel list of byte blobs.
    var layers: [[String: Any]] = []
    var blobs: [Data] = []
    var runningByteOffset = 0

    for layer in payload.layers {
        var arrayEntries: [[String: Any]] = []
        for array in layer.state {
            arrayEntries.append([
                "dtype": array.dtype,
                "shape": array.shape,
                "byte_offset": runningByteOffset,
                "byte_size": array.data.count,
            ])
            runningByteOffset += array.data.count
            blobs.append(array.data)
        }
        layers.append([
            "class_name": layer.className,
            "meta_state": layer.metaState,
            "offset": layer.offset,
            "arrays": arrayEntries,
        ])
    }

    let header: [String: Any] = [
        "format_kind": "tesseract-cache-v1",
        "schema_version": SnapshotManifestSchema.currentVersion,
        "snapshot_id": descriptor.snapshotID,
        "partition_digest": descriptor.partitionDigest,
        "token_offset": descriptor.tokenOffset,
        "checkpoint_type": descriptor.checkpointType,
        "layers": layers,
    ]

    let headerData = try JSONSerialization.data(
        withJSONObject: header,
        options: [.sortedKeys]
    )

    var out = Data(capacity: 8 + headerData.count + runningByteOffset)
    var headerLength = UInt64(headerData.count).littleEndian
    withUnsafeBytes(of: &headerLength) { out.append(contentsOf: $0) }
    out.append(headerData)
    for blob in blobs {
        out.append(blob)
    }
    return out
}
