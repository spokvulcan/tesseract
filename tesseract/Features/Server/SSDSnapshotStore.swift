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
    /// Tree-side consumers may clear the stale committed Snapshot Ref
    /// lazily when a subsequent hydration attempt supplies the node.
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

// MARK: - SSDSnapshotStore

nonisolated final class SSDSnapshotStore: @unchecked Sendable {

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
    private let writerDrainPreludeForTesting: (@Sendable () async -> Void)?

    // MARK: - Public API

    init(
        config: SSDPrefixCacheConfig,
        manifestDebounce: Duration = .milliseconds(500),
        onCommit: @escaping @Sendable (String) -> Void = { _ in },
        onDrop: @escaping @Sendable (String, SSDDropReason) -> Void = { _, _ in },
        writerDrainPreludeForTesting: (@Sendable () async -> Void)? = nil
    ) {
        self.rootURL = config.rootURL
        self.budgetBytes = config.budgetBytes
        self.maxPendingBytes = config.maxPendingBytes
        self.ledger = SnapshotLedger(
            rootURL: config.rootURL,
            budgetBytes: config.budgetBytes,
            manifestDebounce: manifestDebounce
        )
        self.onCommit = onCommit
        self.onDrop = onDrop
        self.writerDrainPreludeForTesting = writerDrainPreludeForTesting

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

        // Enforce the manifest invariant that every snapshots entry
        // must reference a registered partition. The writer cannot
        // autoregister because `PartitionMeta` carries load-bearing
        // fields (modelFingerprint, kvBits, etc.) that only the caller
        // knows at model-load time. Reject here so the caller gets
        // immediate feedback rather than persisting a dangling manifest
        // entry. The ledger check runs before the queue lock — the two
        // locks never nest.
        guard ledger.hasPartition(digest: descriptor.partitionDigest) else {
            PrefixCacheDiagnostics.logSystem(PrefixCacheDiagnostics.SSDAdmitEvent(
                id: descriptor.snapshotID,
                bytes: payloadBytes,
                outcome: .droppedUnregisteredPartition
            ))
            return .rejectedUnregisteredPartition
        }

        var droppedItems: [(id: String, bytes: Int)] = []

        queueLock.lock()

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

        queueLock.unlock()

        for item in droppedItems {
            // Both events fire per bumped item: the admission outcome
            // (`droppedByteBudget` is the terminal verdict for that
            // earlier `tryEnqueue` call), and the lifecycle callback
            // (`storageRefDropCallback`, the stable telemetry name, so
            // the radix node sees its pending ref cleared).
            PrefixCacheDiagnostics.logSystem(PrefixCacheDiagnostics.SSDAdmitEvent(
                id: item.id,
                bytes: item.bytes,
                outcome: .droppedByteBudget
            ))
            PrefixCacheDiagnostics.logSystem(
                PrefixCacheDiagnostics.SnapshotRefDropCallbackEvent(
                    id: item.id,
                    reason: .backpressureOldest
                )
            )
            onDrop(item.id, .backpressureOldest)
        }

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

        return PromptCacheSSDSnapshot(
            enabled: true,
            rootPath: rootURL.path,
            budgetBytes: budgetBytes,
            currentBytes: residency.currentBytes,
            pendingBytes: queuedBytes,
            maxPendingBytes: maxPendingBytes,
            pendingCount: queuedCount,
            snapshotCount: residency.snapshotCount,
            partitionCount: residency.partitionCount
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

        ledger.persistNow()
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
        if let pendingIndex = pending.firstIndex(where: { $0.descriptor.snapshotID == snapshotID }) {
            let removed = pending.remove(at: pendingIndex)
            pendingBytes -= removed.payload.totalBytes
            if pendingBytes < 0 { pendingBytes = 0 }
            queueLock.unlock()
            return
        }
        queueLock.unlock()

        // Not in the pending queue: hand off to the ledger, which
        // atomically removes a committed resident (returning it for file
        // deletion) or tombstones an in-flight write so its later
        // `commit` self-vetoes. The queue lock is released first — the
        // two locks never nest.
        if let resident = ledger.removeOrTombstone(id: snapshotID) {
            try? FileManager.default.removeItem(at: resident.fileURL)
        }
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
        guard !pending.isEmpty else { return nil }
        return pending.removeFirst()
    }

    private func processPendingItem(_ item: PendingWrite) async {
        if ledger.consumeTombstone(id: item.descriptor.snapshotID) {
            releasePendingBytes(item.payload.totalBytes)
            return
        }

        // 1. Admission LRU cut. The ledger returns the admit/drop
        //    decision and a list of committed residents it evicted to
        //    make room, under its own lock. File deletion and `onDrop`
        //    callbacks happen here, outside that lock.
        let (admission, evicted) = ledger.admit(item.descriptor)
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
            if let retryVictim = ledger.retryAfterDiskFull(item.descriptor) {
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

        if ledger.consumeTombstone(id: item.descriptor.snapshotID) {
            try? FileManager.default.removeItem(at: fileURL(for: item.descriptor))
            releasePendingBytes(item.payload.totalBytes)
            return
        }

        // 3. Write succeeded: register the descriptor in the ledger,
        //    release the pending byte budget, and fire the commit
        //    callback. A `false` here is the tombstone self-veto (the
        //    snapshot was deleted while in flight) — drop the file.
        guard ledger.commit(item.descriptor) else {
            try? FileManager.default.removeItem(at: fileURL(for: item.descriptor))
            releasePendingBytes(item.payload.totalBytes)
            return
        }
        releasePendingBytes(item.payload.totalBytes)
        PrefixCacheDiagnostics.logSystem(PrefixCacheDiagnostics.SSDAdmitEvent(
            id: item.descriptor.snapshotID,
            bytes: item.descriptor.bytes,
            outcome: .accepted
        ))
        PrefixCacheDiagnostics.logSystem(
            PrefixCacheDiagnostics.SnapshotRefCommitEvent(id: item.descriptor.snapshotID)
        )
        onCommit(item.descriptor.snapshotID)
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
            deleteResidentFile(resident.fileURL)
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
        expectedFingerprint: String
    ) -> HybridCacheSnapshot? {
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

        // Read the file directly; let `Data(contentsOf:)` surface
        // missing / permission / IO errors via one catch site.
        // `.mappedIfSafe` lets the kernel page in on demand so
        // ~200 MiB snapshots do not spike peak RSS during hydration.
        let url = fileURL(forSnapshotRef: snapshotRef)
        let fileData: Data
        do {
            fileData = try Data(contentsOf: url, options: .mappedIfSafe)
        } catch {
            return failLoad(
                snapshotRef,
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
        PrefixCacheDiagnostics.logSystem(PrefixCacheDiagnostics.SSDMissEvent(
            id: ref.snapshotID,
            reason: reason
        ))
        dropHydrationFailure(id: ref.snapshotID)
        return nil
    }

    private nonisolated func fileURL(
        forSnapshotRef ref: SnapshotRef
    ) -> URL {
        fileURL(
            snapshotID: ref.snapshotID,
            partitionDigest: ref.partitionDigest
        )
    }

    /// Remove the descriptor from the manifest, delete the on-disk
    /// file, and fire `onDrop` with `.hydrationFailure`. Called from
    /// every `loadSync` error path so the node's Snapshot Ref gets
    /// cleared and subsequent lookups miss cleanly.
    private nonisolated func dropHydrationFailure(id: String) {
        if let evicted = ledger.remove(id: id) {
            try? FileManager.default.removeItem(at: evicted.fileURL)
        }
        PrefixCacheDiagnostics.logSystem(
            PrefixCacheDiagnostics.SnapshotRefDropCallbackEvent(
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
        let (header, blobsStart) = try PlaceholderContainerHeader.parse(from: data)

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
