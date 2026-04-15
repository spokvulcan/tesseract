import Foundation
import MLXLMCommon

/// Partition key for isolating radix trees by runtime configuration.
///
/// Tool/template digests are intentionally NOT part of the partition key:
/// different tools/context → different tokens → different radix paths →
/// naturally isolated within one partition.
///
/// `Comparable` so partition iteration can produce a deterministic order
/// (modelID → kvBits → kvGroupSize → modelFingerprint) for stable
/// tie-break behavior in eviction.
struct CachePartitionKey: Hashable, Sendable, Comparable {
    let modelID: String
    let kvBits: Int?
    let kvGroupSize: Int
    /// Stable hex SHA-256 of the loaded model's weight files
    /// (`ModelFingerprint.computeFingerprint`). Folded in so a weight swap
    /// under the same `modelID` cannot surface stale persisted snapshots.
    /// `nil` for RAM-only test fixtures.
    let modelFingerprint: String?

    nonisolated init(
        modelID: String,
        kvBits: Int?,
        kvGroupSize: Int,
        modelFingerprint: String? = nil
    ) {
        self.modelID = modelID
        self.kvBits = kvBits
        self.kvGroupSize = kvGroupSize
        self.modelFingerprint = modelFingerprint
    }

    static func < (lhs: CachePartitionKey, rhs: CachePartitionKey) -> Bool {
        (lhs.modelID, lhs.kvBits ?? -1, lhs.kvGroupSize, lhs.modelFingerprint ?? "")
            < (rhs.modelID, rhs.kvBits ?? -1, rhs.kvGroupSize, rhs.modelFingerprint ?? "")
    }
}

@MainActor
final class PrefixCacheManager {
    private enum PendingBootstrapBoundary {
        case unscoped
        case request(UUID)
    }

    /// Per-partition tree collection. Typed as the concrete
    /// `TieredSnapshotStore` (rather than the `SnapshotStore`
    /// protocol) so the manager can reach `admitSnapshot` and the
    /// storage-ref lifecycle callbacks — those are not part of the
    /// read-only protocol.
    private let store: TieredSnapshotStore
    /// Set by `evictToFitBudget` when the first-ever drain happens
    /// inside an in-flight request. Production passes a per-request ID
    /// so only the matching request-end `recordRequest` call can start
    /// bootstrapping; direct manager tests can omit the ID and use the
    /// `.unscoped` fallback.
    private var pendingBootstrapBoundary: PendingBootstrapBoundary?
    var memoryBudgetBytes: Int
    /// Optional adaptive `alpha` tuner. Production caches attach one;
    /// test/replay caches pass `nil` to avoid recursive recording when
    /// the tuner itself spins up sandboxed caches during grid search.
    let alphaTuner: AlphaTuner?

    init(
        memoryBudgetBytes: Int,
        alphaTuner: AlphaTuner? = nil,
        tieredStore: TieredSnapshotStore? = nil
    ) {
        self.store = tieredStore ?? TieredSnapshotStore(ssdConfig: nil)
        self.memoryBudgetBytes = memoryBudgetBytes
        self.alphaTuner = alphaTuner
    }

    struct CacheStats: Sendable {
        let partitionCount: Int
        let totalNodeCount: Int
        let totalSnapshotBytes: Int
        /// Per-checkpoint-type snapshot counts. Aggregates the incremental
        /// counters from each partition's `TokenRadixTree`.
        let snapshotsByType: [HybridCacheSnapshot.CheckpointType: Int]

        nonisolated var snapshotCount: Int { snapshotsByType.values.reduce(0, +) }
    }

    struct EvictionEvent: Sendable {
        enum Strategy: String, Sendable {
            case utility
            case fallback
        }

        let strategy: Strategy
        let offset: Int
        let checkpointType: HybridCacheSnapshot.CheckpointType
        let freedBytes: Int
        let budgetBytes: Int
        let snapshotBytesAfter: Int
        let normalizedRecency: Double?
        let normalizedFlopEfficiency: Double?
        let utility: Double?
        /// Non-nil when the victim node was kept in the radix tree
        /// because it carried a live `storageRef` (Task 4.1.8 body-drop
        /// guard) — the RAM body was freed but the SSD-backed node
        /// remains reachable for future lookups. Consumers
        /// (`LLMActor.makeHTTPPrefixCacheGeneration`) emit a
        /// `ssdBodyDrop(id:)` diagnostic event for each non-nil
        /// entry. `nil` for plain RAM-only evictions.
        let bodyDroppedStorageRefID: String?

        nonisolated init(
            strategy: Strategy,
            offset: Int,
            checkpointType: HybridCacheSnapshot.CheckpointType,
            freedBytes: Int,
            budgetBytes: Int,
            snapshotBytesAfter: Int,
            normalizedRecency: Double?,
            normalizedFlopEfficiency: Double?,
            utility: Double?,
            bodyDroppedStorageRefID: String? = nil
        ) {
            self.strategy = strategy
            self.offset = offset
            self.checkpointType = checkpointType
            self.freedBytes = freedBytes
            self.budgetBytes = budgetBytes
            self.snapshotBytesAfter = snapshotBytesAfter
            self.normalizedRecency = normalizedRecency
            self.normalizedFlopEfficiency = normalizedFlopEfficiency
            self.utility = utility
            self.bodyDroppedStorageRefID = bodyDroppedStorageRefID
        }
    }

    struct StoreDiagnostics: Sendable {
        let evictions: [EvictionEvent]
        let supersededLeaves: [LeafSupersession]
        let stats: CacheStats
    }

    struct LeafSupersession: Sendable {
        let offset: Int
        let bodyDroppedStorageRefID: String?
    }

    private struct EvictionCandidate {
        let tree: TokenRadixTree
        let node: RadixTreeNode
        let strategy: EvictionEvent.Strategy
        let score: EvictionScore?
    }

    // MARK: - Lookup

    struct LookupResult: Sendable {
        let snapshot: HybridCacheSnapshot?
        let partitionKey: CachePartitionKey?
        let snapshotTokenOffset: Int
        /// Actual token-level match depth in the radix tree, which may be
        /// deeper than `snapshotTokenOffset` when the tree matches beyond the
        /// best stored snapshot.
        let sharedPrefixLength: Int
        let reason: LookupReason
        /// Snapshot ID whose `lastAccessAt` was just bumped by the
        /// SSD LRU bookkeeping (`ssdStore.recordHit(id:)`). Non-nil
        /// only when the hit landed on a committed storage ref —
        /// state 4 (RAM body present + SSD ref committed). Consumers
        /// emit a `ssdRecordHit(id:)` diagnostic event whenever this
        /// is non-nil so the SSD LRU age progression is visible
        /// under operator workload traces.
        let recordedHitSnapshotID: String?

        nonisolated init(
            snapshot: HybridCacheSnapshot?,
            partitionKey: CachePartitionKey?,
            snapshotTokenOffset: Int,
            sharedPrefixLength: Int,
            reason: LookupReason,
            recordedHitSnapshotID: String? = nil
        ) {
            self.snapshot = snapshot
            self.partitionKey = partitionKey
            self.snapshotTokenOffset = snapshotTokenOffset
            self.sharedPrefixLength = sharedPrefixLength
            self.reason = reason
            self.recordedHitSnapshotID = recordedHitSnapshotID
        }

        /// Restore the cached KV/Mamba state. Each call produces an independent deep copy.
        /// Nonisolated because it operates only on the snapshot's deep-copy data.
        nonisolated func restoreCache() -> [any KVCache]? {
            guard let snapshot, let key = partitionKey else { return nil }
            return snapshot.restore(kvBitsHint: key.kvBits, kvGroupSizeHint: key.kvGroupSize)
        }
    }

    /// Handles the `MainActor`-owned `RadixTreeNode` and the
    /// `nonisolated` `SSDSnapshotStore` that LLMActor needs to
    /// hydrate a state-5 node. `@unchecked Sendable` because the
    /// node is MainActor-owned — LLMActor only reads the storage
    /// ref off-main and hops back to MainActor before touching the
    /// node via `PrefixCacheManager.promote(node:snapshot:partitionKey:)`.
    struct SSDHitContext: @unchecked Sendable {
        let storageRef: SnapshotStorageRef
        let ssdStore: SSDSnapshotStore
        let node: RadixTreeNode
    }

    enum LookupReason: CustomStringConvertible, Sendable {
        case hit(snapshotOffset: Int, totalTokens: Int, type: HybridCacheSnapshot.CheckpointType)
        /// State 5 — body absent, committed SSD ref present. LLMActor
        /// hydrates the body via `ssdStore.loadSync(...)` inside
        /// `container.perform`, then promotes the node back to state 4
        /// on MainActor.
        case ssdHit(SSDHitContext)
        case missNoEntries
        case missNoSnapshotInPrefix

        nonisolated var description: String {
            switch self {
            case .hit(let offset, let total, let type):
                "hit(\(type) at \(offset)/\(total))"
            case .ssdHit(let ctx):
                "ssdHit(\(ctx.storageRef.checkpointType) at \(ctx.storageRef.tokenOffset), id=\(ctx.storageRef.snapshotID.prefix(8)))"
            case .missNoEntries:
                "miss(no entries)"
            case .missNoSnapshotInPrefix:
                "miss(no snapshot in prefix)"
            }
        }
    }

    func lookup(tokens: [Int], partitionKey: CachePartitionKey) -> LookupResult {
        guard let tree = store.tree(for: partitionKey) else {
            return LookupResult(
                snapshot: nil, partitionKey: nil,
                snapshotTokenOffset: 0, sharedPrefixLength: 0,
                reason: .missNoEntries
            )
        }

        // Consider state-5 (committed ref, no body) as hittable —
        // LLMActor will hydrate from SSD. Pending refs (state 3) are
        // filtered out inside `findBestSnapshot` because their
        // `committed` flag is false.
        guard let (node, _) = tree.findBestSnapshot(
            tokens: tokens, includeStorageRefs: true
        ) else {
            let treeMatchDepth = tree.findSharedPrefixLength(tokens: tokens)
            return LookupResult(
                snapshot: nil, partitionKey: partitionKey,
                snapshotTokenOffset: 0, sharedPrefixLength: treeMatchDepth,
                reason: .missNoSnapshotInPrefix
            )
        }

        let treeMatchDepth = tree.findSharedPrefixLength(tokens: tokens)

        if let snapshot = node.snapshot {
            // States 1, 2, or 4. On state 4 (committed ref + body)
            // bump the SSD descriptor's `lastAccessAt` so a hot RAM
            // hit does not look stale to the SSD LRU when the body
            // is eventually dropped to state 5.
            var recordedHitID: String?
            if let ref = node.storageRef, ref.committed {
                store.ssdStore?.recordHit(id: ref.snapshotID)
                recordedHitID = ref.snapshotID
            }
            return LookupResult(
                snapshot: snapshot,
                partitionKey: partitionKey,
                snapshotTokenOffset: snapshot.tokenOffset,
                sharedPrefixLength: treeMatchDepth,
                reason: .hit(
                    snapshotOffset: snapshot.tokenOffset,
                    totalTokens: tokens.count,
                    type: snapshot.checkpointType
                ),
                recordedHitSnapshotID: recordedHitID
            )
        }

        // State 5 — body absent, committed ref present. The filter on
        // `findBestSnapshot` guarantees `storageRef?.committed == true`
        // when we reach this branch; force-unwrap is safe. SSD store
        // must also be non-nil because the ref could only have been
        // assigned through the tiered store's admission path.
        guard let ref = node.storageRef, let ssdStore = store.ssdStore else {
            return LookupResult(
                snapshot: nil, partitionKey: partitionKey,
                snapshotTokenOffset: 0, sharedPrefixLength: treeMatchDepth,
                reason: .missNoSnapshotInPrefix
            )
        }
        let context = SSDHitContext(
            storageRef: ref,
            ssdStore: ssdStore,
            node: node
        )
        return LookupResult(
            snapshot: nil,
            partitionKey: partitionKey,
            snapshotTokenOffset: ref.tokenOffset,
            sharedPrefixLength: treeMatchDepth,
            reason: .ssdHit(context)
        )
    }

    /// When a lookup restores at `K` but the tree already matches farther to
    /// `M`, synthesize a checkpoint at `M` so the next request can skip the
    /// already-shared gap. This is layered on top of the existing Phase 2
    /// planner and does not change its speculative branch-point rules.
    func alignmentCheckpointOffset(
        lookupResult: LookupResult,
        totalTokenCount: Int,
        plannedCheckpoints: [(offset: Int, type: HybridCacheSnapshot.CheckpointType)]
    ) -> Int? {
        guard lookupResult.snapshot != nil else { return nil }

        let snapshotOffset = lookupResult.snapshotTokenOffset
        let sharedPrefixLength = lookupResult.sharedPrefixLength
        let alignmentThreshold = 256

        guard snapshotOffset > 0,
              sharedPrefixLength > snapshotOffset,
              sharedPrefixLength < totalTokenCount,
              sharedPrefixLength - snapshotOffset > alignmentThreshold
        else { return nil }

        guard !plannedCheckpoints.contains(where: { $0.offset == sharedPrefixLength }) else {
            return nil
        }

        guard case .hit(let offset, _, _) = lookupResult.reason,
              offset == snapshotOffset
        else {
            return nil
        }

        return sharedPrefixLength
    }

    /// Runs the production lookup + checkpoint planner flow, including the
    /// Phase 3.1 alignment checkpoint merge used by `LLMActor`.
    func lookupAndPlanCheckpoints(
        tokens: [Int],
        stablePrefixOffset: Int?,
        partitionKey: CachePartitionKey
    ) -> (lookup: LookupResult, plan: [(offset: Int, type: HybridCacheSnapshot.CheckpointType)]) {
        let lookup = lookup(tokens: tokens, partitionKey: partitionKey)
        let basePlan = planCheckpoints(
            tokens: tokens,
            stablePrefixOffset: stablePrefixOffset,
            partitionKey: partitionKey
        )
        guard let alignmentOffset = alignmentCheckpointOffset(
            lookupResult: lookup,
            totalTokenCount: tokens.count,
            plannedCheckpoints: basePlan
        ) else {
            return (lookup, basePlan)
        }
        return (lookup, basePlan + [(offset: alignmentOffset, type: .branchPoint)])
    }

    // MARK: - Checkpoint Planning

    /// Determine checkpoint offsets for the upcoming prefill.
    ///
    /// Captures the stable-prefix snapshot plus an optional speculative
    /// branch-point snapshot:
    /// - `stablePrefixOffset`: where `system + tools` end (shared across
    ///   conversations, any request with the same system/tools can hit).
    ///   Stored as `.system` — type-protected from utility eviction
    ///   because it's the cross-conversation hot prefix that an entire
    ///   tree is built on.
    ///
    /// Leaf checkpoint is NOT planned — captured post-generation via storeLeaf().
    /// Existing snapshots at the same offset are skipped.
    func planCheckpoints(
        tokens: [Int],
        stablePrefixOffset: Int?,
        partitionKey: CachePartitionKey
    ) -> [(offset: Int, type: HybridCacheSnapshot.CheckpointType)] {
        var plan: [(offset: Int, type: HybridCacheSnapshot.CheckpointType)] = []
        let tree = store.tree(for: partitionKey)

        /// Returns true if a snapshot of the requested type already exists at
        /// exactly `offset`. A snapshot of a different type (e.g. a leaf stored
        /// at the stable prefix offset) does NOT count — we still want to
        /// capture a proper system snapshot there.
        func alreadyStored(offset: Int, type: HybridCacheSnapshot.CheckpointType) -> Bool {
            guard let tree,
                  let (node, _) = tree.findBestSnapshot(
                      tokens: Array(tokens[0..<offset]), updateAccess: false)
            else { return false }
            return node.tokenOffset == offset && node.snapshot?.checkpointType == type
        }

        if let offset = stablePrefixOffset, offset > 0, offset < tokens.count,
           !alreadyStored(offset: offset, type: .system)
        {
            plan.append((offset: offset, type: .system))
        }

        if let tree,
           let splitOffset = tree.findIntermediateSplitOffsetForInsertion(tokens: tokens),
           splitOffset > 0,
           splitOffset < tokens.count,
           !plan.contains(where: { $0.offset == splitOffset })
        {
            plan.append((offset: splitOffset, type: .branchPoint))
        }

        return plan
    }

    // MARK: - Store

    /// Store mid-prefill snapshots captured during prepareWithCheckpoints().
    ///
    /// `snapshotPayloads` carries the pre-extracted CPU-owned bytes for
    /// each entry in `capturedSnapshots`, positionally aligned (same
    /// count, same order), produced by
    /// `LLMActor.extractSnapshotPayloads(_:ssdEnabled:)` inside the
    /// same `container.perform` scope that captured the snapshots.
    /// An empty array (the default) signals SSD disabled and skips
    /// the admission path; the radix-tree insertion always runs.
    @discardableResult
    func storeSnapshots(
        promptTokens: [Int],
        capturedSnapshots: [HybridCacheSnapshot],
        snapshotPayloads: [SnapshotPayload] = [],
        partitionKey: CachePartitionKey,
        requestID: UUID? = nil
    ) -> StoreDiagnostics {
        guard !capturedSnapshots.isEmpty else {
            return StoreDiagnostics(evictions: [], supersededLeaves: [], stats: stats)
        }

        let tree = store.getOrCreateTree(for: partitionKey)
        tree.insertPath(tokens: promptTokens)

        let payloadsAligned = snapshotPayloads.count == capturedSnapshots.count
        if payloadsAligned {
            registerSSDPartitionIfNeeded(for: partitionKey)
        }
        for (index, snapshot) in capturedSnapshots.enumerated() {
            let offset = snapshot.tokenOffset
            guard offset > 0, offset <= promptTokens.count else { continue }
            let path = Array(promptTokens[0..<offset])
            let node = tree.insertPath(tokens: path)
            guard node.tokenOffset == offset else { continue }
            tree.storeSnapshot(snapshot, on: node)

            guard payloadsAligned else { continue }
            let payload = snapshotPayloads[index]
            let descriptor = makePersistedDescriptor(
                partitionKey: partitionKey,
                pathFromRoot: path,
                snapshot: snapshot,
                payloadBytes: payload.totalBytes
            )
            store.admitSnapshot(
                node: node,
                tree: tree,
                payload: payload,
                descriptor: descriptor
            )
        }

        let evictions = evictToFitBudget(
            requestID: requestID,
            preferredPartitionKey: partitionKey
        )
        return StoreDiagnostics(evictions: evictions, supersededLeaves: [], stats: stats)
    }

    /// Store the leaf snapshot under post-response tokens.
    /// `storedTokens` = re-tokenized (prompt + generated response).
    ///
    /// `leafPayload` is the pre-extracted CPU-owned bytes matching
    /// `leafSnapshot`, produced by
    /// `LLMActor.extractSnapshotPayloads(_:ssdEnabled:)` inside a
    /// `container.perform` scope. When non-nil, the pair is forwarded
    /// to the tiered store's `admitSnapshot` for SSD write-through.
    /// `nil` (the default) is the "SSD disabled" signal and leaves
    /// the radix-tree insertion as the only side effect.
    @discardableResult
    func storeLeaf(
        storedTokens: [Int],
        leafSnapshot: HybridCacheSnapshot,
        leafPayload: SnapshotPayload? = nil,
        partitionKey: CachePartitionKey,
        requestID: UUID? = nil
    ) -> StoreDiagnostics {
        guard leafSnapshot.tokenOffset == storedTokens.count else {
            return StoreDiagnostics(evictions: [], supersededLeaves: [], stats: stats)
        }

        let tree = store.getOrCreateTree(for: partitionKey)
        let node = tree.insertPath(tokens: storedTokens)
        tree.storeSnapshot(leafSnapshot, on: node)

        let supersededLeaves = supersedeAncestorLeaves(
            for: node,
            in: tree
        )

        if let leafPayload {
            registerSSDPartitionIfNeeded(for: partitionKey)
            let descriptor = makePersistedDescriptor(
                partitionKey: partitionKey,
                pathFromRoot: storedTokens,
                snapshot: leafSnapshot,
                payloadBytes: leafPayload.totalBytes
            )
            store.admitSnapshot(
                node: node,
                tree: tree,
                payload: leafPayload,
                descriptor: descriptor
            )
        }

        let evictions = evictToFitBudget(
            requestID: requestID,
            preferredPartitionKey: partitionKey
        )
        return StoreDiagnostics(
            evictions: evictions,
            supersededLeaves: supersededLeaves,
            stats: stats
        )
    }

    private func supersedeAncestorLeaves(
        for node: RadixTreeNode,
        in tree: TokenRadixTree
    ) -> [LeafSupersession] {
        var current = node.parent
        var superseded: [LeafSupersession] = []

        while let ancestor = current {
            let nextAncestor = ancestor.parent
            let snapshot = ancestor.snapshot
            let storageRef = ancestor.storageRef
            let checkpointType = snapshot?.checkpointType ?? storageRef?.checkpointType
            guard checkpointType == .leaf else {
                current = nextAncestor
                continue
            }

            let offset = snapshot?.tokenOffset ?? storageRef?.tokenOffset ?? 0
            let storageRefID = storageRef?.snapshotID
            if let ref = storageRef {
                store.deleteSnapshot(snapshotID: ref.snapshotID)
                ancestor.storageRef = nil
            }

            tree.evictSnapshot(node: ancestor)
            if ancestor.childCount == 1 {
                tree.collapseSingleChildNode(ancestor)
            }

            superseded.append(LeafSupersession(
                offset: offset,
                bodyDroppedStorageRefID: storageRefID
            ))
            current = nextAncestor
        }

        return superseded
    }

    /// Reseed a snapshot at `path` with an explicit `lastAccessTime`.
    /// Used by `AlphaTuner` to restore the production cache state into a
    /// sandbox replay cache while preserving the relative recency of
    /// each restored snapshot.
    func restoreSnapshot(
        path: [Int],
        snapshot: HybridCacheSnapshot,
        partitionKey: CachePartitionKey,
        lastAccessTime: ContinuousClock.Instant
    ) {
        let tree = store.getOrCreateTree(for: partitionKey)
        let node = tree.insertPath(tokens: path)
        tree.storeSnapshot(snapshot, on: node)
        node.lastAccessTime = lastAccessTime
    }

    /// Reattach an SSD-resident `SnapshotStorageRef` to the radix tree
    /// without touching any RAM-resident body. Mirrors
    /// `restoreSnapshot` for state-5 nodes (body absent, ref
    /// committed) — the warm-start path uses it after reading the
    /// on-disk manifest. Callers must have already validated
    /// `storageRef.committed == true`; this helper trusts the input.
    func restoreStorageRef(
        path: [Int],
        storageRef: SnapshotStorageRef,
        partitionKey: CachePartitionKey,
        lastAccessTime: ContinuousClock.Instant
    ) {
        let tree = store.getOrCreateTree(for: partitionKey)
        let node = tree.insertPath(tokens: path)
        node.storageRef = storageRef
        node.lastAccessTime = lastAccessTime
    }

    // MARK: - SSD hydration

    /// State 5 → state 4 transition: attach a freshly hydrated body
    /// to a node that currently has only a committed `storageRef`.
    /// Called from `LLMActor` after `SSDSnapshotStore.loadSync`
    /// succeeds, wrapped in `await MainActor.run { ... }` alongside
    /// the `recordHit` bump.
    ///
    /// Budget reconciliation is deferred to the next natural
    /// eviction call (every request path runs `storeSnapshots` +
    /// `storeLeaf`, which both drain the budget). Running
    /// `evictToFitBudget` inline here would add latency to the
    /// SSD-hit hot path without any benefit — the just-promoted
    /// node has `lastAccessTime = .now` and is the least likely
    /// victim under α=0, so the budget stays above the hard cap
    /// for at most one request cycle.
    func promote(
        node: RadixTreeNode,
        snapshot: HybridCacheSnapshot,
        partitionKey: CachePartitionKey
    ) {
        guard let tree = store.tree(for: partitionKey) else { return }
        tree.storeSnapshot(snapshot, on: node)
    }

    /// Clear a state-5 node's storageRef after a hydration failure
    /// (file missing / fingerprint mismatch / decode error). LLMActor
    /// calls this on the MainActor hop so subsequent lookups on the
    /// same path miss cleanly instead of re-attempting hydration on
    /// a broken file.
    ///
    /// If the node now has no body, no ref, and no children, it is
    /// also removed from the radix tree so it does not leak as a
    /// structural ghost. Pending refs (state 3) never reach this
    /// path — only state-5 hydrations can fail.
    func clearStorageRef(
        node: RadixTreeNode,
        partitionKey: CachePartitionKey
    ) {
        node.storageRef = nil
        guard node.snapshot == nil,
              node.isLeaf,
              let tree = store.tree(for: partitionKey)
        else { return }
        tree.evictNode(node: node)
    }

    // MARK: - Warm start

    /// Restore the radix-tree structure from the SSD manifest so
    /// lookups issued before any fresh capture can hit SSD-resident
    /// snapshots from a previous process.
    ///
    /// Called once per model load from
    /// `LLMActor.ensurePrefixCache()` right after the manager is
    /// constructed. No bodies are loaded — state-5 nodes hydrate
    /// lazily on first lookup via `SSDSnapshotStore.loadSync`. When
    /// the store has no SSD tier configured (`ssdStore == nil`) this
    /// is a no-op.
    ///
    /// Partition fingerprints are validated against the currently
    /// loaded model's fingerprint inside `warmStartLoad`; mismatched
    /// partitions are skipped here and their on-disk directories
    /// are scheduled for async cleanup by the store.
    /// Block until the SSD writer's pending queue drains and the
    /// manifest is persisted to disk. Called by `LLMActor` /
    /// benchmark restart scenarios before an unload so in-flight
    /// writes survive the teardown. No-op when SSD is disabled.
    func flushSSDWrites() async {
        await store.ssdStore?.flushAsync()
    }

    func warmStart(modelFingerprint: String) async throws {
        guard let ssdStore = store.ssdStore else { return }

        let started = Date.timeIntervalSinceReferenceDate
        let outcome = ssdStore.warmStartLoad(expectedFingerprint: modelFingerprint)

        let now: ContinuousClock.Instant = .now
        for partition in outcome.validPartitions {
            let partitionKey = CachePartitionKey(
                modelID: partition.meta.modelID,
                kvBits: partition.meta.kvBits,
                kvGroupSize: partition.meta.kvGroupSize,
                modelFingerprint: partition.meta.modelFingerprint
            )
            // Register with the store so subsequent admissions do
            // not trip the `rejectedUnregisteredPartition` guard.
            store.registerPartition(partition.meta, for: partitionKey)

            for descriptor in partition.descriptors {
                guard let checkpointType = HybridCacheSnapshot.CheckpointType(
                    wireString: descriptor.checkpointType
                ) else { continue }
                let ref = SnapshotStorageRef(
                    snapshotID: descriptor.snapshotID,
                    partitionDigest: descriptor.partitionDigest,
                    tokenOffset: descriptor.tokenOffset,
                    checkpointType: checkpointType,
                    bytesOnDisk: descriptor.bytes,
                    lastAccessTime: now,
                    committed: true
                )
                restoreStorageRef(
                    path: descriptor.pathFromRoot,
                    storageRef: ref,
                    partitionKey: partitionKey,
                    lastAccessTime: now
                )
            }
        }

        let snapshotCount = outcome.validPartitions.reduce(0) {
            $0 + $1.descriptors.count
        }
        let durationSeconds = Date.timeIntervalSinceReferenceDate - started

        // Emit a fingerprint mismatch event per invalidated partition
        // BEFORE the warm-start summary so a `grep` walk of the log
        // shows the failures contributing to the summary count.
        for digest in outcome.invalidatedPartitionDigests {
            PrefixCacheDiagnostics.logSystem(
                PrefixCacheDiagnostics.FingerprintMismatchEvent(partition: digest)
            )
        }
        PrefixCacheDiagnostics.logSystem(
            PrefixCacheDiagnostics.WarmStartCompleteEvent(
                partitionCount: outcome.validPartitions.count,
                snapshotCount: snapshotCount,
                invalidatedPartitionCount: outcome.invalidatedPartitionDigests.count,
                durationSeconds: durationSeconds
            )
        )
    }

    private func makePersistedDescriptor(
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

    private func registerSSDPartitionIfNeeded(for partitionKey: CachePartitionKey) {
        guard store.ssdStore != nil else { return }
        guard let fingerprint = partitionKey.modelFingerprint else {
            assertionFailure(
                "SSD admission requires a model fingerprint on CachePartitionKey"
            )
            Log.agent.error(
                "PrefixCacheManager SSD admission skipped: missing modelFingerprint "
                + "modelID=\(partitionKey.modelID) kvBits=\(String(describing: partitionKey.kvBits)) "
                + "kvGroupSize=\(partitionKey.kvGroupSize)"
            )
            return
        }

        let meta = PartitionMeta(
            modelID: partitionKey.modelID,
            modelFingerprint: fingerprint,
            kvBits: partitionKey.kvBits,
            kvGroupSize: partitionKey.kvGroupSize,
            createdAt: Date().timeIntervalSinceReferenceDate,
            schemaVersion: SnapshotManifestSchema.currentVersion
        )
        store.registerPartition(meta, for: partitionKey)
    }

    // MARK: - Tuner integration

    /// Forward a request lifecycle record to the attached `AlphaTuner`,
    /// if any. Called once per request from the agent layer after all
    /// store-side activity (mid-prefill captures, leaf store or skip)
    /// has completed. The manager builds the `RequestRecord` from raw
    /// inputs so the agent layer doesn't have to know about
    /// `AlphaTuner.SnapshotMetadata`.
    func recordRequest(
        partitionKey: CachePartitionKey,
        promptTokens: [Int],
        capturedSnapshots: [HybridCacheSnapshot],
        leafStore: AlphaTuner.LeafStore?,
        requestID: UUID? = nil
    ) {
        guard let alphaTuner else { return }
        switch pendingBootstrapBoundary {
        case .unscoped?:
            pendingBootstrapBoundary = nil
            if case .waitingForFirstEviction = alphaTuner.phase {
                alphaTuner.notifyFirstEviction(startingInventory: collectSnapshotInventory())
                return
            }
        case .request(let pendingRequestID)? where pendingRequestID == requestID:
            pendingBootstrapBoundary = nil
            if case .waitingForFirstEviction = alphaTuner.phase {
                alphaTuner.notifyFirstEviction(startingInventory: collectSnapshotInventory())
                return
            }
        default:
            break
        }

        let metadata = capturedSnapshots.map { snap in
            AlphaTuner.SnapshotMetadata(
                offset: snap.tokenOffset,
                bytes: snap.memoryBytes,
                type: snap.checkpointType
            )
        }
        alphaTuner.recordRequest(AlphaTuner.RequestRecord(
            partitionKey: partitionKey,
            promptTokens: promptTokens,
            midPrefillSnapshots: metadata,
            leafStore: leafStore
        ))
    }

    /// Walk every partition's snapshot inventory and return a flat list
    /// of `InventoryEntry` records. Used by the alpha tuner to seed the
    /// replay simulation with the production cache state at
    /// bootstrap-window start.
    func collectSnapshotInventory() -> [AlphaTuner.InventoryEntry] {
        var result: [AlphaTuner.InventoryEntry] = []
        for (key, tree) in store.orderedPartitions() {
            for node in tree.allSnapshotNodes() {
                guard let snap = node.snapshot else { continue }
                result.append(AlphaTuner.InventoryEntry(
                    partitionKey: key,
                    path: tree.pathToNode(node),
                    offset: node.tokenOffset,
                    bytes: snap.memoryBytes,
                    type: snap.checkpointType,
                    lastAccessTime: node.lastAccessTime
                ))
            }
        }
        return result
    }

    // MARK: - Eviction

    /// Drop snapshots until `totalSnapshotBytes <= memoryBudgetBytes`. Uses
    /// Marconi utility scoring (`EvictionPolicy`) for eligible nodes and
    /// falls back to oldest-first when only multi-child branch snapshots
    /// remain.
    ///
    /// `preferredPartitionKey` — the partition currently writing (i.e.,
    /// the request that triggered this drain). Eviction prefers to drop
    /// snapshots from this partition first, exhausting its eligible set
    /// before touching other partitions. When `nil`, behaves globally
    /// (Marconi default).
    @discardableResult
    func evictToFitBudget(
        requestID: UUID? = nil,
        preferredPartitionKey: CachePartitionKey? = nil
    ) -> [EvictionEvent] {
        // Pin a single clock reading and a single sorted tree order so all
        // iterations in one drain share the same recency anchor and
        // tie-break ordering.
        let now: ContinuousClock.Instant = .now
        let orderedTrees = store.orderedPartitions().map(\.tree)
        let preferredTree = preferredPartitionKey.flatMap { store.tree(for: $0) }
        var events: [EvictionEvent] = []
        while totalSnapshotBytes > memoryBudgetBytes {
            guard let candidate = findEvictionCandidate(
                now: now,
                orderedTrees: orderedTrees,
                preferredTree: preferredTree
            ),
                  let snapshot = candidate.node.snapshot
            else { break }

            candidate.tree.evictSnapshot(node: candidate.node)
            events.append(EvictionEvent(
                strategy: candidate.strategy,
                offset: candidate.node.tokenOffset,
                checkpointType: snapshot.checkpointType,
                freedBytes: snapshot.memoryBytes,
                budgetBytes: memoryBudgetBytes,
                snapshotBytesAfter: totalSnapshotBytes,
                normalizedRecency: candidate.score?.normalizedRecency,
                normalizedFlopEfficiency: candidate.score?.normalizedFlopEfficiency,
                utility: candidate.score?.utility,
                bodyDroppedStorageRefID: candidate.node.storageRef?.snapshotID
            ))

            // A live `storageRef` means the SSD tier owns a pending or
            // committed copy of the body we just dropped. Hard-removing
            // the node would orphan the persisted file — leave it
            // attached so lookup can still reach it (state 3 or 5).
            // See docs/marconi-hybrid-prefix-cache-implementation-plan.md,
            // "Eviction + demotion" section.
            if candidate.node.storageRef == nil {
                if candidate.node.isLeaf {
                    candidate.tree.evictNode(node: candidate.node)
                } else if candidate.node.childCount == 1 {
                    candidate.tree.collapseSingleChildNode(candidate.node)
                }
            }
        }
        // Mark the first request that ever triggered eviction. The
        // actual inventory snapshot is deferred until `recordRequest`,
        // after the request has finished all stores, so the bootstrap
        // start state includes a later leaf store if the first drain
        // happened during mid-prefill capture.
        if !events.isEmpty,
           let alphaTuner,
           case .waitingForFirstEviction = alphaTuner.phase,
           pendingBootstrapBoundary == nil
        {
            pendingBootstrapBoundary = requestID.map(PendingBootstrapBoundary.request) ?? .unscoped
        }
        return events
    }

    // MARK: - Stats

    var stats: CacheStats {
        var nodes = 0
        var byType: [HybridCacheSnapshot.CheckpointType: Int] = [
            .system: 0, .leaf: 0, .branchPoint: 0,
        ]
        for (_, tree) in store.orderedPartitions() {
            nodes += tree.nodeCount
            for (type, count) in tree.snapshotCountByType {
                byType[type, default: 0] += count
            }
        }
        return CacheStats(
            partitionCount: store.partitionCount,
            totalNodeCount: nodes,
            totalSnapshotBytes: totalSnapshotBytes,
            snapshotsByType: byType
        )
    }

    var totalSnapshotBytes: Int {
        store.totalSnapshotBytes
    }

    // MARK: - Private

    /// Pick one snapshot to evict from the supplied (already-sorted) trees.
    ///
    /// Strategy (in order):
    /// 1. **Preferred utility**: if a `preferredTree` is supplied, score
    ///    its Marconi-eligible nodes (snapshot + `childCount <= 1` +
    ///    non-`.system`) and return the lowest-utility one. This is the
    ///    writing-partition-first rule.
    /// 2. **Global utility**: if the preferred tree has no eligible
    ///    candidates (or none was supplied), score eligible nodes across
    ///    all partitions and return the lowest-utility one. Preserves
    ///    Marconi's "global utility" semantics for single-partition
    ///    configurations and for the spill-over case when the writing
    ///    partition is already drained.
    /// 3. **Preferred fallback**: if both utility paths are empty but
    ///    the preferred tree still has any snapshots (all ineligible —
    ///    e.g., `.system`-only), drop the oldest one from the preferred
    ///    tree.
    /// 4. **Global fallback**: drop the oldest snapshot from any tree,
    ///    including multi-child branches, so the hard budget invariant
    ///    holds in degenerate cases like a zero-budget drain.
    private func findEvictionCandidate(
        now: ContinuousClock.Instant,
        orderedTrees: [TokenRadixTree],
        preferredTree: TokenRadixTree? = nil
    ) -> EvictionCandidate? {
        // 1. Preferred utility — writing-partition-first.
        if let preferredTree {
            let preferredCandidates = preferredTree.eligibleEvictionNodes()
            if let victim = EvictionPolicy.selectVictim(
                candidates: preferredCandidates, now: now
            ) {
                return EvictionCandidate(
                    tree: preferredTree,
                    node: victim.node,
                    strategy: .utility,
                    score: victim.score
                )
            }
        }

        // 2. Global utility — spill to other partitions.
        var nodeToTree: [ObjectIdentifier: TokenRadixTree] = [:]
        var candidates: [RadixTreeNode] = []
        for tree in orderedTrees where tree !== preferredTree {
            for node in tree.eligibleEvictionNodes() {
                nodeToTree[ObjectIdentifier(node)] = tree
                candidates.append(node)
            }
        }
        if let victim = EvictionPolicy.selectVictim(candidates: candidates, now: now),
           let tree = nodeToTree[ObjectIdentifier(victim.node)]
        {
            return EvictionCandidate(
                tree: tree,
                node: victim.node,
                strategy: .utility,
                score: victim.score
            )
        }

        // 3. Preferred fallback — oldest snapshot in the writing partition.
        if let preferredTree,
           let oldest = preferredTree.allSnapshotNodes().min(
               by: { $0.lastAccessTime < $1.lastAccessTime }
           )
        {
            return EvictionCandidate(
                tree: preferredTree,
                node: oldest,
                strategy: .fallback,
                score: nil
            )
        }

        // 4. Global fallback — oldest snapshot anywhere.
        return orderedTrees
            .lazy
            .flatMap { tree in tree.allSnapshotNodes().lazy.map { (tree: tree, node: $0) } }
            .min(by: { $0.node.lastAccessTime < $1.node.lastAccessTime })
            .map { candidate in
                EvictionCandidate(
                    tree: candidate.tree,
                    node: candidate.node,
                    strategy: .fallback,
                    score: nil
                )
            }
    }
}
