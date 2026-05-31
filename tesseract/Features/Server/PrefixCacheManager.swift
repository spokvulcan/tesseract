import Foundation
import MLXLMCommon

/// Normalized TriAttention identity folded into `CachePartitionKey`.
///
/// Every `TriAttentionConfiguration` with `enabled == false` collapses
/// to `.dense` so unused budget/impl fields cannot fragment dense
/// partitions. The `.dense` case is also load-bearing for the on-disk
/// digest: a default dense key must canonicalize to the exact bytes
/// the pre-TriAttention code produced, or persisted snapshots become
/// unreachable.
nonisolated enum TriAttentionPartitionIdentity: Hashable, Sendable, Comparable, Codable {
    case dense
    case triAttention(
        budgetTokens: Int,
        calibrationArtifactIdentity: TriAttentionCalibrationArtifactIdentity?,
        implementationVersion: TriAttentionImplementationVersion,
        prefixProtectionMode: TriAttentionPrefixProtectionMode
    )

    nonisolated static func from(_ configuration: TriAttentionConfiguration) -> Self {
        guard configuration.enabled else { return .dense }
        return .triAttention(
            budgetTokens: configuration.budgetTokens,
            calibrationArtifactIdentity: configuration.calibrationArtifactIdentity,
            implementationVersion: configuration.implementationVersion,
            prefixProtectionMode: configuration.prefixProtectionMode
        )
    }

    nonisolated var isDense: Bool {
        if case .dense = self { return true }
        return false
    }

    private var sortKey: (Int, Int, String, String, String) {
        switch self {
        case .dense:
            return (0, 0, "", "", "")
        case .triAttention(let budget, let artifact, let impl, let mode):
            return (1, budget, artifact?.rawValue ?? "", impl.rawValue, mode.rawValue)
        }
    }

    nonisolated static func < (lhs: Self, rhs: Self) -> Bool {
        lhs.sortKey < rhs.sortKey
    }

    // MARK: Codable

    /// Round-trips through `TriAttentionConfiguration`'s synthesized
    /// Codable so the wire shape matches the runtime config tuple the
    /// rest of the stack already serializes. `from(_:)` collapses any
    /// disabled configuration back to `.dense` so a manifest written
    /// with disabled-but-populated fields cannot fragment dense
    /// partitions on warm start.
    nonisolated init(from decoder: Decoder) throws {
        self = .from(try TriAttentionConfiguration(from: decoder))
    }

    nonisolated func encode(to encoder: Encoder) throws {
        try asConfiguration.encode(to: encoder)
    }

    private var asConfiguration: TriAttentionConfiguration {
        switch self {
        case .dense:
            return .v1Disabled
        case .triAttention(let budget, let artifact, let impl, let mode):
            return TriAttentionConfiguration(
                enabled: true,
                budgetTokens: budget,
                calibrationArtifactIdentity: artifact,
                implementationVersion: impl,
                prefixProtectionMode: mode
            )
        }
    }
}

/// Partition key for isolating radix trees by runtime configuration.
///
/// Tool/template digests are intentionally NOT part of the partition key:
/// different tools/context → different tokens → different radix paths →
/// naturally isolated within one partition.
///
/// `Comparable` so partition iteration can produce a deterministic order
/// (modelID → kvBits → kvGroupSize → modelFingerprint → triAttention)
/// for stable tie-break behavior in eviction.
nonisolated struct CachePartitionKey: Hashable, Sendable, Comparable {
    let modelID: String
    let kvBits: Int?
    let kvGroupSize: Int
    /// Stable hex SHA-256 of the loaded model's weight files
    /// (`ModelFingerprint.computeFingerprint`). Folded in so a weight swap
    /// under the same `modelID` cannot surface stale persisted snapshots.
    /// `nil` for RAM-only test fixtures.
    let modelFingerprint: String?
    let triAttention: TriAttentionPartitionIdentity

    nonisolated init(
        modelID: String,
        kvBits: Int?,
        kvGroupSize: Int,
        modelFingerprint: String? = nil,
        triAttention: TriAttentionPartitionIdentity = .dense
    ) {
        self.modelID = modelID
        self.kvBits = kvBits
        self.kvGroupSize = kvGroupSize
        self.modelFingerprint = modelFingerprint
        self.triAttention = triAttention
    }

    static func < (lhs: CachePartitionKey, rhs: CachePartitionKey) -> Bool {
        let lhsHead = (lhs.modelID, lhs.kvBits ?? -1, lhs.kvGroupSize, lhs.modelFingerprint ?? "")
        let rhsHead = (rhs.modelID, rhs.kvBits ?? -1, rhs.kvGroupSize, rhs.modelFingerprint ?? "")
        if lhsHead != rhsHead { return lhsHead < rhsHead }
        return lhs.triAttention < rhs.triAttention
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
        /// because it carried a live Snapshot Ref — the RAM body was
        /// freed but the SSD-backed node remains reachable for future
        /// lookups. Consumers
        /// (`LLMActor.makeHTTPPrefixCacheGeneration`) emit a
        /// `ssdBodyDrop(id:)` diagnostic event for each non-nil
        /// entry. `nil` for plain RAM-only evictions.
        let bodyDroppedSnapshotRefID: String?

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
            bodyDroppedSnapshotRefID: String? = nil
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
            self.bodyDroppedSnapshotRefID = bodyDroppedSnapshotRefID
        }
    }

    struct StoreDiagnostics: Sendable {
        let evictions: [EvictionEvent]
        let supersededLeaves: [LeafSupersession]
        let stats: CacheStats
    }

    struct LeafSupersession: Sendable {
        let offset: Int
        let bodyDroppedSnapshotRefID: String?
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
        /// only when the hit landed on a committed Snapshot Ref —
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
        nonisolated func restoreCache(
            triAttentionRestoreContext: TriAttentionSnapshotRestoreContext? = nil
        ) -> [any KVCache]? {
            guard let snapshot, let key = partitionKey else { return nil }
            return snapshot.restore(
                kvBitsHint: key.kvBits,
                kvGroupSizeHint: key.kvGroupSize,
                triAttentionRestoreContext: triAttentionRestoreContext
            )
        }
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
                "ssdHit(\(ctx.snapshotRef.checkpointType) at \(ctx.snapshotRef.tokenOffset), id=\(ctx.snapshotRef.snapshotID.prefix(8)))"
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
        // filtered out inside `findBestSnapshot`.
        guard let (node, _) = tree.findBestSnapshot(
            tokens: tokens, includeSnapshotRefs: true
        ) else {
            let treeMatchDepth = tree.findSharedPrefixLength(tokens: tokens)
            return LookupResult(
                snapshot: nil, partitionKey: partitionKey,
                snapshotTokenOffset: 0, sharedPrefixLength: treeMatchDepth,
                reason: .missNoSnapshotInPrefix
            )
        }

        let treeMatchDepth = tree.findSharedPrefixLength(tokens: tokens)

        if let snapshot = node.state.body {
            // States 1, 2, or 4. On state 4 (committed ref + body) the
            // store bumps the SSD descriptor's `lastAccessAt` so a hot
            // RAM hit does not look stale to the SSD LRU when the body
            // is eventually dropped to state 5. `noteLookupHit` returns
            // the bumped snapshot ID, or nil for non-committed states.
            let recordedHitID = store.noteLookupHit(on: node)
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

        // State 5 — body absent, committed ref present. We reach here only
        // after the body branch returned, so the node is body-less; the
        // only body-less *hittable* state `findBestSnapshot` returns is
        // `.ssdOnly` (pending refs are filtered out), so the pattern match
        // is total for this branch. `makeSSDHitContext` returns non-nil
        // because the ref could only have been assigned through the
        // admission path, which requires the SSD tier to be enabled.
        guard case .ssdOnly(let ref) = node.state,
              let context = store.makeSSDHitContext(ref: ref, node: node) else {
            return LookupResult(
                snapshot: nil, partitionKey: partitionKey,
                snapshotTokenOffset: 0, sharedPrefixLength: treeMatchDepth,
                reason: .missNoSnapshotInPrefix
            )
        }
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
    ///
    /// TriAttention-safe: the planner returns an offset, not a state. The
    /// captured `.branchPoint` snapshot at `M` is produced by the prefill
    /// loop using whatever cache type the runtime configured — dense or
    /// TriAttention — and the partition key already isolates the two
    /// modes so a TriAttention checkpoint cannot surface to a dense lookup.
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
            // `includeSnapshotRefs: true` + the body-then-ref `checkpointType`
            // accessor so a checkpoint already persisted to SSD whose RAM
            // body was evicted (`ssdOnly`/committed, body-less) still counts
            // as stored. Without this, every same-prefix request after a warm
            // start (or a RAM-body drop) re-plans, re-extracts, and re-admits
            // the identical snapshot — `admitSnapshot` then supersedes the
            // resident ref, deleting and rewriting byte-identical content.
            guard let tree,
                  let (node, _) = tree.findBestSnapshot(
                      tokens: Array(tokens[0..<offset]),
                      updateAccess: false,
                      includeSnapshotRefs: true)
            else { return false }
            return node.tokenOffset == offset && node.state.checkpointType == type
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

    @discardableResult
    func admit(_ admission: SnapshotAdmission) -> StoreDiagnostics {
        let tree = store.getOrCreateTree(for: admission.partitionKey)
        tree.insertPath(tokens: admission.fullPromptTokens)

        for index in 0..<admission.entries.count {
            let entry = admission.entries[index]
            let path = Array(admission.fullPromptTokens.prefix(entry.path.offset))
            let node = tree.insertPath(tokens: path)
            tree.storeSnapshot(entry.snapshot, on: node)

            guard case .ramAndSSD(let payload) = entry.storage else {
                continue
            }
            registerSSDPartitionIfNeeded(for: admission.partitionKey)
            let descriptor = makePersistedDescriptor(
                partitionKey: admission.partitionKey,
                pathFromRoot: path,
                snapshot: entry.snapshot,
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
            requestID: admission.requestID,
            preferredPartitionKey: admission.partitionKey
        )
        return StoreDiagnostics(
            evictions: evictions,
            supersededLeaves: [],
            stats: stats
        )
    }

    /// Store mid-prefill snapshots captured during prepareWithCheckpoints().
    ///
    /// `snapshotPayloads` carries the pre-extracted CPU-owned bytes for
    /// each entry in `capturedSnapshots`, positionally aligned (same
    /// count, same order), produced by
    /// `LLMActor.extractSnapshotPayloads(_:ssdEnabled:)` inside the
    /// same `container.perform` scope that captured the snapshots.
    /// An empty array (the default) signals SSD disabled; RAM insertion
    /// still routes through Snapshot Admission.
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

        let payloadsAligned =
            snapshotPayloads.count == capturedSnapshots.count
        let candidates = capturedSnapshots.enumerated().map { index, snapshot in
            let storage: SnapshotAdmission.Storage =
                payloadsAligned ? .ramAndSSD(snapshotPayloads[index]) : .ramOnly
            return SnapshotAdmission.CheckpointCandidate(
                snapshot: snapshot,
                storage: storage
            )
        }

        guard let admission = SnapshotAdmission.checkpoints(
            fullPromptTokens: promptTokens,
            candidates: candidates,
            partitionKey: partitionKey,
            requestID: requestID,
        ) else {
            return StoreDiagnostics(evictions: [], supersededLeaves: [], stats: stats)
        }

        return admit(admission)
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
            let state = ancestor.state
            guard state.checkpointType == .leaf else {
                current = nextAncestor
                continue
            }

            let offset = state.tokenOffset ?? 0
            let snapshotRefID = state.refID

            // Fully remove the superseded leaf through the tree (sole
            // mutator). Delete the SSD backing first so discarding the ref
            // cannot orphan a file + manifest entry, then drop any RAM
            // body. Both tree wrappers are strict, so call each only when
            // the captured state says the transition is applicable.
            if let snapshotRefID {
                store.deleteSnapshot(snapshotID: snapshotRefID)
                tree.discardSnapshotRefAfterExplicitDelete(node: ancestor)
            }
            if state.hasResidentBody {
                tree.dropBody(node: ancestor)
            }

            superseded.append(LeafSupersession(
                offset: offset,
                bodyDroppedSnapshotRefID: snapshotRefID
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

    /// Reattach an SSD-resident `SnapshotRef` to the radix tree as an
    /// `ssdOnly` (state-5) node, without touching any RAM body. Mirrors
    /// `restoreSnapshot` — the warm-start path uses it after reading the
    /// on-disk manifest, where every persisted ref is committed by
    /// construction. Routes through `tree.restoreCommittedRef` so the
    /// restore uses the same sole-mutator seam as every other transition.
    ///
    /// Tolerates a non-empty node at `path`: a corrupted-manifest rebuild
    /// or a pre-refactor on-disk state can produce two descriptors that
    /// resolve to the same `pathFromRoot` after `insertPath` compresses
    /// edges. The first descriptor wins (last-wins would orphan the SSD
    /// file the first descriptor points at, since it would be unreachable
    /// from the live tree); the second is dropped so warm-start completes
    /// instead of aborting. The dropped descriptor's SSD backing is
    /// **reclaimed** here (`store.deleteSnapshot`) rather than leaked —
    /// `commitRestoredManifest` already seeded both descriptors into the
    /// manifest + `currentSSDBytes`, so without this the loser's file and
    /// manifest entry would linger, never hittable, inflating the SSD
    /// budget until an unrelated LRU cut happened to evict it.
    func restoreSnapshotRef(
        path: [Int],
        snapshotRef: SnapshotRef,
        partitionKey: CachePartitionKey,
        lastAccessTime: ContinuousClock.Instant
    ) {
        let tree = store.getOrCreateTree(for: partitionKey)
        let node = tree.insertPath(tokens: path)
        guard case .empty = node.state else {
            Log.agent.debug(
                "PrefixCacheManager.restoreSnapshotRef: path collision at "
                + "tokenOffset=\(snapshotRef.tokenOffset) "
                + "id=\(snapshotRef.snapshotID); node already \(node.state.label) — "
                + "reclaiming the dropped descriptor's SSD backing"
            )
            store.deleteSnapshot(snapshotID: snapshotRef.snapshotID)
            return
        }
        tree.restoreCommittedRef(node: node, ref: snapshotRef)
        node.lastAccessTime = lastAccessTime
    }

    // MARK: - SSD hydration

    /// State 5 → state 4 transition: attach a freshly hydrated body
    /// to a node that currently has only a committed ref. Called from
    /// `LLMActor` after `SSDSnapshotStore.loadSync` succeeds, wrapped in
    /// `await MainActor.run { ... }` alongside the `recordHit` bump.
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
        // Forgiving edge: the node was captured before the off-main
        // `loadSync`. If it left `ssdOnly` in that window, the hydration is
        // a logged no-op (the newer state wins) rather than a crash.
        let effect = tree.hydrate(node: node, body: snapshot)
        if case .ignored(let reason) = effect {
            Log.agent.debug(
                "PrefixCacheManager.promote: hydrate ignored — node left ssdOnly before "
                + "the promote hop (reason=\(String(describing: reason)))"
            )
        }
    }

    /// Clear a state-5 node's ref after a hydration failure (file missing
    /// / fingerprint mismatch / decode error). LLMActor calls this on the
    /// MainActor hop — the failing lookup supplies the node — so
    /// subsequent lookups on the same path miss cleanly instead of
    /// re-attempting hydration on a broken file. The tree self-heals
    /// (removes the node) when clearing empties it. Pending refs (state
    /// 3) never reach this path — only state-5 hydrations can fail.
    func clearCommittedSnapshotRefAfterHydrationFailure(
        node: RadixTreeNode,
        partitionKey: CachePartitionKey
    ) {
        guard let tree = store.tree(for: partitionKey) else { return }
        // Forgiving edge (same boundary as `promote`): if the node left the
        // committed/ssdOnly states before this hop, clearing is a logged
        // no-op. `loadSync` already deleted the on-disk backing, so a still
        // -committed node downgrading to `ramOnly` orphans nothing.
        let effect = tree.clearCommittedSnapshotRefAfterHydrationFailure(node: node)
        if case .ignored(let reason) = effect {
            Log.agent.debug(
                "PrefixCacheManager.clearCommittedSnapshotRefAfterHydrationFailure: ignored — "
                + "node left committed/ssdOnly before the failure hop "
                + "(reason=\(String(describing: reason)))"
            )
        }
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
        await store.flush()
    }

    func warmStart(modelFingerprint: String) async throws {
        guard store.isSSDEnabled else { return }

        let started = Date.timeIntervalSinceReferenceDate
        let outcome = store.warmStartLoad(expectedFingerprint: modelFingerprint)

        let now: ContinuousClock.Instant = .now
        var digestMismatchPartitions: [String] = []
        for partition in outcome.validPartitions {
            let partitionKey = CachePartitionKey(
                modelID: partition.meta.modelID,
                kvBits: partition.meta.kvBits,
                kvGroupSize: partition.meta.kvGroupSize,
                modelFingerprint: partition.meta.modelFingerprint,
                triAttention: partition.meta.triAttention
            )
            // `PartitionMeta` (v5) carries the TriAttention identity, so
            // the reconstructed key matches the on-disk digest for both
            // dense and TriAttention partitions. A mismatch now signals
            // a corrupted/inconsistent meta sidecar — drop the partition
            // rather than reattaching it under the wrong key and
            // cross-contaminating at lookup time.
            guard partition.digest == partitionKey.partitionDigest else {
                digestMismatchPartitions.append(partition.digest)
                continue
            }
            // Register with the store so subsequent admissions do
            // not trip the `rejectedUnregisteredPartition` guard.
            store.registerPartition(partition.meta, for: partitionKey)

            for descriptor in partition.descriptors {
                guard let checkpointType = HybridCacheSnapshot.CheckpointType(
                    wireString: descriptor.checkpointType
                ) else { continue }
                let ref = SnapshotRef(
                    snapshotID: descriptor.snapshotID,
                    partitionDigest: descriptor.partitionDigest,
                    tokenOffset: descriptor.tokenOffset,
                    checkpointType: checkpointType,
                    bytesOnDisk: descriptor.bytes
                )
                restoreSnapshotRef(
                    path: descriptor.pathFromRoot,
                    snapshotRef: ref,
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
        for digest in digestMismatchPartitions {
            Log.agent.warning(
                "PrefixCacheManager warmStart dropping partition \(digest): "
                + "on-disk digest does not match digest reconstructed from PartitionMeta"
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
        guard store.isSSDEnabled else { return }
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
            schemaVersion: SnapshotManifestSchema.currentVersion,
            triAttention: partitionKey.triAttention
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
                guard let snap = node.state.body else { continue }
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
            ) else { break }

            // One chokepoint: `dropBody` reconciles the budget, carries
            // out the eviction telemetry inputs, and self-heals the
            // topology. A node whose ref survives the drop (state 2/4 →
            // 3/5) settles in place; a ref-less node (state 1) empties and
            // the tree removes it. The orphan guard that used to gate
            // node removal here is gone — `canEvictNode` is unrepresentable
            // when violated, so there is nothing for a caller to forget.
            //
            // `findEvictionCandidate` filters body-less nodes, so
            // `dropBody` is guaranteed non-ignored and returns the
            // dropped checkpoint type.
            let offset = candidate.node.tokenOffset
            let result = candidate.tree.dropBody(node: candidate.node)
            guard let droppedType = result.droppedCheckpointType else {
                preconditionFailure("dropBody returned no checkpoint type for a body-bearing node")
            }
            events.append(EvictionEvent(
                strategy: candidate.strategy,
                offset: offset,
                checkpointType: droppedType,
                freedBytes: result.droppedBodyBytes,
                budgetBytes: memoryBudgetBytes,
                snapshotBytesAfter: totalSnapshotBytes,
                normalizedRecency: candidate.score?.normalizedRecency,
                normalizedFlopEfficiency: candidate.score?.normalizedFlopEfficiency,
                utility: candidate.score?.utility,
                bodyDroppedSnapshotRefID: result.refID
            ))
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

    func makeTelemetrySnapshot(now: Date = Date()) -> PromptCacheTelemetrySnapshot {
        let cacheStats = stats
        let clockNow: ContinuousClock.Instant = .now
        let trees = store.orderedPartitions().map { key, tree in
            tree.makeTopologySnapshot(partition: key, now: clockNow)
        }
        let snapshotsByType = Dictionary(
            uniqueKeysWithValues: cacheStats.snapshotsByType.map { ($0.key.wireString, $0.value) }
        )

        return PromptCacheTelemetrySnapshot(
            capturedAt: now,
            memoryBudgetBytes: memoryBudgetBytes,
            residentSnapshotBytes: cacheStats.totalSnapshotBytes,
            partitionCount: cacheStats.partitionCount,
            totalNodeCount: cacheStats.totalNodeCount,
            snapshotCount: cacheStats.snapshotCount,
            snapshotsByType: snapshotsByType,
            ssd: store.ssdDiagnosticsSnapshot(),
            trees: trees
        )
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
