import Foundation
import MLXLMCommon

/// Partition key for isolating radix trees by runtime configuration.
///
/// Tool/system digests are intentionally NOT part of the partition key:
/// different tools/context → different tokens → different radix paths →
/// naturally isolated within one partition. The **template-context digest**
/// is the one deliberate exception (issue #98): render-mode flags like the
/// **Preserve-Thinking Render** change how *the same* conversation renders
/// from the first assistant turn on, and the canonical-leaf machinery
/// assumes one render mode per partition — so toggling a flag lands in a
/// fresh partition and mixed renders never share one.
///
/// `Comparable` so partition iteration can produce a deterministic order
/// (modelID → kvBits → kvGroupSize → modelFingerprint → templateContextDigest)
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
    /// The request's `TemplateRenderContext.digest` (issue #98). Defaults to
    /// the canonical render's digest so every existing call site — and every
    /// partition persisted before the field existed — keeps its identity.
    let templateContextDigest: String

    nonisolated init(
        modelID: String,
        kvBits: Int?,
        kvGroupSize: Int,
        modelFingerprint: String? = nil,
        templateContextDigest: String = HTTPPrefixCacheConversation.defaultTemplateContextDigest
    ) {
        self.modelID = modelID
        self.kvBits = kvBits
        self.kvGroupSize = kvGroupSize
        self.modelFingerprint = modelFingerprint
        self.templateContextDigest = templateContextDigest
    }

    static func < (lhs: CachePartitionKey, rhs: CachePartitionKey) -> Bool {
        let lhsHead = (
            lhs.modelID, lhs.kvBits ?? -1, lhs.kvGroupSize,
            lhs.modelFingerprint ?? "", lhs.templateContextDigest
        )
        let rhsHead = (
            rhs.modelID, rhs.kvBits ?? -1, rhs.kvGroupSize,
            rhs.modelFingerprint ?? "", rhs.templateContextDigest
        )
        return lhsHead < rhsHead
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
    /// The live RAM-tier budget — the **Pressure-Reactive Budget**'s
    /// *current* value. Pressure events move it inside `budgetBand`;
    /// tests and the E2E hooks may still set it directly (the band then
    /// reasserts itself on the next pressure event).
    var memoryBudgetBytes: Int
    /// The ceiling/current band the pressure events fold
    /// (`PrefixCacheBudgetBand`). The floor is content-defined and
    /// computed per event — see `budgetFloorBytes()`.
    private(set) var budgetBand: PrefixCacheBudgetBand
    /// Held strongly so the subscription lives exactly as long as this
    /// cache: the handler captures `self` weakly, and dropping the
    /// manager (model unload) deallocates the source, which cancels
    /// delivery.
    private let pressureSource: (any MemoryPressureSource)?
    /// Optional adaptive `alpha` tuner. Production caches attach one;
    /// test/replay caches pass `nil` to avoid recursive recording when
    /// the tuner itself spins up sandboxed caches during grid search.
    let alphaTuner: AlphaTuner?
    /// The one mutable cell holding the eviction weighting. `flopProfile`
    /// is fixed for this cache's model; `alpha` rides the LRU default
    /// until the attached `AlphaTuner` returns a tuned winner from
    /// `recordRequest`. Every eviction and telemetry score reads it by
    /// value — there is no process global. See `CONTEXT.md` → Eviction
    /// tuning (**Eviction Configuration**).
    var evictionConfig: EvictionConfiguration

    /// Lifetime telemetry counters for this cache: hit tokens served,
    /// recovered-vs-terminal eviction outcomes, hydrations. Surfaced on
    /// the telemetry snapshot; never consulted by any policy.
    private(set) var cumulativeCounters = PromptCacheCumulativeCounters()

    /// Extracts a write-through `SnapshotPayload` from a RAM body so
    /// **Snapshot Demotion** can persist an unbacked eviction victim
    /// before dropping it. Injected (production wires the **Server
    /// Completion** module's extraction edge) because payload extraction
    /// knows the safetensors shape, which is not this layer's business.
    /// `nil` (tests, replay caches) disables demotion — every unbacked
    /// drop is terminal, today's pre-demotion behavior.
    private let demotionPayloadExtractor: ((HybridCacheSnapshot) -> SnapshotPayload?)?

    init(
        memoryBudgetBytes: Int,
        evictionConfig: EvictionConfiguration = EvictionConfiguration(),
        alphaTuner: AlphaTuner? = nil,
        tieredStore: TieredSnapshotStore? = nil,
        demotionPayloadExtractor: ((HybridCacheSnapshot) -> SnapshotPayload?)? = nil,
        pressureSource: (any MemoryPressureSource)? = nil
    ) {
        self.store = tieredStore ?? TieredSnapshotStore(ssdConfig: nil)
        self.memoryBudgetBytes = memoryBudgetBytes
        self.budgetBand = PrefixCacheBudgetBand(ceilingBytes: memoryBudgetBytes)
        self.evictionConfig = evictionConfig
        self.alphaTuner = alphaTuner
        self.demotionPayloadExtractor = demotionPayloadExtractor
        self.pressureSource = pressureSource
        pressureSource?.start { [weak self] level in
            self?.applyMemoryPressure(level)
        }
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
        /// (the **Server Completion** module) emit a
        /// `ssdBodyDrop(id:)` diagnostic event for each non-nil
        /// entry. `nil` for plain RAM-only evictions.
        let bodyDroppedSnapshotRefID: String?

        /// Non-nil when the ref-less victim node carried a **Chain-Prefix
        /// Restore** point (ADR-0012): the owning chain's head ID. The
        /// node stays reachable and re-hydrates from the chain's leading
        /// segments, so the drop is recovered, not terminal.
        let chainPrefixOwnerID: String?

        /// Recovered vs terminal — the one classification rule, shared
        /// by the manager's cumulative counters and the per-request
        /// tallies: a surviving ref or chain-prefix point means the node
        /// stays hittable (the next hit pays hydration); a backing-less
        /// drop is a terminal loss (the next hit pays full re-prefill).
        nonisolated var isTerminal: Bool {
            bodyDroppedSnapshotRefID == nil && chainPrefixOwnerID == nil
        }

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
            bodyDroppedSnapshotRefID: String? = nil,
            chainPrefixOwnerID: String? = nil
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
            self.chainPrefixOwnerID = chainPrefixOwnerID
        }
    }

    struct StoreDiagnostics: Sendable {
        let evictions: [EvictionEvent]
        let supersededLeaves: [LeafSupersession]
        let stats: CacheStats
    }

    struct LeafSupersession: Sendable {
        /// What happened to the superseded leaf's SSD backing. See
        /// `CONTEXT.md` → SSD leaf extension (three supersession modes).
        enum Mode: String, Sendable {
            /// A **Leaf Extension Admission** is taking ownership of
            /// the backing's **Segment Chain**. The transfer completes
            /// at the writer's commit, where the node ref is discarded
            /// (`TieredSnapshotStore.markSnapshotRefCommitted`); until
            /// then — and permanently, if the writer drops the
            /// extension — the backing behaves as `preserved`.
            case transferred
            /// The backing (or the whole node, when it had none) was
            /// deleted — a full SSD write replaced it.
            case deleted
            /// The backing was kept (body dropped, ref retained) — the
            /// new leaf has no SSD copy, so the ancestor remains the
            /// warm-start fallback and the next extension base.
            case preserved
        }

        let offset: Int
        let bodyDroppedSnapshotRefID: String?
        let mode: Mode
    }

    /// How `supersedeAncestorLeaves` treats the SSD backings of the
    /// ancestor leaves a fresh leaf admission supersedes.
    private enum LeafSupersessionPolicy {
        /// Full SSD write accepted — ancestor backings are replaced.
        case deleteBackings
        /// **Leaf Extension Admission** accepted — the matching base
        /// transfers its chain; any other (stale) SSD-backed ancestor
        /// is deleted.
        case transferBacking(baseID: String)
        /// The new leaf has no SSD copy (RAM-only admission, or the
        /// enqueue was rejected) — ancestor backings stay reachable as
        /// the warm-start fallback and the next turn's extension base.
        case preserveBackings
    }

    private struct EvictionCandidate {
        let partitionKey: CachePartitionKey
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
        /// A snapshot whose persisted layers fail restoration degrades to a
        /// miss (`nil`) — corrupt warm-start data must never crash a request.
        nonisolated func restoreCache() -> [any KVCache]? {
            guard let snapshot, partitionKey != nil else { return nil }
            do {
                return try snapshot.restore()
            } catch {
                Log.server.error(
                    "snapshot restore failed — treating as cache miss: \(error)"
                )
                return nil
            }
        }
    }

    enum LookupReason: CustomStringConvertible, Sendable {
        case hit(snapshotOffset: Int, totalTokens: Int, type: HybridCacheSnapshot.CheckpointType)
        /// State 5 — body absent, committed SSD ref present. LLMActor
        /// hydrates the body via `ssdStore.loadSync(...)` inside
        /// `container.perform`, then promotes the node back to state 4
        /// on MainActor.
        case ssdHit(SSDHitContext)
        /// Body absent, no own ref, but a **Chain-Prefix Restore** point
        /// (ADR-0012). LLMActor hydrates via `ssdStore.loadSyncPrefix(...)`
        /// inside `container.perform` — composing only the owning chain's
        /// leading segments — then promotes the body on MainActor.
        case chainPrefixHit(ChainPrefixHitContext)
        case missNoEntries
        case missNoSnapshotInPrefix

        nonisolated var description: String {
            switch self {
            case .hit(let offset, let total, let type):
                "hit(\(type) at \(offset)/\(total))"
            case .ssdHit(let ctx):
                "ssdHit(\(ctx.snapshotRef.checkpointType) at \(ctx.snapshotRef.tokenOffset), id=\(ctx.snapshotRef.snapshotID.prefix(8)))"
            case .chainPrefixHit(let ctx):
                "chainPrefixHit(\(ctx.point.checkpointType) at \(ctx.point.boundaryOffset), owner=\(ctx.point.ownerSnapshotID.prefix(8)))"
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
            cumulativeCounters.hitTokens += snapshot.tokenOffset
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
        // after the body branch returned, so the node is body-less.
        // `makeSSDHitContext` returns non-nil because the ref could only
        // have been assigned through the admission path, which requires
        // the SSD tier to be enabled.
        if case .ssdOnly(let ref) = node.state,
           let context = store.makeSSDHitContext(ref: ref, node: node) {
            return LookupResult(
                snapshot: nil,
                partitionKey: partitionKey,
                snapshotTokenOffset: ref.tokenOffset,
                sharedPrefixLength: treeMatchDepth,
                reason: .ssdHit(context)
            )
        }

        // Body-less, no own committed ref — the remaining hittable channel
        // is a **Chain-Prefix Restore** point (ADR-0012): the boundary is
        // backed by the owning chain's leading segments. An own ref always
        // wins over a point (checked above); both hydrate on demand.
        if let point = node.chainPrefixRestorePoint,
           let context = store.makeChainPrefixHitContext(point: point, node: node) {
            return LookupResult(
                snapshot: nil,
                partitionKey: partitionKey,
                snapshotTokenOffset: point.boundaryOffset,
                sharedPrefixLength: treeMatchDepth,
                reason: .chainPrefixHit(context)
            )
        }

        return LookupResult(
            snapshot: nil, partitionKey: partitionKey,
            snapshotTokenOffset: 0, sharedPrefixLength: treeMatchDepth,
            reason: .missNoSnapshotInPrefix
        )
    }

    /// When a lookup restores at `K` but the tree already matches farther to
    /// `M`, synthesize a checkpoint at `M` so the next request can skip the
    /// already-shared gap. This is layered on top of the existing Phase 2
    /// planner and does not change its speculative branch-point rules.
    ///
    /// The planner returns an offset, not a state. The captured
    /// `.branchPoint` snapshot at `M` is produced by the prefill loop, so the
    /// alignment checkpoint stays consistent with the request's cache type.
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
    /// Leaf checkpoint is NOT planned; leaf Snapshot Admission is captured
    /// post-generation by the extraction edge.
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

    /// Plan checkpoints against the settled tree *after* **Snapshot Resolution**,
    /// merging the Phase 3.1 alignment branch-point for `alignTo` when present.
    /// The actor resolves first via **Snapshot Resolution** (which may promote or
    /// clear a node), then plans here, so the post-hydration-failure replan
    /// becomes the ordinary single plan.
    ///
    /// Pass `alignTo: nil` to skip the alignment merge — the pre-resolution
    /// ordering planned against the unhydrated `.ssdHit`, which never aligned,
    /// so an SSD-hydrated hit passes `nil` to preserve that exact behavior.
    func planCheckpoints(
        tokens: [Int],
        stablePrefixOffset: Int?,
        partitionKey: CachePartitionKey,
        alignTo lookupResult: LookupResult?
    ) -> [(offset: Int, type: HybridCacheSnapshot.CheckpointType)] {
        let basePlan = planCheckpoints(
            tokens: tokens,
            stablePrefixOffset: stablePrefixOffset,
            partitionKey: partitionKey
        )
        guard let lookupResult,
              let alignmentOffset = alignmentCheckpointOffset(
                  lookupResult: lookupResult,
                  totalTokenCount: tokens.count,
                  plannedCheckpoints: basePlan
              )
        else { return basePlan }
        return basePlan + [(offset: alignmentOffset, type: .branchPoint)]
    }

    // MARK: - Store

    @discardableResult
    func admit(_ admission: SnapshotAdmission) -> StoreDiagnostics {
        let tree = store.getOrCreateTree(for: admission.partitionKey)
        var supersededLeaves: [LeafSupersession] = []
        let hasSSDEntry = admission.entries.contains { entry in
            if case .ramAndSSD = entry.storage { return true }
            return false
        }
        if hasSSDEntry {
            registerSSDPartitionIfNeeded(for: admission.partitionKey)
        }

        func path(for entry: SnapshotAdmission.Entry) -> [Int] {
            guard entry.path.offset < admission.fullPromptTokens.count else {
                return admission.fullPromptTokens
            }
            return Array(admission.fullPromptTokens.prefix(entry.path.offset))
        }

        func storeRAMEntry(_ entry: SnapshotAdmission.Entry) -> (node: RadixTreeNode, path: [Int]) {
            let path = path(for: entry)
            let node = tree.insertPath(tokens: path)
            tree.storeSnapshot(entry.snapshot, on: node)

            return (node, path)
        }

        // Returns the accepted ref (nil when SSD is disabled, the
        // Survival Gate skipped the write, or the front door rejected
        // the enqueue), so the leaf case can derive its supersession
        // policy from the actual admission outcome.
        @discardableResult
        func admitSSDEntry(
            _ entry: SnapshotAdmission.Entry,
            node: RadixTreeNode,
            path: [Int]
        ) -> SnapshotRef? {
            guard case .ramAndSSD(let payload) = entry.storage else {
                return nil
            }
            // The Survival Gate: checkpoint write-throughs (and any
            // leaf declared non-end-of-turn) only write if the chain
            // would survive the cut its own admission triggers.
            // End-of-turn leaves bypass — the just-finished leaf is
            // the highest-reuse object in the system.
            let bypassesGate = admission.kind == .leaf && admission.leafIsEndOfTurn
            if !bypassesGate,
               !store.survivalGateAdmits(
                   snapshot: entry.snapshot,
                   payloadTotalBytes: payload.totalBytes,
                   scoringConfig: evictionConfig
               ) {
                cumulativeCounters.survivalGateSkips += 1
                return nil
            }
            // Hand the front door domain inputs; the ledger owns the
            // descriptor schema (`SnapshotLedger.makeDescriptor`).
            // `scoringConfig` rides the write to the ledger's
            // terminal-loss cut — the one shared α (ADR-0011).
            return store.admitSnapshot(
                node: node,
                tree: tree,
                partitionKey: admission.partitionKey,
                pathFromRoot: path,
                snapshot: entry.snapshot,
                payload: payload,
                scoringConfig: evictionConfig
            )
        }

        switch admission.kind {
        case .checkpoints:
            tree.insertPath(tokens: admission.fullPromptTokens)
            for entry in admission.entries {
                let stored = storeRAMEntry(entry)
                admitSSDEntry(entry, node: stored.node, path: stored.path)
            }
        case .leaf:
            let entry = admission.entries.first
            let stored = storeRAMEntry(entry)
            // The supersession walk and the SSD enqueue need opposite
            // orders depending on the payload:
            // - **Leaf Extension Admission** and gated (non-end-of-turn)
            //   full writes: enqueue FIRST — the front door must
            //   validate (and, for extensions, shield the base) while
            //   the backings are untouched, and the walk's policy is
            //   read off whether the enqueue was accepted. An accepted
            //   extension transfers the base's backing; an accepted full
            //   write replaces history; any rejection (gate skip,
            //   budget, back-pressure) preserves the ancestor backing as
            //   the warm-start fallback and the next turn's extension
            //   base.
            // - End-of-turn full SSD write (and RAM-only): supersede
            //   FIRST, so the doomed ancestor backings free their budget
            //   before the writer's admission cut sizes up the incoming
            //   write — a near-full tier would otherwise evict an
            //   unrelated resident that didn't need to go.
            if case .ramAndSSD(let payload) = entry.storage,
               payload.extending != nil || !admission.leafIsEndOfTurn {
                let acceptedRef = admitSSDEntry(entry, node: stored.node, path: stored.path)
                let policy: LeafSupersessionPolicy = acceptedRef == nil
                    ? .preserveBackings
                    : payload.extending.map { .transferBacking(baseID: $0.baseSnapshotID) }
                        ?? .deleteBackings
                supersededLeaves.append(contentsOf: supersedeAncestorLeaves(
                    for: stored.node,
                    in: tree,
                    policy: policy
                ))
            } else {
                let policy: LeafSupersessionPolicy
                switch entry.storage {
                case .ramOnly:
                    policy = .preserveBackings
                case .ramAndSSD:
                    // Unconditional delete, even if the enqueue below is
                    // then rejected — the pre-extension behavior. A full
                    // write declares the ancestor history replaced; the
                    // rare rejection (oversized payload, unregistered
                    // partition) costs the warm-start fallback, never
                    // correctness. (End-of-turn leaves bypass the
                    // Survival Gate, so a gate skip can never be the
                    // rejection here.)
                    policy = .deleteBackings
                }
                supersededLeaves.append(contentsOf: supersedeAncestorLeaves(
                    for: stored.node,
                    in: tree,
                    policy: policy
                ))
                admitSSDEntry(entry, node: stored.node, path: stored.path)
            }
        }

        let evictions = evictToFitBudget(
            requestID: admission.requestID,
            preferredPartitionKey: admission.partitionKey
        )
        return StoreDiagnostics(
            evictions: evictions,
            supersededLeaves: supersededLeaves,
            stats: stats
        )
    }

    private func supersedeAncestorLeaves(
        for node: RadixTreeNode,
        in tree: TokenRadixTree,
        policy: LeafSupersessionPolicy
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
            var mode: LeafSupersession.Mode = .deleted

            // The tree is the sole mutator; both wrappers are strict, so
            // call each only when the captured state says the transition
            // is applicable. What happens to the SSD backing rides the
            // policy:
            // - transfer: the matching base's chain will belong to the
            //   new leaf once the writer's fold commits. Until then the
            //   base ref stays live — it is the warm-start fallback and
            //   the hit target for the whole pending window, and the
            //   writer can still drop the extension (budget cut, I/O
            //   failure). `TieredSnapshotStore.markSnapshotRefCommitted`
            //   discards the ref when the fold is durable; a writer
            //   drop leaves it intact (the transfer degrades to
            //   `preserved`). Discarding here would strand the base:
            //   manifest-resident but tree-unreachable.
            // - preserve: keep the ref (and so the node) — the new leaf
            //   has no SSD copy, so the ancestor stays the warm-start
            //   fallback and the next turn's extension base.
            // - delete: the pre-extension behavior — delete the SSD
            //   backing first so discarding the ref cannot orphan a
            //   file + manifest entry.
            if let snapshotRefID {
                switch policy {
                case .transferBacking(let baseID) where snapshotRefID == baseID:
                    mode = .transferred
                case .preserveBackings:
                    mode = .preserved
                default:
                    store.deleteSnapshot(snapshotID: snapshotRefID)
                    tree.discardSnapshotRefAfterExplicitDelete(node: ancestor)
                    mode = .deleted
                }
            }
            if state.hasResidentBody {
                tree.dropBody(node: ancestor)
            }

            superseded.append(LeafSupersession(
                offset: offset,
                bodyDroppedSnapshotRefID: snapshotRefID,
                mode: mode
            ))
            current = nextAncestor
        }

        return superseded
    }

    /// The base a **Leaf Extension Admission** for `tokens` would slice
    /// against: the deepest SSD-backed strict-ancestor leaf on the
    /// path. Read-only — the SSD front door re-validates (and shields)
    /// the base at enqueue, so a stale answer degrades to a rejected
    /// enqueue, never a broken chain. `nil` when SSD is disabled or no
    /// such ancestor exists (then the leaf admits full).
    func extensionBase(
        tokens: [Int],
        partitionKey: CachePartitionKey
    ) -> SnapshotExtension? {
        guard store.isSSDEnabled,
              let tree = store.tree(for: partitionKey),
              let ref = tree.deepestRefBearingLeaf(tokens: tokens)
        else { return nil }
        return SnapshotExtension(
            baseSnapshotID: ref.snapshotID,
            baseOffset: ref.tokenOffset
        )
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
    /// construction. Routes through `store.restoreCommittedRef` so the
    /// restore uses the same sole-mutator seam as every other transition
    /// *and* lands in the router's committed index — a later SSD-tier
    /// eviction of the restored resident must clear this ref eagerly.
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
    /// Returns whether the ref landed on the tree — `false` means the
    /// collision path dropped (and reclaimed) the descriptor, so the
    /// caller must not build anything else on it (issue #99: no restore
    /// points for a dropped owner).
    @discardableResult
    func restoreSnapshotRef(
        path: [Int],
        snapshotRef: SnapshotRef,
        partitionKey: CachePartitionKey,
        lastAccessTime: ContinuousClock.Instant
    ) -> Bool {
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
            return false
        }
        store.restoreCommittedRef(node: node, tree: tree, ref: snapshotRef)
        node.lastAccessTime = lastAccessTime
        return true
    }

    /// Reconstruct the **Chain-Prefix Restore** points of one restored
    /// chain head (ADR-0012, issue #99): every inherited-segment boundary
    /// was a historical leaf extent the commit-time fold consumed, so each
    /// gets a point owned by the head — the restore floor survives the
    /// restart. `prefixBytes` accumulates the shallow→deep segment sizes,
    /// matching the consumed base's chain total the in-session conversion
    /// records. Both warm-start paths (manifest load and directory-walk
    /// rebuild) arrive here with already-validated descriptors; condemned
    /// chains never reach this point.
    private func restoreChainPrefixRestorePoints(
        descriptor: PersistedSnapshotDescriptor,
        checkpointType: HybridCacheSnapshot.CheckpointType,
        partitionKey: CachePartitionKey
    ) {
        var prefixBytes = 0
        var points: [ChainPrefixRestorePoint] = []
        for segment in descriptor.inheritedSegments {
            prefixBytes += segment.bytes
            guard segment.tokenOffset > 0,
                  segment.tokenOffset < descriptor.tokenOffset,
                  segment.tokenOffset <= descriptor.pathFromRoot.count
            else {
                Log.agent.warning(
                    "PrefixCacheManager warmStart: skipping restore point at "
                    + "inconsistent boundary=\(segment.tokenOffset) "
                    + "(chain extent=\(descriptor.tokenOffset), "
                    + "path=\(descriptor.pathFromRoot.count)) "
                    + "owner=\(descriptor.snapshotID.prefix(8))"
                )
                continue
            }
            points.append(
                ChainPrefixRestorePoint(
                    ownerSnapshotID: descriptor.snapshotID,
                    boundaryOffset: segment.tokenOffset,
                    prefixBytes: prefixBytes,
                    checkpointType: checkpointType,
                    partitionDigest: descriptor.partitionDigest
                )
            )
        }
        store.restoreChainPrefixRestorePoints(
            points: points,
            ownerPath: descriptor.pathFromRoot,
            partitionKey: partitionKey
        )
    }

    // MARK: - SSD hydration

    /// State 5 → state 4 transition: attach a freshly hydrated body
    /// to a node that currently has only a committed ref. Called from
    /// `LLMActor` after `SSDSnapshotStore.loadSync` succeeds, wrapped in
    /// `await MainActor.run { ... }` alongside the `recordHit` bump.
    ///
    /// Budget reconciliation is deferred to the next natural
    /// eviction call (request write paths route through `admit`,
    /// which drains the budget). Running
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
            return
        }
        // SSD hit materialized: the hydration count and the restored
        // offset land in the lifetime counters here (not at lookup)
        // so a failed hydration never inflates them.
        cumulativeCounters.hydrations += 1
        cumulativeCounters.hitTokens += snapshot.tokenOffset
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
        let effect = tree.clearCommittedSnapshotRefAfterBackingLoss(node: node)
        if case .ignored(let reason) = effect {
            Log.agent.debug(
                "PrefixCacheManager.clearCommittedSnapshotRefAfterHydrationFailure: ignored — "
                + "node left committed/ssdOnly before the failure hop "
                + "(reason=\(String(describing: reason)))"
            )
        }
    }

    /// Attach a freshly composed chain-prefix body (ADR-0012) to the
    /// pointed node. The chain-side sibling of `promote`: the node holds
    /// no own ref (its backing is borrowed from the owner chain's leading
    /// segments), so this stores the body through `storeSnapshot` rather
    /// than the `hydrate` transition, and the restore point stays — still
    /// valid for re-hydration after the next body eviction. Forgiving: a
    /// body that appeared in the off-main window wins.
    func promoteChainPrefix(
        node: RadixTreeNode,
        snapshot: HybridCacheSnapshot,
        partitionKey: CachePartitionKey
    ) {
        guard let tree = store.tree(for: partitionKey) else { return }
        guard node.state.body == nil else {
            Log.agent.debug(
                "PrefixCacheManager.promoteChainPrefix: node grew a body before the "
                + "promote hop (state=\(node.state.label)) — keeping the newer body"
            )
            return
        }
        tree.storeSnapshot(snapshot, on: node)
        cumulativeCounters.hydrations += 1
        cumulativeCounters.hitTokens += snapshot.tokenOffset
    }

    /// Clear a **Chain-Prefix Restore** point after its hydration failed
    /// (owner evicted mid-window, boundary off the segment grid, broken
    /// leading segment). Subsequent lookups degrade to the next-shallower
    /// backing instead of re-attempting a broken compose. The tree
    /// self-heals when the point was all the node had left.
    func clearChainPrefixRestorePointAfterHydrationFailure(
        node: RadixTreeNode,
        partitionKey: CachePartitionKey
    ) {
        guard let tree = store.tree(for: partitionKey) else { return }
        store.clearChainPrefixRestorePoint(node: node, tree: tree)
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
                templateContextDigest: partition.meta.templateContextDigest
                    ?? HTTPPrefixCacheConversation.defaultTemplateContextDigest
            )
            // The reconstructed key must match the on-disk digest. A mismatch
            // signals a corrupted/inconsistent meta sidecar — drop the
            // partition rather than reattaching it under the wrong key and
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
                    bytesOnDisk: descriptor.totalBytes
                )
                let landed = restoreSnapshotRef(
                    path: descriptor.pathFromRoot,
                    snapshotRef: ref,
                    partitionKey: partitionKey,
                    lastAccessTime: now
                )
                guard landed, !descriptor.inheritedSegments.isEmpty else { continue }
                restoreChainPrefixRestorePoints(
                    descriptor: descriptor,
                    checkpointType: checkpointType,
                    partitionKey: partitionKey
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
            templateContextDigest: partitionKey.templateContextDigest
                == HTTPPrefixCacheConversation.defaultTemplateContextDigest
                ? nil : partitionKey.templateContextDigest
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
        // The tuner returns its tuned winner exactly once, on the call
        // that completes the grid search; assign it to the one mutable
        // cell. No global write, no back-reference from the tuner.
        if let tunedAlpha = alphaTuner.recordRequest(AlphaTuner.RequestRecord(
            partitionKey: partitionKey,
            promptTokens: promptTokens,
            midPrefillSnapshots: metadata,
            leafStore: leafStore
        )) {
            evictionConfig.alpha = tunedAlpha
        }
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

    // MARK: - Pressure-Reactive Budget

    /// Fold one OS memory-pressure event into the budget band and act
    /// on it: a shrink drains down to the new current budget with the
    /// **Budget Floor** protected (every drop demoting via **Snapshot
    /// Demotion** where backing is available); a regrowth just raises
    /// the budget — the cache refills naturally on subsequent
    /// admissions. Invoked by the injected `MemoryPressureSource`;
    /// callable directly by tests through the in-memory peer or
    /// as-is.
    func applyMemoryPressure(_ level: MemoryPressureLevel) {
        let floor = floorContents().bytes
        let previous = budgetBand.currentBytes
        budgetBand = budgetBand.folding(level, floorBytes: floor)
        guard budgetBand.currentBytes != previous else { return }
        memoryBudgetBytes = budgetBand.currentBytes
        if budgetBand.currentBytes < previous {
            evictToFitBudget(respectingFloor: true)
        }
    }

    /// The **Budget Floor** in bytes: what it costs to keep the floor
    /// contents resident right now. Content-defined and recomputed per
    /// pressure event, never a stored constant.
    func budgetFloorBytes() -> Int {
        floorContents().bytes
    }

    /// The floor's membership and cost in one walk: every `.system`
    /// body (the cross-conversation chains every tree is built on) plus
    /// the single most-recently-extended `.leaf` body across all
    /// partitions (the snapshot that buys the next turn's near-instant
    /// TTFT). Deliberately minimal and dumb — a last-resort survival
    /// set, never the protection mechanism (ADR-0011).
    private func floorContents() -> (nodes: Set<ObjectIdentifier>, bytes: Int) {
        var nodes: Set<ObjectIdentifier> = []
        var bytes = 0
        var freshestLeaf: RadixTreeNode?
        for (_, tree) in store.orderedPartitions() {
            for node in tree.allSnapshotNodes() {
                guard let body = node.state.body else { continue }
                switch body.checkpointType {
                case .system:
                    nodes.insert(ObjectIdentifier(node))
                    bytes += body.memoryBytes
                case .leaf:
                    if freshestLeaf == nil
                        || node.lastAccessTime > freshestLeaf!.lastAccessTime {
                        freshestLeaf = node
                    }
                case .branchPoint:
                    break
                }
            }
        }
        if let freshestLeaf, let body = freshestLeaf.state.body {
            nodes.insert(ObjectIdentifier(freshestLeaf))
            bytes += body.memoryBytes
        }
        return (nodes, bytes)
    }

    // MARK: - Eviction

    /// Drop snapshots until `totalSnapshotBytes <= memoryBudgetBytes`. Uses
    /// Marconi utility scoring (`EvictionPolicy`) for eligible nodes and
    /// falls back to oldest-first when only multi-child branch snapshots
    /// remain.
    ///
    /// **Snapshot Demotion** is the first response to every shrink: an
    /// unbacked victim is persisted to SSD (when the tier and the payload
    /// extractor are available) *before* its RAM body drops, so the loss
    /// is recovered — the next hit pays a hydration, not a re-prefill.
    /// Terminal drop is the fallback when SSD backing is unavailable.
    ///
    /// `preferredPartitionKey` — the partition currently writing (i.e.,
    /// the request that triggered this drain). Eviction prefers to drop
    /// snapshots from this partition first, exhausting its eligible set
    /// before touching other partitions. When `nil`, behaves globally
    /// (Marconi default).
    ///
    /// `respectingFloor` — `true` only for pressure-driven drains: the
    /// **Budget Floor** members (`.system` bodies + the single
    /// most-recently-extended leaf) are never victims, so a critical
    /// shrink stops at the survival set. Ordinary admission drains and
    /// the zero-budget test/E2E drains keep the unconditional semantics.
    @discardableResult
    func evictToFitBudget(
        requestID: UUID? = nil,
        preferredPartitionKey: CachePartitionKey? = nil,
        respectingFloor: Bool = false
    ) -> [EvictionEvent] {
        // Pin a single clock reading and a single sorted tree order so all
        // iterations in one drain share the same recency anchor and
        // tie-break ordering.
        let now: ContinuousClock.Instant = .now
        let orderedPartitions = store.orderedPartitions()
        let preferred = preferredPartitionKey.flatMap { key in
            store.tree(for: key).map { (key: key, tree: $0) }
        }
        let protected = respectingFloor ? floorContents().nodes : []
        var events: [EvictionEvent] = []
        while totalSnapshotBytes > memoryBudgetBytes {
            guard let candidate = findEvictionCandidate(
                now: now,
                orderedPartitions: orderedPartitions,
                preferred: preferred,
                protected: protected
            ) else { break }

            // Demote-don't-drop: give an unbacked victim an SSD pending
            // ref before the body drop so the drop settles recoverable
            // (state 1 → 2 → 3) instead of terminal (state 1 → removed).
            demoteBeforeDrop(candidate, now: now)

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
            let event = EvictionEvent(
                strategy: candidate.strategy,
                offset: offset,
                checkpointType: droppedType,
                freedBytes: result.droppedBodyBytes,
                budgetBytes: memoryBudgetBytes,
                snapshotBytesAfter: totalSnapshotBytes,
                normalizedRecency: candidate.score?.normalizedRecency,
                normalizedFlopEfficiency: candidate.score?.normalizedFlopEfficiency,
                utility: candidate.score?.utility,
                bodyDroppedSnapshotRefID: result.refID,
                chainPrefixOwnerID: result.refID == nil
                    ? candidate.node.chainPrefixRestorePoint?.ownerSnapshotID
                    : nil
            )
            if event.isTerminal {
                cumulativeCounters.terminalEvictions += 1
            } else {
                cumulativeCounters.recoveredEvictions += 1
            }
            events.append(event)
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
            tree.makeTopologySnapshot(partition: key, now: clockNow, config: evictionConfig)
        }
        let snapshotsByType = Dictionary(
            uniqueKeysWithValues: cacheStats.snapshotsByType.map { ($0.key.wireString, $0.value) }
        )

        let tuner: PromptCacheTunerSnapshot
        if let alphaTuner {
            tuner = PromptCacheTunerSnapshot(
                phase: alphaTuner.phase.wireString,
                alpha: evictionConfig.alpha,
                bootstrapProgress: alphaTuner.bootstrapWindowCount,
                bootstrapTarget: alphaTuner.bootstrapTarget
            )
        } else {
            tuner = .unavailable
        }

        return PromptCacheTelemetrySnapshot(
            capturedAt: now,
            memoryBudgetBytes: memoryBudgetBytes,
            budgetCeilingBytes: budgetBand.ceilingBytes,
            budgetFloorBytes: budgetFloorBytes(),
            residentSnapshotBytes: cacheStats.totalSnapshotBytes,
            partitionCount: cacheStats.partitionCount,
            totalNodeCount: cacheStats.totalNodeCount,
            snapshotCount: cacheStats.snapshotCount,
            snapshotsByType: snapshotsByType,
            ssd: store.ssdDiagnosticsSnapshot(),
            tuner: tuner,
            counters: cumulativeCounters,
            estimates: evictionConfig.estimates,
            trees: trees
        )
    }

    /// Test-only: zero the lifetime counters so a scenario can assert
    /// deltas from a known baseline.
    func cumulativeCountersResetForTesting() {
        cumulativeCounters = PromptCacheCumulativeCounters()
    }

    // MARK: - Measured-seconds estimators

    /// Fold one observed prefill into the rolling FLOPs/s estimate the
    /// **Eviction Configuration** carries. Called by the **Server
    /// Completion** module after each real chunked prefill.
    func recordPrefillMeasurement(flops: Double, seconds: Double) {
        evictionConfig.estimates = evictionConfig.estimates
            .recordingPrefill(flops: flops, seconds: seconds)
    }

    /// Fold one observed SSD hydration into the rolling bytes/s
    /// estimate. Called by **Snapshot Resolution** after a successful
    /// `loadSync`.
    func recordHydrationMeasurement(bytes: Int, seconds: Double) {
        evictionConfig.estimates = evictionConfig.estimates
            .recordingHydration(bytes: bytes, seconds: seconds)
    }

    // MARK: - Private

    /// **Snapshot Demotion**: persist an unbacked eviction victim to SSD
    /// so the imminent body drop is recovered instead of terminal. A
    /// no-op — leaving the drop terminal, today's pre-demotion behavior —
    /// when the victim already has a ref, the SSD tier is disabled, no
    /// payload extractor was injected, or the partition cannot write to
    /// SSD (no model fingerprint).
    ///
    /// The enqueued descriptor carries the node's real recency converted
    /// to wall clock, and the writer's commit will not re-stamp it — the
    /// flagged invariant that a demotion never refreshes ledger recency.
    /// The front door may still reject the write (budget, back-pressure);
    /// then no ref attaches and the drop settles terminal, which is the
    /// honest outcome.
    private func demoteBeforeDrop(
        _ candidate: EvictionCandidate,
        now: ContinuousClock.Instant
    ) {
        // A chain-prefix-backed node (ADR-0012) skips demotion outright:
        // its bytes already exist on SSD as the owning chain's leading
        // segments, and writing a duplicate copy is exactly the write
        // amplification the restore-point design rejected.
        guard candidate.node.state.ref == nil,
              candidate.node.chainPrefixRestorePoint == nil,
              store.isSSDEnabled,
              let demotionPayloadExtractor,
              candidate.partitionKey.modelFingerprint != nil,
              let snapshot = candidate.node.state.body
        else { return }

        registerSSDPartitionIfNeeded(for: candidate.partitionKey)
        let ageSeconds = (now - candidate.node.lastAccessTime).seconds
        let lastAccessAt = Date().timeIntervalSinceReferenceDate - ageSeconds

        // The Survival Gate: a cold demotion that would not survive the
        // SSD cut its own admission triggers skips the write entirely —
        // churning warmer chains off the tier to store a colder one is
        // a pure loss. The drop then settles terminal, which is the
        // honest outcome. The gate scores the node's real (stale)
        // recency, the same stamp the descriptor would carry. It runs
        // *before* payload extraction: a demotion payload is always the
        // full body, so its byte count is `memoryBytes` — gating on that
        // spares a rejected victim the body-sized tensor copy.
        guard store.survivalGateAdmits(
            snapshot: snapshot,
            payloadTotalBytes: snapshot.memoryBytes,
            lastAccessAt: lastAccessAt,
            scoringConfig: evictionConfig
        ) else {
            cumulativeCounters.survivalGateSkips += 1
            return
        }
        guard let payload = demotionPayloadExtractor(snapshot) else { return }

        store.admitSnapshot(
            node: candidate.node,
            tree: candidate.tree,
            partitionKey: candidate.partitionKey,
            pathFromRoot: candidate.tree.pathToNode(candidate.node),
            snapshot: snapshot,
            payload: payload,
            demotionLastAccessAt: lastAccessAt,
            scoringConfig: evictionConfig
        )
    }

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
        orderedPartitions: [(key: CachePartitionKey, tree: TokenRadixTree)],
        preferred: (key: CachePartitionKey, tree: TokenRadixTree)? = nil,
        protected: Set<ObjectIdentifier> = []
    ) -> EvictionCandidate? {
        func unprotected(_ nodes: [RadixTreeNode]) -> [RadixTreeNode] {
            protected.isEmpty
                ? nodes
                : nodes.filter { !protected.contains(ObjectIdentifier($0)) }
        }

        // 1. Preferred utility — writing-partition-first.
        if let preferred {
            let preferredCandidates = unprotected(preferred.tree.eligibleEvictionNodes())
            if let victim = EvictionPolicy.selectVictim(
                candidates: preferredCandidates, now: now, config: evictionConfig
            ) {
                return EvictionCandidate(
                    partitionKey: preferred.key,
                    tree: preferred.tree,
                    node: victim.node,
                    strategy: .utility,
                    score: victim.score
                )
            }
        }

        // 2. Global utility — spill to other partitions.
        var partitionByNode: [ObjectIdentifier: (key: CachePartitionKey, tree: TokenRadixTree)] = [:]
        var candidates: [RadixTreeNode] = []
        for partition in orderedPartitions where partition.tree !== preferred?.tree {
            for node in unprotected(partition.tree.eligibleEvictionNodes()) {
                partitionByNode[ObjectIdentifier(node)] = partition
                candidates.append(node)
            }
        }
        if let victim = EvictionPolicy.selectVictim(
               candidates: candidates, now: now, config: evictionConfig
           ),
           let partition = partitionByNode[ObjectIdentifier(victim.node)]
        {
            return EvictionCandidate(
                partitionKey: partition.key,
                tree: partition.tree,
                node: victim.node,
                strategy: .utility,
                score: victim.score
            )
        }

        // 3. Preferred fallback — oldest snapshot in the writing partition.
        if let preferred,
           let oldest = unprotected(preferred.tree.allSnapshotNodes()).min(
               by: { $0.lastAccessTime < $1.lastAccessTime }
           )
        {
            return EvictionCandidate(
                partitionKey: preferred.key,
                tree: preferred.tree,
                node: oldest,
                strategy: .fallback,
                score: nil
            )
        }

        // 4. Global fallback — oldest snapshot anywhere.
        return orderedPartitions
            .lazy
            .flatMap { partition in
                unprotected(partition.tree.allSnapshotNodes())
                    .lazy.map { (partition: partition, node: $0) }
            }
            .min(by: { $0.node.lastAccessTime < $1.node.lastAccessTime })
            .map { candidate in
                EvictionCandidate(
                    partitionKey: candidate.partition.key,
                    tree: candidate.partition.tree,
                    node: candidate.node,
                    strategy: .fallback,
                    score: nil
                )
            }
    }
}
