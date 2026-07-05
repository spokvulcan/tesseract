//
//  TieredSnapshotStore.swift
//  tesseract
//
//  Composes the RAM tier (an inline per-partition `TokenRadixTree`
//  dictionary) with `SSDSnapshotStore` behind one interface, and acts as
//  the SSD **router**: it resolves the writer's global `snapshotID` callbacks to
//  the owning `(node, tree)` and forwards them to the tree's transition
//  methods. It is **not** a mutator of node state — that is the tree's
//  job alone. The cross-partition ref indexes (`pendingRefsByID`,
//  `committedRefsByID`) live here because the writer's callback carries
//  only a global ID and the tree is per-partition, so the index cannot
//  live in any single tree.
//
//  **No selective write-through.** Every admission enqueues to SSD
//  unconditionally; the writer runs the admission-time type-protected
//  LRU cut inside `SSDSnapshotStore` when the budget is tight. This store
//  forwards payloads + descriptors to the writer and routes the returned
//  `SnapshotRef` into `tree.admit`.
//
//  **MainActor-isolated by construction.** All routing (pending map
//  writes, callback forwarding) happens on MainActor. The writer's commit
//  / drop callbacks fire from off-main context and hop back via
//  `Task { @MainActor in self?.markSnapshotRefCommitted(...) }` —
//  see `init(ssdConfig:)`.
//
//  **Node removal is the tree's self-heal.** When a drop empties a node
//  (state 3 → removed), `tree.dropRef` removes it respecting topology.
//  This store no longer reaches into node fields or runs radix cleanup
//  itself — the inverted dependency (SSD tier mutating tree-owned state)
//  is gone.
//

import Foundation
import MLXLMCommon

// MARK: - TieredSnapshotStore

/// Two-tier snapshot store — `PrefixCacheManager`'s per-partition
/// `TokenRadixTree` collection. The RAM tier is a plain dictionary:
/// partitions spring into existence on first write and live for the
/// process lifetime (explicit discard is a tiered concern, not a
/// RAM-tier one). The SSD tier is an optional `SSDSnapshotStore` that
/// takes ownership of payload write-through, the admission-time LRU
/// cut, and the debounced manifest persist. When `ssdConfig == nil`
/// (or `ssdConfig.enabled == false`) the store collapses to pure
/// RAM-only behavior: `admitSnapshot` is a no-op that returns `nil`.
@MainActor
final class TieredSnapshotStore {

    // MARK: - Types

    /// Stable link from a pending SSD write to the radix node it
    /// will attach to. Held strongly in `pendingRefsByID` so the
    /// writer's commit / drop callback can always find the node,
    /// even after RAM eviction would have dropped the last strong
    /// reference from the tree. Cleared on commit or drop.
    private struct PendingRef {
        let node: RadixTreeNode
        let tree: TokenRadixTree
    }

    /// Link from a **committed** Snapshot Ref to its node, kept so the
    /// SSD tier's eviction of a committed resident (`.evictedByLRU` /
    /// `.hydrationFailure`) clears the tree's ref *eagerly* — eviction
    /// scoring, **Snapshot Demotion**, and the recovered/terminal
    /// telemetry all treat `node.state.ref != nil` as "backed", so a
    /// stale ref misprices the node and skips its demotion. Weak on
    /// both ends: the tree owns node lifetime, and a dead entry just
    /// means the node is already gone — the clear no-ops.
    private struct CommittedRefOwner {
        weak var node: RadixTreeNode?
        weak var tree: TokenRadixTree?
    }

    // MARK: - Stored state

    /// The RAM tier: every live per-partition tree, keyed by its
    /// **Cache Partition Key**.
    private var trees: [CachePartitionKey: TokenRadixTree] = [:]
    /// The SSD tier. Fully private: callers drive SSD behaviour through
    /// this store's own interface (`noteLookupHit`, `flush`,
    /// `warmStartLoad`, `makeSSDHitContext`, `isSSDEnabled`) and never
    /// name `SSDSnapshotStore` themselves. Assigned once in `init` and
    /// never reassigned thereafter.
    private var ssdStore: SSDSnapshotStore?
    private var pendingRefsByID: [String: PendingRef] = [:]
    private var committedRefsByID: [String: CommittedRefOwner] = [:]
    /// New snapshot ID → the superseded ancestor backings whose deletion
    /// waits for that write's durable commit (enqueue-before-delete,
    /// ADR-0019 — the vLLM offload-safety invariant: never free the
    /// source until the save is durable). A writer drop forgets the
    /// record, degrading the ancestors to `preserved` (the warm-start
    /// fallback stays alive); a commit executes the deletions here.
    private var deferredSupersessionsByID: [String: [DeferredSupersession]] = [:]

    /// One superseded ancestor awaiting its replacement's commit. Weak
    /// node/tree ends, same rationale as `CommittedRefOwner`: a dead
    /// entry means the tree already moved on — only the ledger-side
    /// backing still needs the delete.
    private struct DeferredSupersession {
        weak var node: RadixTreeNode?
        weak var tree: TokenRadixTree?
        let supersededRefID: String
    }
    /// Owner chain head → the nodes holding **Chain-Prefix Restore**
    /// points into its chain (ADR-0012). The eager-clear counterpart of
    /// `committedRefsByID` for the chain-side backing channel: when the
    /// owner dies (explicit delete, LRU cut, hydration failure) its
    /// dependent points are cleared through the same plumbing; when a
    /// later fold consumes the owner, they re-own transitively. Weak on
    /// both ends, same rationale as `CommittedRefOwner`.
    private var chainPrefixDependentsByOwnerID: [String: [CommittedRefOwner]] = [:]

    /// The shared **Storage Activity Gate** (PRD #150) handed to the SSD
    /// tier's writer, retained here so `PrefixCacheManager` can surface
    /// it to the prefill call sites that mark it busy.
    let activityGate: StorageActivityGate?

    // MARK: - Init

    /// Construct a tiered store. Passing `nil` (or an
    /// `SSDPrefixCacheConfig` whose `enabled == false`) produces a
    /// RAM-only store — every `admitSnapshot` call returns `nil`, and
    /// the writer callbacks never fire.
    /// `writerDrainPreludeForTesting` forwards to the SSD tier's writer
    /// gate — test-only, nil in production. It lets a manager-level test
    /// hold a write pending (e.g. to keep an extension base shielded
    /// across a second admission) deterministically, mirroring the hook
    /// `SSDSnapshotStore.init` already exposes.
    init(
        ssdConfig: SSDPrefixCacheConfig? = nil,
        activityGate: StorageActivityGate? = nil,
        writerDrainPreludeForTesting: (@Sendable () async -> Void)? = nil
    ) {
        self.ssdStore = nil
        self.activityGate = activityGate

        guard let ssdConfig, ssdConfig.enabled else { return }

        // The callbacks capture `self` weakly so the tiered store
        // can be deallocated while the SSD writer is idle. Both
        // callbacks hop to MainActor before touching any mutable
        // state — the writer task itself is not MainActor-isolated.
        self.ssdStore = SSDSnapshotStore(
            config: ssdConfig,
            activityGate: activityGate,
            onCommit: { [weak self] info in
                Task { @MainActor [weak self] in
                    self?.markSnapshotRefCommitted(info)
                }
            },
            onDrop: { [weak self] id, reason in
                Task { @MainActor [weak self] in
                    self?.markSnapshotRefDropped(id: id, reason: reason)
                }
            },
            writerDrainPreludeForTesting: writerDrainPreludeForTesting
        )
    }

    // MARK: - RAM tier (per-partition tree collection)

    /// Lookup-only; returns `nil` for partitions that have no tree.
    func tree(for key: CachePartitionKey) -> TokenRadixTree? {
        trees[key]
    }

    /// Lookup-or-allocate. Used on the write path before the first
    /// insertion into a freshly seen partition.
    func getOrCreateTree(for key: CachePartitionKey) -> TokenRadixTree {
        if let existing = trees[key] { return existing }
        let fresh = TokenRadixTree()
        trees[key] = fresh
        return fresh
    }

    /// Deterministic key-sorted iteration. The eviction drain pins a
    /// single ordered list across all rounds of `findEvictionCandidate`
    /// so tie-breaks stay stable; callers rely on the ordering being
    /// consistent across calls within one drain.
    func orderedPartitions() -> [(key: CachePartitionKey, tree: TokenRadixTree)] {
        trees
            .sorted { $0.key < $1.key }
            .map { (key: $0.key, tree: $0.value) }
    }

    /// Live partition count.
    var partitionCount: Int { trees.count }

    /// Sum of every partition's RAM-resident snapshot bytes. Invoked
    /// inside the eviction drain loop condition — must stay
    /// O(partitions) or cheaper.
    var totalSnapshotBytes: Int {
        trees.values.reduce(0) { $0 + $1.totalSnapshotBytes }
    }

    func ssdDiagnosticsSnapshot() -> PromptCacheSSDSnapshot {
        ssdStore?.diagnosticsSnapshot() ?? .disabled
    }

    /// True when an SSD tier is configured. Lets callers gate
    /// SSD-only setup work (warm start, partition registration) without
    /// reaching for the SSD store itself.
    var isSSDEnabled: Bool { ssdStore != nil }

    // MARK: - Lookup support

    /// On a RAM-resident lookup hit, bump the SSD LRU recency for the
    /// node's committed **Snapshot Ref** so a hot RAM hit does not look
    /// stale to the SSD LRU when the body is later dropped to `ssdOnly`.
    /// Returns the snapshot ID whose recency was bumped, or `nil` when
    /// the node holds no committed ref (any **Snapshot State** other
    /// than `committed`) or the SSD tier is disabled. Callers need not
    /// know the node's state shape or that an SSD LRU exists.
    func noteLookupHit(on node: RadixTreeNode) -> String? {
        guard case .committed(_, let ref) = node.state else { return nil }
        ssdStore?.recordHit(id: ref.snapshotID)
        return ref.snapshotID
    }

    /// Build the off-main hydration context for a body-absent
    /// `ssdOnly` node, or `nil` when neither the SSD tier nor a test
    /// hydrating override is configured. The returned `SSDHitContext`
    /// carries the `nonisolated` **Snapshot Hydrating** handle so
    /// **Snapshot Resolution** can `loadSync` off the MainActor — see
    /// the type's note. Production hands the concrete `SSDSnapshotStore`;
    /// a test injects an in-memory peer via
    /// `setHydratingOverrideForTesting` so the composition is assertable
    /// without a loaded model or a temp directory.
    func makeSSDHitContext(ref: SnapshotRef, node: RadixTreeNode) -> SSDHitContext? {
        guard let handle = hydratingOverride ?? ssdStore.map({ $0 as any SnapshotHydrating })
        else { return nil }
        return SSDHitContext(snapshotRef: ref, hydrating: handle, node: node)
    }

    /// Build the off-main hydration context for a **Chain-Prefix
    /// Restore** hit (ADR-0012), or `nil` when neither the SSD tier nor a
    /// test hydrating override is configured. Same shape and isolation
    /// contract as `makeSSDHitContext`.
    func makeChainPrefixHitContext(
        point: ChainPrefixRestorePoint, node: RadixTreeNode
    ) -> ChainPrefixHitContext? {
        guard let handle = hydratingOverride ?? ssdStore.map({ $0 as any SnapshotHydrating })
        else { return nil }
        return ChainPrefixHitContext(point: point, hydrating: handle, node: node)
    }

    // MARK: - SSD lifecycle

    /// Drain the SSD writer's pending queue and persist the manifest.
    /// Awaited by model-unload / restart paths so in-flight writes
    /// survive teardown. No-op when the SSD tier is disabled.
    func flush() async {
        await ssdStore?.flushAsync()
    }

    /// Load the SSD manifest and return the partitions that survive
    /// fingerprint validation against `expectedFingerprint`. Returns
    /// `.empty` when the SSD tier is disabled, so the caller needs no
    /// SSD-presence branch.
    func warmStartLoad(expectedFingerprint: String) -> WarmStartOutcome {
        ssdStore?.warmStartLoad(expectedFingerprint: expectedFingerprint) ?? .empty
    }

    // MARK: - SSD partition registration

    /// Register a `PartitionMeta` with the SSD tier so subsequent
    /// `admitSnapshot` calls whose descriptors carry the same
    /// `partitionDigest` satisfy the manifest invariant. No-op when
    /// the SSD tier is disabled. Must be called once per distinct
    /// `CachePartitionKey` before that partition's first admission;
    /// repeat calls are idempotent (the underlying store overwrites
    /// the stored `PartitionMeta`).
    func registerPartition(_ meta: PartitionMeta, for key: CachePartitionKey) {
        ssdStore?.registerPartition(meta, digest: key.partitionDigest)
    }

    // MARK: - Survival Gate

    /// The **Survival Gate** (PRD #82 slice #90): before an SSD write,
    /// would the incoming chain survive the eviction its own admission
    /// triggers? `true` when the SSD tier is disabled — there is no
    /// write to gate, and the caller's `admitSnapshot` will no-op
    /// anyway. `lastAccessAt` defaults to now (a fresh capture); a
    /// **Snapshot Demotion** passes the node's real stale stamp, which
    /// is what lets the gate catch cold demotions against warmer
    /// chains.
    func survivalGateAdmits(
        snapshot: HybridCacheSnapshot,
        payloadTotalBytes: Int,
        lastAccessAt: TimeInterval? = nil,
        scoringConfig: EvictionConfiguration
    ) -> Bool {
        guard let ssdStore else { return true }
        return ssdStore.survivesAdmissionCut(
            tokenOffset: snapshot.tokenOffset,
            totalBytes: payloadTotalBytes,
            checkpointType: snapshot.checkpointType,
            lastAccessAt: lastAccessAt ?? Date().timeIntervalSinceReferenceDate,
            scoring: scoringConfig
        )
    }

    // MARK: - SSD admission (state 1 → state 2)

    /// Build a descriptor from a captured snapshot's **domain inputs**,
    /// enqueue the payload into the SSD writer's pending queue, and route
    /// the resulting pending **Snapshot Ref** into `tree.admit`. The
    /// caller hands the snapshot, its path-from-root, the partition key,
    /// and the payload — never a hand-shaped `PersistedSnapshotDescriptor`.
    /// The on-disk schema is owned by the ledger, so descriptor
    /// construction goes through `SnapshotLedger.makeDescriptor`.
    ///
    /// Returns the pending `SnapshotRef` on admission, or `nil` when the
    /// SSD tier is disabled **or** the writer rejected the enqueue (budget
    /// / invalid checkpoint type / unregistered partition). The rejection
    /// *reasons* are intentionally not surfaced here — no caller branches
    /// on them, and they are asserted at the `SSDSnapshotStore.tryEnqueue`
    /// seam. A `nil` return means "no ref to track" for both the disabled
    /// and rejected cases.
    ///
    /// **Side effects on admission:**
    /// - The tree applies `admit` (state 1/2/4 → pending write), the sole
    ///   mutator of node state. Re-admission over a still-live ref returns
    ///   the superseded ID; its SSD backing is deleted *before* the new
    ///   pending entry is seeded, closing the SSD-side orphan that the old
    ///   raw node ref overwrite leaked.
    /// - `pendingRefsByID[ref.snapshotID]` is seeded with the
    ///   `(node, tree)` pair so the writer's commit / drop callback can
    ///   always find the node, even after RAM eviction drops the body.
    ///
    /// Non-suspending by construction: descriptor construction is pure and
    /// the underlying `SSDSnapshotStore.tryEnqueue` is `nonisolated` and
    /// acquires a plain `NSLock` for cross-thread safety, so calling this
    /// from a synchronous MainActor Snapshot Admission closure never
    /// forces an `await` on the caller.
    @discardableResult
    func admitSnapshot(
        node: RadixTreeNode,
        tree: TokenRadixTree,
        partitionKey: CachePartitionKey,
        pathFromRoot: [Int],
        snapshot: HybridCacheSnapshot,
        payload: SnapshotPayload,
        demotionLastAccessAt: TimeInterval? = nil,
        scoringConfig: EvictionConfiguration = EvictionConfiguration(),
        condemnedResidentIDs: Set<String> = [],
        mandatory: Bool = false,
        deferrable: Bool = false
    ) -> SnapshotRef? {
        guard let ssdStore else { return nil }

        // A non-nil `demotionLastAccessAt` marks a **Snapshot Demotion**:
        // the descriptor carries the node's real (stale) recency and the
        // writer's commit must not re-stamp it — demoted bodies are the
        // least valuable, and refreshing them would invert the SSD
        // tier's recency signal on every pressure event.
        let descriptor = SnapshotLedger.makeDescriptor(
            partitionKey: partitionKey,
            pathFromRoot: pathFromRoot,
            snapshot: snapshot,
            payloadBytes: payload.totalBytes,
            segmentBaseOffset: payload.extending?.baseOffset ?? 0,
            lastAccessAt: demotionLastAccessAt
        )

        guard
            case .accepted(let ref) = ssdStore.tryEnqueue(
                payload: payload,
                descriptor: descriptor,
                refreshRecencyAtCommit: demotionLastAccessAt == nil,
                scoringConfig: scoringConfig,
                condemnedResidentIDs: condemnedResidentIDs,
                mandatory: mandatory,
                deferrable: deferrable
            )
        else {
            return nil
        }

        let supersededID = tree.admit(node: node, ref: ref)
        if let supersededID {
            // The old write may have already committed to the manifest
            // with its file on disk; deleting it (SSD backing + stale
            // pending entry) before seeding the new entry prevents an
            // orphaned file + manifest entry.
            deleteSnapshot(snapshotID: supersededID)
        }
        pendingRefsByID[ref.snapshotID] = PendingRef(
            node: node,
            tree: tree
        )
        return ref
    }

    /// Remove a snapshot's SSD backing immediately. Used by leaf
    /// supersession when a newly stored descendant leaf makes an older
    /// ancestor leaf obsolete. Safe for pending and committed refs.
    /// Deleting a chain head also clears every restore point into its
    /// chain — the files are about to be unlinked. A deleted snapshot's
    /// own deferred supersessions are forgotten too: its replacement is
    /// gone, so the ancestors it condemned stay preserved.
    func deleteSnapshot(snapshotID: String) {
        pendingRefsByID.removeValue(forKey: snapshotID)
        committedRefsByID.removeValue(forKey: snapshotID)
        deferredSupersessionsByID.removeValue(forKey: snapshotID)
        clearChainPrefixDependents(ownerID: snapshotID)
        ssdStore?.deleteSnapshot(snapshotID: snapshotID)
    }

    /// Record that `supersededRefID`'s backing is replaced by the write
    /// `newSnapshotID` and must be deleted only when that write COMMITS
    /// (enqueue-before-delete, ADR-0019). Until then — and permanently,
    /// if the writer drops the new write — the superseded backing stays
    /// reachable: it is the warm-start fallback and the hit target for
    /// the whole pending window.
    func deferSupersededBackingDeletion(
        until newSnapshotID: String,
        node: RadixTreeNode,
        tree: TokenRadixTree,
        supersededRefID: String
    ) {
        deferredSupersessionsByID[newSnapshotID, default: []].append(
            DeferredSupersession(
                node: node, tree: tree, supersededRefID: supersededRefID
            ))
    }

    /// Execute (on commit) the deferred deletions recorded for a newly
    /// durable write: delete each superseded backing and discard its
    /// tree ref. A ref already gone (SSD-tier eviction consumed the
    /// condemned resident first, or an explicit delete raced) is a
    /// legitimate no-op — `deleteSnapshot` tolerates missing IDs and the
    /// node guard skips a ref the tree already replaced.
    private func executeDeferredSupersessions(for snapshotID: String) {
        guard
            let deferred = deferredSupersessionsByID.removeValue(forKey: snapshotID)
        else { return }
        for entry in deferred {
            deleteSnapshot(snapshotID: entry.supersededRefID)
            guard let node = entry.node,
                let tree = entry.tree,
                node.state.refID == entry.supersededRefID
            else { continue }
            tree.discardSnapshotRefAfterExplicitDelete(node: node)
        }
    }

    /// Whether `snapshotID` is a base currently shielded by a pending
    /// **Leaf Extension Admission**. The manager's supersession walk
    /// consults this before reclaiming an ancestor backing, so it never
    /// strands an in-flight fold that still depends on it. `false` when
    /// SSD is disabled — no extensions, so no shield.
    func isTransferringBase(_ snapshotID: String) -> Bool {
        ssdStore?.isTransferringBase(snapshotID) ?? false
    }

    // MARK: - Warm-start restore

    /// Reattach an SSD-resident committed ref to `node` (warm start).
    /// Forwards to `tree.restoreCommittedRef` — the sole mutator — and
    /// seeds `committedRefsByID`, so a later SSD-tier eviction of the
    /// restored resident clears the tree's ref eagerly, exactly like a
    /// ref that committed through the writer callback.
    func restoreCommittedRef(
        node: RadixTreeNode,
        tree: TokenRadixTree,
        ref: SnapshotRef
    ) {
        tree.restoreCommittedRef(node: node, ref: ref)
        committedRefsByID[ref.snapshotID] = CommittedRefOwner(node: node, tree: tree)
    }

    // MARK: - Writer callbacks (MainActor-isolated)

    /// Writer commit callback. Transitions:
    /// - State 2 (body present, pending ref) → state 4 (body present,
    ///   committed ref). Lookups continue to hit from RAM; the SSD
    ///   copy now exists as insurance.
    /// - State 3 (body absent, pending ref) → state 5 (body absent,
    ///   committed ref). Subsequent lookups can hydrate from SSD.
    ///
    /// The committed ref's `bytesOnDisk` is refreshed with the writer's
    /// `chainBytesOnDisk` — after an extension fold the entry owns its
    /// whole **Segment Chain**, so live telemetry must match what a
    /// warm start would restore.
    ///
    /// For a **Leaf Extension Admission** (`consumedBaseID != nil`),
    /// this is also where the ownership transfer lands on the tree:
    /// the fold consumed the base's manifest entry, so the base's tree
    /// ref — kept live through the pending window as the warm-start
    /// fallback — is discarded here. A writer *drop* never reaches
    /// this path, leaving the base reachable (the transfer degrades to
    /// `preserved`).
    ///
    /// Misses (id not in the pending map) are logged at debug and
    /// otherwise ignored — the node was already evicted via the
    /// drop-on-back-pressure path or the writer fired a duplicate
    /// commit for an ID that never actually landed.
    func markSnapshotRefCommitted(_ info: SSDCommitInfo) {
        guard let pending = pendingRefsByID.removeValue(forKey: info.snapshotID) else {
            Log.agent.debug(
                "TieredSnapshotStore.markSnapshotRefCommitted: id=\(info.snapshotID) "
                    + "not in pending map"
            )
            return
        }

        // The tree is the sole mutator and owns the stale-ID guard: an
        // `.ignored(reason)` here (a later admission superseded the ref,
        // or the node already committed) is logged and not recovered —
        // the newer ref wins.
        let effect = pending.tree.commitRef(
            node: pending.node,
            expectedID: info.snapshotID,
            bytesOnDisk: info.chainBytesOnDisk
        )
        if case .ignored(let reason) = effect {
            Log.agent.debug(
                "TieredSnapshotStore.markSnapshotRefCommitted: id=\(info.snapshotID) ignored "
                    + "(reason=\(String(describing: reason)))"
            )
        } else {
            // The ref is now committed on the node — index it so a later
            // SSD-tier eviction of this resident clears the tree's ref
            // eagerly (see `markSnapshotRefDropped`).
            committedRefsByID[info.snapshotID] = CommittedRefOwner(
                node: pending.node,
                tree: pending.tree
            )
        }

        // Even when the extension's own ref was superseded mid-window
        // (`.ignored` above), the fold is durable and the base entry is
        // gone — convert the base's tree ref unconditionally so no
        // stale committed ref lingers until a failed hydration.
        if let baseID = info.consumedBaseID {
            convertConsumedBaseRef(
                baseID: baseID,
                newOwnerID: info.snapshotID,
                below: pending.node,
                tree: pending.tree
            )
        }

        // The write is durable — NOW the ancestor backings it replaced
        // may be deleted (enqueue-before-delete, ADR-0019). Runs after
        // the commit transition so the node's new committed ref is
        // already the warm-start authority the deletions leave behind.
        executeDeferredSupersessions(for: info.snapshotID)
    }

    /// The fold's tree-side ownership conversion (ADR-0012). Find the
    /// ancestor node still carrying the consumed base's ref and convert
    /// it into a **Chain-Prefix Restore** point owned by the new chain
    /// head — the base's bytes live on as the head's leading segments,
    /// so the boundary stays restorable instead of going dark. Points
    /// previously owned by the base re-own transitively: the head
    /// inherited the base's whole chain, so every boundary it covered
    /// stays covered.
    ///
    /// The base is a strict ancestor of the extension's node by
    /// construction (same token path, shorter offset); ref-bearing nodes
    /// keep their identity across path compression, so the parent walk
    /// finds it. Not finding it is legitimate — an explicit delete or
    /// hydration failure already cleared the node mid-window — and a
    /// no-op.
    private func convertConsumedBaseRef(
        baseID: String,
        newOwnerID: String,
        below node: RadixTreeNode,
        tree: TokenRadixTree
    ) {
        // The fold consumed the base's manifest entry without a writer
        // drop callback, so its committed-index entry is pruned here.
        committedRefsByID.removeValue(forKey: baseID)

        // Transitive re-point: boundaries that resolved through the
        // base's chain now resolve through the new head's.
        if let dependents = chainPrefixDependentsByOwnerID.removeValue(forKey: baseID) {
            var reowned: [CommittedRefOwner] = []
            for dependent in dependents {
                guard let dependentNode = dependent.node,
                    let dependentTree = dependent.tree,
                    dependentNode.chainPrefixRestorePoint?.ownerSnapshotID == baseID
                else { continue }
                dependentTree.reownChainPrefixRestorePoint(
                    node: dependentNode, to: newOwnerID
                )
                reowned.append(dependent)
            }
            if !reowned.isEmpty {
                chainPrefixDependentsByOwnerID[newOwnerID, default: []]
                    .append(contentsOf: reowned)
            }
        }

        var current = node.parent
        while let ancestor = current {
            if ancestor.state.refID == baseID {
                // If this node already carried a point (a fresh leaf re-occupied
                // the boundary after an earlier conversion, then was itself
                // consumed), the overwrite below re-owns it — orphaning its
                // entry in the *prior* owner's dependents bucket. Drop that
                // entry first, so the prior owner's death doesn't trip over a
                // stale no-op entry that no longer matches its node's owner.
                if let priorOwner = ancestor.chainPrefixRestorePoint?.ownerSnapshotID,
                    priorOwner != newOwnerID
                {
                    chainPrefixDependentsByOwnerID[priorOwner]?.removeAll {
                        $0.node === ancestor
                    }
                }
                tree.convertConsumedBaseToChainPrefixRestorePoint(
                    node: ancestor, ownerSnapshotID: newOwnerID
                )
                chainPrefixDependentsByOwnerID[newOwnerID, default: []]
                    .append(CommittedRefOwner(node: ancestor, tree: tree))
                trimChainPrefixRestorePoints(ownerID: newOwnerID)
                return
            }
            current = ancestor.parent
        }
        Log.agent.debug(
            "TieredSnapshotStore.convertConsumedBaseRef: base=\(baseID) "
                + "not on the ancestor path (already cleared)"
        )
    }

    /// Upper bound on **Chain-Prefix Restore** points retained per chain.
    /// Each fold mints one new point and re-owns every prior spine point
    /// forward to the new head — which never dies — so without a cap a
    /// multi-hundred-turn session pins one radix node per turn forever,
    /// inflating `nodeCount` and every tree walk (issue #101 follow-up).
    /// Generous enough that realistic sessions never reach it; a backstop
    /// against unbounded growth, not a tuning knob.
    private static let maxRestorePointsPerChain = 256

    /// Keep the deepest `maxRestorePointsPerChain` restore points for a chain
    /// and clear the shallowest. The shallowest points are the oldest, least
    /// valuable rewind floors — dropping one only costs a deeper re-prefill on
    /// the rare interrupt that rewinds that far back, never correctness. Also
    /// sweeps entries whose node was already reclaimed (dead weak ref).
    private func trimChainPrefixRestorePoints(ownerID: String) {
        guard var dependents = chainPrefixDependentsByOwnerID[ownerID],
            dependents.count > Self.maxRestorePointsPerChain
        else { return }
        // Deepest boundary first; a dead/cleared node sorts last (offset -1).
        dependents.sort {
            ($0.node?.chainPrefixRestorePoint?.boundaryOffset ?? -1)
                > ($1.node?.chainPrefixRestorePoint?.boundaryOffset ?? -1)
        }
        for owner in dependents[Self.maxRestorePointsPerChain...] {
            if let node = owner.node, let tree = owner.tree {
                tree.clearChainPrefixRestorePoint(node: node)
            }
        }
        chainPrefixDependentsByOwnerID[ownerID] =
            Array(dependents.prefix(Self.maxRestorePointsPerChain))
    }

    /// Batch form for one restored chain head. Splits every inherited
    /// boundary in one tree walk, then attaches points only to empty,
    /// point-less nodes so direct snapshot refs still win.
    func restoreChainPrefixRestorePoints(
        points: [ChainPrefixRestorePoint],
        ownerPath: [Int],
        partitionKey: CachePartitionKey
    ) {
        let tree = getOrCreateTree(for: partitionKey)
        let nodesByOffset = tree.ensurePrefixNodes(
            tokens: ownerPath,
            offsets: points.map(\.boundaryOffset)
        )
        for point in points {
            guard let node = nodesByOffset[point.boundaryOffset],
                node.chainPrefixRestorePoint == nil,
                case .empty = node.state
            else {
                Log.agent.debug(
                    "TieredSnapshotStore.restoreChainPrefixRestorePoints: node at "
                        + "boundary=\(point.boundaryOffset) already backed — keeping direct owner"
                )
                continue
            }
            tree.attachChainPrefixRestorePoint(node: node, point: point)
            chainPrefixDependentsByOwnerID[point.ownerSnapshotID, default: []]
                .append(CommittedRefOwner(node: node, tree: tree))
        }
    }

    /// Eagerly clear every **Chain-Prefix Restore** point into a dead
    /// chain — the dependents' counterpart of `clearCommittedResidentRef`.
    /// Dead weak owners and points that already re-owned elsewhere are
    /// legitimate no-ops.
    private func clearChainPrefixDependents(ownerID: String) {
        guard let dependents = chainPrefixDependentsByOwnerID.removeValue(forKey: ownerID)
        else { return }
        for dependent in dependents {
            guard let node = dependent.node,
                let tree = dependent.tree,
                node.chainPrefixRestorePoint?.ownerSnapshotID == ownerID
            else { continue }
            tree.clearChainPrefixRestorePoint(node: node)
        }
    }

    /// Clear one restore point and prune its owner index entry. Used by
    /// point-local hydration failures where the owning chain may still be
    /// valid and should not have all dependents cleared.
    func clearChainPrefixRestorePoint(node: RadixTreeNode, tree: TokenRadixTree) {
        guard let ownerID = node.chainPrefixRestorePoint?.ownerSnapshotID else { return }
        if var dependents = chainPrefixDependentsByOwnerID[ownerID] {
            dependents.removeAll { dependent in
                guard let dependentNode = dependent.node else { return true }
                return dependentNode === node
            }
            if dependents.isEmpty {
                chainPrefixDependentsByOwnerID.removeValue(forKey: ownerID)
            } else {
                chainPrefixDependentsByOwnerID[ownerID] = dependents
            }
        }
        tree.clearChainPrefixRestorePoint(node: node)
    }

    /// Writer drop callback. Three legitimate call paths reach here:
    /// 1. Front-door back-pressure dropped the oldest pending entry
    ///    under the `maxPendingBytes` cap (`.backpressureOldest`).
    /// 2. Admission-time LRU cut dropped the incoming non-system
    ///    write under type protection (`.systemProtectionWins`)
    ///    or exceeded the hard budget (`.exceedsBudget`).
    /// 3. Writer hit a disk-full / IO error that could not be
    ///    recovered after a single eviction-retry pass
    ///    (`.diskFull`, `.writerIOError`).
    ///
    /// Routes through `tree.dropRef`, which decides the transition:
    /// - State 2 (body present, pending ref) → state 1 (body present,
    ///   no ref). The RAM body stays live; the SSD copy never materialized.
    /// - State 3 (body absent, pending ref) → removed-from-tree. The node
    ///   has nothing left, so the tree self-heals (leaf → evict,
    ///   single-child → collapse, multi-child empty → retained junction).
    ///
    /// Calls with an `id` that is not in the pending map are routed to
    /// the committed index — the evicted-by-LRU reason from the SSD-tier
    /// cut (and a `loadSync` hydration failure) lands against a
    /// previously committed resident, whose tree ref must be cleared
    /// *eagerly*: eviction scoring, **Snapshot Demotion**, and the
    /// recovered/terminal telemetry all read `node.state.ref != nil` as
    /// "backed", so a stale committed ref misprices the node, skips its
    /// demotion, and miscounts a terminal loss as recovered.
    func markSnapshotRefDropped(id: String, reason: SSDDropReason) {
        // A dropped write never deletes what it superseded: forget its
        // deferred supersessions so the ancestors degrade to `preserved`
        // — the warm-start fallback and the next turn's extension base
        // stay alive (enqueue-before-delete, ADR-0019).
        deferredSupersessionsByID.removeValue(forKey: id)
        guard let pending = pendingRefsByID.removeValue(forKey: id) else {
            clearCommittedResidentRef(id: id, reason: reason)
            return
        }

        // The tree is the sole mutator and owns the stale-ID guard. State
        // 2 (body present) settles to RAM-only and stays; state 3 (body
        // absent) empties and the tree self-heals (removes the node). An
        // `.ignored(.idMismatch)` means a later admission superseded the
        // ref — newer ref wins, logged and not recovered.
        let effect = pending.tree.dropRef(node: pending.node, expectedID: id)
        if case .ignored(let dropReason) = effect {
            Log.agent.debug(
                "TieredSnapshotStore.markSnapshotRefDropped: id=\(id) ignored "
                    + "(reason=\(String(describing: dropReason)))"
            )
        }
    }

    /// Eagerly clear a committed resident's tree ref after the SSD tier
    /// dropped its backing. A missing index entry, a dead weak owner, or
    /// a node whose current ref ID no longer matches (a re-admission
    /// superseded the ref mid-hop) are all legitimate no-ops: the ref
    /// this drop names is already off the node. State 4 (body present)
    /// settles to RAM-only; state 5 empties and the tree self-heals.
    /// A dead chain head's dependent restore points clear with it —
    /// the same eager backing-loss semantics, on the chain-side channel.
    private func clearCommittedResidentRef(id: String, reason: SSDDropReason) {
        clearChainPrefixDependents(ownerID: id)
        guard let owner = committedRefsByID.removeValue(forKey: id),
            let node = owner.node,
            let tree = owner.tree,
            node.state.refID == id
        else {
            Log.agent.debug(
                "TieredSnapshotStore.markSnapshotRefDropped: id=\(id) not tracked "
                    + "(reason=\(String(describing: reason)))"
            )
            return
        }
        tree.clearCommittedSnapshotRefAfterBackingLoss(node: node)
    }

    // MARK: - Testing hooks

    /// Snapshot of the pending map size. Tests use this to verify
    /// commit / drop callbacks have landed and the lifecycle state
    /// machine is in the expected state without reaching into the
    /// private storage directly.
    var pendingRefCountForTesting: Int { pendingRefsByID.count }

    /// True when the given snapshot ID is currently tracked in the
    /// pending map (i.e. the node is in state 2 or state 3).
    func isPendingForTesting(id: String) -> Bool {
        pendingRefsByID[id] != nil
    }

    /// Test-only white-box door onto the SSD tier. Production code
    /// drives the SSD through this store's sealed interface and must
    /// never reach the `SSDSnapshotStore` directly — tests use this to
    /// assert resident-set / byte-accounting internals.
    var ssdStoreForTesting: SSDSnapshotStore? { ssdStore }

    /// Test-only hydrating override. When set, `makeSSDHitContext` and
    /// `makeChainPrefixHitContext` hand this peer — not the concrete
    /// `SSDSnapshotStore` — into the hit context, so **Snapshot Resolution**'s
    /// SSD-hydration composition is assertable through
    /// `PrefixCacheManager.resolve` with programmed outcomes and no disk.
    /// Production leaves this `nil`, so the concrete store remains the sole
    /// production adapter.
    private var hydratingOverride: (any SnapshotHydrating)?

    func setHydratingOverrideForTesting(_ handle: (any SnapshotHydrating)?) {
        hydratingOverride = handle
    }
}

// MARK: - SSDHitContext

/// Handle that lets **Snapshot Resolution** hydrate a body-absent `ssdOnly`
/// node from the SSD tier off the MainActor. Carries the `MainActor`-owned
/// `RadixTreeNode` and the `nonisolated` **Snapshot Hydrating** handle
/// resolution needs to materialize a state-5 node. `@unchecked Sendable`
/// because the node is MainActor-owned — resolution only reads the
/// **Snapshot Ref** off-main and hops back to MainActor before touching the
/// node via `PrefixCacheManager.promote(node:snapshot:partitionKey:)`.
///
/// It carries the narrow `SnapshotHydrating` handle (`loadSync`,
/// `loadSyncPrefix`, `recordHit`) rather than the concrete `SSDSnapshotStore`
/// so the read stays off the MainActor and a second adapter (an in-memory
/// test peer) could satisfy it — ADR-0001, now sealed.
struct SSDHitContext: @unchecked Sendable {
    let snapshotRef: SnapshotRef
    let hydrating: any SnapshotHydrating
    let node: RadixTreeNode
}

// MARK: - ChainPrefixHitContext

/// Handle that lets **Snapshot Resolution** hydrate a **Chain-Prefix Restore**
/// hit (ADR-0012) off the MainActor — the chain-side sibling of
/// `SSDHitContext`, with the identical isolation contract: the node is
/// MainActor-owned, only the value-type `point` is read off-main, and the
/// caller hops back to MainActor to promote or clear.
struct ChainPrefixHitContext: @unchecked Sendable {
    let point: ChainPrefixRestorePoint
    let hydrating: any SnapshotHydrating
    let node: RadixTreeNode
}
