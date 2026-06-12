//
//  TieredSnapshotStore.swift
//  tesseract
//
//  Composes the RAM tier (`InMemorySnapshotTier`) with `SSDSnapshotStore`
//  behind a single `SnapshotStore` conformance, and acts as the SSD
//  **router**: it resolves the writer's global `snapshotID` callbacks to
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

/// Two-tier snapshot store. The RAM tier is an inline
/// `InMemorySnapshotTier` — identical behavior to the RAM-only
/// `PrefixCacheManager` configuration. The SSD tier is an optional
/// `SSDSnapshotStore` that takes ownership of payload write-through,
/// the admission-time LRU cut, and the debounced manifest persist.
/// When `ssdConfig == nil` (or `ssdConfig.enabled == false`) the
/// store collapses to pure RAM-only behavior: the `SnapshotStore`
/// protocol conformance forwards to the RAM tier, and
/// `admitSnapshot` is a no-op that returns `nil`.
@MainActor
final class TieredSnapshotStore: SnapshotStore {

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

    private let ramTier: InMemorySnapshotTier
    /// The SSD tier. Fully private: callers drive SSD behaviour through
    /// this store's own interface (`noteLookupHit`, `flush`,
    /// `warmStartLoad`, `makeSSDHitContext`, `isSSDEnabled`) and never
    /// name `SSDSnapshotStore` themselves. Assigned once in `init` and
    /// never reassigned thereafter.
    private var ssdStore: SSDSnapshotStore?
    private var pendingRefsByID: [String: PendingRef] = [:]
    private var committedRefsByID: [String: CommittedRefOwner] = [:]

    // MARK: - Init

    /// Construct a tiered store. Passing `nil` (or an
    /// `SSDPrefixCacheConfig` whose `enabled == false`) produces a
    /// RAM-only store that behaves identically to a bare
    /// `InMemorySnapshotTier` — every `admitSnapshot` call returns
    /// `nil`, and the writer callbacks never fire.
    init(ssdConfig: SSDPrefixCacheConfig? = nil) {
        self.ramTier = InMemorySnapshotTier()
        self.ssdStore = nil

        guard let ssdConfig, ssdConfig.enabled else { return }

        // The callbacks capture `self` weakly so the tiered store
        // can be deallocated while the SSD writer is idle. Both
        // callbacks hop to MainActor before touching any mutable
        // state — the writer task itself is not MainActor-isolated.
        self.ssdStore = SSDSnapshotStore(
            config: ssdConfig,
            onCommit: { [weak self] info in
                Task { @MainActor [weak self] in
                    self?.markSnapshotRefCommitted(info)
                }
            },
            onDrop: { [weak self] id, reason in
                Task { @MainActor [weak self] in
                    self?.markSnapshotRefDropped(id: id, reason: reason)
                }
            }
        )
    }

    // MARK: - SnapshotStore protocol (forwards to RAM tier)

    func tree(for key: CachePartitionKey) -> TokenRadixTree? {
        ramTier.tree(for: key)
    }

    func getOrCreateTree(for key: CachePartitionKey) -> TokenRadixTree {
        ramTier.getOrCreateTree(for: key)
    }

    func orderedPartitions() -> [(key: CachePartitionKey, tree: TokenRadixTree)] {
        ramTier.orderedPartitions()
    }

    var partitionCount: Int { ramTier.partitionCount }

    var totalSnapshotBytes: Int { ramTier.totalSnapshotBytes }

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
    /// `ssdOnly` node, or `nil` when the SSD tier is disabled. The
    /// returned `SSDHitContext` carries the `nonisolated`
    /// `SSDSnapshotStore` so `LLMActor` can `loadSync` off the
    /// MainActor — see the type's note. Constructing it here is what
    /// keeps the concrete SSD store private to this seam.
    func makeSSDHitContext(ref: SnapshotRef, node: RadixTreeNode) -> SSDHitContext? {
        guard let ssdStore else { return nil }
        return SSDHitContext(snapshotRef: ref, ssdStore: ssdStore, node: node)
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
        scoringConfig: EvictionConfiguration = EvictionConfiguration()
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

        guard case .accepted(let ref) = ssdStore.tryEnqueue(
            payload: payload,
            descriptor: descriptor,
            refreshRecencyAtCommit: demotionLastAccessAt == nil,
            scoringConfig: scoringConfig
        ) else {
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
    func deleteSnapshot(snapshotID: String) {
        pendingRefsByID.removeValue(forKey: snapshotID)
        committedRefsByID.removeValue(forKey: snapshotID)
        ssdStore?.deleteSnapshot(snapshotID: snapshotID)
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
        // gone — discard the base's tree ref unconditionally so no
        // stale committed ref lingers until a failed hydration.
        if let baseID = info.consumedBaseID {
            discardConsumedBaseRef(
                baseID: baseID,
                below: pending.node,
                tree: pending.tree
            )
        }
    }

    /// Find the ancestor node still carrying the consumed base's ref
    /// and discard it through the tree's explicit-delete seam. The base
    /// is a strict ancestor of the extension's node by construction
    /// (same token path, shorter offset); ref-bearing nodes keep their
    /// identity across path compression, so the parent walk finds it.
    /// Not finding it is legitimate — an explicit delete or hydration
    /// failure already cleared the node mid-window — and a no-op.
    private func discardConsumedBaseRef(
        baseID: String,
        below node: RadixTreeNode,
        tree: TokenRadixTree
    ) {
        // The fold consumed the base's manifest entry without a writer
        // drop callback, so its committed-index entry is pruned here.
        committedRefsByID.removeValue(forKey: baseID)
        var current = node.parent
        while let ancestor = current {
            if ancestor.state.refID == baseID {
                tree.discardSnapshotRefAfterExplicitDelete(node: ancestor)
                return
            }
            current = ancestor.parent
        }
        Log.agent.debug(
            "TieredSnapshotStore.discardConsumedBaseRef: base=\(baseID) "
            + "not on the ancestor path (already cleared)"
        )
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
    private func clearCommittedResidentRef(id: String, reason: SSDDropReason) {
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
}

// MARK: - SSDHitContext

/// Handle that lets `LLMActor` hydrate a body-absent `ssdOnly` node
/// from the SSD tier off the MainActor. Carries the `MainActor`-owned
/// `RadixTreeNode` and the `nonisolated` `SSDSnapshotStore` LLMActor
/// needs to hydrate a state-5 node. `@unchecked Sendable` because the
/// node is MainActor-owned — LLMActor only reads the **Snapshot Ref**
/// off-main and hops back to MainActor before touching the node via
/// `PrefixCacheManager.promote(node:snapshot:partitionKey:)`.
///
/// It deliberately carries the concrete `SSDSnapshotStore` rather than
/// a narrow hydration interface so the read stays off the MainActor;
/// see `docs/adr/0001-ssd-hydration-handle-stays-off-main.md`.
struct SSDHitContext: @unchecked Sendable {
    let snapshotRef: SnapshotRef
    let ssdStore: SSDSnapshotStore
    let node: RadixTreeNode
}
