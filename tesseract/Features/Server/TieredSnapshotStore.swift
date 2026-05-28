//
//  TieredSnapshotStore.swift
//  tesseract
//
//  Composes the RAM tier (`InMemorySnapshotTier`) with `SSDSnapshotStore`
//  behind a single `SnapshotStore` conformance, and acts as the SSD
//  **router**: it resolves the writer's global `snapshotID` callbacks to
//  the owning `(node, tree)` and forwards them to the tree's transition
//  methods. It is **not** a mutator of node state — that is the tree's
//  job alone. The cross-partition `pendingRefsByID` map lives here
//  because the writer's callback carries only a global ID and the tree is
//  per-partition, so the index cannot live in any single tree.
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
//  `Task { @MainActor in self?.markStorageRefCommitted(...) }` —
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

    // MARK: - Stored state

    private let ramTier: InMemorySnapshotTier
    private(set) var ssdStore: SSDSnapshotStore?
    private var pendingRefsByID: [String: PendingRef] = [:]

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
            onCommit: { [weak self] id in
                Task { @MainActor [weak self] in
                    self?.markStorageRefCommitted(id: id)
                }
            },
            onDrop: { [weak self] id, reason in
                Task { @MainActor [weak self] in
                    self?.markStorageRefDropped(id: id, reason: reason)
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

    // MARK: - SSD admission (state 1 → state 2)

    /// Enqueue a freshly captured payload into the SSD writer's
    /// pending queue and route the resulting pending `SnapshotRef`
    /// into `tree.admit`. Returns the raw writer result
    /// so the caller can branch on rejection outcomes
    /// (`rejectedTooLargeForBudget`, `rejectedInvalidCheckpointType`,
    /// `rejectedUnregisteredPartition`). Returns `nil` when the SSD
    /// tier is disabled — callers must treat that as "RAM-only
    /// admission, no ref to track".
    ///
    /// **Side effects on `.accepted`:**
    /// - The tree applies `admit` (state 1/2/4 → pending write), the sole
    ///   mutator of node state. Re-admission over a still-live ref returns
    ///   the superseded ID; its SSD backing is deleted *before* the new
    ///   pending entry is seeded, closing the SSD-side orphan that the old
    ///   raw `node.storageRef = ref` overwrite leaked.
    /// - `pendingRefsByID[ref.snapshotID]` is seeded with the
    ///   `(node, tree)` pair so the writer's commit / drop callback can
    ///   always find the node, even after RAM eviction drops the body.
    ///
    /// Non-suspending by construction: the underlying
    /// `SSDSnapshotStore.tryEnqueue` is `nonisolated` and acquires
    /// a plain `NSLock` for cross-thread safety, so calling this
    /// from a synchronous MainActor closure (the
    /// `PrefixCacheManager.storeSnapshots` / `storeLeaf` call sites)
    /// never forces an `await` on the caller.
    @discardableResult
    func admitSnapshot(
        node: RadixTreeNode,
        tree: TokenRadixTree,
        payload: SnapshotPayload,
        descriptor: PersistedSnapshotDescriptor
    ) -> TryEnqueueResult? {
        guard let ssdStore else { return nil }

        let result = ssdStore.tryEnqueue(
            payload: payload,
            descriptor: descriptor
        )

        if case .accepted(let ref) = result {
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
        }

        return result
    }

    /// Remove a snapshot's SSD backing immediately. Used by leaf
    /// supersession when a newly stored descendant leaf makes an older
    /// ancestor leaf obsolete. Safe for pending and committed refs.
    func deleteSnapshot(snapshotID: String) {
        pendingRefsByID.removeValue(forKey: snapshotID)
        ssdStore?.deleteSnapshot(snapshotID: snapshotID)
    }

    // MARK: - Writer callbacks (MainActor-isolated)

    /// Writer commit callback. Transitions:
    /// - State 2 (body present, pending ref) → state 4 (body present,
    ///   committed ref). Lookups continue to hit from RAM; the SSD
    ///   copy now exists as insurance.
    /// - State 3 (body absent, pending ref) → state 5 (body absent,
    ///   committed ref). Subsequent lookups can hydrate from SSD.
    ///
    /// Misses (id not in the pending map) are logged at debug and
    /// otherwise ignored — the node was already evicted via the
    /// drop-on-back-pressure path or the writer fired a duplicate
    /// commit for an ID that never actually landed.
    func markStorageRefCommitted(id: String) {
        guard let pending = pendingRefsByID.removeValue(forKey: id) else {
            Log.agent.debug(
                "TieredSnapshotStore.markStorageRefCommitted: id=\(id) not in pending map"
            )
            return
        }

        // The tree is the sole mutator and owns the stale-ID guard: an
        // `.ignored(reason)` here (a later admission superseded the ref,
        // or the node already committed) is logged and not recovered —
        // the newer ref wins.
        let effect = pending.tree.commitRef(node: pending.node, expectedID: id)
        if case .ignored(let reason) = effect {
            Log.agent.debug(
                "TieredSnapshotStore.markStorageRefCommitted: id=\(id) ignored "
                + "(reason=\(String(describing: reason)))"
            )
        }
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
    /// Calls with an `id` that is not in the pending map are
    /// legitimate and are ignored at debug level — the evicted-by-LRU
    /// reason from the SSD-tier cut can land against a previously
    /// committed resident whose node is no longer tracked here. The
    /// stale ref on such a node self-cleans on the next lookup when
    /// `loadSync` reports file-missing and the MainActor calls
    /// `tree.clearStorageRef`.
    func markStorageRefDropped(id: String, reason: SSDDropReason) {
        guard let pending = pendingRefsByID.removeValue(forKey: id) else {
            // Not in the pending map: this is a committed-resident drop
            // (`.evictedByLRU` / `.hydrationFailure` for an ID that commit
            // already removed). A logged no-op — the stale committed ref
            // is cleared lazily on the next lookup, when `loadSync`
            // reports file-missing and the MainActor calls
            // `tree.clearStorageRef` with the node already in hand. Do not
            // search the tree per drop callback.
            Log.agent.debug(
                "TieredSnapshotStore.markStorageRefDropped: id=\(id) not in pending map "
                + "(reason=\(String(describing: reason)))"
            )
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
                "TieredSnapshotStore.markStorageRefDropped: id=\(id) ignored "
                + "(reason=\(String(describing: dropReason)))"
            )
        }
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
}
