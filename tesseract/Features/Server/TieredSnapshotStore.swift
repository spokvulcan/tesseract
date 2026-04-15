//
//  TieredSnapshotStore.swift
//  tesseract
//
//  Task 4.1.6 — compose the existing RAM tier (`InMemorySnapshotTier`)
//  with the new `SSDSnapshotStore` behind a single `SnapshotStore`
//  conformance. Owns the five-state storage-ref lifecycle that
//  `PrefixCacheManager` will eventually drive in Task 4.1.7:
//  pending refs are tracked in a `[snapshotID: PendingRef]` map so
//  the writer's commit / drop callbacks can find their node even
//  after a mid-flight RAM eviction has body-dropped the snapshot.
//
//  **No selective write-through.** Every admission enqueues to SSD
//  unconditionally; the writer runs the admission-time type-protected
//  LRU cut inside `SSDSnapshotStore` (Task 4.1.2 already implements
//  it) when the budget is tight. This store merely forwards
//  payloads + descriptors to the writer and attaches the returned
//  `SnapshotStorageRef` to the matching radix node.
//
//  **MainActor-isolated by construction.** All lifecycle mutations
//  (pending map writes, ref `committed` flips, hard-delete cleanups)
//  happen on MainActor. The writer's commit / drop callbacks fire
//  from off-main context and hop back via
//  `Task { @MainActor in self?.markStorageRefCommitted(...) }` —
//  see `init(ssdConfig:)`.
//
//  **Hard-delete on state-3 drop.** When the drop callback finds a
//  node in state 3 (body absent + ref pending), clearing the ref
//  leaves an orphaned structural path node. This store then replays
//  the normal cleanup that `PrefixCacheManager.evictToFitBudget`
//  runs on eviction victims (`evictNode` for leaves, `collapseSingleChildNode`
//  for single-child pass-through nodes), but with the SSD ref
//  already cleared so Task 4.1.8's body-drop guards do not fire.
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
    /// pending queue and attach the resulting pending
    /// `SnapshotStorageRef` to `node`. Returns the raw writer result
    /// so the caller can branch on rejection outcomes
    /// (`rejectedTooLargeForBudget`, `rejectedInvalidCheckpointType`,
    /// `rejectedUnregisteredPartition`). Returns `nil` when the SSD
    /// tier is disabled — callers must treat that as "RAM-only
    /// admission, no ref to track".
    ///
    /// **Side effects on `.accepted`:**
    /// - `node.storageRef` is set to the returned pending ref
    ///   (`committed: false`).
    /// - `pendingRefsByID[ref.snapshotID]` is seeded with the
    ///   `(node, tree)` pair so the writer's commit / drop callback
    ///   can always find the node, even after RAM eviction drops
    ///   the body and `evictNode` would normally prune a leaf.
    ///
    /// Non-suspending by construction: the underlying
    /// `SSDSnapshotStore.tryEnqueue` is `nonisolated` and acquires
    /// a plain `NSLock` for cross-thread safety, so calling this
    /// from a synchronous MainActor closure (the
    /// `PrefixCacheManager.storeSnapshots` / `storeLeaf` call sites
    /// in Task 4.1.7) never forces an `await` on the caller.
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
            node.storageRef = ref
            pendingRefsByID[ref.snapshotID] = PendingRef(
                node: node,
                tree: tree
            )
        }

        return result
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

        let node = pending.node

        // Defensive: the node's ref should still carry the same ID.
        // A mismatch means a later admission replaced the ref under
        // the same node — we surface the inconsistency via debug
        // logging but do not attempt recovery. The newer ref wins.
        guard var ref = node.storageRef, ref.snapshotID == id else {
            Log.agent.debug(
                "TieredSnapshotStore.markStorageRefCommitted: id=\(id) node.storageRef mismatch"
            )
            return
        }

        ref.committed = true
        node.storageRef = ref
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
    /// Transitions:
    /// - State 2 (body present, pending ref) → state 1 (body present,
    ///   no ref). The RAM body stays live; subsequent RAM hits
    ///   continue to work; the SSD copy never materialized.
    /// - State 3 (body absent, pending ref) → removed-from-tree. The
    ///   node has nothing: no RAM body, no committed SSD copy. We
    ///   replay the normal eviction cleanup (`evictNode` for leaves,
    ///   `collapseSingleChildNode` for single-child pass-throughs)
    ///   with the ref already cleared so the Task 4.1.8 guards
    ///   don't fire.
    ///
    /// Calls with an `id` that is not in the pending map are
    /// legitimate and are ignored at debug level — the evicted-by-LRU
    /// reason from the SSD-tier cut can land against a previously
    /// committed resident whose node is no longer tracked here. The
    /// stale `storageRef` on such a node self-cleans on the next
    /// lookup when `loadSync` reports file-missing and fires its own
    /// `markStorageRefDropped` callback (Task 4.1.9).
    func markStorageRefDropped(id: String, reason: SSDDropReason) {
        guard let pending = pendingRefsByID.removeValue(forKey: id) else {
            Log.agent.debug(
                "TieredSnapshotStore.markStorageRefDropped: id=\(id) not in pending map "
                + "(reason=\(String(describing: reason)))"
            )
            return
        }

        let node = pending.node
        let tree = pending.tree

        // Only clear the ref if it still matches — a later
        // admission could have replaced it and the newer ref wins.
        if let ref = node.storageRef, ref.snapshotID == id {
            node.storageRef = nil
        }

        // State 2 (body present) stays in the tree as RAM-only.
        guard node.snapshot == nil else { return }

        // State 3 → removed-from-tree. Replay the normal radix
        // cleanup `PrefixCacheManager.evictToFitBudget` runs on
        // ref-less, body-less victims. Task 4.1.8 adds a storageRef
        // guard to the eviction loop; that guard is a no-op here
        // because the ref was already cleared above.
        if node.isLeaf {
            tree.evictNode(node: node)
        } else if node.childCount == 1 {
            tree.collapseSingleChildNode(node)
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
