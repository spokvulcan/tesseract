import Foundation
import MLXLMCommon

/// Partition key for isolating radix trees by runtime configuration.
///
/// Tool/template digests are intentionally NOT part of the partition key:
/// different tools/context → different tokens → different radix paths → naturally isolated.
///
/// `Comparable` so partition iteration can produce a deterministic order
/// (modelID → kvBits → kvGroupSize) for stable tie-break behavior in
/// eviction.
struct CachePartitionKey: Hashable, Sendable, Comparable {
    let modelID: String
    let kvBits: Int?
    let kvGroupSize: Int

    static func < (lhs: CachePartitionKey, rhs: CachePartitionKey) -> Bool {
        (lhs.modelID, lhs.kvBits ?? -1, lhs.kvGroupSize)
            < (rhs.modelID, rhs.kvBits ?? -1, rhs.kvGroupSize)
    }
}

@MainActor
final class PrefixCacheManager {
    private var trees: [CachePartitionKey: TokenRadixTree] = [:]
    var memoryBudgetBytes: Int

    init(memoryBudgetBytes: Int) {
        self.memoryBudgetBytes = memoryBudgetBytes
    }

    // MARK: - Lookup

    struct LookupResult {
        let snapshot: HybridCacheSnapshot?
        let partitionKey: CachePartitionKey?
        let snapshotTokenOffset: Int
        let sharedPrefixLength: Int
        let reason: LookupReason

        /// Restore the cached KV/Mamba state. Each call produces an independent deep copy.
        /// Nonisolated because it operates only on the snapshot's deep-copy data.
        nonisolated func restoreCache() -> [any KVCache]? {
            guard let snapshot, let key = partitionKey else { return nil }
            return snapshot.restore(kvBitsHint: key.kvBits, kvGroupSizeHint: key.kvGroupSize)
        }
    }

    enum LookupReason: CustomStringConvertible, Sendable {
        case hit(snapshotOffset: Int, totalTokens: Int, type: HybridCacheSnapshot.CheckpointType)
        case missNoEntries
        case missNoSnapshotInPrefix

        nonisolated var description: String {
            switch self {
            case .hit(let offset, let total, let type):
                "hit(\(type) at \(offset)/\(total))"
            case .missNoEntries:
                "miss(no entries)"
            case .missNoSnapshotInPrefix:
                "miss(no snapshot in prefix)"
            }
        }
    }

    func lookup(tokens: [Int], partitionKey: CachePartitionKey) -> LookupResult {
        guard let tree = trees[partitionKey] else {
            return LookupResult(
                snapshot: nil, partitionKey: nil,
                snapshotTokenOffset: 0, sharedPrefixLength: 0,
                reason: .missNoEntries
            )
        }

        guard let (node, sharedLen) = tree.findBestSnapshot(tokens: tokens),
              let snapshot = node.snapshot
        else {
            // No snapshot-bearing node, but the tree may still match a prefix.
            // Report the actual token-level match depth for miss diagnostics.
            let treeMatchDepth = tree.findSharedPrefixLength(tokens: tokens)
            return LookupResult(
                snapshot: nil, partitionKey: partitionKey,
                snapshotTokenOffset: 0, sharedPrefixLength: treeMatchDepth,
                reason: .missNoSnapshotInPrefix
            )
        }

        return LookupResult(
            snapshot: snapshot,
            partitionKey: partitionKey,
            snapshotTokenOffset: snapshot.tokenOffset,
            sharedPrefixLength: sharedLen,
            reason: .hit(
                snapshotOffset: snapshot.tokenOffset,
                totalTokens: tokens.count,
                type: snapshot.checkpointType
            )
        )
    }

    // MARK: - Checkpoint Planning

    /// Determine checkpoint offsets for the upcoming prefill.
    ///
    /// Captures up to two mid-prefill snapshots:
    /// - `stablePrefixOffset`: where `system + tools` end (shared across
    ///   conversations, any request with the same system/tools can hit).
    /// - `lastMessageBoundaryOffset`: where the last message ends, right
    ///   before the assistant-generation prompt. Templates (e.g. Qwen3.5)
    ///   re-render old assistants differently once they're no longer the
    ///   latest turn, so a leaf stored at the full-prompt offset doesn't
    ///   match future requests. A checkpoint at the last-message boundary
    ///   is stable across turns and enables cross-turn prefix reuse.
    ///
    /// Leaf checkpoint is NOT planned — captured post-generation via storeLeaf().
    /// Existing snapshots at the same offset are skipped.
    func planCheckpoints(
        tokens: [Int],
        stablePrefixOffset: Int?,
        lastMessageBoundaryOffset: Int? = nil,
        partitionKey: CachePartitionKey
    ) -> [(offset: Int, type: HybridCacheSnapshot.CheckpointType)] {
        var plan: [(offset: Int, type: HybridCacheSnapshot.CheckpointType)] = []
        let tree = trees[partitionKey]

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

        if let offset = lastMessageBoundaryOffset,
           offset > 0,
           offset < tokens.count,
           offset != stablePrefixOffset,  // avoid duplicate with stable prefix
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
    func storeSnapshots(
        promptTokens: [Int],
        capturedSnapshots: [HybridCacheSnapshot],
        partitionKey: CachePartitionKey
    ) {
        guard !capturedSnapshots.isEmpty else { return }

        let tree = getOrCreateTree(for: partitionKey)
        tree.insertPath(tokens: promptTokens)

        for snapshot in capturedSnapshots {
            tree.storeSnapshot(snapshot, forTokens: promptTokens, atOffset: snapshot.tokenOffset)
        }

        evictToFitBudget()
    }

    /// Store the leaf snapshot under post-response tokens.
    /// storedTokens = re-tokenized (prompt + generated response).
    func storeLeaf(
        storedTokens: [Int],
        leafSnapshot: HybridCacheSnapshot,
        partitionKey: CachePartitionKey
    ) {
        guard leafSnapshot.tokenOffset == storedTokens.count else { return }

        let tree = getOrCreateTree(for: partitionKey)
        let node = tree.insertPath(tokens: storedTokens)
        tree.storeSnapshot(leafSnapshot, on: node)

        evictToFitBudget()
    }

    // MARK: - Eviction

    /// Drop snapshots until `totalSnapshotBytes <= memoryBudgetBytes`. Uses
    /// Marconi utility scoring (`EvictionPolicy`) for eligible nodes and
    /// falls back to oldest-first when only multi-child branch snapshots
    /// remain.
    func evictToFitBudget() {
        // Pin a single clock reading and a single sorted tree order so all
        // iterations in one drain share the same recency anchor and
        // tie-break ordering.
        let now: ContinuousClock.Instant = .now
        let orderedTrees = trees.sorted { $0.key < $1.key }.map(\.value)
        while totalSnapshotBytes > memoryBudgetBytes {
            guard let (tree, node) = findEvictionCandidate(now: now, orderedTrees: orderedTrees)
            else { break }

            tree.evictSnapshot(node: node)

            if node.isLeaf {
                tree.evictNode(node: node)
            } else if node.childCount == 1, node.snapshot == nil {
                tree.collapseSingleChildNode(node)
            }
        }
    }

    // MARK: - Stats

    struct CacheStats: Sendable {
        let partitionCount: Int
        let totalNodeCount: Int
        let totalSnapshotBytes: Int
        /// Per-checkpoint-type snapshot counts. Aggregates the incremental
        /// counters from each partition's `TokenRadixTree`.
        let snapshotsByType: [HybridCacheSnapshot.CheckpointType: Int]

        nonisolated var snapshotCount: Int { snapshotsByType.values.reduce(0, +) }
    }

    var stats: CacheStats {
        var nodes = 0
        var byType: [HybridCacheSnapshot.CheckpointType: Int] = [
            .system: 0, .leaf: 0, .branchPoint: 0,
        ]
        for tree in trees.values {
            nodes += tree.nodeCount
            for (type, count) in tree.snapshotCountByType {
                byType[type, default: 0] += count
            }
        }
        return CacheStats(
            partitionCount: trees.count,
            totalNodeCount: nodes,
            totalSnapshotBytes: totalSnapshotBytes,
            snapshotsByType: byType
        )
    }

    var totalSnapshotBytes: Int {
        trees.values.reduce(0) { $0 + $1.totalSnapshotBytes }
    }

    // MARK: - Private

    private func getOrCreateTree(for key: CachePartitionKey) -> TokenRadixTree {
        if let tree = trees[key] { return tree }
        let tree = TokenRadixTree()
        trees[key] = tree
        return tree
    }

    /// Pick one snapshot to evict from the supplied (already-sorted) trees.
    ///
    /// Primary path scores Marconi-eligible nodes (snapshot + `childCount
    /// <= 1`) and returns the lowest-utility one. Fallback path returns the
    /// oldest snapshot across all nodes (including multi-child branches)
    /// so the hard budget invariant holds in degenerate cases like a
    /// zero-budget drain.
    private func findEvictionCandidate(
        now: ContinuousClock.Instant,
        orderedTrees: [TokenRadixTree]
    ) -> (tree: TokenRadixTree, node: RadixTreeNode)? {
        var nodeToTree: [ObjectIdentifier: TokenRadixTree] = [:]
        var candidates: [RadixTreeNode] = []
        for tree in orderedTrees {
            for node in tree.eligibleEvictionNodes() {
                nodeToTree[ObjectIdentifier(node)] = tree
                candidates.append(node)
            }
        }

        if let victim = EvictionPolicy.selectVictim(candidates: candidates, now: now),
           let tree = nodeToTree[ObjectIdentifier(victim.node)]
        {
            return (tree: tree, node: victim.node)
        }

        return orderedTrees
            .lazy
            .flatMap { tree in tree.allSnapshotNodes().lazy.map { (tree: tree, node: $0) } }
            .min(by: { $0.node.lastAccessTime < $1.node.lastAccessTime })
    }
}
