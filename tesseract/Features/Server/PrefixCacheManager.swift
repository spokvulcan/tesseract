import Foundation
import MLXLMCommon

/// Partition key for isolating radix trees by runtime configuration.
///
/// Tool/template digests from HTTPPrefixCacheKey are intentionally dropped:
/// different tools/context → different tokens → different radix paths → naturally isolated.
struct CachePartitionKey: Hashable {
    let modelID: String
    let kvBits: Int?
    let kvGroupSize: Int
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
        func restoreCache() -> [any KVCache]? {
            guard let snapshot, let key = partitionKey else { return nil }
            return snapshot.restore(kvBitsHint: key.kvBits, kvGroupSizeHint: key.kvGroupSize)
        }
    }

    enum LookupReason: CustomStringConvertible {
        case hit(snapshotOffset: Int, totalTokens: Int, type: HybridCacheSnapshot.CheckpointType)
        case missNoEntries
        case missNoSnapshotInPrefix

        var description: String {
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
            return LookupResult(
                snapshot: nil, partitionKey: nil,
                snapshotTokenOffset: 0, sharedPrefixLength: 0,
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
    /// Phase 1: stable-prefix boundary only (if known and not already stored).
    /// Leaf checkpoint is NOT planned — captured post-generation via storeLeaf().
    func planCheckpoints(
        tokens: [Int],
        stablePrefixOffset: Int?,
        partitionKey: CachePartitionKey
    ) -> [(offset: Int, type: HybridCacheSnapshot.CheckpointType)] {
        var plan: [(offset: Int, type: HybridCacheSnapshot.CheckpointType)] = []

        guard let offset = stablePrefixOffset, offset > 0, offset < tokens.count else {
            return plan
        }

        if let tree = trees[partitionKey],
           let (node, _) = tree.findBestSnapshot(tokens: Array(tokens[0..<offset]), updateAccess: false),
           node.tokenOffset == offset,
           node.snapshot?.checkpointType == .system
        {
            return plan
        }

        plan.append((offset: offset, type: .system))
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

    /// Evict lowest-priority snapshots across all partitions until under budget.
    /// Phase 1: type-based LRU — leaf first, then branchPoint, then system.
    /// Within same type, least-recently-accessed first.
    func evictToFitBudget() {
        while totalSnapshotBytes > memoryBudgetBytes {
            guard let (tree, node) = findEvictionCandidate() else { break }

            tree.evictSnapshot(node: node)

            if node.isLeaf {
                tree.evictNode(node: node)
            } else if node.childCount == 1, node.snapshot == nil {
                tree.collapseSingleChildNode(node)
            }
        }
    }

    // MARK: - Stats

    struct CacheStats {
        let partitionCount: Int
        let totalNodeCount: Int
        let totalSnapshotBytes: Int
        let snapshotCount: Int
    }

    var stats: CacheStats {
        var nodes = 0
        var snapshots = 0
        for tree in trees.values {
            nodes += tree.nodeCount
            snapshots += tree.snapshotCount
        }
        return CacheStats(
            partitionCount: trees.count,
            totalNodeCount: nodes,
            totalSnapshotBytes: totalSnapshotBytes,
            snapshotCount: snapshots
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

    /// Phase 1: evict .leaf first, then .branchPoint, then .system last.
    /// Within same type: oldest lastAccessTime first.
    private func findEvictionCandidate() -> (tree: TokenRadixTree, node: RadixTreeNode)? {
        var best: (tree: TokenRadixTree, node: RadixTreeNode)?
        var bestPriority = Int.max
        var bestTime: ContinuousClock.Instant?

        for tree in trees.values {
            for node in tree.allSnapshotNodes() {
                guard let snap = node.snapshot else { continue }
                let priority = Self.evictionPriority(snap.checkpointType)

                if priority < bestPriority
                    || (priority == bestPriority && bestTime.map({ node.lastAccessTime < $0 }) == true)
                {
                    best = (tree, node)
                    bestPriority = priority
                    bestTime = node.lastAccessTime
                }
            }
        }

        return best
    }

    /// Lower value = evict first.
    private static func evictionPriority(_ type: HybridCacheSnapshot.CheckpointType) -> Int {
        switch type {
        case .leaf: 0
        case .branchPoint: 1
        case .system: 2
        }
    }
}
