import Foundation
import MLXLMCommon

/// Node in a token-level radix (compressed trie) tree.
/// Edge tokens represent the compressed path segment from parent to this node.
final class RadixTreeNode {
    var edgeTokens: [Int]
    var children: [Int: RadixTreeNode]
    var snapshot: HybridCacheSnapshot?
    /// SSD persistence tier back-reference. Non-nil while an SSD
    /// write is pending (`committed == false`) or has landed
    /// (`committed == true`); `nil` for RAM-only nodes and after a
    /// writer drop callback fires. Drives the five-state lifecycle
    /// in `TieredSnapshotStore` — see the plan's
    /// "Storage ref lifecycle" section.
    var storageRef: SnapshotStorageRef?
    /// Cumulative token count from root to the end of this node's edge.
    var tokenOffset: Int
    var lastAccessTime: ContinuousClock.Instant
    weak var parent: RadixTreeNode?

    init(
        edgeTokens: [Int] = [],
        tokenOffset: Int = 0,
        parent: RadixTreeNode? = nil
    ) {
        self.edgeTokens = edgeTokens
        self.children = [:]
        self.snapshot = nil
        self.storageRef = nil
        self.tokenOffset = tokenOffset
        self.lastAccessTime = .now
        self.parent = parent
    }

    var isLeaf: Bool { children.isEmpty }
    var childCount: Int { children.count }
}

/// Token-level radix tree for prefix cache lookup.
/// Stores compressed token paths with optional HybridCacheSnapshot at nodes.
/// Partitioned externally by (modelID, kvBits, kvGroupSize) via PrefixCacheManager.
@MainActor
final class TokenRadixTree {
    private let root: RadixTreeNode
    private(set) var nodeCount: Int = 1
    private(set) var totalSnapshotBytes: Int = 0
    /// Per-checkpoint-type snapshot counts. Maintained incrementally on
    /// every `storeSnapshot` / `evictSnapshot` / `evictNode` so callers
    /// don't have to walk the tree to bucket snapshots by type.
    private(set) var snapshotCountByType: [HybridCacheSnapshot.CheckpointType: Int] = [
        .system: 0, .leaf: 0, .branchPoint: 0,
    ]

    var snapshotCount: Int {
        snapshotCountByType.values.reduce(0, +)
    }

    init() {
        self.root = RadixTreeNode()
    }

    // MARK: - Lookup

    /// Find the deepest node with a snapshot whose offset ≤ the shared prefix length.
    ///
    /// Walks the tree matching tokens. Tracks the deepest snapshot-bearing node.
    /// On lookup hit, updates `lastAccessTime` on the returned node only (not ancestors).
    func findBestSnapshot(tokens: [Int], updateAccess: Bool = true) -> (node: RadixTreeNode, sharedPrefixLength: Int)? {
        var current = root
        var pos = 0
        var bestNode: RadixTreeNode?
        var bestPrefixLength = 0

        // Root can have a snapshot (e.g. empty-prefix checkpoint)
        if current.snapshot != nil {
            bestNode = current
            bestPrefixLength = 0
        }

        while pos < tokens.count {
            guard let child = current.children[tokens[pos]] else { break }

            let edge = child.edgeTokens
            var edgePos = 0
            while edgePos < edge.count && pos < tokens.count && edge[edgePos] == tokens[pos] {
                edgePos += 1
                pos += 1
            }

            if edgePos < edge.count {
                // Diverged mid-edge — this node's snapshot offset is past the divergence point
                break
            }

            current = child
            if child.snapshot != nil {
                bestNode = child
                bestPrefixLength = pos
            }
        }

        guard let node = bestNode else { return nil }
        if updateAccess { node.lastAccessTime = .now }
        return (node: node, sharedPrefixLength: bestPrefixLength)
    }

    /// Returns how many leading tokens from `tokens` match the tree, regardless of
    /// whether any snapshot exists along the path. Useful for miss diagnostics.
    func findSharedPrefixLength(tokens: [Int]) -> Int {
        var current = root
        var pos = 0

        while pos < tokens.count {
            guard let child = current.children[tokens[pos]] else { break }

            let edge = child.edgeTokens
            var edgePos = 0
            while edgePos < edge.count && pos < tokens.count && edge[edgePos] == tokens[pos] {
                edgePos += 1
                pos += 1
            }

            if edgePos < edge.count { break }
            current = child
        }

        return pos
    }

    // MARK: - Insert

    /// Insert a path for the token sequence. Does NOT store a snapshot.
    /// Creates nodes as needed, splitting compressed edges on branch points.
    /// Returns the terminal node at the end of the path.
    @discardableResult
    func insertPath(tokens: [Int]) -> RadixTreeNode {
        guard !tokens.isEmpty else { return root }

        var current = root
        var pos = 0

        while pos < tokens.count {
            guard let child = current.children[tokens[pos]] else {
                let newNode = RadixTreeNode(
                    edgeTokens: Array(tokens[pos...]),
                    tokenOffset: tokens.count,
                    parent: current
                )
                current.children[tokens[pos]] = newNode
                nodeCount += 1
                return newNode
            }

            let edge = child.edgeTokens
            var edgePos = 0
            while edgePos < edge.count && pos < tokens.count && edge[edgePos] == tokens[pos] {
                edgePos += 1
                pos += 1
            }

            if edgePos == edge.count {
                current = child
            } else {
                splitEdge(parent: current, child: child, at: edgePos)
                let splitNode = current.children[edge[0]]!

                if pos < tokens.count {
                    let newNode = RadixTreeNode(
                        edgeTokens: Array(tokens[pos...]),
                        tokenOffset: tokens.count,
                        parent: splitNode
                    )
                    splitNode.children[tokens[pos]] = newNode
                    nodeCount += 1
                    return newNode
                }
                return splitNode
            }
        }
        return current
    }

    /// Attach a snapshot to a node. Use the node returned by `insertPath`.
    func storeSnapshot(_ snapshot: HybridCacheSnapshot, on node: RadixTreeNode) {
        if let old = node.snapshot {
            totalSnapshotBytes -= old.memoryBytes
            snapshotCountByType[old.checkpointType, default: 0] -= 1
        }
        node.snapshot = snapshot
        totalSnapshotBytes += snapshot.memoryBytes
        snapshotCountByType[snapshot.checkpointType, default: 0] += 1
        node.lastAccessTime = .now
    }

    /// Attach a snapshot at a specific offset on an already-inserted token path.
    /// Walks the tree guided by `tokens` to find the node at `offset`, splitting
    /// if needed. Returns false if the path diverges before reaching `offset`.
    ///
    /// Use this for mid-prefill checkpoints where `insertPath` was called once
    /// for the full prompt but snapshots are captured at intermediate offsets
    /// (e.g. stable-prefix boundary at 4000 on an 8000-token prompt).
    @discardableResult
    func storeSnapshot(_ snapshot: HybridCacheSnapshot, forTokens tokens: [Int], atOffset offset: Int) -> Bool {
        guard offset > 0, offset <= tokens.count,
              snapshot.tokenOffset == offset
        else { return false }

        let node = insertPath(tokens: Array(tokens[0..<offset]))
        guard node.tokenOffset == offset else { return false }

        storeSnapshot(snapshot, on: node)
        return true
    }

    // MARK: - Eviction

    /// Remove a node's snapshot. Node structure stays intact.
    func evictSnapshot(node: RadixTreeNode) {
        guard let snap = node.snapshot else { return }
        totalSnapshotBytes -= snap.memoryBytes
        snapshotCountByType[snap.checkpointType, default: 0] -= 1
        node.snapshot = nil
    }

    /// Remove a leaf node and clean up empty snapshot-less ancestors.
    /// Does not remove nodes that have snapshots or other children.
    func evictNode(node: RadixTreeNode) {
        guard node.isLeaf else { return }

        var current: RadixTreeNode? = node
        while let target = current, target !== root {
            guard let parent = target.parent else { break }

            // Remove from parent's children
            let key = target.edgeTokens.first!
            parent.children.removeValue(forKey: key)
            nodeCount -= 1

            if let snap = target.snapshot {
                totalSnapshotBytes -= snap.memoryBytes
                snapshotCountByType[snap.checkpointType, default: 0] -= 1
                target.snapshot = nil
            }
            target.parent = nil

            // Continue cleaning if parent is now an empty leaf with no snapshot
            if parent.isLeaf && parent.snapshot == nil && parent !== root {
                current = parent
            } else {
                break
            }
        }
    }

    /// Snapshot-bearing nodes eligible for Marconi utility scoring:
    /// node has a snapshot AND `childCount <= 1` AND `checkpointType !=
    /// .system`. Multi-child nodes are protected because they hold shared
    /// cache state. **Only** `.system` (stable prefix) snapshots are
    /// type-protected — they sit linearly on the path from root in
    /// single-conversation usage and represent the cross-conversation hot
    /// prefix that the entire tree is built on. `.lastMessageBoundary`
    /// snapshots are explicitly NOT protected because long conversations
    /// accumulate one boundary per turn; protecting all of them would fill
    /// the budget with stale boundaries from old turns and cause the
    /// freshly-stored leaf for the current turn to be the only eligible
    /// eviction candidate, killing it on admission. LRU eviction over the
    /// boundary set keeps the freshest boundary alive (recency wins) while
    /// reaping old ones, which is the behaviour we want.
    /// The hard budget invariant is preserved by `findEvictionCandidate`'s
    /// fallback path, which drops the oldest snapshot from
    /// `allSnapshotNodes()` when the eligible set is empty.
    func eligibleEvictionNodes() -> [RadixTreeNode] {
        var result: [RadixTreeNode] = []
        collectEligible(node: root, into: &result)
        return result
    }

    /// All snapshot-bearing nodes, including multi-child branch nodes.
    /// Used by the eviction fallback when `eligibleEvictionNodes()` is
    /// empty but the hard budget still requires dropping something.
    func allSnapshotNodes() -> [RadixTreeNode] {
        var result: [RadixTreeNode] = []
        collectAllSnapshots(node: root, into: &result)
        return result
    }

    /// Reconstruct the absolute token path from `root` to `node` by
    /// concatenating each ancestor's `edgeTokens` in walk order. Used
    /// by the snapshot inventory builder so the alpha tuner can reseed
    /// a sandbox cache with the same paths.
    func pathToNode(_ node: RadixTreeNode) -> [Int] {
        var segments: [[Int]] = []
        var current: RadixTreeNode? = node
        while let n = current, n !== root {
            segments.append(n.edgeTokens)
            current = n.parent
        }
        return segments.reversed().flatMap { $0 }
    }

    // MARK: - Speculative branch-point detection

    /// Dry-run insertion: returns the absolute token offset at which
    /// `insertPath(tokens:)` would call `splitEdge`, or `nil` if insertion
    /// would not split any compressed edge (empty tokens, empty tree,
    /// node-boundary divergence, or exact match).
    ///
    /// Note: a strict prefix of an existing edge also splits the edge — the
    /// helper reports the offset honestly, and the planner suppresses that
    /// degenerate case via `splitOffset < tokens.count`.
    func findIntermediateSplitOffsetForInsertion(tokens: [Int]) -> Int? {
        guard !tokens.isEmpty else { return nil }

        var current = root
        var pos = 0

        while pos < tokens.count {
            guard let child = current.children[tokens[pos]] else { return nil }

            let edge = child.edgeTokens
            var edgePos = 0
            while edgePos < edge.count && pos < tokens.count && edge[edgePos] == tokens[pos] {
                edgePos += 1
                pos += 1
            }

            if edgePos < edge.count {
                return current.tokenOffset + edgePos
            }

            current = child
        }

        return nil
    }

    /// Collapse a snapshot-less node with exactly one child.
    /// Concatenates the node's edgeTokens into the child edge and re-links parent→child.
    /// Preserves radix compression after snapshot eviction.
    func collapseSingleChildNode(_ node: RadixTreeNode) {
        guard node !== root,
              node.snapshot == nil,
              node.childCount == 1,
              let parent = node.parent,
              let onlyChild = node.children.values.first
        else { return }

        // Merge edges: node.edge + child.edge
        onlyChild.edgeTokens = node.edgeTokens + onlyChild.edgeTokens
        onlyChild.parent = parent

        // Re-link in parent
        let key = node.edgeTokens.first!
        parent.children[key] = onlyChild

        node.parent = nil
        node.children.removeAll()
        nodeCount -= 1
    }

    // MARK: - Private

    /// Split a child's edge at `splitPos`, creating an intermediate node.
    /// Before: parent → child(edge=[a,b,c,d])
    /// After:  parent → intermediate(edge=[a,b]) → child(edge=[c,d])
    private func splitEdge(parent: RadixTreeNode, child: RadixTreeNode, at splitPos: Int) {
        let originalEdge = child.edgeTokens

        let intermediate = RadixTreeNode(
            edgeTokens: Array(originalEdge[..<splitPos]),
            tokenOffset: parent.tokenOffset + splitPos,
            parent: parent
        )

        child.edgeTokens = Array(originalEdge[splitPos...])
        child.parent = intermediate
        intermediate.children[child.edgeTokens[0]] = child

        parent.children[originalEdge[0]] = intermediate
        nodeCount += 1
    }

    private func collectAllSnapshots(node: RadixTreeNode, into result: inout [RadixTreeNode]) {
        if node.snapshot != nil {
            result.append(node)
        }
        for child in node.children.values {
            collectAllSnapshots(node: child, into: &result)
        }
    }

    private func collectEligible(node: RadixTreeNode, into result: inout [RadixTreeNode]) {
        if let snapshot = node.snapshot,
           node.childCount <= 1,
           snapshot.checkpointType != .system
        {
            result.append(node)
        }
        for child in node.children.values {
            collectEligible(node: child, into: &result)
        }
    }
}
