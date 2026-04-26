import Foundation
import MLXLMCommon

/// Draft-side DFlash cache state paired with a target ``HybridCacheSnapshot``.
///
/// The target prefix-cache partition already includes the active target model
/// fingerprint and, when DFlash is bound, the draft config identity. Keeping
/// the companion on the same radix node lets restored HTTP hits recover both
/// target and draft offsets without replaying the cached prefix.
nonisolated struct DFlashDraftCacheSnapshot: @unchecked Sendable {
    let cacheSnapshot: HybridCacheSnapshot

    var tokenOffset: Int { cacheSnapshot.tokenOffset }
    var memoryBytes: Int { cacheSnapshot.memoryBytes }

    nonisolated static func capture(
        cache: [KVCache],
        offset: Int,
        type: HybridCacheSnapshot.CheckpointType
    ) -> DFlashDraftCacheSnapshot? {
        guard let snapshot = HybridCacheSnapshot.capture(
            cache: cache,
            offset: offset,
            type: type
        ) else {
            return nil
        }
        return DFlashDraftCacheSnapshot(cacheSnapshot: snapshot)
    }

    nonisolated func restore() -> [KVCache] {
        cacheSnapshot.restore()
    }
}

/// Node in a token-level radix (compressed trie) tree.
/// Edge tokens represent the compressed path segment from parent to this node.
final class RadixTreeNode {
    var edgeTokens: [Int]
    var children: [Int: RadixTreeNode]
    var snapshot: HybridCacheSnapshot?
    /// Optional DFlash draft-side companion for ``snapshot``. This is
    /// RAM-resident only for now; when the target body is evicted or hydrated
    /// from SSD, the companion is cleared and DFlash skips that restored hit
    /// instead of replaying the prefix.
    var dflashDraftSnapshot: DFlashDraftCacheSnapshot?
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
        self.dflashDraftSnapshot = nil
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
    ///
    /// When `includeStorageRefs` is true, nodes in state 5 (committed
    /// `storageRef` without a resident body) are also treated as
    /// hittable: an SSD-resident snapshot can hydrate on demand, so
    /// it counts as a hit target during lookup. Pending refs (state
    /// 3, `committed == false`) are never hittable — returning one
    /// would race the writer and surface a half-written file.
    ///
    /// `maximumTokenOffsetExclusive` lets generation lookups reject an
    /// exact full-prompt snapshot, because that state has no saved logits
    /// for sampling the first generated token. The tree still reports the
    /// full structural match through `sharedPrefixLength`.
    func findBestSnapshot(
        tokens: [Int],
        updateAccess: Bool = true,
        includeStorageRefs: Bool = false,
        snapshotPredicate: ((HybridCacheSnapshot) -> Bool)? = nil,
        maximumTokenOffsetExclusive: Int? = nil
    ) -> (node: RadixTreeNode, sharedPrefixLength: Int)? {
        var current = root
        var pos = 0
        var bestNode: RadixTreeNode?
        var bestPrefixLength = 0

        func offsetIsAllowed(_ offset: Int) -> Bool {
            guard let maximumTokenOffsetExclusive else { return true }
            return offset < maximumTokenOffsetExclusive
        }

        func isHittable(_ node: RadixTreeNode) -> Bool {
            if let snapshot = node.snapshot {
                guard offsetIsAllowed(snapshot.tokenOffset) else { return false }
                return snapshotPredicate?(snapshot) ?? true
            }
            if includeStorageRefs,
               let storageRef = node.storageRef,
               storageRef.committed,
               offsetIsAllowed(storageRef.tokenOffset)
            {
                return true
            }
            return false
        }

        // Root can have a snapshot (e.g. empty-prefix checkpoint)
        if isHittable(current) {
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
            if isHittable(child) {
                bestNode = child
                bestPrefixLength = pos
            }
        }

        guard let node = bestNode else { return nil }
        if updateAccess { node.lastAccessTime = .now }
        return (node: node, sharedPrefixLength: bestPrefixLength)
    }

    /// Returns the node whose path exactly equals `tokens`, regardless of
    /// whether it currently carries a snapshot or storage ref.
    func findNode(tokens: [Int]) -> RadixTreeNode? {
        guard !tokens.isEmpty else { return root }

        var current = root
        var pos = 0

        while pos < tokens.count {
            guard let child = current.children[tokens[pos]] else { return nil }

            let edge = child.edgeTokens
            guard pos + edge.count <= tokens.count else { return nil }
            for edgePos in 0..<edge.count {
                guard edge[edgePos] == tokens[pos + edgePos] else { return nil }
            }

            pos += edge.count
            current = child
        }

        return current
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
        storeSnapshot(snapshot, dflashDraftSnapshot: nil, on: node)
    }

    /// Attach a target snapshot and optional DFlash draft companion to a node.
    /// Replacing the target body also replaces the companion so stale draft
    /// state cannot survive under a newer target snapshot.
    func storeSnapshot(
        _ snapshot: HybridCacheSnapshot,
        dflashDraftSnapshot: DFlashDraftCacheSnapshot?,
        on node: RadixTreeNode
    ) {
        if let old = node.snapshot {
            totalSnapshotBytes -= old.memoryBytes
            snapshotCountByType[old.checkpointType, default: 0] -= 1
        }
        if let oldDraft = node.dflashDraftSnapshot {
            totalSnapshotBytes -= oldDraft.memoryBytes
        }
        node.snapshot = snapshot
        node.dflashDraftSnapshot = dflashDraftSnapshot
        totalSnapshotBytes += snapshot.memoryBytes
        if let dflashDraftSnapshot {
            totalSnapshotBytes += dflashDraftSnapshot.memoryBytes
        }
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
        if let draft = node.dflashDraftSnapshot {
            totalSnapshotBytes -= draft.memoryBytes
            node.dflashDraftSnapshot = nil
        }
    }

    /// Remove a leaf node and clean up empty snapshot-less ancestors.
    /// Does not remove nodes that have snapshots or other children.
    ///
    /// Task 4.1.8 — regression trap. `evictNode` must not be called
    /// on a node with a live `storageRef`: removing such a node
    /// orphans the SSD-resident copy (the persisted file lives on
    /// disk but nothing in the tree can reach it). The actual
    /// suppression lives at the call site in
    /// `PrefixCacheManager.evictToFitBudget`; this assertion catches
    /// future refactors that route around the guard.
    func evictNode(node: RadixTreeNode) {
        assert(
            node.storageRef == nil,
            "evictNode must not be called on an SSD-backed leaf — "
            + "the storageRef would be orphaned. "
            + "See Task 4.1.8 in docs/marconi-hybrid-prefix-cache-implementation-plan.md."
        )
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
            if let draft = target.dflashDraftSnapshot {
                totalSnapshotBytes -= draft.memoryBytes
                target.dflashDraftSnapshot = nil
            }
            target.parent = nil

            // Continue cleaning if parent is now an empty leaf with no
            // snapshot AND no storageRef. Task 4.1.8: an ancestor with
            // a pending or committed SSD ref must not be swept by the
            // walk — doing so would orphan the persisted file the same
            // way a direct `evictNode` on a state-3/5 victim would.
            if parent.isLeaf
               && parent.snapshot == nil
               && parent.storageRef == nil
               && parent !== root
            {
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
    /// prefix that the entire tree is built on. `.leaf` and
    /// `.branchPoint` snapshots remain fully eligible so the utility
    /// scorer can trade off deeper conversation reuse against shared
    /// prefix protection.
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

    func makeTopologySnapshot(
        partition: CachePartitionKey,
        now: ContinuousClock.Instant = .now
    ) -> PromptCacheTreeSnapshot {
        var nodes: [PromptCacheTreeNodeSnapshot] = []
        var edges: [PromptCacheTreeEdgeSnapshot] = []
        collectTelemetryNode(
            root,
            partitionDigest: partition.partitionDigest,
            parentID: nil,
            path: [],
            depth: 0,
            now: now,
            nodes: &nodes,
            edges: &edges
        )

        return PromptCacheTreeSnapshot(
            id: partition.partitionDigest,
            partitionDigest: partition.partitionDigest,
            partitionSummary: partition.telemetrySummary,
            nodeCount: nodeCount,
            totalSnapshotBytes: totalSnapshotBytes,
            snapshotCount: snapshotCount,
            snapshotsByType: Dictionary(
                uniqueKeysWithValues: snapshotCountByType.map { ($0.key.wireString, $0.value) }
            ),
            nodes: nodes,
            edges: edges
        )
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
    ///
    /// Task 4.1.8 — defense-in-depth against SSD-ref orphaning.
    /// A node with a `storageRef` pins a radix path to a pending or
    /// committed SSD copy, even while its RAM body is absent. Collapsing
    /// such a node would silently drop the ref and orphan the persisted
    /// file. The eviction loop at
    /// `PrefixCacheManager.evictToFitBudget` already suppresses this
    /// call from the primary path; this guard protects against
    /// external callers (tests, future refactors) that reach the
    /// function without that context.
    func collapseSingleChildNode(_ node: RadixTreeNode) {
        guard node !== root,
              node.snapshot == nil,
              node.storageRef == nil,
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

    private func collectTelemetryNode(
        _ node: RadixTreeNode,
        partitionDigest: String,
        parentID: String?,
        path: [Int],
        depth: Int,
        now: ContinuousClock.Instant,
        nodes: inout [PromptCacheTreeNodeSnapshot],
        edges: inout [PromptCacheTreeEdgeSnapshot]
    ) {
        let pathHash = Self.telemetryPathHash(path)
        let nodeID = "\(partitionDigest):\(pathHash)"
        let storageState = Self.telemetryStorageState(node)
        let snapshot = node.snapshot
        let storageRef = node.storageRef
        let checkpointType = snapshot?.checkpointType.wireString ?? storageRef?.checkpointType.wireString
        let snapshotBytes = (snapshot?.memoryBytes ?? 0)
            + (node.dflashDraftSnapshot?.memoryBytes ?? 0)
        let storageBytes = storageRef?.bytesOnDisk ?? 0
        let scores = snapshot.map { _ in telemetryEvictionScore(for: node, now: now) } ?? nil

        nodes.append(PromptCacheTreeNodeSnapshot(
            id: nodeID,
            parentID: parentID,
            pathHash: pathHash,
            tokenOffset: node.tokenOffset,
            pathTokenCount: path.count,
            edgeTokenCount: node.edgeTokens.count,
            childCount: node.childCount,
            depth: depth,
            hasSnapshot: snapshot != nil,
            checkpointType: checkpointType,
            snapshotBytes: snapshotBytes,
            storageState: storageState,
            storageRefID: storageRef?.snapshotID,
            storageBytes: storageBytes,
            lastAccessAgeSeconds: max((now - node.lastAccessTime).seconds, 0),
            normalizedRecency: scores?.normalizedRecency,
            normalizedFlopEfficiency: scores?.normalizedFlopEfficiency,
            utility: scores?.utility
        ))

        for child in node.children.values.sorted(by: Self.telemetryChildSort) {
            let childPath = path + child.edgeTokens
            let childHash = Self.telemetryPathHash(childPath)
            let childID = "\(partitionDigest):\(childHash)"
            edges.append(PromptCacheTreeEdgeSnapshot(
                id: "\(nodeID)->\(childID)",
                parentID: nodeID,
                childID: childID,
                tokenCount: child.edgeTokens.count
            ))
            collectTelemetryNode(
                child,
                partitionDigest: partitionDigest,
                parentID: nodeID,
                path: childPath,
                depth: depth + 1,
                now: now,
                nodes: &nodes,
                edges: &edges
            )
        }
    }

    private func telemetryEvictionScore(
        for node: RadixTreeNode,
        now: ContinuousClock.Instant
    ) -> EvictionScore? {
        guard node.snapshot != nil,
              node.childCount <= 1,
              node.snapshot?.checkpointType != .system
        else { return nil }
        return EvictionPolicy.computeScores(candidates: [node], now: now).first
    }

    private static func telemetryStorageState(_ node: RadixTreeNode) -> PromptCacheStorageState {
        let hasSnapshot = node.snapshot != nil
        guard let ref = node.storageRef else {
            return hasSnapshot ? .ramOnly : .empty
        }
        switch (hasSnapshot, ref.committed) {
        case (true, false): return .pendingWrite
        case (false, false): return .pendingWriteBodyDropped
        case (true, true): return .ramAndSSD
        case (false, true): return .ssdOnly
        }
    }

    private static func telemetryChildSort(_ lhs: RadixTreeNode, _ rhs: RadixTreeNode) -> Bool {
        if lhs.tokenOffset != rhs.tokenOffset { return lhs.tokenOffset < rhs.tokenOffset }
        return lhs.edgeTokens.lexicographicallyPrecedes(rhs.edgeTokens)
    }

    private static func telemetryPathHash(_ tokens: [Int]) -> String {
        guard !tokens.isEmpty else { return "root" }
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for token in tokens {
            var value = UInt64(bitPattern: Int64(token))
            for _ in 0..<8 {
                hash ^= value & 0xff
                hash &*= 0x0000_0100_0000_01b3
                value >>= 8
            }
        }
        return String(format: "%016llx", hash)
    }
}

private extension CachePartitionKey {
    var telemetrySummary: String {
        let kv = kvBits.map { "kv\($0)" } ?? "denseKV"
        let tri: String = triAttention.isDense ? "dense" : "triattention"
        let fingerprint = modelFingerprint.map { String($0.prefix(8)) } ?? "nofp"
        return "\(modelID) · \(kv)/g\(kvGroupSize) · \(tri) · \(fingerprint)"
    }
}
