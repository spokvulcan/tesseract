import Foundation
import MLXLMCommon

/// Node in a token-level radix (compressed trie) tree.
/// Edge tokens represent the compressed path segment from parent to this node.
final class RadixTreeNode {
    var edgeTokens: [Int]
    var children: [Int: RadixTreeNode]
    /// The node's snapshot lifecycle — owns both the RAM body and the
    /// SSD `SnapshotRef`. Mutated **only** by `TokenRadixTree`'s
    /// transition methods (the sole mutator), never assigned from
    /// outside the tree. See `SnapshotState`.
    var state: SnapshotState
    /// **Chain-Prefix Restore** point (ADR-0012): set when a **Leaf
    /// Extension Admission** consumed this node's leaf entry at writer
    /// commit, keeping the boundary restorable from the owning chain's
    /// leading segments. Orthogonal to `state` — a backing channel the
    /// node holds *in addition to* whatever its own lifecycle says.
    /// Mutated **only** by `TokenRadixTree`, same discipline as `state`.
    var chainPrefixRestorePoint: ChainPrefixRestorePoint?
    /// Cumulative token count from root to the end of this node's edge.
    var tokenOffset: Int
    var lastAccessTime: ContinuousClock.Instant
    /// Lookup hits this node has served (bumped alongside
    /// `lastAccessTime`). The **adaptive write eagerness** input
    /// (ADR-0019, PRD #150): a node whose SSD write was deferred while
    /// RAM was healthy earns its backing once the hit count crosses
    /// `SSDWriteEagernessPolicy.hitCountThreshold` — HiCache's
    /// `write_through_selective` precedent.
    var hitCount: Int = 0
    /// One-shot latch for the hit-count promotion write: set when a
    /// promotion has been scheduled or attempted for this node, so a
    /// rejected write (budget, back-pressure) does not retry on every
    /// subsequent hit.
    var ssdPromotionAttempted: Bool = false
    weak var parent: RadixTreeNode?

    init(
        edgeTokens: [Int] = [],
        tokenOffset: Int = 0,
        parent: RadixTreeNode? = nil
    ) {
        self.edgeTokens = edgeTokens
        self.children = [:]
        self.state = .empty
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
    /// Per-checkpoint-type RAM-body counts. Reconciled incrementally at
    /// the single mutation chokepoint (every transition method) from each
    /// state's resident body, so callers don't have to walk the tree to
    /// bucket snapshots by type. An `ssdOnly` node has no resident body
    /// and is not counted.
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
    /// When `includeSnapshotRefs` is true, nodes in state 5 (committed
    /// `SnapshotRef` without a resident body) are also treated as
    /// hittable: an SSD-resident snapshot can hydrate on demand, so
    /// it counts as a hit target during lookup. Pending refs are never
    /// hittable — returning one would race the writer and surface a
    /// half-written file.
    func findBestSnapshot(
        tokens: [Int],
        updateAccess: Bool = true,
        includeSnapshotRefs: Bool = false
    ) -> (node: RadixTreeNode, sharedPrefixLength: Int)? {
        var current = root
        var pos = 0
        var bestNode: RadixTreeNode?
        var bestPrefixLength = 0

        // RAM body → always hittable (`hasResidentBody`). A committed SSD
        // ref (`isHittable` adds state 5) is hittable only when the caller
        // opts in, since it hydrates on demand — and so is a
        // **Chain-Prefix Restore** point (ADR-0012), which hydrates from
        // the owning chain's leading segments. Pending refs are never
        // hittable.
        func isHittable(_ node: RadixTreeNode) -> Bool {
            includeSnapshotRefs
                ? node.state.isHittable || node.chainPrefixRestorePoint != nil
                : node.state.hasResidentBody
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
        if updateAccess {
            node.lastAccessTime = .now
            node.hitCount += 1
        }
        return (node: node, sharedPrefixLength: bestPrefixLength)
    }

    /// The deepest strict-ancestor node on `tokens` whose state is a
    /// `.leaf` carrying a live Snapshot Ref (pending or committed) — the
    /// base a **Leaf Extension Admission** slices against. Strict: a
    /// node at exactly `tokens.count` is the leaf being re-admitted,
    /// never its own base. Read-only — no access-time bump, no
    /// insertion. `nil` when no SSD-backed ancestor leaf exists on the
    /// path (first leaf of a conversation, post-rewind divergence,
    /// RAM-only history).
    func deepestRefBearingLeaf(tokens: [Int]) -> SnapshotRef? {
        var current = root
        var pos = 0
        var best: SnapshotRef?

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
            if pos < tokens.count,
                child.state.checkpointType == .leaf,
                let ref = child.state.ref
            {
                best = ref
            }
        }

        return best
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

    /// Ensure nodes exist at multiple prefix offsets of one token path,
    /// walking the compressed tree once after inserting the full path.
    /// Used by warm-start chain-prefix reconstruction so a chain with many
    /// inherited segments does not rebuild every prefix from the root.
    @discardableResult
    func ensurePrefixNodes(
        tokens: [Int],
        offsets: [Int]
    ) -> [Int: RadixTreeNode] {
        let targets = Array(Set(offsets.filter { $0 >= 0 && $0 <= tokens.count })).sorted()
        guard !targets.isEmpty else { return [:] }

        insertPath(tokens: tokens)

        var result: [Int: RadixTreeNode] = [:]
        var current = root
        var pos = 0
        var targetIndex = 0

        while targetIndex < targets.count, targets[targetIndex] == 0 {
            result[0] = root
            targetIndex += 1
        }

        while targetIndex < targets.count {
            let target = targets[targetIndex]
            if target == pos {
                result[target] = current
                targetIndex += 1
                continue
            }

            guard pos < tokens.count,
                let child = current.children[tokens[pos]]
            else { break }

            let edgeEnd = pos + child.edgeTokens.count
            if target < edgeEnd {
                splitEdge(parent: current, child: child, at: target - pos)
                let splitNode = current.children[tokens[pos]]!
                current = splitNode
                pos = target
                result[target] = splitNode
                targetIndex += 1
            } else {
                current = child
                pos = edgeEnd
                if target == pos {
                    result[target] = current
                    targetIndex += 1
                }
            }
        }

        return result
    }

    /// Attach (or replace) a RAM body on a node. Use the node returned by
    /// `insertPath`. Routes through the `storingBody` transition so the
    /// byte/count budgets reconcile at the single chokepoint.
    func storeSnapshot(_ snapshot: HybridCacheSnapshot, on node: RadixTreeNode) {
        let old = node.state
        let (next, _) = old.storingBody(snapshot)
        commit(next, on: node, from: old)
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
    func storeSnapshot(
        _ snapshot: HybridCacheSnapshot, forTokens tokens: [Int], atOffset offset: Int
    ) -> Bool {
        guard offset > 0, offset <= tokens.count,
            snapshot.tokenOffset == offset
        else { return false }

        let node = insertPath(tokens: Array(tokens[0..<offset]))
        guard node.tokenOffset == offset else { return false }

        storeSnapshot(snapshot, on: node)
        return true
    }

    // MARK: - Snapshot-state transitions (the sole mutator)
    //
    // Every node-state change goes through one of these chokepoints. Each
    // applies the pure `SnapshotState` transition, reconciles the
    // byte/count budgets from state residency, and — when a transition
    // leaves the node `empty` (`.becameEmpty`) — self-heals by removing
    // the node if topology allows. No code outside this type mutates
    // `node.state`.

    /// Attach a freshly enqueued pending ref to a node that already holds
    /// a RAM body (the SSD-admission edge). Strict: an `.ignored` result
    /// here is a programmer error (the caller is admitting onto a
    /// body-less node). Returns the superseded ref's ID when this
    /// re-admission replaced a still-live ref, so the SSD router can
    /// delete its backing before it orphans a file + manifest entry.
    @discardableResult
    func admit(node: RadixTreeNode, ref: SnapshotRef) -> String? {
        let old = node.state
        let (next, effect, supersededID) = old.admitting(ref)
        precondition(
            effect != .ignored(.notResident),
            "admit requires a resident body (state 1/2/4); node was \(old.label)"
        )
        commit(next, on: node, from: old)
        return supersededID
    }

    /// Commit a pending ref (SSD-writer callback, **forgiving**). Returns
    /// the `StateEffect` so the router can log an `.ignored(reason)`
    /// without recovering — newer ref wins. `bytesOnDisk` (when provided)
    /// refreshes the ref with the writer's durable chain byte count —
    /// see `SnapshotState.committing`.
    @discardableResult
    func commitRef(
        node: RadixTreeNode,
        expectedID: String,
        bytesOnDisk: Int? = nil
    ) -> StateEffect {
        let old = node.state
        let (next, effect) = old.committing(expectedID: expectedID, bytesOnDisk: bytesOnDisk)
        commit(next, on: node, from: old)
        return effect
    }

    /// Drop a pending ref (SSD-writer callback, **forgiving**). Returns
    /// the `StateEffect`; self-heals when the drop empties the node
    /// (state 3 → removed).
    @discardableResult
    func dropRef(node: RadixTreeNode, expectedID: String) -> StateEffect {
        let old = node.state
        let (next, effect) = old.droppingRef(expectedID: expectedID)
        commit(next, on: node, from: old)
        if effect == .becameEmpty { selfHeal(node) }
        return effect
    }

    /// Drop a node's RAM body (RAM-budget eviction). Strict: an ignored
    /// result means the node had no resident body. Returns the
    /// `DropBodyResult` carrying the eviction telemetry inputs
    /// (checkpoint type, freed bytes, surviving ref ID). Self-heals when
    /// the drop empties the node (state 1 → removed); a ref-bearing node
    /// (state 2/4) settles in place.
    @discardableResult
    func dropBody(node: RadixTreeNode) -> DropBodyResult {
        let old = node.state
        let (next, result) = old.droppingBody()
        if case .ignored = result.effect {
            preconditionFailure("dropBody requires a resident body; node was \(old.label)")
        }
        commit(next, on: node, from: old)
        if result.effect == .becameEmpty { selfHeal(node) }
        return result
    }

    /// Hydrate a committed-ref node with a freshly loaded body (state 5 →
    /// state 4). **Forgiving:** like the SSD-writer commit/drop edges, this
    /// completes a lookup that captured the node *before* an off-main
    /// `loadSync`. If the node left `ssdOnly` in that window (a concurrent
    /// transition today, or a future off-lease prefetch), the hydration is
    /// a logged no-op — the newer state wins — rather than a `precondition`
    /// that would abort the whole inference server on a benign lost race.
    /// Returns the `StateEffect` so the caller can log the `.ignored` case.
    @discardableResult
    func hydrate(node: RadixTreeNode, body: HybridCacheSnapshot) -> StateEffect {
        let old = node.state
        let (next, effect) = old.hydrating(body)
        if case .ignored = effect { return effect }
        commit(next, on: node, from: old)
        node.lastAccessTime = .now
        return effect
    }

    /// Clear a committed Snapshot Ref whose SSD backing is gone (state 5
    /// → removed, or a still-bodied committed node keeps its body). Two
    /// callers: the hydration-failure hop from `LLMActor`, and the SSD
    /// router's eager clear when the tier's LRU cut evicts a committed
    /// resident. **Forgiving** (same rationale as `hydrate`): both
    /// callers captured the node before an off-main event, so if it left
    /// the committed/`ssdOnly` states in that window the clear is a
    /// logged no-op rather than a process abort. The SSD file is already
    /// deleted by the failing `loadSync` / the tier's cut, so clearing
    /// never orphans a backing. Self-heals when clearing empties the node.
    @discardableResult
    func clearCommittedSnapshotRefAfterBackingLoss(node: RadixTreeNode) -> StateEffect {
        let old = node.state
        let (next, effect) = old.clearingCommittedRefAfterBackingLoss()
        if case .ignored = effect { return effect }
        commit(next, on: node, from: old)
        if effect == .becameEmpty { selfHeal(node) }
        return effect
    }

    /// Discard a Snapshot Ref after the SSD backing has already been
    /// explicitly deleted or cancelled by the caller. Strict: the node
    /// must currently be ref-bearing. Self-heals when the discard empties
    /// the node.
    @discardableResult
    func discardSnapshotRefAfterExplicitDelete(node: RadixTreeNode) -> StateEffect {
        let old = node.state
        let (next, effect) = old.discardingRefAfterExplicitDelete()
        if case .ignored = effect {
            preconditionFailure(
                "discardSnapshotRefAfterExplicitDelete requires a ref-bearing node; "
                    + "node was \(old.label)"
            )
        }
        commit(next, on: node, from: old)
        if effect == .becameEmpty { selfHeal(node) }
        return effect
    }

    /// Warm-start restore: place a node directly in the `ssdOnly` state
    /// (committed ref, no body) through the same sole-mutator seam.
    func restoreCommittedRef(node: RadixTreeNode, ref: SnapshotRef) {
        let old = node.state
        let (next, effect) = old.restoringCommittedRef(ref)
        precondition(
            effect != .ignored(.notResident),
            "restoreCommittedRef requires an empty node; node was \(old.label)"
        )
        commit(next, on: node, from: old)
    }

    // MARK: - Chain-prefix restore points (ADR-0012)

    /// The fold's ownership conversion: a **Leaf Extension Admission**
    /// consumed this node's manifest entry at writer commit, so the
    /// node's own ref is discarded — but instead of going dark, the node
    /// keeps a **Chain-Prefix Restore** point into the new owner's chain,
    /// whose leading segments are exactly the consumed entry's bytes.
    /// Strict: the node must currently be ref-bearing (the caller matched
    /// `refID` to the consumed base). The node is *kept* even when the
    /// discard empties its state — the point keeps it addressable.
    @discardableResult
    func convertConsumedBaseToChainPrefixRestorePoint(
        node: RadixTreeNode,
        ownerSnapshotID: String
    ) -> ChainPrefixRestorePoint {
        guard let baseRef = node.state.ref else {
            preconditionFailure(
                "convertConsumedBaseToChainPrefixRestorePoint requires a ref-bearing node; "
                    + "node was \(node.state.label)"
            )
        }
        let point = ChainPrefixRestorePoint(
            ownerSnapshotID: ownerSnapshotID, consumedBase: baseRef
        )
        node.chainPrefixRestorePoint = point
        let old = node.state
        let (next, _) = old.discardingRefAfterExplicitDelete()
        commit(next, on: node, from: old)
        // No self-heal on `.becameEmpty`: the restore point keeps the
        // boundary reachable, which is the entire purpose of the
        // conversion (`selfHeal` would refuse anyway — belt and braces).
        return point
    }

    /// Warm start's reconstruction arm (issue #99): attach a restore
    /// point recovered from a restored chain head's inherited-segment
    /// descriptors to the empty node at its boundary. Unlike the fold's
    /// conversion there is no ref to discard — the base's manifest entry
    /// died with the fold before the restart. Strict: the caller checks
    /// the node is point-less and empty first.
    func attachChainPrefixRestorePoint(
        node: RadixTreeNode,
        point: ChainPrefixRestorePoint
    ) {
        precondition(
            node.chainPrefixRestorePoint == nil,
            "attachChainPrefixRestorePoint requires a point-less node"
        )
        node.chainPrefixRestorePoint = point
    }

    /// Re-own a restore point in place after a later fold consumed its
    /// current owner: the new head inherits the old owner's whole chain,
    /// so every boundary the old owner covered stays covered. No-op for
    /// a point-less node (the index entry outlived a cleared point).
    func reownChainPrefixRestorePoint(node: RadixTreeNode, to newOwnerID: String) {
        guard let point = node.chainPrefixRestorePoint else { return }
        node.chainPrefixRestorePoint = point.settingOwner(newOwnerID)
    }

    /// Clear a restore point after its owning chain died (explicit
    /// delete, SSD LRU cut, hydration failure) — the same eager
    /// backing-loss semantics as `clearCommittedSnapshotRefAfterBackingLoss`.
    /// Forgiving: clearing a point-less node is a no-op. Self-heals when
    /// the node has nothing else left.
    func clearChainPrefixRestorePoint(node: RadixTreeNode) {
        guard node.chainPrefixRestorePoint != nil else { return }
        node.chainPrefixRestorePoint = nil
        if node.state.isEmpty { selfHeal(node) }
    }

    // MARK: - Mutation chokepoint internals

    /// Install `next` on `node` and reconcile the RAM byte/count budgets
    /// from the difference in resident body between `old` and `next`.
    /// Every transition method funnels through here.
    private func commit(_ next: SnapshotState, on node: RadixTreeNode, from old: SnapshotState) {
        if let oldType = old.body?.checkpointType {
            snapshotCountByType[oldType, default: 0] -= 1
        }
        if let newType = next.body?.checkpointType {
            snapshotCountByType[newType, default: 0] += 1
        }
        totalSnapshotBytes += next.residentBodyBytes - old.residentBodyBytes
        node.state = next
    }

    /// Topology-respecting removal, triggered only on `.becameEmpty`
    /// (`node.state == .empty`, so `canEvictNode` holds and removing the
    /// node cannot orphan an SSD ref). Leaf → detach + sweep empty
    /// ancestors; single-child → collapse; multi-child empty node → stays
    /// as a structural junction. A node holding a **Chain-Prefix
    /// Restore** point is never healed away — its state is empty but the
    /// boundary it addresses is still backed by the owning chain.
    private func selfHeal(_ node: RadixTreeNode) {
        guard node.chainPrefixRestorePoint == nil else { return }
        if node.isLeaf {
            detachEmptyLeaf(node)
        } else if node.childCount == 1 {
            collapseEmptySingleChild(node)
        }
    }

    /// Detach an empty leaf and sweep empty leaf ancestors. Precondition:
    /// `node` is a leaf in the `empty` state. The ancestor walk stops at
    /// any node that still owns a body or ref (`!canEvictNode` or a body),
    /// which can no longer orphan an SSD ref because the walk only
    /// continues through `empty` ancestors.
    private func detachEmptyLeaf(_ node: RadixTreeNode) {
        var current: RadixTreeNode? = node
        while let target = current, target !== root {
            guard let parent = target.parent else { break }

            // Orphan invariant, enforced rather than asserted-by-comment: a
            // node that still owns a live SSD ref (`!canEvictNode`) must
            // never be unlinked, or its on-disk file is orphaned. `selfHeal`
            // only reaches here on `.becameEmpty`, so this holds today — the
            // guard makes `canEvictNode` the load-bearing barrier for any
            // future removal caller instead of a convention.
            guard target.state.canEvictNode else {
                Log.agent.fault(
                    "TokenRadixTree.detachEmptyLeaf refused to unlink a node that still "
                        + "owns an SSD ref (state \(target.state.label)); leaving it in the tree"
                )
                break
            }

            let key = target.edgeTokens.first!
            parent.children.removeValue(forKey: key)
            nodeCount -= 1
            target.parent = nil

            // Continue only through empty leaf ancestors.
            if parent.isLeaf, parent.state.isEmpty, parent !== root {
                current = parent
            } else {
                break
            }
        }
    }

    /// Snapshot-bearing nodes eligible for Marconi utility scoring —
    /// every body-bearing node, no type or topology shielding
    /// (ADR-0019, PRD #149: **no RAM type-shielding**). `.system` and
    /// multi-child bodies used to be excluded here, which let them
    /// survive above a pressure-collapsed budget while the fresh leaf
    /// died; now one recovery-cost policy prices every body, dropping
    /// a body is recoverable (Snapshot Demotion persists unbacked
    /// victims first), and `.system` protection lives where loss is
    /// actually expensive — the SSD ledger's type-protected cut. The
    /// remaining RAM protection is the **Budget Floor** (in-flight
    /// restore pins + the freshest leaf), applied by the caller.
    /// `findEvictionCandidate`'s oldest-first fallback still backstops
    /// the degenerate case where utility scoring yields no victim.
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
        now: ContinuousClock.Instant = .now,
        config: EvictionConfiguration
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
            config: config,
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

    /// Collapse an `empty` node with exactly one child after a transition
    /// emptied it. Concatenates the node's edgeTokens into the child edge
    /// and re-links parent→child, preserving radix compression. Called
    /// only by `selfHeal`, so the `empty` guard (no body, no ref) holds by
    /// construction — collapsing here cannot orphan an SSD ref.
    private func collapseEmptySingleChild(_ node: RadixTreeNode) {
        guard node !== root,
            node.state.isEmpty,
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
        if node.state.body != nil {
            result.append(node)
        }
        for child in node.children.values {
            collectAllSnapshots(node: child, into: &result)
        }
    }

    private func collectEligible(node: RadixTreeNode, into result: inout [RadixTreeNode]) {
        if node.state.body != nil {
            result.append(node)
        }
        for child in node.children.values {
            collectEligible(node: child, into: &result)
        }
    }

    // Evolving MVP mid-refactor (see CLAUDE.md); structural limit kept lenient — splitting deferred.
    // swiftlint:disable:next function_parameter_count
    private func collectTelemetryNode(
        _ node: RadixTreeNode,
        partitionDigest: String,
        parentID: String?,
        path: [Int],
        depth: Int,
        now: ContinuousClock.Instant,
        config: EvictionConfiguration,
        nodes: inout [PromptCacheTreeNodeSnapshot],
        edges: inout [PromptCacheTreeEdgeSnapshot]
    ) {
        let pathHash = Self.telemetryPathHash(path)
        let nodeID = "\(partitionDigest):\(pathHash)"
        let storageState = Self.telemetryStorageState(node)
        let state = node.state
        let snapshot = state.body
        let checkpointType = state.checkpointType?.wireString
        // `.map` yields `EvictionScore??` (telemetryEvictionScore is itself
        // optional); the `?? nil` flattens the double optional and is
        // load-bearing — not redundant (SwiftLint's heuristic misreads it).
        // swiftlint:disable redundant_nil_coalescing
        let scores =
            snapshot.map { _ in
                telemetryEvictionScore(for: node, now: now, config: config)
            } ?? nil
        // swiftlint:enable redundant_nil_coalescing

        nodes.append(
            PromptCacheTreeNodeSnapshot(
                id: nodeID,
                parentID: parentID,
                pathHash: pathHash,
                tokenOffset: node.tokenOffset,
                pathTokenCount: path.count,
                edgeTokenCount: node.edgeTokens.count,
                childCount: node.childCount,
                depth: depth,
                hasSnapshot: state.hasResidentBody,
                checkpointType: checkpointType,
                snapshotBytes: state.residentBodyBytes,
                storageState: storageState,
                snapshotRefID: state.refID,
                storageBytes: state.storageBytes,
                lastAccessAgeSeconds: max((now - node.lastAccessTime).seconds, 0),
                normalizedRecency: scores?.normalizedRecency,
                normalizedFlopEfficiency: scores?.normalizedFlopEfficiency,
                utility: scores?.utility
            ))

        for child in node.children.values.sorted(by: Self.telemetryChildSort) {
            let childPath = path + child.edgeTokens
            let childHash = Self.telemetryPathHash(childPath)
            let childID = "\(partitionDigest):\(childHash)"
            edges.append(
                PromptCacheTreeEdgeSnapshot(
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
                config: config,
                nodes: &nodes,
                edges: &edges
            )
        }
    }

    private func telemetryEvictionScore(
        for node: RadixTreeNode,
        now: ContinuousClock.Instant,
        config: EvictionConfiguration
    ) -> EvictionScore? {
        // Mirrors `collectEligible`: every body scores (no type or
        // topology shielding, ADR-0019 / PRD #149).
        guard node.state.body != nil else { return nil }
        return EvictionPolicy.computeScores(
            candidates: [node], now: now, config: config
        ).first
    }

    private static func telemetryStorageState(_ node: RadixTreeNode) -> PromptCacheStorageState {
        switch node.state {
        case .empty: return .empty
        case .ramOnly: return .ramOnly
        case .pendingWrite: return .pendingWrite
        case .pendingDropped: return .pendingWriteBodyDropped
        case .committed: return .ramAndSSD
        case .ssdOnly: return .ssdOnly
        }
    }

    private static func telemetryChildSort(_ lhs: RadixTreeNode, _ rhs: RadixTreeNode) -> Bool {
        if lhs.tokenOffset != rhs.tokenOffset { return lhs.tokenOffset < rhs.tokenOffset }
        return lhs.edgeTokens.lexicographicallyPrecedes(rhs.edgeTokens)
    }

    private static func telemetryPathHash(_ tokens: [Int]) -> String {
        guard !tokens.isEmpty else { return "root" }
        var hash = TraceBlockDigest.fnvOffsetBasis
        for token in tokens {
            TraceBlockDigest.fold(token: token, into: &hash)
        }
        return TraceBlockDigest.hexDigest(hash)
    }
}

private extension CachePartitionKey {
    var telemetrySummary: String {
        let kv = kvBits.map { "kv\($0)" } ?? "denseKV"
        let fingerprint = modelFingerprint.map { String($0.prefix(8)) } ?? "nofp"
        return "\(modelID) · \(kv)/g\(kvGroupSize) · \(fingerprint)"
    }
}
