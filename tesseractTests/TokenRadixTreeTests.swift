import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// Task 1.4 tests: TokenRadixTree — compressed trie for prefix cache lookup.
@MainActor
struct TokenRadixTreeTests {

    // MARK: - Helpers

    private func makeSnapshot(
        offset: Int,
        type: HybridCacheSnapshot.CheckpointType = .system
    ) -> HybridCacheSnapshot {
        let kv = KVCacheSimple()
        kv.state = [
            MLXArray.zeros([1, 1, max(offset, 1), 64]), MLXArray.zeros([1, 1, max(offset, 1), 64]),
        ]
        return HybridCacheSnapshot.capture(cache: [kv], offset: offset, type: type)!
    }

    private func insertAndStore(
        _ tree: TokenRadixTree,
        tokens: [Int],
        type: HybridCacheSnapshot.CheckpointType = .system
    ) {
        let node = tree.insertPath(tokens: tokens)
        tree.storeSnapshot(makeSnapshot(offset: node.tokenOffset, type: type), on: node)
    }

    // MARK: - 1. emptyTreeReturnsNil

    @Test func emptyTreeReturnsNil() {
        let tree = TokenRadixTree()
        let result = tree.findBestSnapshot(tokens: [1, 2, 3])
        #expect(result == nil)
    }

    // MARK: - 2. insertAndExactMatch

    @Test func insertAndExactMatch() {
        let tree = TokenRadixTree()
        let tokens = Array(1...100)
        insertAndStore(tree, tokens: tokens)

        let result = tree.findBestSnapshot(tokens: tokens)
        #expect(result != nil)
        #expect(result?.node.tokenOffset == 100)
        #expect(result?.sharedPrefixLength == 100)
    }

    // MARK: - 3. snapshotAtPrefixReturnedOnDivergence

    @Test func snapshotAtPrefixReturnedOnDivergence() {
        let tree = TokenRadixTree()
        let prefix = Array(1...50)
        insertAndStore(tree, tokens: prefix)

        let query = prefix + Array(200...210)
        let result = tree.findBestSnapshot(tokens: query)
        #expect(result != nil)
        #expect(result?.node.tokenOffset == 50)
        #expect(result?.sharedPrefixLength == 50)
    }

    // MARK: - 4. deeperSnapshotPreferred

    @Test func deeperSnapshotPreferred() {
        let tree = TokenRadixTree()
        let tokens = Array(1...100)

        let node50 = tree.insertPath(tokens: Array(tokens[0..<50]))
        let node80 = tree.insertPath(tokens: Array(tokens[0..<80]))
        tree.insertPath(tokens: tokens)
        tree.storeSnapshot(makeSnapshot(offset: 50), on: node50)
        tree.storeSnapshot(makeSnapshot(offset: 80), on: node80)

        let result = tree.findBestSnapshot(tokens: tokens)
        #expect(result != nil)
        #expect(result?.node.tokenOffset == 80)
        #expect(result?.sharedPrefixLength == 80)
    }

    // MARK: - 5. snapshotBeyondSharedPrefixNotReturned

    @Test func snapshotBeyondSharedPrefixNotReturned() {
        let tree = TokenRadixTree()
        let tokens = Array(1...100)
        insertAndStore(tree, tokens: tokens)

        let query = Array(1...80) + [999, 998, 997]
        let result = tree.findBestSnapshot(tokens: query)
        #expect(result == nil)
    }

    // MARK: - 6. compressedEdges

    @Test func compressedEdges() {
        let tree = TokenRadixTree()
        tree.insertPath(tokens: [1, 2, 3, 4, 5])
        #expect(tree.nodeCount == 2)
    }

    // MARK: - 7. splitEdgeOnBranch

    @Test func splitEdgeOnBranch() {
        let tree = TokenRadixTree()
        tree.insertPath(tokens: [1, 2, 3, 4])
        #expect(tree.nodeCount == 2)

        tree.insertPath(tokens: [1, 2, 5, 6])
        #expect(tree.nodeCount == 4)

        #expect(tree.findBestSnapshot(tokens: [1, 2, 3, 4]) == nil)
        #expect(tree.findBestSnapshot(tokens: [1, 2, 5, 6]) == nil)

        // Store snapshot at split point — reachable from both paths
        let splitNode = tree.insertPath(tokens: [1, 2])
        tree.storeSnapshot(makeSnapshot(offset: 2), on: splitNode)
        #expect(tree.findBestSnapshot(tokens: [1, 2, 3, 4])?.node.tokenOffset == 2)
        #expect(tree.findBestSnapshot(tokens: [1, 2, 5, 6])?.node.tokenOffset == 2)
    }

    // MARK: - 8. dropBodyKeepsRefBearingNode

    /// Dropping the body of a node that still owns a committed ref leaves
    /// the node in the tree (state 4 → state 5): `canEvictNode` is false,
    /// so self-heal does not fire.
    @Test func dropBodyKeepsRefBearingNode() {
        let tree = TokenRadixTree()
        insertAndStore(tree, tokens: Array(1...10))

        let node = tree.findBestSnapshot(tokens: Array(1...10))!.node
        #expect(node.state.body != nil)

        let ref = PrefixCacheTestFixtures.makeRef(tokenOffset: node.tokenOffset)
        tree.admit(node: node, ref: ref)
        tree.commitRef(node: node, expectedID: ref.snapshotID)
        tree.dropBody(node: node)

        #expect(node.state.body == nil)
        #expect(node.state.ref != nil)  // ssdOnly — node retained
        #expect(tree.nodeCount == 2)
    }

    // MARK: - 9. evictLeafCleansAncestors

    @Test func evictLeafCleansAncestors() {
        let tree = TokenRadixTree()
        tree.insertPath(tokens: [1, 2, 3, 4])
        tree.insertPath(tokens: [1, 2, 5, 6])
        #expect(tree.nodeCount == 4)

        // Store on [5,6] leaf so we can find it, then evict
        insertAndStore(tree, tokens: [1, 2, 5, 6])
        let result = tree.findBestSnapshot(tokens: [1, 2, 5, 6])!
        // Dropping the ref-less leaf's body empties it; the tree self-heals
        // (detaches the leaf, then stops the walk at [1,2] which keeps its
        // [3,4] child).
        tree.dropBody(node: result.node)
        #expect(tree.nodeCount == 3)
    }

    // MARK: - 10. evictLeafPreservesSiblings

    @Test func evictLeafPreservesSiblings() {
        let tree = TokenRadixTree()
        insertAndStore(tree, tokens: [1, 2, 3, 4, 5])
        insertAndStore(tree, tokens: [1, 2, 6, 7])

        let toEvict = tree.findBestSnapshot(tokens: [1, 2, 3, 4, 5])!
        #expect(toEvict.node.tokenOffset == 5)
        tree.dropBody(node: toEvict.node)

        let sibling = tree.findBestSnapshot(tokens: [1, 2, 6, 7])
        #expect(sibling?.node.tokenOffset == 4)
    }

    // MARK: - 11. totalSnapshotBytesAccurate

    @Test func totalSnapshotBytesAccurate() {
        let tree = TokenRadixTree()
        #expect(tree.totalSnapshotBytes == 0)

        insertAndStore(tree, tokens: Array(1...10))
        let bytes1 = tree.totalSnapshotBytes
        #expect(bytes1 > 0)

        // Extend path, add second snapshot
        let node20 = tree.insertPath(tokens: Array(1...20))
        tree.storeSnapshot(makeSnapshot(offset: 20), on: node20)
        let bytes2 = tree.totalSnapshotBytes
        #expect(bytes2 > bytes1)

        let node1 = tree.findBestSnapshot(tokens: Array(1...10))!.node
        tree.dropBody(node: node1)
        #expect(tree.totalSnapshotBytes == bytes2 - bytes1)
    }

    // MARK: - 12. findBestSnapshotOnlyUpdatesReturnedNode

    @Test func findBestSnapshotOnlyUpdatesReturnedNode() {
        let tree = TokenRadixTree()
        let tokens = Array(1...100)
        let node50 = tree.insertPath(tokens: Array(tokens[0..<50]))
        let node80 = tree.insertPath(tokens: Array(tokens[0..<80]))
        tree.insertPath(tokens: tokens)
        tree.storeSnapshot(makeSnapshot(offset: 50), on: node50)
        tree.storeSnapshot(makeSnapshot(offset: 80), on: node80)

        // Lookup [1..50] → updates node 50
        _ = tree.findBestSnapshot(tokens: Array(1...50))
        let time50 = node50.lastAccessTime

        // Lookup [1..100] → returns node 80, should NOT update node 50
        let result = tree.findBestSnapshot(tokens: tokens)!
        #expect(result.node.tokenOffset == 80)
        #expect(result.node.lastAccessTime >= time50)
        #expect(node50.lastAccessTime == time50)
    }

    // MARK: - 13. 50KTokenSequence

    @Test func fiftyKTokenSequence() {
        let tree = TokenRadixTree()
        let tokens = Array(0..<50_000)
        insertAndStore(tree, tokens: tokens)

        let result = tree.findBestSnapshot(tokens: tokens)
        #expect(result?.node.tokenOffset == 50_000)
        #expect(result?.sharedPrefixLength == 50_000)
    }

    // MARK: - 14. eligibleEvictionNodesIncludeMultiChildNodes

    /// **No RAM type-shielding** (ADR-0019, PRD #149): multi-child
    /// nodes join the eligible set — one recovery-cost policy prices
    /// every body. (Pre-#149 they were excluded, which let shared
    /// branch bodies survive above a pressure-collapsed budget while
    /// the fresh leaf died.)
    @Test func eligibleEvictionNodesIncludeMultiChildNodes() {
        let tree = TokenRadixTree()

        tree.insertPath(tokens: [1, 2, 3, 4, 5])
        tree.insertPath(tokens: [1, 2, 6, 7])
        let midNode = tree.insertPath(tokens: [1, 2])
        tree.storeSnapshot(makeSnapshot(offset: 2), on: midNode)  // multi-child
        insertAndStore(tree, tokens: [1, 2, 3, 4, 5], type: .leaf)  // leaf, offset=5
        insertAndStore(tree, tokens: [1, 2, 6, 7], type: .leaf)  // leaf, offset=4

        let eligible = tree.eligibleEvictionNodes()
        let offsets = Set(eligible.map(\.tokenOffset))

        #expect(offsets.contains(5))
        #expect(offsets.contains(4))
        #expect(offsets.contains(2))  // multi-child node is eligible too
    }

    // MARK: - 14b. eligibleEvictionNodesIncludeSystemSnapshots

    /// `.system` bodies lose their RAM eviction immunity (ADR-0019,
    /// PRD #149): their protection moved to the SSD ledger's
    /// type-protected cut, where loss is actually expensive. A demoted
    /// system body costs a hydration on the next cold conversation; a
    /// shielded one used to cost the newest leaf its life under a
    /// pressure-collapsed budget.
    @Test func eligibleEvictionNodesIncludeSystemSnapshots() {
        let tree = TokenRadixTree()

        let sysNode = tree.insertPath(tokens: Array(1...10))
        tree.storeSnapshot(makeSnapshot(offset: 10, type: .system), on: sysNode)
        let leafNode = tree.insertPath(tokens: Array(1...15))
        tree.storeSnapshot(makeSnapshot(offset: 15, type: .leaf), on: leafNode)

        let eligible = tree.eligibleEvictionNodes()
        let offsets = Set(eligible.map(\.tokenOffset))

        #expect(offsets.contains(15))
        #expect(offsets.contains(10))  // system is eligible (uniform eviction)
    }

    // MARK: - 14c. eligibleEvictionNodesAllowsBranchPointAndLeaf

    /// `.branchPoint` snapshots are eligible. Phase 2 branch points
    /// represent speculative captures and have no special protection
    /// rule.
    @Test func eligibleEvictionNodesAllowsBranchPointAndLeaf() {
        let tree = TokenRadixTree()

        let branchNode = tree.insertPath(tokens: Array(1...8))
        tree.storeSnapshot(makeSnapshot(offset: 8, type: .branchPoint), on: branchNode)
        let leafNode = tree.insertPath(tokens: Array(1...12))
        tree.storeSnapshot(makeSnapshot(offset: 12, type: .leaf), on: leafNode)

        let eligible = tree.eligibleEvictionNodes()
        let offsets = Set(eligible.map(\.tokenOffset))

        #expect(offsets.contains(8))
        #expect(offsets.contains(12))
    }

    // MARK: - 15. selfHealCollapsesEmptiedSingleChildNode

    /// Emptying a single-child node via `dropBody` triggers the tree's
    /// self-heal, which collapses it (merging edges) to preserve radix
    /// compression.
    @Test func selfHealCollapsesEmptiedSingleChildNode() {
        let tree = TokenRadixTree()

        tree.insertPath(tokens: [1, 2, 3, 4, 5])
        tree.insertPath(tokens: [1, 2, 6, 7])
        let midNode = tree.insertPath(tokens: [1, 2])
        tree.storeSnapshot(makeSnapshot(offset: 2), on: midNode)
        let leafNode = tree.insertPath(tokens: [1, 2, 3, 4, 5])
        tree.storeSnapshot(makeSnapshot(offset: 5), on: leafNode)
        #expect(tree.nodeCount == 4)

        // Remove the [6,7] leaf by giving it a body and dropping it
        // (self-heal detaches the now-empty leaf).
        let leaf67 = midNode.children[6]!
        tree.storeSnapshot(makeSnapshot(offset: 4), on: leaf67)
        tree.dropBody(node: leaf67)
        #expect(tree.nodeCount == 3)

        // Drop midNode's body: now a single-child empty node → collapse.
        tree.dropBody(node: midNode)

        #expect(tree.nodeCount == 2)
        let final = tree.findBestSnapshot(tokens: [1, 2, 3, 4, 5])
        #expect(final?.node.tokenOffset == 5)
        #expect(final?.sharedPrefixLength == 5)
    }

    // MARK: - Additional edge cases

    @Test func insertSamePath() {
        let tree = TokenRadixTree()
        tree.insertPath(tokens: [1, 2, 3])
        tree.insertPath(tokens: [1, 2, 3])
        #expect(tree.nodeCount == 2)
    }

    @Test func insertPrefixOfExistingPath() {
        let tree = TokenRadixTree()
        tree.insertPath(tokens: [1, 2, 3, 4])
        let prefixNode = tree.insertPath(tokens: [1, 2])
        #expect(tree.nodeCount == 3)
        tree.storeSnapshot(makeSnapshot(offset: 2), on: prefixNode)
        let result = tree.findBestSnapshot(tokens: [1, 2, 3, 4])
        #expect(result?.node.tokenOffset == 2)
    }

    @Test func insertExtensionOfExistingPath() {
        let tree = TokenRadixTree()
        tree.insertPath(tokens: [1, 2])
        tree.insertPath(tokens: [1, 2, 3, 4])
        #expect(tree.nodeCount == 3)
    }

    /// Emptying a multi-child node retains it as a structural junction —
    /// self-heal removes only leaves and single-child nodes.
    @Test func selfHealRetainsMultiChildJunction() {
        let tree = TokenRadixTree()
        tree.insertPath(tokens: [1, 2, 3, 4])
        tree.insertPath(tokens: [1, 2, 5, 6])
        let mid = tree.insertPath(tokens: [1, 2])
        tree.storeSnapshot(makeSnapshot(offset: 2), on: mid)
        tree.dropBody(node: mid)  // empties a 2-child node → retained
        #expect(tree.nodeCount == 4)
        #expect(mid.parent != nil)
    }

    @Test func emptyTokensInsertIsNoOp() {
        let tree = TokenRadixTree()
        tree.insertPath(tokens: [])
        #expect(tree.nodeCount == 1)
    }

    @Test func storeSnapshotReplacesExisting() {
        let tree = TokenRadixTree()
        let node = tree.insertPath(tokens: [1, 2, 3])
        tree.storeSnapshot(makeSnapshot(offset: 3), on: node)
        let bytes1 = tree.totalSnapshotBytes

        tree.storeSnapshot(makeSnapshot(offset: 3), on: node)
        #expect(tree.totalSnapshotBytes == bytes1)
    }

    // MARK: - Mid-prefill checkpoint storage (forTokens:atOffset:)

    @Test func storeSnapshotAtIntermediateOffset() {
        let tree = TokenRadixTree()
        let fullTokens = Array(1...8000)

        // Single insertPath for the full prompt
        tree.insertPath(tokens: fullTokens)

        // Store mid-prefill checkpoint at offset 4000
        let ok = tree.storeSnapshot(
            makeSnapshot(offset: 4000), forTokens: fullTokens, atOffset: 4000
        )
        #expect(ok)

        // Lookup returns the intermediate snapshot, not the leaf
        let result = tree.findBestSnapshot(tokens: fullTokens)
        #expect(result?.node.tokenOffset == 4000)
        #expect(result?.sharedPrefixLength == 4000)
    }

    @Test func storeMultipleIntermediateCheckpoints() {
        let tree = TokenRadixTree()
        let tokens = Array(1...10000)
        tree.insertPath(tokens: tokens)

        // Store checkpoints at 2000, 5000, and 8000
        #expect(tree.storeSnapshot(makeSnapshot(offset: 2000), forTokens: tokens, atOffset: 2000))
        #expect(tree.storeSnapshot(makeSnapshot(offset: 5000), forTokens: tokens, atOffset: 5000))
        #expect(tree.storeSnapshot(makeSnapshot(offset: 8000), forTokens: tokens, atOffset: 8000))

        // Lookup full path returns deepest (8000)
        let full = tree.findBestSnapshot(tokens: tokens)
        #expect(full?.node.tokenOffset == 8000)

        // Lookup partial path returns deepest reachable
        let partial = tree.findBestSnapshot(tokens: Array(tokens[0..<6000]))
        #expect(partial?.node.tokenOffset == 5000)
    }

    @Test func storeAtOffsetBeyondTokensReturnsFalse() {
        let tree = TokenRadixTree()
        let tokens = Array(1...100)
        tree.insertPath(tokens: tokens)

        let ok = tree.storeSnapshot(makeSnapshot(offset: 200), forTokens: tokens, atOffset: 200)
        #expect(!ok)
    }

    @Test func storeAtOffsetZeroReturnsFalse() {
        let tree = TokenRadixTree()
        let tokens = Array(1...100)
        tree.insertPath(tokens: tokens)

        let ok = tree.storeSnapshot(makeSnapshot(offset: 0), forTokens: tokens, atOffset: 0)
        #expect(!ok)
    }

    @Test func mismatchedSnapshotOffsetRejected() {
        let tree = TokenRadixTree()
        let tokens = Array(1...8000)
        tree.insertPath(tokens: tokens)

        // Snapshot captured at 4000 but caller requests atOffset 5000 — must reject
        let ok = tree.storeSnapshot(makeSnapshot(offset: 4000), forTokens: tokens, atOffset: 5000)
        #expect(!ok)
        #expect(tree.totalSnapshotBytes == 0)
    }

    @Test func intermediateSnapshotCoexistsWithLeaf() {
        let tree = TokenRadixTree()
        let tokens = Array(1...100)
        let leaf = tree.insertPath(tokens: tokens)
        tree.storeSnapshot(makeSnapshot(offset: 100), on: leaf)

        // Add intermediate checkpoint
        tree.storeSnapshot(makeSnapshot(offset: 50), forTokens: tokens, atOffset: 50)

        // Full lookup returns leaf (deeper)
        let full = tree.findBestSnapshot(tokens: tokens)
        #expect(full?.node.tokenOffset == 100)

        // Divergent lookup returns intermediate
        let divergent = tree.findBestSnapshot(tokens: Array(1...50) + [999])
        #expect(divergent?.node.tokenOffset == 50)
    }

    // MARK: - Recency on store

    @Test func storeSnapshotRefreshesLastAccessTime() {
        let tree = TokenRadixTree()
        let node = tree.insertPath(tokens: [1, 2, 3])
        let timeAfterInsert = node.lastAccessTime

        // Store snapshot — should refresh the timestamp
        tree.storeSnapshot(makeSnapshot(offset: 3), on: node)
        #expect(node.lastAccessTime >= timeAfterInsert)

        // Replace snapshot — should refresh again
        let timeAfterFirstStore = node.lastAccessTime
        tree.storeSnapshot(makeSnapshot(offset: 3), on: node)
        #expect(node.lastAccessTime >= timeAfterFirstStore)
    }

    @Test func forTokensStoreRefreshesLastAccessTime() {
        let tree = TokenRadixTree()
        let tokens = Array(1...100)
        tree.insertPath(tokens: tokens)

        let timeBefore = ContinuousClock.Instant.now
        tree.storeSnapshot(makeSnapshot(offset: 50), forTokens: tokens, atOffset: 50)

        let node = tree.findBestSnapshot(tokens: Array(1...50))!.node
        #expect(node.lastAccessTime >= timeBefore)
    }

    // MARK: - Speculative branch-point detection

    @Test func findSplitInsideCompressedEdge() {
        let tree = TokenRadixTree()
        tree.insertPath(tokens: [1, 2, 3, 4])
        #expect(tree.findIntermediateSplitOffsetForInsertion(tokens: [1, 2, 5, 6]) == 2)
    }

    @Test func findSplitNodeBoundaryDivergenceReturnsNil() {
        let tree = TokenRadixTree()
        // Materialize an intermediate node at offset 2 by inserting both
        // [1,2,3,4] and [1,2] — the second insert splits the edge.
        tree.insertPath(tokens: [1, 2, 3, 4])
        tree.insertPath(tokens: [1, 2])
        #expect(tree.findIntermediateSplitOffsetForInsertion(tokens: [1, 2, 9]) == nil)
    }

    @Test func findSplitExactExtensionReturnsNil() {
        let tree = TokenRadixTree()
        tree.insertPath(tokens: [1, 2, 3])
        #expect(tree.findIntermediateSplitOffsetForInsertion(tokens: [1, 2, 3, 4, 5]) == nil)
    }

    @Test func findSplitEmptyTreeReturnsNil() {
        let tree = TokenRadixTree()
        #expect(tree.findIntermediateSplitOffsetForInsertion(tokens: [1, 2, 3]) == nil)
    }

    @Test func findSplitEmptyTokensReturnsNil() {
        let tree = TokenRadixTree()
        tree.insertPath(tokens: [1, 2, 3])
        #expect(tree.findIntermediateSplitOffsetForInsertion(tokens: []) == nil)
    }

    @Test func findSplitExactMatchReturnsNil() {
        let tree = TokenRadixTree()
        tree.insertPath(tokens: [1, 2, 3])
        #expect(tree.findIntermediateSplitOffsetForInsertion(tokens: [1, 2, 3]) == nil)
    }

    @Test func findSplitShorterPrefixReturnsSplitOffset() {
        let tree = TokenRadixTree()
        tree.insertPath(tokens: [1, 2, 3, 4, 5])
        #expect(tree.findIntermediateSplitOffsetForInsertion(tokens: [1, 2, 3]) == 3)
    }

    // MARK: - Self-heal respects live refs (the orphan invariant)

    /// A single-child node that still owns a ref is not collapsed: dropping
    /// its body settles it to a ref-bearing state (`canEvictNode == false`),
    /// so self-heal never fires. The old explicit Snapshot Ref guard
    /// is gone — the invariant is now structural (the ref-bearing state is
    /// not `empty`, so `becameEmpty` is never produced).
    @Test func selfHealKeepsSingleChildNodeWithCommittedRef() {
        let tree = TokenRadixTree()
        tree.insertPath(tokens: [1, 2, 3, 4, 5])
        let midNode = tree.insertPath(tokens: [1, 2])
        tree.storeSnapshot(makeSnapshot(offset: 2), on: midNode)
        let ref = PrefixCacheTestFixtures.makeRef(tokenOffset: 2)
        tree.admit(node: midNode, ref: ref)
        tree.commitRef(node: midNode, expectedID: ref.snapshotID)
        let preNodeCount = tree.nodeCount
        let preEdge = midNode.edgeTokens

        tree.dropBody(node: midNode)  // committed → ssdOnly (settled)

        #expect(tree.nodeCount == preNodeCount)
        #expect(midNode.parent != nil)
        #expect(midNode.state.ref != nil)
        #expect(midNode.state.committed)
        #expect(midNode.edgeTokens == preEdge)
        #expect(midNode.childCount == 1)
    }

    @Test func selfHealKeepsSingleChildNodeWithPendingRef() {
        let tree = TokenRadixTree()
        tree.insertPath(tokens: [1, 2, 3, 4, 5])
        let midNode = tree.insertPath(tokens: [1, 2])
        tree.storeSnapshot(makeSnapshot(offset: 2), on: midNode)
        tree.admit(node: midNode, ref: PrefixCacheTestFixtures.makeRef(tokenOffset: 2))
        let preNodeCount = tree.nodeCount

        tree.dropBody(node: midNode)  // pendingWrite → pendingDropped (settled)

        #expect(tree.nodeCount == preNodeCount)
        #expect(midNode.parent != nil)
        #expect(midNode.state.ref != nil)
        #expect(!midNode.state.committed)
    }

    @Test func staleCommitAfterReadmissionReturnsIDMismatch() {
        let tree = TokenRadixTree()
        insertAndStore(tree, tokens: [1, 2, 3])
        let node = tree.findBestSnapshot(tokens: [1, 2, 3])!.node

        let oldRef = PrefixCacheTestFixtures.makeRef(tokenOffset: node.tokenOffset)
        let newRef = PrefixCacheTestFixtures.makeRef(tokenOffset: node.tokenOffset)
        tree.admit(node: node, ref: oldRef)
        tree.admit(node: node, ref: newRef)

        let staleEffect = tree.commitRef(node: node, expectedID: oldRef.snapshotID)

        #expect(staleEffect == .ignored(.idMismatch))
        #expect(node.state.refID == newRef.snapshotID)
        #expect(!node.state.committed)
    }

    /// Self-heal's ancestor walk stops at a ref-bearing ancestor: removing a
    /// leaf cannot sweep up a parent that still owns a ref, so the SSD file
    /// is never orphaned. The walk only continues through `empty` ancestors.
    @Test func selfHealAncestorWalkStopsAtCommittedRef() {
        let tree = TokenRadixTree()
        tree.insertPath(tokens: [1, 2, 3, 4])
        let midNode = tree.insertPath(tokens: [1, 2, 3])
        // Put midNode in state 5 (committed ref, no body) so it has a live
        // ref but is body-less, the exact orphan hazard.
        tree.restoreCommittedRef(
            node: midNode, ref: PrefixCacheTestFixtures.makeRef(tokenOffset: 3))
        #expect(tree.nodeCount == 3)

        // Empty the [4] leaf and let self-heal try to walk upward.
        let leaf = midNode.children[4]!
        tree.storeSnapshot(makeSnapshot(offset: 4), on: leaf)
        tree.dropBody(node: leaf)

        #expect(tree.nodeCount == 2)  // leaf removed, midNode retained
        #expect(midNode.parent != nil)
        #expect(midNode.state.ref != nil)
        #expect(midNode.state.committed)
        #expect(midNode.isLeaf)
    }

    // MARK: - Prompt-cache telemetry topology

    @Test func topologySnapshotCapturesCompressedEdgesAndPathHashes() {
        let tree = TokenRadixTree()
        let node = tree.insertPath(tokens: [1, 2, 3, 4])
        tree.storeSnapshot(makeSnapshot(offset: 4, type: .leaf), on: node)
        let partition = CachePartitionKey(modelID: "telemetry-model", kvBits: 8, kvGroupSize: 64)

        let snapshot = tree.makeTopologySnapshot(
            partition: partition, config: EvictionConfiguration()
        )
        let root = snapshot.nodes.first { $0.parentID == nil }
        let leaf = snapshot.nodes.first { $0.tokenOffset == 4 }
        let edge = snapshot.edges.first

        #expect(snapshot.id == partition.partitionDigest)
        #expect(snapshot.nodeCount == 2)
        #expect(snapshot.nodes.count == 2)
        #expect(snapshot.edges.count == 1)
        #expect(snapshot.snapshotCount == 1)
        #expect(snapshot.snapshotsByType["leaf"] == 1)
        #expect(root?.pathHash == "root")
        #expect(leaf?.pathHash != "root")
        #expect(leaf?.id.hasPrefix("\(partition.partitionDigest):") == true)
        #expect(leaf?.pathTokenCount == 4)
        #expect(leaf?.edgeTokenCount == 4)
        #expect(leaf?.checkpointType == "leaf")
        #expect(leaf?.storageState == .ramOnly)
        #expect(leaf?.hasSnapshot == true)
        #expect(edge?.parentID == root?.id)
        #expect(edge?.childID == leaf?.id)
        #expect(edge?.tokenCount == 4)
    }

    @Test func topologySnapshotCapturesBranchPointsAndStorageStates() {
        let tree = TokenRadixTree()
        tree.insertPath(tokens: [1, 2, 3, 4])
        let pendingLeaf = tree.insertPath(tokens: [1, 2, 9, 10])
        let branch = tree.insertPath(tokens: [1, 2])
        // branch → state 4 (body + committed ref); reported as `ramAndSSD`.
        tree.storeSnapshot(makeSnapshot(offset: 2, type: .branchPoint), on: branch)
        let branchRef = PrefixCacheTestFixtures.makeRef(type: .branchPoint, tokenOffset: 2)
        tree.admit(node: branch, ref: branchRef)
        tree.commitRef(node: branch, expectedID: branchRef.snapshotID)
        // pendingLeaf → state 3 (pending ref, body dropped); reported as
        // `pendingWriteBodyDropped`.
        let pendingRef = PrefixCacheTestFixtures.makeRef(type: .leaf, tokenOffset: 4)
        tree.storeSnapshot(makeSnapshot(offset: 4, type: .leaf), on: pendingLeaf)
        tree.admit(node: pendingLeaf, ref: pendingRef)
        tree.dropBody(node: pendingLeaf)

        let snapshot = tree.makeTopologySnapshot(
            partition: CachePartitionKey(
                modelID: "telemetry-model",
                kvBits: nil,
                kvGroupSize: 64,
                modelFingerprint: "abcdef123456"
            ),
            config: EvictionConfiguration()
        )
        let branchNode = snapshot.nodes.first { $0.tokenOffset == 2 }
        let pendingNode = snapshot.nodes.first { $0.snapshotRefID == pendingRef.snapshotID }

        #expect(snapshot.partitionSummary.contains("telemetry-model"))
        #expect(snapshot.partitionSummary.contains("denseKV"))
        #expect(snapshot.partitionSummary.contains("abcdef12"))
        #expect(branchNode?.checkpointType == "branchPoint")
        #expect(branchNode?.storageState == .ramAndSSD)
        #expect(branchNode?.storageBytes == 1024)
        #expect(branchNode?.snapshotRefID == branchRef.snapshotID)
        #expect(pendingNode?.checkpointType == "leaf")
        #expect(pendingNode?.storageState == .pendingWriteBodyDropped)
        #expect(pendingNode?.hasSnapshot == false)
    }

    @Test func topologySnapshotTraversesLargeBranchyTree() {
        let tree = TokenRadixTree()
        for branch in 0..<160 {
            let node = tree.insertPath(tokens: [1, branch, branch + 1_000, branch + 2_000])
            if branch.isMultiple(of: 10) {
                tree.storeSnapshot(makeSnapshot(offset: 4, type: .leaf), on: node)
            }
        }

        let snapshot = tree.makeTopologySnapshot(
            partition: CachePartitionKey(modelID: "large-tree", kvBits: 4, kvGroupSize: 32),
            config: EvictionConfiguration()
        )
        let nodeIDs = Set(snapshot.nodes.map(\.id))
        let edgeIDs = Set(snapshot.edges.map(\.id))

        #expect(snapshot.nodes.count == tree.nodeCount)
        #expect(snapshot.edges.count == snapshot.nodes.count - 1)
        #expect(nodeIDs.count == snapshot.nodes.count)
        #expect(edgeIDs.count == snapshot.edges.count)
        #expect(snapshot.snapshotCount == 16)
        #expect(snapshot.snapshotsByType["leaf"] == 16)
    }
}
