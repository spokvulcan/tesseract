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
        kv.state = [MLXArray.zeros([1, 1, max(offset, 1), 64]), MLXArray.zeros([1, 1, max(offset, 1), 64])]
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

    // MARK: - 8. evictSnapshotKeepsNode

    @Test func evictSnapshotKeepsNode() {
        let tree = TokenRadixTree()
        insertAndStore(tree, tokens: Array(1...10))

        let node = tree.findBestSnapshot(tokens: Array(1...10))!.node
        #expect(node.snapshot != nil)

        tree.evictSnapshot(node: node)
        #expect(node.snapshot == nil)
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
        tree.evictSnapshot(node: result.node)
        tree.evictNode(node: result.node)
        // [5,6] removed. Intermediate [1,2] still has child [3,4].
        #expect(tree.nodeCount == 3)
    }

    // MARK: - 10. evictLeafPreservesSiblings

    @Test func evictLeafPreservesSiblings() {
        let tree = TokenRadixTree()
        insertAndStore(tree, tokens: [1, 2, 3, 4, 5])
        insertAndStore(tree, tokens: [1, 2, 6, 7])

        let toEvict = tree.findBestSnapshot(tokens: [1, 2, 3, 4, 5])!
        #expect(toEvict.node.tokenOffset == 5)
        tree.evictSnapshot(node: toEvict.node)
        tree.evictNode(node: toEvict.node)

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
        tree.evictSnapshot(node: node1)
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

    // MARK: - 14. eligibleEvictionNodesExcludeMultiChildNodes

    @Test func eligibleEvictionNodesExcludeMultiChildNodes() {
        let tree = TokenRadixTree()

        // Distinct lengths to avoid offset ambiguity. Use `.leaf` for the
        // leaf snapshots so they're not also excluded by the `.system`
        // type-protection guard — this test is about multi-child
        // protection, exercised in isolation.
        tree.insertPath(tokens: [1, 2, 3, 4, 5])
        tree.insertPath(tokens: [1, 2, 6, 7])
        let midNode = tree.insertPath(tokens: [1, 2])
        tree.storeSnapshot(makeSnapshot(offset: 2), on: midNode) // multi-child
        insertAndStore(tree, tokens: [1, 2, 3, 4, 5], type: .leaf) // leaf, offset=5
        insertAndStore(tree, tokens: [1, 2, 6, 7], type: .leaf)    // leaf, offset=4

        let eligible = tree.eligibleEvictionNodes()
        let offsets = Set(eligible.map(\.tokenOffset))

        #expect(offsets.contains(5))
        #expect(offsets.contains(4))
        #expect(!offsets.contains(2)) // multi-child node protected
    }

    // MARK: - 14b. eligibleEvictionNodesExcludeSystemSnapshots

    /// Regression test for the new-user-turn cold-prefill pathology.
    /// `.system` snapshots (stable prefix + last-message boundary) are
    /// excluded from the utility-eviction candidate set even when their
    /// node has `childCount <= 1`. The hard budget invariant is preserved
    /// by `PrefixCacheManager.findEvictionCandidate`'s fallback path
    /// (covered separately by `branchNodeFallbackHonorsHardBudget`).
    @Test func eligibleEvictionNodesExcludeSystemSnapshots() {
        let tree = TokenRadixTree()

        // Linear chain: root → [1..10] (system) → [11..15] (leaf)
        // Both nodes have childCount <= 1; only the .leaf is eligible.
        let sysNode = tree.insertPath(tokens: Array(1...10))
        tree.storeSnapshot(makeSnapshot(offset: 10, type: .system), on: sysNode)
        let leafNode = tree.insertPath(tokens: Array(1...15))
        tree.storeSnapshot(makeSnapshot(offset: 15, type: .leaf), on: leafNode)

        let eligible = tree.eligibleEvictionNodes()
        let offsets = Set(eligible.map(\.tokenOffset))

        #expect(offsets.contains(15))   // leaf is eligible
        #expect(!offsets.contains(10))  // system is protected
    }

    // MARK: - 14c. eligibleEvictionNodesAllowsBranchPointAndLeaf

    /// `.branchPoint` snapshots are still eligible — only `.system` is
    /// type-protected. Phase 2 branch points represent speculative
    /// captures and have no special protection rule.
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

    // MARK: - 15. collapseSingleChildNodeMergesEdges

    @Test func collapseSingleChildNodeMergesEdges() {
        let tree = TokenRadixTree()

        tree.insertPath(tokens: [1, 2, 3, 4, 5])
        tree.insertPath(tokens: [1, 2, 6, 7])
        let midNode = tree.insertPath(tokens: [1, 2])
        tree.storeSnapshot(makeSnapshot(offset: 2), on: midNode)
        let leafNode = tree.insertPath(tokens: [1, 2, 3, 4, 5])
        tree.storeSnapshot(makeSnapshot(offset: 5), on: leafNode)
        #expect(tree.nodeCount == 4)

        // Remove [6,7] leaf
        if let leaf67 = midNode.children[6] {
            tree.evictNode(node: leaf67)
        }
        #expect(tree.nodeCount == 3)

        // Evict snapshot from intermediate, then collapse
        tree.evictSnapshot(node: midNode)
        tree.collapseSingleChildNode(midNode)

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

    @Test func evictNodeOnlyRemovesLeaves() {
        let tree = TokenRadixTree()
        tree.insertPath(tokens: [1, 2, 3, 4])
        tree.insertPath(tokens: [1, 2, 5, 6])
        let mid = tree.insertPath(tokens: [1, 2])
        tree.storeSnapshot(makeSnapshot(offset: 2), on: mid)
        tree.evictNode(node: mid) // no-op: not a leaf
        #expect(tree.nodeCount == 4)
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
}
