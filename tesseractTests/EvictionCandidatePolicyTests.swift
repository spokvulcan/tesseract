//
//  EvictionCandidatePolicyTests.swift
//  tesseractTests
//
//  The **Eviction Candidate Policy** at its own seam (ADR-0049): decision
//  tables over the RAM tier's four-strategy ladder (writing-partition-first,
//  global spill, the Budget Floor) and the SSD tier's terminal-loss
//  ordering. Before the cut these decisions were private to their managers,
//  reachable only by replaying whole-manager admission scenarios; here each
//  test hands the policy a fixed candidate set and asserts which branch
//  fired and which victim it named.
//
//  The ladder's two fallback arms are a residual safety net: with uniform
//  eligibility (ADR-0019) the eligible set equals the snapshot set, so no
//  tree state can reach them today. They stay documented in the ladder,
//  not asserted here.
//

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

@MainActor
struct EvictionCandidatePolicyTests {

    // MARK: - RAM-tier fixtures

    private func makeKey(_ modelID: String) -> CachePartitionKey {
        CachePartitionKey(modelID: modelID, kvBits: nil, kvGroupSize: 64)
    }

    private func makeSnapshot(offset: Int) -> HybridCacheSnapshot {
        let kv = KVCacheSimple()
        kv.state = [
            MLXArray.zeros([1, 1, max(offset, 1), 64]),
            MLXArray.zeros([1, 1, max(offset, 1), 64]),
        ]
        return HybridCacheSnapshot.capture(cache: [kv], offset: offset, type: .system)!
    }

    /// Insert a snapshot-bearing node at `tokens.count` and back-date its
    /// recency. Returns the node so tests can name expected victims and
    /// build the protected set.
    @discardableResult
    private func addNode(
        _ tree: TokenRadixTree, tokens: [Int], accessAge: Duration = .zero
    ) -> RadixTreeNode {
        let node = tree.insertPath(tokens: tokens)
        tree.storeSnapshot(makeSnapshot(offset: node.tokenOffset), on: node)
        if accessAge != .zero {
            node.lastAccessTime = .now - accessAge
        }
        return node
    }

    // MARK: - RAM tier: the ladder

    /// Strategy 1 — writing-partition-first: the preferred partition's own
    /// eligible node is the victim even when another partition holds a
    /// colder (lower-utility) one.
    @Test func preferredPartitionIsDrainedBeforeColderNeighbors() {
        let preferredTree = TokenRadixTree()
        let otherTree = TokenRadixTree()
        let fresh = addNode(preferredTree, tokens: [1, 2], accessAge: .seconds(5))
        addNode(otherTree, tokens: [9, 8], accessAge: .seconds(500))

        let candidate = EvictionCandidatePolicy.candidate(
            now: .now,
            orderedPartitions: [
                (key: makeKey("a"), tree: preferredTree),
                (key: makeKey("b"), tree: otherTree),
            ],
            preferred: (key: makeKey("a"), tree: preferredTree),
            config: EvictionConfiguration()
        )

        #expect(candidate?.node === fresh)
        #expect(candidate?.strategy == .utility)
        #expect(candidate?.score != nil)
        #expect(candidate?.partitionKey == makeKey("a"))
    }

    /// Strategy 2 — spill-over: a drained preferred partition sends the
    /// drain to the other partitions' eligible nodes.
    @Test func drainedPreferredPartitionSpillsToGlobalUtility() {
        let preferredTree = TokenRadixTree()  // no snapshots
        let otherTree = TokenRadixTree()
        let victim = addNode(otherTree, tokens: [9, 8], accessAge: .seconds(60))

        let candidate = EvictionCandidatePolicy.candidate(
            now: .now,
            orderedPartitions: [
                (key: makeKey("a"), tree: preferredTree),
                (key: makeKey("b"), tree: otherTree),
            ],
            preferred: (key: makeKey("a"), tree: preferredTree),
            config: EvictionConfiguration()
        )

        #expect(candidate?.node === victim)
        #expect(candidate?.strategy == .utility)
        #expect(candidate?.partitionKey == makeKey("b"))
    }

    /// No preferred partition — Marconi's plain global utility: with the
    /// LRU default (`alpha = 0`) the oldest node across all partitions is
    /// the victim.
    @Test func globalUtilityNamesTheOldestNodeAcrossPartitions() {
        let treeA = TokenRadixTree()
        let treeB = TokenRadixTree()
        addNode(treeA, tokens: [1, 2], accessAge: .seconds(10))
        let oldest = addNode(treeB, tokens: [9, 8], accessAge: .seconds(300))
        addNode(treeB, tokens: [9, 7], accessAge: .seconds(30))

        let candidate = EvictionCandidatePolicy.candidate(
            now: .now,
            orderedPartitions: [
                (key: makeKey("a"), tree: treeA),
                (key: makeKey("b"), tree: treeB),
            ],
            config: EvictionConfiguration()
        )

        #expect(candidate?.node === oldest)
        #expect(candidate?.strategy == .utility)
    }

    /// The **Budget Floor**: a protected node is never the victim — the
    /// drain passes over it to the next-worst — and when the floor covers
    /// every node the policy names no victim at all rather than betray it.
    @Test func budgetFloorMembersAreNeverVictims() {
        let tree = TokenRadixTree()
        let floorMember = addNode(tree, tokens: [1, 2], accessAge: .seconds(900))
        let victim = addNode(tree, tokens: [1, 3], accessAge: .seconds(5))
        let partitions = [(key: makeKey("a"), tree: tree)]

        let candidate = EvictionCandidatePolicy.candidate(
            now: .now,
            orderedPartitions: partitions,
            preferred: partitions[0],
            protected: [ObjectIdentifier(floorMember)],
            config: EvictionConfiguration()
        )
        #expect(candidate?.node === victim)

        let none = EvictionCandidatePolicy.candidate(
            now: .now,
            orderedPartitions: partitions,
            preferred: partitions[0],
            protected: [ObjectIdentifier(floorMember), ObjectIdentifier(victim)],
            config: EvictionConfiguration()
        )
        #expect(none == nil)
    }

    // MARK: - SSD-tier fixtures

    private func makeDescriptor(
        id: String, tokenOffset: Int = 100, bytes: Int = 4096, lastAccessAt: Double
    ) -> PersistedSnapshotDescriptor {
        PersistedSnapshotDescriptor(
            snapshotID: id,
            partitionDigest: "",
            pathFromRoot: [],
            tokenOffset: tokenOffset,
            checkpointType: "system",
            bytes: bytes,
            createdAt: lastAccessAt,
            lastAccessAt: lastAccessAt,
            fileRelativePath: "",
            schemaVersion: SnapshotManifestSchema.currentVersion
        )
    }

    // MARK: - SSD tier: terminal-loss ordering

    /// The `alpha = 0` fast path is plain LRU — oldest first, ascending
    /// `snapshotID` breaking recency ties.
    @Test func alphaZeroOrdersPlainLRUWithStableTiebreak() {
        let order = EvictionCandidatePolicy.terminalLossOrder(
            [
                makeDescriptor(id: "c", lastAccessAt: 30),
                makeDescriptor(id: "b", lastAccessAt: 10),
                makeDescriptor(id: "a", lastAccessAt: 10),
            ],
            config: EvictionConfiguration(),
            now: 100
        )
        #expect(order.map(\.snapshotID) == ["a", "b", "c"])
    }

    /// With `alpha` engaged, recovery cost reorders recency ties: the
    /// chain that is cheap to re-prefill per byte is the worse victim
    /// (lower utility, earlier in the order) than the dense expensive one
    /// — even though the plain LRU tiebreak would order them the other
    /// way around.
    @Test func alphaBlendsRecoveryCostIntoTheOrder() {
        // Equal recency; "a" < "z" so LRU alone would put `expensive` first.
        let expensive = makeDescriptor(
            id: "a", tokenOffset: 4000, bytes: 1024, lastAccessAt: 50)
        let cheapPerByte = makeDescriptor(
            id: "z", tokenOffset: 400, bytes: 1024 * 1024, lastAccessAt: 50)

        let lru = EvictionCandidatePolicy.terminalLossOrder(
            [expensive, cheapPerByte], config: EvictionConfiguration(), now: 100)
        #expect(lru.map(\.snapshotID) == ["a", "z"])

        let blended = EvictionCandidatePolicy.terminalLossOrder(
            [expensive, cheapPerByte],
            config: EvictionConfiguration(alpha: 1.0),
            now: 100
        )
        #expect(blended.map(\.snapshotID) == ["z", "a"])
    }

    /// A single candidate short-circuits (nothing to normalize against),
    /// and the ordering is a pure derivation: the same inputs and the same
    /// injected `now` give the same order, call after call.
    @Test func orderingIsPureOverItsInputs() {
        let single = EvictionCandidatePolicy.terminalLossOrder(
            [makeDescriptor(id: "only", lastAccessAt: 5)],
            config: EvictionConfiguration(alpha: 1.0),
            now: 100
        )
        #expect(single.map(\.snapshotID) == ["only"])

        let pool = [
            makeDescriptor(id: "a", tokenOffset: 2000, bytes: 512, lastAccessAt: 40),
            makeDescriptor(id: "b", tokenOffset: 100, bytes: 65536, lastAccessAt: 60),
            makeDescriptor(id: "c", tokenOffset: 900, bytes: 4096, lastAccessAt: 20),
        ]
        let config = EvictionConfiguration(alpha: 0.7)
        let first = EvictionCandidatePolicy.terminalLossOrder(pool, config: config, now: 100)
        let second = EvictionCandidatePolicy.terminalLossOrder(pool, config: config, now: 100)
        #expect(first.map(\.snapshotID) == second.map(\.snapshotID))
    }
}
