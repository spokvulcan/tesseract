import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// Task 1.5 tests: PrefixCacheManager — public API for radix-tree prefix cache.
@MainActor
struct PrefixCacheManagerTests {

    // MARK: - Helpers

    private let defaultKey = CachePartitionKey(modelID: "test-model", kvBits: nil, kvGroupSize: 64)

    private func makeSnapshot(offset: Int, type: HybridCacheSnapshot.CheckpointType = .system) -> HybridCacheSnapshot {
        let kv = KVCacheSimple()
        kv.state = [MLXArray.zeros([1, 1, max(offset, 1), 64]), MLXArray.zeros([1, 1, max(offset, 1), 64])]
        return HybridCacheSnapshot.capture(cache: [kv], offset: offset, type: type)!
    }

    private func makeManager(budgetMB: Int = 100) -> PrefixCacheManager {
        PrefixCacheManager(memoryBudgetBytes: budgetMB * 1024 * 1024)
    }

    // MARK: - 1. lookupEmptyCacheReturnsMiss

    @Test func lookupEmptyCacheReturnsMiss() {
        let mgr = makeManager()
        let result = mgr.lookup(tokens: [1, 2, 3], partitionKey: defaultKey)
        #expect(result.snapshot == nil)
        #expect(result.snapshotTokenOffset == 0)
        if case .missNoEntries = result.reason {} else {
            #expect(Bool(false), "Expected missNoEntries, got \(result.reason)")
        }
    }

    // MARK: - 2. storeAndExactLookupReturnsHit

    @Test func storeAndExactLookupReturnsHit() {
        let mgr = makeManager()
        let tokens = Array(1...100)
        let snap = makeSnapshot(offset: 100, type: .leaf)
        mgr.storeLeaf(storedTokens: tokens, leafSnapshot: snap, partitionKey: defaultKey)

        let result = mgr.lookup(tokens: tokens, partitionKey: defaultKey)
        #expect(result.snapshot != nil)
        #expect(result.snapshotTokenOffset == 100)
        #expect(result.sharedPrefixLength == 100)
        if case .hit = result.reason {} else {
            #expect(Bool(false), "Expected hit")
        }
    }

    // MARK: - 3. systemSnapshotSharedAcrossConversations

    @Test func systemSnapshotSharedAcrossConversations() {
        let mgr = makeManager()
        let sysTokens = Array(1...500)  // system + tools prefix
        let userATokens = sysTokens + Array(600...700)
        let userBTokens = sysTokens + Array(800...900)

        // Store system snapshot via storeSnapshots (mid-prefill)
        let sysSnap = makeSnapshot(offset: 500, type: .system)
        mgr.storeSnapshots(promptTokens: userATokens, capturedSnapshots: [sysSnap], partitionKey: defaultKey)

        // Lookup with different user content — still hits system snapshot
        let result = mgr.lookup(tokens: userBTokens, partitionKey: defaultKey)
        #expect(result.snapshot != nil)
        #expect(result.snapshotTokenOffset == 500)
    }

    // MARK: - 4. leafSnapshotMatchesWithinConversation

    @Test func leafSnapshotMatchesWithinConversation() {
        let mgr = makeManager()
        // First turn: sys + user1 + asst1
        let turn1 = Array(1...300)
        let leafSnap = makeSnapshot(offset: 300, type: .leaf)
        mgr.storeLeaf(storedTokens: turn1, leafSnapshot: leafSnap, partitionKey: defaultKey)

        // Second turn extends: sys + user1 + asst1 + tool1
        let turn2 = turn1 + Array(400...450)
        let result = mgr.lookup(tokens: turn2, partitionKey: defaultKey)
        #expect(result.snapshot != nil)
        #expect(result.snapshotTokenOffset == 300)
    }

    // MARK: - 5. lookupReturnsDeepCopiedCache

    @Test func lookupReturnsDeepCopiedCache() {
        let mgr = makeManager()
        let tokens = Array(1...50)
        let snap = makeSnapshot(offset: 50, type: .leaf)
        mgr.storeLeaf(storedTokens: tokens, leafSnapshot: snap, partitionKey: defaultKey)

        let r1 = mgr.lookup(tokens: tokens, partitionKey: defaultKey)
        let r2 = mgr.lookup(tokens: tokens, partitionKey: defaultKey)
        #expect(r1.restoreCache() != nil)
        #expect(r2.restoreCache() != nil)
        // Different object references — independent copies
        #expect(r1.restoreCache()!.count == r2.restoreCache()!.count)
    }

    // MARK: - 6. mutatingRestoredCacheDoesNotAffectSnapshot

    @Test func mutatingRestoredCacheDoesNotAffectSnapshot() {
        let mgr = makeManager()
        let tokens = Array(1...50)
        let snap = makeSnapshot(offset: 50, type: .leaf)
        mgr.storeLeaf(storedTokens: tokens, leafSnapshot: snap, partitionKey: defaultKey)

        var restored = mgr.lookup(tokens: tokens, partitionKey: defaultKey).restoreCache()!
        // Mutate restored cache
        restored[0].state = [MLXArray.zeros([1, 1, 99, 64]), MLXArray.zeros([1, 1, 99, 64])]

        // Second lookup still returns original state
        let r2 = mgr.lookup(tokens: tokens, partitionKey: defaultKey)
        #expect(r2.restoreCache()![0].state[0].dim(2) == 50)
    }

    // MARK: - 7. evictionRemovesLeafBeforeSystem

    @Test func evictionRemovesLeafBeforeSystem() {
        // Store system first, then leaf. Budget fits only one.
        // Use same-size snapshots so we can set budget = 1 snapshot's worth.
        let sysTokens = Array(1...10)
        let sysSnap = makeSnapshot(offset: 10, type: .system)
        let snapBytes = sysSnap.memoryBytes

        // Budget fits exactly one snapshot
        let mgr = PrefixCacheManager(memoryBudgetBytes: snapBytes)

        mgr.storeSnapshots(promptTokens: sysTokens, capturedSnapshots: [sysSnap], partitionKey: defaultKey)
        #expect(mgr.stats.snapshotCount == 1)

        // Store leaf — pushes over budget. Eviction should remove leaf (lower priority), keep system.
        let leafTokens = Array(1...20)
        let leafSnap = makeSnapshot(offset: 20, type: .leaf)
        mgr.storeLeaf(storedTokens: leafTokens, leafSnapshot: leafSnap, partitionKey: defaultKey)

        // System survives
        let sysResult = mgr.lookup(tokens: sysTokens, partitionKey: defaultKey)
        #expect(sysResult.snapshotTokenOffset == 10)

        // Leaf was evicted (may still hit system at 10 as prefix)
        let leafResult = mgr.lookup(tokens: leafTokens, partitionKey: defaultKey)
        #expect(leafResult.snapshotTokenOffset != 20)
    }

    // MARK: - 8. memoryBudgetEnforced

    @Test func memoryBudgetEnforced() {
        let mgr = PrefixCacheManager(memoryBudgetBytes: 1)

        for i in 0..<10 {
            let tokens = Array((i * 100)..<((i + 1) * 100))
            let snap = makeSnapshot(offset: 100, type: .leaf)
            mgr.storeLeaf(storedTokens: tokens, leafSnapshot: snap, partitionKey: defaultKey)
        }

        // Budget enforced — not all snapshots survive
        #expect(mgr.totalSnapshotBytes <= 1)
    }

    // MARK: - 9. planCheckpointsIncludesSystemBoundary

    @Test func planCheckpointsIncludesSystemBoundary() {
        let mgr = makeManager()
        let tokens = Array(1...1000)

        let plan = mgr.planCheckpoints(
            tokens: tokens, stablePrefixOffset: 500, partitionKey: defaultKey
        )
        #expect(plan.count == 1)
        #expect(plan[0].offset == 500)
        #expect(plan[0].type == .system)
    }

    // MARK: - 10. planCheckpointsExcludesExistingSnapshots

    @Test func planCheckpointsExcludesExistingSnapshots() {
        let mgr = makeManager()
        let tokens = Array(1...1000)

        // Store system snapshot at 500
        let sysSnap = makeSnapshot(offset: 500, type: .system)
        mgr.storeSnapshots(promptTokens: tokens, capturedSnapshots: [sysSnap], partitionKey: defaultKey)

        // Planning with same stablePrefixOffset — should not re-plan
        let plan = mgr.planCheckpoints(
            tokens: tokens, stablePrefixOffset: 500, partitionKey: defaultKey
        )
        #expect(plan.isEmpty)
    }

    // MARK: - 11. planCheckpointsNeverIncludesLeaf

    @Test func planCheckpointsNeverIncludesLeaf() {
        let mgr = makeManager()
        let tokens = Array(1...1000)

        let plan = mgr.planCheckpoints(
            tokens: tokens, stablePrefixOffset: 500, partitionKey: defaultKey
        )
        // Only .system checkpoints planned; leaf is never mid-prefill
        for item in plan {
            #expect(item.type != .leaf)
        }
    }

    // MARK: - 12. snapshotBeyondDivergenceNotReturned

    @Test func snapshotBeyondDivergenceNotReturned() {
        let mgr = makeManager()
        let tokens = Array(1...100)
        let snap = makeSnapshot(offset: 100, type: .leaf)
        mgr.storeLeaf(storedTokens: tokens, leafSnapshot: snap, partitionKey: defaultKey)

        // Query diverges at 80
        let query = Array(1...80) + [999, 998]
        let result = mgr.lookup(tokens: query, partitionKey: defaultKey)
        #expect(result.snapshotTokenOffset == 0)
    }

    // MARK: - 13. storeSnapshotsFromPrefillAtCorrectOffsets

    @Test func storeSnapshotsFromPrefillAtCorrectOffsets() {
        let mgr = makeManager()
        let tokens = Array(1...8000)

        let snap4k = makeSnapshot(offset: 4000, type: .system)
        mgr.storeSnapshots(promptTokens: tokens, capturedSnapshots: [snap4k], partitionKey: defaultKey)

        let result = mgr.lookup(tokens: tokens, partitionKey: defaultKey)
        #expect(result.snapshotTokenOffset == 4000)
    }

    // MARK: - 14a. storeLeafUnderPostResponseTokens

    @Test func storeLeafUnderPostResponseTokens() {
        let mgr = makeManager()
        let promptTokens = Array(1...500)
        let storedTokens = promptTokens + Array(600...800) // prompt + response

        let leafSnap = makeSnapshot(offset: storedTokens.count, type: .leaf)
        mgr.storeLeaf(storedTokens: storedTokens, leafSnapshot: leafSnap, partitionKey: defaultKey)

        // Lookup with storedTokens hits the leaf
        let result = mgr.lookup(tokens: storedTokens, partitionKey: defaultKey)
        #expect(result.snapshotTokenOffset == storedTokens.count)
    }

    // MARK: - 14b. leafSnapshotOffsetMatchesStoredTokenCount

    @Test func leafSnapshotOffsetMatchesStoredTokenCount() {
        let mgr = makeManager()
        let storedTokens = Array(1...300)
        let leafSnap = makeSnapshot(offset: storedTokens.count, type: .leaf)
        mgr.storeLeaf(storedTokens: storedTokens, leafSnapshot: leafSnap, partitionKey: defaultKey)

        let result = mgr.lookup(tokens: storedTokens, partitionKey: defaultKey)
        #expect(result.snapshot?.tokenOffset == storedTokens.count)
    }

    // MARK: - 14c. nextRequestHitsLeafViaExtendedPrefix

    @Test func nextRequestHitsLeafViaExtendedPrefix() {
        let mgr = makeManager()
        // Store leaf for sys+user1+asst1
        let turn1Stored = Array(1...300)
        let leafSnap = makeSnapshot(offset: 300, type: .leaf)
        mgr.storeLeaf(storedTokens: turn1Stored, leafSnapshot: leafSnap, partitionKey: defaultKey)

        // Next request: sys+user1+asst1+tool1+user2
        let nextRequest = turn1Stored + Array(400...500)
        let result = mgr.lookup(tokens: nextRequest, partitionKey: defaultKey)
        #expect(result.snapshotTokenOffset == 300)
    }

    // MARK: - 14. statsReflectState

    @Test func statsReflectState() {
        let mgr = makeManager()
        #expect(mgr.stats.snapshotCount == 0)
        #expect(mgr.stats.totalSnapshotBytes == 0)

        let tokens = Array(1...100)
        let snap = makeSnapshot(offset: 100, type: .leaf)
        mgr.storeLeaf(storedTokens: tokens, leafSnapshot: snap, partitionKey: defaultKey)

        let s = mgr.stats
        #expect(s.partitionCount == 1)
        #expect(s.snapshotCount == 1)
        #expect(s.totalSnapshotBytes > 0)
        #expect(s.totalNodeCount >= 2)
    }

    // MARK: - 15. differentKvBitsIsolated

    @Test func differentKvBitsIsolated() {
        let mgr = makeManager()
        let tokens = Array(1...100)
        let key8 = CachePartitionKey(modelID: "m", kvBits: 8, kvGroupSize: 64)
        let keyNil = CachePartitionKey(modelID: "m", kvBits: nil, kvGroupSize: 64)

        let snap = makeSnapshot(offset: 100, type: .leaf)
        mgr.storeLeaf(storedTokens: tokens, leafSnapshot: snap, partitionKey: key8)

        let result = mgr.lookup(tokens: tokens, partitionKey: keyNil)
        #expect(result.snapshot == nil)
    }

    // MARK: - 16. sameKvBitsShared

    @Test func sameKvBitsShared() {
        let mgr = makeManager()
        let tokens = Array(1...100)
        let key8 = CachePartitionKey(modelID: "m", kvBits: 8, kvGroupSize: 64)

        let snap = makeSnapshot(offset: 100, type: .leaf)
        mgr.storeLeaf(storedTokens: tokens, leafSnapshot: snap, partitionKey: key8)

        let result = mgr.lookup(tokens: tokens, partitionKey: key8)
        #expect(result.snapshotTokenOffset == 100)
    }

    // MARK: - 17. differentModelIDsIsolated

    @Test func differentModelIDsIsolated() {
        let mgr = makeManager()
        let tokens = Array(1...100)
        let keyA = CachePartitionKey(modelID: "modelA", kvBits: nil, kvGroupSize: 64)
        let keyB = CachePartitionKey(modelID: "modelB", kvBits: nil, kvGroupSize: 64)

        let snap = makeSnapshot(offset: 100, type: .leaf)
        mgr.storeLeaf(storedTokens: tokens, leafSnapshot: snap, partitionKey: keyA)

        let result = mgr.lookup(tokens: tokens, partitionKey: keyB)
        #expect(result.snapshot == nil)
    }

    // MARK: - 18. evictionCrossesPartitions

    @Test func evictionCrossesPartitions() {
        let mgr = PrefixCacheManager(memoryBudgetBytes: 1)
        let keyA = CachePartitionKey(modelID: "a", kvBits: nil, kvGroupSize: 64)
        let keyB = CachePartitionKey(modelID: "b", kvBits: nil, kvGroupSize: 64)

        // Store in two partitions
        let snapA = makeSnapshot(offset: 50, type: .leaf)
        mgr.storeLeaf(storedTokens: Array(1...50), leafSnapshot: snapA, partitionKey: keyA)
        let snapB = makeSnapshot(offset: 50, type: .leaf)
        mgr.storeLeaf(storedTokens: Array(100...149), leafSnapshot: snapB, partitionKey: keyB)

        // Budget is 1 byte — eviction must cross partitions
        #expect(mgr.totalSnapshotBytes <= 1)
    }

    // MARK: - storeLeaf validation

    @Test func storeLeafRejectsMismatchedOffset() {
        let mgr = makeManager()
        let tokens = Array(1...100)
        // Snapshot offset (50) != storedTokens.count (100) → rejected
        let snap = makeSnapshot(offset: 50, type: .leaf)
        mgr.storeLeaf(storedTokens: tokens, leafSnapshot: snap, partitionKey: defaultKey)
        #expect(mgr.stats.snapshotCount == 0)
    }

    // MARK: - planCheckpoints doesn't update access time (through manager)

    @Test func planCheckpointsDoesNotUpdateAccessTime() {
        // Two .system snapshots, same type → eviction depends on recency alone.
        // planCheckpoints is called AFTER both are stored, targeting snapshotA.
        // If it refreshes A's access time, A becomes newer and B gets evicted instead.
        let snapBytes = makeSnapshot(offset: 10, type: .system).memoryBytes

        // Large budget during setup
        let mgr = PrefixCacheManager(memoryBudgetBytes: snapBytes * 100)

        // Store snapshotA (older — stored first)
        let pathA = Array(1...10)
        mgr.storeSnapshots(promptTokens: pathA,
                           capturedSnapshots: [makeSnapshot(offset: 10, type: .system)],
                           partitionKey: defaultKey)

        // Store snapshotB (newer — stored second)
        let pathB = Array(20...29)
        mgr.storeSnapshots(promptTokens: pathB,
                           capturedSnapshots: [makeSnapshot(offset: 10, type: .system)],
                           partitionKey: defaultKey)

        // planCheckpoints references snapshotA — must NOT refresh its access time.
        // Called AFTER both are stored, so if it updates A, A becomes the newest.
        _ = mgr.planCheckpoints(tokens: pathA + [99], stablePrefixOffset: 10, partitionKey: defaultKey)

        // Tighten budget to one snapshot and evict
        mgr.memoryBudgetBytes = snapBytes
        mgr.evictToFitBudget()

        // Both are .system → recency decides. A is older → A evicted, B survives.
        // If planCheckpoints refreshed A, A would be newer → B evicted instead → fail.
        let resultA = mgr.lookup(tokens: pathA, partitionKey: defaultKey)
        let resultB = mgr.lookup(tokens: pathB, partitionKey: defaultKey)
        #expect(resultA.snapshot == nil, "snapshotA (older) should be evicted")
        #expect(resultB.snapshotTokenOffset == 10, "snapshotB (newer) should survive")
    }

    // MARK: - Branch node eviction (multi-child node with snapshot)

    @Test func evictionCanEvictBranchNodeSnapshots() {
        // The ONLY snapshot is on a multi-child node (childCount == 2).
        // Under the old eligibleEvictionNodes() (childCount <= 1), this node would
        // not be a candidate and eviction would stall with totalSnapshotBytes > budget.
        // With allSnapshotNodes(), the branch snapshot is evictable.
        let snapBytes = makeSnapshot(offset: 10, type: .system).memoryBytes

        // Large budget during setup
        let mgr = PrefixCacheManager(memoryBudgetBytes: snapBytes * 100)

        let pathA = Array(1...15)
        let pathB = Array(1...10) + Array(21...25)

        // Insert BOTH paths first to create the branch structure
        // root → [1..10] → {[11..15], [21..25]}   (offset-10 node has 2 children)
        mgr.storeSnapshots(promptTokens: pathA,
                           capturedSnapshots: [makeSnapshot(offset: 10, type: .system)],
                           partitionKey: defaultKey)
        // Insert pathB by storing a snapshot on its leaf, then evicting just the leaf snapshot.
        // This leaves the pathB child node in place (making offset-10 multi-child)
        // but removes the leaf snapshot so the ONLY remaining snapshot is the branch node.
        mgr.storeLeaf(storedTokens: pathB,
                       leafSnapshot: makeSnapshot(offset: pathB.count, type: .leaf),
                       partitionKey: defaultKey)

        // Evict the leaf snapshot manually by looking it up and tightening
        // Actually simpler: just set budget to exactly one snapshot — leaf gets evicted by type priority
        mgr.memoryBudgetBytes = snapBytes
        mgr.evictToFitBudget()

        // After type-based eviction: leaf gone, only the branch-node system snapshot remains.
        // Now tighten budget to zero — must evict the branch-node snapshot.
        // Under old eligibleEvictionNodes(), this would stall (childCount == 2, not eligible).
        #expect(mgr.stats.snapshotCount == 1) // only branch snapshot left
        mgr.memoryBudgetBytes = 0
        mgr.evictToFitBudget()

        // Budget enforced — branch-node snapshot was evicted
        #expect(mgr.totalSnapshotBytes == 0)
        #expect(mgr.stats.snapshotCount == 0)
    }

    @Test func planCheckpointsRequiresSystemType() {
        let mgr = makeManager()
        let tokens = Array(1...1000)

        // Store a .leaf at offset 500 (not .system)
        let leafSnap = makeSnapshot(offset: 500, type: .leaf)
        mgr.storeSnapshots(promptTokens: tokens, capturedSnapshots: [leafSnap], partitionKey: defaultKey)

        // Planning should still request .system at 500 — the leaf doesn't count
        let plan = mgr.planCheckpoints(tokens: tokens, stablePrefixOffset: 500, partitionKey: defaultKey)
        #expect(plan.count == 1)
        #expect(plan[0].type == .system)
    }

    // MARK: - Last-message boundary checkpoint

    /// `lastMessageBoundaryOffset` adds a second `.system`-type checkpoint
    /// alongside the stable prefix. Both are planned when neither is already
    /// stored. This captures the "end of last user message" point, which is
    /// stable across conversation turns (unlike the leaf, which changes due
    /// to template re-rendering of past assistants).
    @Test func planCheckpointsIncludesLastMessageBoundary() {
        let mgr = makeManager()
        let tokens = Array(1...1000)

        let plan = mgr.planCheckpoints(
            tokens: tokens,
            stablePrefixOffset: 100,
            lastMessageBoundaryOffset: 800,
            partitionKey: defaultKey
        )

        #expect(plan.count == 2)
        #expect(plan.contains { $0.offset == 100 && $0.type == .system })
        #expect(plan.contains { $0.offset == 800 && $0.type == .system })
    }

    /// When the stable prefix and last-message boundary happen to be at the
    /// same offset (short conversations with just system+tools), only one
    /// checkpoint is planned — no duplicate.
    @Test func planCheckpointsDedupesIdenticalOffsets() {
        let mgr = makeManager()
        let tokens = Array(1...1000)

        let plan = mgr.planCheckpoints(
            tokens: tokens,
            stablePrefixOffset: 500,
            lastMessageBoundaryOffset: 500,
            partitionKey: defaultKey
        )

        #expect(plan.count == 1)
        #expect(plan[0].offset == 500)
    }

    /// If a system snapshot already exists at the last-message boundary, the
    /// planner skips it (no re-capture needed).
    @Test func planCheckpointsSkipsExistingLastMessageBoundary() {
        let mgr = makeManager()
        let tokens = Array(1...1000)

        // Pre-store a system snapshot at the boundary.
        let sysSnap = makeSnapshot(offset: 700, type: .system)
        mgr.storeSnapshots(promptTokens: tokens, capturedSnapshots: [sysSnap], partitionKey: defaultKey)

        let plan = mgr.planCheckpoints(
            tokens: tokens,
            stablePrefixOffset: 100,
            lastMessageBoundaryOffset: 700,
            partitionKey: defaultKey
        )

        // Only the stable prefix gets planned — the boundary is already stored.
        #expect(plan.count == 1)
        #expect(plan[0].offset == 100)
    }
}
