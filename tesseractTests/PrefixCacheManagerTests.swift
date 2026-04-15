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

    private func makeUniformSnapshot(
        offset: Int,
        type: HybridCacheSnapshot.CheckpointType = .system
    ) -> HybridCacheSnapshot {
        PrefixCacheTestFixtures.makeUniformSnapshot(offset: offset, type: type)
    }

    private func makeSSDKey(
        fingerprint: String = String(repeating: "a", count: 64)
    ) -> CachePartitionKey {
        CachePartitionKey(
            modelID: "test-model",
            kvBits: nil,
            kvGroupSize: 64,
            sessionAffinity: nil,
            modelFingerprint: fingerprint
        )
    }

    private func makeSSDPayload(
        bytes: Int,
        checkpointType: HybridCacheSnapshot.CheckpointType = .system
    ) -> SnapshotPayload {
        SnapshotPayload(
            tokenOffset: 4_096,
            checkpointType: checkpointType,
            layers: [
                SnapshotPayload.LayerPayload(
                    className: "KVCache",
                    state: [
                        SnapshotPayload.ArrayPayload(
                            data: Data(repeating: 0xAB, count: bytes),
                            dtype: "bfloat16",
                            shape: [1, bytes]
                        )
                    ],
                    metaState: ["meta"],
                    offset: 4_096
                )
            ]
        )
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

    /// With the default `alpha = 0`, utility scoring collapses to pure
    /// recency over the eligible set — the least-recently-accessed snapshot
    /// in the eligible set is evicted. `.system` snapshots are protected
    /// from the eligible set (see `TokenRadixTree.collectEligible`), so
    /// this test uses two `.leaf` snapshots to exercise pure-recency
    /// scoring on its own.
    @Test func evictsLowestUtilityWithinPartition() {
        let snapBytes = makeUniformSnapshot(offset: 10).memoryBytes
        let mgr = PrefixCacheManager(memoryBudgetBytes: snapBytes)

        // Older leaf on its own path.
        let oldTokens = Array(1...10)
        mgr.storeLeaf(
            storedTokens: oldTokens,
            leafSnapshot: makeUniformSnapshot(offset: oldTokens.count, type: .leaf),
            partitionKey: defaultKey
        )
        #expect(mgr.stats.snapshotCount == 1)

        // Newer leaf on an independent path. Same MLXArray shape → same
        // `memoryBytes` → exactly one eviction is needed to fit within
        // `snapBytes`.
        let newTokens = Array(20...29)
        let diagnostics = mgr.storeLeaf(
            storedTokens: newTokens,
            leafSnapshot: makeUniformSnapshot(offset: newTokens.count, type: .leaf),
            partitionKey: defaultKey
        )

        #expect(diagnostics.evictions.count == 1)
        let eviction = diagnostics.evictions[0]
        #expect(eviction.strategy == .utility)
        #expect(eviction.offset == oldTokens.count)
        #expect(eviction.checkpointType == .leaf)
        #expect(eviction.freedBytes == snapBytes)
        #expect(eviction.normalizedRecency != nil)
        #expect(eviction.normalizedFlopEfficiency == 0)
        #expect(eviction.utility != nil)
        #expect(mgr.stats.snapshotCount == 1)
        let newResult = mgr.lookup(tokens: newTokens, partitionKey: defaultKey)
        #expect(newResult.snapshotTokenOffset == newTokens.count)
        let oldResult = mgr.lookup(tokens: oldTokens, partitionKey: defaultKey)
        #expect(oldResult.snapshotTokenOffset == 0)
    }

    /// `.system` snapshots are protected from utility-scored eviction so
    /// the stable-prefix / last-message-boundary snapshot survives even
    /// when its `lastAccessTime` is the oldest in the tree. Regression
    /// test for the new-user-turn cold-prefill pathology.
    @Test func systemSnapshotProtectedFromUtilityEviction() {
        let snapBytes = makeUniformSnapshot(offset: 10).memoryBytes
        let mgr = PrefixCacheManager(memoryBudgetBytes: snapBytes)

        // Older `.system` snapshot — would be the LRU victim under pure
        // recency without the type-protection guard.
        let sysTokens = Array(1...10)
        mgr.storeSnapshots(
            promptTokens: sysTokens,
            capturedSnapshots: [makeUniformSnapshot(offset: 10, type: .system)],
            partitionKey: defaultKey
        )
        #expect(mgr.stats.snapshotCount == 1)

        // Newer `.leaf` on an independent path. The leaf is the only
        // eligible candidate; the system snapshot is protected.
        let leafTokens = Array(20...29)
        let diagnostics = mgr.storeLeaf(
            storedTokens: leafTokens,
            leafSnapshot: makeUniformSnapshot(offset: leafTokens.count, type: .leaf),
            partitionKey: defaultKey
        )

        #expect(diagnostics.evictions.count == 1)
        #expect(diagnostics.evictions[0].strategy == .utility)
        #expect(diagnostics.evictions[0].checkpointType == .leaf)

        // The protected `.system` snapshot must still be reachable; the
        // newer `.leaf` was evicted instead.
        let sysResult = mgr.lookup(tokens: sysTokens, partitionKey: defaultKey)
        #expect(sysResult.snapshotTokenOffset == 10)
        let leafResult = mgr.lookup(tokens: leafTokens, partitionKey: defaultKey)
        #expect(leafResult.snapshot == nil)
    }

    /// **Regression test**: `.lastMessageBoundary` snapshots must NOT be
    /// type-protected — only the cross-conversation `.system` (stable
    /// prefix) is. This is the fix for the long-conversation pathology
    /// where boundaries from every prior turn would accumulate as
    /// "protected" snapshots and starve fresh leaves of admission room.
    @Test func lastMessageBoundaryEvictableButStablePrefixProtected() {
        let snapBytes = makeUniformSnapshot(offset: 10).memoryBytes
        let mgr = PrefixCacheManager(memoryBudgetBytes: snapBytes)

        // Stable prefix snapshot — `.system`, type-protected.
        let stableTokens = Array(1...10)
        mgr.storeSnapshots(
            promptTokens: stableTokens,
            capturedSnapshots: [makeUniformSnapshot(offset: 10, type: .system)],
            partitionKey: defaultKey
        )
        #expect(mgr.stats.snapshotCount == 1)

        // Last-message-boundary snapshot on a different path —
        // `.lastMessageBoundary`, NOT type-protected. This will push
        // budget over and force an eviction.
        let boundaryTokens = Array(20...29)
        let diagnostics = mgr.storeSnapshots(
            promptTokens: boundaryTokens,
            capturedSnapshots: [
                makeUniformSnapshot(
                    offset: boundaryTokens.count, type: .lastMessageBoundary
                )
            ],
            partitionKey: defaultKey
        )

        // Eviction picked the older candidate. Both are now eligible
        // (`.lastMessageBoundary` is no longer type-protected, only
        // multi-child protection still applies). Recency rules: the
        // older `.system` snapshot has older `lastAccessTime` so it's
        // the LRU victim — but `.system` is still type-protected, so
        // utility scoring excludes it from the candidate set. The new
        // `.lastMessageBoundary` is the only candidate → evicted.
        #expect(diagnostics.evictions.count == 1)
        let evicted = diagnostics.evictions[0]
        #expect(evicted.checkpointType == .lastMessageBoundary)

        // Stable prefix survives.
        let stableResult = mgr.lookup(tokens: stableTokens, partitionKey: defaultKey)
        #expect(stableResult.snapshotTokenOffset == 10)
        let boundaryResult = mgr.lookup(tokens: boundaryTokens, partitionKey: defaultKey)
        #expect(boundaryResult.snapshot == nil)
    }

    /// Two `.lastMessageBoundary` snapshots from different turns: the
    /// older one is evicted under recency, the fresher one survives.
    /// This is the correct behaviour for long conversations — only the
    /// most recent boundary is useful, and old ones can age out.
    @Test func olderLastMessageBoundaryEvictedBeforeFreshOne() {
        let snapBytes = makeUniformSnapshot(offset: 10).memoryBytes
        let mgr = PrefixCacheManager(memoryBudgetBytes: snapBytes)

        // Old boundary (turn N-1).
        let oldTokens = Array(1...10)
        mgr.storeSnapshots(
            promptTokens: oldTokens,
            capturedSnapshots: [
                makeUniformSnapshot(offset: 10, type: .lastMessageBoundary)
            ],
            partitionKey: defaultKey
        )
        #expect(mgr.stats.snapshotCount == 1)

        // Fresh boundary (turn N) on an independent path.
        let freshTokens = Array(20...29)
        let diagnostics = mgr.storeSnapshots(
            promptTokens: freshTokens,
            capturedSnapshots: [
                makeUniformSnapshot(
                    offset: freshTokens.count, type: .lastMessageBoundary
                )
            ],
            partitionKey: defaultKey
        )

        // Older boundary evicted, fresh one survives.
        #expect(diagnostics.evictions.count == 1)
        #expect(diagnostics.evictions[0].offset == oldTokens.count)
        #expect(diagnostics.evictions[0].checkpointType == .lastMessageBoundary)

        let freshResult = mgr.lookup(tokens: freshTokens, partitionKey: defaultKey)
        #expect(freshResult.snapshotTokenOffset == freshTokens.count)
        let oldResult = mgr.lookup(tokens: oldTokens, partitionKey: defaultKey)
        #expect(oldResult.snapshot == nil)
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
        // Two .leaf snapshots, same type → eviction depends on recency alone.
        // (.leaf because .system is protected from the utility-eviction set.)
        // planCheckpoints is called AFTER both are stored, walking through
        // snapshotA's path. If it refreshes A's access time, A becomes
        // newer and B gets evicted instead.
        let snapBytes = makeSnapshot(offset: 10, type: .leaf).memoryBytes

        // Large budget during setup
        let mgr = PrefixCacheManager(memoryBudgetBytes: snapBytes * 100)

        // Store snapshotA (older — stored first)
        let pathA = Array(1...10)
        mgr.storeLeaf(
            storedTokens: pathA,
            leafSnapshot: makeSnapshot(offset: pathA.count, type: .leaf),
            partitionKey: defaultKey
        )

        // Store snapshotB (newer — stored second)
        let pathB = Array(20...29)
        mgr.storeLeaf(
            storedTokens: pathB,
            leafSnapshot: makeSnapshot(offset: pathB.count, type: .leaf),
            partitionKey: defaultKey
        )

        // planCheckpoints walks pathA looking for an existing system
        // snapshot at offset 10 — must NOT refresh A's access time.
        _ = mgr.planCheckpoints(
            tokens: pathA + [99], stablePrefixOffset: 10, partitionKey: defaultKey
        )

        // Tighten budget to one snapshot and evict
        mgr.memoryBudgetBytes = snapBytes
        mgr.evictToFitBudget()

        // Both are .leaf → recency decides. A is older → A evicted, B survives.
        // If planCheckpoints refreshed A, A would be newer → B evicted instead → fail.
        let resultA = mgr.lookup(tokens: pathA, partitionKey: defaultKey)
        let resultB = mgr.lookup(tokens: pathB, partitionKey: defaultKey)
        #expect(resultA.snapshot == nil, "snapshotA (older) should be evicted")
        #expect(resultB.snapshotTokenOffset == pathB.count, "snapshotB (newer) should survive")
    }

    /// Multi-child branch snapshots are protected from utility scoring
    /// (Marconi rule: candidates must have `childCount <= 1`). The hard
    /// budget invariant is preserved by a fallback that drops the oldest
    /// snapshot regardless of `childCount` when the eligible set is empty.
    @Test func branchNodeFallbackHonorsHardBudget() {
        let snapBytes = makeSnapshot(offset: 10, type: .system).memoryBytes
        let mgr = PrefixCacheManager(memoryBudgetBytes: snapBytes * 100)

        // root → [1..10] (system snap)
        //            → [11..15]
        //            → [21..25] (leaf)
        //            → [31..35]
        //
        // The extra snapshot-less suffix keeps the system node multi-child
        // even after the leaf snapshot is evicted, so the second drain is
        // forced onto the fallback path.
        let pathA = Array(1...15)
        let pathB = Array(1...10) + Array(21...25)
        let pathC = Array(1...10) + Array(31...35)
        mgr.storeSnapshots(
            promptTokens: pathA,
            capturedSnapshots: [makeSnapshot(offset: 10, type: .system)],
            partitionKey: defaultKey
        )
        mgr.storeLeaf(
            storedTokens: pathB,
            leafSnapshot: makeSnapshot(offset: pathB.count, type: .leaf),
            partitionKey: defaultKey
        )
        mgr.storeSnapshots(
            promptTokens: pathC,
            capturedSnapshots: [makeSnapshot(offset: 10, type: .system)],
            partitionKey: defaultKey
        )

        // Tighten to one snapshot — the eligible leaf is evicted by utility
        // scoring, leaving only the branch-node system snapshot.
        mgr.memoryBudgetBytes = snapBytes
        let utilityEvictions = mgr.evictToFitBudget()
        #expect(utilityEvictions.count == 1)
        #expect(utilityEvictions[0].strategy == .utility)
        #expect(mgr.stats.snapshotCount == 1)

        // Tighten to zero. The remaining snapshot is on a multi-child node
        // (not in the eligible set), so the fallback drops it.
        mgr.memoryBudgetBytes = 0
        let fallbackEvictions = mgr.evictToFitBudget()
        #expect(fallbackEvictions.count == 1)
        let fallback = fallbackEvictions[0]
        #expect(fallback.strategy == .fallback)
        #expect(fallback.checkpointType == .system)
        #expect(fallback.normalizedRecency == nil)
        #expect(fallback.normalizedFlopEfficiency == nil)
        #expect(fallback.utility == nil)
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

        // Stable prefix is `.system` (cross-conversation, type-protected).
        // Last-message boundary is `.lastMessageBoundary` (cross-turn,
        // LRU-evictable so old boundaries don't accumulate forever).
        #expect(plan.count == 2)
        #expect(plan.contains { $0.offset == 100 && $0.type == .system })
        #expect(plan.contains { $0.offset == 800 && $0.type == .lastMessageBoundary })
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

        // Pre-store a `.lastMessageBoundary` snapshot at the boundary
        // offset (matching what the real planner stores there now).
        let boundarySnap = makeSnapshot(offset: 700, type: .lastMessageBoundary)
        mgr.storeSnapshots(
            promptTokens: tokens,
            capturedSnapshots: [boundarySnap],
            partitionKey: defaultKey
        )

        let plan = mgr.planCheckpoints(
            tokens: tokens,
            stablePrefixOffset: 100,
            lastMessageBoundaryOffset: 700,
            partitionKey: defaultKey
        )

        // Only the stable prefix gets planned — the boundary is already stored.
        #expect(plan.count == 1)
        #expect(plan[0].offset == 100)
        #expect(plan[0].type == .system)
    }

    // MARK: - Speculative branch-point candidates

    @Test func divergenceInsideCompressedEdgeCreatesCandidate() {
        let mgr = makeManager()
        let stored = [1, 2, 3, 4]
        mgr.storeLeaf(
            storedTokens: stored,
            leafSnapshot: makeSnapshot(offset: stored.count, type: .leaf),
            partitionKey: defaultKey
        )

        let plan = mgr.planCheckpoints(
            tokens: [1, 2, 5, 6],
            stablePrefixOffset: nil,
            partitionKey: defaultKey
        )

        #expect(plan.count == 1)
        #expect(plan[0].offset == 2)
        #expect(plan[0].type == .branchPoint)
    }

    @Test func exactPathExtensionDoesNotCreateBranchPoint() {
        // Pure extension is already covered by the leaf checkpoint, so no
        // speculative branch point is needed.
        let mgr = makeManager()
        mgr.storeLeaf(
            storedTokens: [1, 2, 3],
            leafSnapshot: makeSnapshot(offset: 3, type: .leaf),
            partitionKey: defaultKey
        )

        let plan = mgr.planCheckpoints(
            tokens: [1, 2, 3, 4, 5],
            stablePrefixOffset: nil,
            partitionKey: defaultKey
        )

        #expect(!plan.contains(where: { $0.type == .branchPoint }))
    }

    @Test func nodeBoundaryDivergenceDoesNotCreateIntermediateCheckpoint() {
        let mgr = makeManager()
        // Storing both [1,2] and [1,2,3,4] materializes node [1,2] as a real
        // boundary, so a [1,2,9] walk diverges at the boundary not mid-edge.
        mgr.storeLeaf(
            storedTokens: [1, 2],
            leafSnapshot: makeSnapshot(offset: 2, type: .leaf),
            partitionKey: defaultKey
        )
        mgr.storeLeaf(
            storedTokens: [1, 2, 3, 4],
            leafSnapshot: makeSnapshot(offset: 4, type: .leaf),
            partitionKey: defaultKey
        )

        let plan = mgr.planCheckpoints(
            tokens: [1, 2, 9, 10],
            stablePrefixOffset: nil,
            partitionKey: defaultKey
        )

        #expect(!plan.contains(where: { $0.type == .branchPoint }))
    }

    @Test func coldTreeNoSpeculativeCandidates() {
        let mgr = makeManager()
        let plan = mgr.planCheckpoints(
            tokens: Array(1...10),
            stablePrefixOffset: nil,
            partitionKey: defaultKey
        )
        #expect(plan.isEmpty)
    }

    @Test func existingSnapshotNotReCandidate() {
        // After a real prior capture the intermediate node is materialized
        // and carries a snapshot — re-running the planner on the same
        // divergent tokens must not re-add the branch point.
        let mgr = makeManager()
        mgr.storeSnapshots(
            promptTokens: [1, 2, 3, 4],
            capturedSnapshots: [makeSnapshot(offset: 2, type: .system)],
            partitionKey: defaultKey
        )

        let plan = mgr.planCheckpoints(
            tokens: [1, 2, 5, 6],
            stablePrefixOffset: nil,
            partitionKey: defaultKey
        )

        #expect(!plan.contains(where: { $0.type == .branchPoint }))
    }

    @Test func atMostOneBranchPointPerSequence() {
        // The walk stops at the first mid-edge divergence by construction —
        // even with multiple plausible split sites along the path, only one
        // candidate is emitted.
        let mgr = makeManager()
        mgr.storeLeaf(
            storedTokens: [1, 2, 3, 4, 5],
            leafSnapshot: makeSnapshot(offset: 5, type: .leaf),
            partitionKey: defaultKey
        )
        mgr.storeLeaf(
            storedTokens: [1, 2, 8, 9, 10],
            leafSnapshot: makeSnapshot(offset: 5, type: .leaf),
            partitionKey: defaultKey
        )

        // Tree: root -> [1,2] -> {[3,4,5], [8,9,10]}. [1,2,3,7,11] descends
        // through [1,2] then diverges mid-edge of [3,4,5] at offset 3.
        let plan = mgr.planCheckpoints(
            tokens: [1, 2, 3, 7, 11],
            stablePrefixOffset: nil,
            partitionKey: defaultKey
        )

        let branchPoints = plan.filter { $0.type == .branchPoint }
        #expect(branchPoints.count == 1)
        #expect(branchPoints.first?.offset == 3)
    }

    @Test func branchPointCoexistsWithSystemCheckpoint() {
        let mgr = makeManager()
        mgr.storeLeaf(
            storedTokens: Array(1...50),
            leafSnapshot: makeSnapshot(offset: 50, type: .leaf),
            partitionKey: defaultKey
        )

        let request = Array(1...30) + [999, 1000]
        let plan = mgr.planCheckpoints(
            tokens: request,
            stablePrefixOffset: 10,
            partitionKey: defaultKey
        )

        #expect(plan.contains { $0.offset == 10 && $0.type == .system })
        #expect(plan.contains { $0.offset == 30 && $0.type == .branchPoint })
    }

    @Test func branchPointSkippedIfSameOffsetAsSystem() {
        let mgr = makeManager()
        mgr.storeLeaf(
            storedTokens: Array(1...50),
            leafSnapshot: makeSnapshot(offset: 50, type: .leaf),
            partitionKey: defaultKey
        )

        let request = Array(1...10) + [999, 1000]
        let plan = mgr.planCheckpoints(
            tokens: request,
            stablePrefixOffset: 10,
            partitionKey: defaultKey
        )

        #expect(plan.count == 1)
        #expect(plan[0].offset == 10)
        #expect(plan[0].type == .system)
    }

    // MARK: - Phase 3.1 alignment checkpoints

    @Test func alignedSnapshotSinglePass() {
        let mgr = makeManager()

        let exactPath = Array(1...200)
        mgr.storeSnapshots(
            promptTokens: exactPath,
            capturedSnapshots: [makeSnapshot(offset: 200, type: .system)],
            partitionKey: defaultKey
        )

        let exactLookup = mgr.lookup(tokens: exactPath + [999], partitionKey: defaultKey)
        #expect(exactLookup.snapshotTokenOffset == 200)
        #expect(exactLookup.sharedPrefixLength == 200)
        #expect(
            mgr.alignmentCheckpointOffset(
                lookupResult: exactLookup,
                totalTokenCount: exactPath.count + 1,
                plannedCheckpoints: []
            ) == nil
        )

        let longPath = Array(1...500)
        mgr.storeSnapshots(
            promptTokens: longPath,
            capturedSnapshots: [makeSnapshot(offset: 200, type: .system)],
            partitionKey: defaultKey
        )

        let plannedLookup = mgr.lookup(tokens: longPath + [1000], partitionKey: defaultKey)
        #expect(plannedLookup.snapshotTokenOffset == 200)
        #expect(plannedLookup.sharedPrefixLength == 500)
        #expect(
            mgr.alignmentCheckpointOffset(
                lookupResult: plannedLookup,
                totalTokenCount: longPath.count + 1,
                plannedCheckpoints: [(offset: 500, type: .system)]
            ) == nil
        )
    }

    @Test func largeGapTriggersTwoPass() {
        let mgr = makeManager()
        let matchedPath = Array(1...500)
        mgr.storeSnapshots(
            promptTokens: matchedPath,
            capturedSnapshots: [makeSnapshot(offset: 100, type: .system)],
            partitionKey: defaultKey
        )

        let lookup = mgr.lookup(tokens: matchedPath + [999], partitionKey: defaultKey)
        #expect(lookup.snapshotTokenOffset == 100)
        #expect(lookup.sharedPrefixLength == 500)

        let alignmentOffset = mgr.alignmentCheckpointOffset(
            lookupResult: lookup,
            totalTokenCount: matchedPath.count + 1,
            plannedCheckpoints: []
        )
        #expect(alignmentOffset == 500)
    }

    @Test func smallGapSinglePass() {
        let mgr = makeManager()
        let matchedPath = Array(1...450)
        mgr.storeSnapshots(
            promptTokens: matchedPath,
            capturedSnapshots: [makeSnapshot(offset: 300, type: .system)],
            partitionKey: defaultKey
        )

        let lookup = mgr.lookup(tokens: matchedPath + [999], partitionKey: defaultKey)
        #expect(lookup.snapshotTokenOffset == 300)
        #expect(lookup.sharedPrefixLength == 450)

        let alignmentOffset = mgr.alignmentCheckpointOffset(
            lookupResult: lookup,
            totalTokenCount: matchedPath.count + 1,
            plannedCheckpoints: []
        )
        #expect(alignmentOffset == nil)
    }

    @Test func lookupAndPlanCheckpointsIncludesAlignmentCheckpoint() {
        let mgr = makeManager()
        let matchedPath = Array(1...500)
        mgr.storeSnapshots(
            promptTokens: matchedPath,
            capturedSnapshots: [makeSnapshot(offset: 100, type: .system)],
            partitionKey: defaultKey
        )

        let request = matchedPath + [999]
        let result = mgr.lookupAndPlanCheckpoints(
            tokens: request,
            stablePrefixOffset: nil,
            partitionKey: defaultKey
        )

        #expect(result.lookup.snapshotTokenOffset == 100)
        #expect(result.lookup.sharedPrefixLength == 500)
        #expect(result.plan.contains { $0.offset == 500 && $0.type == .branchPoint })

        let suffixPlan = result.plan.filter { $0.offset > result.lookup.snapshotTokenOffset }
        #expect(suffixPlan.contains { $0.offset == 500 && $0.type == .branchPoint })
    }

    // MARK: - Session-scoped partitioning

    /// Two `CachePartitionKey`s that differ only by `sessionAffinity` must
    /// be treated as distinct partitions. A snapshot stored under one
    /// session must not be visible to a lookup from another session.
    /// Regression test for the "long-running subagent evicts main agent"
    /// pathology.
    @Test func differentSessionsIsolated() {
        let mgr = makeManager()
        let mainKey = CachePartitionKey(
            modelID: "test-model", kvBits: nil, kvGroupSize: 64,
            sessionAffinity: "ses_main"
        )
        let subKey = CachePartitionKey(
            modelID: "test-model", kvBits: nil, kvGroupSize: 64,
            sessionAffinity: "ses_sub"
        )
        let tokens = Array(1...100)
        // Store the same token sequence under both sessions.
        mgr.storeLeaf(
            storedTokens: tokens,
            leafSnapshot: makeSnapshot(offset: tokens.count, type: .leaf),
            partitionKey: mainKey
        )
        mgr.storeLeaf(
            storedTokens: tokens,
            leafSnapshot: makeSnapshot(offset: tokens.count, type: .leaf),
            partitionKey: subKey
        )

        // Two independent partitions, each with its own snapshot.
        #expect(mgr.stats.partitionCount == 2)
        #expect(mgr.stats.snapshotCount == 2)

        // Each session's lookup hits its own partition.
        let mainResult = mgr.lookup(tokens: tokens, partitionKey: mainKey)
        #expect(mainResult.snapshotTokenOffset == tokens.count)
        let subResult = mgr.lookup(tokens: tokens, partitionKey: subKey)
        #expect(subResult.snapshotTokenOffset == tokens.count)

        // A third session with no prior stores gets a clean miss — not
        // the main or sub partition's snapshot.
        let otherKey = CachePartitionKey(
            modelID: "test-model", kvBits: nil, kvGroupSize: 64,
            sessionAffinity: "ses_other"
        )
        let otherResult = mgr.lookup(tokens: tokens, partitionKey: otherKey)
        #expect(otherResult.snapshot == nil)
    }

    /// `nil` session affinity is a valid partition key and all `nil`
    /// callers share a single partition (pre-session-scoping behavior).
    @Test func nilSessionAffinityShared() {
        let mgr = makeManager()
        let key1 = CachePartitionKey(modelID: "m", kvBits: nil, kvGroupSize: 64)
        let key2 = CachePartitionKey(modelID: "m", kvBits: nil, kvGroupSize: 64)
        #expect(key1 == key2)
        #expect(key1.hashValue == key2.hashValue)

        let tokens = Array(1...50)
        mgr.storeLeaf(
            storedTokens: tokens,
            leafSnapshot: makeSnapshot(offset: tokens.count, type: .leaf),
            partitionKey: key1
        )
        let result = mgr.lookup(tokens: tokens, partitionKey: key2)
        #expect(result.snapshotTokenOffset == tokens.count)
    }

    /// **Acceptance test for the session-scoping fix.** A long-running
    /// subagent fills the cache with its own snapshots. The idle main
    /// agent's tall stable-prefix snapshot must survive, because eviction
    /// prefers the writing partition (subagent) over the idle one (main).
    ///
    /// Setup: main agent stores one `.leaf` snapshot (the only thing
    /// eligible for eviction in that partition — the stable prefix is
    /// `.system` and protected). Subagent then stores many `.leaf`
    /// snapshots, blowing past the budget. Each subagent store triggers
    /// `evictToFitBudget`, which must evict from the subagent's own
    /// partition first.
    @Test func idleMainAgentSurvivesSubagentChurn() {
        let snapBytes = makeUniformSnapshot(offset: 10, type: .leaf).memoryBytes
        // Budget fits exactly 3 snapshots.
        let mgr = PrefixCacheManager(memoryBudgetBytes: snapBytes * 3)

        let mainKey = CachePartitionKey(
            modelID: "test-model", kvBits: nil, kvGroupSize: 64,
            sessionAffinity: "ses_main"
        )
        let subKey = CachePartitionKey(
            modelID: "test-model", kvBits: nil, kvGroupSize: 64,
            sessionAffinity: "ses_sub"
        )

        // Main agent stores one leaf, then goes idle.
        let mainTokens = Array(1...10)
        mgr.storeLeaf(
            storedTokens: mainTokens,
            leafSnapshot: makeUniformSnapshot(offset: mainTokens.count, type: .leaf),
            partitionKey: mainKey
        )
        #expect(mgr.stats.snapshotCount == 1)

        // Subagent churns 10 writes, each a distinct leaf. Budget only
        // fits 3 snapshots, so eviction fires 7 times. Under the
        // preferred-partition rule, all evictions hit the subagent's
        // own partition; the idle main-agent leaf must survive.
        for i in 0..<10 {
            let subTokens = Array((1000 + i * 100)..<(1000 + i * 100 + 10))
            mgr.storeLeaf(
                storedTokens: subTokens,
                leafSnapshot: makeUniformSnapshot(offset: subTokens.count, type: .leaf),
                partitionKey: subKey
            )
        }

        // Cache is at most 3 snapshots total. At least one must be the
        // main agent's surviving leaf.
        #expect(mgr.stats.snapshotCount <= 3)
        let mainResult = mgr.lookup(tokens: mainTokens, partitionKey: mainKey)
        #expect(
            mainResult.snapshotTokenOffset == mainTokens.count,
            "Idle main agent's snapshot should survive subagent churn"
        )
    }

    /// When the writing partition's eligible set is exhausted (all
    /// remaining snapshots are type-protected `.system`), eviction
    /// spills over to other partitions via the global path. Verifies
    /// that session scoping doesn't break the hard-budget invariant.
    @Test func preferredPartitionSpillsToGlobalWhenExhausted() {
        let snapBytes = makeUniformSnapshot(offset: 10, type: .leaf).memoryBytes
        // Budget fits 1 snapshot.
        let mgr = PrefixCacheManager(memoryBudgetBytes: snapBytes)

        let idleKey = CachePartitionKey(
            modelID: "test-model", kvBits: nil, kvGroupSize: 64,
            sessionAffinity: "ses_idle"
        )
        let writingKey = CachePartitionKey(
            modelID: "test-model", kvBits: nil, kvGroupSize: 64,
            sessionAffinity: "ses_writing"
        )

        // Idle partition: one evictable leaf.
        let idleTokens = Array(1...10)
        mgr.storeLeaf(
            storedTokens: idleTokens,
            leafSnapshot: makeUniformSnapshot(offset: idleTokens.count, type: .leaf),
            partitionKey: idleKey
        )
        #expect(mgr.stats.snapshotCount == 1)

        // Writing partition: two `.system` snapshots (type-protected,
        // ineligible in the preferred-utility path) on distinct linear
        // paths. The second store triggers eviction; the writing
        // partition has no eligible utility candidates, so eviction
        // spills over to the idle partition's leaf.
        let writingTokens1 = Array(2000...2009)
        mgr.storeSnapshots(
            promptTokens: writingTokens1,
            capturedSnapshots: [makeUniformSnapshot(offset: 10, type: .system)],
            partitionKey: writingKey
        )
        let writingTokens2 = Array(3000...3009)
        mgr.storeSnapshots(
            promptTokens: writingTokens2,
            capturedSnapshots: [makeUniformSnapshot(offset: 10, type: .system)],
            partitionKey: writingKey
        )

        // Budget must still be respected.
        #expect(mgr.totalSnapshotBytes <= snapBytes)

        // The idle partition's leaf was evicted to make room — spill
        // path worked. At least one of the writing partition's system
        // snapshots survived.
        let idleResult = mgr.lookup(tokens: idleTokens, partitionKey: idleKey)
        #expect(idleResult.snapshot == nil, "Idle leaf should have been evicted via spill")
    }

    // MARK: - Task 4.1.8: Eviction body-drop + cleanup-suppression guards

    /// State 4 (body + committed ref) body-drops to state 5 (body
    /// absent + committed ref). The eviction loop must call
    /// `evictSnapshot` but skip `evictNode`, leaving the node
    /// attached to the tree as an SSD-backed lookup target.
    /// Regression check for the orphan-leak bug flagged on 2026-04-14.
    @Test func evictionBodyDropsStateFourNodeToStateFive() {
        let snapBytes = makeUniformSnapshot(offset: 10, type: .leaf).memoryBytes
        let tieredStore = TieredSnapshotStore(ssdConfig: nil)
        let mgr = PrefixCacheManager(
            memoryBudgetBytes: snapBytes,
            tieredStore: tieredStore
        )

        let firstTokens = Array(1...10)
        mgr.storeLeaf(
            storedTokens: firstTokens,
            leafSnapshot: makeUniformSnapshot(offset: firstTokens.count, type: .leaf),
            partitionKey: defaultKey
        )
        let tree = tieredStore.tree(for: defaultKey)!
        let (firstNode, _) = tree.findBestSnapshot(
            tokens: firstTokens, updateAccess: false
        )!
        firstNode.storageRef = PrefixCacheTestFixtures.makeStorageRef(
            committed: true,
            tokenOffset: firstTokens.count
        )

        let secondTokens = Array(20...29)
        let diag = mgr.storeLeaf(
            storedTokens: secondTokens,
            leafSnapshot: makeUniformSnapshot(offset: secondTokens.count, type: .leaf),
            partitionKey: defaultKey
        )

        #expect(diag.evictions.count == 1)
        #expect(diag.evictions[0].checkpointType == .leaf)
        #expect(firstNode.snapshot == nil)
        #expect(firstNode.storageRef != nil)
        #expect(firstNode.storageRef?.committed == true)
        #expect(firstNode.parent != nil)
        let secondResult = mgr.lookup(tokens: secondTokens, partitionKey: defaultKey)
        #expect(secondResult.snapshotTokenOffset == secondTokens.count)
        let firstResult = mgr.lookup(tokens: firstTokens, partitionKey: defaultKey)
        #expect(firstResult.snapshot == nil)
    }

    /// State 2 (body + pending ref) body-drops to state 3 (body
    /// absent + pending ref). Pending refs pin the node so the
    /// writer's commit callback can later flip it to state 5.
    @Test func evictionBodyDropsStateTwoNodeToStateThree() {
        let snapBytes = makeUniformSnapshot(offset: 10, type: .leaf).memoryBytes
        let tieredStore = TieredSnapshotStore(ssdConfig: nil)
        let mgr = PrefixCacheManager(
            memoryBudgetBytes: snapBytes,
            tieredStore: tieredStore
        )

        let firstTokens = Array(1...10)
        mgr.storeLeaf(
            storedTokens: firstTokens,
            leafSnapshot: makeUniformSnapshot(offset: firstTokens.count, type: .leaf),
            partitionKey: defaultKey
        )
        let tree = tieredStore.tree(for: defaultKey)!
        let (firstNode, _) = tree.findBestSnapshot(
            tokens: firstTokens, updateAccess: false
        )!
        firstNode.storageRef = PrefixCacheTestFixtures.makeStorageRef(
            committed: false,
            tokenOffset: firstTokens.count
        )

        let secondTokens = Array(20...29)
        mgr.storeLeaf(
            storedTokens: secondTokens,
            leafSnapshot: makeUniformSnapshot(offset: secondTokens.count, type: .leaf),
            partitionKey: defaultKey
        )

        #expect(firstNode.snapshot == nil)
        #expect(firstNode.storageRef != nil)
        #expect(firstNode.storageRef?.committed == false)
        #expect(firstNode.parent != nil)
        // State 3 is not a hit target — returning one would race the
        // writer and surface an absent or half-written file.
        let firstResult = mgr.lookup(tokens: firstTokens, partitionKey: defaultKey)
        #expect(firstResult.snapshot == nil)
    }

    /// Regression check: a ref-less state-1 victim is still hard-
    /// deleted. The new guard must not accidentally pin RAM-only
    /// nodes into the tree.
    @Test func evictionHardDeletesRefLessVictim() {
        let snapBytes = makeUniformSnapshot(offset: 10, type: .leaf).memoryBytes
        let tieredStore = TieredSnapshotStore(ssdConfig: nil)
        let mgr = PrefixCacheManager(
            memoryBudgetBytes: snapBytes,
            tieredStore: tieredStore
        )

        let firstTokens = Array(1...10)
        mgr.storeLeaf(
            storedTokens: firstTokens,
            leafSnapshot: makeUniformSnapshot(offset: firstTokens.count, type: .leaf),
            partitionKey: defaultKey
        )
        let tree = tieredStore.tree(for: defaultKey)!
        let (firstNode, _) = tree.findBestSnapshot(
            tokens: firstTokens, updateAccess: false
        )!
        #expect(firstNode.storageRef == nil)
        let initialNodeCount = tree.nodeCount

        let secondTokens = Array(20...29)
        mgr.storeLeaf(
            storedTokens: secondTokens,
            leafSnapshot: makeUniformSnapshot(offset: secondTokens.count, type: .leaf),
            partitionKey: defaultKey
        )

        #expect(firstNode.snapshot == nil)
        #expect(firstNode.parent == nil)
        #expect(tree.nodeCount == initialNodeCount)
        let firstResult = mgr.lookup(tokens: firstTokens, partitionKey: defaultKey)
        #expect(firstResult.snapshot == nil)
    }

    /// `.system` type protection still applies even when the system
    /// node carries a committed storage ref. The `.leaf` victim is
    /// hard-deleted; the `.system` body + ref are preserved.
    @Test func systemTypeProtectionHoldsWithStorageRef() {
        let snapBytes = makeUniformSnapshot(offset: 10, type: .leaf).memoryBytes
        let tieredStore = TieredSnapshotStore(ssdConfig: nil)
        let mgr = PrefixCacheManager(
            memoryBudgetBytes: snapBytes,
            tieredStore: tieredStore
        )

        let sysTokens = Array(1...10)
        mgr.storeSnapshots(
            promptTokens: sysTokens,
            capturedSnapshots: [makeUniformSnapshot(offset: 10, type: .system)],
            partitionKey: defaultKey
        )
        let tree = tieredStore.tree(for: defaultKey)!
        let (sysNode, _) = tree.findBestSnapshot(
            tokens: sysTokens, updateAccess: false
        )!
        sysNode.storageRef = PrefixCacheTestFixtures.makeStorageRef(
            committed: true,
            type: .system,
            tokenOffset: sysTokens.count
        )

        let leafTokens = Array(20...29)
        mgr.storeLeaf(
            storedTokens: leafTokens,
            leafSnapshot: makeUniformSnapshot(offset: leafTokens.count, type: .leaf),
            partitionKey: defaultKey
        )

        #expect(sysNode.snapshot != nil)
        #expect(sysNode.storageRef != nil)
        #expect(sysNode.storageRef?.committed == true)
        let leafResult = mgr.lookup(tokens: leafTokens, partitionKey: defaultKey)
        #expect(leafResult.snapshot == nil)
    }

    // MARK: - SSD hydration: lookup / promote / clearStorageRef

    /// Build a tiered-store-backed manager whose SSD tier points at a
    /// fresh scratch directory. The partition is preregistered so
    /// state-5 refs survive the lookup's fingerprint/manifest checks.
    private func makeSSDManager(
        budgetBytes: Int = 1_000_000
    ) -> (PrefixCacheManager, TieredSnapshotStore, URL) {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("prefix-cache-ssd-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let config = SSDPrefixCacheConfig(
            enabled: true,
            rootURL: root,
            budgetBytes: budgetBytes,
            maxPendingBytes: 10_000_000
        )
        let tieredStore = TieredSnapshotStore(ssdConfig: config)
        let mgr = PrefixCacheManager(
            memoryBudgetBytes: 10_000_000,
            tieredStore: tieredStore
        )
        return (mgr, tieredStore, root)
    }

    /// State 5 lookup (body absent + committed ref) surfaces `.ssdHit`
    /// carrying the ref, the SSD store reference, and the node.
    @Test func lookupReturnsSSDHitForStateFiveNode() {
        let (mgr, tieredStore, root) = makeSSDManager()
        defer { try? FileManager.default.removeItem(at: root) }

        let tokens = Array(1...10)
        let tree = tieredStore.getOrCreateTree(for: defaultKey)
        let node = tree.insertPath(tokens: tokens)
        let ref = PrefixCacheTestFixtures.makeStorageRef(
            committed: true,
            type: .leaf,
            tokenOffset: tokens.count
        )
        node.storageRef = ref
        // Node has a committed ref but no body — state 5.
        #expect(node.snapshot == nil)

        let result = mgr.lookup(tokens: tokens, partitionKey: defaultKey)
        guard case .ssdHit(let ctx) = result.reason else {
            #expect(Bool(false), "Expected .ssdHit, got \(result.reason)")
            return
        }
        #expect(ctx.storageRef.snapshotID == ref.snapshotID)
        #expect(ctx.node === node)
        #expect(result.snapshot == nil)
        #expect(result.snapshotTokenOffset == tokens.count)
    }

    /// State 3 lookup (body absent + pending ref) stays a miss —
    /// returning an SSD hit on an in-flight write would race the
    /// writer and surface an absent or half-written file.
    @Test func lookupTreatsPendingRefAsMiss() {
        let (mgr, tieredStore, root) = makeSSDManager()
        defer { try? FileManager.default.removeItem(at: root) }

        let tokens = Array(1...10)
        let tree = tieredStore.getOrCreateTree(for: defaultKey)
        let node = tree.insertPath(tokens: tokens)
        node.storageRef = PrefixCacheTestFixtures.makeStorageRef(
            committed: false,
            tokenOffset: tokens.count
        )

        let result = mgr.lookup(tokens: tokens, partitionKey: defaultKey)
        if case .missNoSnapshotInPrefix = result.reason {
            // expected
        } else {
            #expect(Bool(false), "Expected miss, got \(result.reason)")
        }
    }

    /// `promote` moves a state-5 node back to state 4 by attaching
    /// the hydrated body. Subsequent lookups surface a normal RAM hit.
    @Test func promoteTransitionsStateFiveToStateFour() {
        let (mgr, tieredStore, root) = makeSSDManager()
        defer { try? FileManager.default.removeItem(at: root) }

        let tokens = Array(1...10)
        let tree = tieredStore.getOrCreateTree(for: defaultKey)
        let node = tree.insertPath(tokens: tokens)
        node.storageRef = PrefixCacheTestFixtures.makeStorageRef(
            committed: true,
            tokenOffset: tokens.count
        )

        let hydrated = makeUniformSnapshot(offset: tokens.count, type: .leaf)
        mgr.promote(node: node, snapshot: hydrated, partitionKey: defaultKey)

        #expect(node.snapshot != nil)
        #expect(node.storageRef != nil)  // ref preserved
        let result = mgr.lookup(tokens: tokens, partitionKey: defaultKey)
        if case .hit = result.reason {
            // expected
        } else {
            #expect(Bool(false), "Expected .hit after promote, got \(result.reason)")
        }
    }

    /// `clearStorageRef` on a state-5 leaf with no body removes the
    /// node from the tree entirely. Subsequent lookups miss via the
    /// normal `missNoSnapshotInPrefix` path.
    @Test func clearStorageRefRemovesLeafWithNoBody() {
        let (mgr, tieredStore, root) = makeSSDManager()
        defer { try? FileManager.default.removeItem(at: root) }

        let tokens = Array(1...10)
        let tree = tieredStore.getOrCreateTree(for: defaultKey)
        let node = tree.insertPath(tokens: tokens)
        node.storageRef = PrefixCacheTestFixtures.makeStorageRef(
            committed: true,
            tokenOffset: tokens.count
        )
        let nodeCountBefore = tree.nodeCount

        mgr.clearStorageRef(node: node, partitionKey: defaultKey)

        #expect(node.storageRef == nil)
        #expect(node.parent == nil)  // detached
        #expect(tree.nodeCount < nodeCountBefore)
        let result = mgr.lookup(tokens: tokens, partitionKey: defaultKey)
        #expect(result.snapshot == nil)
    }

    /// State-4 lookups (body + committed ref) must bump the SSD
    /// descriptor's `lastAccessAt` so hot RAM entries do not look
    /// stale to the SSD LRU when the body is eventually dropped.
    @Test func lookupOnStateFourNodeBumpsSSDRecencyViaRecordHit() {
        let (mgr, tieredStore, root) = makeSSDManager()
        defer { try? FileManager.default.removeItem(at: root) }

        // Store a leaf (body) and register the partition so the SSD
        // store's manifest carries an entry for the ref ID.
        let tokens = Array(1...10)
        mgr.storeLeaf(
            storedTokens: tokens,
            leafSnapshot: makeUniformSnapshot(offset: tokens.count, type: .leaf),
            partitionKey: defaultKey
        )
        let tree = tieredStore.tree(for: defaultKey)!
        let (node, _) = tree.findBestSnapshot(tokens: tokens, updateAccess: false)!

        // Seed the SSD manifest with a descriptor so `recordHit` has
        // a target. Partition must be registered first; the manager's
        // warmStart normally drives this, but we mint the config
        // directly here.
        let ssdStore = tieredStore.ssdStore!
        let meta = PartitionMeta(
            modelID: "test-model",
            modelFingerprint: String(repeating: "a", count: 64),
            kvBits: nil,
            kvGroupSize: 64,
            sessionAffinity: nil,
            createdAt: 0,
            schemaVersion: SnapshotManifestSchema.currentVersion
        )
        let digest = defaultKey.partitionDigest
        ssdStore.registerPartition(meta, digest: digest)

        let ref = SnapshotStorageRef(
            snapshotID: UUID().uuidString,
            partitionDigest: digest,
            tokenOffset: tokens.count,
            checkpointType: .leaf,
            bytesOnDisk: 1024,
            lastAccessTime: .now,
            committed: true
        )
        node.storageRef = ref

        // Manually insert a descriptor with an older lastAccessAt so
        // the recordHit bump is observable on the next read.
        let originalAccess: Double = 0
        let descriptor = PersistedSnapshotDescriptor(
            snapshotID: ref.snapshotID,
            partitionDigest: digest,
            pathFromRoot: tokens,
            tokenOffset: tokens.count,
            checkpointType: HybridCacheSnapshot.CheckpointType.leaf.wireString,
            bytes: 1024,
            createdAt: originalAccess,
            lastAccessAt: originalAccess,
            fileRelativePath: PersistedSnapshotDescriptor.relativeFilePath(
                snapshotID: ref.snapshotID,
                partitionDigest: digest
            ),
            schemaVersion: SnapshotManifestSchema.currentVersion
        )
        ssdStore.seedDescriptorForTesting(descriptor)

        // Lookup is a state-4 hit: body present, committed ref
        // pointing at the seeded descriptor.
        let result = mgr.lookup(tokens: tokens, partitionKey: defaultKey)
        if case .hit = result.reason {
            // expected
        } else {
            #expect(Bool(false), "Expected RAM hit, got \(result.reason)")
        }

        // The recordHit bump should have moved lastAccessAt forward.
        let bumped = ssdStore.lastAccessAtForTesting(id: ref.snapshotID)
        #expect(bumped > originalAccess)
    }

    /// The production store path must register the SSD partition
    /// before the first enqueue so a flush + warm start can recover
    /// the snapshot into a fresh manager.
    @Test func storeSnapshotsWithSSDPayloadsSurviveWarmStart() async throws {
        let (mgr, tieredStore, root) = makeSSDManager()
        defer { try? FileManager.default.removeItem(at: root) }

        let key = makeSSDKey()
        let promptTokens = Array(1...10)
        let snapshot = makeUniformSnapshot(offset: 5, type: .system)
        let payload = makeSSDPayload(bytes: 256, checkpointType: .system)

        mgr.storeSnapshots(
            promptTokens: promptTokens,
            capturedSnapshots: [snapshot],
            snapshotPayloads: [payload],
            partitionKey: key
        )
        await tieredStore.ssdStore!.flushAsync()

        let manifestURL = root.appendingPathComponent("manifest.json")
        #expect(FileManager.default.fileExists(atPath: manifestURL.path))
        let manifest = try JSONDecoder().decode(
            SnapshotManifest.self,
            from: Data(contentsOf: manifestURL)
        )
        #expect(manifest.partitions[key.partitionDigest] != nil)
        #expect(manifest.snapshots.count == 1)

        let config = SSDPrefixCacheConfig(
            enabled: true,
            rootURL: root,
            budgetBytes: 1_000_000,
            maxPendingBytes: 10_000_000
        )
        let restoredStore = TieredSnapshotStore(ssdConfig: config)
        let restoredMgr = PrefixCacheManager(
            memoryBudgetBytes: 10_000_000,
            tieredStore: restoredStore
        )
        try await restoredMgr.warmStart(modelFingerprint: key.modelFingerprint!)

        let lookup = restoredMgr.lookup(tokens: promptTokens, partitionKey: key)
        guard case .ssdHit(let ctx) = lookup.reason else {
            Issue.record("Expected .ssdHit after warm start, got \(lookup.reason)")
            return
        }
        #expect(ctx.storageRef.tokenOffset == snapshot.tokenOffset)
        #expect(restoredStore.ssdStore?.currentSSDBytesForTesting() == payload.totalBytes)
    }
}
