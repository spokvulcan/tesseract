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

    // MARK: - Snapshot Admission Path

    @Test func snapshotAdmissionPathValidatesOffsetsIntoFullPromptTokens() {
        let path = SnapshotAdmissionPath.validating(
            offset: 3,
            fullPromptTokenCount: 5
        )

        #expect(path?.offset == 3)
        #expect(
            SnapshotAdmissionPath.validating(
                offset: 0,
                fullPromptTokenCount: 5
            ) == nil
        )
        #expect(
            SnapshotAdmissionPath.validating(
                offset: 6,
                fullPromptTokenCount: 5
            ) == nil
        )
    }

    @Test func snapshotAdmissionPathValidatesLeafStoredTokenCount() {
        let path = SnapshotAdmissionPath.validatingLeaf(
            offset: 5,
            storedTokenCount: 5
        )

        #expect(path?.offset == 5)
        #expect(
            SnapshotAdmissionPath.validatingLeaf(
                offset: 4,
                storedTokenCount: 5
            ) == nil
        )
    }

    @Test func checkpointSnapshotAdmissionFiltersInvalidPathsAndKeepsMixedStorage() throws {
        let fullPromptTokens = [10, 20, 30, 40]
        let requestID = UUID()
        let ramOnlySnapshot = makeSnapshot(offset: 2, type: .system)
        let invalidSnapshot = makeSnapshot(offset: 5, type: .branchPoint)
        let ssdSnapshot = makeSnapshot(offset: 4, type: .branchPoint)
        let ssdPayload = makeSSDPayload(
            bytes: 256,
            checkpointType: .branchPoint
        )

        let admission = try #require(SnapshotAdmission.checkpoints(
            fullPromptTokens: fullPromptTokens,
            candidates: [
                SnapshotAdmission.CheckpointCandidate(
                    snapshot: ramOnlySnapshot,
                    storage: .ramOnly
                ),
                SnapshotAdmission.CheckpointCandidate(
                    snapshot: invalidSnapshot,
                    storage: .ramOnly
                ),
                SnapshotAdmission.CheckpointCandidate(
                    snapshot: ssdSnapshot,
                    storage: .ramAndSSD(ssdPayload)
                )
            ],
            partitionKey: defaultKey,
            requestID: requestID
        ))

        #expect(admission.fullPromptTokens == fullPromptTokens)
        #expect(admission.partitionKey == defaultKey)
        #expect(admission.requestID == requestID)
        let entries = Array(admission.entries)
        #expect(entries.count == 2)
        #expect(entries[0].path.offset == 2)
        #expect(entries[1].path.offset == 4)
        if case .ramOnly = entries[0].storage {
            // expected
        } else {
            #expect(Bool(false), "Expected first admission entry to be RAM-only")
        }
        if case .ramAndSSD(let payload) = entries[1].storage {
            #expect(payload.totalBytes == ssdPayload.totalBytes)
        } else {
            #expect(Bool(false), "Expected second admission entry to include SSD payload")
        }

        #expect(SnapshotAdmission.checkpoints(
            fullPromptTokens: fullPromptTokens,
            candidates: [
                SnapshotAdmission.CheckpointCandidate(
                    snapshot: invalidSnapshot,
                    storage: .ramOnly
                )
            ],
            partitionKey: defaultKey,
            requestID: requestID
        ) == nil)
    }

    @Test func admitLeafSnapshotAdmissionStoresRAMEntryThroughUnifiedInterface() throws {
        let manager = makeManager()
        let storedTokens = Array(1...5)
        let snapshot = makeSnapshot(offset: storedTokens.count, type: .leaf)
        let admission = try #require(SnapshotAdmission.leaf(
            storedTokens: storedTokens,
            snapshot: snapshot,
            storage: .ramOnly,
            partitionKey: defaultKey,
            requestID: UUID()
        ))

        let diagnostics = manager.admit(admission)

        #expect(diagnostics.evictions.isEmpty)
        #expect(diagnostics.supersededLeaves.isEmpty)
        #expect(diagnostics.stats.snapshotCount == 1)
        let lookup = manager.lookup(tokens: storedTokens, partitionKey: defaultKey)
        #expect(lookup.snapshotTokenOffset == storedTokens.count)
        #expect(SnapshotAdmission.leaf(
            storedTokens: storedTokens,
            snapshot: makeSnapshot(offset: storedTokens.count - 1, type: .leaf),
            storage: .ramOnly,
            partitionKey: defaultKey,
            requestID: UUID()
        ) == nil)
    }

    @Test func admitLeafSnapshotAdmissionSupersedesAncestorLeafThroughUnifiedInterface() throws {
        let manager = makeManager()
        let ancestorTokens = Array(1...5)
        let descendantTokens = Array(1...8)
        let ancestor = try #require(SnapshotAdmission.leaf(
            storedTokens: ancestorTokens,
            snapshot: makeSnapshot(offset: ancestorTokens.count, type: .leaf),
            storage: .ramOnly,
            partitionKey: defaultKey,
            requestID: nil
        ))
        let descendant = try #require(SnapshotAdmission.leaf(
            storedTokens: descendantTokens,
            snapshot: makeSnapshot(offset: descendantTokens.count, type: .leaf),
            storage: .ramOnly,
            partitionKey: defaultKey,
            requestID: nil
        ))

        manager.admit(ancestor)
        let diagnostics = manager.admit(descendant)

        #expect(diagnostics.supersededLeaves.count == 1)
        #expect(diagnostics.supersededLeaves[0].offset == ancestorTokens.count)
        #expect(diagnostics.stats.snapshotCount == 1)
        #expect(manager.lookup(tokens: descendantTokens, partitionKey: defaultKey).snapshotTokenOffset == descendantTokens.count)
        #expect(manager.lookup(tokens: ancestorTokens, partitionKey: defaultKey).snapshotTokenOffset == 0)
    }

    @Test func admitCheckpointSnapshotAdmissionStoresRAMEntriesThroughUnifiedInterface() throws {
        let manager = makeManager()
        let fullPromptTokens = Array(1...8)
        let snapshot = makeSnapshot(offset: 4, type: .system)
        let admission = try #require(SnapshotAdmission.checkpoints(
            fullPromptTokens: fullPromptTokens,
            candidates: [
                SnapshotAdmission.CheckpointCandidate(
                    snapshot: snapshot,
                    storage: .ramOnly
                )
            ],
            partitionKey: defaultKey,
            requestID: UUID()
        ))

        let diagnostics = manager.admit(admission)

        #expect(diagnostics.evictions.isEmpty)
        #expect(diagnostics.supersededLeaves.isEmpty)
        #expect(diagnostics.stats.snapshotCount == 1)
        let lookup = manager.lookup(tokens: fullPromptTokens + [9], partitionKey: defaultKey)
        #expect(lookup.snapshotTokenOffset == 4)
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
        mgr.admit(SnapshotAdmission.leaf(storedTokens: tokens, snapshot: snap, storage: .ramOnly, partitionKey: defaultKey)!)

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

        // Store system snapshot via Snapshot Admission (mid-prefill)
        let sysSnap = makeSnapshot(offset: 500, type: .system)
        mgr.admit(SnapshotAdmission.checkpoints(fullPromptTokens: userATokens, candidates: [.ramOnly(sysSnap)], partitionKey: defaultKey)!)

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
        mgr.admit(SnapshotAdmission.leaf(storedTokens: turn1, snapshot: leafSnap, storage: .ramOnly, partitionKey: defaultKey)!)

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
        mgr.admit(SnapshotAdmission.leaf(storedTokens: tokens, snapshot: snap, storage: .ramOnly, partitionKey: defaultKey)!)

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
        mgr.admit(SnapshotAdmission.leaf(storedTokens: tokens, snapshot: snap, storage: .ramOnly, partitionKey: defaultKey)!)

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
        mgr.admit(SnapshotAdmission.leaf(
            storedTokens: oldTokens,
            snapshot: makeUniformSnapshot(offset: oldTokens.count, type: .leaf),
            storage: .ramOnly,
            partitionKey: defaultKey
        )!)
        #expect(mgr.stats.snapshotCount == 1)

        // Newer leaf on an independent path. Same MLXArray shape → same
        // `memoryBytes` → exactly one eviction is needed to fit within
        // `snapBytes`.
        let newTokens = Array(20...29)
        let diagnostics = mgr.admit(SnapshotAdmission.leaf(
            storedTokens: newTokens,
            snapshot: makeUniformSnapshot(offset: newTokens.count, type: .leaf),
            storage: .ramOnly,
            partitionKey: defaultKey
        )!)

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
        mgr.admit(SnapshotAdmission.checkpoints(
            fullPromptTokens: sysTokens,
            candidates: [.ramOnly(makeUniformSnapshot(offset: 10, type: .system))],
            partitionKey: defaultKey
        )!)
        #expect(mgr.stats.snapshotCount == 1)

        // Newer `.leaf` on an independent path. The leaf is the only
        // eligible candidate; the system snapshot is protected.
        let leafTokens = Array(20...29)
        let diagnostics = mgr.admit(SnapshotAdmission.leaf(
            storedTokens: leafTokens,
            snapshot: makeUniformSnapshot(offset: leafTokens.count, type: .leaf),
            storage: .ramOnly,
            partitionKey: defaultKey
        )!)

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

    /// Only the newest leaf on a structural branch survives. Storing a
    /// descendant leaf must supersede older ancestor leaves immediately.
    @Test func descendantLeafSupersedesAncestorLeaf() {
        let mgr = makeManager()
        let ancestorTokens = Array(1...10)
        let descendantTokens = Array(1...15)

        mgr.admit(SnapshotAdmission.leaf(
            storedTokens: ancestorTokens,
            snapshot: makeUniformSnapshot(offset: ancestorTokens.count, type: .leaf),
            storage: .ramOnly,
            partitionKey: defaultKey
        )!)
        let diagnostics = mgr.admit(SnapshotAdmission.leaf(
            storedTokens: descendantTokens,
            snapshot: makeUniformSnapshot(offset: descendantTokens.count, type: .leaf),
            storage: .ramOnly,
            partitionKey: defaultKey
        )!)

        #expect(diagnostics.supersededLeaves.count == 1)
        #expect(diagnostics.supersededLeaves[0].offset == ancestorTokens.count)
        #expect(mgr.stats.snapshotCount == 1)

        let descendantResult = mgr.lookup(tokens: descendantTokens, partitionKey: defaultKey)
        #expect(descendantResult.snapshotTokenOffset == descendantTokens.count)
        let ancestorResult = mgr.lookup(tokens: ancestorTokens, partitionKey: defaultKey)
        #expect(ancestorResult.snapshot == nil)
    }

    /// A RAM-only descendant supersedes a state-5 ancestor leaf but
    /// **preserves** its committed SSD Snapshot Ref (ADR-0010): the new
    /// leaf has no SSD copy, so the ancestor stays the warm-start
    /// fallback and the next turn's extension base.
    @Test func descendantLeafSupersedesSSDBackedAncestorLeaf() {
        let tieredStore = TieredSnapshotStore(ssdConfig: nil)
        let mgr = PrefixCacheManager(
            memoryBudgetBytes: 100 * 1024 * 1024,
            tieredStore: tieredStore
        )
        let ancestorTokens = Array(1...10)
        let descendantTokens = Array(1...15)

        mgr.admit(SnapshotAdmission.leaf(
            storedTokens: ancestorTokens,
            snapshot: makeUniformSnapshot(offset: ancestorTokens.count, type: .leaf),
            storage: .ramOnly,
            partitionKey: defaultKey
        )!)
        let tree = tieredStore.tree(for: defaultKey)!
        let (ancestorNode, _) = tree.findBestSnapshot(
            tokens: ancestorTokens,
            updateAccess: false
        )!
        // Drive the ancestor to state 5 (committed ref, body dropped)
        // through the tree's sole-mutator seam.
        let ancestorRef = PrefixCacheTestFixtures.makeRef(
            tokenOffset: ancestorTokens.count
        )
        tree.admit(node: ancestorNode, ref: ancestorRef)
        tree.commitRef(node: ancestorNode, expectedID: ancestorRef.snapshotID)
        tree.dropBody(node: ancestorNode)
        #expect(ancestorNode.state.body == nil)
        #expect(ancestorNode.state.refID == ancestorRef.snapshotID)

        let diagnostics = mgr.admit(SnapshotAdmission.leaf(
            storedTokens: descendantTokens,
            snapshot: makeUniformSnapshot(offset: descendantTokens.count, type: .leaf),
            storage: .ramOnly,
            partitionKey: defaultKey
        )!)

        #expect(diagnostics.supersededLeaves.count == 1)
        #expect(diagnostics.supersededLeaves[0].offset == ancestorTokens.count)
        #expect(diagnostics.supersededLeaves[0].bodyDroppedSnapshotRefID == ancestorRef.snapshotID)
        #expect(diagnostics.supersededLeaves[0].mode == .preserved)
        // The ref (and so the state-5 node) survives — only a full SSD
        // re-write or an extension transfer may take the backing.
        #expect(ancestorNode.state.refID == ancestorRef.snapshotID)
        #expect(mgr.stats.snapshotCount == 1)

        let ancestorResult = mgr.lookup(tokens: ancestorTokens, partitionKey: defaultKey)
        #expect(ancestorResult.snapshot == nil)
        let descendantResult = mgr.lookup(tokens: descendantTokens, partitionKey: defaultKey)
        #expect(descendantResult.snapshotTokenOffset == descendantTokens.count)
    }

    /// When the nearest superseded ancestor collapses, the walk must still
    /// continue upward and clear shallower leaves on the same branch.
    @Test func descendantLeafSupersedesAllAncestorLeavesOnBranch() {
        let mgr = makeManager()
        let shallowTokens = Array(1...5)
        let midTokens = Array(1...10)
        let deepTokens = Array(1...15)

        mgr.restoreSnapshot(
            path: shallowTokens,
            snapshot: makeUniformSnapshot(offset: shallowTokens.count, type: .leaf),
            partitionKey: defaultKey,
            lastAccessTime: .now
        )
        mgr.restoreSnapshot(
            path: midTokens,
            snapshot: makeUniformSnapshot(offset: midTokens.count, type: .leaf),
            partitionKey: defaultKey,
            lastAccessTime: .now
        )

        let diagnostics = mgr.admit(SnapshotAdmission.leaf(
            storedTokens: deepTokens,
            snapshot: makeUniformSnapshot(offset: deepTokens.count, type: .leaf),
            storage: .ramOnly,
            partitionKey: defaultKey
        )!)

        #expect(diagnostics.supersededLeaves.count == 2)
        #expect(Set(diagnostics.supersededLeaves.map(\.offset)) == Set([shallowTokens.count, midTokens.count]))
        #expect(mgr.stats.snapshotCount == 1)

        let shallowResult = mgr.lookup(tokens: shallowTokens, partitionKey: defaultKey)
        #expect(shallowResult.snapshot == nil)
        let midResult = mgr.lookup(tokens: midTokens, partitionKey: defaultKey)
        #expect(midResult.snapshot == nil)
        let deepResult = mgr.lookup(tokens: deepTokens, partitionKey: defaultKey)
        #expect(deepResult.snapshotTokenOffset == deepTokens.count)
    }

    /// Leaf supersession is branch-local. Replacing an ancestor leaf on one
    /// branch must preserve sibling leaves and the stable-prefix `.system`
    /// snapshot they share.
    @Test func leafSupersessionPreservesSiblingBranchAndSystemSnapshot() {
        let mgr = makeManager()
        let stableTokens = Array(1...5)
        mgr.admit(SnapshotAdmission.checkpoints(
            fullPromptTokens: stableTokens,
            candidates: [.ramOnly(makeUniformSnapshot(offset: stableTokens.count, type: .system))],
            partitionKey: defaultKey
        )!)

        let ancestorTokens = Array(1...10)
        mgr.admit(SnapshotAdmission.leaf(
            storedTokens: ancestorTokens,
            snapshot: makeUniformSnapshot(offset: ancestorTokens.count, type: .leaf),
            storage: .ramOnly,
            partitionKey: defaultKey
        )!)

        let siblingTokens = Array(1...5) + Array(20...25)
        mgr.admit(SnapshotAdmission.leaf(
            storedTokens: siblingTokens,
            snapshot: makeUniformSnapshot(offset: siblingTokens.count, type: .leaf),
            storage: .ramOnly,
            partitionKey: defaultKey
        )!)

        let descendantTokens = Array(1...15)
        let diagnostics = mgr.admit(SnapshotAdmission.leaf(
            storedTokens: descendantTokens,
            snapshot: makeUniformSnapshot(offset: descendantTokens.count, type: .leaf),
            storage: .ramOnly,
            partitionKey: defaultKey
        )!)

        #expect(diagnostics.supersededLeaves.count == 1)
        #expect(diagnostics.supersededLeaves[0].offset == ancestorTokens.count)

        let stableResult = mgr.lookup(tokens: stableTokens + [99], partitionKey: defaultKey)
        #expect(stableResult.snapshotTokenOffset == stableTokens.count)
        let siblingResult = mgr.lookup(tokens: siblingTokens, partitionKey: defaultKey)
        #expect(siblingResult.snapshotTokenOffset == siblingTokens.count)
        let descendantResult = mgr.lookup(tokens: descendantTokens, partitionKey: defaultKey)
        #expect(descendantResult.snapshotTokenOffset == descendantTokens.count)
    }

    // MARK: - 8. memoryBudgetEnforced

    @Test func memoryBudgetEnforced() {
        let mgr = PrefixCacheManager(memoryBudgetBytes: 1)

        for i in 0..<10 {
            let tokens = Array((i * 100)..<((i + 1) * 100))
            let snap = makeSnapshot(offset: 100, type: .leaf)
            mgr.admit(SnapshotAdmission.leaf(storedTokens: tokens, snapshot: snap, storage: .ramOnly, partitionKey: defaultKey)!)
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
        mgr.admit(SnapshotAdmission.checkpoints(fullPromptTokens: tokens, candidates: [.ramOnly(sysSnap)], partitionKey: defaultKey)!)

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
        mgr.admit(SnapshotAdmission.leaf(storedTokens: tokens, snapshot: snap, storage: .ramOnly, partitionKey: defaultKey)!)

        // Query diverges at 80
        let query = Array(1...80) + [999, 998]
        let result = mgr.lookup(tokens: query, partitionKey: defaultKey)
        #expect(result.snapshotTokenOffset == 0)
    }

    // MARK: - 13. checkpointAdmissionFromPrefillAtCorrectOffsets

    @Test func checkpointAdmissionFromPrefillAtCorrectOffsets() {
        let mgr = makeManager()
        let tokens = Array(1...8000)

        let snap4k = makeSnapshot(offset: 4000, type: .system)
        mgr.admit(SnapshotAdmission.checkpoints(fullPromptTokens: tokens, candidates: [.ramOnly(snap4k)], partitionKey: defaultKey)!)

        let result = mgr.lookup(tokens: tokens, partitionKey: defaultKey)
        #expect(result.snapshotTokenOffset == 4000)
    }

    // MARK: - 14a. leafAdmissionUnderPostResponseTokens

    @Test func leafAdmissionUnderPostResponseTokens() {
        let mgr = makeManager()
        let promptTokens = Array(1...500)
        let storedTokens = promptTokens + Array(600...800) // prompt + response

        let leafSnap = makeSnapshot(offset: storedTokens.count, type: .leaf)
        mgr.admit(SnapshotAdmission.leaf(storedTokens: storedTokens, snapshot: leafSnap, storage: .ramOnly, partitionKey: defaultKey)!)

        // Lookup with storedTokens hits the leaf
        let result = mgr.lookup(tokens: storedTokens, partitionKey: defaultKey)
        #expect(result.snapshotTokenOffset == storedTokens.count)
    }

    // MARK: - 14b. leafSnapshotOffsetMatchesStoredTokenCount

    @Test func leafSnapshotOffsetMatchesStoredTokenCount() {
        let mgr = makeManager()
        let storedTokens = Array(1...300)
        let leafSnap = makeSnapshot(offset: storedTokens.count, type: .leaf)
        mgr.admit(SnapshotAdmission.leaf(storedTokens: storedTokens, snapshot: leafSnap, storage: .ramOnly, partitionKey: defaultKey)!)

        let result = mgr.lookup(tokens: storedTokens, partitionKey: defaultKey)
        #expect(result.snapshot?.tokenOffset == storedTokens.count)
    }

    // MARK: - 14c. nextRequestHitsLeafViaExtendedPrefix

    @Test func nextRequestHitsLeafViaExtendedPrefix() {
        let mgr = makeManager()
        // Store leaf for sys+user1+asst1
        let turn1Stored = Array(1...300)
        let leafSnap = makeSnapshot(offset: 300, type: .leaf)
        mgr.admit(SnapshotAdmission.leaf(storedTokens: turn1Stored, snapshot: leafSnap, storage: .ramOnly, partitionKey: defaultKey)!)

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
        mgr.admit(SnapshotAdmission.leaf(storedTokens: tokens, snapshot: snap, storage: .ramOnly, partitionKey: defaultKey)!)

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
        mgr.admit(SnapshotAdmission.leaf(storedTokens: tokens, snapshot: snap, storage: .ramOnly, partitionKey: key8)!)

        let result = mgr.lookup(tokens: tokens, partitionKey: keyNil)
        #expect(result.snapshot == nil)
    }

    // MARK: - 16. sameKvBitsShared

    @Test func sameKvBitsShared() {
        let mgr = makeManager()
        let tokens = Array(1...100)
        let key8 = CachePartitionKey(modelID: "m", kvBits: 8, kvGroupSize: 64)

        let snap = makeSnapshot(offset: 100, type: .leaf)
        mgr.admit(SnapshotAdmission.leaf(storedTokens: tokens, snapshot: snap, storage: .ramOnly, partitionKey: key8)!)

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
        mgr.admit(SnapshotAdmission.leaf(storedTokens: tokens, snapshot: snap, storage: .ramOnly, partitionKey: keyA)!)

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
        mgr.admit(SnapshotAdmission.leaf(storedTokens: Array(1...50), snapshot: snapA, storage: .ramOnly, partitionKey: keyA)!)
        let snapB = makeSnapshot(offset: 50, type: .leaf)
        mgr.admit(SnapshotAdmission.leaf(storedTokens: Array(100...149), snapshot: snapB, storage: .ramOnly, partitionKey: keyB)!)

        // Budget is 1 byte — eviction must cross partitions
        #expect(mgr.totalSnapshotBytes <= 1)
    }

    // MARK: - Leaf Snapshot Admission validation

    @Test func leafSnapshotAdmissionRejectsMismatchedOffsetAndAdmitsValidPath() throws {
        let tokens = Array(1...100)
        let snap = makeSnapshot(offset: 50, type: .leaf)

        let invalidAdmission = SnapshotAdmission.leaf(
            storedTokens: tokens,
            snapshot: snap,
            storage: .ramOnly,
            partitionKey: defaultKey,
            requestID: nil
        )

        #expect(invalidAdmission == nil)
        let mgr = makeManager()
        let validSnap = makeSnapshot(offset: tokens.count, type: .leaf)
        let validAdmission = try #require(SnapshotAdmission.leaf(
            storedTokens: tokens,
            snapshot: validSnap,
            storage: .ramOnly,
            partitionKey: defaultKey,
            requestID: nil
        ))

        mgr.admit(validAdmission)

        let exact = mgr.lookup(tokens: tokens, partitionKey: defaultKey)
        #expect(exact.snapshotTokenOffset == tokens.count)
        let prefixOnly = mgr.lookup(tokens: Array(1...50), partitionKey: defaultKey)
        #expect(prefixOnly.snapshot == nil)
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
        mgr.admit(SnapshotAdmission.leaf(
            storedTokens: pathA,
            snapshot: makeSnapshot(offset: pathA.count, type: .leaf),
            storage: .ramOnly,
            partitionKey: defaultKey
        )!)

        // Store snapshotB (newer — stored second)
        let pathB = Array(20...29)
        mgr.admit(SnapshotAdmission.leaf(
            storedTokens: pathB,
            snapshot: makeSnapshot(offset: pathB.count, type: .leaf),
            storage: .ramOnly,
            partitionKey: defaultKey
        )!)

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
        mgr.admit(SnapshotAdmission.checkpoints(
            fullPromptTokens: pathA,
            candidates: [.ramOnly(makeSnapshot(offset: 10, type: .system))],
            partitionKey: defaultKey
        )!)
        mgr.admit(SnapshotAdmission.leaf(
            storedTokens: pathB,
            snapshot: makeSnapshot(offset: pathB.count, type: .leaf),
            storage: .ramOnly,
            partitionKey: defaultKey
        )!)
        mgr.admit(SnapshotAdmission.checkpoints(
            fullPromptTokens: pathC,
            candidates: [.ramOnly(makeSnapshot(offset: 10, type: .system))],
            partitionKey: defaultKey
        )!)

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
        mgr.admit(SnapshotAdmission.checkpoints(fullPromptTokens: tokens, candidates: [.ramOnly(leafSnap)], partitionKey: defaultKey)!)

        // Planning should still request .system at 500 — the leaf doesn't count
        let plan = mgr.planCheckpoints(tokens: tokens, stablePrefixOffset: 500, partitionKey: defaultKey)
        #expect(plan.count == 1)
        #expect(plan[0].type == .system)
    }

    // MARK: - Speculative branch-point candidates

    @Test func divergenceInsideCompressedEdgeCreatesCandidate() {
        let mgr = makeManager()
        let stored = [1, 2, 3, 4]
        mgr.admit(SnapshotAdmission.leaf(
            storedTokens: stored,
            snapshot: makeSnapshot(offset: stored.count, type: .leaf),
            storage: .ramOnly,
            partitionKey: defaultKey
        )!)

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
        mgr.admit(SnapshotAdmission.leaf(
            storedTokens: [1, 2, 3],
            snapshot: makeSnapshot(offset: 3, type: .leaf),
            storage: .ramOnly,
            partitionKey: defaultKey
        )!)

        let plan = mgr.planCheckpoints(
            tokens: [1, 2, 3, 4, 5],
            stablePrefixOffset: nil,
            partitionKey: defaultKey
        )

        #expect(!plan.contains(where: { $0.type == .branchPoint }))
    }

    @Test func nodeBoundaryDivergenceDoesNotCreateIntermediateCheckpoint() {
        let mgr = makeManager()
        // A materialized snapshot-bearing boundary at [1,2] should keep the
        // divergence on [1,2,9,10] at the node boundary, not mid-edge. Use a
        // `.system` snapshot here because descendant leaf stores supersede
        // ancestor leaves under the newest-per-branch policy.
        mgr.admit(SnapshotAdmission.checkpoints(
            fullPromptTokens: [1, 2],
            candidates: [.ramOnly(makeSnapshot(offset: 2, type: .system))],
            partitionKey: defaultKey
        )!)
        mgr.admit(SnapshotAdmission.leaf(
            storedTokens: [1, 2, 3, 4],
            snapshot: makeSnapshot(offset: 4, type: .leaf),
            storage: .ramOnly,
            partitionKey: defaultKey
        )!)

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
        mgr.admit(SnapshotAdmission.checkpoints(
            fullPromptTokens: [1, 2, 3, 4],
            candidates: [.ramOnly(makeSnapshot(offset: 2, type: .system))],
            partitionKey: defaultKey
        )!)

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
        mgr.admit(SnapshotAdmission.leaf(
            storedTokens: [1, 2, 3, 4, 5],
            snapshot: makeSnapshot(offset: 5, type: .leaf),
            storage: .ramOnly,
            partitionKey: defaultKey
        )!)
        mgr.admit(SnapshotAdmission.leaf(
            storedTokens: [1, 2, 8, 9, 10],
            snapshot: makeSnapshot(offset: 5, type: .leaf),
            storage: .ramOnly,
            partitionKey: defaultKey
        )!)

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
        mgr.admit(SnapshotAdmission.leaf(
            storedTokens: Array(1...50),
            snapshot: makeSnapshot(offset: 50, type: .leaf),
            storage: .ramOnly,
            partitionKey: defaultKey
        )!)

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
        mgr.admit(SnapshotAdmission.leaf(
            storedTokens: Array(1...50),
            snapshot: makeSnapshot(offset: 50, type: .leaf),
            storage: .ramOnly,
            partitionKey: defaultKey
        )!)

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
        mgr.admit(SnapshotAdmission.checkpoints(
            fullPromptTokens: exactPath,
            candidates: [.ramOnly(makeSnapshot(offset: 200, type: .system))],
            partitionKey: defaultKey
        )!)

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
        mgr.admit(SnapshotAdmission.checkpoints(
            fullPromptTokens: longPath,
            candidates: [.ramOnly(makeSnapshot(offset: 200, type: .system))],
            partitionKey: defaultKey
        )!)

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
        mgr.admit(SnapshotAdmission.checkpoints(
            fullPromptTokens: matchedPath,
            candidates: [.ramOnly(makeSnapshot(offset: 100, type: .system))],
            partitionKey: defaultKey
        )!)

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
        mgr.admit(SnapshotAdmission.checkpoints(
            fullPromptTokens: matchedPath,
            candidates: [.ramOnly(makeSnapshot(offset: 300, type: .system))],
            partitionKey: defaultKey
        )!)

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

    @Test func planCheckpointsWithAlignToIncludesAlignmentCheckpoint() {
        let mgr = makeManager()
        let matchedPath = Array(1...500)
        mgr.admit(SnapshotAdmission.checkpoints(
            fullPromptTokens: matchedPath,
            candidates: [.ramOnly(makeSnapshot(offset: 100, type: .system))],
            partitionKey: defaultKey
        )!)

        let request = matchedPath + [999]
        // Drive the live prefill path: resolve first (Snapshot Resolution does
        // the lookup), then plan against the settled tree aligned to that hit.
        let lookup = mgr.lookup(tokens: request, partitionKey: defaultKey)
        #expect(lookup.snapshotTokenOffset == 100)
        #expect(lookup.sharedPrefixLength == 500)

        let plan = mgr.planCheckpoints(
            tokens: request,
            stablePrefixOffset: nil,
            partitionKey: defaultKey,
            alignTo: lookup
        )

        #expect(plan.contains { $0.offset == 500 && $0.type == .branchPoint })

        let suffixPlan = plan.filter { $0.offset > lookup.snapshotTokenOffset }
        #expect(suffixPlan.contains { $0.offset == 500 && $0.type == .branchPoint })
    }

    // MARK: - Global partitioning

    /// OpenCode session IDs no longer affect prefix-cache routing. Two
    /// equivalent model-config keys share the same radix tree, so an
    /// identical first-turn prompt may hit the stored leaf immediately.
    @Test func equivalentKeysShareLeafAcrossSessions() {
        let mgr = makeManager()
        let keyA = CachePartitionKey(modelID: "test-model", kvBits: nil, kvGroupSize: 64)
        let keyB = CachePartitionKey(modelID: "test-model", kvBits: nil, kvGroupSize: 64)
        let tokens = Array(1...100)

        mgr.admit(SnapshotAdmission.leaf(
            storedTokens: tokens,
            snapshot: makeSnapshot(offset: tokens.count, type: .leaf),
            storage: .ramOnly,
            partitionKey: keyA
        )!)

        #expect(mgr.stats.partitionCount == 1)
        #expect(mgr.stats.snapshotCount == 1)

        let result = mgr.lookup(tokens: tokens, partitionKey: keyB)
        #expect(result.snapshotTokenOffset == tokens.count)
    }

    /// Cross-session sharing is still bounded by the global partition
    /// identity. A different model fingerprint must map to a different
    /// partition and miss cleanly.
    @Test func differentFingerprintsRemainIsolated() {
        let mgr = makeManager()
        let keyA = CachePartitionKey(
            modelID: "m",
            kvBits: nil,
            kvGroupSize: 64,
            modelFingerprint: String(repeating: "a", count: 64)
        )
        let keyB = CachePartitionKey(
            modelID: "m",
            kvBits: nil,
            kvGroupSize: 64,
            modelFingerprint: String(repeating: "b", count: 64)
        )

        let tokens = Array(1...50)
        mgr.admit(SnapshotAdmission.leaf(
            storedTokens: tokens,
            snapshot: makeSnapshot(offset: tokens.count, type: .leaf),
            storage: .ramOnly,
            partitionKey: keyA
        )!)

        let result = mgr.lookup(tokens: tokens, partitionKey: keyB)
        #expect(result.snapshot == nil)
    }

    /// Stable-prefix reuse is global too. After storing a `.system`
    /// snapshot for one request shape, a different request that shares the
    /// same prefix must hit that checkpoint even when the user suffix
    /// diverges.
    @Test func stablePrefixSharedAcrossEquivalentKeys() {
        let mgr = makeManager()
        let keyA = CachePartitionKey(modelID: "test-model", kvBits: nil, kvGroupSize: 64)
        let keyB = CachePartitionKey(modelID: "test-model", kvBits: nil, kvGroupSize: 64)

        let original = [1, 2, 3, 10, 11]
        let differentUser = [1, 2, 3, 20, 21]
        mgr.admit(SnapshotAdmission.checkpoints(
            fullPromptTokens: original,
            candidates: [.ramOnly(makeSnapshot(offset: 3, type: .system))],
            partitionKey: keyA
        )!)

        let result = mgr.lookup(tokens: differentUser, partitionKey: keyB)
        #expect(result.snapshotTokenOffset == 3)
        #expect(result.sharedPrefixLength == 3)
        if case .hit(let offset, _, let type) = result.reason {
            #expect(offset == 3)
            #expect(type == .system)
        } else {
            #expect(Bool(false), "Expected stable-prefix hit, got \(result.reason)")
        }
    }

    /// When the writing partition's eligible set is exhausted (all
    /// remaining snapshots are type-protected `.system`), eviction
    /// spills over to other model partitions via the global path.
    @Test func preferredPartitionSpillsToGlobalWhenExhausted() {
        let snapBytes = makeUniformSnapshot(offset: 10, type: .leaf).memoryBytes
        // Budget fits 1 snapshot.
        let mgr = PrefixCacheManager(memoryBudgetBytes: snapBytes)

        let idleKey = CachePartitionKey(
            modelID: "idle-model", kvBits: nil, kvGroupSize: 64
        )
        let writingKey = CachePartitionKey(
            modelID: "writing-model", kvBits: nil, kvGroupSize: 64
        )

        // Idle partition: one evictable leaf.
        let idleTokens = Array(1...10)
        mgr.admit(SnapshotAdmission.leaf(
            storedTokens: idleTokens,
            snapshot: makeUniformSnapshot(offset: idleTokens.count, type: .leaf),
            storage: .ramOnly,
            partitionKey: idleKey
        )!)
        #expect(mgr.stats.snapshotCount == 1)

        // Writing partition: two `.system` snapshots (type-protected,
        // ineligible in the preferred-utility path) on distinct linear
        // paths. The second store triggers eviction; the writing
        // partition has no eligible utility candidates, so eviction
        // spills over to the idle partition's leaf.
        let writingTokens1 = Array(2000...2009)
        mgr.admit(SnapshotAdmission.checkpoints(
            fullPromptTokens: writingTokens1,
            candidates: [.ramOnly(makeUniformSnapshot(offset: 10, type: .system))],
            partitionKey: writingKey
        )!)
        let writingTokens2 = Array(3000...3009)
        mgr.admit(SnapshotAdmission.checkpoints(
            fullPromptTokens: writingTokens2,
            candidates: [.ramOnly(makeUniformSnapshot(offset: 10, type: .system))],
            partitionKey: writingKey
        )!)

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
    /// absent + committed ref). `tree.dropBody` settles in place (the
    /// surviving ref makes `canEvictNode` false), leaving the node
    /// attached as an SSD-backed lookup target.
    /// Regression check for the orphan-leak bug flagged on 2026-04-14.
    @Test func evictionBodyDropsStateFourNodeToStateFive() {
        let snapBytes = makeUniformSnapshot(offset: 10, type: .leaf).memoryBytes
        let tieredStore = TieredSnapshotStore(ssdConfig: nil)
        let mgr = PrefixCacheManager(
            memoryBudgetBytes: snapBytes,
            tieredStore: tieredStore
        )

        let firstTokens = Array(1...10)
        mgr.admit(SnapshotAdmission.leaf(
            storedTokens: firstTokens,
            snapshot: makeUniformSnapshot(offset: firstTokens.count, type: .leaf),
            storage: .ramOnly,
            partitionKey: defaultKey
        )!)
        let tree = tieredStore.tree(for: defaultKey)!
        let (firstNode, _) = tree.findBestSnapshot(
            tokens: firstTokens, updateAccess: false
        )!
        let ref = PrefixCacheTestFixtures.makeRef(tokenOffset: firstTokens.count)
        tree.admit(node: firstNode, ref: ref)
        tree.commitRef(node: firstNode, expectedID: ref.snapshotID)

        let secondTokens = Array(20...29)
        let diag = mgr.admit(SnapshotAdmission.leaf(
            storedTokens: secondTokens,
            snapshot: makeUniformSnapshot(offset: secondTokens.count, type: .leaf),
            storage: .ramOnly,
            partitionKey: defaultKey
        )!)

        #expect(diag.evictions.count == 1)
        #expect(diag.evictions[0].checkpointType == .leaf)
        #expect(firstNode.state.body == nil)
        #expect(firstNode.state.ref != nil)
        #expect(firstNode.state.committed)
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
        mgr.admit(SnapshotAdmission.leaf(
            storedTokens: firstTokens,
            snapshot: makeUniformSnapshot(offset: firstTokens.count, type: .leaf),
            storage: .ramOnly,
            partitionKey: defaultKey
        )!)
        let tree = tieredStore.tree(for: defaultKey)!
        let (firstNode, _) = tree.findBestSnapshot(
            tokens: firstTokens, updateAccess: false
        )!
        // State 2 (body + pending ref): admit a ref but do not commit it.
        let ref = PrefixCacheTestFixtures.makeRef(tokenOffset: firstTokens.count)
        tree.admit(node: firstNode, ref: ref)

        let secondTokens = Array(20...29)
        mgr.admit(SnapshotAdmission.leaf(
            storedTokens: secondTokens,
            snapshot: makeUniformSnapshot(offset: secondTokens.count, type: .leaf),
            storage: .ramOnly,
            partitionKey: defaultKey
        )!)

        #expect(firstNode.state.body == nil)
        #expect(firstNode.state.ref != nil)
        #expect(!firstNode.state.committed)
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
        mgr.admit(SnapshotAdmission.leaf(
            storedTokens: firstTokens,
            snapshot: makeUniformSnapshot(offset: firstTokens.count, type: .leaf),
            storage: .ramOnly,
            partitionKey: defaultKey
        )!)
        let tree = tieredStore.tree(for: defaultKey)!
        let (firstNode, _) = tree.findBestSnapshot(
            tokens: firstTokens, updateAccess: false
        )!
        #expect(firstNode.state.ref == nil)
        let initialNodeCount = tree.nodeCount

        let secondTokens = Array(20...29)
        mgr.admit(SnapshotAdmission.leaf(
            storedTokens: secondTokens,
            snapshot: makeUniformSnapshot(offset: secondTokens.count, type: .leaf),
            storage: .ramOnly,
            partitionKey: defaultKey
        )!)

        #expect(firstNode.state.body == nil)
        #expect(firstNode.parent == nil)
        #expect(tree.nodeCount == initialNodeCount)
        let firstResult = mgr.lookup(tokens: firstTokens, partitionKey: defaultKey)
        #expect(firstResult.snapshot == nil)
    }

    /// `.system` type protection still applies even when the system
    /// node carries a committed Snapshot Ref. The `.leaf` victim is
    /// hard-deleted; the `.system` body + ref are preserved.
    @Test func systemTypeProtectionHoldsWithSnapshotRef() {
        let snapBytes = makeUniformSnapshot(offset: 10, type: .leaf).memoryBytes
        let tieredStore = TieredSnapshotStore(ssdConfig: nil)
        let mgr = PrefixCacheManager(
            memoryBudgetBytes: snapBytes,
            tieredStore: tieredStore
        )

        let sysTokens = Array(1...10)
        mgr.admit(SnapshotAdmission.checkpoints(
            fullPromptTokens: sysTokens,
            candidates: [.ramOnly(makeUniformSnapshot(offset: 10, type: .system))],
            partitionKey: defaultKey
        )!)
        let tree = tieredStore.tree(for: defaultKey)!
        let (sysNode, _) = tree.findBestSnapshot(
            tokens: sysTokens, updateAccess: false
        )!
        let ref = PrefixCacheTestFixtures.makeRef(
            type: .system,
            tokenOffset: sysTokens.count
        )
        tree.admit(node: sysNode, ref: ref)
        tree.commitRef(node: sysNode, expectedID: ref.snapshotID)

        let leafTokens = Array(20...29)
        mgr.admit(SnapshotAdmission.leaf(
            storedTokens: leafTokens,
            snapshot: makeUniformSnapshot(offset: leafTokens.count, type: .leaf),
            storage: .ramOnly,
            partitionKey: defaultKey
        )!)

        #expect(sysNode.state.body != nil)
        #expect(sysNode.state.ref != nil)
        #expect(sysNode.state.committed)
        let leafResult = mgr.lookup(tokens: leafTokens, partitionKey: defaultKey)
        #expect(leafResult.snapshot == nil)
    }

    // MARK: - SSD hydration: lookup / promote / clearCommittedSnapshotRefAfterHydrationFailure

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
        let ref = PrefixCacheTestFixtures.makeRef(
            type: .leaf,
            tokenOffset: tokens.count
        )
        // Node has a committed ref but no body — state 5.
        tree.restoreCommittedRef(node: node, ref: ref)
        #expect(node.state.body == nil)

        let result = mgr.lookup(tokens: tokens, partitionKey: defaultKey)
        guard case .ssdHit(let ctx) = result.reason else {
            #expect(Bool(false), "Expected .ssdHit, got \(result.reason)")
            return
        }
        #expect(ctx.snapshotRef.snapshotID == ref.snapshotID)
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
        // State 3 (pending ref, body dropped): store a body, admit a
        // pending ref, then drop the body.
        let node = tree.insertPath(tokens: tokens)
        tree.storeSnapshot(makeUniformSnapshot(offset: tokens.count, type: .leaf), on: node)
        tree.admit(node: node, ref: PrefixCacheTestFixtures.makeRef(tokenOffset: tokens.count))
        tree.dropBody(node: node)

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
        tree.restoreCommittedRef(
            node: node,
            ref: PrefixCacheTestFixtures.makeRef(tokenOffset: tokens.count)
        )

        let hydrated = makeUniformSnapshot(offset: tokens.count, type: .leaf)
        mgr.promote(node: node, snapshot: hydrated, partitionKey: defaultKey)

        #expect(node.state.body != nil)
        #expect(node.state.ref != nil)  // ref preserved
        let result = mgr.lookup(tokens: tokens, partitionKey: defaultKey)
        if case .hit = result.reason {
            // expected
        } else {
            #expect(Bool(false), "Expected .hit after promote, got \(result.reason)")
        }
    }

    /// `clearCommittedSnapshotRefAfterHydrationFailure` on a state-5 leaf with no body removes the
    /// node from the tree entirely. Subsequent lookups miss via the
    /// normal `missNoSnapshotInPrefix` path.
    @Test func clearCommittedSnapshotRefAfterHydrationFailureRemovesLeafWithNoBody() {
        let (mgr, tieredStore, root) = makeSSDManager()
        defer { try? FileManager.default.removeItem(at: root) }

        let tokens = Array(1...10)
        let tree = tieredStore.getOrCreateTree(for: defaultKey)
        let node = tree.insertPath(tokens: tokens)
        tree.restoreCommittedRef(
            node: node,
            ref: PrefixCacheTestFixtures.makeRef(tokenOffset: tokens.count)
        )
        let nodeCountBefore = tree.nodeCount

        mgr.clearCommittedSnapshotRefAfterHydrationFailure(node: node, partitionKey: defaultKey)

        #expect(node.state.ref == nil)
        #expect(node.parent == nil)  // detached
        #expect(tree.nodeCount < nodeCountBefore)
        let result = mgr.lookup(tokens: tokens, partitionKey: defaultKey)
        #expect(result.snapshot == nil)
    }

    /// `hydrate` is a forgiving edge: it completes a lookup that captured
    /// the node before an off-main `loadSync`. If the node is no longer
    /// `ssdOnly` at the promote hop, hydration is a logged no-op (newer
    /// state wins) instead of a `precondition` that would abort the whole
    /// server. Pre-fix this trapped.
    @Test func hydrateOnNonSSDOnlyNodeIsForgivingNoOp() {
        let (_, tieredStore, root) = makeSSDManager()
        defer { try? FileManager.default.removeItem(at: root) }

        let tokens = Array(1...10)
        let tree = tieredStore.getOrCreateTree(for: defaultKey)
        let node = tree.insertPath(tokens: tokens)  // state empty, not ssdOnly

        let effect = tree.hydrate(
            node: node,
            body: makeUniformSnapshot(offset: tokens.count, type: .leaf)
        )

        #expect(effect == .ignored(.notResident))
        #expect(node.state.isEmpty)  // unchanged — body not attached
    }

    /// `clearCommittedSnapshotRefAfterHydrationFailure` is forgiving for
    /// the same reason: a node that left the committed/`ssdOnly` states
    /// before the failure hop is a logged no-op, not a process abort.
    @Test func clearCommittedSnapshotRefOnNonCommittedNodeIsForgivingNoOp() {
        let (_, tieredStore, root) = makeSSDManager()
        defer { try? FileManager.default.removeItem(at: root) }

        let tokens = Array(1...10)
        let tree = tieredStore.getOrCreateTree(for: defaultKey)
        let node = tree.insertPath(tokens: tokens)
        // ramOnly (state 1): a RAM body, no ref — not committed/ssdOnly.
        tree.storeSnapshot(makeUniformSnapshot(offset: tokens.count, type: .leaf), on: node)

        let effect = tree.clearCommittedSnapshotRefAfterHydrationFailure(node: node)

        #expect(effect == .ignored(.notResident))
        #expect(node.state.body != nil)  // body untouched
    }

    /// `planCheckpoints` must treat a checkpoint already persisted to SSD
    /// whose RAM body was dropped (`ssdOnly`, body-less) as already-stored.
    /// Otherwise every same-prefix request after a warm start (or a body
    /// eviction) re-plans, re-extracts, and re-admits the identical system
    /// snapshot — `admitSnapshot` then supersedes the resident ref,
    /// deleting and rewriting byte-identical content.
    @Test func planCheckpointsSkipsSystemAlreadyPersistedAsSSDOnly() {
        let (mgr, tieredStore, root) = makeSSDManager()
        defer { try? FileManager.default.removeItem(at: root) }

        let tokens = Array(1...20)
        let stableOffset = 10
        let tree = tieredStore.getOrCreateTree(for: defaultKey)

        // Drive the stable-prefix node to `ssdOnly`: store a system body,
        // admit + commit a ref, then drop the body — exactly the state a
        // warm-restored or RAM-evicted system checkpoint sits in.
        let node = tree.insertPath(tokens: Array(tokens[0..<stableOffset]))
        tree.storeSnapshot(
            makeUniformSnapshot(offset: stableOffset, type: .system), on: node
        )
        let ref = PrefixCacheTestFixtures.makeRef(type: .system, tokenOffset: stableOffset)
        tree.admit(node: node, ref: ref)
        tree.commitRef(node: node, expectedID: ref.snapshotID)
        tree.dropBody(node: node)  // committed → ssdOnly (body-less)
        #expect(node.state.body == nil)
        #expect(node.state.label == "ssdOnly")

        let plan = mgr.planCheckpoints(
            tokens: tokens,
            stablePrefixOffset: stableOffset,
            partitionKey: defaultKey
        )

        // The system checkpoint at `stableOffset` is already on SSD; it
        // must NOT be re-planned.
        #expect(
            !plan.contains { $0.offset == stableOffset && $0.type == .system }
        )
    }

    /// The invariant the LLMActor hydration-failure replan relies on: an
    /// `ssdOnly` system checkpoint suppresses re-planning while present,
    /// but once a failed hydration clears the node from the tree,
    /// `planCheckpoints` re-plans the (now-absent) checkpoint. Without
    /// LLMActor's post-failure replan, the suppressed-then-cleared
    /// checkpoint would go un-recaptured for a full request cycle.
    @Test func planReplansSystemAfterSSDOnlyNodeClearedOnHydrationFailure() {
        let (mgr, tieredStore, root) = makeSSDManager()
        defer { try? FileManager.default.removeItem(at: root) }

        let tokens = Array(1...20)
        let stableOffset = 10
        let tree = tieredStore.getOrCreateTree(for: defaultKey)
        let node = tree.insertPath(tokens: Array(tokens[0..<stableOffset]))
        tree.restoreCommittedRef(
            node: node,
            ref: PrefixCacheTestFixtures.makeRef(type: .system, tokenOffset: stableOffset)
        )

        // While the ssdOnly node is present, the system checkpoint is
        // treated as already stored and suppressed.
        let planBefore = mgr.planCheckpoints(
            tokens: tokens, stablePrefixOffset: stableOffset, partitionKey: defaultKey
        )
        #expect(!planBefore.contains { $0.offset == stableOffset && $0.type == .system })

        // A failed hydration clears the node (forgiving clear removes the
        // now-bodyless leaf).
        mgr.clearCommittedSnapshotRefAfterHydrationFailure(node: node, partitionKey: defaultKey)

        // The lost checkpoint is now re-planned.
        let planAfter = mgr.planCheckpoints(
            tokens: tokens, stablePrefixOffset: stableOffset, partitionKey: defaultKey
        )
        #expect(planAfter.contains { $0.offset == stableOffset && $0.type == .system })
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
        mgr.admit(SnapshotAdmission.leaf(
            storedTokens: tokens,
            snapshot: makeUniformSnapshot(offset: tokens.count, type: .leaf),
            storage: .ramOnly,
            partitionKey: defaultKey
        )!)
        let tree = tieredStore.tree(for: defaultKey)!
        let (node, _) = tree.findBestSnapshot(tokens: tokens, updateAccess: false)!

        // Seed the SSD manifest with a descriptor so `recordHit` has
        // a target. Partition must be registered first; the manager's
        // warmStart normally drives this, but we mint the config
        // directly here.
        let ssdStore = tieredStore.ssdStoreForTesting!
        let meta = PartitionMeta(
            modelID: "test-model",
            modelFingerprint: String(repeating: "a", count: 64),
            kvBits: nil,
            kvGroupSize: 64,
            createdAt: 0,
            schemaVersion: SnapshotManifestSchema.currentVersion
        )
        let digest = defaultKey.partitionDigest
        ssdStore.registerPartition(meta, digest: digest)

        let ref = SnapshotRef(
            snapshotID: UUID().uuidString,
            partitionDigest: digest,
            tokenOffset: tokens.count,
            checkpointType: .leaf,
            bytesOnDisk: 1024
        )
        // Drive the node to state 4 (body + committed ref) through the
        // tree's sole-mutator seam: admit a pending ref, then commit it.
        tree.admit(node: node, ref: ref)
        tree.commitRef(node: node, expectedID: ref.snapshotID)

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
    @Test func checkpointAdmissionWithSSDPayloadsSurvivesWarmStart() async throws {
        let (mgr, tieredStore, root) = makeSSDManager()
        defer { try? FileManager.default.removeItem(at: root) }

        let key = makeSSDKey()
        let promptTokens = Array(1...10)
        let snapshot = makeUniformSnapshot(offset: 5, type: .system)
        let payload = makeSSDPayload(bytes: 256, checkpointType: .system)

        mgr.admit(SnapshotAdmission.checkpoints(
            fullPromptTokens: promptTokens,
            candidates: [.ramAndSSD(snapshot, payload: payload)],
            partitionKey: key
        )!)
        await tieredStore.ssdStoreForTesting!.flushAsync()

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
        #expect(ctx.snapshotRef.tokenOffset == snapshot.tokenOffset)
        #expect(restoredStore.ssdStoreForTesting?.currentSSDBytesForTesting() == payload.totalBytes)
    }

    /// Warm-start tolerates two persisted descriptors that resolve to the
    /// same `pathFromRoot` (corrupted-manifest rebuild, a stale entry left
    /// after a crash before debounced persist, etc.). The first descriptor
    /// wins; the second is dropped — last-wins would leak the first SSD
    /// file because it would be unreachable from the live tree. The
    /// dropped descriptor's SSD backing is reclaimed (not leaked): without
    /// the reclaim, its bytes + manifest entry linger, never hittable,
    /// inflating the SSD budget.
    @Test func restoreSnapshotRefIsIdempotentOnPathCollisionAndReclaimsLoser() {
        let (mgr, tieredStore, root) = makeSSDManager()
        defer { try? FileManager.default.removeItem(at: root) }

        let path = Array(1...8)
        let digest = defaultKey.partitionDigest
        let ssdStore = tieredStore.ssdStoreForTesting!

        // Register the partition and seed BOTH descriptors into the
        // manifest + SSD byte budget, mirroring `commitRestoredManifest`
        // for two on-disk files that compress to one `pathFromRoot`.
        let meta = PartitionMeta(
            modelID: "test-model",
            modelFingerprint: String(repeating: "a", count: 64),
            kvBits: nil,
            kvGroupSize: 64,
            createdAt: 0,
            schemaVersion: SnapshotManifestSchema.currentVersion
        )
        ssdStore.registerPartition(meta, digest: digest)

        func seed(bytes: Int) -> SnapshotRef {
            let id = UUID().uuidString
            ssdStore.seedDescriptorForTesting(PersistedSnapshotDescriptor(
                snapshotID: id,
                partitionDigest: digest,
                pathFromRoot: path,
                tokenOffset: path.count,
                checkpointType: HybridCacheSnapshot.CheckpointType.leaf.wireString,
                bytes: bytes,
                createdAt: 0,
                lastAccessAt: 0,
                fileRelativePath: PersistedSnapshotDescriptor.relativeFilePath(
                    snapshotID: id, partitionDigest: digest
                ),
                schemaVersion: SnapshotManifestSchema.currentVersion
            ))
            return SnapshotRef(
                snapshotID: id,
                partitionDigest: digest,
                tokenOffset: path.count,
                checkpointType: .leaf,
                bytesOnDisk: bytes
            )
        }

        let first = seed(bytes: 1024)
        let second = seed(bytes: 2048)
        #expect(ssdStore.currentSSDBytesForTesting() == 1024 + 2048)

        let now: ContinuousClock.Instant = .now
        mgr.restoreSnapshotRef(
            path: path, snapshotRef: first,
            partitionKey: defaultKey, lastAccessTime: now
        )
        // Second call resolves to the same node via `insertPath`'s edge
        // walk: first-wins, and the loser's backing is reclaimed.
        mgr.restoreSnapshotRef(
            path: path, snapshotRef: second,
            partitionKey: defaultKey, lastAccessTime: now
        )

        // First-wins: the surviving node still points at `first`.
        let tree = tieredStore.tree(for: defaultKey)!
        let (node, _) = tree.findBestSnapshot(
            tokens: path, updateAccess: false, includeSnapshotRefs: true
        )!
        #expect(node.state.refID == first.snapshotID)

        // The dropped (second) descriptor's backing is reclaimed: its
        // bytes are gone from the SSD budget and its manifest entry
        // removed — only `first` remains.
        #expect(ssdStore.currentSSDBytesForTesting() == first.bytesOnDisk)
        #expect(
            !ssdStore.residentIDsByRecencyForTesting().contains(second.snapshotID)
        )
    }

    // MARK: - SSD admission (dense partition)

    @Test func checkpointAdmissionStillAdmitsSSDForDensePartition() async throws {
        let (mgr, tieredStore, root) = makeSSDManager()
        defer { try? FileManager.default.removeItem(at: root) }

        let key = makeSSDKey()
        let promptTokens = Array(1...10)
        let snapshot = makeUniformSnapshot(offset: 5, type: .system)
        let payload = makeSSDPayload(bytes: 256, checkpointType: .system)

        mgr.admit(SnapshotAdmission.checkpoints(
            fullPromptTokens: promptTokens,
            candidates: [.ramAndSSD(snapshot, payload: payload)],
            partitionKey: key
        )!)
        await tieredStore.ssdStoreForTesting!.flushAsync()

        #expect(tieredStore.ssdStoreForTesting?.currentSSDBytesForTesting() == payload.totalBytes)
    }

    @Test func admitCheckpointSnapshotAdmissionStoresSSDEntriesThroughUnifiedInterface() async throws {
        let (mgr, tieredStore, root) = makeSSDManager()
        defer { try? FileManager.default.removeItem(at: root) }

        let key = makeSSDKey()
        let fullPromptTokens = Array(1...10)
        let snapshot = makeUniformSnapshot(offset: 5, type: .system)
        let payload = makeSSDPayload(bytes: 256, checkpointType: .system)
        let admission = try #require(SnapshotAdmission.checkpoints(
            fullPromptTokens: fullPromptTokens,
            candidates: [
                SnapshotAdmission.CheckpointCandidate(
                    snapshot: snapshot,
                    storage: .ramAndSSD(payload)
                )
            ],
            partitionKey: key,
            requestID: UUID()
        ))

        let diagnostics = mgr.admit(admission)
        await tieredStore.ssdStoreForTesting!.flushAsync()

        #expect(diagnostics.evictions.isEmpty)
        #expect(diagnostics.supersededLeaves.isEmpty)
        #expect(diagnostics.stats.snapshotCount == 1)
        #expect(tieredStore.ssdStoreForTesting?.currentSSDBytesForTesting() == payload.totalBytes)
    }

    @Test func admitLeafSnapshotAdmissionStoresSSDEntryThroughUnifiedInterface() async throws {
        let (mgr, tieredStore, root) = makeSSDManager()
        defer { try? FileManager.default.removeItem(at: root) }

        let key = makeSSDKey()
        let storedTokens = Array(1...10)
        let snapshot = makeUniformSnapshot(offset: storedTokens.count, type: .leaf)
        let payload = makeSSDPayload(bytes: 256, checkpointType: .leaf)
        let admission = try #require(SnapshotAdmission.leaf(
            storedTokens: storedTokens,
            snapshot: snapshot,
            storage: .ramAndSSD(payload),
            partitionKey: key,
            requestID: UUID()
        ))

        let diagnostics = mgr.admit(admission)
        await tieredStore.ssdStoreForTesting!.flushAsync()

        #expect(diagnostics.evictions.isEmpty)
        #expect(diagnostics.supersededLeaves.isEmpty)
        #expect(diagnostics.stats.snapshotCount == 1)
        #expect(tieredStore.ssdStoreForTesting?.currentSSDBytesForTesting() == payload.totalBytes)
    }

    @Test func leafSSDAdmissionFreesSupersededAncestorBeforeBudgetCut() async throws {
        let payloadBytes = 256
        let (mgr, tieredStore, root) = makeSSDManager(budgetBytes: payloadBytes * 2)
        defer { try? FileManager.default.removeItem(at: root) }

        let key = makeSSDKey()
        let unrelatedTokens = Array(100...109)
        let ancestorTokens = Array(1...10)
        let descendantTokens = Array(1...15)

        mgr.admit(SnapshotAdmission.leaf(
            storedTokens: unrelatedTokens,
            snapshot: makeUniformSnapshot(offset: unrelatedTokens.count, type: .leaf),
            storage: .ramAndSSD(makeSSDPayload(bytes: payloadBytes, checkpointType: .leaf)),
            partitionKey: key
        )!)
        await tieredStore.ssdStoreForTesting!.flushAsync()

        mgr.admit(SnapshotAdmission.leaf(
            storedTokens: ancestorTokens,
            snapshot: makeUniformSnapshot(offset: ancestorTokens.count, type: .leaf),
            storage: .ramAndSSD(makeSSDPayload(bytes: payloadBytes, checkpointType: .leaf)),
            partitionKey: key
        )!)
        await tieredStore.ssdStoreForTesting!.flushAsync()

        let tree = try #require(tieredStore.tree(for: key))
        let unrelatedID = try #require(tree.findBestSnapshot(
            tokens: unrelatedTokens,
            updateAccess: false,
            includeSnapshotRefs: true
        )?.0.state.refID)
        let ancestorID = try #require(tree.findBestSnapshot(
            tokens: ancestorTokens,
            updateAccess: false,
            includeSnapshotRefs: true
        )?.0.state.refID)
        #expect(tieredStore.ssdStoreForTesting?.currentSSDBytesForTesting() == payloadBytes * 2)

        let diagnostics = mgr.admit(SnapshotAdmission.leaf(
            storedTokens: descendantTokens,
            snapshot: makeUniformSnapshot(offset: descendantTokens.count, type: .leaf),
            storage: .ramAndSSD(makeSSDPayload(bytes: payloadBytes, checkpointType: .leaf)),
            partitionKey: key
        )!)
        await tieredStore.ssdStoreForTesting!.flushAsync()

        let descendantID = try #require(tree.findBestSnapshot(
            tokens: descendantTokens,
            updateAccess: false,
            includeSnapshotRefs: true
        )?.0.state.refID)
        let residentIDs = tieredStore.ssdStoreForTesting!.residentIDsByRecencyForTesting()
        #expect(diagnostics.supersededLeaves.map(\.bodyDroppedSnapshotRefID) == [ancestorID])
        #expect(residentIDs.contains(unrelatedID))
        #expect(residentIDs.contains(descendantID))
        #expect(!residentIDs.contains(ancestorID))
        #expect(tieredStore.ssdStoreForTesting?.currentSSDBytesForTesting() == payloadBytes * 2)
    }
}
