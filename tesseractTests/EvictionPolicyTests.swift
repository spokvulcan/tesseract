import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// Tests for the Marconi-style FLOP-aware eviction policy.
///
/// `EvictionPolicy.alpha` and `.modelProfile` are static, so the suite is
/// `.serialized` to keep tests from racing on those globals. Each test that
/// mutates them resets in a `defer`.
@MainActor
@Suite(.serialized)
struct EvictionPolicyTests {

    // MARK: - Helpers

    private let defaultKey = CachePartitionKey(
        modelID: "test-model", kvBits: nil, kvGroupSize: 64
    )
    private let gib = 1024 * 1024 * 1024

    private func makeManager(budgetMB: Int = 100) -> PrefixCacheManager {
        PrefixCacheManager(memoryBudgetBytes: budgetMB * 1024 * 1024)
    }

    /// Synthetic radix node carrying a snapshot of approximately
    /// `snapshotBytesTarget` bytes. The materialized parent chain lets
    /// `EvictionPolicy.parentRelativeFlops` resolve `node.parent?.tokenOffset`.
    ///
    /// `snapshotBytesTarget` only needs to make the *ratio* between
    /// candidates correct — min-max normalization is scale-invariant — so
    /// callers should pick the smallest values that still encode the
    /// relative ordering they care about. Tests deliberately stay in the
    /// kilobyte range; allocating real GPU memory in the megabyte range
    /// just to assert a comparison would burn ~1 GB of unified memory per
    /// suite run.
    private func makeNode(
        tokenOffset: Int,
        parentOffset: Int = 0,
        snapshotBytesTarget: Int = 4096,
        type: HybridCacheSnapshot.CheckpointType = .system,
        accessAge: Duration = .zero
    ) -> RadixTreeNode {
        let parent = RadixTreeNode(edgeTokens: [], tokenOffset: parentOffset)
        let node = RadixTreeNode(
            edgeTokens: [],
            tokenOffset: tokenOffset,
            parent: parent
        )
        // Each `[1, 1, length, 64]` float32 array is `length * 64 * 4` bytes,
        // and there are two arrays per KVCacheSimple snapshot.
        let length = max(1, snapshotBytesTarget / (2 * 64 * 4))
        let kv = KVCacheSimple()
        kv.state = [
            MLXArray.zeros([1, 1, length, 64]),
            MLXArray.zeros([1, 1, length, 64]),
        ]
        node.snapshot = HybridCacheSnapshot.capture(cache: [kv], offset: tokenOffset, type: type)
        if accessAge != .zero {
            node.lastAccessTime = .now - accessAge
        }
        return node
    }

    private func resetPolicyDefaults() {
        PrefixCacheTestFixtures.resetPolicyDefaults()
    }

    private func makeUniformSnapshot(
        offset: Int,
        type: HybridCacheSnapshot.CheckpointType = .system
    ) -> HybridCacheSnapshot {
        PrefixCacheTestFixtures.makeUniformSnapshot(offset: offset, type: type)
    }

    // MARK: - Tests

    /// Marconi rule: nodes with `2+` children represent shared prefixes and
    /// are excluded from the utility-scored candidate set. Setup leaves the
    /// branch with `childCount == 2` after the single forced eviction so it
    /// stays protected for the life of the test.
    @Test func candidateSetExcludesMultiChildNodes() {
        defer { resetPolicyDefaults() }
        let mgr = makeManager(budgetMB: 1000)

        // root → [1..10] (system snap)
        //            → [11..15] leaf
        //            → [21..25] leaf
        //            → [31..35] leaf
        let pathA = Array(1...10) + Array(11...15)
        let pathB = Array(1...10) + Array(21...25)
        let pathC = Array(1...10) + Array(31...35)

        mgr.storeSnapshots(
            promptTokens: pathA,
            capturedSnapshots: [makeUniformSnapshot(offset: 10, type: .system)],
            partitionKey: defaultKey
        )
        mgr.storeLeaf(
            storedTokens: pathA,
            leafSnapshot: makeUniformSnapshot(offset: pathA.count, type: .leaf),
            partitionKey: defaultKey
        )
        mgr.storeLeaf(
            storedTokens: pathB,
            leafSnapshot: makeUniformSnapshot(offset: pathB.count, type: .leaf),
            partitionKey: defaultKey
        )
        mgr.storeLeaf(
            storedTokens: pathC,
            leafSnapshot: makeUniformSnapshot(offset: pathC.count, type: .leaf),
            partitionKey: defaultKey
        )
        #expect(mgr.stats.snapshotCount == 4)

        let snapBytes = makeUniformSnapshot(offset: 10).memoryBytes
        mgr.memoryBudgetBytes = snapBytes * 3
        mgr.evictToFitBudget()

        #expect(mgr.stats.snapshotCount == 3)

        let probe = Array(1...10) + [999]
        let result = mgr.lookup(tokens: probe, partitionKey: defaultKey)
        #expect(result.snapshotTokenOffset == 10)
    }

    /// `deltaF` is parent-relative: holding `nodeOffset` constant, a
    /// snapshot whose parent sits closer to the root saves more FLOPs
    /// (longer unique suffix) and scores higher on the FLOP term.
    @Test func parentRelativeFlopsUsed() {
        defer { resetPolicyDefaults() }

        let dfShallow = EvictionPolicy.parentRelativeFlops(
            nodeOffset: 4096, parentOffset: 100
        )
        let dfDeep = EvictionPolicy.parentRelativeFlops(
            nodeOffset: 4096, parentOffset: 3000
        )

        #expect(dfShallow > dfDeep)
        #expect(dfShallow > 0)
        #expect(dfDeep > 0)
    }

    @Test func parentRelativeFlopsZeroWhenParentAtNode() {
        defer { resetPolicyDefaults() }
        let df = EvictionPolicy.parentRelativeFlops(nodeOffset: 100, parentOffset: 100)
        #expect(df == 0)
    }

    @Test func minMaxNormalizationHandlesDegenerateCase() {
        defer { resetPolicyDefaults() }

        #expect(EvictionPolicy.normalize([5.0, 5.0, 5.0]) == [1.0, 1.0, 1.0])
        #expect(EvictionPolicy.normalize([42.0]) == [1.0])
        #expect(EvictionPolicy.normalize([]).isEmpty)

        let healthy = EvictionPolicy.normalize([10.0, 20.0, 30.0])
        #expect(healthy[0] == 0.0)
        #expect(healthy[1] == 0.5)
        #expect(healthy[2] == 1.0)
    }

    @Test func recentAccessBoostsUtility() {
        defer { resetPolicyDefaults() }
        EvictionPolicy.alpha = 0.0

        let older = makeNode(tokenOffset: 1024, accessAge: .seconds(60))
        let newer = makeNode(tokenOffset: 1024, accessAge: .seconds(1))

        let victim = EvictionPolicy.selectVictim(
            candidates: [older, newer], now: .now
        )
        #expect(victim?.node === older)
        #expect(victim?.score.normalizedRecency == 0.0)
    }

    @Test func higherFlopEfficiencyBoostsUtility() {
        defer { resetPolicyDefaults() }
        EvictionPolicy.alpha = 1.0

        let now: ContinuousClock.Instant = .now

        // Same memoryBytes for both → ratio comes from `tokenOffset` alone.
        let tall = makeNode(tokenOffset: 4096)
        tall.lastAccessTime = now
        let short = makeNode(tokenOffset: 256)
        short.lastAccessTime = now

        let victim = EvictionPolicy.selectVictim(
            candidates: [tall, short], now: now
        )
        #expect(victim?.node === short)

        let scores = EvictionPolicy.computeScores(candidates: [tall, short], now: now)
        #expect(scores[0].normalizedFlopEfficiency > scores[1].normalizedFlopEfficiency)
    }

    /// Eviction must consider every partition's eligible nodes globally —
    /// not "lowest within tree A, then tree B" but "lowest across all
    /// trees combined."
    @Test func lowestUtilityEvictedAcrossPartitions() {
        defer { resetPolicyDefaults() }
        let snapBytes = makeUniformSnapshot(offset: 100, type: .leaf).memoryBytes
        let mgr = PrefixCacheManager(memoryBudgetBytes: snapBytes * 2)

        let keyA = CachePartitionKey(modelID: "a", kvBits: nil, kvGroupSize: 64)
        let keyB = CachePartitionKey(modelID: "b", kvBits: nil, kvGroupSize: 64)

        mgr.storeLeaf(
            storedTokens: Array(1...100),
            leafSnapshot: makeUniformSnapshot(offset: 100, type: .leaf),
            partitionKey: keyA
        )
        mgr.storeLeaf(
            storedTokens: Array(200...299),
            leafSnapshot: makeUniformSnapshot(offset: 100, type: .leaf),
            partitionKey: keyB
        )
        #expect(mgr.stats.snapshotCount == 2)

        mgr.memoryBudgetBytes = snapBytes
        mgr.evictToFitBudget()

        #expect(mgr.stats.snapshotCount == 1)
        let resultA = mgr.lookup(tokens: Array(1...100), partitionKey: keyA)
        let resultB = mgr.lookup(tokens: Array(200...299), partitionKey: keyB)
        #expect(resultA.snapshot == nil)
        #expect(resultB.snapshotTokenOffset == 100)
    }

    /// When eviction empties a snapshot from a node that still has exactly
    /// one child, `evictToFitBudget` collapses the snapshot-less node into
    /// its child to preserve compressed-radix structure. Both nodes use
    /// `.branchPoint` so the type-protection guard on `.system` doesn't
    /// interfere — the test is about the collapse mechanism, not type
    /// priority.
    @Test func singleChildEvictionCollapsesNode() {
        defer { resetPolicyDefaults() }

        let snapBytes = makeUniformSnapshot(offset: 10).memoryBytes
        let mgr = PrefixCacheManager(memoryBudgetBytes: snapBytes * 100)

        // root → node1 (branchPoint, offset 10) → node2 (leaf, offset 20)
        // Storage order: branchPoint first → older. Leaf second → newer.
        mgr.storeSnapshots(
            promptTokens: Array(1...10),
            capturedSnapshots: [makeUniformSnapshot(offset: 10, type: .branchPoint)],
            partitionKey: defaultKey
        )
        let leafTokens = Array(1...20)
        mgr.storeLeaf(
            storedTokens: leafTokens,
            leafSnapshot: makeUniformSnapshot(offset: 20, type: .leaf),
            partitionKey: defaultKey
        )
        let nodeCountBefore = mgr.stats.totalNodeCount

        mgr.memoryBudgetBytes = snapBytes
        mgr.evictToFitBudget()

        let result = mgr.lookup(tokens: leafTokens, partitionKey: defaultKey)
        #expect(result.snapshotTokenOffset == 20)
        #expect(mgr.stats.totalNodeCount == nodeCountBefore - 1)
    }

    @Test func memoryBudgetRespected() {
        defer { resetPolicyDefaults() }
        let snap = makeUniformSnapshot(offset: 100, type: .leaf)
        let snapBytes = snap.memoryBytes
        let mgr = PrefixCacheManager(memoryBudgetBytes: snapBytes * 3)

        for i in 0..<10 {
            let tokens = Array((i * 100 + 1)...((i + 1) * 100))
            mgr.storeLeaf(
                storedTokens: tokens,
                leafSnapshot: makeUniformSnapshot(offset: 100, type: .leaf),
                partitionKey: defaultKey
            )
        }

        #expect(mgr.totalSnapshotBytes <= snapBytes * 3)
        #expect(mgr.stats.snapshotCount <= 3)
        #expect(mgr.stats.snapshotCount >= 1)
    }

    /// Regression test for the LRU-only eviction limitation: a tall
    /// main-agent checkpoint loses to fresher subagent checkpoints under
    /// pure recency. With `alpha > 0`, the F/B term saves the tall one
    /// because its FLOP savings dominate the score.
    @Test func tallMainAgentSurvivesSubagentChurn() {
        defer { resetPolicyDefaults() }
        EvictionPolicy.alpha = 1.0

        // Bytes targets are kept in the kilobyte range — only the *ratio*
        // between candidates matters because of min-max normalization.
        let mainAgent = makeNode(
            tokenOffset: 76_800,
            snapshotBytesTarget: 200 * 1024,
            accessAge: .seconds(30)
        )
        let sub1 = makeNode(
            tokenOffset: 25_000, snapshotBytesTarget: 60 * 1024,
            accessAge: .milliseconds(100)
        )
        let sub2 = makeNode(
            tokenOffset: 25_000, snapshotBytesTarget: 60 * 1024,
            accessAge: .milliseconds(200)
        )
        let sub3 = makeNode(
            tokenOffset: 25_000, snapshotBytesTarget: 60 * 1024,
            accessAge: .milliseconds(300)
        )

        let victim = EvictionPolicy.selectVictim(
            candidates: [mainAgent, sub1, sub2, sub3]
        )
        #expect(victim?.node !== mainAgent)
    }

    /// 1 main agent + 3 parallel subagents share a tight budget under
    /// utility scoring. All four tall-prefix snapshots survive while
    /// shorter leaf-like entries evict first.
    @Test func parallelSubagentsCoexistWithMainAgent() {
        defer { resetPolicyDefaults() }
        EvictionPolicy.alpha = 1.0

        let main = makeNode(
            tokenOffset: 80_000, snapshotBytesTarget: 200 * 1024,
            accessAge: .milliseconds(50)
        )
        let s1 = makeNode(
            tokenOffset: 25_000, snapshotBytesTarget: 60 * 1024,
            accessAge: .milliseconds(150)
        )
        let s2 = makeNode(
            tokenOffset: 25_000, snapshotBytesTarget: 60 * 1024,
            accessAge: .milliseconds(150)
        )
        let s3 = makeNode(
            tokenOffset: 25_000, snapshotBytesTarget: 60 * 1024,
            accessAge: .milliseconds(150)
        )
        let leaf1 = makeNode(
            tokenOffset: 500, snapshotBytesTarget: 4 * 1024,
            type: .leaf, accessAge: .seconds(2)
        )
        let leaf2 = makeNode(
            tokenOffset: 500, snapshotBytesTarget: 4 * 1024,
            type: .leaf, accessAge: .seconds(3)
        )

        var remaining: [RadixTreeNode] = [main, s1, s2, s3, leaf1, leaf2]
        let leaves: Set<ObjectIdentifier> = [
            ObjectIdentifier(leaf1), ObjectIdentifier(leaf2),
        ]

        for _ in 0..<2 {
            guard let victim = EvictionPolicy.selectVictim(candidates: remaining)
            else { break }
            #expect(leaves.contains(ObjectIdentifier(victim.node)))
            remaining.removeAll { $0 === victim.node }
        }

        let survivors = Set(remaining.map(ObjectIdentifier.init))
        #expect(survivors == [
            ObjectIdentifier(main),
            ObjectIdentifier(s1),
            ObjectIdentifier(s2),
            ObjectIdentifier(s3),
        ])
    }

    // MARK: - Prefix cache budget sizing

    @Test func budgetScalesWithRAM() {
        let modelBytes = Int64(10 * gib)

        let budget32 = LLMActor.autoSizedPrefixCacheMemoryBudgetBytes(
            totalMemoryBytes: UInt64(32 * gib),
            modelMemoryBytes: modelBytes
        )
        let budget48 = LLMActor.autoSizedPrefixCacheMemoryBudgetBytes(
            totalMemoryBytes: UInt64(48 * gib),
            modelMemoryBytes: modelBytes
        )
        let budget64 = LLMActor.autoSizedPrefixCacheMemoryBudgetBytes(
            totalMemoryBytes: UInt64(64 * gib),
            modelMemoryBytes: modelBytes
        )

        // (total - 10 GiB model - 20 GiB headroom) / 2
        #expect(budget32 == 1 * gib)
        #expect(budget48 == 9 * gib)
        #expect(budget64 == 17 * gib)
        #expect(budget32 < budget48)
        #expect(budget48 < budget64)
    }

    @Test func budgetAccountsForModel() {
        let totalMemoryBytes = UInt64(48 * gib)

        let budget4 = LLMActor.autoSizedPrefixCacheMemoryBudgetBytes(
            totalMemoryBytes: totalMemoryBytes,
            modelMemoryBytes: Int64(4 * gib)
        )
        let budget10 = LLMActor.autoSizedPrefixCacheMemoryBudgetBytes(
            totalMemoryBytes: totalMemoryBytes,
            modelMemoryBytes: Int64(10 * gib)
        )
        let budget20 = LLMActor.autoSizedPrefixCacheMemoryBudgetBytes(
            totalMemoryBytes: totalMemoryBytes,
            modelMemoryBytes: Int64(20 * gib)
        )

        // (48 GiB - model - 20 GiB headroom) / 2
        #expect(budget4 == 12 * gib)
        #expect(budget10 == 9 * gib)
        #expect(budget20 == 4 * gib)
        #expect(budget4 > budget10)
        #expect(budget10 > budget20)
    }

    @Test func budgetUsesRawFormulaWithoutHardCap() {
        let budget = LLMActor.autoSizedPrefixCacheMemoryBudgetBytes(
            totalMemoryBytes: UInt64(128 * gib),
            modelMemoryBytes: Int64(10 * gib)
        )

        // (128 GiB - 10 GiB model - 20 GiB headroom) / 2
        #expect(budget == 49 * gib)
    }

    @Test func budgetClampsToZeroWhenHeadroomExhausted() {
        let exactFit = LLMActor.autoSizedPrefixCacheMemoryBudgetBytes(
            totalMemoryBytes: UInt64(16 * gib),
            modelMemoryBytes: Int64(12 * gib)
        )
        let overcommitted = LLMActor.autoSizedPrefixCacheMemoryBudgetBytes(
            totalMemoryBytes: UInt64(16 * gib),
            modelMemoryBytes: Int64(13 * gib)
        )

        #expect(exactFit == 0)
        #expect(overcommitted == 0)
    }

    // MARK: - Profile detection

    /// Run `body` against a fresh temp directory containing `config.json`.
    /// The directory is removed when `body` returns or throws.
    private func withConfigFixture<T>(
        _ json: String?,
        _ body: (URL) throws -> T
    ) throws -> T {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("evictpolicy-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        if let json {
            try json.write(
                to: dir.appendingPathComponent("config.json"),
                atomically: true, encoding: .utf8
            )
        }
        return try body(dir)
    }

    /// 4B-PARO-shaped config (`hidden_size=2560`) yields a profile that
    /// matches the `qwen35_4B_PARO` fallback.
    @Test func detectModelFlopProfile_qwen35_4B_PARO() throws {
        let json = #"""
        {
          "model_type": "qwen3_5",
          "text_config": {
            "model_type": "qwen3_5_text",
            "num_hidden_layers": 32,
            "hidden_size": 2560,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "full_attention_interval": 4
          }
        }
        """#
        try withConfigFixture(json) { dir in
            #expect(LLMActor.detectModelFlopProfile(directory: dir) == .qwen35_4B_PARO)
        }
    }

    /// 9B-PARO-shaped config (`hidden_size=4096`) yields a profile distinct
    /// from the 4B fallback. Regression test for the original P1 finding —
    /// hardcoded constants would have scored the 9B model with 4B FLOPs.
    @Test func detectModelFlopProfile_qwen35_9B_PARO() throws {
        let json = #"""
        {
          "model_type": "qwen3_5",
          "text_config": {
            "model_type": "qwen3_5_text",
            "num_hidden_layers": 32,
            "hidden_size": 4096,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "full_attention_interval": 4
          }
        }
        """#
        try withConfigFixture(json) { dir in
            let profile = LLMActor.detectModelFlopProfile(directory: dir)
            #expect(profile != nil)
            #expect(profile?.hiddenSize == 4096)
            #expect(profile?.attentionLayers == 8)
            #expect(profile?.ssmLayers == 24)
            #expect(profile?.mlpLayers == 32)
            #expect(profile?.ssmStateDim == 32 * 128)
            #expect(profile != .qwen35_4B_PARO)
        }
    }

    /// LLM-only path keeps fields at the top level (no `text_config`
    /// nesting). The detector handles both shapes.
    @Test func detectModelFlopProfile_llmOnlyTopLevelFields() throws {
        let json = #"""
        {
          "model_type": "qwen3_5",
          "num_hidden_layers": 32,
          "hidden_size": 2560,
          "linear_num_value_heads": 32,
          "linear_key_head_dim": 128,
          "full_attention_interval": 4
        }
        """#
        try withConfigFixture(json) { dir in
            #expect(LLMActor.detectModelFlopProfile(directory: dir) == .qwen35_4B_PARO)
        }
    }

    /// Non-Qwen3.5 model_type returns nil so the caller falls back to the
    /// default profile rather than scoring with mismatched constants.
    @Test func detectModelFlopProfile_unknownModelTypeReturnsNil() throws {
        let json = #"""
        { "model_type": "llama", "hidden_size": 4096, "num_hidden_layers": 32 }
        """#
        try withConfigFixture(json) { dir in
            #expect(LLMActor.detectModelFlopProfile(directory: dir) == nil)
        }
    }

    /// Missing config.json returns nil.
    @Test func detectModelFlopProfile_missingConfigReturnsNil() throws {
        try withConfigFixture(nil) { dir in
            #expect(LLMActor.detectModelFlopProfile(directory: dir) == nil)
        }
    }
}
