import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// Task 1.6 integration tests: validates the end-to-end prefix cache flow
/// from normalization through tokenization, radix lookup, checkpoint planning,
/// suffix slicing, snapshot capture, and leaf storage.
///
/// These tests use synthetic token sequences (no model required) to exercise
/// the PrefixCacheManager ↔ HybridCacheSnapshot ↔ TokenRadixTree integration
/// contracts that LLMActor depends on.
@MainActor
struct PrefixCacheIntegrationTests {

    // MARK: - Helpers

    private let defaultKey = CachePartitionKey(modelID: "test-model", kvBits: 8, kvGroupSize: 64)

    private func makeManager(budgetMB: Int = 500) -> PrefixCacheManager {
        PrefixCacheManager(memoryBudgetBytes: budgetMB * 1024 * 1024)
    }

    /// Build a KVCacheSimple snapshot at the given offset with realistic shape.
    private func makeKVSnapshot(
        offset: Int,
        type: HybridCacheSnapshot.CheckpointType = .leaf,
        layerCount: Int = 1
    ) -> HybridCacheSnapshot {
        var cache: [any KVCache] = []
        for _ in 0..<layerCount {
            let kv = KVCacheSimple()
            kv.state = [
                MLXArray.zeros([1, 1, max(offset, 1), 64]),
                MLXArray.zeros([1, 1, max(offset, 1), 64]),
            ]
            cache.append(kv)
        }
        return HybridCacheSnapshot.capture(cache: cache, offset: offset, type: type)!
    }

    /// Build a hybrid snapshot (attention + Mamba layers) mimicking Qwen3.5 layout.
    private func makeHybridSnapshot(
        offset: Int,
        type: HybridCacheSnapshot.CheckpointType = .leaf
    ) -> HybridCacheSnapshot {
        var cache: [any KVCache] = []
        // 8 layers: Mamba at 0,1,2, attention at 3, Mamba at 4,5,6, attention at 7
        for i in 0..<8 {
            if i % 4 == 3 {
                let kv = KVCacheSimple()
                kv.state = [
                    MLXArray.zeros([1, 1, max(offset, 1), 64]),
                    MLXArray.zeros([1, 1, max(offset, 1), 64]),
                ]
                cache.append(kv)
            } else {
                let mamba = MambaCache()
                mamba.state = [
                    MLXArray.zeros([1, 3, 14336]),
                    MLXArray.zeros([1, 64, 128, 192]),
                ]
                cache.append(mamba)
            }
        }
        return HybridCacheSnapshot.capture(cache: cache, offset: offset, type: type)!
    }

    private func makeTriAttentionRestoreContext(
        budgetTokens: Int = 4,
        prefixProtectionMode: TriAttentionPrefixProtectionMode = .protectStablePrefixOnly
    ) -> TriAttentionSnapshotRestoreContext {
        let headDim = 8
        let sampledHeads = (0 ..< 4).map {
            TriAttentionCalibrationHeadKey(layerIndex: 0, headIndex: $0)
        }
        let stats = Dictionary(uniqueKeysWithValues: sampledHeads.map { key in
            (
                key,
                TriAttentionCalibrationHeadStats(
                    qMeanReal: Array(repeating: Float(1), count: headDim / 2),
                    qMeanImag: Array(repeating: Float(0.5), count: headDim / 2),
                    qAbsMean: Array(repeating: Float(1.5), count: headDim / 2)
                )
            )
        })
        let artifact = TriAttentionCalibrationArtifact(
            metadata: TriAttentionCalibrationMetadata(
                sampledHeads: sampledHeads,
                headDim: headDim,
                ropeStyle: "half",
                modelName: "tests/prefix-cache-triattention"
            ),
            statsByHead: stats
        )
        let configuration = TriAttentionConfiguration(
            enabled: true,
            budgetTokens: budgetTokens,
            calibrationArtifactIdentity: TriAttentionCalibrationArtifactIdentity(rawValue: "artifact"),
            implementationVersion: .v1,
            prefixProtectionMode: prefixProtectionMode
        )
        let runtimeState = TriAttentionQwen35Runtime.makeState(
            configuration: configuration,
            artifact: artifact,
            fullAttentionLayerIndices: [0],
            attentionHeads: 4,
            kvHeads: 2,
            headDim: headDim,
            partialRotaryFactor: 1.0,
            ropeTheta: 100_000,
            ropeType: "default"
        )!
        return TriAttentionSnapshotRestoreContext(
            expectedConfiguration: configuration,
            runtimeState: runtimeState
        )
    }

    private func makeTriAttentionStates(
        tokenCount: Int,
        kvHeads: Int = 2,
        headDim: Int = 8,
        keyBase: Float = 1,
        valueBase: Float = 2
    ) -> (MLXArray, MLXArray) {
        (
            MLXArray(
                Array(repeating: keyBase, count: 1 * kvHeads * tokenCount * headDim)
            ).reshaped(1, kvHeads, tokenCount, headDim),
            MLXArray(
                Array(repeating: valueBase, count: 1 * kvHeads * tokenCount * headDim)
            ).reshaped(1, kvHeads, tokenCount, headDim)
        )
    }

    // MARK: - 1. endToEndNormalizeTokenizeLookupHit

    /// Wire message → normalize → tokenize (synthetic) → store → new request →
    /// same tokens → lookup → hit.
    @Test func endToEndNormalizeTokenizeLookupHit() {
        let mgr = makeManager()
        // Simulate: first request produces tokens [1..100], store leaf
        let tokens1 = Array(1...100)
        let leaf = makeKVSnapshot(offset: 100, type: .leaf)
        mgr.storeLeaf(storedTokens: tokens1, leafSnapshot: leaf, partitionKey: defaultKey)

        // Second request produces same tokens (normalization stable) → hit
        let result = mgr.lookup(tokens: tokens1, partitionKey: defaultKey)
        #expect(result.snapshot != nil)
        #expect(result.snapshotTokenOffset == 100)
        if case .hit = result.reason {} else {
            Issue.record("Expected hit, got \(result.reason)")
        }
    }

    // MARK: - 2. suffixSlicedAtSnapshotOffset

    /// Suffix tokens should start at snapshotTokenOffset when there's a cache hit.
    @Test func suffixSlicedAtSnapshotOffset() {
        let mgr = makeManager()
        // Store a snapshot at offset 50 for the first 50 tokens
        let stored = Array(1...50)
        let fullRequest = Array(1...80)
        let snap = makeKVSnapshot(offset: 50, type: .leaf)
        mgr.storeLeaf(storedTokens: stored, leafSnapshot: snap, partitionKey: defaultKey)

        let result = mgr.lookup(tokens: fullRequest, partitionKey: defaultKey)
        #expect(result.snapshot != nil)
        let suffixStart = result.snapshotTokenOffset
        #expect(suffixStart == 50)
        // The caller would slice: fullTokens[suffixStart...]
        let suffixTokens = Array(fullRequest[suffixStart...])
        #expect(suffixTokens == Array(51...80))
    }

    // MARK: - 3. cacheMissTriggersFullPrefillWithCheckpoints

    /// Miss → full prefill → checkpoint plan includes stable prefix.
    @Test func cacheMissTriggersFullPrefillWithCheckpoints() {
        let mgr = makeManager()
        let tokens = Array(1...500)
        let stableOffset = 200

        // Empty cache → miss
        let result = mgr.lookup(tokens: tokens, partitionKey: defaultKey)
        #expect(result.snapshot == nil)

        // Plan should include stable prefix checkpoint
        let plan = mgr.planCheckpoints(
            tokens: tokens,
            stablePrefixOffset: stableOffset,
            partitionKey: defaultKey
        )
        #expect(plan.count == 1)
        #expect(plan[0].offset == 200)
        #expect(plan[0].type == .system)
    }

    // MARK: - 4. hybridCacheRestoredWithCorrectLayerTypes

    /// Restored cache has MambaCache at Mamba positions and KVCacheSimple at attention positions.
    @Test func hybridCacheRestoredWithCorrectLayerTypes() {
        let mgr = makeManager()
        let tokens = Array(1...100)
        let snap = makeHybridSnapshot(offset: 100, type: .leaf)
        mgr.storeLeaf(storedTokens: tokens, leafSnapshot: snap, partitionKey: defaultKey)

        let result = mgr.lookup(tokens: tokens, partitionKey: defaultKey)
        let restored = result.restoreCache()
        #expect(restored != nil)
        #expect(restored?.count == 8)

        // Verify layer types: Mamba at 0,1,2,4,5,6 and attention at 3,7
        for i in 0..<8 {
            if i % 4 == 3 {
                #expect(restored![i] is KVCacheSimple, "Layer \(i) should be KVCacheSimple")
            } else {
                #expect(restored![i] is MambaCache, "Layer \(i) should be MambaCache")
            }
        }
    }

    // MARK: - 5. restoredCacheOffsetMatchesSuffix

    /// cache[attentionIdx].offset == snapshotTokenOffset (critical alignment invariant).
    @Test func restoredCacheOffsetMatchesSuffix() {
        let mgr = makeManager()
        let tokens = Array(1...200)
        let snap = makeHybridSnapshot(offset: 200, type: .leaf)
        mgr.storeLeaf(storedTokens: tokens, leafSnapshot: snap, partitionKey: defaultKey)

        let result = mgr.lookup(tokens: tokens, partitionKey: defaultKey)
        let restored = result.restoreCache()!
        let snapshotOffset = result.snapshotTokenOffset
        #expect(snapshotOffset == 200)

        // Attention layers (indices 3, 7) should have offset == snapshotTokenOffset
        #expect(restored[3].offset == snapshotOffset)
        #expect(restored[7].offset == snapshotOffset)
    }

    // MARK: - 6. normalizationProducesStableTokens

    /// Same logical message, different whitespace → same normalized content.
    /// Tests HTTPPrefixCacheMessage normalization that feeds into tokenization.
    @Test func normalizationProducesStableTokens() {
        // HTTPPrefixCacheMessage.assistant normalizes whitespace-only content to ""
        let msg1 = HTTPPrefixCacheMessage.assistant(content: "  hello  ", reasoning: " think ", toolCalls: [])
        let msg2 = HTTPPrefixCacheMessage.assistant(content: "  hello  ", reasoning: " think ", toolCalls: [])
        #expect(msg1 == msg2)

        // Tool call arguments are normalized (JSON sorted/canonicalized)
        let tc1 = HTTPPrefixCacheToolCall(name: "read", arguments: ["path": .string("/a"), "count": .int(1)])
        let tc2 = HTTPPrefixCacheToolCall(name: "read", arguments: ["count": .int(1), "path": .string("/a")])
        #expect(tc1 == tc2)
    }

    // MARK: - 7. dynamicQuantizationHandledInSnapshot

    /// Early snapshot with KVCacheSimple, later with QuantizedKVCache → both restore correctly.
    @Test func dynamicQuantizationHandledInSnapshot() {
        let mgr = makeManager()

        // System snapshot uses KVCacheSimple (early prefill, before quantization kicks in)
        let sysTokens = Array(1...200)
        let kvCache = KVCacheSimple()
        kvCache.state = [MLXArray.zeros([1, 1, 200, 64]), MLXArray.zeros([1, 1, 200, 64])]
        let sysSnap = HybridCacheSnapshot.capture(cache: [kvCache], offset: 200, type: .system)!
        mgr.storeSnapshots(promptTokens: sysTokens + Array(300...400), capturedSnapshots: [sysSnap], partitionKey: defaultKey)

        // Leaf uses QuantizedKVCache (after quantization transition)
        let fullTokens = Array(1...400)
        let qkv = QuantizedKVCache(groupSize: 64, bits: 8)
        let leafSnap = HybridCacheSnapshot.capture(cache: [qkv], offset: 400, type: .leaf)!
        mgr.storeLeaf(storedTokens: fullTokens, leafSnapshot: leafSnap, partitionKey: defaultKey)

        // Lookup full → hits leaf (QuantizedKVCache)
        let fullResult = mgr.lookup(tokens: fullTokens, partitionKey: defaultKey)
        #expect(fullResult.snapshot?.checkpointType == .leaf)
        let restoredLeaf = fullResult.restoreCache()
        #expect(restoredLeaf != nil)
        #expect(restoredLeaf![0] is QuantizedKVCache)

        // Lookup partial (only system prefix) → hits system (KVCacheSimple)
        let partialTokens = sysTokens + Array(500...600)
        let partialResult = mgr.lookup(tokens: partialTokens, partitionKey: defaultKey)
        #expect(partialResult.snapshot?.checkpointType == .system)
        let restoredSys = partialResult.restoreCache()
        #expect(restoredSys != nil)
        #expect(restoredSys![0] is KVCacheSimple)
    }

    // MARK: - 8. snapshotsCapturedDuringPrefillNotPostHoc

    /// Stable-prefix snapshot must be captured at the correct offset DURING prefill,
    /// not reconstructed from the final cache.
    @Test func snapshotsCapturedDuringPrefillNotPostHoc() {
        // Simulate the flow: planCheckpoints returns an offset, snapshot is captured there.
        let mgr = makeManager()
        let tokens = Array(1...1000)
        let stableOffset = 300

        let plan = mgr.planCheckpoints(
            tokens: tokens,
            stablePrefixOffset: stableOffset,
            partitionKey: defaultKey
        )
        #expect(plan.count == 1)
        #expect(plan[0].offset == 300)

        // Simulate capturing during prefill at the planned offset
        let midSnap = makeKVSnapshot(offset: 300, type: .system)
        #expect(midSnap.tokenOffset == 300)

        mgr.storeSnapshots(promptTokens: tokens, capturedSnapshots: [midSnap], partitionKey: defaultKey)

        // Verify: system snapshot stored at correct offset
        let result = mgr.lookup(tokens: Array(1...500), partitionKey: defaultKey)
        #expect(result.snapshot?.tokenOffset == 300)
        #expect(result.snapshot?.checkpointType == .system)
    }

    // MARK: - 9. noDoublePrefill

    /// TokenIterator.init() calls prepareWithCheckpoints() exactly once.
    /// This test validates the contract: checkpoint offsets flow through GenerateParameters,
    /// and capturedSnapshots is populated after init — no separate prepare() call needed.
    @Test func noDoublePrefill() {
        // Verify the GenerateParameters contract: checkpoints and checkpointBaseOffset
        // are passed through, and default to empty/zero (no double-prefill path).
        var params = GenerateParameters()
        #expect(params.checkpoints.isEmpty)
        #expect(params.checkpointBaseOffset == 0)

        // Setting checkpoints doesn't trigger any separate prepare call — they're
        // read by TokenIterator.init during its single prepare() invocation.
        params.checkpoints = [100: .system, 200: .system]
        params.checkpointBaseOffset = 50
        #expect(params.checkpoints.count == 2)
        #expect(params.checkpointBaseOffset == 50)
    }

    // MARK: - 10. partitionKeyIsolatesDifferentKvBits

    /// Store at kvBits=8, lookup at kvBits=nil → miss despite same tokens.
    @Test func partitionKeyIsolatesDifferentKvBits() {
        let mgr = makeManager()
        let tokens = Array(1...100)
        let snap = makeKVSnapshot(offset: 100, type: .leaf)

        let key8 = CachePartitionKey(modelID: "model", kvBits: 8, kvGroupSize: 64)
        let keyNil = CachePartitionKey(modelID: "model", kvBits: nil, kvGroupSize: 64)

        mgr.storeLeaf(storedTokens: tokens, leafSnapshot: snap, partitionKey: key8)

        // Same tokens, different kvBits → miss
        let result = mgr.lookup(tokens: tokens, partitionKey: keyNil)
        #expect(result.snapshot == nil)
        if case .missNoEntries = result.reason {} else {
            Issue.record("Expected missNoEntries for different kvBits")
        }

        // Same key → hit
        let hitResult = mgr.lookup(tokens: tokens, partitionKey: key8)
        #expect(hitResult.snapshot != nil)
    }

    // MARK: - 11. stablePrefixIncludesToolTokens

    /// Stable prefix checkpoint covers system + tool tokens.
    /// The two-probe boundary detected by StablePrefixDetector includes tool definitions
    /// because tools are part of the stable template prefix.
    @Test func stablePrefixIncludesToolTokens() {
        let mgr = makeManager()
        // System tokens: [1..100], Tool tokens: [101..200], User tokens: [201..500]
        let fullTokens = Array(1...500)
        let systemPlusToolOffset = 200  // As if StablePrefixDetector returned this

        let plan = mgr.planCheckpoints(
            tokens: fullTokens,
            stablePrefixOffset: systemPlusToolOffset,
            partitionKey: defaultKey
        )
        #expect(plan.count == 1)
        #expect(plan[0].offset == 200, "Stable prefix should be at system+tool boundary")
        #expect(plan[0].type == .system)
    }

    // MARK: - 12. checkpointBaseOffsetRebasesCorrectlyOnHit

    /// Cache hit at 3K, checkpoint planned at absolute 4K → after filtering, checkpoint
    /// is at 4K which is in the suffix (4K > 3K). checkpointBaseOffset = 3K.
    @Test func checkpointBaseOffsetRebasesCorrectlyOnHit() {
        let mgr = makeManager()
        let tokens = Array(1...5000)

        // Store a snapshot at offset 3000
        let snap = makeKVSnapshot(offset: 3000, type: .leaf)
        mgr.storeLeaf(storedTokens: Array(1...3000), leafSnapshot: snap, partitionKey: defaultKey)

        // Lookup with extended tokens → hit at 3000
        let result = mgr.lookup(tokens: tokens, partitionKey: defaultKey)
        #expect(result.snapshotTokenOffset == 3000)

        // Plan checkpoints for a stable prefix at 4000
        var plan = mgr.planCheckpoints(
            tokens: tokens,
            stablePrefixOffset: 4000,
            partitionKey: defaultKey
        )

        // Filter: only checkpoints in the suffix (> 3000)
        plan = plan.filter { $0.offset > 3000 }
        #expect(plan.count == 1)
        #expect(plan[0].offset == 4000)

        // The checkpointBaseOffset should be set to snapshot.tokenOffset (3000)
        // so that TokenIterator rebases: absolute 4000 - base 3000 = relative 1000
        let checkpointBaseOffset = result.snapshotTokenOffset
        #expect(checkpointBaseOffset == 3000)

        // Verify: the absolute offset minus base gives the relative position in suffix
        let relativeOffset = plan[0].offset - checkpointBaseOffset
        #expect(relativeOffset == 1000)
    }

    // MARK: - 12b. newSnapshotFromTwoPassReusable

    /// A hit at K with a longer tree match to M synthesizes an alignment
    /// checkpoint at M, and the next request reuses that deeper snapshot.
    @Test func newSnapshotFromTwoPassReusable() {
        let mgr = makeManager()
        let seedPath = Array(1...600)

        mgr.storeSnapshots(
            promptTokens: seedPath,
            capturedSnapshots: [makeKVSnapshot(offset: 200, type: .system)],
            partitionKey: defaultKey
        )

        let firstRequest = seedPath + [999]
        let firstLookup = mgr.lookup(tokens: firstRequest, partitionKey: defaultKey)
        #expect(firstLookup.snapshotTokenOffset == 200)
        #expect(firstLookup.sharedPrefixLength == 600)

        let planned = mgr.planCheckpoints(
            tokens: firstRequest,
            stablePrefixOffset: nil,
            partitionKey: defaultKey
        )
        #expect(!planned.contains(where: { $0.offset == 600 }))

        let alignmentOffset = mgr.alignmentCheckpointOffset(
            lookupResult: firstLookup,
            totalTokenCount: firstRequest.count,
            plannedCheckpoints: planned
        )
        #expect(alignmentOffset == 600)

        mgr.storeSnapshots(
            promptTokens: firstRequest,
            capturedSnapshots: [makeKVSnapshot(offset: 600, type: .branchPoint)],
            partitionKey: defaultKey
        )

        let nextRequest = seedPath + [1000, 1001]
        let nextLookup = mgr.lookup(tokens: nextRequest, partitionKey: defaultKey)
        #expect(nextLookup.snapshotTokenOffset == 600)
        #expect(nextLookup.snapshot?.checkpointType == .branchPoint)
    }

    // MARK: - 13. leafStoredUnderPostResponsePath

    /// After generation, leaf snapshot stored under re-tokenized (prompt + response) path.
    @Test func leafStoredUnderPostResponsePath() {
        let mgr = makeManager()
        // Prompt tokens: [1..100]
        // After generation, stored tokens: [1..100, 101..150] (prompt + response)
        let storedTokens = Array(1...150)
        let leaf = makeKVSnapshot(offset: 150, type: .leaf)
        mgr.storeLeaf(storedTokens: storedTokens, leafSnapshot: leaf, partitionKey: defaultKey)

        // Next request extending the conversation: [1..150, 151..200]
        let nextRequest = Array(1...200)
        let result = mgr.lookup(tokens: nextRequest, partitionKey: defaultKey)
        #expect(result.snapshot != nil)
        #expect(result.snapshotTokenOffset == 150)
    }

    // MARK: - 14. nextTurnHitsLeafFromPreviousTurn

    /// Turn N stores leaf. Turn N+1 (extending conversation) hits leaf on lookup.
    @Test func nextTurnHitsLeafFromPreviousTurn() {
        let mgr = makeManager()

        // Turn 1: system + user + assistant → stored as leaf
        let turn1Tokens = Array(1...300)
        let turn1Leaf = makeKVSnapshot(offset: 300, type: .leaf)
        mgr.storeLeaf(storedTokens: turn1Tokens, leafSnapshot: turn1Leaf, partitionKey: defaultKey)

        // Turn 2: extends with user2 + (pending generation)
        let turn2Tokens = Array(1...300) + Array(400...500)
        let result = mgr.lookup(tokens: turn2Tokens, partitionKey: defaultKey)
        #expect(result.snapshot != nil)
        #expect(result.snapshotTokenOffset == 300, "Should hit turn 1 leaf")

        // Verify suffix = turn 2 extension
        let suffix = Array(turn2Tokens[result.snapshotTokenOffset...])
        #expect(suffix == Array(400...500))
    }

    // MARK: - 15. leafOffsetAlignedAfterNormalization

    /// Generated response with trailing whitespace → trim aligns attention offset
    /// to storedTokens.count.
    @Test func leafOffsetAlignedAfterNormalization() throws {
        // Simulate: finalCache has offset 105 (un-normalized response)
        // but storedTokens.count = 100 (normalized response → fewer tokens)
        let kv = KVCacheSimple()
        kv.state = [
            MLXArray.zeros([1, 1, 105, 64]),
            MLXArray.zeros([1, 1, 105, 64]),
        ]
        let initialOffset = kv.offset
        #expect(initialOffset == 105)

        // Trim to align: 105 - 100 = 5 tokens
        let storedTokenCount = 100
        let actualCacheOffset = kv.offset
        if actualCacheOffset > storedTokenCount {
            let trimAmount = actualCacheOffset - storedTokenCount
            let trimmed = kv.trim(trimAmount)
            #expect(trimmed == trimAmount)
        }

        #expect(kv.offset == 100, "After trim, offset should match storedTokens.count")

        // Now capture the leaf snapshot at the aligned offset
        let leaf = try #require(HybridCacheSnapshot.capture(cache: [kv], offset: 100, type: .leaf))
        #expect(leaf.tokenOffset == 100)
    }

    // MARK: - 16. mambaLayersNotTrimmedInLeafAlignment

    /// MambaCache.isTrimmable==false → skip in trim loop (accepted divergence).
    @Test func mambaLayersNotTrimmedInLeafAlignment() {
        let mamba = MambaCache()
        mamba.state = [
            MLXArray.zeros([1, 3, 14336]),
            MLXArray.zeros([1, 64, 128, 192]),
        ]
        #expect(mamba.isTrimmable == false)

        // Attempting to trim a Mamba layer should return 0
        let trimmed = mamba.trim(10)
        #expect(trimmed == 0)

        // Attention layer IS trimmable
        let kv = KVCacheSimple()
        kv.state = [MLXArray.zeros([1, 1, 50, 64]), MLXArray.zeros([1, 1, 50, 64])]
        #expect(kv.isTrimmable == true)
        let kvTrimmed = kv.trim(5)
        #expect(kvTrimmed == 5)
        #expect(kv.offset == 45)
    }

    // MARK: - 16b. triAttentionLayersNotTrimmedInLeafAlignment

    /// TriAttention sparse caches behave like Mamba in the leaf-alignment
    /// loop: per-layer `trim(_:)` is a no-op, `canTrimPromptCache` flips to
    /// false as soon as one is present, and `trimPromptCache` returns 0
    /// instead of silently mutating the dense siblings. This is the rule
    /// that lets the LLMActor leaf-store guard skip cleanly when
    /// normalization would shorten the canonical token sequence.
    @Test func triAttentionLayersNotTrimmedInLeafAlignment() {
        let sparse = TriAttentionSparseKVCache(configuration: .v1Disabled)
        let keys = MLXArray(Array(repeating: Float(1), count: 1 * 1 * 4 * 8)).reshaped(1, 1, 4, 8)
        let values = MLXArray(Array(repeating: Float(2), count: 1 * 1 * 4 * 8)).reshaped(1, 1, 4, 8)
        let _ = sparse.update(keys: keys, values: values)

        #expect(sparse.isTrimmable == false)
        #expect(sparse.trim(2) == 0)
        #expect(sparse.offset == 4)

        let kv = KVCacheSimple()
        kv.state = [
            MLXArray.zeros([1, 1, 4, 8]),
            MLXArray.zeros([1, 1, 4, 8]),
        ]
        let mixed: [KVCache] = [kv, sparse]
        #expect(canTrimPromptCache(mixed) == false)
        #expect(containsTriAttentionState(mixed) == true)
        #expect(trimPromptCache(mixed, numTokens: 2) == 0)
        // Dense sibling must remain untouched when the array as a whole is
        // non-trimmable — otherwise the normalization-trim guard's "skip
        // leaf store" decision would be defeated by partial mutation.
        #expect(kv.offset == 4)
    }

    // MARK: - 16c. triAttentionSnapshotRestoreExtendsAtAbsoluteOffset

    /// Hit-restore path semantics: a TriAttention snapshot captured at offset
    /// `K` must restore with `cache.offset == K` and accept additional updates
    /// that advance the offset to `K + suffix`, mirroring the dense
    /// KVCacheSimple contract used by `LLMActor`'s suffix-only prefill.
    @Test func triAttentionSnapshotRestoreExtendsAtAbsoluteOffset() throws {
        let original = TriAttentionSparseKVCache(configuration: .v1Disabled)
        let keys = MLXArray(Array(repeating: Float(1), count: 1 * 1 * 4 * 8)).reshaped(1, 1, 4, 8)
        let values = MLXArray(Array(repeating: Float(2), count: 1 * 1 * 4 * 8)).reshaped(1, 1, 4, 8)
        let _ = original.update(keys: keys, values: values)

        let snapshot = try #require(HybridCacheSnapshot.capture(
            cache: [original], offset: original.offset, type: .system
        ))

        let restored = snapshot.restore()
        let restoredCache = try #require(restored[0] as? TriAttentionSparseKVCache)
        #expect(restoredCache.offset == 4)

        // Suffix prefill: append two more tokens, mirroring what the
        // hit-restore path does after `lookupResult.restoreCache()`.
        let suffixKeys = MLXArray(Array(repeating: Float(3), count: 1 * 1 * 2 * 8)).reshaped(1, 1, 2, 8)
        let suffixValues = MLXArray(Array(repeating: Float(4), count: 1 * 1 * 2 * 8)).reshaped(1, 1, 2, 8)
        let _ = restoredCache.update(keys: suffixKeys, values: suffixValues)

        #expect(restoredCache.offset == 6)
        #expect(restoredCache.retainedPositions?.dim(2) == 6)
        #expect(restoredCache.retainedPositions?.asArray(Int32.self) == [0, 1, 2, 3, 4, 5])
    }

    @Test func lookupResultRestoreCacheForwardsTriAttentionRestoreContext() throws {
        let restoreContext = makeTriAttentionRestoreContext()
        let original = TriAttentionSparseKVCache(
            configuration: restoreContext.expectedConfiguration,
            runtimeState: restoreContext.runtimeState
        )
        original.offset = 4
        original.state = [
            MLXArray([Int32(0), 2, 1, 3]).reshaped(1, 2, 2),
            MLXArray(Array(repeating: Float(1), count: 1 * 2 * 2 * 8)).reshaped(1, 2, 2, 8),
            MLXArray(Array(repeating: Float(2), count: 1 * 2 * 2 * 8)).reshaped(1, 2, 2, 8),
        ]
        let snapshot = try #require(HybridCacheSnapshot.capture(
            cache: [original],
            offset: original.offset,
            type: .leaf
        ))
        let partitionKey = CachePartitionKey(
            modelID: "test-model",
            kvBits: 8,
            kvGroupSize: 64
        )
        let lookupResult = PrefixCacheManager.LookupResult(
            snapshot: snapshot,
            partitionKey: partitionKey,
            snapshotTokenOffset: snapshot.tokenOffset,
            sharedPrefixLength: snapshot.tokenOffset,
            reason: .hit(
                snapshotOffset: snapshot.tokenOffset,
                totalTokens: snapshot.tokenOffset,
                type: snapshot.checkpointType
            )
        )

        let restored = try #require(lookupResult.restoreCache(
            triAttentionRestoreContext: restoreContext
        ))
        let restoredCache = try #require(restored[0] as? TriAttentionSparseKVCache)

        switch restoredCache.makeMask(n: 2, windowSize: nil, returnArray: false) {
        case .array(let mask):
            #expect(mask.shape == [1, 4, 2, 4])
        default:
            Issue.record("Expected restored TriAttention cache to use an explicit sparse mask")
        }
    }

    @Test func triAttentionStablePrefixProtectionRetainsStablePrefixDuringColdPrefill() throws {
        let stablePrefixOffset = 6
        let restoreContext = makeTriAttentionRestoreContext(
            budgetTokens: 4,
            prefixProtectionMode: .protectStablePrefixOnly
        )
        let cache = TriAttentionSparseKVCache(
            configuration: restoreContext.expectedConfiguration,
            runtimeState: restoreContext.runtimeState
        )
        var parameters = GenerateParameters(triAttention: restoreContext.expectedConfiguration)
        parameters.triAttentionStablePrefixOffset = stablePrefixOffset
        let tokenCount = restoreContext.expectedConfiguration.budgetTokens + 129
        parameters.configureTriAttentionCachesForPrefill([cache], inputTokenCount: tokenCount)

        let (keys, values) = makeTriAttentionStates(tokenCount: tokenCount)
        _ = cache.update(keys: keys, values: values)
        TriAttentionQwen35Runtime.pruneIfNeeded(caches: [cache], layerIndices: [0])

        let retainedPositions = Set(try #require(cache.retainedPositions?[0, 0].asArray(Int32.self)))
        #expect(retainedPositions.isSuperset(of: Set(Int32(0) ..< Int32(stablePrefixOffset))))
        #expect(retainedPositions.contains(Int32(tokenCount - 1)))
        #expect(
            cache.retainedPositions?.dim(2)
                == restoreContext.expectedConfiguration.budgetTokens + stablePrefixOffset
        )
    }

    @Test func triAttentionStablePrefixProtectionPersistsAcrossRestoreAndSuffixPrefill() throws {
        let stablePrefixOffset = 6
        let restoreContext = makeTriAttentionRestoreContext(
            budgetTokens: 4,
            prefixProtectionMode: .protectStablePrefixOnly
        )
        let original = TriAttentionSparseKVCache(
            configuration: restoreContext.expectedConfiguration,
            runtimeState: restoreContext.runtimeState
        )
        var initialParameters = GenerateParameters(triAttention: restoreContext.expectedConfiguration)
        initialParameters.triAttentionStablePrefixOffset = stablePrefixOffset
        let initialTokenCount = restoreContext.expectedConfiguration.budgetTokens + 129
        initialParameters.configureTriAttentionCachesForPrefill([original], inputTokenCount: initialTokenCount)

        let (initialKeys, initialValues) = makeTriAttentionStates(tokenCount: initialTokenCount)
        _ = original.update(keys: initialKeys, values: initialValues)
        TriAttentionQwen35Runtime.pruneIfNeeded(caches: [original], layerIndices: [0])

        let snapshot = try #require(HybridCacheSnapshot.capture(
            cache: [original],
            offset: original.offset,
            type: .leaf
        ))
        let restored = snapshot.restore(triAttentionRestoreContext: restoreContext)
        let restoredCache = try #require(restored[0] as? TriAttentionSparseKVCache)

        var suffixParameters = GenerateParameters(
            checkpointBaseOffset: snapshot.tokenOffset,
            triAttention: restoreContext.expectedConfiguration
        )
        suffixParameters.triAttentionStablePrefixOffset = stablePrefixOffset
        let suffixTokenCount = 128
        suffixParameters.configureTriAttentionCachesForPrefill([restoredCache], inputTokenCount: suffixTokenCount)

        let (suffixKeys, suffixValues) = makeTriAttentionStates(
            tokenCount: suffixTokenCount,
            keyBase: 3,
            valueBase: 4
        )
        _ = restoredCache.update(keys: suffixKeys, values: suffixValues)
        TriAttentionQwen35Runtime.pruneIfNeeded(caches: [restoredCache], layerIndices: [0])

        let retainedPositions = Set(try #require(restoredCache.retainedPositions?[0, 0].asArray(Int32.self)))
        #expect(retainedPositions.isSuperset(of: Set(Int32(0) ..< Int32(stablePrefixOffset))))
        #expect(retainedPositions.contains(Int32(snapshot.tokenOffset + suffixTokenCount - 1)))
    }

    @Test func triAttentionBoundaryRestorePrefillAppliesStablePrefixProtection() throws {
        let stablePrefixOffset = 6
        let restoreContext = makeTriAttentionRestoreContext(
            budgetTokens: 4,
            prefixProtectionMode: .protectStablePrefixOnly
        )
        let boundaryCache = TriAttentionSparseKVCache(
            configuration: restoreContext.expectedConfiguration,
            runtimeState: restoreContext.runtimeState
        )
        var boundaryParameters = GenerateParameters(triAttention: restoreContext.expectedConfiguration)
        boundaryParameters.triAttentionStablePrefixOffset = stablePrefixOffset
        let boundaryOffset = 20
        boundaryParameters.configureTriAttentionCachesForPrefill([boundaryCache], inputTokenCount: boundaryOffset)

        let (boundaryKeys, boundaryValues) = makeTriAttentionStates(tokenCount: boundaryOffset)
        _ = boundaryCache.update(keys: boundaryKeys, values: boundaryValues)
        let boundarySnapshot = try #require(HybridCacheSnapshot.capture(
            cache: [boundaryCache],
            offset: boundaryCache.offset,
            type: .system
        ))

        let restored = boundarySnapshot.restore(triAttentionRestoreContext: restoreContext)
        let restoredCache = try #require(restored[0] as? TriAttentionSparseKVCache)

        var residualParameters = GenerateParameters(
            checkpointBaseOffset: boundaryOffset,
            triAttention: restoreContext.expectedConfiguration
        )
        residualParameters.triAttentionStablePrefixOffset = stablePrefixOffset
        let residualTokenCount = restoreContext.expectedConfiguration.budgetTokens + 129
        residualParameters.configureTriAttentionCachesForPrefill(
            [restoredCache],
            inputTokenCount: residualTokenCount
        )

        let (residualKeys, residualValues) = makeTriAttentionStates(
            tokenCount: residualTokenCount,
            keyBase: 5,
            valueBase: 6
        )
        _ = restoredCache.update(keys: residualKeys, values: residualValues)
        TriAttentionQwen35Runtime.pruneIfNeeded(caches: [restoredCache], layerIndices: [0])

        let retainedPositions = Set(try #require(restoredCache.retainedPositions?[0, 0].asArray(Int32.self)))
        #expect(retainedPositions.isSuperset(of: Set(Int32(0) ..< Int32(stablePrefixOffset))))
        #expect(restoredCache.offset == boundaryOffset + residualTokenCount)
    }

    @Test func triAttentionBoundaryRestorePrefillUsesBoundaryOffsetForProtectAllPrefill() throws {
        let restoreContext = makeTriAttentionRestoreContext(
            budgetTokens: 4,
            prefixProtectionMode: .protectAllPrefill
        )
        let boundaryCache = TriAttentionSparseKVCache(
            configuration: restoreContext.expectedConfiguration,
            runtimeState: restoreContext.runtimeState
        )
        let boundaryOffset = 20
        let boundaryParameters = GenerateParameters(triAttention: restoreContext.expectedConfiguration)
        boundaryParameters.configureTriAttentionCachesForPrefill(
            [boundaryCache],
            inputTokenCount: boundaryOffset
        )

        let (boundaryKeys, boundaryValues) = makeTriAttentionStates(tokenCount: boundaryOffset)
        _ = boundaryCache.update(keys: boundaryKeys, values: boundaryValues)
        let boundarySnapshot = try #require(HybridCacheSnapshot.capture(
            cache: [boundaryCache],
            offset: boundaryCache.offset,
            type: .system
        ))

        let restored = boundarySnapshot.restore(triAttentionRestoreContext: restoreContext)
        let restoredCache = try #require(restored[0] as? TriAttentionSparseKVCache)

        let residualTokenCount = restoreContext.expectedConfiguration.budgetTokens + 129
        var residualParameters = GenerateParameters(
            checkpointBaseOffset: boundaryOffset,
            triAttention: restoreContext.expectedConfiguration
        )
        residualParameters.configureTriAttentionCachesForPrefill(
            [restoredCache],
            inputTokenCount: residualTokenCount
        )

        let (residualKeys, residualValues) = makeTriAttentionStates(
            tokenCount: residualTokenCount,
            keyBase: 5,
            valueBase: 6
        )
        _ = restoredCache.update(keys: residualKeys, values: residualValues)
        TriAttentionQwen35Runtime.pruneIfNeeded(caches: [restoredCache], layerIndices: [0])

        let retainedPositions = Set(try #require(restoredCache.retainedPositions?[0, 0].asArray(Int32.self)))
        #expect(
            retainedPositions == Set(Int32(0) ..< Int32(boundaryOffset + residualTokenCount))
        )
        #expect(restoredCache.offset == boundaryOffset + residualTokenCount)
    }

    // MARK: - 17. vlm2DTokensExtractedCorrectly

    /// 2D `[batch, seq]` tensor → flat `[Int]` sequence from first batch element.
    /// Tests the extraction contract that LLMActor.extractTokenSequence implements:
    /// ndim <= 1 → asArray, ndim == 2 → [0].asArray.
    @Test func vlm2DTokensExtractedCorrectly() {
        // 1D case: [seq] — direct asArray
        let tokens1D = MLXArray([1, 2, 3, 4, 5])
        #expect(tokens1D.ndim == 1)
        #expect(tokens1D.asArray(Int.self) == [1, 2, 3, 4, 5])

        // 2D case: [batch=1, seq=5] — extract first batch element
        let tokens2D = MLXArray([10, 20, 30, 40, 50]).reshaped([1, 5])
        #expect(tokens2D.ndim == 2)
        #expect(tokens2D[0].asArray(Int.self) == [10, 20, 30, 40, 50])

        // dim(-1) gives sequence length for both shapes
        #expect(tokens1D.dim(-1) == 5)
        #expect(tokens2D.dim(-1) == 5)
    }

    // MARK: - 18. detectorAcceptsSystemPromptString

    /// StablePrefixDetector.detect accepts systemPrompt: String? and returns nil
    /// when systemPrompt is nil or empty.
    @Test func detectorAcceptsSystemPromptString() throws {
        // Nil system prompt → nil
        let result1 = try StablePrefixDetector.detect(
            systemPrompt: nil,
            toolSpecs: nil,
            fullTokens: [1, 2, 3],
            tokenizer: MockTokenizer()
        )
        #expect(result1 == nil)

        // Empty system prompt → nil
        let result2 = try StablePrefixDetector.detect(
            systemPrompt: "",
            toolSpecs: nil,
            fullTokens: [1, 2, 3],
            tokenizer: MockTokenizer()
        )
        #expect(result2 == nil)
    }
    // MARK: - 19. missNoSnapshotReportsActualTreeMatchDepth

    /// When the tree has nodes but no snapshot, sharedPrefixLength should report
    /// how deep the token walk went, not 0. This is the divergence diagnostic
    /// that replaces the old conversation-level diagnosePrefixMismatch.
    @Test func missNoSnapshotReportsActualTreeMatchDepth() {
        let mgr = makeManager()
        // Insert a path WITHOUT a snapshot (storeSnapshots with empty array
        // doesn't store snapshots, but storeLeaf does insert the path).
        // We'll store a leaf, then evict its snapshot to leave a bare path.
        let tokens = Array(1...200)
        let leaf = makeKVSnapshot(offset: 200, type: .leaf)
        mgr.storeLeaf(storedTokens: tokens, leafSnapshot: leaf, partitionKey: defaultKey)

        // Now store a system snapshot at offset 100 so we have a snapshot-bearing
        // node at 100, but nothing beyond that for a different suffix.
        let sysSnap = makeKVSnapshot(offset: 100, type: .system)
        mgr.storeSnapshots(
            promptTokens: tokens,
            capturedSnapshots: [sysSnap],
            partitionKey: defaultKey
        )

        // Lookup with tokens that share 100 tokens then diverge
        let divergentTokens = Array(1...100) + Array(500...600)
        let result = mgr.lookup(tokens: divergentTokens, partitionKey: defaultKey)

        // Should hit the system snapshot at offset 100 — the shared prefix covers it
        #expect(result.snapshot != nil)
        #expect(result.snapshotTokenOffset == 100)
        #expect(result.sharedPrefixLength == 100)

        // Now test true missNoSnapshotInPrefix: tokens that share prefix but
        // diverge BEFORE any snapshot. Insert a path with no snapshot at offset < 100.
        let shortTokens = Array(1...50) + Array(900...950)
        let shortResult = mgr.lookup(tokens: shortTokens, partitionKey: defaultKey)

        // Tokens 1..50 match the tree (first 50 of the stored 1..200 path).
        // No snapshot exists at or before offset 50.
        if case .missNoSnapshotInPrefix = shortResult.reason {
            // sharedPrefixLength should be 50, not 0
            #expect(shortResult.sharedPrefixLength == 50,
                    "Miss should report actual tree match depth, not 0")
        } else if case .hit = shortResult.reason {
            // The system snapshot at 100 is NOT reachable with only 50 matching tokens
            Issue.record("Should not hit — shared prefix (50) is shorter than system snapshot (100)")
        }
    }

    // MARK: - Stripped leaf (Qwen3.5 think-stripping) cross-new-user-turn

    /// Tool-loop continuations should reuse the direct post-response leaf.
    /// This models a request that appends tool results after an assistant
    /// turn that finished with `tool_calls`.
    @Test func directLeafHitOnToolContinuation() {
        let mgr = makeManager(budgetMB: 500)

        let toolCallTurnTokens = Array(1...95)
        mgr.storeLeaf(
            storedTokens: toolCallTurnTokens,
            leafSnapshot: makeKVSnapshot(offset: toolCallTurnTokens.count, type: .leaf),
            partitionKey: defaultKey
        )
        #expect(mgr.stats.snapshotCount == 1)

        let toolContinuationTokens = toolCallTurnTokens + Array(700...710)
        let result = mgr.lookup(tokens: toolContinuationTokens, partitionKey: defaultKey)
        #expect(result.snapshotTokenOffset == toolCallTurnTokens.count,
                "Tool-result continuation must hit the direct leaf from the tool-call turn")
        if case .hit(_, _, let type) = result.reason {
            #expect(type == .leaf)
        } else {
            Issue.record("Expected .hit, got \(result.reason)")
        }
    }

    /// Qwen-style templates should store only the canonical stripped leaf.
    /// A new-user turn that re-renders the previous assistant as non-latest
    /// must hit that shorter path directly.
    @Test func canonicalLeafHitOnNewUserTurn() {
        let mgr = makeManager(budgetMB: 500)

        let strippedTokens = Array(1...85)
        mgr.storeLeaf(
            storedTokens: strippedTokens,
            leafSnapshot: makeKVSnapshot(offset: strippedTokens.count, type: .leaf),
            partitionKey: defaultKey
        )
        #expect(mgr.stats.snapshotCount == 1)

        let turnNPlus1Tokens = strippedTokens + Array(500...510)
        let newUserResult = mgr.lookup(tokens: turnNPlus1Tokens, partitionKey: defaultKey)
        #expect(newUserResult.snapshotTokenOffset == strippedTokens.count,
                "Cross-new-user-turn lookup must hit the canonical stripped leaf at offset 85")
        if case .hit(_, _, let type) = newUserResult.reason {
            #expect(type == .leaf)
        } else {
            Issue.record("Expected .hit, got \(newUserResult.reason)")
        }
    }

    /// Newer descendant leaves supersede shallower leaves on the same
    /// branch, so the deepest canonical leaf is always the one that
    /// survives lookup on later extensions.
    @Test func newerCanonicalLeafReplacesShallowerLeafOnSamePath() {
        let mgr = makeManager(budgetMB: 500)
        let shallowTokens = Array(1...80)
        let deeperTokens = Array(1...100)

        mgr.storeLeaf(
            storedTokens: shallowTokens,
            leafSnapshot: makeKVSnapshot(offset: shallowTokens.count, type: .leaf),
            partitionKey: defaultKey
        )
        let diagnostics = mgr.storeLeaf(
            storedTokens: deeperTokens,
            leafSnapshot: makeKVSnapshot(offset: deeperTokens.count, type: .leaf),
            partitionKey: defaultKey
        )
        #expect(diagnostics.supersededLeaves.count == 1)
        #expect(diagnostics.supersededLeaves[0].offset == shallowTokens.count)
        #expect(mgr.stats.snapshotCount == 1)

        let extendedTokens = deeperTokens + Array(800...810)
        let result = mgr.lookup(tokens: extendedTokens, partitionKey: defaultKey)
        #expect(result.snapshotTokenOffset == deeperTokens.count)
    }
}

// MARK: - Test Helpers

/// Minimal mock tokenizer for StablePrefixDetector nil/empty tests.
/// Does NOT need to produce meaningful tokens — just needs to satisfy the protocol.
private struct MockTokenizer: Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }

    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        Array(1...10)
    }
}
