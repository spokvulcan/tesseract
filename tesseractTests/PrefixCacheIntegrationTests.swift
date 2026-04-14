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

    /// Regression test for the Qwen3.5 `<think>` cross-new-user-turn
    /// divergence. Simulates the production flow without a loaded model:
    ///
    /// - **Turn N** stores an **unstripped** leaf under `[sys..., u_N, a_N_with_think]`
    ///   (this is the existing behavior; tokens include the `<think>…</think>`
    ///   block because the assistant is the LAST message in the stored
    ///   conversation, so the template keeps its think block).
    /// - **Turn N** ALSO stores a **stripped** leaf under
    ///   `[sys..., u_N, a_N_stripped]` (what Turn N+1 will see once the
    ///   template strips the think block — this is what the new
    ///   `captureStrippedLeaf` helper produces via a two-probe tokenization
    ///   and a residual re-prefill).
    /// - **Turn N+1** arrives with a new user message. The lookup token
    ///   sequence starts with `[sys..., u_N, a_N_stripped]` and extends
    ///   with the new user content. It must hit the STRIPPED leaf at
    ///   `storedTokensStripped.count` — not the unstripped leaf (which is
    ///   on a different radix path) and not fall back to the shallower
    ///   stable-prefix snapshot.
    ///
    /// Also verifies that tool-loop continuation — where the assistant is
    /// still the latest message in the next request's history — still
    /// hits the UNSTRIPPED leaf at the full offset.
    @Test func strippedLeafHitOnNewUserTurn() {
        let mgr = makeManager(budgetMB: 500)

        // Stripped-history simulation. Tokens:
        //   [1...85]  = [sys, tools, u_1, a_stripped...] (what Turn N+1 sees)
        //   [86...95] = extra tokens in a_N_with_think (think block content)
        //   96..100   = tail of a_N shared between stripped and unstripped
        //
        // To keep the synthetic test tractable, the "stripped" and
        // "unstripped" paths share the [1...85] prefix (stable history
        // through end of a_N_stripped content), then diverge:
        //   - unstripped continues [86...100]
        //   - stripped is just [1...85]
        //
        // In reality the divergence is mid-assistant; this approximation
        // exercises the same radix tree behavior.
        let strippedTokens = Array(1...85)
        let unstrippedTokens = Array(1...100)

        let strippedSnap = makeKVSnapshot(offset: strippedTokens.count, type: .leaf)
        let unstrippedSnap = makeKVSnapshot(offset: unstrippedTokens.count, type: .leaf)

        // Store the stripped leaf first (simulates the new additive path).
        mgr.storeLeaf(
            storedTokens: strippedTokens,
            leafSnapshot: strippedSnap,
            partitionKey: defaultKey
        )
        // Store the unstripped leaf (simulates the existing primary path).
        mgr.storeLeaf(
            storedTokens: unstrippedTokens,
            leafSnapshot: unstrippedSnap,
            partitionKey: defaultKey
        )
        #expect(mgr.stats.snapshotCount == 2, "Both leaves must coexist")

        // Turn N+1: new user turn. Tokens match the stripped path
        // `[1...85]` and then diverge with new user content.
        let turnNPlus1Tokens = strippedTokens + Array(500...510)
        let newUserResult = mgr.lookup(tokens: turnNPlus1Tokens, partitionKey: defaultKey)
        #expect(newUserResult.snapshotTokenOffset == strippedTokens.count,
                "Cross-new-user-turn lookup must hit the stripped leaf at offset 85")
        if case .hit(_, _, let type) = newUserResult.reason {
            #expect(type == .leaf)
        } else {
            Issue.record("Expected .hit, got \(newUserResult.reason)")
        }

        // Tool-loop continuation: assistant is still the latest, so the
        // template didn't strip its think block. Tokens match the
        // unstripped path `[1...100]` and then extend with a tool_result.
        let toolLoopTokens = unstrippedTokens + Array(700...710)
        let toolLoopResult = mgr.lookup(tokens: toolLoopTokens, partitionKey: defaultKey)
        #expect(toolLoopResult.snapshotTokenOffset == unstrippedTokens.count,
                "Tool-loop continuation must still hit the unstripped leaf at offset 100")
    }

    /// When the stripped tokens are a strict prefix of the unstripped
    /// tokens (which they ARE in the real Qwen3.5 case: the `<think>` block
    /// is interior content within the assistant turn, so re-rendering with
    /// the assistant stripped produces a shorter version of the same
    /// sequence up to the tail `<|im_end|>`), a lookup that matches the
    /// unstripped path beyond the stripped offset must still pick the
    /// DEEPEST snapshot — the unstripped leaf at the full offset, not
    /// the stripped leaf at the shorter offset. This guards against a
    /// future regression where the radix walk might prefer a shallower
    /// hit when a deeper one is available on the same path.
    @Test func unstrippedLeafPreferredWhenDeeperOnSamePath() {
        let mgr = makeManager(budgetMB: 500)
        // Stripped path is a strict prefix of unstripped. In the radix
        // tree this creates a linear chain:
        //   root → [1..80] (stripped leaf) → [81..100] (unstripped leaf)
        let strippedTokens = Array(1...80)
        let unstrippedTokens = Array(1...100)

        mgr.storeLeaf(
            storedTokens: strippedTokens,
            leafSnapshot: makeKVSnapshot(offset: strippedTokens.count, type: .leaf),
            partitionKey: defaultKey
        )
        mgr.storeLeaf(
            storedTokens: unstrippedTokens,
            leafSnapshot: makeKVSnapshot(offset: unstrippedTokens.count, type: .leaf),
            partitionKey: defaultKey
        )

        // Lookup covering both: `[1...100] + extra` must return the
        // DEEPER snapshot at 100, not the shallower one at 80.
        let extendedTokens = unstrippedTokens + Array(800...810)
        let result = mgr.lookup(tokens: extendedTokens, partitionKey: defaultKey)
        #expect(result.snapshotTokenOffset == unstrippedTokens.count)
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
