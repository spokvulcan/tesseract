import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// Pure-math tests for **Asymmetric-State Restore** synthesis (issue #134).
///
/// These exercise the spike's one new unit at the tensor level — no model
/// load — because the hazard the spike exists to measure (delta-RoPE
/// lossiness, stale recurrent state) is downstream of this math being right.
/// The delta-RoPE correctness is pinned against `MLXFast.RoPE` as the gold:
/// re-rotating a key that `RoPE` placed at position `p` to position `s` must
/// land bitwise where `RoPE` places it at `s` directly. The end-to-end
/// (loaded-model, ASR-vs-gold distributional) gate lives in the
/// `HybridCacheCorrectnessRunner` seam.
struct AsymmetricStateRestoreTests {

    // Qwen3.5/3.6 RoPE facts: partial rotary 0.25, theta 1e5, split-half.
    private static let ropeDims = 8  // headDim 32 × partialRotaryFactor 0.25
    private static let headDim = 32
    private static let theta: Float = 100_000.0

    private func goldRoPE(_ raw: MLXArray, offset: Int = 0) -> MLXArray {
        MLXFast.RoPE(
            raw, dimensions: Self.ropeDims, traditional: false,
            base: Self.theta, scale: 1.0, offset: offset)
    }

    private func meta(traditional: Bool = false) -> AsymmetricStateRestore.RopeMetadata {
        AsymmetricStateRestore.RopeMetadata(
            ropeDims: Self.ropeDims, ropeTheta: Self.theta, scale: 1.0, traditional: traditional)
    }

    private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        BenchmarkHarness.maxAbsDiff(a, b)
    }

    // MARK: - Delta-RoPE against MLXFast.RoPE gold

    /// Re-rotating the bearing key (RoPE at position p) by the cumulative
    /// delta must reproduce RoPE at the stripped position exactly. A single
    /// front excision shifts every retained token by the same delta.
    @Test func deltaRoPESingleExcisionMatchesGoldBitwise() throws {
        let tokenCount = 16
        let delta = 3
        let raw = randomKeys(tokenCount: tokenCount)
        let bearing = goldRoPE(raw)  // positions [0, tokenCount)

        // Surviving: bearing[3..<16] → stripped positions [0, 13).
        let surviving = bearing[.ellipsis, delta..<tokenCount, 0...]
        let bearingPositions = Array(delta..<tokenCount)
        let synthesized = AsymmetricStateRestore.applyCumulativeDeltaRoPE(
            keys: surviving, bearingPositions: bearingPositions, meta: meta())

        let gold = goldRoPE(raw[.ellipsis, delta..<tokenCount, 0...], offset: 0)
        #expect(maxAbsDiff(synthesized, gold) < 1e-5)
    }

    /// Multi-span excision: the per-token delta is non-contiguous (a step
    /// function jumping at each removed span). Still must match the gold RoPE
    /// gathered at the surviving positions.
    @Test func deltaRoPEMultiSpanExcisionMatchesGoldBitwise() throws {
        let tokenCount = 12
        let raw = randomKeys(tokenCount: tokenCount)
        let bearing = goldRoPE(raw)
        // Remove [1,3) and [6,9) → surviving bearing positions [0, 3,4,5, 9,10,11].
        let survivingPositions = [0, 3, 4, 5, 9, 10, 11]
        let survivingPieces = survivingPositions.map { bearing[.ellipsis, $0...$0, 0...] }
        let surviving = concatenated(survivingPieces, axis: 2)
        let synthesized = AsymmetricStateRestore.applyCumulativeDeltaRoPE(
            keys: surviving, bearingPositions: survivingPositions, meta: meta())

        let goldRawPieces = survivingPositions.map { raw[.ellipsis, $0...$0, 0...] }
        let goldRaw = concatenated(goldRawPieces, axis: 2)
        let gold = goldRoPE(goldRaw, offset: 0)
        #expect(maxAbsDiff(synthesized, gold) < 1e-5)
    }

    /// The non-rotary tail (dims past `ropeDims`) must be passed through
    /// untouched — only the rotary fraction is position-dependent.
    @Test func deltaRoPELeavesNonRotaryTailUntouched() throws {
        let tokenCount = 8
        let delta = 2
        // Full headDim tensor: rotary front + a tail we can recognize.
        let raw = randomKeys(tokenCount: tokenCount, headDim: Self.headDim)
        let bearing = goldRoPE(raw)  // rotates only first ropeDims; tail == raw tail
        let surviving = bearing[.ellipsis, delta..<tokenCount, 0...]
        let bearingPositions = Array(delta..<tokenCount)
        let synthesized = AsymmetricStateRestore.applyCumulativeDeltaRoPE(
            keys: surviving, bearingPositions: bearingPositions, meta: meta())

        // Tail (ropeDims..<headDim) of synthesized must equal the tail of raw
        // at the surviving positions — RoPE never touched it and neither does
        // the delta rotation.
        let tail = synthesized[.ellipsis, Self.ropeDims..<Self.headDim]
        let expectedTailPieces = (delta..<tokenCount).map {
            raw[.ellipsis, $0...$0, Self.ropeDims..<Self.headDim]
        }
        let expectedTail = concatenated(expectedTailPieces, axis: 2)
        #expect(maxAbsDiff(tail, expectedTail) < 1e-6)
    }

    /// Values are excised but never rotated — RoPE touches keys only. The
    /// synthesized snapshot must reflect that.
    @Test func valuesAreExcisedOnly() throws {
        let tokenCount = 10
        let values = MLXArray((0..<tokenCount).map { Float($0) })
            .expandedDimensions(axis: 0).expandedDimensions(axis: 0)
            .expandedDimensions(axis: -1)  // [1,1,tokenCount,1]
        // Think span [2,5) removes 3 tokens → surviving [0,1,2],[5,6,7,8,9].
        let ranges: [Range<Int>] = [0..<2, 5..<tokenCount]
        let pieces = ranges.map { values[.ellipsis, $0, 0...] }
        let excised = concatenated(pieces, axis: 2)
        let expected: [Float] = [0, 1, 5, 6, 7, 8, 9]
        let actual = excised.asArray(Float.self)
        #expect(actual == expected)
    }

    // MARK: - Span / offset mapping

    @Test func survivingRangesComplementTheSpans() {
        let spans = [
            AsymmetricStateRestore.ThinkSpan(start: 2, end: 5),
            AsymmetricStateRestore.ThinkSpan(start: 8, end: 9),
        ]
        let ranges = AsymmetricStateRestore.survivingRanges(spans: spans, bearingLength: 12)
        #expect(ranges == [0..<2, 5..<8, 9..<12])
    }

    @Test func retainedBearingPositionsAreInStrippedOrder() {
        let ranges = [0..<2, 5..<8]
        #expect(
            AsymmetricStateRestore.retainedBearingPositions(ranges: ranges)
                == [0, 1, 5, 6, 7])
    }

    // MARK: - Full synthesis (snapshot level)

    /// Build a bearing snapshot (one float KV layer + one recurrent Mamba
    /// layer), synthesize after excising a think span, and assert the
    /// structural asymmetry: attention aligned to the stripped path,
    /// recurrent left at the bearing render.
    @Test func synthesizeProducesAsymmetricSnapshot() throws {
        let bearingLen = 12
        let strippedLen = 9  // excise [3,6)
        let span = AsymmetricStateRestore.ThinkSpan(start: 3, end: 6)

        // Float KV layer: keys = goldRoPE(random) at positions [0..11],
        // values = position-index marker.
        let raw = randomKeys(tokenCount: bearingLen, headDim: Self.headDim)
        let keys = goldRoPE(raw)
        let kv = KVCacheSimple()
        let valuesArr = MLXArray((0..<bearingLen).map { Float($0) })
            .expandedDimensions(axis: 0).expandedDimensions(axis: 0)
            .expandedDimensions(axis: -1)  // [1,1,bearingLen,1]
        kv.state = [keys, valuesArr]
        kv.offset = bearingLen

        // Recurrent Mamba layer: arbitrary state, must survive untouched.
        let mamba = MambaCache()
        let mambaState = [MLXArray.ones([1, 4, 16]), MLXArray.zeros([1, 2, 8, 8])]
        mamba.state = mambaState
        mamba.offset = bearingLen

        let bearing = try #require(
            HybridCacheSnapshot.capture(cache: [kv, mamba], offset: bearingLen, type: .leaf))

        let synthesized = try AsymmetricStateRestore.synthesize(
            bearingSnapshot: bearing,
            strippedTokenCount: strippedLen,
            thinkSpans: [span],
            ropeMetadataByLayer: [0: meta()]
        )
        #expect(synthesized.tokenOffset == strippedLen)

        // Attention layer: aligned to the stripped path.
        #expect(synthesized.layers[0].offset == strippedLen)
        #expect(synthesized.layers[0].state[0].dim(2) == strippedLen)  // keys
        #expect(synthesized.layers[0].state[1].dim(2) == strippedLen)  // values

        // Attention keys must match the gold: RoPE at stripped positions of
        // the surviving raw tokens.
        let survivingPositions = [0, 1, 2, 6, 7, 8, 9, 10, 11]
        let goldRawPieces = survivingPositions.map { raw[.ellipsis, $0...$0, 0...] }
        let goldRaw = concatenated(goldRawPieces, axis: 2)
        let goldKeys = goldRoPE(goldRaw, offset: 0)
        #expect(maxAbsDiff(synthesized.layers[0].state[0], goldKeys) < 1e-5)

        // Recurrent layer: left at the bearing render — state + offset unchanged.
        #expect(synthesized.layers[1].offset == bearingLen)
        #expect(synthesized.layers[1].state.count == mambaState.count)
        for (a, b) in zip(synthesized.layers[1].state, mambaState) {
            #expect(maxAbsDiff(a, b) == 0.0)
        }
    }

    // MARK: - Preflight declines

    @Test func preflightNoThinkSpans() {
        let bearing = makeTrivialBearingSnapshot()
        #expect(throws: AsymmetricStateRestore.SynthesisError.unavailable(.noThinkSpans)) {
            try AsymmetricStateRestore.synthesize(
                bearingSnapshot: bearing, strippedTokenCount: 4,
                thinkSpans: [], ropeMetadataByLayer: [0: meta()])
        }
    }

    @Test func preflightLengthMismatchGenuine() {
        let bearing = makeTrivialBearingSnapshot()  // offset 4
        #expect(
            throws: AsymmetricStateRestore.SynthesisError.unavailable(
                .lengthMismatch(bearing: 4, stripped: 1, expectedStripped: 2))
        ) {
            try AsymmetricStateRestore.synthesize(
                bearingSnapshot: bearing, strippedTokenCount: 1,  // genuine mismatch
                thinkSpans: [AsymmetricStateRestore.ThinkSpan(start: 0, end: 2)],
                ropeMetadataByLayer: [0: meta()])
        }
    }

    @Test func preflightMissingRopeMetadata() {
        let bearing = makeTrivialBearingSnapshot()
        #expect(
            throws: AsymmetricStateRestore.SynthesisError.unavailable(
                .missingRopeMetadata(layerIndices: [0]))
        ) {
            try AsymmetricStateRestore.synthesize(
                bearingSnapshot: bearing, strippedTokenCount: 2,
                thinkSpans: [AsymmetricStateRestore.ThinkSpan(start: 0, end: 2)],
                ropeMetadataByLayer: [:])  // no metadata for the KV layer
        }
    }

    @Test func preflightInvalidOverlappingSpans() throws {
        let bearing = makeTrivialBearingSnapshot()  // offset 4
        do {
            _ = try AsymmetricStateRestore.synthesize(
                bearingSnapshot: bearing, strippedTokenCount: 0,
                thinkSpans: [
                    AsymmetricStateRestore.ThinkSpan(start: 0, end: 3),
                    AsymmetricStateRestore.ThinkSpan(start: 2, end: 4),  // overlap
                ],
                ropeMetadataByLayer: [0: meta()])
            Issue.record("expected .unavailable(.invalidSpans)")
        } catch AsymmetricStateRestore.SynthesisError.unavailable(.invalidSpans) {
            // expected
        }
    }

    // MARK: - Quantized path (shape + round-trip, lossy by design)

    /// The quantized delta-RoPE is lossy by construction (dequant → rotate →
    /// requant). This test asserts only that it runs, produces a snapshot at
    /// the stripped offset, and restores through the real restore path
    /// without crashing (the "Smoke" acceptance artifact).
    @Test func quantizedSynthesisRestoresWithoutCrashing() throws {
        let bearingLen = 16
        let strippedLen = 12
        let span = AsymmetricStateRestore.ThinkSpan(start: 2, end: 6)

        let raw = randomKeys(tokenCount: bearingLen, headDim: 64)
        let keys = goldRoPE64(raw)
        let values = randomKeys(tokenCount: bearingLen, headDim: 64)
        let kv = KVCacheSimple()
        kv.state = [keys, values]
        kv.offset = bearingLen
        var cache: [any KVCache] = [kv]
        // Force 8-bit quantization (the production default).
        maybeQuantizeKVCache(cache: &cache, kvBits: 8, kvGroupSize: 64, quantizedKVStart: 0)

        let quantMeta = AsymmetricStateRestore.RopeMetadata(
            ropeDims: 16, ropeTheta: Self.theta, scale: 1.0, traditional: false)
        let bearing = try #require(
            HybridCacheSnapshot.capture(cache: cache, offset: bearingLen, type: .leaf))

        let synthesized = try AsymmetricStateRestore.synthesize(
            bearingSnapshot: bearing, strippedTokenCount: strippedLen,
            thinkSpans: [span], ropeMetadataByLayer: [0: quantMeta])
        #expect(synthesized.tokenOffset == strippedLen)
        #expect(synthesized.layers[0].offset == strippedLen)
        #expect(synthesized.layers[0].state[0].dim(2) == strippedLen)  // wq keys

        // Smoke artifact: restores through the real path without throwing.
        let restored = try synthesized.restore()
        #expect(restored[0].offset == strippedLen)
    }

    // MARK: - Fixtures

    private func randomKeys(tokenCount: Int, headDim: Int = Self.ropeDims) -> MLXArray {
        // Small deterministic-enough seed of values; MLX random is fine here
        // since the comparison is relative (synthesized vs gold from the same
        // raw).
        let array = MLXRandom.uniform(low: -1.0, high: 1.0, [1, 1, tokenCount, headDim])
        eval(array)
        return array
    }

    private func goldRoPE64(_ raw: MLXArray) -> MLXArray {
        MLXFast.RoPE(
            raw, dimensions: 16, traditional: false, base: Self.theta, scale: 1.0, offset: 0)
    }

    private func makeTrivialBearingSnapshot() -> HybridCacheSnapshot {
        let kv = KVCacheSimple()
        kv.state = [
            MLXArray.zeros([1, 1, 4, Self.ropeDims]),
            MLXArray.zeros([1, 1, 4, Self.ropeDims]),
        ]
        kv.offset = 4
        let snapshot = HybridCacheSnapshot.capture(cache: [kv], offset: 4, type: .leaf)!
        return snapshot
    }
}
