import Foundation
import MLX
import MLXLMCommon

// swiftlint:disable type_body_length
/// **Asymmetric-State Restore** (ASR) — CONTEXT.md, ADR-0009, issue #134.
///
/// The experimental single-prefill counter to the **Think-Strip Rewind**.
/// Rather than re-prefilling the think-stripped **Tool Stretch** (the
/// **Speculative Canonical Prefill** path, which loses its race as stretches
/// grow), ASR *derives* a snapshot for the stripped token path from the
/// think-bearing snapshot by pure array surgery — no model forward pass:
///
/// 1. **N middle-excisions** — one per span the canonical future render
///    drops, derived by **Render-Diff Excision** (aligning the bearing token
///    path against the future shared path — never by scanning for literal
///    `<think>` delimiters, which conversation *content* can carry as data;
///    the 2026-06-27 live declines in issue #134 were exactly that) —
///    cutting those K/V ranges out of each sliceable (attention) layer along
///    the token axis and concatenating the survivors.
/// 2. **Cumulative delta-RoPE** on the retained attention keys: every retained
///    token's absolute position shifts left by the cumulative removed-think
///    length, and the cached keys are already post-RoPE, so only the rotary
///    fraction of each head dim is re-rotated to its new position.
/// 3. The non-sliceable recurrent (**MambaCache**/**ArraysCache**) state is
///    **left at the bearing render** — kept as-is, advanced through the
///    stretch with the think blocks included, because recurrent state is
///    irreversible (ADR-0007, ADR-0009).
///
/// The result is a deliberate per-layer-family asymmetry: attention layers
/// aligned to the stripped path, recurrent layers still carrying the bearing
/// render — hence *asymmetric*. Correctness is unproven by construction
/// (recurrent state is stale by design); this module is the unit the spike
/// measures. It plugs into the existing **Speculative Canonical Prefill**
/// scheduling and swaps only the body.
///
/// All operations are pure array ops on already-captured snapshot arrays, so
/// they run on a Metal-affine thread (`container.perform`) but perform no
/// model work. KV quantizes to 8-bit by default (`AgentGeneration.kvBits = 8`),
/// so delta-RoPE is lossy there (dequant → rotate → requant on the rotary
/// fraction); exact in float KV. Synthesized snapshots are **RAM-only**: their
/// sliceable layers are a concatenation of non-contiguous pieces, which does
/// not fit the segment-chain contiguous-suffix model (ADR-0010).
nonisolated enum AsymmetricStateRestore {

    /// Per-attention-layer RoPE metadata, read from the live model config
    /// (not hard-coded) so the ASR gate is the render context, not a model
    /// name (user story #11). Carries exactly what the delta rotation needs.
    struct RopeMetadata: Sendable, Equatable {
        /// The rotary-fraction width `partialRotaryFactor × headDim`. Only the
        /// first `ropeDims` elements of each head are rotated by RoPE; the rest
        /// is position-free and untouched by the delta rotation.
        let ropeDims: Int
        /// `rope_theta`.
        let ropeTheta: Float
        /// Position scale from the RoPE layer (`1.0` for the default rope type
        /// Qwen3.5/3.6 ship). Folds into the rotation angle exactly as
        /// `MLXFast.RoPE` applies it.
        let scale: Float
        /// `true` for the interleaved (traditional) pairing, `false` for the
        /// split-half pairing Qwen3.5/3.6 use.
        let traditional: Bool

        init(ropeDims: Int, ropeTheta: Float, scale: Float = 1.0, traditional: Bool = false) {
            self.ropeDims = ropeDims
            self.ropeTheta = ropeTheta
            self.scale = scale
            self.traditional = traditional
        }

        /// Build the per-attention-layer metadata map keyed by cache-array
        /// layer index. Only sliceable (attention) layers get an entry —
        /// recurrent layers carry no RoPE. Pure — unit-tested directly.
        static func mapByLayer(
            for layerClassNames: [String],
            ropeDims: Int,
            ropeTheta: Float,
            scale: Float = 1.0,
            traditional: Bool = false
        ) -> [Int: AsymmetricStateRestore.RopeMetadata] {
            let metadata = AsymmetricStateRestore.RopeMetadata(
                ropeDims: ropeDims, ropeTheta: ropeTheta,
                scale: scale, traditional: traditional)
            var map: [Int: AsymmetricStateRestore.RopeMetadata] = [:]
            for (idx, className) in layerClassNames.enumerated()
            where AsymmetricStateRestore.isSliceableLayerClass(className) {
                map[idx] = metadata
            }
            return map
        }

        /// The RoPE scalars extracted from a model config, decoupled from the
        /// per-layer map so the live path can read them once at model load
        /// (from the config directory) and rebuild the layer-keyed map per
        /// bearing snapshot's layer classes. Pure value, no I/O.
        struct RopeScalars: Sendable, Equatable {
            let ropeDims: Int
            let ropeTheta: Float
            let scale: Float
            let traditional: Bool
        }

        /// Read the RoPE scalars from a Qwen3.5/3.6 model directory's
        /// `config.json`, walking the VLM `text_config` nesting and the
        /// `rope_parameters` merge the vendor decoder performs. Returns `nil`
        /// when the config lacks the scalars ASR needs (the caller treats that
        /// as a preflight-unavailable decline). Best-effort: only the default
        /// rope type (`scale = 1.0`, `traditional = false`) is fully modeled
        /// here, which is what Qwen3.5/3.6 ship; a linear `rope_scaling.factor`
        /// folds into `scale = 1 / factor`.
        static func scalars(from modelDirectory: URL) -> RopeScalars? {
            let configURL = modelDirectory.appendingPathComponent("config.json")
            guard let data = try? Data(contentsOf: configURL),
                let raw = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            else { return nil }
            // VLM nesting: prefer text_config; fall back to top-level for a
            // pure-LLM config.
            var config = raw
            if let textConfig = raw["text_config"] as? [String: Any] {
                config = merged(into: textConfig, overridingWith: raw)
            }
            // rope_parameters (Qwen3.5) overrides the bare keys.
            if let rope = config["rope_parameters"] as? [String: Any] {
                config = merged(into: config, overridingWith: rope)
            }
            guard let theta = anyConfig(config, "rope_theta") ?? anyConfig(raw, "rope_theta"),
                let partial = anyConfig(config, "partial_rotary_factor")
                    ?? anyConfig(raw, "partial_rotary_factor")
            else { return nil }
            let headDim: Int
            if let hd = anyConfig(config, "head_dim") ?? anyConfig(raw, "head_dim") {
                headDim = Int(hd)
            } else {
                headDim = Self.defaultHeadDim(from: config)
            }
            let ropeDims = max(1, Int(Float(headDim) * Float(partial)))
            var scale = 1.0
            if let scaling = config["rope_scaling"] as? [String: Any],
                let type = scaling["type"] as? String ?? scaling["rope_type"] as? String,
                type == "linear",
                let factor = scaling["factor"] as? Double, factor != 0
            {
                scale = 1.0 / factor
            }
            return RopeScalars(
                ropeDims: ropeDims, ropeTheta: Float(theta),
                scale: Float(scale), traditional: false)
        }

        /// Convenience: read the scalars and build the layer-keyed map in one
        /// call (the path the loaded-model harness uses).
        static func readFromConfigDirectory(
            _ modelDirectory: URL,
            layerClassNames: [String]
        ) -> [Int: AsymmetricStateRestore.RopeMetadata]? {
            guard let scalars = scalars(from: modelDirectory) else { return nil }
            return mapByLayer(
                for: layerClassNames, ropeDims: scalars.ropeDims,
                ropeTheta: scalars.ropeTheta, scale: scalars.scale,
                traditional: scalars.traditional)
        }

        /// Resolve a config scalar that may be serialized as Double, Int, or
        /// numeric String.
        private static func anyConfig(_ config: [String: Any], _ key: String) -> Double? {
            guard let value = config[key] else { return nil }
            if let d = value as? Double { return d }
            if let i = value as? Int { return Double(i) }
            if let s = value as? String { return Double(s) }
            return nil
        }

        private static func defaultHeadDim(from config: [String: Any]) -> Int {
            guard let hidden = anyConfig(config, "hidden_size"),
                let heads = anyConfig(config, "num_attention_heads"), heads != 0
            else { return 128 }
            return Int(hidden / heads)
        }

        private static func merged(
            into base: [String: Any], overridingWith override: [String: Any]
        ) -> [String: Any] {
            var merged = base
            for (k, v) in override { merged[k] = v }
            return merged
        }
    }

    /// One token span to excise, in **bearing-path** token offsets. Half-open
    /// `[start, end)` — removed from each sliceable layer's K/V. Derived by
    /// **Render-Diff Excision** (`renderDiffExcision`): a span is a token run
    /// the canonical future render drops, typically a `<think>` block plus its
    /// template-owned separators, but defined by the render diff — never by
    /// scanning for literal delimiter text, which conversation content can
    /// carry as data (issue #134's live declines).
    struct ExcisionSpan: Sendable, Equatable, Hashable {
        let start: Int
        let end: Int

        /// Number of tokens this span removes.
        var length: Int { end - start }
    }

    /// Why synthesis declined before touching any state. Decided purely from
    /// the inputs, so a pass that will return one of these spends **no**
    /// idle-window budget and the caller falls back to the existing
    /// speculative prefill in full (user story #12). Distinct from a
    /// mid-synthesis failure, which throws after some budget is spent.
    enum UnavailableReason: Sendable, Equatable {
        /// The render diff dropped nothing — the future path extends the
        /// bearing path unchanged, so there is no single-prefill benefit and
        /// synthesis would just copy the bearing capture.
        case nothingExcised
        /// Alignment found no shared prefix at all between the bearing path
        /// and the future path (a different render entirely).
        case noAlignedPrefix
        /// The aligned depth sits inside the image prefix — a restore there
        /// has no valid boundary (mirrors the pass's boundary guards).
        case belowWarmOffset(aligned: Int, minimum: Int)
        /// The aligned depth consumed the whole future path — no residual for
        /// the extension prefill to cover, so no admissible deeper leaf.
        case noResidual(aligned: Int, admit: Int)
        /// Spans not sorted, overlapping, empty, or out of the bearing path.
        case invalidSpans(detail: String)
        /// `strippedTokenCount` does not equal `bearingLength − Σ span lengths`.
        case lengthMismatch(bearing: Int, stripped: Int, expectedStripped: Int)
        /// `ropeMetadataByLayer` is missing entries for these sliceable layers.
        case missingRopeMetadata(layerIndices: [Int])
        /// A sliceable layer is a class the spike does not operate on yet
        /// (e.g. `RotatingKVCache`/`ChunkedKVCache`, whose temporal-order /
        /// chunk structure complicates token-axis excision).
        case unsupportedSliceLayer(layerIndex: Int, className: String)
        /// The bearing snapshot has no layers to synthesize from.
        case emptySnapshot
    }

    /// A synthesis failure. The case distinguishes the two fallback channels
    /// the caller (the speculative-prefill body) treats differently.
    enum SynthesisError: Error, Equatable {
        /// Preflight: decided before any surgery, so the pass spent **no**
        /// idle-window budget. The caller falls back to the existing
        /// speculative prefill in full (user story #12).
        case unavailable(UnavailableReason)
        /// Surgery had begun (some idle-window budget spent) before the shape
        /// mismatch was discovered. The caller aborts and admits nothing
        /// deeper than the canonical leaf rather than chaining a full
        /// re-prefill onto the sunk cost (user story #14).
        case midSynthesis(MidSynthesisError)
    }

    /// A mid-synthesis failure detail (see ``SynthesisError/midSynthesis(_:)``).
    struct MidSynthesisError: Error, Equatable {
        let detail: String

        init(_ detail: String) { self.detail = detail }
    }

    // MARK: - Synthesis

    /// Synthesize a stripped-path snapshot from the think-bearing capture.
    ///
    /// - Preflight: every check decidable from the inputs alone runs first, so
    ///   an `.unavailable` outcome spends no idle-window budget.
    /// - Surgery: the excision + delta-RoPE + recurrent-as-is pass. A shape
    ///   mismatch discovered here throws `.midSynthesis`; the caller degrades
    ///   to a cache miss (admit nothing deeper than the canonical leaf).
    ///
    /// The returned snapshot's `tokenOffset` is `strippedTokenCount`; each
    /// sliceable layer's offset is `strippedTokenCount` (attention aligned to
    /// the stripped path); each non-sliceable recurrent layer is kept exactly
    /// as the bearing capture left it — its state and offset still reflect the
    /// full bearing render.
    static func synthesize(
        bearingSnapshot: HybridCacheSnapshot,
        strippedTokenCount: Int,
        spans: [ExcisionSpan],
        ropeMetadataByLayer: [Int: RopeMetadata]
    ) throws -> HybridCacheSnapshot {
        // 1. Preflight — spans.
        guard !bearingSnapshot.layers.isEmpty else {
            throw SynthesisError.unavailable(.emptySnapshot)
        }
        guard !spans.isEmpty else {
            throw SynthesisError.unavailable(.nothingExcised)
        }
        let sortedSpans = try validateSpans(
            spans, bearingLength: bearingSnapshot.tokenOffset
        )
        let excisedTotal = sortedSpans.reduce(0) { $0 + $1.length }
        let expectedStripped = bearingSnapshot.tokenOffset - excisedTotal
        guard expectedStripped == strippedTokenCount else {
            throw SynthesisError.unavailable(
                .lengthMismatch(
                    bearing: bearingSnapshot.tokenOffset,
                    stripped: strippedTokenCount,
                    expectedStripped: expectedStripped
                ))
        }

        // 2. Preflight — every sliceable layer is supported and has RoPE
        //    metadata. Done before any surgery so these remain no-budget
        //    declines.
        var missingMeta: [Int] = []
        var unsupported: (Int, String)?
        for (idx, layer) in bearingSnapshot.layers.enumerated() {
            guard isSliceable(className: layer.className) else { continue }
            if Self.isUnsupportedSliceClass(layer.className) {
                unsupported = (idx, layer.className)
                break
            }
            if ropeMetadataByLayer[idx] == nil {
                missingMeta.append(idx)
            }
        }
        if let unsupported {
            throw SynthesisError.unavailable(
                .unsupportedSliceLayer(
                    layerIndex: unsupported.0, className: unsupported.1
                ))
        }
        if !missingMeta.isEmpty {
            throw SynthesisError.unavailable(.missingRopeMetadata(layerIndices: missingMeta))
        }

        // 3. Surgery.
        let survivingRanges = Self.survivingRanges(
            spans: sortedSpans, bearingLength: bearingSnapshot.tokenOffset
        )
        // The original bearing position of each retained token, in stripped
        // order — drives the per-token cumulative delta-RoPE.
        let bearingPositions = Self.retainedBearingPositions(ranges: survivingRanges)

        var newLayers: [HybridCacheSnapshot.LayerState] = []
        newLayers.reserveCapacity(bearingSnapshot.layers.count)
        var totalBytes = 0

        for (idx, layer) in bearingSnapshot.layers.enumerated() {
            guard isSliceable(className: layer.className) else {
                // Non-sliceable recurrent state — left at the bearing render
                // (ADR-0007: position-free; ADR-0009: not rewindable). State,
                // metaState, and offset all carry the bearing capture exactly.
                newLayers.append(layer)
                totalBytes += layer.state.reduce(0) { $0 + $1.nbytes }
                continue
            }
            // isSliceable && !isUnsupportedSliceClass ⇒ KVCache / QuantizedKVCache.
            let meta = ropeMetadataByLayer[idx]!
            let synthesized: (state: [MLXArray], metaState: [String])
            switch layer.className {
            case "KVCache", "KVCacheSimple":
                synthesized = try synthesizeKVSimple(
                    state: layer.state,
                    survivingRanges: survivingRanges,
                    bearingPositions: bearingPositions,
                    meta: meta,
                    strippedOffset: expectedStripped
                )
            case "QuantizedKVCache":
                synthesized = try synthesizeQuantized(
                    state: layer.state,
                    metaState: layer.metaState,
                    survivingRanges: survivingRanges,
                    bearingPositions: bearingPositions,
                    meta: meta,
                    strippedOffset: expectedStripped
                )
            default:
                throw SynthesisError.midSynthesis(
                    .init(
                        "layer \(idx) unrecognized sliceable class \(layer.className)"))
            }
            for array in synthesized.state {
                totalBytes += array.nbytes
            }
            newLayers.append(
                HybridCacheSnapshot.LayerState(
                    className: layer.className,
                    state: synthesized.state,
                    metaState: synthesized.metaState,
                    offset: expectedStripped
                )
            )
        }

        return HybridCacheSnapshot(
            tokenOffset: expectedStripped,
            layers: newLayers,
            checkpointType: .leaf,
            memoryBytes: totalBytes,
            createdAt: .now
        )
    }

    // MARK: - Preflight helpers

    /// Sort, bounds-check, and de-duplicate the spans. Returns them sorted by
    /// `start` or throws `.unavailable(.invalidSpans(...))`.
    private static func validateSpans(
        _ spans: [ExcisionSpan], bearingLength: Int
    ) throws -> [ExcisionSpan] {
        let sorted = spans.sorted { $0.start < $1.start }
        var previousEnd = 0
        for span in sorted {
            guard span.start < span.end else {
                throw SynthesisError.unavailable(
                    .invalidSpans(
                        detail: "empty or reversed span [\(span.start), \(span.end))"))
            }
            guard span.start >= previousEnd else {
                throw SynthesisError.unavailable(
                    .invalidSpans(
                        detail: "overlapping spans at \(span.start) (prev end \(previousEnd))"))
            }
            guard span.start >= 0, span.end <= bearingLength else {
                throw SynthesisError.unavailable(
                    .invalidSpans(
                        detail:
                            "span [\(span.start), \(span.end)) out of bounds (bearing \(bearingLength))"
                    ))
            }
            previousEnd = span.end
        }
        return sorted
    }

    /// The token classes the snapshot can excise along the token axis. The
    /// recurrent `MambaCache`/`ArraysCache` family is deliberately excluded
    /// (position-free, not sliceable — ADR-0007/0010). Internal (shared with
    /// ``RopeMetadata/mapByLayer``).
    static func isSliceableLayerClass(_ className: String) -> Bool {
        switch className {
        case "KVCache", "KVCacheSimple", "QuantizedKVCache":
            return true
        default:
            return false
        }
    }

    /// Private alias kept for the in-body preflight/surgery readability.
    private static func isSliceable(className: String) -> Bool {
        isSliceableLayerClass(className)
    }

    /// Sliceable in principle but not operated on by this spike: their
    /// temporal-order (`RotatingKVCache`) or chunk (`ChunkedKVCache`)
    /// structure complicates token-axis excision. Preflight declines so the
    /// pass falls back to the speculative prefill.
    private static func isUnsupportedSliceClass(_ className: String) -> Bool {
        className == "RotatingKVCache" || className == "ChunkedKVCache"
    }

    /// The half-open `[start, end)` token ranges that *survive* excision,
    /// i.e. the complement of the think spans within `[0, bearingLength)`.
    static func survivingRanges(
        spans: [ExcisionSpan], bearingLength: Int
    ) -> [Range<Int>] {
        var ranges: [Range<Int>] = []
        var cursor = 0
        for span in spans {
            if span.start > cursor {
                ranges.append(cursor..<span.start)
            }
            cursor = span.end
        }
        if cursor < bearingLength {
            ranges.append(cursor..<bearingLength)
        }
        return ranges
    }

    /// The original bearing position of every retained token, in stripped
    /// order. Retained token `k` sits at bearing position `positions[k]` and
    /// at stripped position `k`, so its cumulative left-shift is
    /// `k − positions[k]` (≤ 0).
    static func retainedBearingPositions(ranges: [Range<Int>]) -> [Int] {
        ranges.flatMap { $0 }
    }

    // MARK: - Surgery: float KV (KVCacheSimple)

    /// Excise think spans from float K/V and re-rotate retained keys to their
    /// stripped positions. Exact (no quantization). Values are excised only —
    /// RoPE touches keys, not values.
    private static func synthesizeKVSimple(
        state: [MLXArray],
        survivingRanges: [Range<Int>],
        bearingPositions: [Int],
        meta: RopeMetadata,
        strippedOffset: Int
    ) throws -> (state: [MLXArray], metaState: [String]) {
        guard state.count >= 2 else {
            throw SynthesisError.midSynthesis(
                .init("KVCache state has \(state.count) arrays; need ≥ 2"))
        }
        let keys = state[0]
        let values = state[1]
        let excisedKeys = exciseAlongTokenAxis(keys, ranges: survivingRanges)
        let excisedValues = exciseAlongTokenAxis(values, ranges: survivingRanges)
        let rerotatedKeys = applyCumulativeDeltaRoPE(
            keys: excisedKeys, bearingPositions: bearingPositions, meta: meta
        )
        return (
            state: [
                HybridCacheSnapshot.deepCopyState(rerotatedKeys),
                HybridCacheSnapshot.deepCopyState(excisedValues),
            ],
            metaState: [""]
        )
    }

    // MARK: - Surgery: quantized KV

    /// Quantized KV is 8-bit by default, so delta-RoPE is lossy here: route
    /// through upstream's `toUnquantized` → excise + re-rotate →
    /// `toQuantized`. Excision along the token axis is clean (quantization
    /// groups pack along the head dim, ADR-0010), so the loss is confined to
    /// the rotary fraction's quantization group — exactly what the spike
    /// measures.
    private static func synthesizeQuantized(
        state: [MLXArray],
        metaState: [String],
        survivingRanges: [Range<Int>],
        bearingPositions: [Int],
        meta: RopeMetadata,
        strippedOffset: Int
    ) throws -> (state: [MLXArray], metaState: [String]) {
        // metaState = [step, offset, groupSize, bits] (validated on restore).
        guard metaState.count == 4,
            let groupSize = Int(metaState[2]),
            let bits = Int(metaState[3])
        else {
            throw SynthesisError.midSynthesis(
                .init("QuantizedKVCache metaState unreadable: \(metaState)"))
        }
        let step = Int(metaState[0]) ?? 256
        let quant = QuantizedKVCache(groupSize: groupSize, bits: bits)
        quant.state = state
        quant.offset = Int(metaState[1]) ?? strippedOffset

        // Dequantize → float KV, excise + re-rotate, requantize. Reuses
        // upstream's exact dequant/requant so the loss models the production
        // path (the cache's own mode and group layout).
        let asSimple = quant.toUnquantized()
        guard let simpleKeys = asSimple.state.first, asSimple.state.count >= 2 else {
            throw SynthesisError.midSynthesis(.init("quantized→simple produced no state"))
        }
        let simpleValues = asSimple.state[1]
        let excisedKeys = exciseAlongTokenAxis(simpleKeys, ranges: survivingRanges)
        let excisedValues = exciseAlongTokenAxis(simpleValues, ranges: survivingRanges)
        let rerotatedKeys = applyCumulativeDeltaRoPE(
            keys: excisedKeys, bearingPositions: bearingPositions, meta: meta
        )

        let rebuilt = KVCacheSimple()
        rebuilt.state = [rerotatedKeys, excisedValues]
        let requanted = rebuilt.toQuantized(groupSize: groupSize, bits: bits)
        let newState = requanted.state.map { HybridCacheSnapshot.deepCopyState($0) }
        return (
            state: newState,
            metaState: [String(step), String(strippedOffset), String(groupSize), String(bits)]
        )
    }

    // MARK: - Surgery primitives

    /// Concatenate the surviving token ranges along the token axis (axis 2 of
    /// `[B, kvHeads, seqLen, headDim]`). For quantized caches each component
    /// array shares this axis layout, so the same slice applies.
    private static func exciseAlongTokenAxis(
        _ array: MLXArray, ranges: [Range<Int>]
    ) -> MLXArray {
        guard !ranges.isEmpty else {
            return MLXArray.zeros(array.shape, dtype: array.dtype)
        }
        let pieces = ranges.map { array[.ellipsis, $0.lowerBound..<$0.upperBound, 0...] }
        return concatenated(pieces, axis: 2)
    }

    /// Re-rotate retained attention keys from their bearing positions to their
    /// stripped positions. The cached keys are already post-RoPE, so this
    /// applies the rotation that maps position `p` → position `s = p − delta`
    /// for each retained token, touching only the rotary fraction of each head
    /// dim. Exact in float; the loss under quantization happens around this
    /// call (dequant/requant).
    ///
    /// Conventions mirror `MLXFast.RoPE`: split-half pairing for
    /// `traditional == false` (Qwen3.5/3.6), interleaved for `true`; the
    /// angle for frequency `i` at a position shift `d` is
    /// `scale · d · ropeTheta^(−2i/ropeDims)`. The bearing key already carries
    /// the rotation for position `p`; multiplying by the rotation for
    /// `d = s − p` yields the rotation for position `s` (rotations compose
    /// within a pair).
    static func applyCumulativeDeltaRoPE(
        keys: MLXArray, bearingPositions: [Int], meta: RopeMetadata
    ) -> MLXArray {
        let headDim = keys.dim(-1)
        let ropeDims = meta.ropeDims
        let retainedCount = keys.dim(-2)
        precondition(
            bearingPositions.count == retainedCount,
            "bearingPositions (\(bearingPositions.count)) must match retained tokens (\(retainedCount))"
        )
        guard ropeDims >= 2, headDim >= ropeDims else { return keys }
        let half = ropeDims / 2

        // Per-frequency theta: [half]. theta_i = ropeTheta^(-2i/ropeDims).
        let frequencyIndices = MLXArray((0..<half).map { Float($0) })
        let exponents = (frequencyIndices * Float(-2.0 / Float(ropeDims)))
        let theta = MLX.pow(meta.ropeTheta, exponents)  // [half]

        // Per-retained-token delta d_k = strippedPos − bearingPos = k − p_k (≤ 0).
        let deltas = MLXArray(
            (0..<retainedCount).map { k in meta.scale * Float(k - bearingPositions[k]) }
        )  // [retainedCount]
        // angles[k, i] = d_k * theta_i  → shape [retainedCount, half].
        let angles = deltas.expandedDimensions(axis: 1) * theta.expandedDimensions(axis: 0)
        let cos = MLX.cos(angles).asType(keys.dtype)  // [retainedCount, half]
        let sin = MLX.sin(angles).asType(keys.dtype)

        let rotated: MLXArray
        // Broadcast cos/sin over the batch + heads axes: [retainedCount, half]
        // → [1, 1, retainedCount, half].
        let cosB = cos.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
        let sinB = sin.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
        if meta.traditional {
            // Interleaved: pairs are (2i, 2i+1). Reshape the rotary fraction
            // to [..., retainedCount, half, 2] so the last axis is (even, odd),
            // apply the 2D rotation, then flatten back.
            let rotary = keys[.ellipsis, 0..<ropeDims]
                .reshaped(keys.dim(0), keys.dim(1), retainedCount, half, 2)
            let even = rotary[.ellipsis, 0]  // x_2i  → [B, kvH, L, half]
            let odd = rotary[.ellipsis, 1]  // x_2i+1
            // MLXFast.RoPE traditional rotates (even, odd) by +angle.
            let newEven = even * cosB - odd * sinB
            let newOdd = even * sinB + odd * cosB
            let recombined = concatenated(
                [newEven.expandedDimensions(axis: -1), newOdd.expandedDimensions(axis: -1)],
                axis: -1)  // [B, kvH, L, half, 2]
            rotated = recombined.reshaped(
                keys.dim(0), keys.dim(1), retainedCount, ropeDims)
        } else {
            // Split-half: pairs are (i, i + half) for i in [0, half).
            let lo = keys[.ellipsis, 0..<half]  // [B, kvH, L, half]
            let hi = keys[.ellipsis, half..<ropeDims]  // [B, kvH, L, half]
            let newLo = lo * cosB - hi * sinB
            let newHi = lo * sinB + hi * cosB
            rotated = concatenated([newLo, newHi], axis: -1)  // [B, kvH, L, ropeDims]
        }

        // Splice the non-rotary tail (if any) back untouched.
        if ropeDims == headDim {
            return rotated
        }
        let tail = keys[.ellipsis, ropeDims..<headDim]
        return concatenated([rotated, tail], axis: -1)
    }

    // MARK: - Trigger entry point

    /// The ASR-specific inputs the **Speculative Canonical Prefill** seed
    /// carries in addition to its existing fields. When present and ASR is
    /// enabled, the speculative pass attempts synthesis first (the body this
    /// spike swaps in) instead of the restore + chunked-extension re-prefill.
    struct Plan: Sendable {
        /// The think-bearing render through end-of-stretch — *not* the
        /// canonical leaf. Its attention layers hold the `<think>` K/V to
        /// excise; its recurrent state is advanced through the whole stretch
        /// (thinks included) and is kept as-is. Produced and held for the pass
        /// by the trigger (a distinct capture from the canonical leaf, which
        /// sits at the Think-Strip Rewind divergence and covers none of the
        /// stretch — PRD issue #134).
        let bearingSnapshot: HybridCacheSnapshot
        /// The bearing render's token path, aligned to `bearingSnapshot`
        /// (`count == bearingSnapshot.tokenOffset`, guaranteed by the arm-time
        /// offset-match gate). The pass derives the excision spans from it by
        /// **Render-Diff Excision** against the future shared path once the
        /// probe resolves — spans are *not* precomputed at arm time, because
        /// only the actual future render says what the template drops.
        let bearingTokens: [Int]
        /// Per-attention-layer RoPE metadata (read from the live model config,
        /// not hard-coded — user story #11).
        let ropeMetadataByLayer: [Int: RopeMetadata]
        /// Wall-clock seconds the trigger spent capturing the bearing snapshot
        /// (the deep-copy of the live final cache). Reported in the
        /// `AsymmetricStateRestoreEvent` so the PRD's "bearing capture may be
        /// dominant" gate is measurable (issue #134). Synthesis itself is timed
        /// inside `synthesizeBoundary`.
        let bearingCaptureSeconds: TimeInterval
    }

    /// The tri-state outcome of an ASR boundary synthesis — drives the
    /// speculative-prefill body's fallback decision.
    enum BoundaryOutcome: Sendable {
        /// A stripped-path snapshot was synthesized; the speculative pass
        /// restores it and extends through the residual next-user-turn header,
        /// admitting the deeper leaf at `admitPath.count`. The full re-prefill
        /// of the stripped conversation is skipped.
        case synthesized(HybridCacheSnapshot)
        /// Preflight declined before any surgery — the caller falls back to the
        /// existing speculative prefill in full (user story #12).
        case unavailable
        /// Synthesis began then failed — admit nothing deeper than the canonical
        /// leaf and do not chain a re-prefill on top of the sunk cost (user
        /// story #14).
        case aborted
    }

    // MARK: - Render-Diff Excision

    /// The product of aligning the bearing token path against the future
    /// shared path (**Render-Diff Excision**, CONTEXT.md): the spans the
    /// future render drops, the depth of the future path the retained tokens
    /// reconstruct, and whether alignment ended early at a re-tokenized seam.
    struct RenderDiffAlignment: Sendable, Equatable {
        /// Excisions in bearing-path coordinates, sorted, non-overlapping.
        /// Includes the tail cut when alignment ends before the bearing end
        /// (a seam, or the future path running out). Invariant: excising
        /// these from `bearingTokens` yields exactly
        /// `admitPath[0..<alignedDepth]`.
        let spans: [ExcisionSpan]
        /// Number of future-path tokens the retained bearing tokens
        /// reconstruct — the synthesized snapshot's token offset.
        let alignedDepth: Int
        /// `true` when alignment stopped at a token the bearing cache never
        /// held (a re-tokenized seam, e.g. a `"\n"+"\n"` pair the stripped
        /// render merges into one `"\n\n"` token). Synthesis then proceeds at
        /// the shallower `alignedDepth` (**partial synthesis**) and the
        /// residual re-prefill covers the tail.
        let seamCut: Bool
    }

    /// Derive the excision spans by aligning the bearing render against the
    /// actual future shared path. Two-pointer walk: matching tokens advance
    /// both cursors; on a mismatch, the smallest bearing skip whose next
    /// `resyncWindow` tokens re-match the future path becomes an excision
    /// span. Unresolvable mismatches end the alignment (partial synthesis).
    ///
    /// This replaces scanning for literal `<think>` delimiters — conversation
    /// content can carry those as data (a tool `read` of CONTEXT.md did, live
    /// on 2026-06-27, poisoning the span set and forcing the pass's decline).
    /// Deriving from the render diff makes future-path compatibility hold by
    /// construction: content-borne delimiters appear in both renders and
    /// align away, and a render that strips nothing yields no spans (a
    /// correct no-op). Pure — unit-tested directly.
    static func renderDiffExcision(
        bearingTokens: [Int],
        admitPath: [Int],
        resyncWindow: Int = 16
    ) -> RenderDiffAlignment {
        var spans: [ExcisionSpan] = []
        var b = 0
        var f = 0
        while b < bearingTokens.count, f < admitPath.count {
            if bearingTokens[b] == admitPath[f] {
                b += 1
                f += 1
                continue
            }
            guard
                let resync = resyncIndex(
                    bearingTokens: bearingTokens, from: b + 1,
                    admitPath: admitPath, at: f, window: resyncWindow)
            else {
                // Unresolvable seam: cut the bearing tail and stop here.
                spans.append(ExcisionSpan(start: b, end: bearingTokens.count))
                return RenderDiffAlignment(spans: spans, alignedDepth: f, seamCut: true)
            }
            spans.append(ExcisionSpan(start: b, end: resync))
            b = resync
        }
        if f == admitPath.count, b < bearingTokens.count {
            // The future path ran out with bearing tokens left (the admit
            // path's LCP ended early): trim the tail so the retained tokens
            // are exactly the aligned prefix.
            spans.append(ExcisionSpan(start: b, end: bearingTokens.count))
        }
        return RenderDiffAlignment(spans: spans, alignedDepth: f, seamCut: false)
    }

    /// The smallest bearing index `e ≥ from` where the bearing path re-matches
    /// `admitPath[at...]` for `window` consecutive tokens (clamped near the
    /// ends, minimum one token). `nil` when no such index exists.
    private static func resyncIndex(
        bearingTokens: [Int], from: Int,
        admitPath: [Int], at futureIndex: Int,
        window: Int
    ) -> Int? {
        guard from < bearingTokens.count, futureIndex < admitPath.count else { return nil }
        for e in from..<bearingTokens.count {
            let width = min(
                max(1, window),
                bearingTokens.count - e,
                admitPath.count - futureIndex
            )
            var match = true
            for j in 0..<width where bearingTokens[e + j] != admitPath[futureIndex + j] {
                match = false
                break
            }
            if match { return e }
        }
        return nil
    }

    /// The token path left after excising the given spans.
    static func strippedTokens(
        in bearingTokens: [Int],
        spans: [ExcisionSpan]
    ) -> [Int] {
        survivingRanges(spans: spans, bearingLength: bearingTokens.count)
            .flatMap { bearingTokens[$0] }
    }

    // Evolving MVP mid-refactor (see CLAUDE.md); structural limit kept lenient.
    /// The cohesive ASR body the **Speculative Canonical Prefill** trigger
    /// calls in place of resolving a restore boundary: derive the excision
    /// spans from the actual future path (**Render-Diff Excision**), then
    /// synthesize a stripped-path snapshot from the bearing capture by pure
    /// tensor surgery. The synthesized snapshot sits at the alignment's
    /// `alignedDepth` (the future-path prefix the retained tokens
    /// reconstruct — the full stripped conversation when alignment runs
    /// clean, shallower on a re-tokenized seam); the speculative pass then
    /// restores it and extends through the residual in `admitPath`, admitting
    /// the deeper leaf at `admitPath.count` — so only the residual is
    /// re-prefilled, not the whole stripped conversation.
    ///
    /// Runs entirely off the request critical path (inside the background
    /// speculative pass), so a failure can never corrupt a live turn (user
    /// story #22): a `.midSynthesis` failure admits nothing and leaves the
    /// cache exactly as the canonical leaf left it; a `.unavailable` decline is
    /// surfaced so the caller falls back to the full re-prefill (user story
    /// #12). Synthesis is timed for the ADR-0009 performance gate; the bearing
    /// capture cost is carried in `plan.bearingCaptureSeconds` and reported
    /// alongside it (the PRD's "bearing capture may be dominant" gate).
    static func synthesizeBoundary(
        plan: Plan,
        admitPath: [Int],
        minimumWarmOffset: Int,
        testMode: Bool = false,
        diagnostics: PrefixCacheDiagnostics.Context
    ) -> BoundaryOutcome {
        let bearingSnapshot = plan.bearingSnapshot
        let synthesisStart = Date.timeIntervalSinceReferenceDate

        // Render-Diff Excision: derive the spans from the actual future path.
        // Alignment is part of the timed synthesis (it walks the whole
        // bearing path), but it is preflight in budget terms — pure integer
        // work, no tensor touched yet.
        let alignment = renderDiffExcision(
            bearingTokens: plan.bearingTokens, admitPath: admitPath
        )
        let spans = alignment.spans
        let excisedTokens = spans.reduce(0) { $0 + $1.length }
        let alignedDepth = alignment.alignedDepth

        func logOutcome(
            _ outcome: PrefixCacheDiagnostics.AsymmetricStateRestoreEvent.Outcome,
            synthesisSeconds: TimeInterval,
            unavailableReason: String? = nil
        ) {
            diagnostics.log(
                PrefixCacheDiagnostics.AsymmetricStateRestoreEvent(
                    outcome: outcome,
                    bearingOffset: bearingSnapshot.tokenOffset,
                    strippedOffset: alignedDepth,
                    spanCount: spans.count,
                    excisedTokens: excisedTokens,
                    captureSeconds: plan.bearingCaptureSeconds,
                    synthesisSeconds: synthesisSeconds,
                    seamCut: alignment.seamCut,
                    admitPathLength: admitPath.count,
                    unavailableReason: unavailableReason))
        }

        func decline(_ reason: UnavailableReason) -> BoundaryOutcome {
            let elapsed = Date.timeIntervalSinceReferenceDate - synthesisStart
            logOutcome(
                .unavailable, synthesisSeconds: elapsed,
                unavailableReason: String(describing: reason))
            if testMode {
                logAlignmentForensics(
                    plan: plan, admitPath: admitPath, alignment: alignment,
                    diagnostics: diagnostics)
            }
            return .unavailable
        }

        // Alignment guards — mirror the pass's boundary guards so a decline
        // here spends no tensor budget and the caller falls back cleanly.
        guard !spans.isEmpty else { return decline(.nothingExcised) }
        guard alignedDepth > 0 else { return decline(.noAlignedPrefix) }
        guard alignedDepth >= minimumWarmOffset else {
            return decline(.belowWarmOffset(aligned: alignedDepth, minimum: minimumWarmOffset))
        }
        guard alignedDepth < admitPath.count else {
            return decline(.noResidual(aligned: alignedDepth, admit: admitPath.count))
        }
        if testMode, alignment.seamCut {
            // A seam ended alignment early: still synthesizable at the
            // shallower depth, but the divergence is worth a forensics line.
            logAlignmentForensics(
                plan: plan, admitPath: admitPath, alignment: alignment,
                diagnostics: diagnostics)
        }

        let synthesized: HybridCacheSnapshot
        do {
            synthesized = try synthesize(
                bearingSnapshot: bearingSnapshot,
                strippedTokenCount: alignedDepth,
                spans: spans,
                ropeMetadataByLayer: plan.ropeMetadataByLayer
            )
        } catch AsymmetricStateRestore.SynthesisError.unavailable(let reason) {
            // Preflight: no idle-window budget spent — the caller falls back
            // to the existing speculative prefill in full.
            let elapsed = Date.timeIntervalSinceReferenceDate - synthesisStart
            logOutcome(
                .unavailable,
                synthesisSeconds: elapsed,
                unavailableReason: String(describing: reason))
            return .unavailable
        } catch AsymmetricStateRestore.SynthesisError.midSynthesis(let failure) {
            // Sunk some idle-window budget; safer to abort than chain a full
            // re-prefill on top. Admit nothing deeper than the canonical leaf.
            let elapsed = Date.timeIntervalSinceReferenceDate - synthesisStart
            logOutcome(
                .midSynthesis,
                synthesisSeconds: elapsed,
                unavailableReason: failure.detail)
            return .aborted
        } catch {
            let elapsed = Date.timeIntervalSinceReferenceDate - synthesisStart
            logOutcome(
                .midSynthesis,
                synthesisSeconds: elapsed,
                unavailableReason: error.localizedDescription)
            return .aborted
        }
        let synthesisSeconds = Date.timeIntervalSinceReferenceDate - synthesisStart
        logOutcome(.synthesized, synthesisSeconds: synthesisSeconds)
        return .synthesized(synthesized)
    }

    /// **Asymmetric-State Restore test mode** forensics: on a decline or a
    /// seam cut, log the token-ID windows around the first divergence so the
    /// mismatch is diagnosable from the JSONL alone (with the tokenizer
    /// offline) — no more archaeology sessions to explain a decline. Token
    /// IDs, not text: the pass has no tokenizer, and IDs are exactly what the
    /// alignment compared.
    private static func logAlignmentForensics(
        plan: Plan,
        admitPath: [Int],
        alignment: RenderDiffAlignment,
        diagnostics: PrefixCacheDiagnostics.Context
    ) {
        let f = alignment.alignedDepth
        // The bearing-side cursor where alignment stopped: the start of the
        // last span when it was a seam/tail cut, else the bearing end.
        let b = alignment.spans.last.map(\.start) ?? plan.bearingTokens.count
        func window(_ tokens: [Int], around index: Int) -> String {
            let lo = max(0, index - 8)
            let hi = min(tokens.count, index + 8)
            return "\(lo)..<\(hi):\(Array(tokens[lo..<hi]))"
        }
        diagnostics.logSkip(
            stage: "asymmetricStateRestoreForensics",
            reason: alignment.seamCut ? "seam-cut" : "declined",
            extraFields: [
                ("alignedDepth", "\(f)"),
                ("bearingCursor", "\(b)"),
                ("spanCount", "\(alignment.spans.count)"),
                (
                    "spans",
                    alignment.spans.prefix(8).map { "[\($0.start),\($0.end))" }
                        .joined(separator: " ")
                ),
                ("bearingWindow", window(plan.bearingTokens, around: b)),
                ("admitWindow", window(admitPath, around: f)),
            ]
        )
    }
}
// swiftlint:enable type_body_length
