import Foundation
@preconcurrency import MLX
@preconcurrency import MLXLMCommon

/// Fused Metal kernels for the per-token hot path.
///
/// A frame runs about two thousand small kernels (15 code-predictor passes and
/// a talker step), and each costs a few microseconds of GPU time however
/// little it does. These fuse the chains that dominate the count, computing
/// exactly what the MLX ops they replace compute: the same casts, the same
/// math functions, the same random draws.
///
/// q/k RMSNorm + RoPE and the residual add + RMSNorm are MLXLMCommon's
/// (`attentionNormRope`, `rmsNormResidual`); the top-k draw is this package's.
/// Each kernel's name starts with `fastmath_`, so the fork compiles it as the
/// package's MLX kernels are compiled (fast math on). Compiled without it,
/// `exp2` in the rotary frequencies comes out an ulp away now and then, and a
/// product near a bf16 rounding boundary lands one step off.
enum Qwen3TTSKernels {
    /// Per kernel: off runs the MLX ops it replaces (tests and the bench
    /// compare the two).
    nonisolated(unsafe) static var sampler = true
    nonisolated(unsafe) static var normRoPE = true
    nonisolated(unsafe) static var addNorm = true

    /// All three at once.
    static var enabled: Bool {
        get { sampler || normRoPE || addNorm }
        set { (sampler, normRoPE, addNorm) = (newValue, newValue, newValue) }
    }

    enum Kernel: CaseIterable {
        case normRoPE, addNorm, sampler
    }

    /// Bench: sees each fused call's outputs, with a closure computing what
    /// the MLX ops give for the same inputs.
    nonisolated(unsafe) static var audit: ((Kernel, [MLXArray], () -> [MLXArray]) -> Void)?

    // MARK: - q/k RMSNorm + RoPE

    /// For one position: RMSNorm each query and key head with its weight, then
    /// rotate it (non-traditional RoPE at `offset`). `qkv` is the stacked
    /// projection `[1, 1, (heads + 2·kvHeads)·headDim]`. Returns q
    /// `[1, heads, 1, headDim]` and k `[1, kvHeads, 1, headDim]`, as
    /// `MLXFast.rmsNorm` then `MLXFast.RoPE` would; nil when switched off or
    /// when the kernel doesn't take these dtypes (fp32, mixed weights).
    static func qkNormRoPE(
        _ qkv: MLXArray, qWeight: MLXArray, kWeight: MLXArray, heads: Int, kvHeads: Int,
        headDim: Int, eps: Float, base: Float, offset: Int
    ) -> (q: MLXArray, k: MLXArray)? {
        guard normRoPE,
            let fused = attentionNormRope(
                rows: qkv, queryOffset: 0, queryHeadStride: headDim, queryHeads: heads,
                keyOffset: heads * headDim, keyHeadStride: headDim, keyHeads: kvHeads,
                headDim: headDim, queryWeight: qWeight, keyWeight: kWeight, eps: eps,
                rope: PlainRoPEParameters(dimensions: headDim, base: base, scale: 1),
                offset: MLXArray([Int32(offset)]))
        else { return nil }
        audit?(.normRoPE, [fused.queries, fused.keys]) {
            let split = qkv.reshaped(1, 1, heads + 2 * kvHeads, headDim)
            func rotated(_ x: MLXArray, _ weight: MLXArray) -> MLXArray {
                MLXFast.RoPE(
                    MLXFast.rmsNorm(x, weight: weight, eps: eps).transposed(0, 2, 1, 3),
                    dimensions: headDim, traditional: false, base: base, scale: 1, offset: offset)
            }
            return [
                rotated(split[0..., 0..., ..<heads, 0...], qWeight),
                rotated(split[0..., 0..., heads ..< (heads + kvHeads), 0...], kWeight),
            ]
        }
        return (fused.queries, fused.keys)
    }

    // MARK: - Residual add + RMSNorm

    /// Whether `runQwen3TTSLayers` fuses the residual adds for `x`: switched
    /// on, and one position.
    static func canAddNorm(_ x: MLXArray) -> Bool {
        addNorm && x.ndim == 3 && x.dim(0) == 1 && x.dim(1) == 1
    }

    /// `sum = x + y`, and `sum` RMS-normalized with `weight`: one kernel,
    /// or MLX's add then rms_norm when the dtypes differ (the kernel's
    /// values either way).
    static func addRMSNorm(_ x: MLXArray, _ y: MLXArray, weight: MLXArray, eps: Float)
        -> (sum: MLXArray, normed: MLXArray)
    {
        func ops() -> [MLXArray] {
            let sum = x + y
            return [sum, MLXFast.rmsNorm(sum, weight: weight, eps: eps)]
        }
        guard x.shape == y.shape, y.dtype == x.dtype, weight.dtype == x.dtype else {
            let reference = ops()
            return (reference[0], reference[1])
        }
        let (sum, normed) = rmsNormResidual(x, y, weight: weight, eps: eps)
        audit?(.addNorm, [sum, normed], ops)
        return (sum, normed)
    }

    // MARK: - Top-k categorical sample

    /// Draws a token from `logits` `[1, vocab]` at `temperature`, keeping
    /// every logit at least the k-th largest (ties stay): what
    /// `categorical(which(x < kth, -inf, x), key:)` with `x = logits /
    /// temperature` computes, with the same uniform draw, so the same token.
    /// `divisor` is `temperature` as MLX divides by it, rounded to the
    /// logits' type (`[1]` float32, kept by `Qwen3TTSSampler`).
    static func topKSample(
        _ logits: MLXArray, temperature: Float, divisor: MLXArray, topK: Int,
        random: MLXRandom.RandomState
    ) -> MLXArray {
        let vocab = logits.dim(-1)
        // categorical's own draw: uniform [0, 1) of the logits' shape, float32.
        let noise = MLXRandom.uniform(
            low: Float(0), high: Float(1), [1, vocab], dtype: .float32, key: random)
        let lowBit: Int
        switch logits.dtype {
        case .bfloat16: lowBit = 16
        case .float16: lowBit = 13
        default: lowBit = 0
        }
        let token = topKKernel(
            [logits, noise, divisor],
            template: [
                ("T", logits.dtype), ("V", vocab), ("K", topK), ("LOW_BIT", lowBit),
            ],
            grid: (1024, 1, 1),
            threadGroup: (1024, 1, 1),
            outputShapes: [[1, 1]],
            outputDTypes: [.int32])[0]
        audit?(.sampler, [token]) {
            // categorical: argmax(gumbel + logits), gumbel = -log(-log(u)).
            let filtered = Qwen3TTSSampler.filter(logits / temperature, topK: topK, topP: 1)
            return [argMax(-log(-log(noise)) + filtered, axis: -1).asType(.int32).reshaped(1, 1)]
        }
        return token
    }

    /// Whether `topKSample` computes this sampler's draw.
    static func canSample(_ logits: MLXArray, temperature: Float, topK: Int, topP: Float) -> Bool {
        sampler && temperature > 0 && topK > 0 && topK < logits.dim(-1)
            && (topP <= 0 || topP >= 1) && logits.ndim == 2 && logits.dim(0) == 1
            && [.bfloat16, .float16, .float32].contains(logits.dtype)
    }

    private static let topKKernel = MLXFast.metalKernel(
        name: "fastmath_qwen3tts_top_k_sample",
        inputNames: ["logits", "noise", "temperature"],
        outputNames: ["token"],
        source: """
            constexpr uint THREADS = 1024;
            constexpr uint PER = (V + THREADS - 1) / THREADS;
            uint tid = thread_position_in_threadgroup.x;
            uint lane = thread_index_in_simdgroup;
            uint group = simdgroup_index_in_threadgroup;
            threadgroup uint counts[32];
            threadgroup float bestValues[32];
            threadgroup uint bestIndices[32];

            // x = logits / temperature, rounded to T as MLX's divide rounds it;
            // an order-preserving key of each value's float bits.
            float x[PER];
            uint key[PER];
            for (uint e = 0; e < PER; e++) {
                uint idx = tid + e * THREADS;
                if (idx < V) {
                    x[e] = static_cast<float>(static_cast<T>(static_cast<float>(logits[idx]) / temperature[0]));
                    uint bits = as_type<uint>(x[e]);
                    key[e] = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
                } else {
                    x[e] = -INFINITY;
                    key[e] = 0;
                }
            }

            // The k-th largest value's key: the largest t with at least K keys
            // >= t, found bit by bit (T's values leave the low bits zero).
            uint threshold = 0;
            for (int bit = 31; bit >= LOW_BIT; bit--) {
                uint candidate = threshold | (1u << bit);
                uint c = 0;
                for (uint e = 0; e < PER; e++) { c += key[e] >= candidate ? 1u : 0u; }
                c = simd_sum(c);
                if (lane == 0) { counts[group] = c; }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                uint total = simd_sum(counts[lane]);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (total >= K) { threshold = candidate; }
            }

            // argmax(x + gumbel) over the kept values, gumbel = -log(-log(u))
            // as MLX computes it; the first index wins a tie.
            float best = -INFINITY;
            uint bestIndex = 0xFFFFFFFFu;
            for (uint e = 0; e < PER; e++) {
                uint idx = tid + e * THREADS;
                if (idx < V && key[e] >= threshold) {
                    float g = -metal::precise::log(-metal::precise::log(noise[idx]));
                    float score = g + x[e];
                    if (score > best || (score == best && idx < bestIndex)) {
                        best = score;
                        bestIndex = idx;
                    }
                }
            }
            float m = simd_max(best);
            uint mi = simd_min(best == m ? bestIndex : 0xFFFFFFFFu);
            if (lane == 0) {
                bestValues[group] = m;
                bestIndices[group] = mi;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (group == 0) {
                float v = bestValues[lane];
                float mm = simd_max(v);
                uint ii = simd_min(v == mm ? bestIndices[lane] : 0xFFFFFFFFu);
                if (lane == 0) { token[0] = ii == 0xFFFFFFFFu ? 0 : int(ii); }
            }
            """)
}
