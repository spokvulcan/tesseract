import Foundation
@preconcurrency import MLX

/// Fused Metal kernels for the per-token hot path.
///
/// A frame runs about two thousand small kernels (15 code-predictor passes and
/// a talker step), and each costs a few microseconds of GPU time however
/// little it does. These fuse the chains that dominate the count, computing
/// exactly what the MLX ops they replace compute: the same casts, the same
/// math functions, the same random draws.
///
/// Each kernel's name starts with `fastmath_`, so the fork compiles it as the
/// package's MLX kernels are compiled (fast math on). Compiled without it,
/// `exp2` in the rotary frequencies comes out an ulp away now and then, and
/// a product near a bf16 rounding boundary lands one step off.
enum Qwen3TTSKernels {
    /// Off runs the plain MLX ops (tests compare the two).
    nonisolated(unsafe) static var enabled = true
    /// Per kernel, for bisecting.
    nonisolated(unsafe) static var sampler = true
    nonisolated(unsafe) static var normRoPE = true
    nonisolated(unsafe) static var addNorm = true
    /// Bench: see every fused call's inputs and outputs.
    nonisolated(unsafe) static var audit: ((Qwen3TTSAttention, MLXArray, MLXArray, MLXArray, Int) -> Void)?
    nonisolated(unsafe) static var auditAddNorm: ((MLXArray, MLXArray, MLXArray, Float, MLXArray, MLXArray) -> Void)?
    nonisolated(unsafe) static var auditSample: ((MLXArray, MLXArray, Float, Int, MLXArray) -> Void)?

    // MARK: - q/k RMSNorm + RoPE

    /// For one position: RMSNorm each query and key head with its weight, then
    /// rotate it (non-traditional RoPE at `offset`). `qkv` is the stacked
    /// projection `[1, 1, (heads + 2·kvHeads)·headDim]`. Returns q
    /// `[1, heads, 1, headDim]` and k `[1, kvHeads, 1, headDim]`, as
    /// `MLXFast.rmsNorm` then `MLXFast.RoPE` would.
    static func qkNormRoPE(
        _ qkv: MLXArray, qWeight: MLXArray, kWeight: MLXArray, heads: Int, kvHeads: Int,
        headDim: Int, eps: Float, base: Float, offset: Int
    ) -> (q: MLXArray, k: MLXArray) {
        let outputs = qkNormRoPEKernel(
            [qkv, qWeight, kWeight, MLXArray([Int32(offset)]), MLXArray([eps, log2(base)])],
            template: [("T", qkv.dtype), ("HEAD_DIM", headDim), ("QHEADS", heads)],
            grid: (headDim / 4, heads + kvHeads, 1),
            threadGroup: (headDim / 4, 1, 1),
            outputShapes: [[1, heads, 1, headDim], [1, kvHeads, 1, headDim]],
            outputDTypes: [qkv.dtype, qkv.dtype])
        return (outputs[0], outputs[1])
    }

    static func canNormRoPE(headDim: Int) -> Bool {
        enabled && normRoPE && headDim % 8 == 0 && headDim <= 512
    }

    private static let qkNormRoPEKernel = MLXFast.metalKernel(
        name: "fastmath_qwen3tts_qk_norm_rope",
        inputNames: ["qkv", "qw", "kw", "offset", "params"],
        outputNames: ["q", "k"],
        source: """
            // One threadgroup per head; four elements a thread, as MLX's
            // rms_norm reads a row this size, so the sum of squares is its sum.
            uint head = threadgroup_position_in_grid.y;
            uint t = thread_position_in_threadgroup.x;
            uint lane = thread_index_in_simdgroup;
            uint group = simdgroup_index_in_threadgroup;
            threadgroup float sums[32];
            threadgroup float normed[HEAD_DIM];

            float xs[4];
            float acc = 0;
            for (uint e = 0; e < 4; e++) {
                xs[e] = static_cast<float>(qkv[head * HEAD_DIM + t * 4 + e]);
                acc += xs[e] * xs[e];
            }
            acc = simd_sum(acc);
            if (group == 0) { sums[lane] = 0; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0) { sums[group] = acc; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float total = simd_sum(sums[lane]);
            float inv = metal::precise::rsqrt(total / HEAD_DIM + params[0]);
            bool isQ = head < QHEADS;
            for (uint e = 0; e < 4; e++) {
                uint i = t * 4 + e;
                T w = isQ ? qw[i] : kw[i];
                // The normalized value is cast to T before the weight
                // multiplies it, as in MLX's rms_norm.
                T n = w * static_cast<T>(xs[e] * inv);
                normed[i] = static_cast<float>(n);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // RoPE, as MLX's rope_single: pair (j, j + half), frequency
            // exp2(-j / half * log2(base)), fast sin and cos.
            constexpr uint HALF = HEAD_DIM / 2;
            for (uint e = 0; e < 4; e++) {
                uint i = t * 4 + e;
                uint j = i < HALF ? i : i - HALF;
                float d = static_cast<float>(j) / static_cast<float>(HALF);
                float theta = static_cast<float>(offset[0]) * metal::exp2(-d * params[1]);
                float c = metal::fast::cos(theta);
                float s = metal::fast::sin(theta);
                float x1 = normed[j];
                float x2 = normed[j + HALF];
                float r = i < HALF ? x1 * c - x2 * s : x1 * s + x2 * c;
                if (isQ) {
                    q[head * HEAD_DIM + i] = static_cast<T>(r);
                } else {
                    k[(head - QHEADS) * HEAD_DIM + i] = static_cast<T>(r);
                }
            }
            """)

    // MARK: - Residual add + RMSNorm

    /// For one position: `sum = x + y`, and `sum` RMS-normalized with
    /// `weight`, as MLX's add then rms_norm compute them. `x` and `y` are
    /// `[1, 1, D]`; D a multiple of 128.
    static func addRMSNorm(_ x: MLXArray, _ y: MLXArray, weight: MLXArray, eps: Float)
        -> (sum: MLXArray, normed: MLXArray)
    {
        let d = x.dim(-1)
        let outputs = addRMSNormKernel(
            [x, y, weight, MLXArray([eps])],
            template: [("T", x.dtype), ("D", d)],
            grid: (d / 4, 1, 1),
            threadGroup: (d / 4, 1, 1),
            outputShapes: [x.shape, x.shape],
            outputDTypes: [x.dtype, x.dtype])
        auditAddNorm?(x, y, weight, eps, outputs[0], outputs[1])
        return (outputs[0], outputs[1])
    }

    static func canAddNorm(_ x: MLXArray) -> Bool {
        enabled && addNorm && x.ndim == 3 && x.dim(0) == 1 && x.dim(1) == 1 && x.dim(2) % 128 == 0
            && x.dim(2) <= 4096
    }

    private static let addRMSNormKernel = MLXFast.metalKernel(
        name: "fastmath_qwen3tts_add_rms_norm",
        inputNames: ["x", "y", "w", "params"],
        outputNames: ["sum", "normed"],
        source: """
            // Four elements a thread, as MLX's rms_norm reads them.
            uint tid = thread_position_in_threadgroup.x;
            uint lane = thread_index_in_simdgroup;
            uint group = simdgroup_index_in_threadgroup;
            threadgroup float sums[32];
            T s[4];
            float acc = 0;
            for (uint e = 0; e < 4; e++) {
                uint idx = tid * 4 + e;
                s[e] = static_cast<T>(static_cast<float>(x[idx]) + static_cast<float>(y[idx]));
                sum[idx] = s[e];
                float v = static_cast<float>(s[e]);
                acc += v * v;
            }
            acc = simd_sum(acc);
            if (group == 0) { sums[lane] = 0; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0) { sums[group] = acc; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float total = simd_sum(sums[lane]);
            float inv = metal::precise::rsqrt(total / D + params[0]);
            for (uint e = 0; e < 4; e++) {
                uint idx = tid * 4 + e;
                normed[idx] = w[idx] * static_cast<T>(static_cast<float>(s[e]) * inv);
            }
            """)

    // MARK: - Top-k categorical sample

    /// Draws a token from `logits` `[1, vocab]` at `temperature`, keeping
    /// every logit at least the k-th largest (ties stay): what
    /// `categorical(which(x < kth, -inf, x), key:)` with `x = logits /
    /// temperature` computes, with the same uniform draw, so the same token.
    static func topKSample(
        _ logits: MLXArray, temperature: Float, topK: Int, random: MLXRandom.RandomState
    ) -> MLXArray {
        let vocab = logits.dim(-1)
        // categorical's own draw: uniform [0, 1) of the logits' shape, float32.
        let noise = MLXRandom.uniform(
            low: Float(0), high: Float(1), [1, vocab], dtype: .float32, key: random)
        // Divided as MLX divides by a scalar: the scalar in the logits' type.
        let t = MLXArray([temperature]).asType(logits.dtype).asType(.float32)
        let lowBit: Int
        switch logits.dtype {
        case .bfloat16: lowBit = 16
        case .float16: lowBit = 13
        default: lowBit = 0
        }
        let token = topKKernel(
            [logits, noise, t],
            template: [
                ("T", logits.dtype), ("V", vocab), ("K", topK), ("LOW_BIT", lowBit),
            ],
            grid: (1024, 1, 1),
            threadGroup: (1024, 1, 1),
            outputShapes: [[1, 1]],
            outputDTypes: [.int32])[0]
        auditSample?(logits, noise, temperature, topK, token)
        return token
    }

    /// Whether `topKSample` computes this sampler's draw.
    static func canSample(_ logits: MLXArray, temperature: Float, topK: Int, topP: Float) -> Bool {
        enabled && sampler && temperature > 0 && topK > 0 && topK < logits.dim(-1)
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
