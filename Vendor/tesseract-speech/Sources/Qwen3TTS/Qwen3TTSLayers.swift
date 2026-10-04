import Foundation
@preconcurrency import MLX
@preconcurrency import MLXLMCommon
import MLXNN

// The Qwen3 decoder layer the talker and the code predictor share.
//
// The checkpoint stores q/k/v and gate/up as separate projections; loading
// stacks each set along its output rows (`Qwen3TTSWeights`), so a layer runs
// two matmuls where the reference runs five. Quantization groups run along
// the input dimension, so stacking rows is exact.
//
// Rotary positions: the talker's config names Qwen's interleaved 3D M-RoPE,
// but text-to-speech feeds the same position to all three axes, and then the
// interleave picks between identical frequencies. That is plain RoPE, which
// MLX runs as one fused kernel per tensor.

/// Which projections loading stacked into one matmul. A checkpoint quantized
/// per layer can give q, k and v different bit widths, and those can't be
/// stacked; they then run as separate matmuls, as in the reference.
struct Qwen3TTSFusion: Sendable {
    var qkv = true
    var gateUp = true
}

/// One head's attention logits over a span of key positions, taken for the
/// last query of each call: where the talker looks in the text while it
/// speaks (ADR-0077). Only the talker's alignment layer carries one, and
/// only while a generation asks for word timing.
final class Qwen3TTSAlignmentProbe {
    let head: Int
    /// Key positions: the prompt's text track.
    let span: Range<Int>
    /// The latest call's logits, `[span.count]` float32, lazy. Positions not
    /// in the cache yet (the streaming-text layout's later text) are left
    /// out, so it can be shorter than the span.
    var scores: MLXArray?

    init(head: Int, span: Range<Int>) {
        self.head = head
        self.span = span
    }
}

/// Qwen3 attention: q/k/v in one projection, RMSNorm on each head's queries
/// and keys, rotary positions, grouped-query attention over a KV cache.
final class Qwen3TTSAttention: Module {
    /// Set on the alignment layer during a generation that times its words.
    var alignmentProbe: Qwen3TTSAlignmentProbe?

    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float
    let ropeBase: Float

    @ModuleInfo(key: "qkv_proj") var qkvProj: Linear?
    @ModuleInfo(key: "q_proj") var qProj: Linear?
    @ModuleInfo(key: "k_proj") var kProj: Linear?
    @ModuleInfo(key: "v_proj") var vProj: Linear?
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(
        hiddenSize: Int, heads: Int, kvHeads: Int, headDim: Int, ropeBase: Float,
        rmsNormEps: Float, bias: Bool, fused: Bool
    ) {
        self.heads = heads
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.scale = 1 / Float(headDim).squareRoot()
        self.ropeBase = ropeBase
        if fused {
            _qkvProj.wrappedValue = Linear(
                hiddenSize, (heads + 2 * kvHeads) * headDim, bias: bias)
        } else {
            _qProj.wrappedValue = Linear(hiddenSize, heads * headDim, bias: bias)
            _kProj.wrappedValue = Linear(hiddenSize, kvHeads * headDim, bias: bias)
            _vProj.wrappedValue = Linear(hiddenSize, kvHeads * headDim, bias: bias)
        }
        _oProj.wrappedValue = Linear(heads * headDim, hiddenSize, bias: bias)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, cache: KVCache) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        let offset = cache.offset
        let q: MLXArray
        let k: MLXArray
        let v: MLXArray
        if let qkvProj {
            let qkv = qkvProj(x)
            let split = qkv.reshaped(batch, length, heads + 2 * kvHeads, headDim)
            v = split[0..., 0..., (heads + kvHeads)..., 0...].transposed(0, 2, 1, 3)
            if length == 1, batch == 1,
                let fused = Qwen3TTSKernels.qkNormRoPE(
                    qkv, qWeight: qNorm.weight, kWeight: kNorm.weight, heads: heads,
                    kvHeads: kvHeads, headDim: headDim, eps: qNorm.eps, base: ropeBase,
                    offset: offset)
            {
                // One position (every decode step): both norms and both
                // rotations in one kernel.
                (q, k) = fused
            } else {
                (q, k) = normAndRotate(
                    split[0..., 0..., ..<heads, 0...],
                    split[0..., 0..., heads ..< (heads + kvHeads), 0...], offset: offset)
            }
        } else {
            v = vProj!(x).reshaped(batch, length, kvHeads, headDim).transposed(0, 2, 1, 3)
            (q, k) = normAndRotate(
                qProj!(x).reshaped(batch, length, heads, headDim),
                kProj!(x).reshaped(batch, length, kvHeads, headDim), offset: offset)
        }
        let (keys, values) = cache.update(keys: k, values: v)
        if let probe = alignmentProbe {
            probe.scores = alignmentScores(probe, q: q, keys: keys, length: length)
        }
        // `.causal` aligns to the last key, so a prompt after a restored
        // prefix sees the whole prefix.
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: keys, values: values, scale: scale,
            mask: length > 1 ? .causal : .none)
        return oProj(out.transposed(0, 2, 1, 3).reshaped(batch, length, heads * headDim))
    }

    /// The probe's head, last query, over the cached keys in its span: one
    /// `[1, D] × [D, span]` product in float32. Nil before any key of the
    /// span is cached.
    private func alignmentScores(
        _ probe: Qwen3TTSAlignmentProbe, q: MLXArray, keys: MLXArray, length: Int
    ) -> MLXArray? {
        let end = min(probe.span.upperBound, keys.dim(2))
        guard probe.span.lowerBound < end else { return nil }
        let kvHead = probe.head / (heads / kvHeads)
        let query = q[0..., probe.head ..< (probe.head + 1), (length - 1) ..< length, 0...]
        let span = keys[0..., kvHead ..< (kvHead + 1), probe.span.lowerBound ..< end, 0...]
        return (matmul(query.asType(.float32), span.asType(.float32).transposed(0, 1, 3, 2))
            * scale).reshaped(-1)
    }

    /// Queries and keys `[B, L, heads, D]` normalized per head and rotated,
    /// head-major: the MLX ops the fused kernel replaces.
    private func normAndRotate(_ q: MLXArray, _ k: MLXArray, offset: Int) -> (MLXArray, MLXArray) {
        func rotated(_ x: MLXArray) -> MLXArray {
            MLXFast.RoPE(
                x.transposed(0, 2, 1, 3), dimensions: headDim, traditional: false, base: ropeBase,
                scale: 1, offset: offset)
        }
        return (rotated(qNorm(q)), rotated(kNorm(k)))
    }
}

/// SwiGLU, gate and up in one projection.
final class Qwen3TTSMLP: Module {
    @ModuleInfo(key: "gate_up_proj") var gateUpProj: Linear?
    @ModuleInfo(key: "gate_proj") var gateProj: Linear?
    @ModuleInfo(key: "up_proj") var upProj: Linear?
    @ModuleInfo(key: "down_proj") var downProj: Linear
    /// Sees each product `silu(gate) · up`, while the Neural Engine graphs
    /// measure how large it grows (`NeuralPrecision`).
    var productProbe: ((MLXArray) -> Void)?

    init(hiddenSize: Int, intermediateSize: Int, fused: Bool) {
        if fused {
            _gateUpProj.wrappedValue = Linear(hiddenSize, 2 * intermediateSize, bias: false)
        } else {
            _gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
            _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        }
        _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let product: MLXArray
        if let gateUpProj {
            let (gate, up) = gateUpProj(x).split(axis: -1)
            product = compiledSwiGLU(gate, up)
        } else {
            product = compiledSwiGLU(gateProj!(x), upProj!(x))
        }
        productProbe?(product)
        return downProj(product)
    }
}

private let compiledSwiGLU: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    compile(shapeless: true) { gate, up in silu(gate) * up }
}()

final class Qwen3TTSDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: Qwen3TTSAttention
    @ModuleInfo var mlp: Qwen3TTSMLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm

    init(
        hiddenSize: Int, intermediateSize: Int, heads: Int, kvHeads: Int, headDim: Int,
        ropeBase: Float, rmsNormEps: Float, attentionBias: Bool, fusion: Qwen3TTSFusion
    ) {
        _attention.wrappedValue = Qwen3TTSAttention(
            hiddenSize: hiddenSize, heads: heads, kvHeads: kvHeads, headDim: headDim,
            ropeBase: ropeBase, rmsNormEps: rmsNormEps, bias: attentionBias, fused: fusion.qkv)
        _mlp.wrappedValue = Qwen3TTSMLP(
            hiddenSize: hiddenSize, intermediateSize: intermediateSize, fused: fusion.gateUp)
        _inputNorm.wrappedValue = RMSNorm(dimensions: hiddenSize, eps: rmsNormEps)
        _postNorm.wrappedValue = RMSNorm(dimensions: hiddenSize, eps: rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, cache: KVCache) -> MLXArray {
        let h = x + attention(inputNorm(x), cache: cache)
        return h + mlp(postNorm(h))
    }

    /// One position with fused residual adds: takes the residual stream and
    /// its normalized copy, returns the next residual and its copy
    /// normalized by `next` (the following layer's input norm, or the
    /// stack's final norm).
    func step(_ x: MLXArray, normed: MLXArray, cache: KVCache, next: RMSNorm)
        -> (residual: MLXArray, normed: MLXArray)
    {
        let (h, n) = Qwen3TTSKernels.addRMSNorm(
            x, attention(normed, cache: cache), weight: postNorm.weight, eps: postNorm.eps)
        let (out, nextNormed) = Qwen3TTSKernels.addRMSNorm(
            h, mlp(n), weight: next.weight, eps: next.eps)
        return (out, nextNormed)
    }
}

/// Runs `layers` over `x` and returns the final norm of the last position:
/// `[1, 1, D]`. A single position takes the fused path.
///
/// `layerByLayer` evaluates as it goes, for a prompt: each layer is queued
/// once built and the one before it waited for. The GPU never idles, and
/// command buffers in flight hold at most two layers' activations, where a
/// whole-prompt graph holds about ten. MLX's pool keeps what they held, at
/// sizes only that prompt length reuses, so this cuts what each new length
/// leaves there by about 70 %. The values are the same.
func runQwen3TTSLayers(
    _ x: MLXArray, layers: [Qwen3TTSDecoderLayer], cache: [KVCache], finalNorm: RMSNorm,
    layerByLayer: Bool = false
) -> MLXArray {
    if Qwen3TTSKernels.canAddNorm(x), !layers.isEmpty {
        var residual = x
        var normed = layers[0].inputNorm(x)
        for (i, layer) in layers.enumerated() {
            let next = i + 1 < layers.count ? layers[i + 1].inputNorm : finalNorm
            (residual, normed) = layer.step(residual, normed: normed, cache: cache[i], next: next)
        }
        return normed
    }
    var h = x
    var previous: MLXArray?
    for (layer, layerCache) in zip(layers, cache) {
        h = layer(h, cache: layerCache)
        if layerByLayer {
            asyncEval(h)
            if let previous { eval(previous) }
            previous = h
        }
    }
    return finalNorm(h[0..., (-1)..., 0...])
}
