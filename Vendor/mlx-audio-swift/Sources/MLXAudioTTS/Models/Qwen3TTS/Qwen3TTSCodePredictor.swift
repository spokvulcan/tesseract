// Port of code predictor classes from mlx_audio/tts/models/qwen3_tts/talker.py
// Multi-codebook token prediction sub-model

@preconcurrency import MLX
import MLXNN
@preconcurrency import MLXLMCommon
import Foundation

private let compiledSiluMul: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(shapeless: true) { gate, up in
    silu(gate) * up
}

private func computeRopeFreqs(dim: Int, base: Float) -> MLXArray {
    let indices = MLXArray(stride(from: 0, to: dim, by: 2)).asType(.float32)
    let exponent = indices / Float(dim)
    return MLX.pow(MLXArray(base), exponent)
}

// MARK: - Code Predictor Attention

final class CodePredictorAttention: Module {
    let numHeads: Int
    let numKvHeads: Int
    let headDim: Int
    let scale: Float
    let ropeFreqs: MLXArray

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    private var fusedQKVReady = false
    private var fusedQKVWeight: MLXArray?
    private var fusedQKVBias: MLXArray?

    init(config: Qwen3TTSTalkerCodePredictorConfig, layerIdx: Int) {
        self.numHeads = config.numAttentionHeads
        self.numKvHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = 1.0 / Foundation.sqrt(Float(headDim))
        self.ropeFreqs = computeRopeFreqs(dim: config.headDim, base: config.ropeTheta)

        self._qProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: config.attentionBias)
        self._kProj.wrappedValue = Linear(config.hiddenSize, numKvHeads * headDim, bias: config.attentionBias)
        self._vProj.wrappedValue = Linear(config.hiddenSize, numKvHeads * headDim, bias: config.attentionBias)
        self._oProj.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: config.attentionBias)
        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
    }

    @inline(__always)
    private func ensureFusedQKV() {
        if fusedQKVReady { return }
        fusedQKVWeight = concatenated([qProj.weight, kProj.weight, vProj.weight], axis: 0)
        if let qBias = qProj.bias, let kBias = kProj.bias, let vBias = vProj.bias {
            fusedQKVBias = concatenated([qBias, kBias, vBias], axis: 0)
        } else {
            fusedQKVBias = nil
        }
        // Materialize once to avoid rebuilding/retaining large concat graphs on hot path.
        if let weight = fusedQKVWeight, let bias = fusedQKVBias {
            eval(weight, bias)
        } else if let weight = fusedQKVWeight {
            eval(weight)
        }
        fusedQKVReady = true
    }

    @inline(__always)
    private func projectQKV(_ x: MLXArray, batch: Int, seqLen: Int) -> (MLXArray, MLXArray, MLXArray) {
        ensureFusedQKV()

        let qkv: MLXArray
        if let bias = fusedQKVBias, let weight = fusedQKVWeight {
            qkv = addMM(bias, x, weight.T)
        } else if let weight = fusedQKVWeight {
            qkv = matmul(x, weight.T)
        } else {
            let q = qProj(x).reshaped(batch, seqLen, numHeads, headDim)
            let k = kProj(x).reshaped(batch, seqLen, numKvHeads, headDim)
            let v = vProj(x).reshaped(batch, seqLen, numKvHeads, headDim)
            return (q, k, v)
        }

        let qSize = numHeads * headDim
        let kSize = numKvHeads * headDim
        let parts = split(qkv, indices: [qSize, qSize + kSize], axis: -1)
        let q = parts[0].reshaped(batch, seqLen, numHeads, headDim)
        let k = parts[1].reshaped(batch, seqLen, numKvHeads, headDim)
        let v = parts[2].reshaped(batch, seqLen, numKvHeads, headDim)
        return (q, k, v)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
        cache: KVCacheSimple? = nil
    ) -> MLXArray {
        if let cache {
            return callAsFunction(x, mask: mask, cache: cache)
        }

        let (batch, seqLen, _) = (x.dim(0), x.dim(1), x.dim(2))

        var (q, k, v) = projectQKV(x, batch: batch, seqLen: seqLen)

        q = qNorm(q)
        k = kNorm(k)

        q = q.transposed(0, 2, 1, 3)  // [batch, numHeads, seqLen, headDim]
        k = k.transposed(0, 2, 1, 3)
        let vt = v.transposed(0, 2, 1, 3)

        // Fused RoPE AFTER transpose — x.shape[-2] must be seqLen, not numHeads
        let offset = cache?.offset ?? 0
        q = MLXFast.RoPE(
            q,
            dimensions: headDim,
            traditional: false,
            base: nil,
            scale: 1.0,
            offset: offset,
            freqs: ropeFreqs
        )
        k = MLXFast.RoPE(
            k,
            dimensions: headDim,
            traditional: false,
            base: nil,
            scale: 1.0,
            offset: offset,
            freqs: ropeFreqs
        )

        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: vt, scale: scale, mask: mask
        )
        return oProj(output.transposed(0, 2, 1, 3).reshaped(batch, seqLen, -1))
    }

    @inline(__always)
    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
        cache: KVCacheSimple
    ) -> MLXArray {
        let (batch, seqLen, _) = (x.dim(0), x.dim(1), x.dim(2))

        var (q, k, v) = projectQKV(x, batch: batch, seqLen: seqLen)

        q = qNorm(q)
        k = kNorm(k)

        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        let vt = v.transposed(0, 2, 1, 3)

        let offset = cache.offset
        q = MLXFast.RoPE(
            q,
            dimensions: headDim,
            traditional: false,
            base: nil,
            scale: 1.0,
            offset: offset,
            freqs: ropeFreqs
        )
        k = MLXFast.RoPE(
            k,
            dimensions: headDim,
            traditional: false,
            base: nil,
            scale: 1.0,
            offset: offset,
            freqs: ropeFreqs
        )

        let (ck, cv) = cache.update(keys: k, values: vt)
        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: ck, values: cv, scale: scale, mask: mask
        )
        return oProj(output.transposed(0, 2, 1, 3).reshaped(batch, seqLen, -1))
    }
}

// MARK: - Code Predictor MLP

final class CodePredictorMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(config: Qwen3TTSTalkerCodePredictorConfig) {
        self._gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(compiledSiluMul(gateProj(x), upProj(x)))
    }
}

// MARK: - Code Predictor Decoder Layer

final class CodePredictorDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: CodePredictorAttention
    @ModuleInfo var mlp: CodePredictorMLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm

    init(config: Qwen3TTSTalkerCodePredictorConfig, layerIdx: Int) {
        self._selfAttn.wrappedValue = CodePredictorAttention(config: config, layerIdx: layerIdx)
        self._mlp.wrappedValue = CodePredictorMLP(config: config)
        self._inputLayernorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayernorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
        cache: KVCacheSimple? = nil
    ) -> MLXArray {
        if let cache {
            return callAsFunction(x, mask: mask, cache: cache)
        }

        var out = x + selfAttn(inputLayernorm(x), mask: mask, cache: cache)
        out = out + mlp(postAttentionLayernorm(out))
        return out
    }

    @inline(__always)
    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
        cache: KVCacheSimple
    ) -> MLXArray {
        var out = x + selfAttn(inputLayernorm(x), mask: mask, cache: cache)
        out = out + mlp(postAttentionLayernorm(out))
        return out
    }
}

// MARK: - Code Predictor Model (inner)

final class CodePredictorModel: Module {
    let config: Qwen3TTSTalkerCodePredictorConfig
    @ModuleInfo(key: "codec_embedding") var codecEmbedding: [Embedding]
    let layers: [CodePredictorDecoderLayer]
    @ModuleInfo var norm: RMSNorm
    // rotaryEmb kept for weight loading compatibility (weights reference it)
    let rotaryEmb: Qwen3TTSRotaryEmbedding

    init(config: Qwen3TTSTalkerCodePredictorConfig, talkerHiddenSize: Int) {
        self.config = config
        self._codecEmbedding.wrappedValue = (0 ..< config.numCodeGroups - 1).map { _ in
            Embedding(embeddingCount: config.vocabSize, dimensions: talkerHiddenSize)
        }
        self.layers = (0 ..< config.numHiddenLayers).map { CodePredictorDecoderLayer(config: config, layerIdx: $0) }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.rotaryEmb = Qwen3TTSRotaryEmbedding(
            dim: config.headDim,
            maxPositionEmbeddings: config.maxPositionEmbeddings,
            base: config.ropeTheta
        )
    }

    func callAsFunction(
        _ inputsEmbeds: MLXArray,
        mask: MLXArray? = nil,
        cache: [KVCacheSimple]? = nil
    ) -> MLXArray {
        let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode =
            if let mask {
                .array(mask)
            } else {
                inputsEmbeds.dim(1) > 1 ? .causal : .none
            }

        var x = inputsEmbeds
        if let cache {
            for i in layers.indices {
                x = layers[i](x, mask: attentionMask, cache: cache[i])
            }
        } else {
            for layer in layers {
                x = layer(x, mask: attentionMask, cache: nil)
            }
        }
        return norm(x)
    }

    /// Specialized path for seqLen=1 autoregressive decoding.
    @inline(__always)
    func callAsFunctionSingleToken(_ inputsEmbeds: MLXArray, cache: [KVCacheSimple]) -> MLXArray {
        var x = inputsEmbeds
        for i in layers.indices {
            x = layers[i](x, mask: .none, cache: cache[i])
        }
        return norm(x)
    }

    func makeCache() -> [KVCacheSimple] {
        layers.map { _ in KVCacheSimple() }
    }
}

// MARK: - Code Predictor (public)

final class Qwen3TTSCodePredictor: Module {
    let config: Qwen3TTSTalkerCodePredictorConfig
    let numCodeGroups: Int
    let talkerHiddenSize: Int

    @ModuleInfo(key: "small_to_mtp_projection") var projection: Linear?
    @ModuleInfo var model: CodePredictorModel
    @ModuleInfo(key: "lm_head") var lmHead: [Linear]

    var codecEmbedding: [Embedding] { model.codecEmbedding }

    init(config: Qwen3TTSTalkerCodePredictorConfig, talkerHiddenSize: Int) {
        self.config = config
        self.numCodeGroups = config.numCodeGroups
        self.talkerHiddenSize = talkerHiddenSize

        if config.hiddenSize != talkerHiddenSize {
            self._projection.wrappedValue = Linear(talkerHiddenSize, config.hiddenSize, bias: true)
        } else {
            self._projection.wrappedValue = nil
        }

        self._model.wrappedValue = CodePredictorModel(config: config, talkerHiddenSize: talkerHiddenSize)
        self._lmHead.wrappedValue = (0 ..< config.numCodeGroups - 1).map { _ in
            Linear(config.hiddenSize, config.vocabSize, bias: false)
        }
    }

    func callAsFunction(
        _ inputsEmbeds: MLXArray,
        cache: [KVCacheSimple]? = nil,
        generationStep: Int = 0
    ) -> (MLXArray, [KVCacheSimple]?, Int) {
        var embeds = inputsEmbeds
        if let proj = projection {
            embeds = proj(embeds)
        }

        let x = model(embeds, cache: cache)
        let logits = lmHead[generationStep](x)
        return (logits, cache, generationStep + 1)
    }

    /// Fast path for autoregressive codebook generation loops.
    /// Returns logits for the configured codebook head.
    @inline(__always)
    func predictStep(
        _ inputsEmbeds: MLXArray,
        cache: [KVCacheSimple],
        generationStep: Int
    ) -> MLXArray {
        var embeds = inputsEmbeds
        if let proj = projection {
            embeds = proj(embeds)
        }
        let x = model(embeds, cache: cache)
        let lastHidden = x[0..., (-1)..., 0...]
        return lmHead[generationStep](lastHidden)
    }

    /// Fast path when the predictor input is a single token (`seqLen=1`).
    @inline(__always)
    func predictStepSingleToken(
        _ inputsEmbeds: MLXArray,
        cache: [KVCacheSimple],
        generationStep: Int
    ) -> MLXArray {
        var embeds = inputsEmbeds
        if let proj = projection {
            embeds = proj(embeds)
        }
        let x = model.callAsFunctionSingleToken(embeds, cache: cache)
        return lmHead[generationStep](x)
    }

    /// Prime code predictor KV cache with context embeddings (no sampling/logits needed).
    /// This enables subsequent autoregressive passes to run as single-token (`seqLen=1`) steps.
    func prefill(_ inputsEmbeds: MLXArray, cache: [KVCacheSimple]) {
        var embeds = inputsEmbeds
        if let proj = projection {
            embeds = proj(embeds)
        }
        _ = model(embeds, cache: cache)
    }

    func makeCache() -> [KVCacheSimple] {
        model.makeCache()
    }
}
