// Port of code predictor classes from mlx_audio/tts/models/qwen3_tts/talker.py
// Multi-codebook token prediction sub-model

@preconcurrency import MLX
import MLXNN
@preconcurrency import MLXLMCommon
import Foundation

// MARK: - Code Predictor Attention

final class CodePredictorAttention: Module {
    let numHeads: Int
    let numKvHeads: Int
    let headDim: Int
    let scale: Float
    let ropeBase: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(config: Qwen3TTSTalkerCodePredictorConfig, layerIdx: Int) {
        self.numHeads = config.numAttentionHeads
        self.numKvHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = 1.0 / Foundation.sqrt(Float(headDim))
        self.ropeBase = config.ropeTheta

        self._qProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: config.attentionBias)
        self._kProj.wrappedValue = Linear(config.hiddenSize, numKvHeads * headDim, bias: config.attentionBias)
        self._vProj.wrappedValue = Linear(config.hiddenSize, numKvHeads * headDim, bias: config.attentionBias)
        self._oProj.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: config.attentionBias)
        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        cache: (any KVCache)? = nil
    ) -> MLXArray {
        let (batch, seqLen, _) = (x.dim(0), x.dim(1), x.dim(2))

        var q = qProj(x).reshaped(batch, seqLen, numHeads, headDim)
        var k = kProj(x).reshaped(batch, seqLen, numKvHeads, headDim)
        let v = vProj(x).reshaped(batch, seqLen, numKvHeads, headDim)

        q = qNorm(q)
        k = kNorm(k)

        q = q.transposed(0, 2, 1, 3)  // [batch, numHeads, seqLen, headDim]
        k = k.transposed(0, 2, 1, 3)
        let vt = v.transposed(0, 2, 1, 3)

        // Fused RoPE AFTER transpose — x.shape[-2] must be seqLen, not numHeads
        let offset = cache?.offset ?? 0
        q = MLXFast.RoPE(q, dimensions: headDim, traditional: false, base: ropeBase, scale: 1.0, offset: offset)
        k = MLXFast.RoPE(k, dimensions: headDim, traditional: false, base: ropeBase, scale: 1.0, offset: offset)

        if let cache {
            let (ck, cv) = cache.update(keys: k, values: vt)
            let output = MLXFast.scaledDotProductAttention(
                queries: q, keys: ck, values: cv, scale: scale, mask: mask
            )
            return oProj(output.transposed(0, 2, 1, 3).reshaped(batch, seqLen, -1))
        } else {
            let output = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: vt, scale: scale, mask: mask
            )
            return oProj(output.transposed(0, 2, 1, 3).reshaped(batch, seqLen, -1))
        }
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
        downProj(silu(gateProj(x)) * upProj(x))
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
        mask: MLXArray? = nil,
        cache: (any KVCache)? = nil
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
        cache: [any KVCache]? = nil
    ) -> MLXArray {
        let seqLen = inputsEmbeds.dim(1)

        var causalMask = mask
        if causalMask == nil && seqLen > 1 {
            causalMask = MultiHeadAttention.createAdditiveCausalMask(seqLen).asType(inputsEmbeds.dtype)
        }

        var x = inputsEmbeds
        for (i, layer) in layers.enumerated() {
            x = layer(x, mask: causalMask, cache: cache?[i])
        }
        return norm(x)
    }

    func makeCache() -> [any KVCache] {
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
        cache: [any KVCache]? = nil,
        generationStep: Int = 0
    ) -> (MLXArray, [any KVCache]?, Int) {
        var embeds = inputsEmbeds
        if let proj = projection {
            embeds = proj(embeds)
        }

        let x = model(embeds, cache: cache)
        let logits = lmHead[generationStep](x)
        return (logits, cache, generationStep + 1)
    }

    func makeCache() -> [any KVCache] {
        model.makeCache()
    }
}
