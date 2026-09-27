import Foundation
@preconcurrency import MLX
@preconcurrency import MLXLMCommon
import MLXNN

// MARK: - Talker transformer

/// The talker's 28-layer transformer. Its input embeds sum a text track and
/// a codec track; the text embedding table lives outside the module
/// (`Qwen3TTSTextEmbedding`), so only the codec table is a parameter here.
final class Qwen3TTSTalkerModel: Module {
    @ModuleInfo(key: "codec_embedding") var codecEmbedding: Embedding
    let layers: [Qwen3TTSDecoderLayer]
    @ModuleInfo var norm: RMSNorm

    init(config: Qwen3TTSTalkerConfig, fusion: Qwen3TTSFusion) {
        _codecEmbedding.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        layers = (0 ..< config.numHiddenLayers).map { _ in
            Qwen3TTSDecoderLayer(
                hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize,
                heads: config.numAttentionHeads, kvHeads: config.numKeyValueHeads,
                headDim: config.headDim, ropeBase: config.ropeTheta,
                rmsNormEps: config.rmsNormEps, attentionBias: config.attentionBias,
                fusion: fusion)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }
}

/// Text embeddings to the talker's width: two linears with SiLU between.
final class Qwen3TTSTextProjection: Module {
    @ModuleInfo(key: "linear_fc1") var fc1: Linear
    @ModuleInfo(key: "linear_fc2") var fc2: Linear

    init(inputSize: Int, outputSize: Int) {
        _fc1.wrappedValue = Linear(inputSize, inputSize, bias: true)
        _fc2.wrappedValue = Linear(inputSize, outputSize, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(silu(fc1(x)))
    }
}

/// The talker: picks each frame's first codebook (the words and the
/// prosody), and owns the code predictor that fills in the other fifteen.
final class Qwen3TTSTalker: Module {
    let config: Qwen3TTSTalkerConfig

    @ModuleInfo var model: Qwen3TTSTalkerModel
    @ModuleInfo(key: "text_projection") var textProjection: Qwen3TTSTextProjection
    @ModuleInfo(key: "codec_head") var codecHead: Linear
    @ModuleInfo(key: "code_predictor") var codePredictor: Qwen3TTSCodePredictor

    init(config: Qwen3TTSTalkerConfig, fusion: Qwen3TTSFusion = .init()) {
        self.config = config
        _model.wrappedValue = Qwen3TTSTalkerModel(config: config, fusion: fusion)
        _textProjection.wrappedValue = Qwen3TTSTextProjection(
            inputSize: config.textHiddenSize, outputSize: config.hiddenSize)
        _codecHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        _codePredictor.wrappedValue = Qwen3TTSCodePredictor(
            config: config.codePredictorConfig ?? .defaults, talkerHiddenSize: config.hiddenSize,
            fusion: fusion)
    }

    /// The codec embedding of `codes`, `[1, n]` int32.
    func embedCodec(_ codes: MLXArray) -> MLXArray {
        model.codecEmbedding(codes)
    }

    /// Runs `embeds` `[1, n, hidden]` through the transformer, appending to
    /// `cache`. Returns the first codebook's logits `[1, vocab]` and the
    /// final hidden state `[1, 1, hidden]`, both for the last position only.
    func callAsFunction(_ embeds: MLXArray, cache: [KVCache]) -> (logits: MLXArray, hidden: MLXArray) {
        let hidden = runQwen3TTSLayers(
            embeds, layers: model.layers, cache: cache, finalNorm: model.norm)
        return (codecHead(hidden).squeezed(axis: 1), hidden)
    }

    /// `callAsFunction` for a prompt. From `layerByLayerLength` positions it
    /// is evaluated a layer at a time as it is built, so its activations
    /// don't pile up in MLX's pool; a shorter prompt's are small, and the
    /// waits would cost first-audio time. Returns before the last layer has
    /// run.
    func prefill(_ embeds: MLXArray, cache: [KVCache]) -> (logits: MLXArray, hidden: MLXArray) {
        let hidden = runQwen3TTSLayers(
            embeds, layers: model.layers, cache: cache, finalNorm: model.norm,
            layerByLayer: embeds.dim(1) >= Self.layerByLayerLength)
        return (codecHead(hidden).squeezed(axis: 1), hidden)
    }

    static let layerByLayerLength = 32

    /// A KV cache per layer, preallocated for `capacity` positions.
    func makeCache(capacity: Int) -> [KVCache] {
        model.layers.map { _ in
            let cache = KVCacheSimple()
            cache.reserveCapacity(capacity)
            return cache
        }
    }
}

// MARK: - Code predictor

final class Qwen3TTSCodePredictorModel: Module {
    @ModuleInfo(key: "codec_embedding") var codecEmbedding: [Embedding]
    let layers: [Qwen3TTSDecoderLayer]
    @ModuleInfo var norm: RMSNorm

    init(
        config: Qwen3TTSTalkerCodePredictorConfig, talkerHiddenSize: Int, fusion: Qwen3TTSFusion
    ) {
        _codecEmbedding.wrappedValue = (0 ..< config.numCodeGroups - 1).map { _ in
            Embedding(embeddingCount: config.vocabSize, dimensions: talkerHiddenSize)
        }
        layers = (0 ..< config.numHiddenLayers).map { _ in
            Qwen3TTSDecoderLayer(
                hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize,
                heads: config.numAttentionHeads, kvHeads: config.numKeyValueHeads,
                headDim: config.headDim, ropeBase: config.ropeTheta,
                rmsNormEps: config.rmsNormEps, attentionBias: config.attentionBias,
                fusion: fusion)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }
}

/// Fills in codebooks 1...15 of a frame, one at a time, each conditioned on
/// the talker's hidden state and the codes before it: 15 short sequential
/// passes of a 5-layer transformer per frame.
final class Qwen3TTSCodePredictor: Module {
    let numCodeGroups: Int

    @ModuleInfo(key: "small_to_mtp_projection") var projection: Linear?
    @ModuleInfo var model: Qwen3TTSCodePredictorModel
    @ModuleInfo(key: "lm_head") var lmHead: [Linear]

    init(
        config: Qwen3TTSTalkerCodePredictorConfig, talkerHiddenSize: Int, fusion: Qwen3TTSFusion
    ) {
        numCodeGroups = config.numCodeGroups
        _projection.wrappedValue =
            config.hiddenSize != talkerHiddenSize
            ? Linear(talkerHiddenSize, config.hiddenSize, bias: true) : nil
        _model.wrappedValue = Qwen3TTSCodePredictorModel(
            config: config, talkerHiddenSize: talkerHiddenSize, fusion: fusion)
        _lmHead.wrappedValue = (0 ..< config.numCodeGroups - 1).map { _ in
            Linear(config.hiddenSize, config.vocabSize, bias: false)
        }
    }

    var codecEmbedding: [Embedding] { model.codecEmbedding }

    /// A cache per layer for one frame's passes (positions 0...numCodeGroups).
    func makeCache() -> [KVCache] {
        model.layers.map { _ in
            let cache = KVCacheSimple()
            cache.reserveCapacity(numCodeGroups + 1)
            return cache
        }
    }

    /// One frame: `hidden` is the talker's last hidden state `[1, 1, D]`,
    /// `firstEmbedding` the talker-side embedding of the frame's first code.
    /// `sample` turns `[1, vocab]` logits into a `[1, 1]` code. Returns the
    /// fifteen codes and the sum of all sixteen codes' embeddings, which is
    /// the codec half of the talker's next input. Lazy: nothing is evaluated.
    func predict(
        hidden: MLXArray, firstEmbedding: MLXArray, cache: [KVCache],
        sample: (_ group: Int, _ logits: MLXArray) -> MLXArray
    ) -> (codes: [MLXArray], embeddingSum: MLXArray) {
        for layerCache in cache { layerCache.trim(layerCache.offset) }
        var input = concatenated([hidden, firstEmbedding], axis: 1)
        var embeddingSum = firstEmbedding
        var codes: [MLXArray] = []
        codes.reserveCapacity(numCodeGroups - 1)
        for group in 0 ..< numCodeGroups - 1 {
            let x = projection.map { $0(input) } ?? input
            let last = runQwen3TTSLayers(
                x, layers: model.layers, cache: cache, finalNorm: model.norm)
            let logits = lmHead[group](last).squeezed(axis: 1)
            let code = sample(group, logits)
            codes.append(code)
            let embedding = model.codecEmbedding[group](code)
            embeddingSum = embeddingSum + embedding
            input = embedding
        }
        return (codes, embeddingSum)
    }
}
