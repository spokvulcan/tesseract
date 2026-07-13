import Foundation

public struct MossGPT2Config: Codable, Sendable {
    public var modelType: String
    public var vocabSize: Int
    public var nPositions: Int
    public var nCtx: Int
    public var nEmbd: Int
    public var nLayer: Int
    public var nHead: Int
    public var nInner: Int?
    public var activationFunction: String
    public var residPdrop: Float
    public var embdPdrop: Float
    public var attnPdrop: Float
    public var layerNormEpsilon: Float
    public var initializerRange: Float
    public var scaleAttnWeights: Bool
    public var scaleAttnByInverseLayerIdx: Bool
    public var positionEmbeddingType: String
    public var ropeBase: Float
    public var padTokenID: Int
    public var bosTokenID: Int
    public var eosTokenID: Int
    public var tieWordEmbeddings: Bool
    public var useCache: Bool

    public init(
        modelType: String = "gpt2",
        vocabSize: Int = 16_384,
        nPositions: Int = 32_768,
        nCtx: Int = 32_768,
        nEmbd: Int = 768,
        nLayer: Int = 12,
        nHead: Int = 12,
        nInner: Int? = 3_072,
        activationFunction: String = "gelu_new",
        residPdrop: Float = 0,
        embdPdrop: Float = 0,
        attnPdrop: Float = 0,
        layerNormEpsilon: Float = 1e-5,
        initializerRange: Float = 0.02,
        scaleAttnWeights: Bool = true,
        scaleAttnByInverseLayerIdx: Bool = false,
        positionEmbeddingType: String = "rope",
        ropeBase: Float = 10_000,
        padTokenID: Int = 3,
        bosTokenID: Int = 1,
        eosTokenID: Int = 2,
        tieWordEmbeddings: Bool = true,
        useCache: Bool = true
    ) {
        self.modelType = modelType
        self.vocabSize = vocabSize
        self.nPositions = nPositions
        self.nCtx = nCtx
        self.nEmbd = nEmbd
        self.nLayer = nLayer
        self.nHead = nHead
        self.nInner = nInner
        self.activationFunction = activationFunction
        self.residPdrop = residPdrop
        self.embdPdrop = embdPdrop
        self.attnPdrop = attnPdrop
        self.layerNormEpsilon = layerNormEpsilon
        self.initializerRange = initializerRange
        self.scaleAttnWeights = scaleAttnWeights
        self.scaleAttnByInverseLayerIdx = scaleAttnByInverseLayerIdx
        self.positionEmbeddingType = positionEmbeddingType
        self.ropeBase = ropeBase
        self.padTokenID = padTokenID
        self.bosTokenID = bosTokenID
        self.eosTokenID = eosTokenID
        self.tieWordEmbeddings = tieWordEmbeddings
        self.useCache = useCache
    }

    public var hiddenSize: Int { nEmbd }
    public var numAttentionHeads: Int { nHead }
    public var headDim: Int { nEmbd / nHead }
    public var intermediateSize: Int { nInner ?? 4 * nEmbd }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case nPositions = "n_positions"
        case nCtx = "n_ctx"
        case nEmbd = "n_embd"
        case nLayer = "n_layer"
        case nHead = "n_head"
        case nInner = "n_inner"
        case activationFunction = "activation_function"
        case residPdrop = "resid_pdrop"
        case embdPdrop = "embd_pdrop"
        case attnPdrop = "attn_pdrop"
        case layerNormEpsilon = "layer_norm_epsilon"
        case initializerRange = "initializer_range"
        case scaleAttnWeights = "scale_attn_weights"
        case scaleAttnByInverseLayerIdx = "scale_attn_by_inverse_layer_idx"
        case positionEmbeddingType = "position_embedding_type"
        case ropeBase = "rope_base"
        case padTokenID = "pad_token_id"
        case bosTokenID = "bos_token_id"
        case eosTokenID = "eos_token_id"
        case tieWordEmbeddings = "tie_word_embeddings"
        case useCache = "use_cache"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case intermediateSize = "intermediate_size"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "gpt2"
        self.vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 16_384
        self.nPositions = try c.decodeIfPresent(Int.self, forKey: .nPositions) ?? 32_768
        self.nCtx = try c.decodeIfPresent(Int.self, forKey: .nCtx) ?? nPositions
        self.nEmbd = try c.decodeIfPresent(Int.self, forKey: .nEmbd)
            ?? c.decodeIfPresent(Int.self, forKey: .hiddenSize)
            ?? 768
        self.nLayer = try c.decodeIfPresent(Int.self, forKey: .nLayer)
            ?? c.decodeIfPresent(Int.self, forKey: .numHiddenLayers)
            ?? 12
        self.nHead = try c.decodeIfPresent(Int.self, forKey: .nHead)
            ?? c.decodeIfPresent(Int.self, forKey: .numAttentionHeads)
            ?? 12
        self.nInner = try c.decodeIfPresent(Int.self, forKey: .nInner)
            ?? c.decodeIfPresent(Int.self, forKey: .intermediateSize)
            ?? 3_072
        self.activationFunction = try c.decodeIfPresent(String.self, forKey: .activationFunction) ?? "gelu_new"
        self.residPdrop = try c.decodeIfPresent(Float.self, forKey: .residPdrop) ?? 0
        self.embdPdrop = try c.decodeIfPresent(Float.self, forKey: .embdPdrop) ?? 0
        self.attnPdrop = try c.decodeIfPresent(Float.self, forKey: .attnPdrop) ?? 0
        self.layerNormEpsilon = try c.decodeIfPresent(Float.self, forKey: .layerNormEpsilon) ?? 1e-5
        self.initializerRange = try c.decodeIfPresent(Float.self, forKey: .initializerRange) ?? 0.02
        self.scaleAttnWeights = try c.decodeIfPresent(Bool.self, forKey: .scaleAttnWeights) ?? true
        self.scaleAttnByInverseLayerIdx = try c.decodeIfPresent(Bool.self, forKey: .scaleAttnByInverseLayerIdx) ?? false
        self.positionEmbeddingType = try c.decodeIfPresent(String.self, forKey: .positionEmbeddingType) ?? "rope"
        self.ropeBase = try c.decodeIfPresent(Float.self, forKey: .ropeBase) ?? 10_000
        self.padTokenID = try c.decodeIfPresent(Int.self, forKey: .padTokenID) ?? 3
        self.bosTokenID = try c.decodeIfPresent(Int.self, forKey: .bosTokenID) ?? 1
        self.eosTokenID = try c.decodeIfPresent(Int.self, forKey: .eosTokenID) ?? 2
        self.tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        self.useCache = try c.decodeIfPresent(Bool.self, forKey: .useCache) ?? true
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(modelType, forKey: .modelType)
        try c.encode(vocabSize, forKey: .vocabSize)
        try c.encode(nPositions, forKey: .nPositions)
        try c.encode(nCtx, forKey: .nCtx)
        try c.encode(nEmbd, forKey: .nEmbd)
        try c.encode(nLayer, forKey: .nLayer)
        try c.encode(nHead, forKey: .nHead)
        try c.encodeIfPresent(nInner, forKey: .nInner)
        try c.encode(activationFunction, forKey: .activationFunction)
        try c.encode(residPdrop, forKey: .residPdrop)
        try c.encode(embdPdrop, forKey: .embdPdrop)
        try c.encode(attnPdrop, forKey: .attnPdrop)
        try c.encode(layerNormEpsilon, forKey: .layerNormEpsilon)
        try c.encode(initializerRange, forKey: .initializerRange)
        try c.encode(scaleAttnWeights, forKey: .scaleAttnWeights)
        try c.encode(scaleAttnByInverseLayerIdx, forKey: .scaleAttnByInverseLayerIdx)
        try c.encode(positionEmbeddingType, forKey: .positionEmbeddingType)
        try c.encode(ropeBase, forKey: .ropeBase)
        try c.encode(padTokenID, forKey: .padTokenID)
        try c.encode(bosTokenID, forKey: .bosTokenID)
        try c.encode(eosTokenID, forKey: .eosTokenID)
        try c.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try c.encode(useCache, forKey: .useCache)
    }
}

public struct MossTTSNanoConfig: Codable, Sendable {
    public var modelType: String
    public var modelPath: String?
    public var gpt2Config: MossGPT2Config
    public var nVQ: Int
    public var audioVocabSize: Int
    public var audioCodebookSizes: [Int]
    public var audioPadTokenID: Int
    public var padTokenID: Int
    public var imStartTokenID: Int
    public var imEndTokenID: Int
    public var audioStartTokenID: Int
    public var audioEndTokenID: Int
    public var audioUserSlotTokenID: Int
    public var audioAssistantSlotTokenID: Int
    public var audioTokenizerType: String
    public var audioTokenizerPretrainedNameOrPath: String?
    public var audioTokenizerSampleRate: Int
    public var tokenizerUseFast: Bool
    public var attnImplementation: String
    public var localTransformerLayers: Int
    public var localTransformerAttnImplementation: String
    public var initializerRange: Float
    public var maxPositionEmbeddings: Int
    public var hiddenSize: Int
    public var vocabSize: Int

    public init(
        modelType: String = "moss_tts_nano",
        modelPath: String? = nil,
        gpt2Config: MossGPT2Config = MossGPT2Config(),
        nVQ: Int = 16,
        audioVocabSize: Int = 1_024,
        audioCodebookSizes: [Int]? = nil,
        audioPadTokenID: Int = 1_024,
        padTokenID: Int = 3,
        imStartTokenID: Int = 4,
        imEndTokenID: Int = 5,
        audioStartTokenID: Int = 6,
        audioEndTokenID: Int = 7,
        audioUserSlotTokenID: Int = 8,
        audioAssistantSlotTokenID: Int = 9,
        audioTokenizerType: String = "moss-audio-tokenizer-nano",
        audioTokenizerPretrainedNameOrPath: String? = nil,
        audioTokenizerSampleRate: Int = 48_000,
        tokenizerUseFast: Bool = false,
        attnImplementation: String = "sdpa",
        localTransformerLayers: Int = 1,
        localTransformerAttnImplementation: String? = nil,
        initializerRange: Float = 0.02,
        maxPositionEmbeddings: Int? = nil,
        hiddenSize: Int? = nil,
        vocabSize: Int? = nil
    ) throws {
        let sizes = audioCodebookSizes ?? Array(repeating: audioVocabSize, count: nVQ)
        guard sizes.count == nVQ else {
            throw DecodingError.dataCorrupted(.init(
                codingPath: [],
                debugDescription: "audio_codebook_sizes must have one entry per VQ channel"
            ))
        }

        self.modelType = modelType
        self.modelPath = modelPath
        self.gpt2Config = gpt2Config
        self.nVQ = nVQ
        self.audioVocabSize = audioVocabSize
        self.audioCodebookSizes = sizes
        self.audioPadTokenID = audioPadTokenID
        self.padTokenID = padTokenID
        self.imStartTokenID = imStartTokenID
        self.imEndTokenID = imEndTokenID
        self.audioStartTokenID = audioStartTokenID
        self.audioEndTokenID = audioEndTokenID
        self.audioUserSlotTokenID = audioUserSlotTokenID
        self.audioAssistantSlotTokenID = audioAssistantSlotTokenID
        self.audioTokenizerType = audioTokenizerType
        self.audioTokenizerPretrainedNameOrPath = audioTokenizerPretrainedNameOrPath
        self.audioTokenizerSampleRate = audioTokenizerSampleRate
        self.tokenizerUseFast = tokenizerUseFast
        self.attnImplementation = attnImplementation
        self.localTransformerLayers = localTransformerLayers
        self.localTransformerAttnImplementation = localTransformerAttnImplementation ?? attnImplementation
        self.initializerRange = initializerRange
        self.maxPositionEmbeddings = maxPositionEmbeddings ?? gpt2Config.nPositions
        self.hiddenSize = hiddenSize ?? gpt2Config.nEmbd
        self.vocabSize = vocabSize ?? gpt2Config.vocabSize
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case modelPath = "model_path"
        case gpt2Config = "gpt2_config"
        case nVQ = "n_vq"
        case audioVocabSize = "audio_vocab_size"
        case audioCodebookSizes = "audio_codebook_sizes"
        case audioPadTokenID = "audio_pad_token_id"
        case padTokenID = "pad_token_id"
        case imStartTokenID = "im_start_token_id"
        case imEndTokenID = "im_end_token_id"
        case audioStartTokenID = "audio_start_token_id"
        case audioEndTokenID = "audio_end_token_id"
        case audioUserSlotTokenID = "audio_user_slot_token_id"
        case audioAssistantSlotTokenID = "audio_assistant_slot_token_id"
        case audioTokenizerType = "audio_tokenizer_type"
        case audioTokenizerPretrainedNameOrPath = "audio_tokenizer_pretrained_name_or_path"
        case audioTokenizerSampleRate = "audio_tokenizer_sample_rate"
        case tokenizerUseFast = "tokenizer_use_fast"
        case attnImplementation = "attn_implementation"
        case localTransformerLayers = "local_transformer_layers"
        case localTransformerAttnImplementation = "local_transformer_attn_implementation"
        case initializerRange = "initializer_range"
        case maxPositionEmbeddings = "max_position_embeddings"
        case hiddenSize = "hidden_size"
        case vocabSize = "vocab_size"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let gpt = try c.decodeIfPresent(MossGPT2Config.self, forKey: .gpt2Config) ?? MossGPT2Config()
        let nVQ = try c.decodeIfPresent(Int.self, forKey: .nVQ) ?? 16
        let audioVocabSize = try c.decodeIfPresent(Int.self, forKey: .audioVocabSize) ?? 1_024
        let audioCodebookSizes = try c.decodeIfPresent([Int].self, forKey: .audioCodebookSizes)

        try self.init(
            modelType: "moss_tts_nano",
            modelPath: try c.decodeIfPresent(String.self, forKey: .modelPath),
            gpt2Config: gpt,
            nVQ: nVQ,
            audioVocabSize: audioVocabSize,
            audioCodebookSizes: audioCodebookSizes,
            audioPadTokenID: try c.decodeIfPresent(Int.self, forKey: .audioPadTokenID) ?? 1_024,
            padTokenID: try c.decodeIfPresent(Int.self, forKey: .padTokenID) ?? 3,
            imStartTokenID: try c.decodeIfPresent(Int.self, forKey: .imStartTokenID) ?? 4,
            imEndTokenID: try c.decodeIfPresent(Int.self, forKey: .imEndTokenID) ?? 5,
            audioStartTokenID: try c.decodeIfPresent(Int.self, forKey: .audioStartTokenID) ?? 6,
            audioEndTokenID: try c.decodeIfPresent(Int.self, forKey: .audioEndTokenID) ?? 7,
            audioUserSlotTokenID: try c.decodeIfPresent(Int.self, forKey: .audioUserSlotTokenID) ?? 8,
            audioAssistantSlotTokenID: try c.decodeIfPresent(Int.self, forKey: .audioAssistantSlotTokenID) ?? 9,
            audioTokenizerType: try c.decodeIfPresent(String.self, forKey: .audioTokenizerType) ?? "moss-audio-tokenizer-nano",
            audioTokenizerPretrainedNameOrPath: try c.decodeIfPresent(String.self, forKey: .audioTokenizerPretrainedNameOrPath),
            audioTokenizerSampleRate: try c.decodeIfPresent(Int.self, forKey: .audioTokenizerSampleRate) ?? 48_000,
            tokenizerUseFast: try c.decodeIfPresent(Bool.self, forKey: .tokenizerUseFast) ?? false,
            attnImplementation: try c.decodeIfPresent(String.self, forKey: .attnImplementation) ?? "sdpa",
            localTransformerLayers: try c.decodeIfPresent(Int.self, forKey: .localTransformerLayers) ?? 1,
            localTransformerAttnImplementation: try c.decodeIfPresent(String.self, forKey: .localTransformerAttnImplementation),
            initializerRange: try c.decodeIfPresent(Float.self, forKey: .initializerRange) ?? 0.02,
            maxPositionEmbeddings: try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings),
            hiddenSize: try c.decodeIfPresent(Int.self, forKey: .hiddenSize),
            vocabSize: try c.decodeIfPresent(Int.self, forKey: .vocabSize)
        )
    }

    public func localGPT2Config() -> MossGPT2Config {
        MossGPT2Config(
            modelType: gpt2Config.modelType,
            vocabSize: gpt2Config.vocabSize,
            nPositions: nVQ + 1,
            nCtx: nVQ + 1,
            nEmbd: gpt2Config.nEmbd,
            nLayer: localTransformerLayers,
            nHead: gpt2Config.nHead,
            nInner: gpt2Config.nInner,
            activationFunction: gpt2Config.activationFunction,
            residPdrop: gpt2Config.residPdrop,
            embdPdrop: gpt2Config.embdPdrop,
            attnPdrop: gpt2Config.attnPdrop,
            layerNormEpsilon: gpt2Config.layerNormEpsilon,
            initializerRange: gpt2Config.initializerRange,
            scaleAttnWeights: gpt2Config.scaleAttnWeights,
            scaleAttnByInverseLayerIdx: gpt2Config.scaleAttnByInverseLayerIdx,
            positionEmbeddingType: gpt2Config.positionEmbeddingType,
            ropeBase: gpt2Config.ropeBase,
            padTokenID: gpt2Config.padTokenID,
            bosTokenID: gpt2Config.bosTokenID,
            eosTokenID: gpt2Config.eosTokenID,
            tieWordEmbeddings: gpt2Config.tieWordEmbeddings,
            useCache: gpt2Config.useCache
        )
    }
}
