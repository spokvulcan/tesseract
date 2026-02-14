import Foundation
import MLX
import MLXFast
import MLXNN

final class Qwen3RMSNorm: Module {
    let weight: MLXArray
    let eps: Float

    init(hiddenSize: Int, eps: Float = 1e-6) {
        self.weight = MLXArray.ones([hiddenSize])
        self.eps = eps
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let inputDtype = hiddenStates.dtype
        let hs = hiddenStates.asType(.float32)
        let variance = MLX.mean(hs * hs, axis: -1, keepDims: true)
        let normed = hs * MLX.rsqrt(variance + MLXArray(eps))
        return (weight.asType(.float32) * normed).asType(inputDtype)
    }
}

final class Qwen3RotaryEmbedding: Module {
    let dim: Int
    let base: Float
    let scalingFactor: Float
    let invFreq: MLXArray

    init(dim: Int, maxPositionEmbeddings: Int = 40960, base: Float = 1_000_000.0, scalingFactor: Float = 1.0) {
        self.dim = dim
        self.base = base
        self.scalingFactor = scalingFactor
        let indices = stride(from: 0, to: dim, by: 2).map { Float($0) / Float(dim) }
        self.invFreq = 1.0 / MLX.pow(MLXArray(base), MLXArray(indices))
    }

    func callAsFunction(_ x: MLXArray, positionIds: MLXArray) -> (cos: MLXArray, sin: MLXArray) {
        var posIds = positionIds
        if posIds.ndim == 1 {
            posIds = posIds.expandedDimensions(axis: 0)
        }
        let invF = invFreq.expandedDimensions(axes: [0, 1])
        let pos = posIds.asType(.float32).expandedDimensions(axis: -1)
        let freqs = pos * invF
        let emb = MLX.concatenated([freqs, freqs], axis: -1)
        let cosEmb = MLX.cos(emb) * MLXArray(scalingFactor)
        let sinEmb = MLX.sin(emb) * MLXArray(scalingFactor)
        return (cosEmb.asType(x.dtype), sinEmb.asType(x.dtype))
    }
}

final class Qwen3MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        downProj(silu(gateProj(hiddenStates)) * upProj(hiddenStates))
    }
}

final class Qwen3Attention: Module {
    let hiddenSize: Int
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let headDim: Int
    let numKeyValueGroups: Int
    let scaling: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: Qwen3RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: Qwen3RMSNorm

    init(
        hiddenSize: Int,
        numAttentionHeads: Int,
        numKeyValueHeads: Int,
        headDim: Int,
        attentionBias: Bool,
        rmsNormEps: Float
    ) {
        self.hiddenSize = hiddenSize
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.numKeyValueGroups = numAttentionHeads / numKeyValueHeads
        self.scaling = 1.0 / Foundation.sqrt(Float(headDim))

        self._qProj.wrappedValue = Linear(hiddenSize, numAttentionHeads * headDim, bias: attentionBias)
        self._kProj.wrappedValue = Linear(hiddenSize, numKeyValueHeads * headDim, bias: attentionBias)
        self._vProj.wrappedValue = Linear(hiddenSize, numKeyValueHeads * headDim, bias: attentionBias)
        self._oProj.wrappedValue = Linear(numAttentionHeads * headDim, hiddenSize, bias: attentionBias)
        self._qNorm.wrappedValue = Qwen3RMSNorm(hiddenSize: headDim, eps: rmsNormEps)
        self._kNorm.wrappedValue = Qwen3RMSNorm(hiddenSize: headDim, eps: rmsNormEps)
    }

    func callAsFunction(
        hiddenStates: MLXArray,
        attentionMask: MLXArray?,
        positionEmbeddings: (cos: MLXArray, sin: MLXArray)?
    ) -> MLXArray {
        let (bsz, qLen, _) = (hiddenStates.dim(0), hiddenStates.dim(1), hiddenStates.dim(2))

        var queryStates = qProj(hiddenStates).reshaped(bsz, qLen, numAttentionHeads, headDim)
        var keyStates = kProj(hiddenStates).reshaped(bsz, qLen, numKeyValueHeads, headDim)
        var valueStates = vProj(hiddenStates).reshaped(bsz, qLen, numKeyValueHeads, headDim)

        queryStates = qNorm(queryStates)
        keyStates = kNorm(keyStates)

        queryStates = queryStates.transposed(0, 2, 1, 3)
        keyStates = keyStates.transposed(0, 2, 1, 3)
        valueStates = valueStates.transposed(0, 2, 1, 3)

        if let (cos, sin) = positionEmbeddings {
            (queryStates, keyStates) = Self.applyRotaryPosEmb(
                q: queryStates, k: keyStates, cos: cos, sin: sin
            )
        }

        // Expand KV heads if GQA
        if numKeyValueHeads != numAttentionHeads {
            keyStates = Self.repeatKV(keyStates, nRep: numKeyValueGroups)
            valueStates = Self.repeatKV(valueStates, nRep: numKeyValueGroups)
        }

        let kvLen = keyStates.dim(2)
        var mask = attentionMask
        if let m = mask {
            mask = m[0..., 0..., 0..., ..<kvLen]
        }

        let qF32 = queryStates.asType(.float32)
        let kF32 = keyStates.asType(.float32)
        let vF32 = valueStates.asType(.float32)

        var attnOutput = MLXFast.scaledDotProductAttention(
            queries: qF32, keys: kF32, values: vF32, scale: scaling, mask: mask
        )
        attnOutput = attnOutput.asType(queryStates.dtype)
        attnOutput = attnOutput.transposed(0, 2, 1, 3).reshaped(bsz, qLen, numAttentionHeads * headDim)
        return oProj(attnOutput)
    }

    static func repeatKV(_ hiddenStates: MLXArray, nRep: Int) -> MLXArray {
        if nRep == 1 { return hiddenStates }
        let (batch, numKVHeads, slen, headDim) = (
            hiddenStates.dim(0), hiddenStates.dim(1), hiddenStates.dim(2), hiddenStates.dim(3)
        )
        let expanded = hiddenStates.expandedDimensions(axis: 2)
        let broadcasted = MLX.broadcast(expanded, to: [batch, numKVHeads, nRep, slen, headDim])
        return broadcasted.reshaped(batch, numKVHeads * nRep, slen, headDim)
    }

    static func applyRotaryPosEmb(
        q: MLXArray, k: MLXArray, cos: MLXArray, sin: MLXArray
    ) -> (MLXArray, MLXArray) {
        let cosE = cos.expandedDimensions(axis: 1)
        let sinE = sin.expandedDimensions(axis: 1)
        let qEmbed = q * cosE + rotateHalf(q) * sinE
        let kEmbed = k * cosE + rotateHalf(k) * sinE
        return (qEmbed, kEmbed)
    }

    static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let halfDim = x.dim(-1) / 2
        let x1 = x[.ellipsis, ..<halfDim]
        let x2 = x[.ellipsis, halfDim...]
        return MLX.concatenated([-x2, x1], axis: -1)
    }
}

final class Qwen3DecoderLayer: Module {
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: Qwen3RMSNorm
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen3Attention
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: Qwen3RMSNorm
    @ModuleInfo var mlp: Qwen3MLP

    init(config: Flux2Configuration.TextEncoder) {
        self._inputLayernorm.wrappedValue = Qwen3RMSNorm(hiddenSize: config.hiddenSize, eps: config.rmsNormEps)
        self._selfAttn.wrappedValue = Qwen3Attention(
            hiddenSize: config.hiddenSize,
            numAttentionHeads: config.numAttentionHeads,
            numKeyValueHeads: config.numKeyValueHeads,
            headDim: config.headDim,
            attentionBias: config.attentionBias,
            rmsNormEps: config.rmsNormEps
        )
        self._postAttentionLayernorm.wrappedValue = Qwen3RMSNorm(hiddenSize: config.hiddenSize, eps: config.rmsNormEps)
        self._mlp.wrappedValue = Qwen3MLP(hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
    }

    func callAsFunction(
        hiddenStates: MLXArray,
        attentionMask: MLXArray?,
        positionEmbeddings: (cos: MLXArray, sin: MLXArray)?
    ) -> MLXArray {
        var residual = hiddenStates
        var hs = inputLayernorm(hiddenStates)
        hs = selfAttn(hiddenStates: hs, attentionMask: attentionMask, positionEmbeddings: positionEmbeddings)
        hs = residual + hs
        residual = hs
        hs = postAttentionLayernorm(hs)
        hs = mlp(hs)
        return residual + hs
    }
}

final class Qwen3TextEncoder: Module {
    let config: Flux2Configuration.TextEncoder
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [Qwen3DecoderLayer]
    @ModuleInfo var norm: Qwen3RMSNorm
    let rotaryEmb: Qwen3RotaryEmbedding

    init(config: Flux2Configuration.TextEncoder) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in
            Qwen3DecoderLayer(config: config)
        }
        self._norm.wrappedValue = Qwen3RMSNorm(hiddenSize: config.hiddenSize, eps: config.rmsNormEps)
        self.rotaryEmb = Qwen3RotaryEmbedding(
            dim: config.headDim,
            maxPositionEmbeddings: config.maxPositionEmbeddings,
            base: config.ropeTheta,
            scalingFactor: config.attentionScaling
        )
    }

    func callAsFunction(
        inputIds: MLXArray,
        attentionMask: MLXArray? = nil,
        targetHiddenStateLayers: Set<Int>? = nil
    ) -> (MLXArray, [Int: MLXArray]?) {
        let (batchSize, seqLen) = (inputIds.dim(0), inputIds.dim(1))
        if flux2Profiling {
            NSLog("[MLXImageGen] TextEncoder input: batch=%d, seqLen=%d", batchSize, seqLen)
        }

        var hiddenStates = embedTokens(inputIds)

        let mask: MLXArray
        if let attentionMask {
            mask = attentionMask
        } else {
            mask = MLXArray.ones([batchSize, seqLen], type: Int32.self)
        }

        // Build 4D attention mask: padding + causal
        let paddingMask = MLX.where(
            mask .== 1,
            MLXArray.zeros([batchSize, seqLen]).asType(hiddenStates.dtype),
            MLXArray.full([batchSize, seqLen], values: MLXArray(-Float.infinity)).asType(hiddenStates.dtype)
        ).expandedDimensions(axes: [1, 2])

        let causalMask: MLXArray
        if seqLen == 1 {
            causalMask = MLXArray.zeros([batchSize, 1, 1, 1]).asType(hiddenStates.dtype)
        } else {
            let idx = MLXArray(0..<Int32(seqLen))
            let j = idx.expandedDimensions(axis: 0)
            let i = idx.expandedDimensions(axis: 1)
            let triBool = j .> i
            let zeros2D = MLXArray.zeros([seqLen, seqLen]).asType(hiddenStates.dtype)
            let negInf2D = MLXArray.full([seqLen, seqLen], values: MLXArray(-Float.infinity)).asType(hiddenStates.dtype)
            let triMask = MLX.where(triBool, negInf2D, zeros2D)
                .expandedDimensions(axes: [0, 1])
            causalMask = MLX.broadcast(triMask, to: [batchSize, 1, seqLen, seqLen])
        }

        let attentionMask4D = causalMask + paddingMask

        // Position IDs
        let positionIds = MLX.broadcast(
            MLXArray(0..<Int32(seqLen)).expandedDimensions(axis: 0),
            to: [batchSize, seqLen]
        )
        let positionEmbeddings = rotaryEmb(hiddenStates, positionIds: positionIds)

        var hiddenStatesMap: [Int: MLXArray]? = targetHiddenStateLayers != nil ? [:] : nil
        for (idx, layer) in layers.enumerated() {
            hiddenStates = layer(
                hiddenStates: hiddenStates,
                attentionMask: attentionMask4D,
                positionEmbeddings: positionEmbeddings
            )
            if let targets = targetHiddenStateLayers, targets.contains(idx) {
                hiddenStatesMap?[idx] = hiddenStates
            }
            if flux2Profiling && (idx == 0 || (idx + 1) % 6 == 0 || idx == layers.count - 1) {
                NSLog("[MLXImageGen] [PROFILE] Layer %d/%d done", idx + 1, layers.count)
            }
        }

        hiddenStates = norm(hiddenStates)
        return (hiddenStates, hiddenStatesMap)
    }

    func getPromptEmbeds(
        inputIds: MLXArray,
        attentionMask: MLXArray? = nil,
        hiddenStateLayers: [Int] = [9, 18, 27]
    ) -> MLXArray {
        // hiddenStateLayers uses 0-indexed layer numbers (output of layer N, not including initial embed)
        let targetSet = Set(hiddenStateLayers)
        let (_, hiddenStatesMap) = self(
            inputIds: inputIds,
            attentionMask: attentionMask,
            targetHiddenStateLayers: targetSet
        )
        guard let hsMap = hiddenStatesMap else {
            fatalError("Hidden states not available for prompt embedding")
        }
        let stacked = MLX.stacked(hiddenStateLayers.map { hsMap[$0]! }, axis: 1)
        let (batchSize, numLayers, seqLen, hiddenDim) = (
            stacked.dim(0), stacked.dim(1), stacked.dim(2), stacked.dim(3)
        )
        let transposed = stacked.transposed(0, 2, 1, 3)
        return transposed.reshaped(batchSize, seqLen, numLayers * hiddenDim)
    }
}
