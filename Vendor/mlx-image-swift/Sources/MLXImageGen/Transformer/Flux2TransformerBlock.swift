import MLX
import MLXNN

final class Flux2TransformerBlock: Module {
    @ModuleInfo var norm1: LayerNorm
    @ModuleInfo(key: "norm1_context") var norm1Context: LayerNorm
    @ModuleInfo var attn: Flux2Attention
    @ModuleInfo var norm2: LayerNorm
    @ModuleInfo var ff: Flux2FeedForward
    @ModuleInfo(key: "norm2_context") var norm2Context: LayerNorm
    @ModuleInfo(key: "ff_context") var ffContext: Flux2FeedForward

    init(dim: Int, numAttentionHeads: Int, attentionHeadDim: Int, mlpRatio: Float = 3.0) {
        self._norm1.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        self._norm1Context.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        self._attn.wrappedValue = Flux2Attention(
            dim: dim, heads: numAttentionHeads, dimHead: attentionHeadDim, addedKVProjDim: dim
        )
        self._norm2.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        self._ff.wrappedValue = Flux2FeedForward(dim: dim, mult: mlpRatio)
        self._norm2Context.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        self._ffContext.wrappedValue = Flux2FeedForward(dim: dim, mult: mlpRatio)
    }

    func callAsFunction(
        hiddenStates: MLXArray,
        encoderHiddenStates: MLXArray,
        tembModParamsImg: [ModulationParams],
        tembModParamsTxt: [ModulationParams],
        imageRotaryEmb: (cos: MLXArray, sin: MLXArray)?
    ) -> (encoderOut: MLXArray, hiddenOut: MLXArray) {
        let imgMSA = tembModParamsImg[0]
        let imgMLP = tembModParamsImg[1]
        let txtMSA = tembModParamsTxt[0]
        let txtMLP = tembModParamsTxt[1]

        // Image stream: norm + modulate
        var normHS = norm1(hiddenStates)
        normHS = (1 + imgMSA.scale) * normHS + imgMSA.shift
        // Text stream: norm + modulate
        var normEHS = norm1Context(encoderHiddenStates)
        normEHS = (1 + txtMSA.scale) * normEHS + txtMSA.shift

        // Joint attention
        let (attnOutput, encoderAttnOutput) = attn(
            hiddenStates: normHS,
            encoderHiddenStates: normEHS,
            imageRotaryEmb: imageRotaryEmb
        )

        var hs = hiddenStates + imgMSA.gate * attnOutput
        var ehs = encoderHiddenStates + txtMSA.gate * encoderAttnOutput!

        // Image FFN
        normHS = norm2(hs)
        normHS = (1 + imgMLP.scale) * normHS + imgMLP.shift
        hs = hs + imgMLP.gate * ff(normHS)

        // Text FFN
        normEHS = norm2Context(ehs)
        normEHS = (1 + txtMLP.scale) * normEHS + txtMLP.shift
        ehs = ehs + txtMLP.gate * ffContext(normEHS)

        return (ehs, hs)
    }
}
