import MLX
import MLXNN

final class Flux2SingleTransformerBlock: Module {
    @ModuleInfo var norm: LayerNorm
    @ModuleInfo var attn: Flux2ParallelSelfAttention

    init(dim: Int, numAttentionHeads: Int, attentionHeadDim: Int, mlpRatio: Float = 3.0) {
        self._norm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        self._attn.wrappedValue = Flux2ParallelSelfAttention(
            dim: dim, heads: numAttentionHeads, dimHead: attentionHeadDim, mlpRatio: mlpRatio
        )
    }

    func callAsFunction(
        hiddenStates: MLXArray,
        tembModParams: ModulationParams,
        imageRotaryEmb: (cos: MLXArray, sin: MLXArray)?
    ) -> MLXArray {
        var normHS = norm(hiddenStates)
        normHS = (1 + tembModParams.scale) * normHS + tembModParams.shift
        let attnOutput = attn(hiddenStates: normHS, imageRotaryEmb: imageRotaryEmb)
        return hiddenStates + tembModParams.gate * attnOutput
    }
}
