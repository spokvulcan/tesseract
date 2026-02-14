import MLX
import MLXNN

/// Context refiner block — same structure as ZImageTransformerBlock but without adaLN modulation.
/// No timestep conditioning, no gates.
final class ZImageContextBlock: Module {
    @ModuleInfo var attention: ZImageAttention
    @ModuleInfo(key: "feed_forward") var feedForward: ZImageFeedForward
    @ModuleInfo(key: "attention_norm1") var attentionNorm1: RMSNorm
    @ModuleInfo(key: "attention_norm2") var attentionNorm2: RMSNorm
    @ModuleInfo(key: "ffn_norm1") var ffnNorm1: RMSNorm
    @ModuleInfo(key: "ffn_norm2") var ffnNorm2: RMSNorm

    init(dim: Int, nHeads: Int, normEps: Float = 1e-5) {
        self._attention.wrappedValue = ZImageAttention(dim: dim, nHeads: nHeads, eps: 1e-5)
        self._feedForward.wrappedValue = ZImageFeedForward(dim: dim, hiddenDim: Int(Float(dim) / 3.0 * 8.0))
        self._attentionNorm1.wrappedValue = RMSNorm(dimensions: dim, eps: normEps)
        self._attentionNorm2.wrappedValue = RMSNorm(dimensions: dim, eps: normEps)
        self._ffnNorm1.wrappedValue = RMSNorm(dimensions: dim, eps: normEps)
        self._ffnNorm2.wrappedValue = RMSNorm(dimensions: dim, eps: normEps)
    }

    func callAsFunction(
        x: MLXArray,
        attnMask: MLXArray?,
        freqsCis: MLXArray?
    ) -> MLXArray {
        var hs = x
        let attnOut = attention(
            hiddenStates: attentionNorm1(hs),
            attentionMask: attnMask,
            freqsCis: freqsCis
        )
        hs = hs + attentionNorm2(attnOut)

        let ffnOut = feedForward(ffnNorm1(hs))
        hs = hs + ffnNorm2(ffnOut)
        return hs
    }
}
