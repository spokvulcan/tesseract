import MLX
import MLXNN

/// Single-stream transformer block with adaLN modulation using tanh gates.
/// Used by both noise_refiner and main layers.
/// Modulation: Linear(min(dim,256) → 4*dim) → split → (1+scale_msa, tanh(gate_msa), 1+scale_mlp, tanh(gate_mlp))
final class ZImageTransformerBlock: Module {
    @ModuleInfo var attention: ZImageAttention
    @ModuleInfo(key: "feed_forward") var feedForward: ZImageFeedForward
    @ModuleInfo(key: "attention_norm1") var attentionNorm1: RMSNorm
    @ModuleInfo(key: "attention_norm2") var attentionNorm2: RMSNorm
    @ModuleInfo(key: "ffn_norm1") var ffnNorm1: RMSNorm
    @ModuleInfo(key: "ffn_norm2") var ffnNorm2: RMSNorm
    @ModuleInfo(key: "adaLN_modulation") var adaLNModulation: Linear

    let dim: Int

    init(dim: Int, nHeads: Int, normEps: Float = 1e-5) {
        self.dim = dim
        self._attention.wrappedValue = ZImageAttention(dim: dim, nHeads: nHeads, eps: 1e-5)
        self._feedForward.wrappedValue = ZImageFeedForward(dim: dim, hiddenDim: Int(Float(dim) / 3.0 * 8.0))
        self._attentionNorm1.wrappedValue = RMSNorm(dimensions: dim, eps: normEps)
        self._attentionNorm2.wrappedValue = RMSNorm(dimensions: dim, eps: normEps)
        self._ffnNorm1.wrappedValue = RMSNorm(dimensions: dim, eps: normEps)
        self._ffnNorm2.wrappedValue = RMSNorm(dimensions: dim, eps: normEps)
        self._adaLNModulation.wrappedValue = Linear(min(dim, 256), 4 * dim)
    }

    func callAsFunction(
        x: MLXArray,
        attnMask: MLXArray?,
        freqsCis: MLXArray?,
        tEmb: MLXArray
    ) -> MLXArray {
        // Compute modulation: [B, 4*dim] → expand → [B, 1, 4*dim]
        let modulation = adaLNModulation(tEmb).expandedDimensions(axis: 1)
        let splits = MLX.split(modulation, parts: 4, axis: 2)
        let scaleMsa = 1.0 + splits[0]
        let gateMsa = MLX.tanh(splits[1])
        let scaleMlp = 1.0 + splits[2]
        let gateMlp = MLX.tanh(splits[3])

        // Attention with modulation
        var hs = x
        let attnOut = attention(
            hiddenStates: attentionNorm1(hs) * scaleMsa,
            attentionMask: attnMask,
            freqsCis: freqsCis
        )
        hs = hs + gateMsa * attentionNorm2(attnOut)

        // FFN with modulation
        let ffnOut = feedForward(ffnNorm1(hs) * scaleMlp)
        hs = hs + gateMlp * ffnNorm2(ffnOut)
        return hs
    }
}
