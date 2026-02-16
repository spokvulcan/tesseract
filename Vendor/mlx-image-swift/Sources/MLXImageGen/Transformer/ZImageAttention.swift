import Foundation
import MLX
import MLXFast
import MLXNN

/// Self-attention with QK norm + complex-interleaved RoPE + SDPA.
/// All projections are bias=false. to_out is a single Linear (not a list).
final class ZImageAttention: Module {
    let dim: Int
    let nHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm

    init(dim: Int, nHeads: Int, eps: Float = 1e-5) {
        self.dim = dim
        self.nHeads = nHeads
        self.headDim = dim / nHeads
        self.scale = pow(Float(dim / nHeads), -0.5)

        self._toQ.wrappedValue = Linear(dim, dim, bias: false)
        self._toK.wrappedValue = Linear(dim, dim, bias: false)
        self._toV.wrappedValue = Linear(dim, dim, bias: false)
        self._toOut.wrappedValue = Linear(dim, dim, bias: false)
        self._normQ.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        self._normK.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
    }

    func callAsFunction(
        hiddenStates: MLXArray,
        attentionMask: MLXArray?,
        freqsCis: MLXArray?
    ) -> MLXArray {
        let (batchSize, seqLen, _) = (hiddenStates.dim(0), hiddenStates.dim(1), hiddenStates.dim(2))

        // Project to Q, K, V → [B, S, H, D]
        var query = toQ(hiddenStates).reshaped(batchSize, seqLen, nHeads, headDim)
        var key = toK(hiddenStates).reshaped(batchSize, seqLen, nHeads, headDim)
        var value = toV(hiddenStates).reshaped(batchSize, seqLen, nHeads, headDim)

        // QK normalization
        query = normQ(query)
        key = normK(key)

        // Apply RoPE in BSHD format (before transpose)
        if let fc = freqsCis {
            query = ZImageRopeEmbedder.applyRotaryEmb(query, freqsCis: fc)
            key = ZImageRopeEmbedder.applyRotaryEmb(key, freqsCis: fc)
        }

        // Transpose to [B, H, S, D] for SDPA
        query = query.transposed(0, 2, 1, 3)
        key = key.transposed(0, 2, 1, 3)
        value = value.transposed(0, 2, 1, 3)

        // Convert boolean mask to additive mask
        var mask: MLXArray? = nil
        if let attentionMask {
            // attentionMask: [1, seqLen] bool → [1, 1, 1, seqLen] additive
            mask = MLX.where(
                attentionMask[.ellipsis, .newAxis, .newAxis, 0...],
                MLXArray(Float(0.0)),
                MLXArray(-Float.infinity)
            )
        }

        var hs = MLXFast.scaledDotProductAttention(
            queries: query, keys: key, values: value, scale: scale, mask: mask
        )

        // [B, H, S, D] → [B, S, H*D]
        hs = hs.transposed(0, 2, 1, 3).reshaped(batchSize, seqLen, dim)
        return toOut(hs)
    }
}
