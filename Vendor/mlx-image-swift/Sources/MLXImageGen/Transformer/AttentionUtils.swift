import Foundation
import MLX
import MLXFast
import MLXNN

enum AttentionUtils {
    static func processQKV(
        hiddenStates: MLXArray,
        toQ: Linear,
        toK: Linear,
        toV: Linear,
        normQ: RMSNorm,
        normK: RMSNorm,
        numHeads: Int,
        headDim: Int
    ) -> (query: MLXArray, key: MLXArray, value: MLXArray) {
        let batchSize = hiddenStates.dim(0)
        let seqLen = hiddenStates.dim(1)

        var query = toQ(hiddenStates)
        var key = toK(hiddenStates)
        let value = toV(hiddenStates)

        // Reshape [B, S, H*D] -> [B, H, S, D]
        let queryR = query.reshaped(batchSize, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)
        let keyR = key.reshaped(batchSize, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)
        let valueR = value.reshaped(batchSize, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)

        // Apply normalization in float32 for stability — stays f32 for RoPE
        query = normQ(queryR.asType(.float32))
        key = normK(keyR.asType(.float32))

        return (query, key, valueR)
    }

    static func computeAttention(
        query: MLXArray,
        key: MLXArray,
        value: MLXArray,
        batchSize: Int,
        numHeads: Int,
        headDim: Int,
        mask: MLXArray? = nil
    ) -> MLXArray {
        let scale: Float = 1.0 / Foundation.sqrt(Float(query.dim(-1)))
        let hiddenStates = MLXFast.scaledDotProductAttention(
            queries: query, keys: key, values: value, scale: scale, mask: mask
        )
        // [B, H, S, D] -> [B, S, H, D] -> [B, S, H*D]
        return hiddenStates.transposed(0, 2, 1, 3).reshaped(batchSize, -1, numHeads * headDim)
    }

    static func applyRopeBSHD(
        xq: MLXArray, xk: MLXArray, cos: MLXArray, sin: MLXArray,
        outputDtype: DType? = nil
    ) -> (MLXArray, MLXArray) {
        let outDtype = outputDtype ?? xq.dtype
        let xqF = xq.dtype == .float32 ? xq : xq.asType(.float32)
        let xkF = xk.dtype == .float32 ? xk : xk.asType(.float32)
        // cos/sin shape: [seqLen, dim] -> [1, 1, seqLen, dim]
        let cosB = cos.reshaped(1, 1, cos.dim(0), cos.dim(1))
        let sinB = sin.reshaped(1, 1, sin.dim(0), sin.dim(1))

        func mix(_ x: MLXArray) -> MLXArray {
            let x2 = x.reshaped(x.dim(0), x.dim(1), x.dim(2), -1, 2)
            let real = x2[.ellipsis, 0]
            let imag = x2[.ellipsis, 1]
            let out0 = real * cosB + (-imag) * sinB
            let out1 = imag * cosB + real * sinB
            let out2 = MLX.stacked([out0, out1], axis: -1)
            return out2.reshaped(x.shape)
        }

        return (mix(xqF).asType(outDtype), mix(xkF).asType(outDtype))
    }
}
