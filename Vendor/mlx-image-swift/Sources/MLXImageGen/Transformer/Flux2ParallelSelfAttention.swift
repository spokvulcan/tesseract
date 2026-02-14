import MLX
import MLXNN

final class Flux2ParallelSelfAttention: Module {
    let heads: Int
    let dimHead: Int
    let innerDim: Int
    let mlpHiddenDim: Int

    @ModuleInfo(key: "to_qkv_mlp_proj") var toQKVMlpProj: Linear
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm
    let mlpAct: Flux2SwiGLU
    @ModuleInfo(key: "to_out") var toOut: Linear

    init(dim: Int, heads: Int, dimHead: Int, mlpRatio: Float = 3.0) {
        self.heads = heads
        self.dimHead = dimHead
        self.innerDim = heads * dimHead
        self.mlpHiddenDim = Int(Float(dim) * mlpRatio)

        self._toQKVMlpProj.wrappedValue = Linear(dim, innerDim * 3 + mlpHiddenDim * 2, bias: false)
        self._normQ.wrappedValue = RMSNorm(dimensions: dimHead, eps: 1e-5)
        self._normK.wrappedValue = RMSNorm(dimensions: dimHead, eps: 1e-5)
        self.mlpAct = Flux2SwiGLU()
        self._toOut.wrappedValue = Linear(innerDim + mlpHiddenDim, dim, bias: false)
    }

    func callAsFunction(
        hiddenStates: MLXArray,
        imageRotaryEmb: (cos: MLXArray, sin: MLXArray)?
    ) -> MLXArray {
        let proj = toQKVMlpProj(hiddenStates)
        let splitResult = proj.split(indices: [innerDim * 3], axis: -1)
        let qkv = splitResult[0]
        let mlpHidden = splitResult[1]

        let qkvParts = qkv.split(parts: 3, axis: -1)
        let (batch, seqLen, _) = (hiddenStates.dim(0), hiddenStates.dim(1), hiddenStates.dim(2))

        var query = qkvParts[0].reshaped(batch, seqLen, heads, dimHead).transposed(0, 2, 1, 3)
        var key = qkvParts[1].reshaped(batch, seqLen, heads, dimHead).transposed(0, 2, 1, 3)
        let value = qkvParts[2].reshaped(batch, seqLen, heads, dimHead).transposed(0, 2, 1, 3)

        let inputDtype = query.dtype
        // Normalize in f32, stay in f32 for RoPE
        query = normQ(query.asType(.float32))
        key = normK(key.asType(.float32))

        if let (cos, sin) = imageRotaryEmb {
            (query, key) = AttentionUtils.applyRopeBSHD(
                xq: query, xk: key, cos: cos, sin: sin,
                outputDtype: inputDtype
            )
        }

        var result = AttentionUtils.computeAttention(
            query: query, key: key, value: value,
            batchSize: batch, numHeads: heads, headDim: dimHead
        )

        let mlpOut = mlpAct(mlpHidden)
        result = MLX.concatenated([result, mlpOut], axis: -1)
        return toOut(result)
    }
}
