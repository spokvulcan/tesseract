import MLX
import MLXNN

final class Flux2Attention: Module {
    let heads: Int
    let dimHead: Int
    let innerDim: Int
    let addedKVProjDim: Int?

    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm
    @ModuleInfo(key: "to_out") var toOut: Linear

    @ModuleInfo(key: "norm_added_q") var normAddedQ: RMSNorm?
    @ModuleInfo(key: "norm_added_k") var normAddedK: RMSNorm?
    @ModuleInfo(key: "add_q_proj") var addQProj: Linear?
    @ModuleInfo(key: "add_k_proj") var addKProj: Linear?
    @ModuleInfo(key: "add_v_proj") var addVProj: Linear?
    @ModuleInfo(key: "to_add_out") var toAddOut: Linear?

    init(dim: Int, heads: Int, dimHead: Int, addedKVProjDim: Int? = nil) {
        self.heads = heads
        self.dimHead = dimHead
        self.innerDim = heads * dimHead
        self.addedKVProjDim = addedKVProjDim

        self._toQ.wrappedValue = Linear(dim, innerDim, bias: false)
        self._toK.wrappedValue = Linear(dim, innerDim, bias: false)
        self._toV.wrappedValue = Linear(dim, innerDim, bias: false)
        self._normQ.wrappedValue = RMSNorm(dimensions: dimHead, eps: 1e-5)
        self._normK.wrappedValue = RMSNorm(dimensions: dimHead, eps: 1e-5)
        self._toOut.wrappedValue = Linear(innerDim, dim, bias: false)

        if let addedKVProjDim {
            self._normAddedQ.wrappedValue = RMSNorm(dimensions: dimHead, eps: 1e-5)
            self._normAddedK.wrappedValue = RMSNorm(dimensions: dimHead, eps: 1e-5)
            self._addQProj.wrappedValue = Linear(addedKVProjDim, innerDim, bias: false)
            self._addKProj.wrappedValue = Linear(addedKVProjDim, innerDim, bias: false)
            self._addVProj.wrappedValue = Linear(addedKVProjDim, innerDim, bias: false)
            self._toAddOut.wrappedValue = Linear(innerDim, dim, bias: false)
        }
    }

    func callAsFunction(
        hiddenStates: MLXArray,
        encoderHiddenStates: MLXArray?,
        imageRotaryEmb: (cos: MLXArray, sin: MLXArray)?
    ) -> (MLXArray, MLXArray?) {
        var (query, key, value) = AttentionUtils.processQKV(
            hiddenStates: hiddenStates,
            toQ: toQ, toK: toK, toV: toV,
            normQ: normQ, normK: normK,
            numHeads: heads, headDim: dimHead
        )

        var encoderOutput: MLXArray? = nil
        if let encoderHiddenStates, let addQProj, let addKProj, let addVProj,
           let normAddedQ, let normAddedK {
            let (encQuery, encKey, encValue) = AttentionUtils.processQKV(
                hiddenStates: encoderHiddenStates,
                toQ: addQProj, toK: addKProj, toV: addVProj,
                normQ: normAddedQ, normK: normAddedK,
                numHeads: heads, headDim: dimHead
            )
            query = MLX.concatenated([encQuery, query], axis: 2)
            key = MLX.concatenated([encKey, key], axis: 2)
            value = MLX.concatenated([encValue, value], axis: 2)
        }

        if let (cos, sin) = imageRotaryEmb {
            (query, key) = AttentionUtils.applyRopeBSHD(
                xq: query, xk: key, cos: cos, sin: sin,
                outputDtype: hiddenStates.dtype
            )
        }

        var result = AttentionUtils.computeAttention(
            query: query, key: key, value: value,
            batchSize: hiddenStates.dim(0), numHeads: heads, headDim: dimHead
        )

        if let encoderHiddenStates, addedKVProjDim != nil, let toAddOut {
            let encSeqLen = encoderHiddenStates.dim(1)
            encoderOutput = toAddOut(result[0..., ..<encSeqLen, 0...])
            result = result[0..., encSeqLen..., 0...]
        }

        result = toOut(result)
        return (result, encoderOutput)
    }
}
