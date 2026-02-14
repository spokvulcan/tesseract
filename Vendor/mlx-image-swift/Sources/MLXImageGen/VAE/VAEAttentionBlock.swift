import Foundation
import MLX
import MLXFast
import MLXNN

final class VAEAttentionBlock: Module {
    @ModuleInfo(key: "group_norm") var groupNorm: GroupNorm
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear

    init(channels: Int, groups: Int = 32, eps: Float = 1e-6) {
        self._groupNorm.wrappedValue = GroupNorm(groupCount: groups, dimensions: channels, eps: eps, pytorchCompatible: true)
        self._toQ.wrappedValue = Linear(channels, channels)
        self._toK.wrappedValue = Linear(channels, channels)
        self._toV.wrappedValue = Linear(channels, channels)
        self._toOut.wrappedValue = Linear(channels, channels)
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        // Input: NCHW -> NHWC
        let nhwc = hiddenStates.transposed(0, 2, 3, 1)
        let (batch, height, width, channels) = (nhwc.dim(0), nhwc.dim(1), nhwc.dim(2), nhwc.dim(3))

        let normed = groupNorm(nhwc.asType(.float32)).asType(.bfloat16)
        let seqLen = height * width

        // Q/K/V: [B, H*W, 1, C] -> transpose to [B, 1, H*W, C]
        var q = toQ(normed).reshaped(batch, seqLen, 1, channels).transposed(0, 2, 1, 3)
        var k = toK(normed).reshaped(batch, seqLen, 1, channels).transposed(0, 2, 1, 3)
        var v = toV(normed).reshaped(batch, seqLen, 1, channels).transposed(0, 2, 1, 3)

        let scale: Float = 1.0 / Foundation.sqrt(Float(channels))
        var attended = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: nil
        )
        attended = attended.transposed(0, 2, 1, 3).reshaped(batch, height, width, channels)
        attended = toOut(attended)

        let result = nhwc + attended
        return result.transposed(0, 3, 1, 2)
    }
}
