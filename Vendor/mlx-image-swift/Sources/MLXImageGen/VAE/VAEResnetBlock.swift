import MLX
import MLXNN

final class VAEResnetBlock2D: Module {
    @ModuleInfo var norm1: GroupNorm
    @ModuleInfo var conv1: Conv2d
    @ModuleInfo var norm2: GroupNorm
    @ModuleInfo var conv2: Conv2d
    @ModuleInfo(key: "conv_shortcut") var convShortcut: Conv2d?

    init(inChannels: Int, outChannels: Int, eps: Float = 1e-6, groups: Int = 32) {
        self._norm1.wrappedValue = GroupNorm(groupCount: groups, dimensions: inChannels, eps: eps, pytorchCompatible: true)
        self._conv1.wrappedValue = Conv2d(
            inputChannels: inChannels, outputChannels: outChannels,
            kernelSize: IntOrPair(3), stride: IntOrPair(1), padding: IntOrPair(1)
        )
        self._norm2.wrappedValue = GroupNorm(groupCount: groups, dimensions: outChannels, eps: eps, pytorchCompatible: true)
        self._conv2.wrappedValue = Conv2d(
            inputChannels: outChannels, outputChannels: outChannels,
            kernelSize: IntOrPair(3), stride: IntOrPair(1), padding: IntOrPair(1)
        )
        if inChannels != outChannels {
            self._convShortcut.wrappedValue = Conv2d(
                inputChannels: inChannels, outputChannels: outChannels,
                kernelSize: IntOrPair(1), stride: IntOrPair(1), padding: IntOrPair(0)
            )
        }
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        // Input is NCHW, transpose to NHWC for GroupNorm and Conv2d
        var residual = hiddenStates.transposed(0, 2, 3, 1)

        var hs = hiddenStates.transposed(0, 2, 3, 1)
        hs = norm1(hs.asType(.float32)).asType(.bfloat16)
        hs = silu(hs)
        hs = conv1(hs)
        hs = norm2(hs.asType(.float32)).asType(.bfloat16)
        hs = silu(hs)
        hs = conv2(hs)

        if let convShortcut {
            residual = convShortcut(residual)
        }

        hs = hs + residual
        return hs.transposed(0, 3, 1, 2)
    }
}
