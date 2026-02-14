import MLX
import MLXFast
import MLXNN

/// Conv2d that transposes NCHW <-> NHWC around MLX's native NHWC conv
final class NCHWConv2d: Module {
    @ModuleInfo var conv: Conv2d

    init(inChannels: Int, outChannels: Int, kernelSize: Int = 3, stride: Int = 1, padding: Int = 1) {
        self._conv.wrappedValue = Conv2d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: IntOrPair(kernelSize),
            stride: IntOrPair(stride),
            padding: IntOrPair(padding)
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Input: NCHW -> NHWC for conv -> back to NCHW
        let nhwc = x.transposed(0, 2, 3, 1)
        let out = conv(nhwc)
        return out.transposed(0, 3, 1, 2)
    }
}

/// GroupNorm that operates on NCHW tensors (transposes internally)
final class NCHWGroupNorm: Module {
    @ModuleInfo(key: "group_norm") var groupNorm: GroupNorm

    init(numGroups: Int, dims: Int, eps: Float = 1e-6) {
        self._groupNorm.wrappedValue = GroupNorm(groupCount: numGroups, dimensions: dims, eps: eps, pytorchCompatible: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let nhwc = x.transposed(0, 2, 3, 1)
        let out = groupNorm(nhwc.asType(.float32)).asType(.bfloat16)
        return out.transposed(0, 3, 1, 2)
    }
}

/// Stride-2 conv downsample with asymmetric padding (NCHW)
final class Downsample2D: Module {
    @ModuleInfo var conv: Conv2d

    init(channels: Int, outChannels: Int? = nil) {
        let oc = outChannels ?? channels
        // Conv with stride=2, no built-in padding (we pad manually for asymmetric)
        self._conv.wrappedValue = Conv2d(
            inputChannels: channels,
            outputChannels: oc,
            kernelSize: IntOrPair(3),
            stride: IntOrPair(2),
            padding: IntOrPair(0)
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Asymmetric padding: pad right+1, bottom+1 (matches diffusers Downsample2D)
        // Input is NCHW → pad H (axis 2) and W (axis 3) on the high side only
        let padded = MLX.padded(x, widths: [IntOrPair(0), IntOrPair(0), IntOrPair((0, 1)), IntOrPair((0, 1))])
        // NCHW -> NHWC for conv
        let nhwc = padded.transposed(0, 2, 3, 1)
        let out = conv(nhwc)
        return out.transposed(0, 3, 1, 2)
    }
}

/// Nearest-neighbor 2x upsample + Conv (NCHW)
final class Upsample2D: Module {
    @ModuleInfo var conv: Conv2d

    init(channels: Int, outChannels: Int? = nil) {
        let oc = outChannels ?? channels
        self._conv.wrappedValue = Conv2d(
            inputChannels: channels,
            outputChannels: oc,
            kernelSize: IntOrPair(3),
            stride: IntOrPair(1),
            padding: IntOrPair(1)
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Nearest-neighbor 2x upsample on NCHW: repeat along H and W axes
        var hs = MLX.repeated(x, count: 2, axis: 2)
        hs = MLX.repeated(hs, count: 2, axis: 3)
        // NCHW -> NHWC for conv
        hs = hs.transposed(0, 2, 3, 1)
        hs = conv(hs)
        return hs.transposed(0, 3, 1, 2)
    }
}
