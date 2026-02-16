import MLX
import MLXNN

final class DownEncoderBlock2D: Module {
    @ModuleInfo var resnets: [VAEResnetBlock2D]
    @ModuleInfo var downsamplers: [Downsample2D]

    init(inChannels: Int, outChannels: Int, numLayers: Int = 2, eps: Float = 1e-6, groups: Int = 32, addDownsample: Bool = true) {
        self._resnets.wrappedValue = (0..<numLayers).map { i in
            VAEResnetBlock2D(
                inChannels: i == 0 ? inChannels : outChannels,
                outChannels: outChannels,
                eps: eps,
                groups: groups
            )
        }
        if addDownsample {
            self._downsamplers.wrappedValue = [Downsample2D(channels: outChannels)]
        } else {
            self._downsamplers.wrappedValue = []
        }
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        var hs = hiddenStates
        for resnet in resnets {
            hs = resnet(hs)
        }
        for downsampler in downsamplers {
            hs = downsampler(hs)
        }
        return hs
    }
}

final class Flux2VAEEncoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "mid_block") var midBlock: UNetMidBlock2D
    @ModuleInfo(key: "down_blocks") var downBlocks: [DownEncoderBlock2D]
    @ModuleInfo(key: "conv_norm_out") var convNormOut: GroupNorm
    @ModuleInfo(key: "conv_out") var convOut: Conv2d

    init(config: Flux2Configuration.VAE) {
        // conv_in: 3 → first block_out_channel (128)
        let firstChannel = config.blockOutChannels.first!
        self._convIn.wrappedValue = Conv2d(
            inputChannels: config.outChannels, outputChannels: firstChannel,
            kernelSize: IntOrPair(3), stride: IntOrPair(1), padding: IntOrPair(1)
        )

        // Down blocks: forward channels
        var blocks = [DownEncoderBlock2D]()
        for (i, outChannel) in config.blockOutChannels.enumerated() {
            let inChannel = i == 0 ? firstChannel : config.blockOutChannels[i - 1]
            let isFinalBlock = i == config.blockOutChannels.count - 1
            blocks.append(DownEncoderBlock2D(
                inChannels: inChannel,
                outChannels: outChannel,
                numLayers: config.layersPerBlock,
                eps: 1e-6,
                groups: config.normNumGroups,
                addDownsample: !isFinalBlock
            ))
        }
        self._downBlocks.wrappedValue = blocks

        let lastChannel = config.blockOutChannels.last!
        self._midBlock.wrappedValue = UNetMidBlock2D(
            channels: lastChannel, eps: 1e-6, groups: config.normNumGroups, addAttention: true
        )

        self._convNormOut.wrappedValue = GroupNorm(
            groupCount: config.normNumGroups, dimensions: lastChannel, eps: 1e-6, pytorchCompatible: true
        )
        // conv_out: 512 → 64 (2 × latentChannels for mean + logvar)
        self._convOut.wrappedValue = Conv2d(
            inputChannels: lastChannel, outputChannels: config.latentChannels * 2,
            kernelSize: IntOrPair(3), stride: IntOrPair(1), padding: IntOrPair(1)
        )
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        // Input: NCHW -> NHWC for conv_in
        var hs = hiddenStates.transposed(0, 2, 3, 1)
        hs = convIn(hs)
        hs = hs.transposed(0, 3, 1, 2) // back to NCHW for down_blocks

        for downBlock in downBlocks {
            hs = downBlock(hs)
        }

        hs = midBlock(hs)

        // conv_norm_out operates in NHWC
        hs = hs.transposed(0, 2, 3, 1)
        hs = convNormOut(hs.asType(.float32)).asType(.bfloat16)
        hs = silu(hs)
        hs = convOut(hs)
        return hs.transposed(0, 3, 1, 2) // back to NCHW: [B, 64, H/8, W/8]
    }
}
