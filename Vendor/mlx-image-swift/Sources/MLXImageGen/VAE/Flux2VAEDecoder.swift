import MLX
import MLXNN

final class UNetMidBlock2D: Module {
    @ModuleInfo var resnets: [VAEResnetBlock2D]
    @ModuleInfo var attentions: [VAEAttentionBlock]

    init(channels: Int, eps: Float = 1e-6, groups: Int = 32, addAttention: Bool = true) {
        self._resnets.wrappedValue = [
            VAEResnetBlock2D(inChannels: channels, outChannels: channels, eps: eps, groups: groups),
            VAEResnetBlock2D(inChannels: channels, outChannels: channels, eps: eps, groups: groups),
        ]
        if addAttention {
            self._attentions.wrappedValue = [VAEAttentionBlock(channels: channels, groups: groups, eps: eps)]
        } else {
            self._attentions.wrappedValue = []
        }
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        var hs = resnets[0](hiddenStates)
        if !attentions.isEmpty {
            hs = attentions[0](hs)
        }
        hs = resnets[1](hs)
        return hs
    }
}

final class UpDecoderBlock2D: Module {
    @ModuleInfo var resnets: [VAEResnetBlock2D]
    @ModuleInfo var upsamplers: [Upsample2D]

    init(inChannels: Int, outChannels: Int, numLayers: Int = 3, eps: Float = 1e-6, groups: Int = 32, addUpsample: Bool = true) {
        self._resnets.wrappedValue = (0..<numLayers).map { i in
            VAEResnetBlock2D(
                inChannels: i == 0 ? inChannels : outChannels,
                outChannels: outChannels,
                eps: eps,
                groups: groups
            )
        }
        if addUpsample {
            self._upsamplers.wrappedValue = [Upsample2D(channels: outChannels, outChannels: outChannels)]
        } else {
            self._upsamplers.wrappedValue = []
        }
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        var hs = hiddenStates
        for resnet in resnets {
            hs = resnet(hs)
        }
        for upsampler in upsamplers {
            hs = upsampler(hs)
        }
        return hs
    }
}

final class Flux2VAEDecoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "mid_block") var midBlock: UNetMidBlock2D
    @ModuleInfo(key: "up_blocks") var upBlocks: [UpDecoderBlock2D]
    @ModuleInfo(key: "conv_norm_out") var convNormOut: GroupNorm
    @ModuleInfo(key: "conv_out") var convOut: Conv2d

    init(config: Flux2Configuration.VAE) {
        // conv_in: latent_channels -> last block_out_channel
        let lastChannel = config.blockOutChannels.last!
        self._convIn.wrappedValue = Conv2d(
            inputChannels: config.inChannels, outputChannels: lastChannel,
            kernelSize: IntOrPair(3), stride: IntOrPair(1), padding: IntOrPair(1)
        )

        self._midBlock.wrappedValue = UNetMidBlock2D(
            channels: lastChannel, eps: 1e-6, groups: config.normNumGroups, addAttention: true
        )

        // Up blocks: reversed channels
        let reversedChannels = config.blockOutChannels.reversed().map { $0 }
        var blocks = [UpDecoderBlock2D]()
        for (i, outChannel) in reversedChannels.enumerated() {
            let prevChannel = i == 0 ? outChannel : reversedChannels[i - 1]
            let isFinalBlock = i == reversedChannels.count - 1
            blocks.append(UpDecoderBlock2D(
                inChannels: prevChannel,
                outChannels: outChannel,
                numLayers: config.layersPerBlock + 1,
                eps: 1e-6,
                groups: config.normNumGroups,
                addUpsample: !isFinalBlock
            ))
        }
        self._upBlocks.wrappedValue = blocks

        let firstChannel = config.blockOutChannels.first!
        self._convNormOut.wrappedValue = GroupNorm(
            groupCount: config.normNumGroups, dimensions: firstChannel, eps: 1e-6, pytorchCompatible: true
        )
        self._convOut.wrappedValue = Conv2d(
            inputChannels: firstChannel, outputChannels: config.outChannels,
            kernelSize: IntOrPair(3), stride: IntOrPair(1), padding: IntOrPair(1)
        )
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        // Input: NCHW -> NHWC for conv_in
        var hs = hiddenStates.transposed(0, 2, 3, 1)
        hs = convIn(hs)
        hs = hs.transposed(0, 3, 1, 2) // back to NCHW for mid_block

        hs = midBlock(hs)

        for upBlock in upBlocks {
            hs = upBlock(hs)
        }

        // conv_norm_out operates in NHWC
        hs = hs.transposed(0, 2, 3, 1)
        hs = convNormOut(hs.asType(.float32)).asType(.bfloat16)
        hs = silu(hs)
        hs = convOut(hs)
        return hs.transposed(0, 3, 1, 2) // back to NCHW
    }
}

// MARK: - Batch norm statistics (learned running_mean/running_var)

final class Flux2BatchNormStats: Module {
    // snake_case names match safetensors keys: bn.running_mean, bn.running_var
    var running_mean: MLXArray
    var running_var: MLXArray
    let eps: Float

    init(numFeatures: Int, eps: Float = 1e-4) {
        self.running_mean = MLXArray.zeros([numFeatures])
        self.running_var = MLXArray.ones([numFeatures])
        self.eps = eps
    }
}

// MARK: - Full VAE (batch norm denorm + unpatchify + post_quant_conv + decoder)

final class Flux2VAE: Module {
    @ModuleInfo var decoder: Flux2VAEDecoder
    @ModuleInfo(key: "post_quant_conv") var postQuantConv: Conv2d
    @ModuleInfo var bn: Flux2BatchNormStats

    init(config: Flux2Configuration.VAE) {
        self._decoder.wrappedValue = Flux2VAEDecoder(config: config)
        self._postQuantConv.wrappedValue = Conv2d(
            inputChannels: config.latentChannels,
            outputChannels: config.latentChannels,
            kernelSize: IntOrPair(1), stride: IntOrPair(1), padding: IntOrPair(0)
        )
        self._bn.wrappedValue = Flux2BatchNormStats(
            numFeatures: 4 * config.latentChannels, eps: 1e-4
        )
    }

    /// Decode packed latents [B, 128, latH, latW] → image [B, 3, H, W]
    func decodePacked(_ packedLatents: MLXArray) -> MLXArray {
        // 1. BN denormalize (reverse batch norm using learned running stats)
        let bnMean = bn.running_mean.reshaped(1, -1, 1, 1)
        let bnStd = MLX.sqrt(bn.running_var.reshaped(1, -1, 1, 1) + MLXArray(bn.eps))
        var latents = packedLatents * bnStd + bnMean

        // 2. Unpatchify: [B, 128, latH, latW] → [B, 32, latH*2, latW*2]
        latents = Flux2LatentCreator.unpatchifyLatents(latents)

        // 3. post_quant_conv: Conv2d(32→32, 1×1) — needs NHWC for MLX conv
        latents = latents.transposed(0, 2, 3, 1)
        latents = postQuantConv(latents)
        latents = latents.transposed(0, 3, 1, 2)

        // 4. Decode
        return decoder(latents)
    }
}
