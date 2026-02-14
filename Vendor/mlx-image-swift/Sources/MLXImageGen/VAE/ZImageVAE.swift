import MLX
import MLXNN

/// Z-Image VAE: simple decoder wrapper with scaling/shifting.
/// No batch norm stats, no post_quant_conv, no unpatchify — just scale + shift + decode.
/// Reuses Flux2VAEDecoder since the decoder architecture is identical (just different inChannels).
final class ZImageVAE: Module {
    @ModuleInfo var decoder: Flux2VAEDecoder
    let scalingFactor: Float
    let shiftFactor: Float

    init(config: ZImageConfiguration.VAE) {
        self.scalingFactor = config.scalingFactor
        self.shiftFactor = config.shiftFactor

        // Build decoder with Z-Image's channel configuration
        let vaeConfig = Flux2Configuration.VAE(
            inChannels: config.latentChannels,  // 16 (vs FLUX's 32)
            outChannels: 3,
            latentChannels: config.latentChannels,
            blockOutChannels: [128, 256, 512, 512],
            layersPerBlock: 2,  // +1 in UpDecoderBlock2D init = 3 resnets per block
            normNumGroups: 32,
            scaleFactor: config.spatialScale
        )
        self._decoder.wrappedValue = Flux2VAEDecoder(config: vaeConfig)
    }

    /// Decode latents [B, 16, H/8, W/8] → image [B, 3, H, W]
    func decode(_ latents: MLXArray) -> MLXArray {
        // Scale and shift: (latents / scaling_factor) + shift_factor
        let scaled = latents / MLXArray(scalingFactor) + MLXArray(shiftFactor)
        return decoder(scaled)
    }
}
