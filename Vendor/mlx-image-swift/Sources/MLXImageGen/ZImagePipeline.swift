import CoreGraphics
import Foundation
import MLX
import MLXNN
import MLXRandom
import os
import Tokenizers
import HuggingFace

private let logger = Logger(subsystem: "app.tesseract.agent", category: "image")

public actor ZImagePipeline {
    private let transformer: ZImageTransformer
    private let textEncoder: Qwen3TextEncoder
    private let vae: ZImageVAE
    private let tokenizer: Tokenizer
    private let config: ZImageConfiguration.Pipeline

    public init(modelDirectory: URL, config: ZImageConfiguration.Pipeline = .base) async throws {
        self.config = config
        self.transformer = ZImageTransformer(config: config.transformer)

        // Reuse Qwen3TextEncoder with same config as FLUX (same architecture)
        let textEncoderConfig = Flux2Configuration.TextEncoder.klein4B
        self.textEncoder = Qwen3TextEncoder(config: textEncoderConfig)

        self.vae = ZImageVAE(config: config.vae)

        // Load tokenizer
        let tokenizerDir = modelDirectory.appendingPathComponent("tokenizer")
        logger.info("Loading Z-Image tokenizer from: \(tokenizerDir.path, privacy: .public)")
        self.tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerDir)

        // Load weights
        logger.info("Loading Z-Image transformer weights...")
        try ZImageWeightLoader.loadTransformerWeights(from: modelDirectory, into: transformer)

        logger.info("Loading Z-Image text encoder weights...")
        try ZImageWeightLoader.loadTextEncoderWeights(from: modelDirectory, into: textEncoder)

        logger.info("Loading Z-Image VAE weights...")
        try ZImageWeightLoader.loadVAEWeights(from: modelDirectory, into: vae)

        logger.info("All Z-Image weights loaded successfully")
    }

    public func generateImage(
        prompt: String,
        negativePrompt: String? = nil,
        width: Int = 1024,
        height: Int = 1024,
        numSteps: Int = 50,
        guidance: Float = 4.0,
        seed: UInt64 = 0,
        onProgress: (@Sendable (Int, Int) -> Void)? = nil
    ) async throws -> CGImage {
        let actualSeed = seed == 0 ? UInt64.random(in: 1...UInt64.max) : seed
        logger.info("Z-Image generating \(width, privacy: .public)x\(height, privacy: .public), \(numSteps, privacy: .public) steps, guidance=\(guidance, privacy: .public), seed=\(actualSeed, privacy: .public)")

        // Cap MLX buffer cache
        Memory.cacheLimit = 256 * 1024 * 1024

        let genStart = CFAbsoluteTimeGetCurrent()

        // 1. Encode prompt → [numValidTokens, 2560]
        let textStart = CFAbsoluteTimeGetCurrent()
        let capFeats = try ZImagePromptEncoder.encodePrompt(
            prompt: prompt,
            tokenizer: tokenizer,
            textEncoder: textEncoder,
            maxSequenceLength: config.maxSequenceLength
        )
        eval(capFeats)
        Memory.clearCache()
        let textTime = CFAbsoluteTimeGetCurrent() - textStart
        logger.info("Z-Image text encoding: \(String(format: "%.3f", textTime), privacy: .public)s, tokens=\(capFeats.dim(0), privacy: .public)")

        // 2. Encode negative prompt for CFG (if guidance > 1.0)
        var negativeCapFeats: MLXArray? = nil
        if guidance > 1.0 {
            let negText = (negativePrompt?.trimmingCharacters(in: .whitespaces).isEmpty == false)
                ? negativePrompt! : " "
            negativeCapFeats = try ZImagePromptEncoder.encodePrompt(
                prompt: negText,
                tokenizer: tokenizer,
                textEncoder: textEncoder,
                maxSequenceLength: config.maxSequenceLength
            )
            eval(negativeCapFeats!)
            Memory.clearCache()
        }

        // 3. Create noise latents [16, 1, H/8, W/8]
        var latents = ZImageLatentCreator.createNoise(
            seed: actualSeed,
            height: height,
            width: width,
            spatialScale: config.vae.spatialScale
        )
        eval(latents)

        // 4. Create scheduler
        // imageSeqLen = (H/16) * (W/16) after patchification
        let imageSeqLen = (height / 16) * (width / 16)
        let scheduler = FlowMatchEulerScheduler(numInferenceSteps: numSteps, imageSeqLen: imageSeqLen)

        // 5. Denoising loop
        let denoiseStart = CFAbsoluteTimeGetCurrent()

        for i in 0..<numSteps {
            let stepStart = CFAbsoluteTimeGetCurrent()

            // Compute timestep: 1 - sigma_t
            let sigmaT = scheduler.sigmas[i]
            let timestep = 1.0 - sigmaT

            // Forward pass with positive prompt
            let noise = transformer(
                x: latents,
                timestep: timestep,
                sigmas: scheduler.sigmas,
                capFeats: capFeats
            )
            eval(noise)

            // CFG: combine with negative prompt prediction
            var combinedNoise = noise
            if let negFeats = negativeCapFeats {
                let negNoise = transformer(
                    x: latents,
                    timestep: timestep,
                    sigmas: scheduler.sigmas,
                    capFeats: negFeats
                )
                eval(negNoise)
                combinedNoise = noise + MLXArray(guidance) * (noise - negNoise)
            }

            // Scheduler step
            latents = scheduler.step(noise: combinedNoise, timestepIndex: i, latents: latents)
            eval(latents)

            Memory.clearCache()
            onProgress?(i + 1, numSteps)
            let stepTime = CFAbsoluteTimeGetCurrent() - stepStart
            let activeMB = Float(Memory.activeMemory) / 1e6
            logger.info("Z-Image step \(i + 1, privacy: .public)/\(numSteps, privacy: .public): \(String(format: "%.3f", stepTime), privacy: .public)s, active=\(String(format: "%.1f", activeMB), privacy: .public)MB")
        }

        let denoiseTime = CFAbsoluteTimeGetCurrent() - denoiseStart
        logger.info("Z-Image denoising total: \(String(format: "%.3f", denoiseTime), privacy: .public)s (\(numSteps, privacy: .public) steps)")

        // 6. Unpack latents: [16, 1, H/8, W/8] → [1, 16, H/8, W/8]
        let unpackedLatents = ZImageLatentCreator.unpackLatents(latents)

        // 7. VAE decode
        let vaeStart = CFAbsoluteTimeGetCurrent()
        var decoded = vae.decode(unpackedLatents)
        eval(decoded)
        Memory.clearCache()
        let vaeTime = CFAbsoluteTimeGetCurrent() - vaeStart
        logger.info("Z-Image VAE decode: \(String(format: "%.3f", vaeTime), privacy: .public)s")

        // 8. Convert to image: [-1, 1] → [0, 1]
        decoded = MLX.clip(decoded / 2 + 0.5, min: 0, max: 1)
        let image = arrayToCGImage(decoded)
        Memory.clearCache()

        let totalTime = CFAbsoluteTimeGetCurrent() - genStart
        let peakMB = Float(Memory.peakMemory) / 1e6
        logger.info("Z-Image generation complete in \(String(format: "%.1f", totalTime), privacy: .public)s, peak=\(String(format: "%.1f", peakMB), privacy: .public)MB")

        return image
    }

    private func arrayToCGImage(_ array: MLXArray) -> CGImage {
        // Input: [1, 3, H, W] → [H, W, 3] as UInt8
        let rgb = array[0].transposed(1, 2, 0)
        let uint8 = (rgb.asType(.float32) * 255).asType(.uint8)
        eval(uint8)

        let height = uint8.dim(0)
        let width = uint8.dim(1)

        let data = uint8.asData(noCopy: false)
        let provider = CGDataProvider(data: data as CFData)!
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)

        return CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 24,
            bytesPerRow: width * 3,
            space: colorSpace,
            bitmapInfo: bitmapInfo,
            provider: provider,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        )!
    }
}
