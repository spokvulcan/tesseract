import CoreGraphics
import Foundation
import MLX
import MLXNN
import MLXRandom
import os
import Tokenizers
import HuggingFace

let flux2Profiling = ProcessInfo.processInfo.arguments.contains("--flux2-profile")
private let logger = Logger(subsystem: "app.tesseract.agent", category: "image")

/// Configuration for glyph injection during denoising.
public struct GlyphInjectionConfig: Sendable {
    /// Text strings to rasterize and inject as glyph structure.
    public let glyphTexts: [String]
    /// Which denoising steps to inject at (0-indexed).
    public let injectAtSteps: [Int]
    /// Base injection strength (0..1). Cosine-annealed across injection steps.
    public let strength: Float

    public init(glyphTexts: [String], injectAtSteps: [Int] = [1, 2], strength: Float = 0.5) {
        self.glyphTexts = glyphTexts
        self.injectAtSteps = injectAtSteps
        self.strength = strength
    }

    /// Convenience for single text.
    public init(glyphText: String, injectAtSteps: [Int] = [1, 2], strength: Float = 0.5) {
        self.glyphTexts = [glyphText]
        self.injectAtSteps = injectAtSteps
        self.strength = strength
    }
}

public actor Flux2Pipeline {
    private let transformer: Flux2Transformer
    private let textEncoder: Qwen3TextEncoder
    private let vae: Flux2VAE
    private let tokenizer: Tokenizer
    private let config: Flux2Configuration.Pipeline
    private let modelDirectory: URL

    public init(modelDirectory: URL, config: Flux2Configuration.Pipeline = .klein4B) async throws {
        self.config = config
        self.modelDirectory = modelDirectory
        self.transformer = Flux2Transformer(config: config.transformer)
        self.textEncoder = Qwen3TextEncoder(config: config.textEncoder)
        self.vae = Flux2VAE(config: config.vae)

        // Load tokenizer from tokenizer/ subdirectory (HuggingFace diffusers layout)
        let tokenizerDir = modelDirectory.appendingPathComponent("tokenizer")
        logger.info("Loading tokenizer from: \(tokenizerDir.path, privacy: .public)")
        self.tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerDir)

        // Load weights
        logger.info("Loading transformer weights...")
        try Flux2WeightLoader.loadTransformerWeights(from: modelDirectory, into: transformer)

        logger.info("Loading text encoder weights...")
        try Flux2WeightLoader.loadTextEncoderWeights(from: modelDirectory, into: textEncoder)

        logger.info("Loading VAE weights...")
        try Flux2WeightLoader.loadVAEWeights(from: modelDirectory, into: vae)

        logger.info("All weights loaded successfully")
    }

    public func generateImage(
        prompt: String,
        width: Int = 1024,
        height: Int = 1024,
        numSteps: Int = 4,
        zeroInitSteps: Int = 1,
        glyphInjection: GlyphInjectionConfig? = nil,
        seed: UInt64 = 0,
        onProgress: (@Sendable (Int, Int) -> Void)? = nil
    ) async throws -> CGImage {
        let actualSeed = seed == 0 ? UInt64.random(in: 1...UInt64.max) : seed
        logger.info("Generating \(width)x\(height) image, \(numSteps) steps, seed=\(actualSeed), glyph=\(glyphInjection != nil), profiling=\(flux2Profiling)")

        // Cap MLX buffer cache to prevent unbounded memory growth
        Memory.cacheLimit = 256 * 1024 * 1024  // 256 MB

        let genStart = CFAbsoluteTimeGetCurrent()

        // 1. Encode prompt
        let textStart = CFAbsoluteTimeGetCurrent()
        let (promptEmbeds, textIds) = try Flux2PromptEncoder.encodePrompt(
            prompt: prompt,
            tokenizer: tokenizer,
            textEncoder: textEncoder,
            maxSequenceLength: config.maxSequenceLength,
            hiddenStateLayers: config.textEncoder.hiddenStateLayers
        )
        eval(promptEmbeds, textIds)
        Memory.clearCache()
        if flux2Profiling {
            let textTime = CFAbsoluteTimeGetCurrent() - textStart
            logger.info("[PROFILE] Text encoding: \(String(format: "%.3f", textTime))s")
        }

        // 2. Create latents
        let (packedLatents, latentIds, latentHeight, latentWidth) = Flux2LatentCreator.preparePackedLatents(
            seed: actualSeed,
            height: height,
            width: width,
            batchSize: 1,
            numLatentChannels: config.vae.latentChannels,
            vaeScaleFactor: config.vae.scaleFactor
        )
        eval(packedLatents, latentIds)

        // 3. Create scheduler
        let imageSeqLen = latentHeight * latentWidth
        let scheduler = FlowMatchEulerScheduler(numInferenceSteps: numSteps, imageSeqLen: imageSeqLen)

        // 3.5. Precompute rotary embeddings (identical across all denoising steps)
        let rotaryEmb = transformer.computeRotaryEmb(imgIds: latentIds, txtIds: textIds)
        eval(rotaryEmb.cos, rotaryEmb.sin)

        // 3.6. Prepare glyph latent if injection is enabled
        var glyphLatentPacked: MLXArray? = nil
        var glyphNoise: MLXArray? = nil
        let injectStepSet: Set<Int>
        let glyphStrength: Float

        if let injection = glyphInjection {
            let glyphStart = CFAbsoluteTimeGetCurrent()
            logger.info("Preparing glyph injection for \(injection.glyphTexts.count) text(s)")

            // Rasterize text(s) → [1, 3, H, W] in [-1, 1]
            let glyphImage = GlyphRasterizer.rasterize(
                texts: injection.glyphTexts,
                width: width, height: height
            )
            eval(glyphImage)

            // Encode through VAE → [1, 128, latH, latW]
            var glyphEncoded = try encodeImage(glyphImage)
            eval(glyphEncoded)

            // Apply Log-Gabor filter to isolate glyph edges
            glyphEncoded = LogGaborFilter.apply(glyphEncoded)
            eval(glyphEncoded)

            // Pack to match denoising latent format: [1, 128, latH, latW] → [1, seqLen, 128]
            glyphLatentPacked = Flux2LatentCreator.packLatents(glyphEncoded)
            eval(glyphLatentPacked!)

            // Pre-generate noise for sigma blending (same shape as packed latents)
            glyphNoise = MLXRandom.normal(
                glyphLatentPacked!.shape,
                key: MLXRandom.key(actualSeed &+ 1)
            ).asType(.bfloat16)
            eval(glyphNoise!)

            Memory.clearCache()
            injectStepSet = Set(injection.injectAtSteps)
            glyphStrength = injection.strength

            if flux2Profiling {
                let glyphTime = CFAbsoluteTimeGetCurrent() - glyphStart
                logger.info("[PROFILE] Glyph preparation: \(String(format: "%.3f", glyphTime))s")
            }
        } else {
            injectStepSet = []
            glyphStrength = 0
        }

        // 4. Denoising loop
        var latents = packedLatents
        let denoiseStart = CFAbsoluteTimeGetCurrent()

        for i in 0..<numSteps {
            let stepStart = CFAbsoluteTimeGetCurrent()

            // CFG-Zero*: zero velocity at early steps (pure noise → skip transformer)
            if i < zeroInitSteps {
                // Zero velocity = latents unchanged, skip transformer forward pass
                onProgress?(i + 1, numSteps)
                if flux2Profiling {
                    logger.info("[PROFILE] Step \(i + 1)/\(numSteps): zero-init (skipped)")
                } else {
                    logger.info("Step \(i + 1)/\(numSteps) zero-init (skipped)")
                }
                continue
            }

            let timestep = scheduler.timesteps[i]

            let noisePred = transformer(
                hiddenStates: latents,
                encoderHiddenStates: promptEmbeds,
                timestep: timestep,
                imgIds: latentIds,
                txtIds: textIds,
                guidance: nil,
                precomputedRotaryEmb: rotaryEmb
            )
            eval(noisePred)

            latents = scheduler.step(noise: noisePred, timestepIndex: i, latents: latents)
            eval(latents)

            // Glyph injection: blend filtered glyph structure into latents
            if injectStepSet.contains(i), let glyphPacked = glyphLatentPacked, let noise = glyphNoise {
                let sigma = scheduler.sigmas[i + 1]  // sigma AFTER this step
                let sigmaFloat: Float = sigma.item()

                // Noise-align glyph to current noise level: (1-σ)·glyph + σ·noise
                let glyphAligned = (1.0 - sigmaFloat) * glyphPacked + sigmaFloat * noise

                // Unpack to spatial for Log-Gabor filtering
                var glyphSpatial = Flux2LatentCreator.unpackLatents(
                    glyphAligned, height: height, width: width,
                    vaeScaleFactor: config.vae.scaleFactor
                )
                glyphSpatial = LogGaborFilter.apply(glyphSpatial)
                let glyphRepacked = Flux2LatentCreator.packLatents(glyphSpatial)

                // Cosine annealing: λ decays across injection steps
                let injectionIndex = injectStepSet.sorted().firstIndex(of: i)!
                let totalInjections = injectStepSet.count
                let progress = Float(injectionIndex) / Float(max(totalInjections - 1, 1))
                let lambda = glyphStrength * 0.5 * (1.0 + cos(Float.pi * progress))

                // Blend: z̃ = (1-λ)·z + λ·z_filtered
                latents = (1.0 - lambda) * latents + lambda * glyphRepacked
                eval(latents)

                if flux2Profiling {
                    logger.info("[PROFILE] Glyph injected at step \(i), λ=\(String(format: "%.3f", lambda)), σ=\(String(format: "%.3f", sigmaFloat))")
                }
            }

            Memory.clearCache()
            onProgress?(i + 1, numSteps)
            let stepTime = CFAbsoluteTimeGetCurrent() - stepStart
            let activeMB = Float(Memory.activeMemory) / 1e6
            if flux2Profiling {
                logger.info("[PROFILE] Step \(i + 1)/\(numSteps): \(String(format: "%.3f", stepTime))s, active=\(String(format: "%.1f", activeMB))MB")
            } else {
                logger.info("Step \(i + 1)/\(numSteps) complete, active=\(String(format: "%.1f", activeMB))MB")
            }
        }

        if flux2Profiling {
            let denoiseTime = CFAbsoluteTimeGetCurrent() - denoiseStart
            logger.info("[PROFILE] Denoising total: \(String(format: "%.3f", denoiseTime))s (\(numSteps) steps)")
        }
        Memory.clearCache()

        // 5. VAE decode (BN denorm → unpatchify → post_quant_conv → decoder)
        let vaeStart = CFAbsoluteTimeGetCurrent()
        let unpackedLatents = Flux2LatentCreator.unpackLatents(latents, height: height, width: width, vaeScaleFactor: config.vae.scaleFactor)

        var decoded = vae.decodePacked(unpackedLatents)
        eval(decoded)
        Memory.clearCache()
        if flux2Profiling {
            let vaeTime = CFAbsoluteTimeGetCurrent() - vaeStart
            logger.info("[PROFILE] VAE decode: \(String(format: "%.3f", vaeTime))s")
        }

        // 6. Convert to image
        // VAE outputs [-1, 1] — denormalize to [0, 1]
        decoded = MLX.clip(decoded / 2 + 0.5, min: 0, max: 1)
        let image = try arrayToCGImage(decoded)
        Memory.clearCache()
        let totalTime = CFAbsoluteTimeGetCurrent() - genStart
        let peakMB = Float(Memory.peakMemory) / 1e6
        if flux2Profiling {
            logger.info("[PROFILE] Total generation: \(String(format: "%.3f", totalTime))s, peak=\(String(format: "%.1f", peakMB))MB")
        } else {
            logger.info("Generation complete in \(String(format: "%.1f", totalTime))s, peak=\(String(format: "%.1f", peakMB))MB")
        }
        return image
    }

    private var encoderContainer: Flux2VAEEncoderContainer?

    /// Lazily initialize and load VAE encoder weights (~200MB)
    func ensureEncoderLoaded() throws {
        guard encoderContainer == nil else { return }
        logger.info("Loading VAE encoder weights (lazy)...")
        let container = Flux2VAEEncoderContainer(config: config.vae)
        try Flux2WeightLoader.loadVAEEncoderWeights(from: modelDirectory, into: container)
        encoderContainer = container
        logger.info("VAE encoder loaded")
    }

    /// Encode image to packed+normalized latents [B, 128, H/16, W/16]
    func encodeImage(_ image: MLXArray) throws -> MLXArray {
        try ensureEncoderLoaded()
        return encoderContainer!.encode(image, bn: vae.bn)
    }

    private func arrayToCGImage(_ array: MLXArray) -> CGImage {
        // Input: [1, 3, H, W] -> [H, W, 3] as UInt8
        let rgb = array[0].transposed(1, 2, 0) // [H, W, 3]
        let uint8 = (rgb.asType(.float32) * 255).asType(.uint8)
        eval(uint8)

        let height = uint8.dim(0)
        let width = uint8.dim(1)

        // Get raw bytes
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
