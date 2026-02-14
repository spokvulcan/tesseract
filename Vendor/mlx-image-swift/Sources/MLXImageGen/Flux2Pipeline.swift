import CoreGraphics
import Foundation
import MLX
import MLXNN
import MLXRandom
import Tokenizers
import HuggingFace

public actor Flux2Pipeline {
    private let transformer: Flux2Transformer
    private let textEncoder: Qwen3TextEncoder
    private let vae: Flux2VAE
    private let tokenizer: Tokenizer
    private let config: Flux2Configuration.Pipeline

    public init(modelDirectory: URL, config: Flux2Configuration.Pipeline = .klein4B) async throws {
        self.config = config
        self.transformer = Flux2Transformer(config: config.transformer)
        self.textEncoder = Qwen3TextEncoder(config: config.textEncoder)
        self.vae = Flux2VAE(config: config.vae)

        // Load tokenizer from tokenizer/ subdirectory (HuggingFace diffusers layout)
        let tokenizerDir = modelDirectory.appendingPathComponent("tokenizer")
        NSLog("[MLXImageGen] Loading tokenizer from: %@", tokenizerDir.path)
        self.tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerDir)

        // Load weights
        NSLog("[MLXImageGen] Loading transformer weights...")
        try Flux2WeightLoader.loadTransformerWeights(from: modelDirectory, into: transformer)

        NSLog("[MLXImageGen] Loading text encoder weights...")
        try Flux2WeightLoader.loadTextEncoderWeights(from: modelDirectory, into: textEncoder)

        NSLog("[MLXImageGen] Loading VAE weights...")
        try Flux2WeightLoader.loadVAEWeights(from: modelDirectory, into: vae)

        NSLog("[MLXImageGen] All weights loaded successfully")
    }

    public func generateImage(
        prompt: String,
        width: Int = 1024,
        height: Int = 1024,
        numSteps: Int = 4,
        guidanceScale: Float = 3.5,
        seed: UInt64 = 0,
        onProgress: (@Sendable (Int, Int) -> Void)? = nil
    ) async throws -> CGImage {
        let actualSeed = seed == 0 ? UInt64.random(in: 1...UInt64.max) : seed
        NSLog("[MLXImageGen] Generating %dx%d image, %d steps, seed=%llu", width, height, numSteps, actualSeed)

        // Cap MLX buffer cache to prevent unbounded memory growth
        Memory.cacheLimit = 256 * 1024 * 1024  // 256 MB
        NSLog("[MLXImageGen] Memory before generation: active=%.1fMB cache=%.1fMB",
              Float(Memory.activeMemory) / 1e6, Float(Memory.cacheMemory) / 1e6)

        // 1. Encode prompt
        NSLog("[MLXImageGen] Encoding prompt...")
        let (promptEmbeds, textIds) = try Flux2PromptEncoder.encodePrompt(
            prompt: prompt,
            tokenizer: tokenizer,
            textEncoder: textEncoder,
            maxSequenceLength: config.maxSequenceLength,
            hiddenStateLayers: config.textEncoder.hiddenStateLayers
        )
        eval(promptEmbeds, textIds)
        Memory.clearCache()
        NSLog("[MLXImageGen] After text encoding: active=%.1fMB cache=%.1fMB",
              Float(Memory.activeMemory) / 1e6, Float(Memory.cacheMemory) / 1e6)

        // 2. Create latents
        NSLog("[MLXImageGen] Creating latents...")
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

        // 4. Denoising loop
        NSLog("[MLXImageGen] Starting denoising (%d steps)...", numSteps)
        var latents = packedLatents
        let guidance = MLXArray(guidanceScale)

        for i in 0..<numSteps {
            let timestep = scheduler.timesteps[i]

            let noisePred = transformer(
                hiddenStates: latents,
                encoderHiddenStates: promptEmbeds,
                timestep: timestep,
                imgIds: latentIds,
                txtIds: textIds,
                guidance: guidance
            )
            eval(noisePred)

            latents = scheduler.step(noise: noisePred, timestepIndex: i, latents: latents)
            eval(latents)

            Memory.clearCache()
            onProgress?(i + 1, numSteps)
            NSLog("[MLXImageGen] Step %d/%d complete, active=%.1fMB cache=%.1fMB",
                  i + 1, numSteps, Float(Memory.activeMemory) / 1e6, Float(Memory.cacheMemory) / 1e6)
        }

        Memory.clearCache()

        // 5. VAE decode (BN denorm → unpatchify → post_quant_conv → decoder)
        NSLog("[MLXImageGen] Decoding with VAE...")
        let unpackedLatents = Flux2LatentCreator.unpackLatents(latents, height: height, width: width, vaeScaleFactor: config.vae.scaleFactor)

        var decoded = vae.decodePacked(unpackedLatents)
        eval(decoded)
        Memory.clearCache()
        NSLog("[MLXImageGen] After VAE decode: active=%.1fMB cache=%.1fMB",
              Float(Memory.activeMemory) / 1e6, Float(Memory.cacheMemory) / 1e6)

        // 6. Convert to image
        NSLog("[MLXImageGen] Converting to CGImage...")
        // VAE outputs [-1, 1] — denormalize to [0, 1]
        decoded = MLX.clip(decoded / 2 + 0.5, min: 0, max: 1)
        let image = try arrayToCGImage(decoded)
        Memory.clearCache()
        NSLog("[MLXImageGen] Generation complete: active=%.1fMB cache=%.1fMB peak=%.1fMB",
              Float(Memory.activeMemory) / 1e6, Float(Memory.cacheMemory) / 1e6, Float(Memory.peakMemory) / 1e6)
        return image
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
