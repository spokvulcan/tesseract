import Foundation
import MLX
import MLXNN
import HuggingFace

public enum Flux2WeightLoader {
    /// Load transformer weights from safetensors files
    public static func loadTransformerWeights(
        from directory: URL,
        into model: Module
    ) throws {
        let weights = try loadSafetensors(from: directory, prefix: "transformer")
        NSLog("[MLXImageGen] Loaded %d raw transformer weights", weights.count)
        let remapped = remapTransformerWeights(weights)
        NSLog("[MLXImageGen] Remapped to %d transformer weights", remapped.count)
        try model.update(parameters: ModuleParameters.unflattened(remapped), verify: .noUnusedKeys)
    }

    /// Load text encoder weights from safetensors files
    public static func loadTextEncoderWeights(
        from directory: URL,
        into model: Module
    ) throws {
        let weights = try loadSafetensors(from: directory, prefix: "text_encoder")
        NSLog("[MLXImageGen] Loaded %d raw text encoder weights", weights.count)
        let remapped = remapTextEncoderWeights(weights)
        NSLog("[MLXImageGen] Remapped to %d text encoder weights", remapped.count)
        try model.update(parameters: ModuleParameters.unflattened(remapped), verify: .noUnusedKeys)
    }

    /// Load VAE weights (decoder + post_quant_conv + bn) from safetensors files
    public static func loadVAEWeights(
        from directory: URL,
        into model: Module
    ) throws {
        let weights = try loadSafetensors(from: directory, prefix: "vae")
        NSLog("[MLXImageGen] Loaded %d raw VAE weights", weights.count)
        let remapped = remapVAEWeights(weights)
        NSLog("[MLXImageGen] Remapped to %d VAE weights", remapped.count)
        try model.update(parameters: ModuleParameters.unflattened(remapped), verify: .noUnusedKeys)
    }

    /// Load VAE encoder weights (encoder + quant_conv) into an already-initialized Flux2VAE
    public static func loadVAEEncoderWeights(
        from directory: URL,
        into model: Module
    ) throws {
        let weights = try loadSafetensors(from: directory, prefix: "vae")
        NSLog("[MLXImageGen] Loaded %d raw VAE weights for encoder", weights.count)
        let remapped = remapVAEEncoderWeights(weights)
        NSLog("[MLXImageGen] Remapped to %d encoder weights", remapped.count)
        try model.update(parameters: ModuleParameters.unflattened(remapped), verify: .noUnusedKeys)
    }

    // MARK: - Private helpers

    private static func loadSafetensors(from directory: URL, prefix: String) throws -> [String: MLXArray] {
        let fileManager = FileManager.default

        // Collect safetensors from root directory
        let rootFiles = (try? fileManager.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil))?
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent } ?? []

        // Try loading from root safetensors with prefix filtering (single-file format)
        var allWeights = [String: MLXArray]()
        for file in rootFiles {
            let weights = try MLX.loadArrays(url: file)
            for (key, value) in weights {
                if key.hasPrefix(prefix + ".") {
                    let strippedKey = String(key.dropFirst(prefix.count + 1))
                    allWeights[strippedKey] = value
                } else if prefix == "vae" && (key.hasPrefix("decoder.") || key.hasPrefix("encoder.") || key.hasPrefix("post_quant_conv.") || key.hasPrefix("quant_conv.") || key.hasPrefix("bn.")) {
                    allWeights[key] = value
                }
            }
        }

        // If no prefix-matched keys in root, check the corresponding subdirectory
        // (diffusers split format: text_encoder/*.safetensors, transformer/*.safetensors, etc.)
        if allWeights.isEmpty {
            let subdir = directory.appendingPathComponent(prefix)
            if fileManager.fileExists(atPath: subdir.path) {
                let subFiles = (try? fileManager.contentsOfDirectory(at: subdir, includingPropertiesForKeys: nil))?
                    .filter { $0.pathExtension == "safetensors" }
                    .sorted { $0.lastPathComponent < $1.lastPathComponent } ?? []
                for file in subFiles {
                    let weights = try MLX.loadArrays(url: file)
                    allWeights.merge(weights) { _, new in new }
                }
            }
        }

        return allWeights
    }

    private static func remapTransformerWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = [String: MLXArray]()
        for (key, value) in weights {
            let newKey = key.replacingOccurrences(of: "to_out.0.", with: "to_out.")
            result[newKey] = value
        }
        return result
    }

    private static func remapTextEncoderWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = [String: MLXArray]()
        for (key, value) in weights {
            var newKey = key
            if newKey.hasPrefix("model.") {
                newKey = String(newKey.dropFirst("model.".count))
            }
            result[newKey] = value
        }
        return result
    }

    private static func remapVAEWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = [String: MLXArray]()
        for (key, value) in weights {
            // Skip encoder and quant_conv — only keep decoder, post_quant_conv, bn
            guard key.hasPrefix("decoder.") || key.hasPrefix("post_quant_conv.") || key.hasPrefix("bn.") else { continue }
            // Skip PyTorch-only BatchNorm tracking counter
            if key == "bn.num_batches_tracked" { continue }

            var newKey = key
            // Remap to_out.0. → to_out. (only in decoder)
            if newKey.hasPrefix("decoder.") {
                newKey = newKey.replacingOccurrences(of: "to_out.0.", with: "to_out.")
            }

            // Transpose conv2d weights from PyTorch [out, in, kH, kW] to MLX [out, kH, kW, in]
            if newKey.contains("conv") && newKey.hasSuffix(".weight") && value.ndim == 4 {
                result[newKey] = value.transposed(0, 2, 3, 1)
            } else {
                result[newKey] = value
            }
        }
        return result
    }

    private static func remapVAEEncoderWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = [String: MLXArray]()
        for (key, value) in weights {
            // Only keep encoder and quant_conv weights
            guard key.hasPrefix("encoder.") || key.hasPrefix("quant_conv.") else { continue }

            var newKey = key
            // Remap to_out.0. → to_out. (attention output projection)
            if newKey.hasPrefix("encoder.") {
                newKey = newKey.replacingOccurrences(of: "to_out.0.", with: "to_out.")
            }

            // Transpose conv2d weights from PyTorch [out, in, kH, kW] to MLX [out, kH, kW, in]
            if newKey.contains("conv") && newKey.hasSuffix(".weight") && value.ndim == 4 {
                result[newKey] = value.transposed(0, 2, 3, 1)
            } else {
                result[newKey] = value
            }
        }
        return result
    }
}
