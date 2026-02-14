import Foundation
import MLX
import MLXNN
import HuggingFace

public enum ZImageWeightLoader {
    /// Load transformer weights from safetensors files.
    public static func loadTransformerWeights(
        from directory: URL,
        into model: Module
    ) throws {
        let weights = try loadSafetensors(from: directory, prefix: "transformer")
        NSLog("[MLXImageGen] Loaded %d raw Z-Image transformer weights", weights.count)
        let remapped = remapTransformerWeights(weights)
        NSLog("[MLXImageGen] Remapped to %d Z-Image transformer weights", remapped.count)
        try model.update(parameters: ModuleParameters.unflattened(remapped), verify: .noUnusedKeys)
    }

    /// Load text encoder weights from safetensors files.
    public static func loadTextEncoderWeights(
        from directory: URL,
        into model: Module
    ) throws {
        let weights = try loadSafetensors(from: directory, prefix: "text_encoder")
        NSLog("[MLXImageGen] Loaded %d raw Z-Image text encoder weights", weights.count)
        let remapped = remapTextEncoderWeights(weights)
        NSLog("[MLXImageGen] Remapped to %d Z-Image text encoder weights", remapped.count)
        try model.update(parameters: ModuleParameters.unflattened(remapped), verify: .noUnusedKeys)
    }

    /// Load VAE decoder weights from safetensors files.
    public static func loadVAEWeights(
        from directory: URL,
        into model: Module
    ) throws {
        let weights = try loadSafetensors(from: directory, prefix: "vae")
        NSLog("[MLXImageGen] Loaded %d raw Z-Image VAE weights", weights.count)
        let remapped = remapVAEWeights(weights)
        NSLog("[MLXImageGen] Remapped to %d Z-Image VAE weights", remapped.count)
        try model.update(parameters: ModuleParameters.unflattened(remapped), verify: .noUnusedKeys)
    }

    // MARK: - Private helpers

    private static func loadSafetensors(from directory: URL, prefix: String) throws -> [String: MLXArray] {
        let fileManager = FileManager.default

        // Collect safetensors from root directory
        let rootFiles = (try? fileManager.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil))?
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent } ?? []

        var allWeights = [String: MLXArray]()
        for file in rootFiles {
            let weights = try MLX.loadArrays(url: file)
            for (key, value) in weights {
                if key.hasPrefix(prefix + ".") {
                    let strippedKey = String(key.dropFirst(prefix.count + 1))
                    allWeights[strippedKey] = value
                } else if prefix == "vae" && (key.hasPrefix("decoder.") || key.hasPrefix("encoder.")) {
                    allWeights[key] = value
                }
            }
        }

        // If no prefix-matched keys in root, check the corresponding subdirectory
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

    // MARK: - Transformer remapping

    private static func remapTransformerWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = [String: MLXArray]()
        for (key, value) in weights {
            var newKey = key

            // 1. all_x_embedder.2-1. → x_embedder.
            newKey = newKey.replacingOccurrences(of: "all_x_embedder.2-1.", with: "x_embedder.")
            // 2. all_final_layer.2-1. → final_layer.
            newKey = newKey.replacingOccurrences(of: "all_final_layer.2-1.", with: "final_layer.")
            // 3. cap_embedder.0. → cap_norm.
            newKey = newKey.replacingOccurrences(of: "cap_embedder.0.", with: "cap_norm.")
            // 4. cap_embedder.1. → cap_linear.
            newKey = newKey.replacingOccurrences(of: "cap_embedder.1.", with: "cap_linear.")
            // 5. t_embedder.mlp.0. → t_embedder.linear1.
            newKey = newKey.replacingOccurrences(of: "t_embedder.mlp.0.", with: "t_embedder.linear1.")
            // 6. t_embedder.mlp.2. → t_embedder.linear2.
            newKey = newKey.replacingOccurrences(of: "t_embedder.mlp.2.", with: "t_embedder.linear2.")
            // 7. adaLN_modulation.0. → adaLN_modulation. (for transformer/refiner blocks)
            newKey = newKey.replacingOccurrences(of: ".adaLN_modulation.0.", with: ".adaLN_modulation.")
            // 8. adaLN_modulation.1. → adaLN_modulation. (for final layer, after step 2)
            newKey = newKey.replacingOccurrences(of: ".adaLN_modulation.1.", with: ".adaLN_modulation.")
            // 9. to_out.0. → to_out.
            newKey = newKey.replacingOccurrences(of: ".to_out.0.", with: ".to_out.")

            result[newKey] = value
        }
        return result
    }

    // MARK: - Text encoder remapping

    private static func remapTextEncoderWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = [String: MLXArray]()
        for (key, value) in weights {
            var newKey = key
            // Strip model. prefix (same as FLUX)
            if newKey.hasPrefix("model.") {
                newKey = String(newKey.dropFirst("model.".count))
            }
            result[newKey] = value
        }
        return result
    }

    // MARK: - VAE remapping

    private static func remapVAEWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = [String: MLXArray]()
        for (key, value) in weights {
            // Only keep decoder weights (no encoder, no post_quant_conv, no bn)
            guard key.hasPrefix("decoder.") else { continue }

            var newKey = key
            // Remap to_out.0. → to_out.
            newKey = newKey.replacingOccurrences(of: "to_out.0.", with: "to_out.")

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
