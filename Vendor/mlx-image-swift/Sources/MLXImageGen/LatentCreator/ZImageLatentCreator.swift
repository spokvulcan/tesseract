import MLX
import MLXRandom

/// Latent noise creation and packing/unpacking for Z-Image.
/// Latent format: [16, 1, H/8, W/8] (channel-first with temporal dim).
public enum ZImageLatentCreator {
    /// Create initial noise latents.
    /// - Returns: [16, 1, H/8, W/8] in bf16
    public static func createNoise(seed: UInt64, height: Int, width: Int, spatialScale: Int = 8) -> MLXArray {
        MLXRandom.normal(
            [16, 1, height / spatialScale, width / spatialScale],
            key: MLXRandom.key(seed)
        ).asType(.bfloat16)
    }

    /// Pack latents for transformer input: insert temporal dim if needed, squeeze batch.
    /// [16, 1, H/8, W/8] → [16, 1, H/8, W/8] (no-op for 4D, squeeze 5D)
    public static func packLatents(_ latents: MLXArray) -> MLXArray {
        var result = latents
        if result.ndim == 5 {
            // [B, C, F, H, W] → squeeze F dim if size 1
            result = result.squeezed(axis: 2)
        }
        // Insert temporal dim if 4D: [C, H, W] or [C, 1, H, W]
        if result.ndim == 3 {
            result = result.expandedDimensions(axis: 1)
        }
        return result
    }

    /// Unpack latents from transformer output to VAE input format.
    /// Adds batch dim, squeezes temporal dim: → [1, 16, H/8, W/8]
    public static func unpackLatents(_ latents: MLXArray) -> MLXArray {
        var result = latents
        // Add batch dim if needed
        if result.ndim == 4 {
            result = result.expandedDimensions(axis: 0)
        }
        // Squeeze temporal dim: [B, C, 1, H, W] → [B, C, H, W]
        if result.ndim == 5 {
            result = result.squeezed(axis: 2)
        }
        return result
    }
}
