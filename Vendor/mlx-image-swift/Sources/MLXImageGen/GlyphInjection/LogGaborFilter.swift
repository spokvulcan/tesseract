import Foundation
import MLX
import MLXFFT

/// Log-Gabor frequency filter for isolating glyph edge structure in latent space.
/// Passes mid-to-high frequencies (text edges/strokes), suppresses DC and very high noise.
enum LogGaborFilter {

    /// Apply isotropic Log-Gabor band-pass filter to a 4D tensor [B, C, H, W].
    /// Operates per-channel using rfft2/irfft2.
    static func apply(
        _ input: MLXArray,
        centerFrequency: Float = 0.25,
        bandwidth: Float = 0.55
    ) -> MLXArray {
        let height = input.dim(2)
        let width = input.dim(3)

        // Build 2D isotropic Log-Gabor filter in frequency domain
        let filter = buildFilter(
            height: height, width: width,
            centerFrequency: centerFrequency,
            bandwidth: bandwidth
        )

        // Apply filter per batch×channel by reshaping to [B*C, H, W]
        let batchSize = input.dim(0)
        let channels = input.dim(1)
        let flat = input.reshaped(batchSize * channels, height, width).asType(.float32)

        // rfft2: [B*C, H, W] → [B*C, H, W/2+1] (complex)
        let spectrum = MLXFFT.rfft2(flat)

        // Multiply by filter (broadcast over batch dimension)
        let filtered = spectrum * filter

        // irfft2: [B*C, H, W/2+1] → [B*C, H, W] (real)
        let result = MLXFFT.irfft2(filtered, s: [height, width])

        return result.reshaped(batchSize, channels, height, width).asType(input.dtype)
    }

    /// Build isotropic Log-Gabor filter in rfft2 frequency domain [H, W/2+1].
    /// G(ρ) = exp(-log(ρ/f₀)² / (2·log(σ/f₀)²))
    private static func buildFilter(
        height: Int, width: Int,
        centerFrequency f0: Float,
        bandwidth sigma: Float
    ) -> MLXArray {
        let halfW = width / 2 + 1

        // Create frequency grid (normalized 0..0.5)
        // For rfft2: rows go 0..H-1, cols go 0..W/2
        let hFreqs = frequencyAxis(n: height)  // [H]
        let wFreqs = MLXArray(Array(stride(from: Float(0), to: Float(halfW), by: 1)))
            / MLXArray(Float(width))  // [W/2+1]

        // 2D grid: [H, W/2+1]
        let hGrid = hFreqs.expandedDimensions(axis: 1)  // [H, 1]
        let wGrid = wFreqs.expandedDimensions(axis: 0)  // [1, W/2+1]

        // Radial frequency: ρ = sqrt(u² + v²)
        let radius = MLX.sqrt(hGrid * hGrid + wGrid * wGrid)

        // Avoid log(0) — set DC to small value
        let safeRadius = MLX.maximum(radius, MLXArray(Float(1e-6)))

        // Log-Gabor: G(ρ) = exp(-log(ρ/f₀)² / (2·log(σ/f₀)²))
        let logRatioF0 = MLX.log(safeRadius / MLXArray(f0))
        let logSigmaF0 = Foundation.log(sigma / f0)
        let denominator = 2.0 * logSigmaF0 * logSigmaF0

        var filter = MLX.exp(-(logRatioF0 * logRatioF0) / MLXArray(denominator))

        // Zero out DC component at [0, 0] — no DC pass-through
        // Set filter[0, 0] = 0 by multiplying with a mask where only DC is zero
        let dcMask = 1.0 - (radius .== MLXArray(Float(0))).asType(.float32)
        filter = filter * dcMask

        return filter.asType(.float32)
    }

    /// FFT frequency axis: [0, 1, ..., n/2-1, -n/2, ..., -1] / n (normalized)
    private static func frequencyAxis(n: Int) -> MLXArray {
        var freqs = [Float](repeating: 0, count: n)
        for i in 0..<n {
            if i <= n / 2 {
                freqs[i] = Float(i) / Float(n)
            } else {
                freqs[i] = Float(i - n) / Float(n)
            }
        }
        return MLXArray(freqs)
    }
}
