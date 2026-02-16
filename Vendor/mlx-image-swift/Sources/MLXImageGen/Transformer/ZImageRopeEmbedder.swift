import Foundation
import MLX

/// 3D precomputed cos/sin RoPE tables for Z-Image transformer.
/// Precomputes per-axis frequency tables, then indexes by 3D position IDs at call time.
/// Python: RopeEmbedder(theta=256, axes_dims=[32,48,48], axes_lens=[1024,512,512])
final class ZImageRopeEmbedder {
    let axesDims: [Int]
    /// Precomputed [cos, sin] tables per axis: each is [axisLen, axisDim//2, 2]
    let freqsCis: [MLXArray]

    init(theta: Float = 256.0, axesDims: [Int] = [32, 48, 48], axesLens: [Int] = [1024, 512, 512]) {
        self.axesDims = axesDims
        self.freqsCis = Self.precomputeFreqsCis(axesDims: axesDims, axesLens: axesLens, theta: theta)
    }

    /// Lookup RoPE embeddings by position IDs.
    /// - Parameter ids: [seqLen, 3] integer position IDs (one column per axis)
    /// - Returns: [seqLen, totalDim//2, 2] where totalDim = sum(axesDims)
    func callAsFunction(_ ids: MLXArray) -> MLXArray {
        var results = [MLXArray]()
        for (i, _) in axesDims.enumerated() {
            let index = ids[.ellipsis, i].asType(.int32)
            results.append(freqsCis[i][index])
        }
        return MLX.concatenated(results, axis: 1)
    }

    /// Precompute cos/sin tables for each axis.
    private static func precomputeFreqsCis(axesDims: [Int], axesLens: [Int], theta: Float) -> [MLXArray] {
        var tables = [MLXArray]()
        for (d, e) in zip(axesDims, axesLens) {
            let scale = MLXArray((0..<(d / 2)).map { Float($0) * 2.0 / Float(d) })
            let freqs = 1.0 / MLX.pow(MLXArray(theta), scale)
            let timestep = MLXArray((0..<e).map { Float($0) })
            // [e] x [d/2] → [e, d/2]
            let outerProduct = timestep.expandedDimensions(axis: -1) * freqs.expandedDimensions(axis: 0)
            let cosFreqs = MLX.cos(outerProduct)
            let sinFreqs = MLX.sin(outerProduct)
            // Stack to [e, d/2, 2]
            let freqsCisI = MLX.stacked([cosFreqs, sinFreqs], axis: -1)
            tables.append(freqsCisI)
        }
        return tables
    }

    /// Apply complex-interleaved RoPE to Q/K in BSHD format (before transpose to BHSD).
    /// - Parameters:
    ///   - x: [batch, seqLen, nHeads, headDim]
    ///   - freqsCis: [seqLen, headDim//2, 2] from callAsFunction
    /// - Returns: same shape as x with RoPE applied
    static func applyRotaryEmb(_ x: MLXArray, freqsCis: MLXArray) -> MLXArray {
        let (batchSize, seqLen, nHeads, headDim) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        // Reshape to complex pairs: [B, S, H, D/2, 2]
        let x2 = x.reshaped(batchSize, seqLen, nHeads, headDim / 2, 2)
        // Broadcast freqs_cis: [1, S, 1, D/2, 2]
        let fc = freqsCis.expandedDimensions(axes: [0, 2])
        let xReal = x2[.ellipsis, 0]
        let xImag = x2[.ellipsis, 1]
        let freqsCos = fc[.ellipsis, 0]
        let freqsSin = fc[.ellipsis, 1]
        let outReal = xReal * freqsCos - xImag * freqsSin
        let outImag = xReal * freqsSin + xImag * freqsCos
        let outStacked = MLX.stacked([outReal, outImag], axis: -1)
        return outStacked.reshaped(batchSize, seqLen, nHeads, headDim)
    }
}
