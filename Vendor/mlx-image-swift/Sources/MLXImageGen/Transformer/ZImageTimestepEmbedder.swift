import Foundation
import MLX
import MLXNN

/// Sinusoidal positional encoding → 2-layer MLP for timestep embedding.
/// Python: TimestepEmbedder(out_size=min(dim, 256), mid_size=1024, frequency_embedding_size=256)
final class ZImageTimestepEmbedder: Module {
    let frequencyEmbeddingSize: Int

    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear

    init(outSize: Int = 256, midSize: Int = 1024, frequencyEmbeddingSize: Int = 256) {
        self.frequencyEmbeddingSize = frequencyEmbeddingSize
        self._linear1.wrappedValue = Linear(frequencyEmbeddingSize, midSize)
        self._linear2.wrappedValue = Linear(midSize, outSize)
    }

    func callAsFunction(_ t: MLXArray) -> MLXArray {
        let tFreq = Self.timestepEmbedding(t, dim: frequencyEmbeddingSize)
        var emb = linear1(tFreq)
        emb = silu(emb)
        emb = linear2(emb)
        return emb
    }

    /// Sinusoidal timestep embedding matching Python's _timestep_embedding.
    static func timestepEmbedding(_ t: MLXArray, dim: Int, maxPeriod: Float = 10000.0) -> MLXArray {
        let half = dim / 2
        let freqScale = MLXArray(
            (0..<half).map { -Foundation.log(maxPeriod) * Float($0) / Float(half) }
        )
        let freqs = MLX.exp(freqScale)
        let args = t[.ellipsis, .newAxis].asType(.float32) * freqs[.newAxis, .ellipsis]
        let embedding = MLX.concatenated([MLX.cos(args), MLX.sin(args)], axis: -1)
        if dim % 2 != 0 {
            return MLX.concatenated([embedding, MLXArray.zeros(like: embedding[.ellipsis, ..<1])], axis: -1)
        }
        return embedding
    }
}
