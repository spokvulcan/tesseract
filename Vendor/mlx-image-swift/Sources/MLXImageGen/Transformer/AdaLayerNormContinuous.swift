import MLX
import MLXNN

final class AdaLayerNormContinuous: Module {
    let embeddingDim: Int
    @ModuleInfo var linear: Linear
    @ModuleInfo var norm: LayerNorm

    init(_ embeddingDim: Int, _ conditioningEmbeddingDim: Int) {
        self.embeddingDim = embeddingDim
        self._linear.wrappedValue = Linear(conditioningEmbeddingDim, embeddingDim * 2, bias: false)
        self._norm.wrappedValue = LayerNorm(dimensions: embeddingDim, eps: 1e-6, affine: false)
    }

    func callAsFunction(_ x: MLXArray, _ textEmbeddings: MLXArray) -> MLXArray {
        let emb = linear(silu(textEmbeddings))
        let chunkSize = embeddingDim
        let scale = emb[0..., (0 * chunkSize)..<(1 * chunkSize)]
        let shift = emb[0..., (1 * chunkSize)..<(2 * chunkSize)]
        return norm(x) * (1 + scale.expandedDimensions(axis: 1)) + shift.expandedDimensions(axis: 1)
    }
}
