import MLX
import MLXNN

/// Final output layer: LayerNorm(affine=false) + adaLN scale + Linear projection.
/// Python: FinalLayer(hidden_size=3840, out_channels=64)
final class ZImageFinalLayer: Module {
    @ModuleInfo var norm: LayerNorm
    @ModuleInfo var linear: Linear
    @ModuleInfo(key: "adaLN_modulation") var adaLNModulation: Linear

    init(hiddenSize: Int, outChannels: Int) {
        self._norm.wrappedValue = LayerNorm(dimensions: hiddenSize, eps: 1e-6, affine: false)
        self._linear.wrappedValue = Linear(hiddenSize, outChannels)
        self._adaLNModulation.wrappedValue = Linear(min(hiddenSize, 256), hiddenSize)
    }

    func callAsFunction(_ x: MLXArray, _ c: MLXArray) -> MLXArray {
        // scale = 1.0 + adaLN_modulation(silu(c))
        let scale = 1.0 + adaLNModulation(silu(c)).expandedDimensions(axis: 1)
        let normed = norm(x) * scale
        return linear(normed)
    }
}
