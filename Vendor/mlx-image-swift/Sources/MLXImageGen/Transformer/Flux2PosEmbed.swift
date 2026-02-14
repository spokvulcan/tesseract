import MLX
import MLXNN

final class Flux2PosEmbed: Module {
    let theta: Int
    let axesDim: [Int]

    init(theta: Int = 2000, axesDim: [Int] = [32, 32, 32, 32]) {
        self.theta = theta
        self.axesDim = axesDim
    }

    func callAsFunction(_ ids: MLXArray) -> (cos: MLXArray, sin: MLXArray) {
        let pos = ids.asType(.float32)
        var cosOut = [MLXArray]()
        var sinOut = [MLXArray]()
        for (i, dim) in axesDim.enumerated() {
            let (cos, sin) = get1DRope(dim: dim, pos: pos[.ellipsis, i])
            cosOut.append(cos)
            sinOut.append(sin)
        }
        let freqsCos = MLX.concatenated(cosOut, axis: -1)
        let freqsSin = MLX.concatenated(sinOut, axis: -1)
        return (freqsCos, freqsSin)
    }

    private func get1DRope(dim: Int, pos: MLXArray) -> (cos: MLXArray, sin: MLXArray) {
        let scale = MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) / Float(dim) })
        let omega = 1.0 / MLX.pow(MLXArray(Float(theta)), scale)
        let posExpanded = pos.expandedDimensions(axis: -1)
        let omegaExpanded = omega.expandedDimensions(axis: 0)
        let out = posExpanded * omegaExpanded
        return (MLX.cos(out), MLX.sin(out))
    }
}
