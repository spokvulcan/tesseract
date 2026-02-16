import MLX
import MLXFast
import MLXNN

private let compiledSwiGLU: @Sendable (MLXArray, MLXArray) -> MLXArray =
    compile(shapeless: true) { gate, up in silu(gate) * up }

final class Flux2SwiGLU: Module {
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = x.split(parts: 2, axis: -1)
        return compiledSwiGLU(parts[0], parts[1])
    }
}

final class Flux2FeedForward: Module {
    @ModuleInfo(key: "linear_in") var linearIn: Linear
    let act: Flux2SwiGLU
    @ModuleInfo(key: "linear_out") var linearOut: Linear

    init(dim: Int, mult: Float = 3.0) {
        let innerDim = Int(Float(dim) * mult)
        self._linearIn.wrappedValue = Linear(dim, innerDim * 2, bias: false)
        self.act = Flux2SwiGLU()
        self._linearOut.wrappedValue = Linear(innerDim, dim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linearOut(act(linearIn(x)))
    }
}
