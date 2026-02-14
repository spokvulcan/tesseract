import MLX
import MLXNN

final class Flux2Modulation: Module {
    let modParamSets: Int
    @ModuleInfo var linear: Linear

    init(dim: Int, modParamSets: Int = 2) {
        self.modParamSets = modParamSets
        self._linear.wrappedValue = Linear(dim, dim * 3 * modParamSets, bias: false)
    }

    func callAsFunction(_ temb: MLXArray) -> [ModulationParams] {
        var mod = silu(temb)
        mod = linear(mod)
        if mod.ndim == 2 {
            mod = mod.expandedDimensions(axis: 1)
        }
        let chunks = mod.split(parts: 3 * modParamSets, axis: -1)
        var result = [ModulationParams]()
        for i in 0..<modParamSets {
            result.append(ModulationParams(
                shift: chunks[3 * i],
                scale: chunks[3 * i + 1],
                gate: chunks[3 * i + 2]
            ))
        }
        return result
    }
}

struct ModulationParams {
    let shift: MLXArray
    let scale: MLXArray
    let gate: MLXArray
}
