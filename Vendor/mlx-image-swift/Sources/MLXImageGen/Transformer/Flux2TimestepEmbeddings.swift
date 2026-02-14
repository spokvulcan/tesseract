import Foundation
import MLX
import MLXNN

final class Flux2TimestepEmbedder: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(inChannels: Int, embeddingDim: Int) {
        self._linear1.wrappedValue = Linear(inChannels, embeddingDim, bias: false)
        self._linear2.wrappedValue = Linear(embeddingDim, embeddingDim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(silu(linear1(x)))
    }
}

final class Flux2TimestepGuidanceEmbeddings: Module {
    let inChannels: Int
    let guidanceEmbeds: Bool
    @ModuleInfo(key: "timestep_embedder") var timestepEmbedder: Flux2TimestepEmbedder
    @ModuleInfo(key: "guidance_embedder") var guidanceEmbedder: Flux2TimestepEmbedder?

    init(inChannels: Int = 256, embeddingDim: Int = 6144, guidanceEmbeds: Bool = false) {
        self.inChannels = inChannels
        self.guidanceEmbeds = guidanceEmbeds
        self._timestepEmbedder.wrappedValue = Flux2TimestepEmbedder(
            inChannels: inChannels, embeddingDim: embeddingDim
        )
        if guidanceEmbeds {
            self._guidanceEmbedder.wrappedValue = Flux2TimestepEmbedder(
                inChannels: inChannels, embeddingDim: embeddingDim
            )
        }
    }

    func callAsFunction(_ timestep: MLXArray, guidance: MLXArray?) -> MLXArray {
        let tsEmb = Self.timestepEmbedding(timestep.asType(.float32), dim: inChannels)
        let timestepsEmb = timestepEmbedder(tsEmb)

        if let guidance, let gEmbedder = guidanceEmbedder {
            let gEmb = Self.timestepEmbedding(guidance.asType(.float32), dim: inChannels)
            let guidanceEmb = gEmbedder(gEmb)
            return timestepsEmb + guidanceEmb
        }
        return timestepsEmb
    }

    static func timestepEmbedding(_ timesteps: MLXArray, dim: Int, flipSinToCos: Bool = true) -> MLXArray {
        let half = dim / 2
        let logFactor = -log(10000.0)
        let freqs = MLX.exp(MLXArray(Float(logFactor)) * MLXArray(stride(from: 0, to: half, by: 1).map { Float($0) / Float(half) }))
        let args = timesteps.expandedDimensions(axis: -1) * freqs.expandedDimensions(axis: 0)
        var emb = MLX.concatenated([MLX.sin(args), MLX.cos(args)], axis: -1)
        if flipSinToCos {
            let sinPart = emb[.ellipsis, half...]
            let cosPart = emb[.ellipsis, ..<half]
            emb = MLX.concatenated([sinPart, cosPart], axis: -1)
        }
        if dim % 2 == 1 {
            emb = MLX.concatenated([emb, MLXArray.zeros([emb.dim(0), 1])], axis: -1)
        }
        return emb
    }
}
