import MLX
import MLXNN

/// Z-Image single-stream DiT transformer.
/// Architecture: patchify → x_embed → noise_refiner(2) → cap_embed → context_refiner(2) →
///   concat → main_layers(30) → final_layer → unpatchify → negate
final class ZImageTransformer: Module {
    let config: ZImageConfiguration.Transformer

    @ModuleInfo(key: "t_embedder") var tEmbedder: ZImageTimestepEmbedder
    @ModuleInfo(key: "x_embedder") var xEmbedder: Linear
    @ModuleInfo(key: "final_layer") var finalLayer: ZImageFinalLayer
    @ModuleInfo(key: "cap_norm") var capNorm: RMSNorm
    @ModuleInfo(key: "cap_linear") var capLinear: Linear
    // Plain MLXArray parameters — property names match safetensors keys
    var x_pad_token: MLXArray
    var cap_pad_token: MLXArray
    @ModuleInfo(key: "noise_refiner") var noiseRefiner: [ZImageTransformerBlock]
    @ModuleInfo(key: "context_refiner") var contextRefiner: [ZImageContextBlock]
    @ModuleInfo var layers: [ZImageTransformerBlock]

    let ropeEmbedder: ZImageRopeEmbedder

    init(config: ZImageConfiguration.Transformer) {
        self.config = config
        let dim = config.dim
        let embedDim = config.fPatchSize * config.patchSize * config.patchSize * config.inChannels  // 1*2*2*16 = 64

        self._tEmbedder.wrappedValue = ZImageTimestepEmbedder(
            outSize: min(dim, 256), midSize: 1024
        )
        self._xEmbedder.wrappedValue = Linear(embedDim, dim)
        self._finalLayer.wrappedValue = ZImageFinalLayer(hiddenSize: dim, outChannels: embedDim)
        self._capNorm.wrappedValue = RMSNorm(dimensions: config.capFeatDim, eps: config.normEps)
        self._capLinear.wrappedValue = Linear(config.capFeatDim, dim)
        self.x_pad_token = MLXArray.zeros([1, dim])
        self.cap_pad_token = MLXArray.zeros([1, dim])

        self._noiseRefiner.wrappedValue = (0..<config.nRefinerLayers).map { _ in
            ZImageTransformerBlock(dim: dim, nHeads: config.nHeads, normEps: config.normEps)
        }
        self._contextRefiner.wrappedValue = (0..<config.nRefinerLayers).map { _ in
            ZImageContextBlock(dim: dim, nHeads: config.nHeads, normEps: config.normEps)
        }
        self._layers.wrappedValue = (0..<config.nLayers).map { _ in
            ZImageTransformerBlock(dim: dim, nHeads: config.nHeads, normEps: config.normEps)
        }
        self.ropeEmbedder = ZImageRopeEmbedder(
            theta: config.ropeTheta,
            axesDims: config.axesDims,
            axesLens: config.axesLens
        )
    }

    func callAsFunction(
        x: MLXArray,
        timestep: MLXArray,
        sigmas: MLXArray,
        capFeats: MLXArray
    ) -> MLXArray {
        // Time embedding: timestep * t_scale → sinusoidal → MLP
        var ts = timestep
        if ts.ndim == 0 { ts = ts.reshaped(1) }
        let tEmb = tEmbedder(ts.asType(.float32) * MLXArray(config.tScale))

        // Patchify image and caption
        let (xEmb, capEmb, imageSize, xPosIds, capPosIds, xPadMask, capPadMask) = Self.patchify(
            image: x,
            capFeats: capFeats,
            patchSize: config.patchSize,
            fPatchSize: config.fPatchSize
        )

        // Image embedding
        var xEmbedded = xEmbedder(xEmb)
        // Zero out padded positions with x_pad_token
        xEmbedded = MLX.where(xPadMask[.ellipsis, .newAxis], x_pad_token, xEmbedded)
        let xFreqsCis = ropeEmbedder(xPosIds)
        let xLen = xEmbedded.dim(0)
        // Add batch dimension: [seqLen, dim] → [1, seqLen, dim]
        xEmbedded = xEmbedded.expandedDimensions(axis: 0)

        // Noise refiner
        let xAttnMask = MLXArray.ones([1, xLen]).asType(.bool)
        for layer in noiseRefiner {
            xEmbedded = layer(x: xEmbedded, attnMask: xAttnMask, freqsCis: xFreqsCis, tEmb: tEmb)
        }

        // Caption embedding
        var capEmbedded = capLinear(capNorm(capEmb))
        capEmbedded = MLX.where(capPadMask[.ellipsis, .newAxis], cap_pad_token, capEmbedded)
        let capFreqsCis = ropeEmbedder(capPosIds)
        let capLen = capEmbedded.dim(0)
        capEmbedded = capEmbedded.expandedDimensions(axis: 0)

        // Context refiner
        let capAttnMask = MLXArray.ones([1, capLen]).asType(.bool)
        for layer in contextRefiner {
            capEmbedded = layer(x: capEmbedded, attnMask: capAttnMask, freqsCis: capFreqsCis)
        }

        // Unify and main layers
        var unified = MLX.concatenated([xEmbedded, capEmbedded], axis: 1)
        let unifiedFreqsCis = MLX.concatenated([xFreqsCis, capFreqsCis], axis: 0)
        let unifiedAttnMask = MLXArray.ones([1, unified.dim(1)]).asType(.bool)

        for layer in layers {
            unified = layer(x: unified, attnMask: unifiedAttnMask, freqsCis: unifiedFreqsCis, tEmb: tEmb)
        }

        // Final layer and unpatchify
        unified = finalLayer(unified, tEmb)
        let output = Self.unpatchify(
            x: unified[0, ..<xLen],
            size: imageSize,
            patchSize: config.patchSize,
            fPatchSize: config.fPatchSize,
            outChannels: config.inChannels
        )
        // Negate output (Z-Image convention)
        return -output
    }

    // MARK: - Patchify / Unpatchify

    /// Patchify image [C, F, H, W] into patch tokens and prepare position IDs.
    static func patchify(
        image: MLXArray,
        capFeats: MLXArray,
        patchSize: Int,
        fPatchSize: Int
    ) -> (
        xEmb: MLXArray, capEmb: MLXArray, imageSize: (Int, Int, Int),
        xPosIds: MLXArray, capPosIds: MLXArray,
        xPadMask: MLXArray, capPadMask: MLXArray
    ) {
        let pH = patchSize
        let pW = patchSize
        let pF = fPatchSize

        // Caption padding to multiple of 32
        let capOriLen = capFeats.dim(0)
        let capPaddingLen = (32 - capOriLen % 32) % 32
        let totalCapLen = capOriLen + capPaddingLen
        let capPosIds = createCoordGrid(
            size: (totalCapLen, 1, 1), start: (1, 0, 0)
        ).reshaped(-1, 3)
        let capPadMask = MLX.concatenated([
            MLXArray.zeros([capOriLen]).asType(.bool),
            MLXArray.ones([capPaddingLen]).asType(.bool)
        ])
        let capPadded: MLXArray
        if capPaddingLen > 0 {
            let lastToken = capFeats[capOriLen - 1].expandedDimensions(axis: 0)  // [1, featDim]
            let repeated = MLX.repeated(lastToken, count: capPaddingLen, axis: 0)
            capPadded = MLX.concatenated([capFeats, repeated], axis: 0)
        } else {
            capPadded = capFeats
        }

        // Image patchification
        let (C, F, H, W) = (image.dim(0), image.dim(1), image.dim(2), image.dim(3))
        let imageSize = (F, H, W)
        let fTokens = F / pF
        let hTokens = H / pH
        let wTokens = W / pW

        // [C, F, H, W] → [C, F_t, pF, H_t, pH, W_t, pW]
        var img = image.reshaped(C, fTokens, pF, hTokens, pH, wTokens, pW)
        // transpose(1,3,5,2,4,6,0) → [F_t, H_t, W_t, pF, pH, pW, C]
        img = img.transposed(1, 3, 5, 2, 4, 6, 0)
        // flatten → [F_t*H_t*W_t, pF*pH*pW*C]
        let numPatches = fTokens * hTokens * wTokens
        let patchDim = pF * pH * pW * C
        img = img.reshaped(numPatches, patchDim)

        // Image padding to multiple of 32
        let imgOriLen = numPatches
        let imgPaddingLen = (32 - imgOriLen % 32) % 32
        var xPosIds = createCoordGrid(
            size: (fTokens, hTokens, wTokens), start: (totalCapLen + 1, 0, 0)
        ).reshaped(-1, 3)

        if imgPaddingLen > 0 {
            xPosIds = MLX.concatenated([
                xPosIds,
                MLXArray.zeros([imgPaddingLen, 3], type: Int32.self)
            ], axis: 0)
            let lastPatch = img[imgOriLen - 1].expandedDimensions(axis: 0)  // [1, patchDim]
            let repeated = MLX.repeated(lastPatch, count: imgPaddingLen, axis: 0)
            img = MLX.concatenated([img, repeated], axis: 0)
        }

        let xPadMask = MLX.concatenated([
            MLXArray.zeros([imgOriLen]).asType(.bool),
            MLXArray.ones([imgPaddingLen]).asType(.bool)
        ])

        return (img, capPadded, imageSize, xPosIds, capPosIds, xPadMask, capPadMask)
    }

    /// Unpatchify patch tokens back to image [C, F, H, W].
    static func unpatchify(
        x: MLXArray,
        size: (Int, Int, Int),
        patchSize: Int,
        fPatchSize: Int,
        outChannels: Int
    ) -> MLXArray {
        let pH = patchSize
        let pW = patchSize
        let pF = fPatchSize
        let (F, H, W) = size
        let fTokens = F / pF
        let hTokens = H / pH
        let wTokens = W / pW
        let oriLen = fTokens * hTokens * wTokens

        // Take only the original (non-padded) tokens
        var out = x[..<oriLen]
        // [oriLen, pF*pH*pW*C] → [F_t, H_t, W_t, pF, pH, pW, C]
        out = out.reshaped(fTokens, hTokens, wTokens, pF, pH, pW, outChannels)
        // transpose(6,0,3,1,4,2,5) → [C, F_t, pF, H_t, pH, W_t, pW]
        out = out.transposed(6, 0, 3, 1, 4, 2, 5)
        // flatten → [C, F, H, W]
        return out.reshaped(outChannels, F, H, W)
    }

    /// Create a 3D coordinate grid with offsets.
    static func createCoordGrid(size: (Int, Int, Int), start: (Int, Int, Int)) -> MLXArray {
        let (s0, s1, s2) = size
        let (st0, st1, st2) = start

        let a0 = MLXArray((st0..<(st0 + s0)).map { Int32($0) })
        let a1 = MLXArray((st1..<(st1 + s1)).map { Int32($0) })
        let a2 = MLXArray((st2..<(st2 + s2)).map { Int32($0) })

        // meshgrid with "ij" indexing
        let g0 = MLX.broadcast(
            a0.reshaped(s0, 1, 1),
            to: [s0, s1, s2]
        )
        let g1 = MLX.broadcast(
            a1.reshaped(1, s1, 1),
            to: [s0, s1, s2]
        )
        let g2 = MLX.broadcast(
            a2.reshaped(1, 1, s2),
            to: [s0, s1, s2]
        )
        return MLX.stacked([g0, g1, g2], axis: -1)
    }
}
