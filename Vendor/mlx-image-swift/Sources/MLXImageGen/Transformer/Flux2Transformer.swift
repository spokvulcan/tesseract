import MLX
import MLXNN

final class Flux2Transformer: Module {
    let outChannels: Int
    let innerDim: Int

    @ModuleInfo(key: "pos_embed") var posEmbed: Flux2PosEmbed
    @ModuleInfo(key: "time_guidance_embed") var timeGuidanceEmbed: Flux2TimestepGuidanceEmbeddings
    @ModuleInfo(key: "double_stream_modulation_img") var doubleStreamModulationImg: Flux2Modulation
    @ModuleInfo(key: "double_stream_modulation_txt") var doubleStreamModulationTxt: Flux2Modulation
    @ModuleInfo(key: "single_stream_modulation") var singleStreamModulation: Flux2Modulation
    @ModuleInfo(key: "x_embedder") var xEmbedder: Linear
    @ModuleInfo(key: "context_embedder") var contextEmbedder: Linear
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [Flux2TransformerBlock]
    @ModuleInfo(key: "single_transformer_blocks") var singleTransformerBlocks: [Flux2SingleTransformerBlock]
    @ModuleInfo(key: "norm_out") var normOut: AdaLayerNormContinuous
    @ModuleInfo(key: "proj_out") var projOut: Linear

    init(config: Flux2Configuration.Transformer) {
        self.outChannels = config.outChannels
        self.innerDim = config.innerDim

        self._posEmbed.wrappedValue = Flux2PosEmbed(
            theta: config.ropeTheta, axesDim: config.axesDimsRope
        )
        self._timeGuidanceEmbed.wrappedValue = Flux2TimestepGuidanceEmbeddings(
            inChannels: config.timestepGuidanceChannels,
            embeddingDim: innerDim,
            guidanceEmbeds: config.guidanceEmbeds
        )
        self._doubleStreamModulationImg.wrappedValue = Flux2Modulation(dim: innerDim, modParamSets: 2)
        self._doubleStreamModulationTxt.wrappedValue = Flux2Modulation(dim: innerDim, modParamSets: 2)
        self._singleStreamModulation.wrappedValue = Flux2Modulation(dim: innerDim, modParamSets: 1)
        let dim = innerDim
        self._xEmbedder.wrappedValue = Linear(config.inChannels, dim, bias: false)
        self._contextEmbedder.wrappedValue = Linear(config.jointAttentionDim, dim, bias: false)

        self._transformerBlocks.wrappedValue = (0..<config.numLayers).map { _ in
            Flux2TransformerBlock(
                dim: dim,
                numAttentionHeads: config.numAttentionHeads,
                attentionHeadDim: config.attentionHeadDim,
                mlpRatio: config.mlpRatio
            )
        }
        self._singleTransformerBlocks.wrappedValue = (0..<config.numSingleLayers).map { _ in
            Flux2SingleTransformerBlock(
                dim: dim,
                numAttentionHeads: config.numAttentionHeads,
                attentionHeadDim: config.attentionHeadDim,
                mlpRatio: config.mlpRatio
            )
        }
        self._normOut.wrappedValue = AdaLayerNormContinuous(dim, dim)
        self._projOut.wrappedValue = Linear(
            innerDim, config.patchSize * config.patchSize * outChannels, bias: false
        )
    }

    func callAsFunction(
        hiddenStates: MLXArray,
        encoderHiddenStates: MLXArray,
        timestep: MLXArray,
        imgIds: MLXArray,
        txtIds: MLXArray,
        guidance: MLXArray?
    ) -> MLXArray {
        // Prepare timestep
        var ts = timestep
        if ts.ndim == 0 {
            ts = MLXArray.full([hiddenStates.dim(0)], values: ts, dtype: hiddenStates.dtype)
        }
        ts = ts.asType(hiddenStates.dtype)
        let tsScale: MLXArray = MLX.where(
            MLX.max(ts) .<=  1.0, MLXArray(Float(1000.0)), MLXArray(Float(1.0))
        ).asType(hiddenStates.dtype)
        ts = ts * tsScale

        // Prepare guidance
        var g: MLXArray? = guidance
        if var gVal = g {
            if gVal.ndim == 0 {
                gVal = MLXArray.full([hiddenStates.dim(0)], values: gVal, dtype: hiddenStates.dtype)
            }
            gVal = gVal.asType(hiddenStates.dtype)
            let gScale: MLXArray = MLX.where(
                MLX.max(gVal) .<= 1.0, MLXArray(Float(1000.0)), MLXArray(Float(1.0))
            ).asType(hiddenStates.dtype)
            g = gVal * gScale
        }

        let temb = timeGuidanceEmbed(ts, guidance: g).asType(hiddenStates.dtype)

        var hs = xEmbedder(hiddenStates)
        var ehs = contextEmbedder(encoderHiddenStates)

        // Strip batch dim from IDs if present
        let imgIdsFlat = imgIds.ndim == 3 ? imgIds[0] : imgIds
        let txtIdsFlat = txtIds.ndim == 3 ? txtIds[0] : txtIds

        let imageRotaryEmb = posEmbed(imgIdsFlat)
        let textRotaryEmb = posEmbed(txtIdsFlat)
        let concatRotaryEmb = (
            cos: MLX.concatenated([textRotaryEmb.cos, imageRotaryEmb.cos], axis: 0),
            sin: MLX.concatenated([textRotaryEmb.sin, imageRotaryEmb.sin], axis: 0)
        )

        let tembModParamsImg = doubleStreamModulationImg(temb)
        let tembModParamsTxt = doubleStreamModulationTxt(temb)

        // Double-stream blocks
        for block in transformerBlocks {
            let result = block(
                hiddenStates: hs,
                encoderHiddenStates: ehs,
                tembModParamsImg: tembModParamsImg,
                tembModParamsTxt: tembModParamsTxt,
                imageRotaryEmb: concatRotaryEmb
            )
            ehs = result.encoderOut
            hs = result.hiddenOut
        }

        // Concatenate for single-stream
        hs = MLX.concatenated([ehs, hs], axis: 1)

        let tembModParamsSingle = singleStreamModulation(temb)[0]
        for block in singleTransformerBlocks {
            hs = block(
                hiddenStates: hs,
                tembModParams: tembModParamsSingle,
                imageRotaryEmb: concatRotaryEmb
            )
        }

        // Strip text tokens
        hs = hs[0..., ehs.dim(1)..., 0...]

        // Output
        hs = normOut(hs, temb)
        hs = projOut(hs)
        return hs
    }
}
