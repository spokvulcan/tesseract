import MLX
import MLXRandom

public enum Flux2LatentCreator {
    /// Pack latents from [B, C, H, W] to [B, H*W, C]
    public static func packLatents(_ latents: MLXArray) -> MLXArray {
        let (batchSize, numChannels, height, width) = (
            latents.dim(0), latents.dim(1), latents.dim(2), latents.dim(3)
        )
        return latents.reshaped(batchSize, numChannels, height * width).transposed(0, 2, 1)
    }

    /// Unpack latents from [B, seqLen, C] back to [B, C, H, W]
    public static func unpackLatents(
        _ latents: MLXArray, height: Int, width: Int, vaeScaleFactor: Int = 8
    ) -> MLXArray {
        if latents.ndim == 4 { return latents }
        let (batchSize, _, channels) = (latents.dim(0), latents.dim(1), latents.dim(2))
        let latentHeight = height / (vaeScaleFactor * 2)
        let latentWidth = width / (vaeScaleFactor * 2)
        return latents.reshaped(batchSize, latentHeight, latentWidth, channels).transposed(0, 3, 1, 2)
    }

    /// Create grid position IDs for latents [B, seqLen, 4] with (t, h, w, layer)
    public static func prepareGridIds(_ latents: MLXArray, tCoord: Int = 0) -> MLXArray {
        let (batchSize, _, height, width) = (latents.dim(0), latents.dim(1), latents.dim(2), latents.dim(3))
        let hIds = MLXArray(0..<Int32(height))
        let wIds = MLXArray(0..<Int32(width))

        let hGrid = MLX.broadcast(hIds.expandedDimensions(axis: 1), to: [height, width])
        let wGrid = MLX.broadcast(wIds.expandedDimensions(axis: 0), to: [height, width])

        let flatH = hGrid.reshaped(-1)
        let flatW = wGrid.reshaped(-1)
        let t = MLXArray.full([flatH.dim(0)], values: MLXArray(Int32(tCoord)), type: Int32.self)
        let layerIds = MLXArray.zeros([flatH.dim(0)], type: Int32.self)

        let coords = MLX.stacked([t, flatH, flatW, layerIds], axis: 1).expandedDimensions(axis: 0)
        return MLX.broadcast(coords, to: [batchSize, coords.dim(1), coords.dim(2)])
    }

    /// Prepare packed latents and grid IDs
    public static func preparePackedLatents(
        seed: UInt64,
        height: Int,
        width: Int,
        batchSize: Int = 1,
        numLatentChannels: Int = 32,
        vaeScaleFactor: Int = 8
    ) -> (latents: MLXArray, latentIds: MLXArray, latentHeight: Int, latentWidth: Int) {
        let h = 2 * (height / (vaeScaleFactor * 2))
        let w = 2 * (width / (vaeScaleFactor * 2))
        let latentHeight = h / 2
        let latentWidth = w / 2

        let latents = MLXRandom.normal(
            [batchSize, numLatentChannels * 4, latentHeight, latentWidth],
            key: MLXRandom.key(seed)
        ).asType(.bfloat16)

        let latentIds = prepareGridIds(latents, tCoord: 0)
        let packed = packLatents(latents)

        return (packed, latentIds, latentHeight, latentWidth)
    }

    /// Patchify: [B, C, H, W] -> [B, C*4, H/2, W/2] (inverse of unpatchify)
    public static func patchifyLatents(_ latents: MLXArray) -> MLXArray {
        let (batchSize, channels, height, width) = (
            latents.dim(0), latents.dim(1), latents.dim(2), latents.dim(3)
        )
        let halfH = height / 2
        let halfW = width / 2
        // [B, C, H, W] -> [B, C, H/2, 2, W/2, 2] -> [B, C, 2, 2, H/2, W/2] -> [B, C*4, H/2, W/2]
        let reshaped = latents.reshaped(batchSize, channels, halfH, 2, halfW, 2)
        let transposed = reshaped.transposed(0, 1, 3, 5, 2, 4)
        return transposed.reshaped(batchSize, channels * 4, halfH, halfW)
    }

    /// Unpatchify: [B, C*4, H/2, W/2] -> [B, C, H, W]
    public static func unpatchifyLatents(_ latents: MLXArray) -> MLXArray {
        let (batchSize, channels4, halfH, halfW) = (
            latents.dim(0), latents.dim(1), latents.dim(2), latents.dim(3)
        )
        let channels = channels4 / 4
        // Reverse of patchify: [B, C*4, H/2, W/2] -> [B, C, 2, 2, H/2, W/2] -> [B, C, H, W]
        let reshaped = latents.reshaped(batchSize, channels, 2, 2, halfH, halfW)
        let transposed = reshaped.transposed(0, 1, 4, 2, 5, 3)
        return transposed.reshaped(batchSize, channels, halfH * 2, halfW * 2)
    }
}
