import Foundation
import MLX

public enum ZImageConfiguration {
    public struct Transformer: Sendable {
        public let dim: Int
        public let nHeads: Int
        public let headDim: Int
        public let nLayers: Int
        public let nRefinerLayers: Int
        public let patchSize: Int
        public let fPatchSize: Int
        public let inChannels: Int
        public let capFeatDim: Int
        public let normEps: Float
        public let ropeTheta: Float
        public let tScale: Float
        public let axesDims: [Int]
        public let axesLens: [Int]
        public let ffnHiddenDim: Int

        public static let base = Transformer(
            dim: 3840,
            nHeads: 30,
            headDim: 128,
            nLayers: 30,
            nRefinerLayers: 2,
            patchSize: 2,
            fPatchSize: 1,
            inChannels: 16,
            capFeatDim: 2560,
            normEps: 1e-5,
            ropeTheta: 256.0,
            tScale: 1000.0,
            axesDims: [32, 48, 48],
            axesLens: [1024, 512, 512],
            ffnHiddenDim: 10240  // int(3840 / 3 * 8)
        )
    }

    public struct VAE: Sendable {
        public let scalingFactor: Float
        public let shiftFactor: Float
        public let latentChannels: Int
        public let spatialScale: Int

        public static let base = VAE(
            scalingFactor: 0.3611,
            shiftFactor: 0.1159,
            latentChannels: 16,
            spatialScale: 8
        )
    }

    public struct Pipeline: Sendable {
        public let transformer: Transformer
        public let vae: VAE
        public let maxSequenceLength: Int
        public let numTrainTimesteps: Int
        public let defaultNumSteps: Int
        public let defaultGuidance: Float

        public static let base = Pipeline(
            transformer: .base,
            vae: .base,
            maxSequenceLength: 512,
            numTrainTimesteps: 1000,
            defaultNumSteps: 50,
            defaultGuidance: 4.0
        )
    }
}
