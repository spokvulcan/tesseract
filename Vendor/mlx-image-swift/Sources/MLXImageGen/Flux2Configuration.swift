import Foundation
import MLX

public enum Flux2Configuration {
    public struct Transformer: Sendable {
        public let patchSize: Int
        public let inChannels: Int
        public let outChannels: Int
        public let numLayers: Int
        public let numSingleLayers: Int
        public let attentionHeadDim: Int
        public let numAttentionHeads: Int
        public let jointAttentionDim: Int
        public let timestepGuidanceChannels: Int
        public let mlpRatio: Float
        public let axesDimsRope: [Int]
        public let ropeTheta: Int
        public let guidanceEmbeds: Bool

        public var innerDim: Int { numAttentionHeads * attentionHeadDim }

        public static let klein4B = Transformer(
            patchSize: 1,
            inChannels: 128,
            outChannels: 128,
            numLayers: 5,
            numSingleLayers: 20,
            attentionHeadDim: 128,
            numAttentionHeads: 24,
            jointAttentionDim: 7680,
            timestepGuidanceChannels: 256,
            mlpRatio: 3.0,
            axesDimsRope: [32, 32, 32, 32],
            ropeTheta: 2000,
            guidanceEmbeds: false
        )
    }

    public struct TextEncoder: Sendable {
        public let vocabSize: Int
        public let hiddenSize: Int
        public let numHiddenLayers: Int
        public let numAttentionHeads: Int
        public let numKeyValueHeads: Int
        public let intermediateSize: Int
        public let maxPositionEmbeddings: Int
        public let ropeTheta: Float
        public let rmsNormEps: Float
        public let headDim: Int
        public let attentionBias: Bool
        public let attentionScaling: Float
        public let hiddenStateLayers: [Int]

        public static let klein4B = TextEncoder(
            vocabSize: 151936,
            hiddenSize: 2560,
            numHiddenLayers: 36,
            numAttentionHeads: 32,
            numKeyValueHeads: 8,
            intermediateSize: 9728,
            maxPositionEmbeddings: 40960,
            ropeTheta: 1_000_000.0,
            rmsNormEps: 1e-6,
            headDim: 128,
            attentionBias: false,
            attentionScaling: 1.0,
            hiddenStateLayers: [9, 18, 27]
        )
    }

    public struct VAE: Sendable {
        public let inChannels: Int
        public let outChannels: Int
        public let latentChannels: Int
        public let blockOutChannels: [Int]
        public let layersPerBlock: Int
        public let normNumGroups: Int
        public let scaleFactor: Int

        public static let flux2 = VAE(
            inChannels: 32,
            outChannels: 3,
            latentChannels: 32,
            blockOutChannels: [128, 256, 512, 512],
            layersPerBlock: 2,
            normNumGroups: 32,
            scaleFactor: 8
        )
    }

    public struct Pipeline: Sendable {
        public let transformer: Transformer
        public let textEncoder: TextEncoder
        public let vae: VAE
        public let maxSequenceLength: Int
        public let numTrainTimesteps: Int

        public static let klein4B = Pipeline(
            transformer: .klein4B,
            textEncoder: .klein4B,
            vae: .flux2,
            maxSequenceLength: 512,
            numTrainTimesteps: 1000
        )
    }
}
