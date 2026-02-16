import Foundation
import MLX

public struct FlowMatchEulerScheduler {
    public let numInferenceSteps: Int
    public let sigmas: MLXArray
    public let timesteps: MLXArray

    private static let numTrainTimesteps = 1000

    public init(numInferenceSteps: Int, imageSeqLen: Int) {
        self.numInferenceSteps = numInferenceSteps
        let (ts, sigs) = Self.getTimestepsAndSigmas(
            imageSeqLen: imageSeqLen,
            numInferenceSteps: numInferenceSteps
        )
        self.timesteps = ts
        self.sigmas = sigs
    }

    public func step(noise: MLXArray, timestepIndex: Int, latents: MLXArray) -> MLXArray {
        let dt = (sigmas[timestepIndex + 1] - sigmas[timestepIndex]).asType(latents.dtype)
        return latents + dt * noise.asType(latents.dtype)
    }

    static func getTimestepsAndSigmas(
        imageSeqLen: Int,
        numInferenceSteps: Int,
        numTrainTimesteps: Int = 1000
    ) -> (timesteps: MLXArray, sigmas: MLXArray) {
        // Linearly spaced sigmas from 1.0 to 1/steps
        let sigmaValues = (0..<numInferenceSteps).map { i in
            Float(1.0) - Float(i) * (Float(1.0) - Float(1.0) / Float(numInferenceSteps)) / Float(numInferenceSteps - 1)
        }
        var sigmas = MLXArray(sigmaValues)

        let mu = computeEmpiricalMu(imageSeqLen: imageSeqLen, numSteps: numInferenceSteps)
        sigmas = timeShiftExponentialArray(mu: mu, sigmaPower: 1.0, t: sigmas)

        let timesteps = sigmas * Float(numTrainTimesteps)
        // Append zero sigma at the end
        sigmas = MLX.concatenated([sigmas, MLXArray.zeros([1])], axis: 0)
        return (timesteps, sigmas)
    }

    static func computeEmpiricalMu(imageSeqLen: Int, numSteps: Int) -> Float {
        let a1: Float = 8.73809524e-05
        let b1: Float = 1.89833333
        let a2: Float = 0.00016927
        let b2: Float = 0.45666666

        if imageSeqLen > 4300 {
            return a2 * Float(imageSeqLen) + b2
        }
        let m200 = a2 * Float(imageSeqLen) + b2
        let m10 = a1 * Float(imageSeqLen) + b1
        let a = (m200 - m10) / 190.0
        let b = m200 - 200.0 * a
        return a * Float(numSteps) + b
    }

    static func timeShiftExponentialArray(mu: Float, sigmaPower: Float, t: MLXArray) -> MLXArray {
        let expMu = MLXArray(exp(mu))
        return expMu / (expMu + MLX.pow(1.0 / t - 1.0, MLXArray(sigmaPower)))
    }
}
