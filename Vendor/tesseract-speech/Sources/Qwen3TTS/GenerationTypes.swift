import Foundation
@preconcurrency import MLX

/// Statistics for one finished generation.
public struct AudioGenerationInfo: Sendable {
    public let promptTokenCount: Int
    public let generationTokenCount: Int
    public let prefillTime: TimeInterval
    public let generateTime: TimeInterval
    public let tokensPerSecond: Double
    public let peakMemoryUsage: Double

    public init(
        promptTokenCount: Int,
        generationTokenCount: Int,
        prefillTime: TimeInterval,
        generateTime: TimeInterval,
        tokensPerSecond: Double,
        peakMemoryUsage: Double
    ) {
        self.promptTokenCount = promptTokenCount
        self.generationTokenCount = generationTokenCount
        self.prefillTime = prefillTime
        self.generateTime = generateTime
        self.tokensPerSecond = tokensPerSecond
        self.peakMemoryUsage = peakMemoryUsage
    }
}

/// What a generation stream yields.
public enum AudioGeneration: Sendable {
    /// The first-codebook token sampled at a step.
    case token(Int)
    /// Statistics, once the generation ends.
    case info(AudioGenerationInfo)
    /// Decoded audio: a chunk when streaming, else the whole utterance.
    case audio(MLXArray)
}

public enum AudioGenerationError: Error, LocalizedError {
    case modelNotInitialized(String)
    case invalidInput(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotInitialized(let message):
            return "Model not initialized: \(message)"
        case .invalidInput(let message):
            return "Invalid input: \(message)"
        }
    }
}
