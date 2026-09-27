import Foundation
@preconcurrency import MLX

/// What a generation stream yields.
public enum AudioGeneration: Sendable {
    /// A chunk of decoded audio.
    case audio(MLXArray)
    /// Once, after the last chunk: the codec frames the generation rendered,
    /// one `[Int32]` of `numCodeGroups` codes per frame. A reference take
    /// keeps them.
    case codeFrames([[Int32]])
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
