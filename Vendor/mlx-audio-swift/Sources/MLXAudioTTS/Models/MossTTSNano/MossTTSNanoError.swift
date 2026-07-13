import Foundation

public enum MossTTSNanoError: Error, CustomStringConvertible {
    case invalidInput(String)
    case notImplemented(String)
    case tokenizerNotInitialized
    case audioTokenizerNotInitialized

    public var description: String {
        switch self {
        case .invalidInput(let message):
            message
        case .notImplemented(let message):
            message
        case .tokenizerNotInitialized:
            "MOSS-TTS-Nano tokenizer is not initialized."
        case .audioTokenizerNotInitialized:
            "MOSS-TTS-Nano audio tokenizer is not initialized."
        }
    }
}
