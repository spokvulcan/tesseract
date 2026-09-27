import Foundation

/// What a generation stream yields.
public enum AudioGeneration: Sendable {
    /// A chunk of decoded audio: 24 kHz mono samples in [-1, 1].
    case audio([Float])
    /// Once, after the last chunk: the codec frames the generation rendered,
    /// one `[Int32]` of `numCodeGroups` codes per frame. A reference take
    /// keeps them.
    case codeFrames([[Int32]])
    /// Once, before any `alignment`, when the generation was asked to follow
    /// an alignment head: the text track that head looks at.
    case textTrack(Qwen3TTSTextTrack)
    /// One per rendered frame, in order and ahead of that frame's audio: the
    /// alignment head's attention logits over the text track, from the query
    /// that predicted the frame. `-infinity` where a position wasn't in the
    /// prompt yet (the streaming-text layout's later text).
    case alignment([Float])
}

/// A talker attention head that follows the text while the voice speaks it:
/// its attention sits on the token being said, frame by frame (ADR-0077).
/// A property of the network, found by measurement: the 1.7B checkpoints'
/// is layer 3, head 0; the 0.6B's is layer 6, head 5
/// (docs/research/2026-09-27-word-timing-from-attention.md).
public struct Qwen3TTSAlignmentHead: Sendable, Equatable {
    public let layer: Int
    public let head: Int

    public init(layer: Int, head: Int) {
        self.layer = layer
        self.head = head
    }
}

/// The text an alignment head reads, as positions of the prompt: a take's
/// text first (in-context layout), then the new text, then TTS EOS.
public struct Qwen3TTSTextTrack: Sendable, Equatable {
    /// How many leading positions are the take's text.
    public let referenceTokenCount: Int
    /// Where each token of the new text starts in it, in characters.
    public let characterOffsets: [Int]

    public init(referenceTokenCount: Int, characterOffsets: [Int]) {
        self.referenceTokenCount = referenceTokenCount
        self.characterOffsets = characterOffsets
    }

    /// Positions in an `alignment` row: the take's tokens, the new text's,
    /// and EOS.
    public var width: Int { referenceTokenCount + characterOffsets.count + 1 }
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
