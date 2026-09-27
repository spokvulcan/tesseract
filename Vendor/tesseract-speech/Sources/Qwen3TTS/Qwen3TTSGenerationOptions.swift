import Foundation

/// How the talker and the code predictor pick their tokens.
///
/// The talker samples the first codebook of each 80 ms frame, which carries
/// the words and the prosody: its temperature is how expressive the reading
/// is. The code predictor fills in the other codebooks, the acoustic detail
/// where most of the timbre lives: a lower detail temperature keeps the voice
/// steadier without flattening the delivery (research/voice-consistency-
/// 2026-09-26, ADR-0072). The order matches Qwen's reference sampler:
/// temperature first, then top-k and top-p; the repetition penalty looks only
/// at the most recent talker tokens.
public struct Qwen3TTSSampling: Sendable, Equatable {
    public var temperature: Float
    public var topP: Float
    public var repetitionPenalty: Float
    public var detailTemperature: Float
    /// Upper bound on frames per generation; the model also caps it at six
    /// frames per text token.
    public var maxTokens: Int

    /// Qwen's reference values for what the app doesn't tune: top-k for both
    /// models, the code predictor's top-p, and how many of the latest talker
    /// tokens the repetition penalty counts.
    static let topK = 50
    static let detailTopP: Float = 1.0
    static let repetitionWindow = 64

    public init(
        temperature: Float = 0.9,
        topP: Float = 1.0,
        repetitionPenalty: Float = 1.05,
        detailTemperature: Float = 0.5,
        maxTokens: Int = 4096
    ) {
        self.temperature = temperature
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.detailTemperature = detailTemperature
        self.maxTokens = maxTokens
    }
}

/// Speech this model rendered, kept so later generations continue in the
/// same voice: the codec frames (one row of `numCodeGroups` codes per 80 ms)
/// and the text they speak. Generation places it with Qwen's in-context
/// layout, inside the assistant turn, the way a clone prompt carries its
/// reference clip. The frames come from our own generation, so no audio is
/// ever encoded.
public struct Qwen3TTSReference: Sendable, Equatable {
    public let codeFrames: [[Int32]]
    public let text: String

    public init(codeFrames: [[Int32]], text: String) {
        self.codeFrames = codeFrames
        self.text = text
    }
}
