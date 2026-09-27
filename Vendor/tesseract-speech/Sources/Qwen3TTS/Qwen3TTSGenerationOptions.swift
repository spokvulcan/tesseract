import Foundation

/// How the talker and the code predictor pick their tokens.
///
/// The talker samples the first codebook of each 80 ms frame, which carries
/// the words and the prosody: its temperature is how expressive the reading
/// is. The code predictor fills in the other codebooks, the acoustic detail
/// where most of the timbre lives: a lower detail temperature keeps the voice
/// steadier without flattening the delivery (research/voice-consistency-
/// 2026-09-26, ADR-0072). The order matches Qwen's reference sampler:
/// temperature first, then top-k, top-p and min-p; the repetition penalty
/// looks only at the most recent talker tokens.
public struct Qwen3TTSSampling: Sendable, Equatable {
    public var temperature: Float
    public var topK: Int
    public var topP: Float
    public var minP: Float
    public var repetitionPenalty: Float
    /// How many of the latest talker tokens the repetition penalty counts.
    public var repetitionWindow: Int
    public var detailTemperature: Float
    public var detailTopK: Int
    public var detailTopP: Float
    /// Upper bound on frames per generation; the model also caps it at six
    /// frames per text token.
    public var maxTokens: Int

    public init(
        temperature: Float = 0.9,
        topK: Int = 50,
        topP: Float = 1.0,
        minP: Float = 0,
        repetitionPenalty: Float = 1.05,
        repetitionWindow: Int = 64,
        detailTemperature: Float = 0.5,
        detailTopK: Int = 50,
        detailTopP: Float = 1.0,
        maxTokens: Int = 4096
    ) {
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.repetitionWindow = repetitionWindow
        self.detailTemperature = detailTemperature
        self.detailTopK = detailTopK
        self.detailTopP = detailTopP
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
