import Foundation
@preconcurrency import MLX
import Tokenizers

/// How the talker reads the text it speaks. Both layouts are Qwen's.
public enum Qwen3TTSTextLayout: Sendable, Equatable {
    /// Qwen's streaming-text mode: the first text token rides with codec BOS
    /// and each later one with a generated frame, as text arriving live
    /// would. What the engine has always run (Qwen's `non_streaming_mode =
    /// False`; also what mlx-audio does).
    case interleaved
    /// Qwen's non-streaming mode, the default of its VoiceDesign and
    /// CustomVoice APIs: the whole text sits in the prompt, over codec pad,
    /// before the first frame.
    case upfront
}

/// A talker prompt, split where the voice prefix cache splits it.
struct Qwen3TTSPrompt {
    /// The description's user turn. Causal attention means it reads the same
    /// whatever follows, so its KV is cached per description.
    var instruct: MLXArray?
    /// Everything after the instruct turn, through the position the first
    /// frame is predicted from.
    var body: MLXArray
    /// Text embeds the interleaved layout feeds one per frame; after them,
    /// and throughout the other layouts, each frame gets `pad`.
    var trailingText: MLXArray?
    var pad: MLXArray
    /// The spoken text's tokens, which cap how many frames it can take.
    var textTokenCount: Int

    var trailingCount: Int { trailingText?.dim(1) ?? 0 }

    /// The text track's input for frame `t`: its trailing text token, then
    /// pad.
    func text(forFrame t: Int) -> MLXArray {
        t < trailingCount ? trailingText![0..., t ..< (t + 1), 0...] : pad
    }
}

/// Lays out talker prompts the way Qwen's `generate` does: a text track and a
/// codec track summed position by position.
///
/// Every prompt is `[instruct] role codecPrefix …`: the optional user turn,
/// `<|im_start|>assistant\n`, then the codec tags (think/nothink, the
/// language, a CustomVoice speaker, pad, BOS) under a text track of pads
/// ending in TTS BOS. What follows depends on the layout.
final class Qwen3TTSPromptBuilder {
    private let talkerConfig: Qwen3TTSTalkerConfig
    private let tokenizer: Tokenizer
    private let talker: Qwen3TTSTalker
    private let textEmbedding: Qwen3TTSTextEmbedding

    /// TTS BOS/EOS/PAD on the text track, `[1, 1, D]` each.
    private let ttsBos: MLXArray
    private let ttsEos: MLXArray
    let ttsPad: MLXArray

    init(
        config: Qwen3TTSModelConfig, tokenizer: Tokenizer, talker: Qwen3TTSTalker,
        textEmbedding: Qwen3TTSTextEmbedding
    ) throws {
        self.talkerConfig = config.talkerConfig ?? .defaults
        self.tokenizer = tokenizer
        self.talker = talker
        self.textEmbedding = textEmbedding
        let special = talker.textProjection(
            try textEmbedding([
                Int32(config.ttsBosTokenId), Int32(config.ttsEosTokenId),
                Int32(config.ttsPadTokenId),
            ]))
        ttsBos = special[0..., 0 ..< 1, 0...]
        ttsEos = special[0..., 1 ..< 2, 0...]
        ttsPad = special[0..., 2 ..< 3, 0...]
        eval(ttsBos, ttsEos, ttsPad)
    }

    // MARK: - Pieces

    private func tokens(_ text: String) -> [Int32] {
        tokenizer.encode(text: text).map { Int32($0) }
    }

    /// Text tokens on the text track: embedded, then projected to the
    /// talker's width.
    private func embedText(_ ids: [Int32]) throws -> MLXArray {
        talker.textProjection(try textEmbedding(ids))
    }

    private func codec(_ ids: [Int]) -> MLXArray {
        talker.embedCodec(MLXArray(ids.map { Int32($0) }).reshaped(1, -1))
    }

    /// The codec tags: think + language, or nothink; a speaker; pad, BOS.
    private func codecPrefix(languageID: Int?, speaker: MLXArray?) -> MLXArray {
        let t = talkerConfig
        let tags =
            languageID.map { [t.codecThinkId, t.codecThinkBosId, $0, t.codecThinkEosId] }
            ?? [t.codecNothinkId, t.codecThinkBosId, t.codecThinkEosId]
        return concatenated(
            [codec(tags)] + [speaker].compactMap { $0 } + [codec([t.codecPadId, t.codecBosId])],
            axis: 1)
    }

    /// Pads, then TTS BOS, over every codec tag but the last (codec BOS).
    private func prefixTrack(_ codecPrefix: MLXArray) -> MLXArray {
        let n = codecPrefix.dim(1)
        let text = concatenated(
            [broadcast(ttsPad, to: [1, n - 2, ttsPad.dim(-1)]), ttsBos], axis: 1)
        return text + codecPrefix[0..., ..<(n - 1), 0...]
    }

    private func overCodecPad(_ text: MLXArray) -> MLXArray {
        text + codec([talkerConfig.codecPadId])
    }

    /// The codec token for `language`; nil for "auto" or one the checkpoint
    /// doesn't list. A CustomVoice speaker with a dialect speaks it when the
    /// language is Chinese or auto (Qwen's rule).
    func languageID(_ language: String?, speaker: String? = nil) -> Int? {
        let name = (language ?? "auto").lowercased()
        if let speaker, ["chinese", "auto"].contains(name),
            let dialect = talkerConfig.spkIsDialect?[speaker.lowercased()]?.dialectName,
            let dialectID = talkerConfig.codecLanguageId?[dialect]
        {
            return dialectID
        }
        guard name != "auto" else { return nil }
        return talkerConfig.codecLanguageId?[name]
    }

    // MARK: - Prompts

    /// The user turn for a voice description (or CustomVoice instruction).
    func instruct(_ description: String?) throws -> MLXArray? {
        guard let description, !description.isEmpty else { return nil }
        return try embedText(tokens("<|im_start|>user\n\(description)<|im_end|>\n"))
    }

    /// A prompt from the description alone. `speaker` selects a CustomVoice
    /// speaker, whose dialect replaces the language when the language is
    /// Chinese or auto (Qwen's rule).
    func plain(
        text: String, instruct: String?, language: String?, speaker: String?,
        layout: Qwen3TTSTextLayout
    ) throws -> Qwen3TTSPrompt {
        let ids = tokens("<|im_start|>assistant\n\(text)<|im_end|>\n<|im_start|>assistant\n")
        // <|im_start|> assistant \n … <|im_end|> \n <|im_start|> assistant \n
        let textEnd = ids.count - 5
        guard textEnd > 3 else {
            throw AudioGenerationError.invalidInput("There is no text to speak.")
        }
        let embeds = try embedText(ids)

        var speakerEmbed: MLXArray?
        if let speaker {
            guard let id = talkerConfig.spkId?[speaker.lowercased()]?.intValue else {
                throw AudioGenerationError.invalidInput(
                    "This checkpoint has no speaker named \(speaker).")
            }
            speakerEmbed = codec([id])
        }

        let codecTags = codecPrefix(
            languageID: languageID(language, speaker: speaker), speaker: speakerEmbed)
        let head = [embeds[0..., ..<3, 0...], prefixTrack(codecTags)]
        let codecBos = codecTags[0..., (-1)..., 0...]
        var prompt = Qwen3TTSPrompt(
            instruct: try self.instruct(instruct), body: embeds, trailingText: nil, pad: ttsPad,
            textTokenCount: textEnd - 3)
        switch layout {
        case .interleaved:
            prompt.body = concatenated(head + [embeds[0..., 3 ..< 4, 0...] + codecBos], axis: 1)
            prompt.trailingText = concatenated(
                [embeds[0..., 4 ..< textEnd, 0...], ttsEos], axis: 1)
        case .upfront:
            let spoken = overCodecPad(
                concatenated([embeds[0..., 3 ..< textEnd, 0...], ttsEos], axis: 1))
            prompt.body = concatenated(head + [spoken, ttsPad + codecBos], axis: 1)
        }
        return prompt
    }

    /// A prompt that continues a Reference Take in Qwen's in-context layout
    /// (its `non_streaming_mode` form): the take's text then the new text,
    /// then EOS, all over codec pad; then codec BOS and the take's frames
    /// under text pad. Generation continues the frames with the new words.
    func reference(
        text: String, take: Qwen3TTSReference, instruct: String?, language: String?
    ) throws -> Qwen3TTSPrompt {
        let groups = talkerConfig.numCodeGroups
        guard !take.codeFrames.isEmpty, take.codeFrames.allSatisfy({ $0.count == groups }) else {
            throw AudioGenerationError.invalidInput(
                "A reference take needs at least one frame of \(groups) codes per frame.")
        }
        let target = tokens("<|im_start|>assistant\n\(text)<|im_end|>\n<|im_start|>assistant\n")
        let referenceIDs = tokens("<|im_start|>assistant\n\(take.text)<|im_end|>\n")
        let targetEnd = target.count - 5
        guard targetEnd > 3 else {
            throw AudioGenerationError.invalidInput("There is no text to speak.")
        }
        // One lookup for the role, the take's text and the new text.
        let spokenIDs = Array(referenceIDs[min(3, referenceIDs.count) ..< max(3, referenceIDs.count - 2)])
        let embeds = try embedText(Array(target[..<3]) + spokenIDs + Array(target[3 ..< targetEnd]))

        let frames = MLXArray(take.codeFrames.flatMap { $0 }).reshaped(1, take.codeFrames.count, groups)
        var codes = talker.embedCodec(frames[0..., 0..., 0])
        for g in 1 ..< groups {
            codes = codes + talker.codePredictor.codecEmbedding[g - 1](frames[0..., 0..., g])
        }
        let codecTrack = concatenated([codec([talkerConfig.codecBosId]), codes], axis: 1) + ttsPad
        let textTrack = overCodecPad(concatenated([embeds[0..., 3..., 0...], ttsEos], axis: 1))

        let codecTags = codecPrefix(languageID: languageID(language), speaker: nil)
        return Qwen3TTSPrompt(
            instruct: try self.instruct(instruct),
            body: concatenated(
                [embeds[0..., ..<3, 0...], prefixTrack(codecTags), textTrack, codecTrack], axis: 1),
            trailingText: nil, pad: ttsPad, textTokenCount: targetEnd - 3)
    }
}
