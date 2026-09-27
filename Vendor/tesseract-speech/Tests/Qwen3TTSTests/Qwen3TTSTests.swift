// Qwen3-TTS model tests on tiny random-weight models and fixed logits: no
// checkpoint, but MLX needs Metal, so run them through xcodebuild
// (docs/testing.md).

import Foundation
@preconcurrency import MLX
import MLXLMCommon
import Testing
import Tokenizers

@testable import Qwen3TTS

@Suite("Qwen3TTS")
struct Qwen3TTSTests {

    @Test func customVoicePromptSplitsSpeakerAndInstruction() {
        let combined = Qwen3TTSModel.parseCustomVoicePrompt("Vivian, very happy and excited.")
        #expect(combined?.speaker == "Vivian")
        #expect(combined?.instruction == "very happy and excited.")

        let speakerOnly = Qwen3TTSModel.parseCustomVoicePrompt(" Vivian ")
        #expect(speakerOnly?.speaker == "Vivian")
        #expect(speakerOnly?.instruction == nil)

        #expect(Qwen3TTSModel.parseCustomVoicePrompt(nil)?.speaker == nil)
        #expect(Qwen3TTSModel.parseCustomVoicePrompt("   ")?.speaker == nil)
    }

    @Test func voiceDesignGeneratesAndStreams() async throws {
        let fixture = try await TinyModel.make(ttsModelType: "voice_design")
        defer { fixture.cleanUp() }
        let parameters = GenerateParameters(
            maxTokens: 2, temperature: 0.7, topP: 0.95, repetitionPenalty: 1.0)

        let audio = try await fixture.model.generate(
            text: "target voice prompt one two three four five",
            voice: "sample voice", language: "English", generationParameters: parameters)
        #expect(audio.ndim == 1)
        #expect(audio.shape[0] > 0)

        let streamed = try await collect(
            fixture.model.generateStream(
                text: "target voice prompt one two three four five",
                voice: "sample voice", language: "English",
                generationParameters: parameters, streamingInterval: 0.05))
        #expect(streamed.tokenCount > 0)
        #expect(streamed.infoCount == 1)
        #expect(streamed.lastAudio?.ndim == 1)
    }

    @Test func customVoiceSpeakerGenerates() async throws {
        let fixture = try await TinyModel.make(
            ttsModelType: "custom_voice",
            speakerIDEntries: #""ryan": 100"#,
            dialectEntries: #""ryan": false"#)
        defer { fixture.cleanUp() }

        let audio = try await fixture.model.generate(
            text: "target voice prompt one two three four five",
            voice: "ryan", language: "English",
            generationParameters: GenerateParameters(
                maxTokens: 2, temperature: 0.7, topP: 0.95, repetitionPenalty: 1.0))
        #expect(audio.ndim == 1)
        #expect(audio.shape[0] > 0)
    }

    /// The engine loads VoiceDesign and CustomVoice checkpoints only; a Base
    /// checkpoint fails before any weights are read.
    @Test func baseCheckpointsAreRejected() async throws {
        let directory = try makeTemporaryDirectory("qwen3-tts-base")
        defer { try? FileManager.default.removeItem(at: directory) }
        try #"{"model_type": "qwen3_tts", "tts_model_type": "base"}"#
            .write(to: directory.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)

        await #expect(throws: AudioGenerationError.self) {
            _ = try await Qwen3TTSModel.fromModelDirectory(directory)
        }
    }

    /// The codec's encoder (reference-audio cloning only, ~225 MB of F32 in
    /// the shipped checkpoints) is dropped before loading, never read.
    @Test func speechTokenizerSanitizeDropsTheEncoder() {
        let weights: [String: MLXArray] = [
            "encoder.encoder.layers.0.conv.weight": MLXArray.zeros([4, 1, 7]),
            "encoder.quantizer.rvq_first.input_proj.weight": MLXArray.zeros([4, 4, 1]),
            "encoder_model.downsample.conv.conv.weight": MLXArray.zeros([4, 4, 2]),
            "decoder.pre_transformer.norm.weight": MLXArray.zeros([4]),
        ]
        let sanitized = Qwen3TTSSpeechTokenizer.sanitize(weights: weights)
        #expect(Set(sanitized.keys) == ["decoder.pre_transformer.norm.weight"])
    }

    // EOS is filtered like every other token. Upstream wrote EOS's pre-filter
    // logit back after top-k/top-p/min-p, so EOS kept a chance on every talker
    // step and could end long-form audio mid-syllable. Python mlx-audio made
    // the same fix:
    // https://github.com/Blaizzy/mlx-audio/commit/d2d02bf36148d68198fb3d314d65c4d4f5340752
    // In this row EOS holds 18% of the mass but sits outside the 0.8 nucleus,
    // which tokens 7 (49%) and 8 (33%) close.
    private static let eosTokenID = 3050

    private static func eosJustOutsideTheNucleus() -> [Float] {
        var row = [Float](repeating: -20, count: 3072)
        row[7] = 3.0
        row[8] = 2.6
        row[eosTokenID] = 2.0
        return row
    }

    @Test func eosOutsideTheTopPNucleusIsFilteredToNegativeInfinity() {
        let row = Self.eosJustOutsideTheNucleus()
        let filtered = Qwen3TTSModel.filterLogits(
            MLXArray(row).reshaped(1, row.count), topK: 0, topP: 0.8, minP: 0
        ).asArray(Float.self)

        #expect(filtered[Self.eosTokenID] == -Float.infinity)
        #expect(filtered.indices.filter { filtered[$0].isFinite } == [7, 8])
    }

    @Test func eosOutsideTheTopPNucleusIsNeverSampled() async throws {
        let fixture = try await TinyModel.make(ttsModelType: "voice_design")
        defer { fixture.cleanUp() }

        // The talker's call with the Voice Engine's settings: t=0.6, top-p 0.8,
        // top-k off, penalty 1.3, the special tokens other than EOS suppressed.
        let row = Self.eosJustOutsideTheNucleus()
        let logits = MLXArray(row).reshaped(1, 1, row.count)
        let suppressTokens = (2048 ..< row.count).filter { $0 != Self.eosTokenID }
        MLXRandom.seed(0)
        var draws: [Int: Int] = [:]
        for _ in 0 ..< 200 {
            let token = fixture.model.sampleToken(
                logits,
                temperature: 0.6,
                topP: 0.8,
                topK: 0,
                repetitionPenalty: 1.3,
                generatedTokens: [],
                suppressTokens: suppressTokens,
                minP: 0
            )
            draws[Int(token[0, 0].item(Int32.self)), default: 0] += 1
        }

        #expect(draws[Self.eosTokenID] == nil)
        #expect(Set(draws.keys) == [7, 8])
    }
}

// MARK: - Fixtures

struct TinyModel {
    let model: Qwen3TTSModel
    let tokenizerDirectory: URL

    func cleanUp() {
        try? FileManager.default.removeItem(at: tokenizerDirectory)
    }

    /// Two talker layers of width 16, two code groups, random weights, and a
    /// default-shaped decoder: enough to run every generation path.
    static func make(
        ttsModelType: String,
        speakerIDEntries: String = "",
        dialectEntries: String = ""
    ) async throws -> TinyModel {
        let tokenizerDirectory = try makeTinyTokenizerDirectory()
        let spkIdJSON = speakerIDEntries.isEmpty ? "" : #""spk_id": {"# + speakerIDEntries + "},"
        let spkDialectJSON =
            dialectEntries.isEmpty ? "" : #""spk_is_dialect": {"# + dialectEntries + "},"
        let configJSON = """
            {
              "model_type": "qwen3_tts",
              "tts_model_type": "\(ttsModelType)",
              "tts_model_size": "tiny",
              "tokenizer_type": "qwen3_tts_tokenizer_12hz",
              "im_start_token_id": 4,
              "im_end_token_id": 5,
              "tts_pad_token_id": 21,
              "tts_bos_token_id": 22,
              "tts_eos_token_id": 23,
              "sample_rate": 24000,
              "talker_config": {
                "vocab_size": 3072,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "head_dim": 4,
                "max_position_embeddings": 128,
                "num_code_groups": 2,
                "text_hidden_size": 16,
                "text_vocab_size": 64,
                "codec_eos_token_id": 3050,
                "codec_think_id": 3051,
                "codec_nothink_id": 3052,
                "codec_think_bos_id": 3053,
                "codec_think_eos_id": 3054,
                "codec_pad_id": 3055,
                "codec_bos_id": 3056,
                "codec_language_id": {
                  "english": 3057
                },
                \(spkIdJSON)
                \(spkDialectJSON)
                "code_predictor_config": {
                  "vocab_size": 2048,
                  "hidden_size": 16,
                  "intermediate_size": 32,
                  "num_hidden_layers": 1,
                  "num_attention_heads": 4,
                  "num_key_value_heads": 4,
                  "head_dim": 4,
                  "max_position_embeddings": 128,
                  "num_code_groups": 2
                }
              },
              "tokenizer_config": {
                "decoder_config": {}
              }
            }
            """

        let config = try JSONDecoder().decode(Qwen3TTSModelConfig.self, from: Data(configJSON.utf8))
        let model = Qwen3TTSModel(config: config)
        model.tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerDirectory)
        model.speechTokenizer = Qwen3TTSSpeechTokenizer(config: try #require(config.tokenizerConfig))
        return TinyModel(model: model, tokenizerDirectory: tokenizerDirectory)
    }
}

func makeTemporaryDirectory(_ prefix: String) throws -> URL {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("\(prefix)-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    return directory
}

private func makeTinyTokenizerDirectory() throws -> URL {
    let directory = try makeTemporaryDirectory("tiny-qwen3-tokenizer")

    let tokenizerConfig = """
        {
          "tokenizer_class": "GPT2Tokenizer",
          "bos_token": "<bos>",
          "eos_token": "<eos>",
          "unk_token": "<unk>",
          "pad_token": "<pad>",
          "model_max_length": 128,
          "do_lower_case": false
        }
        """

    let tokenizerData = """
        {
          "version": "1.0",
          "truncation": null,
          "padding": null,
          "added_tokens": [
            { "id": 0, "content": "<bos>", "special": true },
            { "id": 1, "content": "<pad>", "special": true },
            { "id": 2, "content": "<eos>", "special": true },
            { "id": 3, "content": "<unk>", "special": true },
            { "id": 4, "content": "<|im_start|>", "special": true },
            { "id": 5, "content": "<|im_end|>", "special": true }
          ],
          "model": {
            "type": "BPE",
            "vocab": {
              "<bos>": 0,
              "<pad>": 1,
              "<eos>": 2,
              "<unk>": 3,
              "<|im_start|>": 4,
              "<|im_end|>": 5,
              "assistant": 6,
              "user": 7,
              "one": 8,
              "two": 9,
              "three": 10,
              "four": 11,
              "five": 12,
              "target": 13,
              "voice": 14,
              "prompt": 15,
              "sample": 16,
              "english": 17
            },
            "merges": [],
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "unk_token": "<unk>"
          },
          "normalizer": {
            "type": "Lowercase"
          },
          "pre_tokenizer": {
            "type": "Whitespace"
          }
        }
        """

    try tokenizerConfig.write(
        to: directory.appendingPathComponent("tokenizer_config.json"), atomically: true,
        encoding: .utf8)
    try tokenizerData.write(
        to: directory.appendingPathComponent("tokenizer.json"), atomically: true, encoding: .utf8)
    return directory
}

func collect(
    _ stream: AsyncThrowingStream<AudioGeneration, Error>
) async throws -> (tokenCount: Int, infoCount: Int, lastAudio: MLXArray?) {
    var tokenCount = 0
    var infoCount = 0
    var lastAudio: MLXArray?
    for try await event in stream {
        switch event {
        case .token: tokenCount += 1
        case .info: infoCount += 1
        case .audio(let audio): lastAudio = audio
        }
    }
    return (tokenCount, infoCount, lastAudio)
}
