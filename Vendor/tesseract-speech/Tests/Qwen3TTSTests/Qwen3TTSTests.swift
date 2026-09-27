// Qwen3-TTS model tests on tiny random-weight models and fixed logits: no
// checkpoint, but MLX needs Metal, so run them through xcodebuild
// (docs/testing.md).

import Foundation
@preconcurrency import MLX
@preconcurrency import MLXLMCommon
import MLXNN
import Testing
import Tokenizers

@testable import Qwen3TTS

@Suite("Qwen3TTS", .serialized)
struct Qwen3TTSTests {

    // MARK: - Generation

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

        let streamed = try await collect(
            fixture.model.generateStream(
                text: "target voice prompt one two three four five",
                voice: "sample voice", language: "English",
                sampling: TinyModel.sampling, streamingInterval: 0.05))
        #expect(streamed.chunks > 0)
        #expect(streamed.codeFrames?.isEmpty == false, "the frames follow the audio")
        #expect(streamed.lastAudio?.isEmpty == false)
        let frames = try #require(streamed.codeFrames)
        #expect(
            streamed.allAudio.count == frames.count * fixture.model.samplesPerFrame,
            "every accepted frame is decoded, once")
    }

    /// A render is reproducible from its seed, whatever the chunking.
    @Test func aSeedReproducesTheRenderAcrossChunkings() async throws {
        let fixture = try await TinyModel.make(ttsModelType: "voice_design")
        defer { fixture.cleanUp() }
        let sampling = Qwen3TTSSampling(temperature: 0.9, topP: 1, maxTokens: 12)
        var renders: [([Float], [[Int32]])] = []
        // The tiny decoder runs 250 frames a second: chunks of 1, 2 and 5.
        for interval in [0.004, 0.008, 0.02] {
            let r = try await collect(
                fixture.model.generateStream(
                    text: "one two three four five", voice: "sample voice", language: "English",
                    sampling: sampling, seed: 11, streamingInterval: interval))
            renders.append((r.allAudio, try #require(r.codeFrames)))
        }
        for r in renders.dropFirst() {
            #expect(r.1 == renders[0].1, "same seed, same frames")
            #expect(r.0.count == renders[0].0.count)
            #expect(maxAbsDifference(r.0, renders[0].0) < 1e-3, "chunking never changes the audio")
        }
    }

    /// Generations on one model never overlap: a second one waits for the
    /// first, and each renders what it renders alone. (A cancelled stream's
    /// generation runs on to the end of its frame after its consumer and the
    /// engine's GPU lease have gone; the next one must not rewind the kept
    /// KV caches under it.)
    @Test func overlappingGenerationsRenderAsIfAlone() async throws {
        let fixture = try await TinyModel.make(ttsModelType: "voice_design")
        defer { fixture.cleanUp() }
        let model = fixture.model
        func stream(_ text: String, seed: UInt64) -> AsyncThrowingStream<AudioGeneration, Error> {
            model.generateStream(
                text: text, voice: "sample voice", language: "English",
                sampling: Qwen3TTSSampling(temperature: 0.9, topP: 1, maxTokens: 60), seed: seed,
                streamingInterval: 0.004)
        }
        let longText = "a longer line that goes on rendering for a good while after this"
        let shortText = "one two three four five"
        let longAlone = try await collect(stream(longText, seed: 9))
        let shortAlone = try await collect(stream(shortText, seed: 5))
        #expect((longAlone.codeFrames?.count ?? 0) > 10, "the long render outlasts its first chunk")

        // The long render is under way (its first chunk is out) when the
        // short one is asked for.
        var long = stream(longText, seed: 9).makeAsyncIterator()
        var longAudio: [Float] = []
        var longFrames: [[Int32]]?
        if case .audio(let audio) = try await long.next() { longAudio += audio }
        let short = try await collect(stream(shortText, seed: 5))
        while let event = try await long.next() {
            switch event {
            case .audio(let audio): longAudio += audio
            case .codeFrames(let frames): longFrames = frames
            case .textTrack, .alignment: break
            }
        }
        #expect(short.codeFrames == shortAlone.codeFrames)
        #expect(short.allAudio == shortAlone.allAudio)
        #expect(longFrames == longAlone.codeFrames)
        #expect(longAudio == longAlone.allAudio)
    }

    @Test func customVoiceSpeakerGenerates() async throws {
        let fixture = try await TinyModel.make(
            ttsModelType: "custom_voice",
            speakerIDEntries: #""ryan": 100"#,
            dialectEntries: #""ryan": false"#)
        defer { fixture.cleanUp() }

        let rendered = try await collect(
            fixture.model.generateStream(
                text: "target voice prompt one two three four five",
                voice: "ryan", language: "English", sampling: TinyModel.sampling))
        #expect(!rendered.allAudio.isEmpty)

        await #expect(throws: AudioGenerationError.self) {
            _ = try await collect(
                fixture.model.generateStream(
                    text: "one two", voice: "nobody", language: "English",
                    sampling: TinyModel.sampling))
        }
    }

    /// A dialect speaker speaks its dialect only when the language is
    /// Chinese or auto, as in Qwen's generate.
    @Test func aDialectReplacesOnlyChineseOrAuto() async throws {
        let fixture = try await TinyModel.make(
            ttsModelType: "custom_voice",
            speakerIDEntries: #""eric": 101"#,
            dialectEntries: #""eric": "sichuan_dialect""#,
            extraLanguages: #", "chinese": 3058, "sichuan_dialect": 3059"#)
        defer { fixture.cleanUp() }
        let prompts = fixture.model.prompts
        #expect(prompts.languageID("auto", speaker: "eric") == 3059)
        #expect(prompts.languageID("Chinese", speaker: "eric") == 3059)
        #expect(prompts.languageID("English", speaker: "eric") == 3057)
        #expect(prompts.languageID("auto") == nil)
    }

    /// A Reference Take is the frames a generation rendered plus the text it
    /// spoke; conditioning on it renders the next words in that voice.
    @Test func aReferenceTakeConditionsTheNextGeneration() async throws {
        let fixture = try await TinyModel.make(ttsModelType: "voice_design")
        defer { fixture.cleanUp() }

        let take = try await collect(
            fixture.model.generateStream(
                text: "one two three", voice: "sample voice", language: "English",
                sampling: TinyModel.sampling, streamingInterval: 0.05))
        let frames = try #require(take.codeFrames)
        #expect(!frames.isEmpty)
        #expect(frames.allSatisfy { $0.count == 2 }, "one code per group per frame")

        let streamed = try await collect(
            fixture.model.generateStream(
                text: "four five", voice: "sample voice", language: "English",
                reference: Qwen3TTSReference(codeFrames: frames, text: "one two three"),
                sampling: TinyModel.sampling, streamingInterval: 0.05))
        #expect(streamed.codeFrames != nil)
        #expect(streamed.lastAudio?.isEmpty == false)
    }

    /// The description leads every prompt as the user turn, the same
    /// embedding whatever follows, so one voice prefix cache serves every
    /// layout.
    @Test func everyPromptLeadsWithTheDescription() async throws {
        let fixture = try await TinyModel.make(ttsModelType: "voice_design")
        defer { fixture.cleanUp() }
        let prompts = fixture.model.prompts
        let take = Qwen3TTSReference(codeFrames: [[1, 2], [3, 4], [5, 6]], text: "one two")
        let reference = try prompts.reference(
            text: "three four", take: take, instruct: "sample voice", language: "English")
        let plain = try prompts.plain(
            text: "three four", instruct: "sample voice", language: "English", speaker: nil,
            layout: .interleaved)
        let instructTokens = fixture.model.tokenizer
            .encode(text: "<|im_start|>user\nsample voice<|im_end|>\n").count
        let a = try #require(reference.instruct)
        let b = try #require(plain.instruct)
        #expect(a.dim(1) == instructTokens)
        #expect((a .== b).all().item(Bool.self))

        let bare = try prompts.reference(
            text: "three four", take: take, instruct: nil, language: "English")
        #expect(bare.instruct == nil)
        #expect(bare.body.dim(1) == reference.body.dim(1))
    }

    /// Qwen's two text layouts hold the same text: interleaved feeds it one
    /// token per frame, upfront puts it all in the prompt (with EOS, then
    /// pad over codec BOS).
    @Test func theUpfrontLayoutPrefillsTheWholeText() async throws {
        let fixture = try await TinyModel.make(ttsModelType: "voice_design")
        defer { fixture.cleanUp() }
        let prompts = fixture.model.prompts
        let interleaved = try prompts.plain(
            text: "one two three four", instruct: nil, language: "English", speaker: nil,
            layout: .interleaved)
        let upfront = try prompts.plain(
            text: "one two three four", instruct: nil, language: "English", speaker: nil,
            layout: .upfront)
        #expect(interleaved.trailingCount == interleaved.textTokenCount)
        #expect(upfront.trailingCount == 0)
        #expect(upfront.body.dim(1) == interleaved.body.dim(1) + interleaved.trailingCount + 1)

        let rendered = try await collect(
            fixture.model.generateStream(
                text: "one two three four", voice: "sample voice", language: "English",
                sampling: TinyModel.sampling, layout: .upfront))
        #expect(!rendered.allAudio.isEmpty)
    }

    // MARK: - Word timing (ADR-0077)

    /// Both layouts place the text track right after the codec tags: the
    /// streaming layout's first token ends the prompt, and a take's text
    /// comes before the new text.
    @Test func thePromptKnowsWhereItsTextTrackSits() async throws {
        let fixture = try await TinyModel.make(ttsModelType: "voice_design")
        defer { fixture.cleanUp() }
        let prompts = fixture.model.prompts
        let plain = try prompts.plain(
            text: "one two three", instruct: nil, language: "English", speaker: nil,
            layout: .interleaved)
        #expect(plain.textSpan.lowerBound == plain.body.dim(1) - 1)
        #expect(plain.textSpan.count == plain.textTokenCount + 1)
        #expect(plain.targetTokens.count == plain.textTokenCount)
        #expect(plain.referenceTokenCount == 0)

        let take = Qwen3TTSReference(codeFrames: [[1, 2], [3, 4]], text: "four five")
        let reference = try prompts.reference(
            text: "one two three", take: take, instruct: nil, language: "English")
        #expect(reference.textSpan.lowerBound == plain.textSpan.lowerBound)
        #expect(
            reference.textSpan.count
                == reference.referenceTokenCount + reference.textTokenCount + 1)
        #expect(reference.targetTokens == plain.targetTokens)
        // After the text track: codec BOS and the take's two frames.
        #expect(reference.body.dim(1) == reference.textSpan.upperBound + 1 + 2)
    }

    /// The probe reads one head's logits for the last query over its span of
    /// cached keys, through the prompt's path and the fused one-position
    /// path alike.
    @Test func theProbeReadsOneHeadOverItsSpan() throws {
        let random = MLXRandom.RandomState(seed: 17)
        let (heads, kvHeads, headDim, hidden) = (4, 2, 16, 64)
        let (layers, _) = try randomDecoderLayers(
            1, hidden: hidden, heads: heads, kvHeads: kvHeads, headDim: headDim,
            intermediate: 64, scale: 0.05, random: random)
        let attention = layers[0].attention
        let cache = KVCacheSimple()
        let probe = Qwen3TTSAlignmentProbe(head: 3, span: 2 ..< 8)
        attention.alignmentProbe = probe

        func expected(_ normed: MLXArray, offset: Int) -> [Float] {
            let length = normed.dim(1)
            let split = attention.qkvProj!(normed).reshaped(1, length, heads + 2 * kvHeads, headDim)
            let queries = MLXFast.RoPE(
                attention.qNorm(split[0..., 0..., ..<heads, 0...]).transposed(0, 2, 1, 3),
                dimensions: headDim, traditional: false, base: attention.ropeBase, scale: 1,
                offset: offset)
            let keys = cache.state[0]
            let query = queries[0..., 3 ..< 4, (length - 1) ..< length, 0...].asType(.float32)
            // Head 3 of 4 reads KV head 1 of 2.
            let span = keys[0..., 1 ..< 2, 2 ..< min(8, keys.dim(2)), 0...].asType(.float32)
            return (matmul(query, span.transposed(0, 1, 3, 2)) * attention.scale).reshaped(-1)
                .asArray(Float.self)
        }

        let prompt = layers[0].inputNorm(
            (MLXRandom.normal([1, 10, hidden], key: random) * 0.5).asType(.bfloat16))
        _ = attention(prompt, cache: cache)
        let first = try #require(probe.scores).asArray(Float.self)
        #expect(first.count == 6)
        #expect(maxAbsDifference(first, expected(prompt, offset: 0)) < 1e-3)

        let step = layers[0].inputNorm(
            (MLXRandom.normal([1, 1, hidden], key: random) * 0.5).asType(.bfloat16))
        _ = attention(step, cache: cache)
        let next = try #require(probe.scores).asArray(Float.self)
        #expect(maxAbsDifference(next, expected(step, offset: 10)) < 1e-3)
    }

    /// With an alignment head the stream carries the text track once and a
    /// row per frame, each ahead of its frame's audio; the render itself is
    /// the same sample for sample.
    @Test func anAlignmentHeadAddsRowsAndChangesNoSample() async throws {
        let fixture = try await TinyModel.make(ttsModelType: "voice_design")
        defer { fixture.cleanUp() }
        func render(_ head: Qwen3TTSAlignmentHead?) async throws -> Collected {
            try await collect(
                fixture.model.generateStream(
                    text: "one two three four", voice: "sample voice", language: "English",
                    sampling: TinyModel.sampling, seed: 3, streamingInterval: 0.004,
                    alignment: head))
        }
        let plain = try await render(nil)
        let timed = try await render(Qwen3TTSAlignmentHead(layer: 1, head: 2))
        #expect(timed.codeFrames == plain.codeFrames)
        #expect(timed.allAudio == plain.allAudio)
        #expect(plain.textTrack == nil)
        #expect(plain.alignment.isEmpty)

        let prompt = try fixture.model.prompts.plain(
            text: "one two three four", instruct: "sample voice", language: "English",
            speaker: nil, layout: .interleaved)
        let track = try #require(timed.textTrack)
        let frames = try #require(timed.codeFrames)
        #expect(track.referenceTokenCount == 0)
        #expect(track.characterOffsets.count == prompt.textTokenCount)
        #expect(track.width == prompt.textSpan.count)
        #expect(timed.alignment.count == frames.count)
        #expect(timed.alignment.allSatisfy { $0.count == track.width })
        for (frame, samplesBefore) in timed.audioFramesAtRow.enumerated() {
            #expect(samplesBefore <= frame * fixture.model.samplesPerFrame)
        }
        // The streaming layout feeds the text with the frames: the first
        // row sees only the first token; EOS comes in last.
        #expect(timed.alignment[0][0].isFinite)
        #expect(timed.alignment[0][track.width - 1] == -.infinity)

        let missing = try await render(Qwen3TTSAlignmentHead(layer: 9, head: 0))
        #expect(missing.textTrack == nil, "a head the talker doesn't have is ignored")
    }

    /// Continuing a take, the rows cover the take's text before the new text.
    @Test func aContinuationsRowsCoverTheTakesText() async throws {
        let fixture = try await TinyModel.make(ttsModelType: "voice_design")
        defer { fixture.cleanUp() }
        let take = Qwen3TTSReference(codeFrames: [[1, 2], [3, 4], [5, 6]], text: "one two")
        let timed = try await collect(
            fixture.model.generateStream(
                text: "three four five", voice: "sample voice", language: "English",
                reference: take, sampling: TinyModel.sampling, seed: 4,
                alignment: Qwen3TTSAlignmentHead(layer: 0, head: 1)))
        let prompt = try fixture.model.prompts.reference(
            text: "three four five", take: take, instruct: "sample voice", language: "English")
        let track = try #require(timed.textTrack)
        #expect(track.referenceTokenCount == prompt.referenceTokenCount)
        #expect(track.width == prompt.textSpan.count)
        #expect(!timed.alignment.isEmpty)
        // The whole text track is in the prompt from the start.
        #expect(
            timed.alignment.allSatisfy { row in
                row.count == track.width && row.allSatisfy(\.isFinite)
            })
    }

    @Test func aTakeWithTheWrongCodebookCountIsRejected() async throws {
        let fixture = try await TinyModel.make(ttsModelType: "voice_design")
        defer { fixture.cleanUp() }
        #expect(throws: AudioGenerationError.self) {
            _ = try fixture.model.prompts.reference(
                text: "one", take: Qwen3TTSReference(codeFrames: [[1, 2, 3]], text: "one"),
                instruct: nil, language: "English")
        }
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

    // MARK: - Sampling

    /// Temperature applies before top-p, as in Qwen's sampler. In this row the
    /// raw 0.8 nucleus holds tokens 7 and 8, but at temperature 0.5 token 7
    /// alone carries 88% of the mass, so 8 must never be drawn.
    @Test func temperatureTightensTheNucleusBeforeSampling() {
        var row = [Float](repeating: -20, count: 3072)
        row[7] = 2.0
        row[8] = 1.0
        let raw = Qwen3TTSSampler.filter(
            MLXArray(row).reshaped(1, row.count), topK: 0, topP: 0.8
        ).asArray(Float.self)
        #expect(raw.indices.filter { raw[$0].isFinite } == [7, 8])

        let sampler = Qwen3TTSSampler(temperature: 0.5, topK: 0, topP: 0.8)
        let random = MLXRandom.RandomState(seed: 0)
        var draws: Set<Int32> = []
        for _ in 0 ..< 200 {
            draws.insert(sampler(MLXArray(row).reshaped(1, row.count), random: random).item(Int32.self))
        }
        #expect(draws == [7])
    }

    /// Top-k keeps every logit at least the k-th largest, so a tie at the
    /// cut keeps both (transformers' TopKLogitsWarper).
    @Test func topKKeepsTiesAtTheCut() {
        var row = [Float](repeating: -5, count: 64)
        row[3] = 4
        row[9] = 2
        row[11] = 2
        let kept = Qwen3TTSSampler.filter(MLXArray(row).reshaped(1, 64), topK: 2, topP: 1)
            .asArray(Float.self)
        #expect(kept.indices.filter { kept[$0].isFinite } == [3, 9, 11])
    }

    /// The repetition penalty reads only the last `window` talker tokens.
    @Test func theRepetitionPenaltyOnlyCountsRecentTokens() {
        var row = [Float](repeating: -20, count: 3072)
        row[7] = 2.0
        row[8] = 1.9
        let greedy = Qwen3TTSSampling(temperature: 0, repetitionPenalty: 1.5)
        var sampler = Qwen3TTSTalkerSampler(
            sampling: greedy, vocabSize: 3072, eosTokenID: 3050, dtype: .float32, window: 2)
        let random = MLXRandom.RandomState(seed: 0)
        let logits = MLXArray(row).reshaped(1, row.count)
        #expect(sampler(logits, frame: 5, random: random).item(Int32.self) == 7)
        #expect(sampler(logits, frame: 6, random: random).item(Int32.self) == 8, "7 is penalized")
        // 7 and 8 both in the window of two: 7 again, as without a penalty.
        #expect(sampler(logits, frame: 7, random: random).item(Int32.self) == 7)
    }

    /// EOS is filtered like every other token (Tesseract #567; Python
    /// mlx-audio made the same fix). In this row EOS holds 18% of the mass
    /// but sits outside the 0.8 nucleus, which tokens 7 (49%) and 8 (33%)
    /// close.
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
        let filtered = Qwen3TTSSampler.filter(
            MLXArray(row).reshaped(1, row.count), topK: 0, topP: 0.8
        ).asArray(Float.self)
        #expect(filtered[Self.eosTokenID] == -Float.infinity)
        #expect(filtered.indices.filter { filtered[$0].isFinite } == [7, 8])
    }

    @Test func eosOutsideTheTopPNucleusIsNeverSampled() {
        let row = Self.eosJustOutsideTheNucleus()
        // The settings the bug shipped under: t=0.6, top-p 0.8, top-k off,
        // penalty 1.3; a fresh sampler per draw, so nothing is penalized.
        let sampling = Qwen3TTSSampling(temperature: 0.6, topP: 0.8, repetitionPenalty: 1.3)
        let random = MLXRandom.RandomState(seed: 0)
        var draws: [Int32: Int] = [:]
        for _ in 0 ..< 200 {
            var sampler = Qwen3TTSTalkerSampler(
                sampling: sampling, vocabSize: 3072, eosTokenID: Self.eosTokenID, dtype: .float32)
            let token = sampler(MLXArray(row).reshaped(1, row.count), frame: 10, random: random)
            draws[token.item(Int32.self), default: 0] += 1
        }
        #expect(draws[Int32(Self.eosTokenID)] == nil)
        #expect(Set(draws.keys) == [7, 8])
    }

    /// Qwen's min_new_tokens = 2: EOS can't end the first two frames, and
    /// the control tokens above the codec range never come out.
    @Test func eosWaitsForTwoFramesAndControlTokensNeverCome() {
        var row = [Float](repeating: -20, count: 3072)
        row[Self.eosTokenID] = 10
        row[2100] = 20  // a control token: suppressed
        row[5] = 1
        let greedy = Qwen3TTSSampling(temperature: 0, repetitionPenalty: 1)
        var sampler = Qwen3TTSTalkerSampler(
            sampling: greedy, vocabSize: 3072, eosTokenID: Self.eosTokenID, dtype: .float32)
        let random = MLXRandom.RandomState(seed: 0)
        let logits = MLXArray(row).reshaped(1, row.count)
        #expect(sampler(logits, frame: 0, random: random).item(Int32.self) == 5)
        #expect(sampler(logits, frame: 1, random: random).item(Int32.self) == 5)
        #expect(sampler(logits, frame: 2, random: random).item(Int32.self) == Int32(Self.eosTokenID))
    }

    // MARK: - Fused kernels

    /// The fused sampler draws the token the MLX ops draw: same division,
    /// same top-k cut (ties kept), same uniform draw, same Gumbel-max.
    @Test(arguments: [DType.bfloat16, .float32])
    func theFusedSamplerDrawsWhatMLXDraws(dtype: DType) {
        let vocab = 2048
        var agree = 0
        for trial in 0 ..< 60 {
            let logits = (MLXRandom.normal([1, vocab], key: MLXRandom.RandomState(seed: UInt64(trial)))
                * 3).asType(dtype)
            let sampler = Qwen3TTSSampler(temperature: 0.5 + Float(trial % 3) * 0.2, topK: 50, topP: 1)
            Qwen3TTSKernels.enabled = false
            let reference = sampler(logits, random: MLXRandom.RandomState(seed: 99)).item(Int32.self)
            Qwen3TTSKernels.enabled = true
            let fused = sampler(logits, random: MLXRandom.RandomState(seed: 99)).item(Int32.self)
            if reference == fused { agree += 1 }
        }
        #expect(agree == 60, "\(agree) of 60 draws agree")
    }

    /// The fused q/k RMSNorm + RoPE computes MLX's rms_norm then rope, bit
    /// for bit: the same sum order, casts and math functions, compiled with
    /// fast math as MLX's kernels are. Compiled without it, one value in
    /// 100,000 to 200,000 lands a bf16 step away, so this checks some 4
    /// million.
    @Test func theFusedNormRoPEMatchesMLX() throws {
        let (heads, kvHeads, headDim) = (16, 8, 128)
        let random = MLXRandom.RandomState(seed: 5)
        let qkv = (MLXRandom.normal([1, 1, (heads + 2 * kvHeads) * headDim], key: random) * 2)
            .asType(.bfloat16)
        let qw = (1 + MLXRandom.normal([headDim], key: random) * 0.3).asType(.bfloat16)
        let kw = (1 + MLXRandom.normal([headDim], key: random) * 0.3).asType(.bfloat16)
        var differ = MLXArray(Int32(0))
        for offset in 0 ..< 1400 {
            let (q, k) = try #require(
                Qwen3TTSKernels.qkNormRoPE(
                    qkv, qWeight: qw, kWeight: kw, heads: heads, kvHeads: kvHeads,
                    headDim: headDim, eps: 1e-6, base: 1_000_000, offset: offset))
            let split = qkv.reshaped(1, 1, heads + 2 * kvHeads, headDim)
            let refQ = MLXFast.RoPE(
                MLXFast.rmsNorm(split[0..., 0..., ..<heads, 0...], weight: qw, eps: 1e-6)
                    .transposed(0, 2, 1, 3),
                dimensions: headDim, traditional: false, base: 1_000_000, scale: 1, offset: offset)
            let refK = MLXFast.RoPE(
                MLXFast.rmsNorm(split[0..., 0..., heads ..< (heads + kvHeads), 0...], weight: kw, eps: 1e-6)
                    .transposed(0, 2, 1, 3),
                dimensions: headDim, traditional: false, base: 1_000_000, scale: 1, offset: offset)
            if offset == 0 {
                #expect(q.shape == [1, heads, 1, headDim])
                #expect(k.shape == [1, kvHeads, 1, headDim])
            }
            differ = differ + (q .!= refQ).sum() + (k .!= refK).sum()
        }
        let mismatches = differ.item(Int32.self)
        #expect(mismatches == 0, "\(mismatches) of \(1400 * 3072) values differ")
    }

    /// The fused residual add + RMSNorm is MLX's add then rms_norm.
    @Test(arguments: [1024, 2048])
    func theFusedAddNormMatchesMLX(width: Int) {
        let random = MLXRandom.RandomState(seed: 8)
        let x = MLXRandom.normal([1, 1, width], key: random).asType(.bfloat16)
        let y = (MLXRandom.normal([1, 1, width], key: random) * 0.5).asType(.bfloat16)
        let w = (1 + MLXRandom.normal([width], key: random) * 0.2).asType(.bfloat16)
        let (sum, normed) = Qwen3TTSKernels.addRMSNorm(x, y, weight: w, eps: 1e-6)
        let refSum = x + y
        let refNormed = MLXFast.rmsNorm(refSum, weight: w, eps: 1e-6)
        #expect((sum .== refSum).all().item(Bool.self), "the sum is bit-exact")
        let mismatches = (normed .!= refNormed).sum().item(Int32.self)
        #expect(mismatches == 0, "\(mismatches) of \(width) normalized values differ")
    }

    /// One attention step, fused and not, bit for bit.
    @Test(arguments: [64, 128])
    func theFusedAttentionStepIsExact(headDim: Int) {
        let (hidden, heads, kvHeads) = (1024, 16, 8)
        let attention = Qwen3TTSAttention(
            hiddenSize: hidden, heads: heads, kvHeads: kvHeads, headDim: headDim,
            ropeBase: 1_000_000, rmsNormEps: 1e-6, bias: false, fused: true)
        attention.apply { $0.asType(.bfloat16) }
        eval(attention.parameters())
        let random = MLXRandom.RandomState(seed: 21)
        let prompt = MLXRandom.normal([1, 6, hidden], key: random).asType(.bfloat16)
        let step = MLXRandom.normal([1, 1, hidden], key: random).asType(.bfloat16)
        func run(fused: Bool) -> MLXArray {
            Qwen3TTSKernels.enabled = fused
            defer { Qwen3TTSKernels.enabled = true }
            let cache = KVCacheSimple()
            _ = attention(prompt, cache: cache)
            return attention(step, cache: cache)
        }
        let mismatches = (run(fused: true) .!= run(fused: false)).sum().item(Int32.self)
        #expect(mismatches == 0, "head \(headDim): \(mismatches) of \(hidden) outputs differ")
    }

    /// A whole layer stack, fused and not, bit for bit, over many seeded
    /// weights and inputs.
    @Test(arguments: [64, 128])
    func theFusedLayerStackMatchesTheUnfusedOne(headDim: Int) throws {
        let (hidden, heads, kvHeads, inter) = (1024, 16, 8, 1536)
        var failures: [String] = []
        for seed in 0 ..< 20 {
            let random = MLXRandom.RandomState(seed: UInt64(seed))
            let (layers, norm) = try randomDecoderLayers(
                3, hidden: hidden, heads: heads, kvHeads: kvHeads, headDim: headDim,
                intermediate: inter, scale: 0.03, random: random)
            let prompt = MLXRandom.normal([1, 5, hidden], key: random).asType(.bfloat16)
            let step = MLXRandom.normal([1, 1, hidden], key: random).asType(.bfloat16)
            func run(fused: Bool) -> MLXArray {
                Qwen3TTSKernels.enabled = fused
                defer { Qwen3TTSKernels.enabled = true }
                let caches: [KVCache] = layers.map { _ in KVCacheSimple() }
                _ = runQwen3TTSLayers(prompt, layers: layers, cache: caches, finalNorm: norm)
                let out = runQwen3TTSLayers(step, layers: layers, cache: caches, finalNorm: norm)
                eval(out)
                return out
            }
            let mismatches = (run(fused: true) .!= run(fused: false)).sum().item(Int32.self)
            if mismatches != 0 { failures.append("seed \(seed): \(mismatches)") }
        }
        #expect(failures.isEmpty, "head \(headDim): \(failures.joined(separator: ", "))")
    }

    /// A prompt evaluated a layer at a time computes what the one graph
    /// does: the same last position and the same keys and values cached.
    @Test(arguments: [40, 90])
    func aLayerAtATimePrefillMatchesTheOneGraph(length: Int) throws {
        let (hidden, heads, kvHeads, headDim, inter) = (256, 4, 2, 64, 384)
        let random = MLXRandom.RandomState(seed: 17)
        let (layers, norm) = try randomDecoderLayers(
            4, hidden: hidden, heads: heads, kvHeads: kvHeads, headDim: headDim,
            intermediate: inter, scale: 0.05, random: random)
        let prompt = MLXRandom.normal([1, length, hidden], key: random).asType(.bfloat16)
        let oneGraph: [KVCache] = layers.map { _ in KVCacheSimple() }
        let layered: [KVCache] = layers.map { _ in KVCacheSimple() }
        let a = runQwen3TTSLayers(prompt, layers: layers, cache: oneGraph, finalNorm: norm)
        let b = runQwen3TTSLayers(
            prompt, layers: layers, cache: layered, finalNorm: norm, layerByLayer: true)
        #expect((a .== b).all().item(Bool.self))
        for (x, y) in zip(oneGraph, layered) {
            #expect(x.offset == y.offset)
            for (p, q) in zip(x.state, y.state) { #expect((p .== q).all().item(Bool.self)) }
        }
    }

    /// The fork's causal SDPA, at the query counts prompts bring, with and
    /// without a cached prefix, against attention written out in float32.
    @Test(arguments: [0, 37])
    func causalAttentionMatchesAttentionWrittenOut(prefix: Int) {
        let (heads, kvHeads, headDim) = (16, 8, 128)
        let random = MLXRandom.RandomState(seed: 21)
        let scale = 1 / Float(headDim).squareRoot()
        for queries in [1, 2, 7, 8, 9, 12, 16, 17, 31, 33, 64, 100] {
            let keys = prefix + queries
            let q = MLXRandom.normal([1, heads, queries, headDim], key: random).asType(.bfloat16)
            let k = MLXRandom.normal([1, kvHeads, keys, headDim], key: random).asType(.bfloat16)
            let v = MLXRandom.normal([1, kvHeads, keys, headDim], key: random).asType(.bfloat16)
            let fast = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale, mask: .causal)
            // Query i sees keys 0 through prefix + i.
            let kf = repeated(k.asType(.float32), count: heads / kvHeads, axis: 1)
            let vf = repeated(v.asType(.float32), count: heads / kvHeads, axis: 1)
            let scores = matmul(q.asType(.float32), kf.transposed(0, 1, 3, 2)) * scale
            let rows = MLXArray(Int32(0) ..< Int32(queries)).reshaped(queries, 1) + Int32(prefix)
            let columns = MLXArray(Int32(0) ..< Int32(keys)).reshaped(1, keys)
            let masked = which(columns .<= rows, scores, MLXArray(-Float.infinity))
            let reference = matmul(softmax(masked, axis: -1), vf)
            let difference = abs(fast.asType(.float32) - reference).max().item(Float.self)
            // bf16 output: its rounding is about 0.008 at these magnitudes.
            #expect(difference < 0.03, "\(queries) queries over \(keys) keys: \(difference)")
        }
    }

    // MARK: - Layers

    /// Stacking q/k/v (and gate/up) into one projection is the same function.
    @Test func stackedProjectionsMatchSeparateOnes() {
        let random = MLXRandom.RandomState(seed: 3)
        let (hidden, heads, kvHeads, headDim, inter) = (32, 4, 2, 8, 48)
        let q = MLXRandom.normal([heads * headDim, hidden], key: random) * 0.2
        let k = MLXRandom.normal([kvHeads * headDim, hidden], key: random) * 0.2
        let v = MLXRandom.normal([kvHeads * headDim, hidden], key: random) * 0.2
        let o = MLXRandom.normal([hidden, heads * headDim], key: random) * 0.2
        let gate = MLXRandom.normal([inter, hidden], key: random) * 0.2
        let up = MLXRandom.normal([inter, hidden], key: random) * 0.2
        let down = MLXRandom.normal([hidden, inter], key: random) * 0.2
        func layer(fused: Bool) throws -> Qwen3TTSDecoderLayer {
            let l = Qwen3TTSDecoderLayer(
                hiddenSize: hidden, intermediateSize: inter, heads: heads, kvHeads: kvHeads,
                headDim: headDim, ropeBase: 10_000, rmsNormEps: 1e-6, attentionBias: false,
                fusion: Qwen3TTSFusion(qkv: fused, gateUp: fused))
            var p: [String: MLXArray] = [
                "self_attn.o_proj.weight": o, "mlp.down_proj.weight": down,
            ]
            if fused {
                p["self_attn.qkv_proj.weight"] = concatenated([q, k, v], axis: 0)
                p["mlp.gate_up_proj.weight"] = concatenated([gate, up], axis: 0)
            } else {
                p["self_attn.q_proj.weight"] = q
                p["self_attn.k_proj.weight"] = k
                p["self_attn.v_proj.weight"] = v
                p["mlp.gate_proj.weight"] = gate
                p["mlp.up_proj.weight"] = up
            }
            try l.update(parameters: ModuleParameters.unflattened(p), verify: .noUnusedKeys)
            return l
        }
        let x = MLXRandom.normal([1, 5, hidden], key: random)
        let a = try! layer(fused: true)(x, cache: KVCacheSimple())
        let b = try! layer(fused: false)(x, cache: KVCacheSimple())
        #expect(abs(a - b).max().item(Float.self) < 1e-4)
    }

    // MARK: - Codec decoder

    /// Decoding a stream a chunk at a time gives the one-pass audio, at any
    /// chunk size: the causal convs carry their context, the transposed convs
    /// their overlap (bias added once), the transformer its window.
    @Test(arguments: [8, 1000])
    func theDecoderIsChunkSizeInvariant(window: Int) throws {
        let decoder = try TinyModel.decoder(slidingWindow: window)
        let codes = TinyModel.randomCodes(frames: 29, groups: 2, seed: 5)
        let whole = try decoder.decodeAll(codes).asArray(Float.self)
        #expect(whole.count == 29 * decoder.samplesPerFrame)
        for chunk in [1, 3, 7] {
            var stream = decoder.makeStream()
            var samples: [Float] = []
            var start = 0
            while start < 29 {
                let end = min(start + chunk, 29)
                let audio = try decoder.decode(codes[0..., start ..< end, 0...], stream: &stream)
                eval([audio] + stream.arrays)
                samples += audio.asArray(Float.self)
                start = end
            }
            let peak = whole.map { abs($0) }.max() ?? 1
            let difference = maxAbsDifference(samples, whole)
            #expect(difference < 1e-5, "window \(window) chunk \(chunk): \(difference) (peak \(peak))")
            #expect(peak < 1, "the test audio isn't clipped")
        }
    }

    /// The transformer attends over a sliding window (72 frames in the
    /// checkpoints, 8 here): a frame far enough back no longer shapes the
    /// audio. Full attention would carry the first frame to the end.
    @Test func theDecoderForgetsFramesBeyondItsWindow() throws {
        let decoder = try TinyModel.decoder(slidingWindow: 8)
        var a = TinyModel.randomCodes(frames: 80, groups: 2, seed: 9)
        let flat = a.asArray(Int32.self)
        var changed = flat
        changed[0] = (flat[0] + 1) % 2048
        changed[1] = (flat[1] + 7) % 2048
        let b = MLXArray(changed).reshaped(1, 80, 2)
        a = MLXArray(flat).reshaped(1, 80, 2)
        let x = try decoder.decodeAll(a).asArray(Float.self)
        let y = try decoder.decodeAll(b).asArray(Float.self)
        let perFrame = decoder.samplesPerFrame
        #expect(maxAbsDifference(Array(x[..<perFrame]), Array(y[..<perFrame])) > 0)
        // Window 8, plus the tiny conv stack's reach: its small upsampling
        // rates stretch the dilated convs over tens of frames.
        let late = 55 * perFrame
        #expect(maxAbsDifference(Array(x[late...]), Array(y[late...])) < 1e-6)
    }

    @Test func theSlidingWindowMaskIsCausalAndBounded() {
        func mask(_ q: Int, _ c: Int, _ w: Int) -> [[Bool]]? {
            guard case .array(let m) = Qwen3TTSCodecDecoder.slidingWindowMask(
                queries: q, cached: c, window: w)
            else { return nil }
            let flat = m.asArray(Bool.self)
            return (0 ..< q).map { i in Array(flat[(i * (c + q)) ..< ((i + 1) * (c + q))]) }
        }
        #expect(mask(1, 5, 8) == nil, "one query that sees every carried key")
        let m = try! #require(mask(3, 7, 8))
        for i in 0 ..< 3 {
            for j in 0 ..< 10 {
                let query = 7 + i
                #expect(m[i][j] == (j <= query && j > query - 8), "query \(i) key \(j)")
            }
        }
    }

    /// Loading reads only the decoder half: encoder tensors are skipped.
    @Test func theDecoderLoadsWithoutTheEncoder() throws {
        let directory = try makeTemporaryDirectory("tiny-speech-tokenizer")
        defer { try? FileManager.default.removeItem(at: directory) }
        let config = TinyModel.decoderConfig(slidingWindow: 8)
        var tensors = TinyModel.decoderTensors(config)
            .reduce(into: [String: MLXArray]()) { $0["decoder.\($1.key)"] = $1.value }
        tensors["encoder.encoder.layers.0.conv.weight"] = MLXArray.zeros([4, 1, 7])
        try MLX.save(arrays: tensors, url: directory.appendingPathComponent("model.safetensors"))
        try TinyModel.speechTokenizerConfigJSON(slidingWindow: 8)
            .write(to: directory.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let decoder = try Qwen3TTSCodecDecoder(directory: directory, dtype: .float32)
        #expect(decoder.samplesPerFrame == TinyModel.samplesPerFrame)
    }

    // MARK: - Neural Engine codec

    /// The Core ML conv stack this package writes (protobuf, weight blobs,
    /// polyphase upsamplers, carried conv state) computes what the MLX conv
    /// stack does. On the CPU here, so the test doesn't need the Neural
    /// Engine; the app checks placement and agreement again at load.
    @Test func theCoreMLConvStackMatchesMLX() async throws {
        let decoder = try TinyModel.decoder(slidingWindow: 8, dtype: .float16)
        let cache = try makeTemporaryDirectory("neural-codec")
        defer { try? FileManager.default.removeItem(at: cache) }
        let (codec, _) = try await Qwen3TTSNeuralCodec.load(
            decoder: decoder, frames: 3, cacheDirectory: cache, sourceKey: "tiny",
            computeUnits: .cpuOnly, requireNeuralEngine: false)
        #expect(codec.frames == 3)

        let codes = TinyModel.randomCodes(frames: 7, groups: 2, seed: 4)
        var stream = decoder.makeStream()
        let latent = decoder.latent(codes, stream: &stream)
        let expected = try decoder.synthesize(latent, stream: &stream).asArray(Float.self)
        // Three calls, the last one frame long (padded inside).
        let actual = try codec.decodeAll(latent: latent.asType(.float16).asArray(Float16.self))
        #expect(actual.count == expected.count)
        let snr = Qwen3TTSNeuralCodec.snr(actual, expected)
        #expect(snr > 40, "Core ML vs MLX: \(snr) dB")

        // A second load finds the compiled model in the cache.
        let (again, _) = try await Qwen3TTSNeuralCodec.load(
            decoder: decoder, frames: 3, cacheDirectory: cache, sourceKey: "tiny",
            computeUnits: .cpuOnly, requireNeuralEngine: false)
        #expect(again.frames == 3)
        let cached = try FileManager.default.contentsOfDirectory(atPath: cache.path)
        #expect(cached.filter { $0.hasSuffix(".mlmodelc") }.count == 1)
    }

    // MARK: - Text embedding

    /// An unquantized table is read from the checkpoint file row by row.
    @Test func theTextEmbeddingReadsRowsFromTheFile() throws {
        let directory = try makeTemporaryDirectory("text-embedding")
        defer { try? FileManager.default.removeItem(at: directory) }
        let table = (MLXRandom.normal([40, 6], key: MLXRandom.RandomState(seed: 1)))
            .asType(.bfloat16)
        try MLX.save(
            arrays: ["talker.model.text_embedding.weight": table, "other": MLXArray.ones([3])],
            url: directory.appendingPathComponent("model.safetensors"))
        let embedding = try #require(
            Qwen3TTSTextEmbedding(key: "talker.model.text_embedding.weight", directory: directory))
        #expect(embedding.count == 40)
        #expect(embedding.dimensions == 6)
        let rows = try embedding([3, 0, 39, 3])
        #expect(rows.shape == [1, 4, 6])
        #expect(rows.dtype == .bfloat16)
        let expected = table[MLXArray([Int32(3), 0, 39, 3])].reshaped(1, 4, 6)
        #expect((rows .== expected).all().item(Bool.self))
        #expect(throws: AudioGenerationError.self) { _ = try embedding([40]) }
    }
}

// MARK: - Fixtures

struct TinyModel {
    let model: Qwen3TTSModel
    let tokenizerDirectory: URL

    static let sampling = Qwen3TTSSampling(
        temperature: 0.7, topP: 0.95, repetitionPenalty: 1.0, maxTokens: 4)

    /// 2 × 2 × 2 × 3 × 2 × 2: the decoder's total upsampling.
    static let samplesPerFrame = 96

    func cleanUp() {
        try? FileManager.default.removeItem(at: tokenizerDirectory)
    }

    /// The tiny decoder's config, as its checkpoint's JSON writes it.
    static func decoderConfigJSON(slidingWindow: Int) -> String {
        """
        {"latent_dim": 32, "codebook_dim": 16, "codebook_size": 2048, "decoder_dim": 48,
         "hidden_size": 16, "intermediate_size": 32, "num_attention_heads": 2,
         "num_key_value_heads": 2, "head_dim": 8, "num_hidden_layers": 2,
         "num_quantizers": 2, "num_semantic_quantizers": 1, "sliding_window": \(slidingWindow),
         "upsample_rates": [2, 3, 2, 2], "upsampling_ratios": [2, 2]}
        """
    }

    static func decoderConfig(slidingWindow: Int) -> Qwen3TTSTokenizerDecoderConfig {
        try! JSONDecoder().decode(
            Qwen3TTSTokenizerDecoderConfig.self,
            from: Data(decoderConfigJSON(slidingWindow: slidingWindow).utf8))
    }

    static func speechTokenizerConfigJSON(slidingWindow: Int) -> String {
        """
        {"decode_upsample_rate": \(samplesPerFrame),
         "decoder_config": \(decoderConfigJSON(slidingWindow: slidingWindow))}
        """
    }

    /// Random decoder weights in the checkpoint's (PyTorch) layout.
    static func decoderTensors(_ c: Qwen3TTSTokenizerDecoderConfig, seed: UInt64 = 1)
        -> [String: MLXArray]
    {
        let random = MLXRandom.RandomState(seed: seed)
        // Scaled to fan-in, as a trained network is, so activations stay
        // near 1 through thirty layers and the audio isn't clipped flat.
        func r(_ shape: [Int], _ scale: Float? = nil) -> MLXArray {
            let fanIn = shape.count > 1 ? shape.dropFirst().reduce(1, *) : 1
            return MLXRandom.normal(shape, key: random)
                * (scale ?? 1 / Float(fanIn).squareRoot())
        }
        var t: [String: MLXArray] = [:]
        let half = c.codebookDim / 2
        for q in 0 ..< c.numQuantizers {
            let group = q < c.numSemanticQuantizers ? "rvq_first" : "rvq_rest"
            let index = q < c.numSemanticQuantizers ? q : q - c.numSemanticQuantizers
            let base = "quantizer.\(group).vq.layers.\(index)._codebook"
            t["\(base).embedding_sum"] = r([c.codebookSize, half], 1)
            t["\(base).cluster_usage"] = abs(r([c.codebookSize], 1)) + 0.5
        }
        for group in ["rvq_first", "rvq_rest"] {
            t["quantizer.\(group).output_proj.weight"] = r([c.codebookDim, half, 1])
        }
        t["pre_conv.conv.weight"] = r([c.latentDim, c.codebookDim, 3])
        t["pre_conv.conv.bias"] = r([c.latentDim])
        let h = c.hiddenSize
        let width = c.numAttentionHeads * c.headDim
        t["pre_transformer.input_proj.weight"] = r([h, c.latentDim])
        t["pre_transformer.input_proj.bias"] = r([h])
        for i in 0 ..< c.numHiddenLayers {
            let p = "pre_transformer.layers.\(i)"
            t["\(p).input_layernorm.weight"] = 1 + r([h], 0.1)
            t["\(p).post_attention_layernorm.weight"] = 1 + r([h], 0.1)
            t["\(p).self_attn.q_proj.weight"] = r([width, h])
            t["\(p).self_attn.k_proj.weight"] = r([width, h])
            t["\(p).self_attn.v_proj.weight"] = r([width, h])
            t["\(p).self_attn.o_proj.weight"] = r([h, width])
            t["\(p).self_attn_layer_scale.scale"] = r([h], 0.5)
            t["\(p).mlp.gate_proj.weight"] = r([c.intermediateSize, h])
            t["\(p).mlp.up_proj.weight"] = r([c.intermediateSize, h])
            t["\(p).mlp.down_proj.weight"] = r([h, c.intermediateSize])
            t["\(p).mlp_layer_scale.scale"] = r([h], 0.5)
        }
        t["pre_transformer.norm.weight"] = 1 + r([h], 0.1)
        t["pre_transformer.output_proj.weight"] = r([c.latentDim, h])
        t["pre_transformer.output_proj.bias"] = r([c.latentDim])
        let d = c.latentDim
        for (i, factor) in c.upsamplingRatios.enumerated() {
            let p = "upsample.\(i)"
            t["\(p).0.conv.weight"] = r([d, d, factor], 1 / Float(d).squareRoot())
            t["\(p).0.conv.bias"] = r([d])
            t["\(p).1.dwconv.conv.weight"] = r([d, 1, 7])
            t["\(p).1.dwconv.conv.bias"] = r([d])
            t["\(p).1.norm.weight"] = 1 + r([d], 0.1)
            t["\(p).1.norm.bias"] = r([d], 0.1)
            t["\(p).1.pwconv1.weight"] = r([4 * d, d])
            t["\(p).1.pwconv1.bias"] = r([4 * d])
            t["\(p).1.pwconv2.weight"] = r([d, 4 * d])
            t["\(p).1.pwconv2.bias"] = r([d])
            t["\(p).1.gamma"] = r([d], 0.5)
        }
        t["decoder.0.conv.weight"] = r([c.decoderDim, d, 7])
        t["decoder.0.conv.bias"] = r([c.decoderDim])
        for (i, rate) in c.upsampleRates.enumerated() {
            let p = "decoder.\(i + 1).block"
            let inDim = c.decoderDim >> i
            let outDim = c.decoderDim >> (i + 1)
            t["\(p).0.alpha"] = r([inDim], 0.1)
            t["\(p).0.beta"] = 1 + r([inDim], 0.1)
            t["\(p).1.conv.weight"] = r([inDim, outDim, 2 * rate], 1 / Float(2 * inDim).squareRoot())
            t["\(p).1.conv.bias"] = r([outDim])
            for u in 2 ... 4 {
                let q = "\(p).\(u)"
                // A small residual branch, as in a trained network: the
                // stream's activations stay near 1 instead of doubling per unit.
                t["\(q).act1.alpha"] = r([outDim], 0.1)
                t["\(q).act1.beta"] = 1 + r([outDim], 0.1)
                t["\(q).conv1.conv.weight"] = r([outDim, outDim, 7])
                t["\(q).conv1.conv.bias"] = r([outDim], 0.05)
                t["\(q).act2.alpha"] = r([outDim], 0.1)
                t["\(q).act2.beta"] = 1 + r([outDim], 0.1)
                t["\(q).conv2.conv.weight"] = r([outDim, outDim, 1], 0.1 / Float(outDim).squareRoot())
                t["\(q).conv2.conv.bias"] = r([outDim], 0.05)
            }
        }
        let n = c.upsampleRates.count
        let outputDim = c.decoderDim >> n
        t["decoder.\(n + 1).alpha"] = r([outputDim], 0.1)
        t["decoder.\(n + 1).beta"] = 1 + r([outputDim], 0.1)
        t["decoder.\(n + 2).conv.weight"] = r([1, outputDim, 7], 0.1 / Float(7 * outputDim).squareRoot())
        t["decoder.\(n + 2).conv.bias"] = r([1])
        return t
    }

    static func decoder(slidingWindow: Int, dtype: DType = .float32) throws -> Qwen3TTSCodecDecoder {
        let config = decoderConfig(slidingWindow: slidingWindow)
        return try Qwen3TTSCodecDecoder(
            config: config, samplesPerFrame: samplesPerFrame, tensors: decoderTensors(config),
            dtype: dtype)
    }

    static func randomCodes(frames: Int, groups: Int, seed: UInt64) -> MLXArray {
        MLXRandom.randInt(
            Int32(0) ..< Int32(2048), [1, frames, groups], key: MLXRandom.RandomState(seed: seed))
    }

    /// Two talker layers of width 16, two code groups, random weights, and a
    /// small decoder: enough to run every generation path.
    static func make(
        ttsModelType: String,
        speakerIDEntries: String = "",
        dialectEntries: String = "",
        extraLanguages: String = ""
    ) async throws -> TinyModel {
        let tokenizerDirectory = try makeTinyTokenizerDirectory()
        let spkIdJSON = speakerIDEntries.isEmpty ? "" : #""spk_id": {"# + speakerIDEntries + "},"
        let spkDialectJSON =
            dialectEntries.isEmpty ? "" : #""spk_is_dialect": {"# + dialectEntries + "},"
        let configJSON = """
            {
              "model_type": "qwen3_tts",
              "tts_model_type": "\(ttsModelType)",
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
                "num_key_value_heads": 2,
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
                  "english": 3057\(extraLanguages)
                },
                \(spkIdJSON)
                \(spkDialectJSON)
                "code_predictor_config": {
                  "vocab_size": 2048,
                  "hidden_size": 16,
                  "intermediate_size": 32,
                  "num_hidden_layers": 1,
                  "num_attention_heads": 4,
                  "num_key_value_heads": 2,
                  "head_dim": 4,
                  "max_position_embeddings": 128,
                  "num_code_groups": 2
                }
              }
            }
            """
        let config = try JSONDecoder().decode(Qwen3TTSModelConfig.self, from: Data(configJSON.utf8))
        let talkerConfig = try #require(config.talkerConfig)
        let talker = Qwen3TTSTalker(config: talkerConfig)
        eval(talker.parameters())
        let text = Embedding(embeddingCount: 64, dimensions: 16)
        eval(text.parameters())
        let model = try Qwen3TTSModel(
            config: config, talker: talker,
            textEmbedding: Qwen3TTSTextEmbedding(embedding: text, count: 64, dimensions: 16),
            codecDecoder: decoder(slidingWindow: 8),
            tokenizer: try await AutoTokenizer.from(modelFolder: tokenizerDirectory))
        return TinyModel(model: model, tokenizerDirectory: tokenizerDirectory)
    }
}

func makeTemporaryDirectory(_ prefix: String) throws -> URL {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("\(prefix)-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    return directory
}

func maxAbsDifference(_ a: [Float], _ b: [Float]) -> Float {
    guard a.count == b.count else { return .infinity }
    return zip(a, b).map { abs($0 - $1) }.max() ?? 0
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

/// `count` decoder layers with random bf16 weights (norm weights near 1,
/// the rest `scale`-sized), and a final norm.
func randomDecoderLayers(
    _ count: Int, hidden: Int, heads: Int, kvHeads: Int, headDim: Int, intermediate: Int,
    scale: Float, random: MLXRandom.RandomState
) throws -> (layers: [Qwen3TTSDecoderLayer], norm: RMSNorm) {
    let layers = (0 ..< count).map { _ in
        Qwen3TTSDecoderLayer(
            hiddenSize: hidden, intermediateSize: intermediate, heads: heads, kvHeads: kvHeads,
            headDim: headDim, ropeBase: 1_000_000, rmsNormEps: 1e-6, attentionBias: false,
            fusion: Qwen3TTSFusion())
    }
    for layer in layers {
        let parameters = layer.parameters().flattened().map { key, value in
            (key, (key.hasSuffix("norm.weight") || key.hasSuffix("layernorm.weight"))
                ? (1 + MLXRandom.normal(value.shape, key: random) * 0.2).asType(.bfloat16)
                : (MLXRandom.normal(value.shape, key: random) * scale).asType(.bfloat16))
        }
        try layer.update(parameters: ModuleParameters.unflattened(parameters), verify: .all)
    }
    let norm = RMSNorm(dimensions: hidden, eps: 1e-6)
    norm.apply { $0.asType(.bfloat16) }
    eval(layers.map { $0.parameters() }, norm.parameters())
    return (layers, norm)
}

struct Collected {
    var chunks = 0
    var allAudio: [Float] = []
    var lastAudio: [Float]?
    var codeFrames: [[Int32]]?
    var textTrack: Qwen3TTSTextTrack?
    var alignment: [[Float]] = []
    /// How many frames of audio had arrived when each alignment row did.
    var audioFramesAtRow: [Int] = []
}

func collect(_ stream: AsyncThrowingStream<AudioGeneration, Error>) async throws -> Collected {
    var c = Collected()
    for try await event in stream {
        switch event {
        case .audio(let audio):
            c.chunks += 1
            c.allAudio += audio
            c.lastAudio = audio
        case .codeFrames(let frames): c.codeFrames = frames
        case .textTrack(let track): c.textTrack = track
        case .alignment(let row):
            c.alignment.append(row)
            c.audioFramesAtRow.append(c.allAudio.count)
        }
    }
    return c
}
