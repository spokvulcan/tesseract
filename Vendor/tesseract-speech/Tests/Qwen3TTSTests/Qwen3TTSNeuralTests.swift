// The talker and the code predictor on the Neural Engine (ADR-0084), on tiny
// random-weight models: the Core ML graphs compiled for the CPU against MLX,
// so no Neural Engine is needed. The real checkpoint's numbers come from
// `qwen3-tts-bench --mode neural-voice`.

import CoreML
import Foundation
@preconcurrency import MLX
import MLXNN
import Testing

@testable import Qwen3TTS

@Suite("Qwen3TTS on the Neural Engine", .serialized)
struct Qwen3TTSNeuralTests {

    // MARK: - The talker

    /// Position by position, the talker's step gives MLX's logits, hidden
    /// state and alignment-head scores, with the cache carried in the
    /// model's state between calls, and another session's steps in between
    /// changing nothing.
    @Test func theTalkerStepsAsMLXDoes() async throws {
        let talker = try NeuralFixture.talker()
        let directory = try makeTemporaryDirectory("neural-talker")
        defer { try? FileManager.default.removeItem(at: directory) }
        let head = Qwen3TTSAlignmentHead(layer: 1, head: 3)
        let length = 12
        let (neural, _) = try await Qwen3TTSNeuralTalker.load(
            talker: talker, contextLength: length, alignment: head, cacheDirectory: directory,
            sourceKey: "tiny", computeUnits: .cpuOnly, requireNeuralEngine: false)

        let probe = Qwen3TTSAlignmentProbe(head: head.head, span: 0 ..< length)
        let attention = talker.model.layers[head.layer].attention
        attention.alignmentProbe = probe
        defer { attention.alignmentProbe = nil }

        let inputs = MLXRandom.normal([1, 9, NeuralFixture.hidden], key: MLXRandom.RandomState(seed: 3))
        let cache = talker.makeCache(capacity: length)
        let session = try neural.makeSession()
        // The alignment rows are pooled: a row of one or two small scores
        // says little on its own.
        var rows: [Float] = []
        var expectedRows: [Float] = []
        for (i, row) in Qwen3TTSModel.rows(inputs).enumerated() {
            let (logits, hidden, scores) = Device.withDefaultDevice(.cpu) {
                let (logits, hidden) = talker(inputs[0..., i ..< (i + 1), 0...], cache: cache)
                return (
                    logits.asArray(Float.self), hidden.asArray(Float.self),
                    probe.scores!.asArray(Float.self)
                )
            }
            let step = try row.withUnsafeBufferPointer { try neural.step($0, session: session) }
            if i == 4 {
                let other = try neural.makeSession()
                for row in Qwen3TTSModel.rows(inputs * -2).prefix(3) {
                    _ = try row.withUnsafeBufferPointer { try neural.step($0, session: other) }
                }
            }
            let alignment = try #require(step.alignment)
            #expect(alignment.count == i + 1)
            rows += alignment
            expectedRows += scores
            let agreement = [snr(step.logits, logits), snr(step.hidden.floats(), hidden)]
            #expect(agreement.allSatisfy { $0 > 40 }, "position \(i): \(agreement) dB")
        }
        #expect(session.position == 9)
        #expect(snr(rows, expectedRows) > 35)
    }

    /// The cache holds `contextLength` positions and no more.
    @Test func theTalkerRefusesAPositionPastItsCache() async throws {
        let talker = try NeuralFixture.talker()
        let directory = try makeTemporaryDirectory("neural-talker-full")
        defer { try? FileManager.default.removeItem(at: directory) }
        let (neural, _) = try await Qwen3TTSNeuralTalker.load(
            talker: talker, contextLength: 2, alignment: nil, cacheDirectory: directory,
            sourceKey: "tiny", computeUnits: .cpuOnly, requireNeuralEngine: false)
        let session = try neural.makeSession()
        let x = [Float16](repeating: 0.1, count: NeuralFixture.hidden)
        try x.withUnsafeBufferPointer { x in
            #expect(try neural.step(x, session: session).alignment == nil)
            _ = try neural.step(x, session: session)
            #expect(throws: AudioGenerationError.self) { try neural.step(x, session: session) }
        }
    }

    // MARK: - The code predictor

    /// Greedy, a frame's codes and embedding sum are MLX's. Drawn, every code
    /// is the best of its top k plus the same Gumbel noise, as MLX's logits
    /// rank them (the reference follows the graph's codes, so a near tie
    /// that fp16 settles the other way can't derail the rest).
    @Test func theCodePredictorDrawsAsMLXDoes() async throws {
        let talker = try NeuralFixture.talker()
        let predictor = talker.codePredictor
        let directory = try makeTemporaryDirectory("neural-code-predictor")
        defer { try? FileManager.default.removeItem(at: directory) }
        let (neural, _) = try await Qwen3TTSNeuralCodePredictor.load(
            predictor: predictor, topK: 50, cacheDirectory: directory, sourceKey: "tiny",
            computeUnits: .cpuOnly, requireNeuralEngine: false)
        #expect(neural.passes == NeuralFixture.codeGroups - 1)

        let random = MLXRandom.RandomState(seed: 5)
        let hidden = MLXRandom.normal([1, 1, NeuralFixture.hidden], key: random)
        let code0 = MLXRandom.normal([1, 1, NeuralFixture.hidden], key: random)
        let hiddenArray = try NeuralFixture.column(hidden)
        let code0Row = Qwen3TTSModel.rows(code0)[0]

        // Greedy.
        let (expectedCodes, expectedSum) = Device.withDefaultDevice(.cpu) {
            let (codes, sum) = predictor.predict(
                hidden: hidden, firstEmbedding: code0, cache: predictor.makeCache()
            ) { _, logits in argMax(logits, axis: -1, keepDims: true).asType(.int32) }
            return (concatenated(codes, axis: 1).asArray(Int32.self), sum.asArray(Float.self))
        }
        var unused = NeuralRandom(seed: 1)
        let greedy = try code0Row.withUnsafeBufferPointer {
            try neural.frame(hidden: hiddenArray, code0: $0, temperature: 0, random: &unused)
        }
        #expect(greedy.codes == expectedCodes)
        #expect(snr(greedy.embeddingSum.floats(), expectedSum) > 35)

        // Drawn at temperature 0.7, on the noise the same seed gives.
        let temperature: Float = 0.7
        var draws = NeuralRandom(seed: 9)
        let drawn = try code0Row.withUnsafeBufferPointer {
            try neural.frame(hidden: hiddenArray, code0: $0, temperature: temperature, random: &draws)
        }
        var replay = NeuralRandom(seed: 9)
        var noise = [Float](repeating: 0, count: neural.passes * NeuralFixture.codes)
        replay.fillGumbel(&noise)
        var shortfalls: [Float] = []
        let followedSum = Device.withDefaultDevice(.cpu) {
            predictor.predict(hidden: hidden, firstEmbedding: code0, cache: predictor.makeCache()) {
                group, logits in
                let z = Qwen3TTSSampler.filter(logits / temperature, topK: 50, topP: 1)
                    .asArray(Float.self)
                let values = z.indices.map { z[$0] + noise[group * NeuralFixture.codes + $0] }
                let code = Int(drawn.codes[group])
                shortfalls.append(values.max()! - values[code])
                return MLXArray([Int32(code)]).reshaped(1, 1)
            }.embeddingSum.asArray(Float.self)
        }
        #expect(shortfalls.allSatisfy { $0 < 0.05 }, "below the best by \(shortfalls)")
        #expect(snr(drawn.embeddingSum.floats(), followedSum) > 35)
    }

    /// A layer whose MLP product outgrows fp16 (a "massive activation") is
    /// found by measuring and keeps its down projection in fp16 with the
    /// product scaled to fit: the frame still matches MLX, where the CPU's
    /// fp16 would otherwise overflow.
    @Test func anOutlierLayerKeepsItsDownProjectionInFp16() async throws {
        let talker = try NeuralFixture.talker(quantized: false)
        let predictor = talker.codePredictor
        // Channel 5 of the second layer's MLP, inflated.
        let mlp = predictor.model.layers[1].mlp
        let gateUp = try #require(mlp.gateUpProj).weight
        let intermediate = gateUp.dim(0) / 2
        let boost = MLXArray((0 ..< gateUp.dim(0)).map {
            $0 == 5 || $0 == intermediate + 5 ? Float(600) : 1
        }).reshaped(-1, 1)
        try mlp.update(
            parameters: ModuleParameters.unflattened(["gate_up_proj.weight": gateUp * boost]),
            verify: .noUnusedKeys)

        let random = MLXRandom.RandomState(seed: 5)
        let hidden = MLXRandom.normal([1, 1, NeuralFixture.hidden], key: random)
        let code0 = MLXRandom.normal([1, 1, NeuralFixture.hidden], key: random)
        let ((expectedCodes, expectedSum), products) = Device.withDefaultDevice(.cpu) {
            measuringProducts(of: predictor.model.layers) {
                let (codes, sum) = predictor.predict(
                    hidden: hidden, firstEmbedding: code0, cache: predictor.makeCache()
                ) { _, logits in argMax(logits, axis: -1, keepDims: true).asType(.int32) }
                return (concatenated(codes, axis: 1).asArray(Int32.self), sum.asArray(Float.self))
            }
        }
        try #require(products[1] > 65_504, "the product must outgrow fp16: \(products)")
        let precision = NeuralPrecision(largestProducts: products)
        #expect(precision.outliers.keys.sorted() == [1])

        let directory = try makeTemporaryDirectory("neural-outlier")
        defer { try? FileManager.default.removeItem(at: directory) }
        let (neural, _) = try await Qwen3TTSNeuralCodePredictor.load(
            predictor: predictor, topK: 50, precision: precision, cacheDirectory: directory,
            sourceKey: "tiny", computeUnits: .cpuOnly, requireNeuralEngine: false)
        var unused = NeuralRandom(seed: 1)
        let frame = try Qwen3TTSModel.rows(code0)[0].withUnsafeBufferPointer {
            try neural.frame(
                hidden: try NeuralFixture.column(hidden), code0: $0, temperature: 0,
                random: &unused)
        }
        #expect(frame.codes == expectedCodes)
        #expect(snr(frame.embeddingSum.floats(), expectedSum) > 35)
    }

    /// Noise can lift a code inside the top k above the best, never one
    /// outside it.
    @Test func theCodePredictorDrawsOnlyFromTheTopK() async throws {
        let talker = try NeuralFixture.talker()
        let predictor = talker.codePredictor
        let directory = try makeTemporaryDirectory("neural-top-k")
        defer { try? FileManager.default.removeItem(at: directory) }
        let (neural, _) = try await Qwen3TTSNeuralCodePredictor.load(
            predictor: predictor, topK: 50, cacheDirectory: directory, sourceKey: "tiny",
            computeUnits: .cpuOnly, requireNeuralEngine: false)
        let random = MLXRandom.RandomState(seed: 8)
        let hidden = MLXRandom.normal([1, 1, NeuralFixture.hidden], key: random)
        let code0 = MLXRandom.normal([1, 1, NeuralFixture.hidden], key: random)
        let hiddenArray = try NeuralFixture.column(hidden)

        // The first pass's codes, best first, as MLX ranks them.
        var ranking: [Int] = []
        Device.withDefaultDevice(.cpu) {
            _ = predictor.predict(hidden: hidden, firstEmbedding: code0, cache: predictor.makeCache()) {
                group, logits in
                if group == 0 {
                    let values = logits.asArray(Float.self)
                    ranking = values.indices.sorted { values[$0] > values[$1] }
                }
                return argMax(logits, axis: -1, keepDims: true).asType(.int32)
            }
        }

        /// The first code drawn when `boosted` gets noise far above the rest.
        func firstCode(boosting boosted: Int) throws -> Int {
            let noise = try MLMultiArray(
                shape: [1, NSNumber(value: NeuralFixture.codes), 1, NSNumber(value: neural.passes)],
                dataType: .float16)
            noise.fill(0)
            noise.withUnsafeMutableBufferPointer(ofType: Float16.self) { buffer, strides in
                buffer[boosted * strides[1]] = 40
            }
            return try Qwen3TTSModel.rows(code0)[0].withUnsafeBufferPointer {
                Int(
                    try neural.frame(
                        hidden: hiddenArray, code0: $0, noise: noise, inverseTemperature: 1
                    ).codes[0])
            }
        }
        #expect(try firstCode(boosting: ranking[44]) == ranking[44])
        #expect(try firstCode(boosting: ranking[59]) == ranking[0])
    }

    // MARK: - Sampling on the host

    /// The talker's first code: never a control code, never EOS in the
    /// first two frames, a recent code penalized.
    @Test func theHostSamplerAppliesTheTalkersRules() {
        var sampler = Qwen3TTSHostTalkerSampler(
            sampling: Qwen3TTSSampling(temperature: 0, repetitionPenalty: 1.05), vocabSize: 3072,
            eosTokenID: 3050)
        var logits = [Float](repeating: -5, count: 3072)
        logits[3050] = 9  // EOS
        logits[3060] = 8  // a control code
        logits[7] = 2
        logits[9] = 1.95
        var random = NeuralRandom(seed: 1)
        #expect(sampler(logits, frame: 0, random: &random) == 7)
        // 7 was just drawn: 2 / 1.05 < 1.95.
        #expect(sampler(logits, frame: 1, random: &random) == 9)
        #expect(sampler(logits, frame: 2, random: &random) == 3050)
    }

    /// Top-k keeps ties at the cut; top-p keeps the nucleus.
    @Test func theHostDrawKeepsTiesAndTheNucleus() {
        var random = NeuralRandom(seed: 2)
        var logits = [Float](repeating: -10, count: 100)
        logits[0] = 5
        (1 ... 3).forEach { logits[$0] = 4.5 }
        let tied = Set(
            (0 ..< 400).map { _ in
                Qwen3TTSHostTalkerSampler.draw(
                    logits, temperature: 1, topK: 2, topP: 1, random: &random)
            })
        #expect(tied == [0, 1, 2, 3])

        logits[0] = 12
        let nucleus = Set(
            (0 ..< 200).map { _ in
                Qwen3TTSHostTalkerSampler.draw(
                    logits, temperature: 1, topK: 50, topP: 0.9, random: &random)
            })
        #expect(nucleus == [0])
    }

    // MARK: - Generation

    /// A prepared voice renders through the Neural Engine path (here on the
    /// CPU): audio for every frame, an alignment row per frame, and the same
    /// render again for the same seed.
    @Test func aPreparedVoiceRendersOnTheNeuralEnginePath() async throws {
        let fixture = try await TinyModel.make(
            ttsModelType: "custom_voice", speakerIDEntries: #""ryan": 100"#,
            dialectEntries: #""ryan": false"#)
        defer { fixture.cleanUp() }
        let directory = try makeTemporaryDirectory("neural-voice")
        defer { try? FileManager.default.removeItem(at: directory) }
        let head = Qwen3TTSAlignmentHead(layer: 1, head: 1)
        let report = try await fixture.model.prepareNeuralVoice(
            cacheDirectory: directory, contextLength: 64, alignment: head, computeUnits: .cpuOnly)
        #expect(report.contains("dB"))

        func render(seed: UInt64) async throws -> Collected {
            try await collect(
                fixture.model.generateStream(
                    text: "target voice prompt one two three", voice: "ryan", language: "English",
                    sampling: Qwen3TTSSampling(maxTokens: 6), seed: seed, streamingInterval: 0.08,
                    alignment: head))
        }
        let first = try await render(seed: 4)
        let frames = try #require(first.codeFrames)
        #expect(frames.count >= 2)
        #expect(frames.allSatisfy { $0.count == 2 })
        #expect(first.allAudio.count == frames.count * TinyModel.samplesPerFrame)
        #expect(first.alignment.count == frames.count)
        let again = try await render(seed: 4)
        #expect(again.codeFrames == frames)
        #expect(again.allAudio == first.allAudio)
    }
}

// MARK: - Fixtures

enum NeuralFixture {
    static let hidden = 64
    static let codeGroups = 5
    static let codes = 64

    /// A small talker, 8-bit unless `quantized` is off: three layers of
    /// width 64, and a code predictor of width 32 (so its input projection
    /// runs) with five code groups of 64 codes. Random weights; the norms'
    /// weights near 1.
    static func talker(seed: UInt64 = 7, quantized: Bool = true) throws -> Qwen3TTSTalker {
        let json = """
            {
              "vocab_size": 3072, "hidden_size": 64, "intermediate_size": 96,
              "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 2,
              "head_dim": 16, "max_position_embeddings": 128, "num_code_groups": 5,
              "text_hidden_size": 32, "text_vocab_size": 64,
              "codec_eos_token_id": 3050, "codec_think_id": 3051, "codec_nothink_id": 3052,
              "codec_think_bos_id": 3053, "codec_think_eos_id": 3054, "codec_pad_id": 3055,
              "codec_bos_id": 3056,
              "code_predictor_config": {
                "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
                "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
                "head_dim": 8, "max_position_embeddings": 128, "num_code_groups": 5
              }
            }
            """
        let config = try JSONDecoder().decode(Qwen3TTSTalkerConfig.self, from: Data(json.utf8))
        MLXRandom.seed(seed)
        let talker = Qwen3TTSTalker(config: config)
        let random = MLXRandom.RandomState(seed: seed)
        let norms = talker.parameters().flattened().filter { $0.0.hasSuffix("norm.weight") }
            .map { key, value in (key, 1 + MLXRandom.normal(value.shape, key: random) * 0.2) }
        try talker.update(parameters: ModuleParameters.unflattened(norms), verify: .noUnusedKeys)
        if quantized { quantize(model: talker, groupSize: 32, bits: 8) }
        eval(talker.parameters())
        return talker
    }

    /// `[1, 1, D]` as the `[1, D, 1, 1]` fp16 column the graphs take.
    static func column(_ x: MLXArray) throws -> MLMultiArray {
        let row = Qwen3TTSModel.rows(x)[0]
        let array = try MLMultiArray(shape: [1, NSNumber(value: row.count), 1, 1], dataType: .float16)
        row.withUnsafeBufferPointer { array.copy(from: $0) }
        return array
    }
}

/// `actual` against `expected`, in dB.
func snr(_ actual: [Float], _ expected: [Float]) -> Double {
    Qwen3TTSNeuralCodec.snr(actual, expected)
}
