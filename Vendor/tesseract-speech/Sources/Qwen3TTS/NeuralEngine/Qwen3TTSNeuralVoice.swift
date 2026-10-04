import Accelerate
import CoreML
import Foundation
@preconcurrency import MLX

// The talker and the code predictor on the Neural Engine (ADR-0084, 0088): the
// Core ML models `Qwen3TTSNeuralGraphs` writes, built from the checkpoint on
// this machine and cached, and what drives them from Swift.

// MARK: - The talker

/// The talker's step on the Neural Engine: one position per call, the KV
/// cache kept in the model's state.
///
/// Positions are bounded by `contextLength`, the cache the graph was built
/// for: a prompt and its frames must fit.
final class Qwen3TTSNeuralTalker: @unchecked Sendable {
    let contextLength: Int
    let hidden: Int
    let headDim: Int
    let ropeBase: Float
    let vocabulary: Int
    let cacheWidth: Int
    /// The head whose scores each step returns, when the graph has one.
    let alignment: Qwen3TTSAlignmentHead?
    private let model: MLModel

    private init(
        talker: Qwen3TTSTalker, contextLength: Int, alignment: Qwen3TTSAlignmentHead?,
        model: MLModel
    ) {
        let g = NeuralGeometry(talker.model.layers[0])
        self.contextLength = contextLength
        self.hidden = g.hidden
        self.headDim = g.headDim
        self.ropeBase = g.ropeBase
        self.vocabulary = talker.codecHead.shape.0
        self.cacheWidth = talker.model.layers.count * g.kvWidth
        self.alignment = alignment
        self.model = model
    }

    /// The talker for `talker`'s weights, from `cacheDirectory` or built
    /// there with `precision`. Throws when the compute plan would put any op
    /// off the Neural Engine (unless `requireNeuralEngine` is off, for tests
    /// on the CPU).
    static func load(
        talker: Qwen3TTSTalker, contextLength: Int, alignment: Qwen3TTSAlignmentHead?,
        precision: NeuralPrecision = .int8, cacheDirectory: URL, sourceKey: String,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine, requireNeuralEngine: Bool = true
    ) async throws -> (talker: Qwen3TTSNeuralTalker, placement: NeuralPlacement) {
        let head = alignment.map { "\($0.layer).\($0.head)" } ?? "none"
        let compiled = try await NeuralModelStore.compiled(
            key: NeuralModelStore.key(
                "qwen3tts-talker",
                "v\(Qwen3TTSNeuralTalkerGraph.version) L\(contextLength) a\(head) \(precision) "
                    + sourceKey),
            in: cacheDirectory, target: .coreML8,
            metadata: [
                "com.tesseract.qwen3tts.talker": "step, \(contextLength) positions",
                "com.tesseract.qwen3tts.graphVersion": "\(Qwen3TTSNeuralTalkerGraph.version)",
            ]
        ) {
            Qwen3TTSNeuralTalkerGraph.build(
                $0, talker: talker, contextLength: contextLength, alignment: alignment,
                precision: precision)
        }
        let (model, placement) = try await NeuralModelStore.load(
            compiled, computeUnits: computeUnits, requireNeuralEngine: requireNeuralEngine,
            what: "The talker")
        return (
            Qwen3TTSNeuralTalker(
                talker: talker, contextLength: contextLength, alignment: alignment, model: model),
            placement
        )
    }

    /// One generation's cache and inputs. Its steps run one at a time.
    final class Session: @unchecked Sendable {
        let state: MLState
        /// The next position: how many the cache holds.
        fileprivate(set) var position = 0
        fileprivate let x: MLMultiArray
        fileprivate let cos: MLMultiArray
        fileprivate let sin: MLMultiArray
        fileprivate let mask: MLMultiArray

        fileprivate init(state: MLState, hidden: Int, headDim: Int, length: Int) throws {
            self.state = state
            x = try MLMultiArray(shape: [1, NSNumber(value: hidden), 1, 1], dataType: .float16)
            cos = try MLMultiArray(shape: [1, 1, NSNumber(value: headDim), 1], dataType: .float16)
            sin = try MLMultiArray(shape: [1, 1, NSNumber(value: headDim), 1], dataType: .float16)
            mask = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: length)], dataType: .float16)
            mask.fill(Float16(neuralExcluded))
        }
    }

    func makeSession() throws -> Session {
        try Session(state: model.makeState(), hidden: hidden, headDim: headDim, length: contextLength)
    }

    /// What one step returns.
    struct Step: @unchecked Sendable {
        /// The codec logits, `vocabulary` values.
        var logits: [Float]
        /// The final hidden state `[1, hidden, 1, 1]` fp16: the code
        /// predictor's input as it is.
        var hidden: MLMultiArray
        /// The alignment head's scaled scores over the positions before
        /// this one, then this one: `position + 1` values.
        var alignment: [Float]?
    }

    /// Runs `x` (`hidden` values) at `session`'s next position and stores
    /// its keys and values there.
    func step(_ x: UnsafeBufferPointer<Float16>, session: Session) throws -> Step {
        let p = session.position
        guard p < contextLength else {
            throw AudioGenerationError.invalidInput(
                "The talker's cache holds \(contextLength) positions.")
        }
        precondition(x.count == hidden)
        session.x.copy(from: x)
        let (cos, sin) = neuralRopeTables(positions: p ..< (p + 1), headDim: headDim, base: ropeBase)
        session.cos.copy(cos)
        session.sin.copy(sin)
        let output = try model.prediction(
            from: MLDictionaryFeatureProvider(dictionary: [
                "x": MLFeatureValue(multiArray: session.x),
                "cos": MLFeatureValue(multiArray: session.cos),
                "sin": MLFeatureValue(multiArray: session.sin),
                "mask": MLFeatureValue(multiArray: session.mask),
            ]), using: session.state)
        func array(_ name: String) throws -> MLMultiArray {
            guard let value = output.featureValue(for: name)?.multiArrayValue else {
                throw AudioGenerationError.modelNotInitialized("The talker returned no \(name).")
            }
            return value
        }
        for name in ["keys", "values"] {
            let new = try array("new_\(name)")
            session.state.withMultiArray(for: name) { $0.setColumn(p, from: new) }
        }
        session.mask.setValue(0, at: p)
        session.position = p + 1
        var alignmentRow: [Float]?
        if alignment != nil {
            let scores = try array("alignment").floats()
            alignmentRow = Array(scores[..<p]) + [scores[contextLength]]
        }
        return Step(
            logits: try array("logits").floats(), hidden: try array("hidden"),
            alignment: alignmentRow)
    }
}

// MARK: - The code predictor

/// The code predictor on the Neural Engine: a whole frame per call, its
/// codes sampled in the graph on Gumbel noise from the host.
///
/// Its input buffers are reused: one frame at a time.
final class Qwen3TTSNeuralCodePredictor: @unchecked Sendable {
    /// Codes per call: every codebook after the first.
    let passes: Int
    let vocabulary: Int
    let talkerHidden: Int
    private let model: MLModel
    private let code0: MLMultiArray
    private let noise: MLMultiArray
    private let inverseTemperature: MLMultiArray
    /// Scratch for the noise, `passes × vocabulary`, pass-major.
    private var gumbel: [Float]

    private init(predictor: Qwen3TTSCodePredictor, model: MLModel) throws {
        passes = predictor.numCodeGroups - 1
        vocabulary = predictor.lmHead[0].shape.0
        talkerHidden = predictor.codecEmbedding[0].shape.1
        self.model = model
        code0 = try MLMultiArray(shape: [1, NSNumber(value: talkerHidden), 1, 1], dataType: .float16)
        noise = try MLMultiArray(
            shape: [1, NSNumber(value: vocabulary), 1, NSNumber(value: passes)], dataType: .float16)
        inverseTemperature = try MLMultiArray(shape: [1, 1, 1, 1], dataType: .float16)
        gumbel = [Float](repeating: 0, count: passes * vocabulary)
    }

    /// The code predictor for `predictor`'s weights, from `cacheDirectory`
    /// or built there with `precision`, sampling among the top `topK`;
    /// `forced`, the bench's variant that takes the codes and returns logits.
    static func load(
        predictor: Qwen3TTSCodePredictor, topK: Int, precision: NeuralPrecision = .int8,
        forced: Bool = false, cacheDirectory: URL, sourceKey: String,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine, requireNeuralEngine: Bool = true
    ) async throws -> (predictor: Qwen3TTSNeuralCodePredictor, placement: NeuralPlacement) {
        let variant = forced ? "forced" : "k\(topK)"
        let compiled = try await NeuralModelStore.compiled(
            key: NeuralModelStore.key(
                "qwen3tts-code-predictor",
                "v\(Qwen3TTSNeuralCodePredictorGraph.version) \(variant) \(precision) "
                    + sourceKey),
            in: cacheDirectory, target: .coreML8,
            metadata: [
                "com.tesseract.qwen3tts.codePredictor": "one frame, \(variant)",
                "com.tesseract.qwen3tts.graphVersion": "\(Qwen3TTSNeuralCodePredictorGraph.version)",
            ]
        ) {
            Qwen3TTSNeuralCodePredictorGraph.build(
                $0, predictor: predictor, topK: topK, precision: precision, forced: forced)
        }
        let (model, placement) = try await NeuralModelStore.load(
            compiled, computeUnits: computeUnits, requireNeuralEngine: requireNeuralEngine,
            what: "The code predictor")
        return (try Qwen3TTSNeuralCodePredictor(predictor: predictor, model: model), placement)
    }

    /// One frame: `hidden` is the talker step's, `code0` the talker-side
    /// embedding of the frame's first code. Draws at `temperature` with
    /// noise from `random`; at 0, takes the best code. Returns the other
    /// codes and the sum of all the frame's code embeddings, the codec half
    /// of the talker's next input.
    func frame(
        hidden: MLMultiArray, code0 row: UnsafeBufferPointer<Float16>, temperature: Float,
        random: inout NeuralRandom
    ) throws -> (codes: [Int32], embeddingSum: MLMultiArray) {
        if temperature > 0 {
            random.fillGumbel(&gumbel)
            noise.setColumns(passMajor: gumbel, columns: passes)
        } else {
            noise.fill(0)
        }
        return try frame(
            hidden: hidden, code0: row, noise: noise,
            inverseTemperature: temperature > 0 ? 1 / temperature : 1)
    }

    /// One frame on the given noise `[1, vocabulary, 1, passes]`.
    func frame(
        hidden: MLMultiArray, code0 row: UnsafeBufferPointer<Float16>, noise: MLMultiArray,
        inverseTemperature scale: Float
    ) throws -> (codes: [Int32], embeddingSum: MLMultiArray) {
        code0.copy(from: row)
        inverseTemperature.setValue(Float16(scale), at: 0)
        let output = try model.prediction(
            from: MLDictionaryFeatureProvider(dictionary: [
                "hidden": MLFeatureValue(multiArray: hidden),
                "code0_embed": MLFeatureValue(multiArray: code0),
                "noise": MLFeatureValue(multiArray: noise),
                "inverse_temperature": MLFeatureValue(multiArray: inverseTemperature),
            ]))
        guard let codes = output.featureValue(for: "codes")?.multiArrayValue,
            let sum = output.featureValue(for: "embed_sum")?.multiArrayValue
        else {
            throw AudioGenerationError.modelNotInitialized("The code predictor returned nothing.")
        }
        return (codes.floats().map { Int32($0.rounded()) }, sum)
    }

    /// The forced variant's logits for a frame whose other codes are
    /// `codes`: `passes × vocabulary`, pass after pass.
    func logits(hidden: MLMultiArray, code0 row: UnsafeBufferPointer<Float16>, codes: [Int32])
        throws -> [Float]
    {
        code0.copy(from: row)
        noise.fill(0)
        noise.withUnsafeMutableBufferPointer(ofType: Float16.self) { buffer, strides in
            for (pass, code) in codes.enumerated() {
                buffer[Int(code) * strides[1] + pass * strides[3]] = 1
            }
        }
        let output = try model.prediction(
            from: MLDictionaryFeatureProvider(dictionary: [
                "hidden": MLFeatureValue(multiArray: hidden),
                "code0_embed": MLFeatureValue(multiArray: code0),
                "codes_onehot": MLFeatureValue(multiArray: noise),
            ]))
        guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
            throw AudioGenerationError.modelNotInitialized("The code predictor returned nothing.")
        }
        // [1, V, 1, passes] in index order is code-major: transpose.
        let values = logits.floats()
        return (0 ..< passes).flatMap { pass in
            (0 ..< vocabulary).map { values[$0 * passes + pass] }
        }
    }
}

// MARK: - Sampling on the host

/// A seeded generator for the Neural Engine path's draws (xoshiro256**,
/// seeded through SplitMix64): the same seed gives the same render.
struct NeuralRandom: Sendable {
    private var s: (UInt64, UInt64, UInt64, UInt64)

    init(seed: UInt64) {
        var z = seed
        func splitMix() -> UInt64 {
            z &+= 0x9E37_79B9_7F4A_7C15
            var x = z
            x = (x ^ (x >> 30)) &* 0xBF58_476D_1CE4_E5B9
            x = (x ^ (x >> 27)) &* 0x94D0_49BB_1331_11EB
            return x ^ (x >> 31)
        }
        s = (splitMix(), splitMix(), splitMix(), splitMix())
    }

    mutating func next() -> UInt64 {
        let result = ((s.1 &* 5) << 7 | (s.1 &* 5) >> 57) &* 9
        let t = s.1 << 17
        s.2 ^= s.0
        s.3 ^= s.1
        s.1 ^= s.2
        s.0 ^= s.3
        s.2 ^= t
        s.3 = s.3 << 45 | s.3 >> 19
        return result
    }

    /// Uniform in (0, 1), never either end.
    mutating func uniform() -> Float {
        (Float(next() >> 40) + 0.5) / Float(1 << 24)
    }

    /// Fills `values` with standard Gumbel noise, `-log(-log(u))`.
    mutating func fillGumbel(_ values: inout [Float]) {
        for i in values.indices { values[i] = uniform() }
        var count = Int32(values.count)
        values.withUnsafeMutableBufferPointer { buffer in
            let p = buffer.baseAddress!
            vvlogf(p, p, &count)
            vDSP_vneg(p, 1, p, 1, vDSP_Length(buffer.count))
            vvlogf(p, p, &count)
            vDSP_vneg(p, 1, p, 1, vDSP_Length(buffer.count))
        }
    }
}

/// The talker's sampler on the host: `Qwen3TTSTalkerSampler`'s processors
/// (control codes suppressed, EOS held back for the first frames, the
/// windowed repetition penalty) and Qwen's draw (temperature, top-k keeping
/// ties, top-p), over a step's logits in Swift.
struct Qwen3TTSHostTalkerSampler {
    let temperature: Float
    let topK: Int
    let topP: Float
    let penalty: Float
    let eos: Int
    let window: Int
    /// The codes no frame may start with: the control range but EOS.
    private let suppressed: Range<Int>
    /// The last `window` codes, oldest first.
    private(set) var recent: [Int] = []

    init(
        sampling: Qwen3TTSSampling, vocabSize: Int, eosTokenID: Int,
        window: Int = Qwen3TTSSampling.repetitionWindow
    ) {
        temperature = sampling.temperature
        topK = Qwen3TTSSampling.topK
        topP = sampling.topP
        penalty = sampling.repetitionPenalty
        eos = eosTokenID
        self.window = window
        suppressed = max(0, vocabSize - 1024) ..< vocabSize
    }

    mutating func callAsFunction(_ logits: [Float], frame: Int, random: inout NeuralRandom) -> Int {
        var x = logits
        if penalty != 1 {
            for token in Set(recent) {
                x[token] = x[token] < 0 ? x[token] * penalty : x[token] / penalty
            }
        }
        for id in suppressed where id != eos { x[id] = -.infinity }
        if frame < Qwen3TTSTalkerSampler.minFrames { x[eos] = -.infinity }
        let token = Self.draw(x, temperature: temperature, topK: topK, topP: topP, random: &random)
        recent.append(token)
        if recent.count > window { recent.removeFirst(recent.count - window) }
        return token
    }

    /// Temperature, top-k (ties at the cut kept), top-p, then a draw; at
    /// temperature 0 the best token.
    static func draw(
        _ logits: [Float], temperature: Float, topK: Int, topP: Float, random: inout NeuralRandom
    ) -> Int {
        guard temperature > 0 else {
            // The first of equal bests, as `argMax`.
            return logits.indices.max {
                logits[$0] < logits[$1] || (logits[$0] == logits[$1] && $0 > $1)
            }!
        }
        let scaled = logits.map { $0 / temperature }
        var order = scaled.indices.filter { scaled[$0] > -.infinity }
        guard !order.isEmpty else { return 0 }
        order.sort { scaled[$0] > scaled[$1] }
        if topK > 0, topK < order.count {
            let cut = scaled[order[topK - 1]]
            order = order.filter { scaled[$0] >= cut }
        }
        let best = scaled[order[0]]
        var weights = order.map { exp(scaled[$0] - best) }
        if topP > 0, topP < 1 {
            // Smallest first, drop while the running share stays within 1 - top-p.
            let total = weights.reduce(0, +)
            var running: Float = 0
            for i in weights.indices.reversed() {
                running += weights[i] / total
                if running <= 1 - topP { weights[i] = 0 } else { break }
            }
        }
        let total = weights.reduce(0, +)
        var u = random.uniform() * total
        for (i, w) in weights.enumerated() {
            u -= w
            if u <= 0, w > 0 { return order[i] }
        }
        return order[weights.lastIndex { $0 > 0 } ?? 0]
    }
}

// MARK: - Multi-array access

extension MLMultiArray {
    /// Every value, fp16 read as Float, in index order.
    func floats() -> [Float] {
        let shape = self.shape.map(\.intValue)
        let strides = self.strides.map(\.intValue)
        var out = [Float](repeating: 0, count: shape.reduce(1, *))
        withUnsafeBufferPointer(ofType: Float16.self) { buffer in
            var index = [Int](repeating: 0, count: shape.count)
            for i in out.indices {
                var offset = 0
                for d in shape.indices { offset += index[d] * strides[d] }
                out[i] = Float(buffer[offset])
                var d = shape.count - 1
                while d >= 0 {
                    index[d] += 1
                    if index[d] < shape[d] { break }
                    index[d] = 0
                    d -= 1
                }
            }
        }
        return out
    }

    /// `[1, C, 1, 1]` (or any shape of `values.count` channels along axis 1)
    /// from `values`.
    func copy(from values: UnsafeBufferPointer<Float16>) {
        withUnsafeMutableBufferPointer(ofType: Float16.self) { buffer, strides in
            let stride = strides[1]
            for (c, v) in values.enumerated() { buffer[c * stride] = v }
        }
    }

    /// A `[1, 1, n, 1]` rotation table from `values`.
    func copy(_ values: [Float]) {
        withUnsafeMutableBufferPointer(ofType: Float16.self) { buffer, strides in
            let stride = strides[2]
            for (i, v) in values.enumerated() { buffer[i * stride] = Float16(v) }
        }
    }

    /// The last axis's element `i` of a `[1, 1, 1, n]` array.
    func setValue(_ value: Float16, at i: Int) {
        withUnsafeMutableBufferPointer(ofType: Float16.self) { buffer, strides in
            buffer[i * strides[3]] = value
        }
    }

    func fill(_ value: Float16) {
        withUnsafeMutableBufferPointer(ofType: Float16.self) { buffer, _ in
            buffer.update(repeating: value)
        }
    }

    /// Column `p` of a `[1, C, 1, L]` array from a `[1, C, 1, 1]` one.
    func setColumn(_ p: Int, from column: MLMultiArray) {
        let source = column.strides[1].intValue
        column.withUnsafeBufferPointer(ofType: Float16.self) { values in
            withUnsafeMutableBufferPointer(ofType: Float16.self) { buffer, strides in
                let (channel, position) = (strides[1], strides[3])
                for c in 0 ..< shape[1].intValue {
                    buffer[c * channel + p * position] = values[c * source]
                }
            }
        }
    }

    /// A `[1, C, 1, n]` array from `values`, `n × C`, column-major (one
    /// column after another).
    func setColumns(passMajor values: [Float], columns n: Int) {
        let channels = shape[1].intValue
        withUnsafeMutableBufferPointer(ofType: Float16.self) { buffer, strides in
            let (channel, position) = (strides[1], strides[3])
            for p in 0 ..< n {
                for c in 0 ..< channels {
                    buffer[c * channel + p * position] = Float16(values[p * channels + c])
                }
            }
        }
    }
}
