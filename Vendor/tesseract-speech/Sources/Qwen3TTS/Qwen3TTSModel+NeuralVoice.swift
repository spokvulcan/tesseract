import CoreML
import Foundation
@preconcurrency import MLX
import MLXNN
import os

/// The talker and the code predictor on the Neural Engine, and what the host
/// keeps beside them: the talker's codec embedding table, whose row for a
/// frame's first code the code predictor starts from.
final class Qwen3TTSNeuralVoice: @unchecked Sendable {
    let talker: Qwen3TTSNeuralTalker
    let codePredictor: Qwen3TTSNeuralCodePredictor
    /// The prompts' text projection in dense fp32, which MLX's CPU runs
    /// through Accelerate: with the checkpoint's 8-bit weights it takes about
    /// 30 ms per text token there, all before the first audio.
    let prompts: Qwen3TTSPromptBuilder
    /// What each graph was built with.
    let talkerPrecision: NeuralPrecision
    let codePredictorPrecision: NeuralPrecision
    /// `vocabulary × hidden` fp16, row-major.
    private let codecTable: [Float16]
    private let hidden: Int

    init(
        talker: Qwen3TTSNeuralTalker, codePredictor: Qwen3TTSNeuralCodePredictor,
        talkerPrecision: NeuralPrecision, codePredictorPrecision: NeuralPrecision,
        codecEmbedding: MLXArray, prompts: Qwen3TTSPromptBuilder
    ) {
        self.talker = talker
        self.codePredictor = codePredictor
        self.prompts = prompts
        self.talkerPrecision = talkerPrecision
        self.codePredictorPrecision = codePredictorPrecision
        hidden = codecEmbedding.dim(1)
        codecTable = codecEmbedding.asType(.float16).asArray(Float16.self)
    }

    /// Code `code`'s row of the talker's codec embedding.
    func withCodecRow<R>(_ code: Int, _ body: (UnsafeBufferPointer<Float16>) throws -> R) rethrows -> R {
        try codecTable.withUnsafeBufferPointer {
            try body(UnsafeBufferPointer(rebasing: $0[(code * hidden) ..< ((code + 1) * hidden)]))
        }
    }
}

// MARK: - Preparing

/// A stage of preparing the Neural Engine voice, for its progress.
public enum NeuralVoicePhase: Sendable, Equatable {
    /// MLX measures the checkpoint on the CPU: the first time only.
    case measuring
    /// A graph is built and compiled, or loaded as built before.
    case building(String)
    /// The graphs are checked against MLX: the first time only.
    case checking
}

/// What a preparation found that the next one can trust: the precision MLX
/// measured, and the check the graphs passed with it.
struct NeuralVoiceManifest: Codable {
    var talker: NeuralPrecision
    var codePredictor: NeuralPrecision
    var check: String
}

extension Qwen3TTSModel {
    /// Moves the talker and the code predictor to the Neural Engine
    /// (ADR-0084, ADR-0085). MLX first runs a short probe on the CPU, which
    /// measures how large each layer's MLP product grows (`NeuralPrecision`);
    /// then their Core ML models are built from this checkpoint with that
    /// precision (or loaded, built before, from `cacheDirectory`), every op
    /// must land on the Neural Engine, and they must follow MLX on the probe.
    /// From then on generation runs them there, and runs MLX only on the
    /// CPU: the prompt's embeddings and the codec's front end.
    ///
    /// A preparation that passed leaves a manifest in `cacheDirectory`, so
    /// the next one for this checkpoint and these graphs skips the probe and
    /// the check (MLX on the CPU, the slow part) and only loads the models.
    ///
    /// `contextLength` bounds a prompt and its frames. `alignment` is the
    /// head word timing reads, fixed in the talker's graph; generations
    /// asking for another get no rows. With `.cpuOnly` (tests) the models
    /// run on the CPU and placement isn't checked. Returns what the compute
    /// plans and the check found.
    @discardableResult
    public func prepareNeuralVoice(
        cacheDirectory: URL, contextLength: Int = 512, alignment: Qwen3TTSAlignmentHead? = nil,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        onPhase: (@Sendable (NeuralVoicePhase) -> Void)? = nil
    ) async throws -> String {
        let requireNeuralEngine = computeUnits != .cpuOnly
        let source = checkpointKey()
        let alignment = validated(alignment)
        let head = alignment.map { "\($0.layer).\($0.head)" } ?? "none"
        let manifestURL = cacheDirectory.appendingPathComponent(
            NeuralModelStore.key(
                "qwen3tts-voice",
                "talker v\(Qwen3TTSNeuralTalkerGraph.version) L\(contextLength) a\(head) "
                    + "code predictor v\(Qwen3TTSNeuralCodePredictorGraph.version) \(computeUnits.rawValue) "
                    + source) + ".json")
        let manifest = try? JSONDecoder().decode(
            NeuralVoiceManifest.self, from: Data(contentsOf: manifestURL))
        let clock = ContinuousClock()
        var started = clock.now
        var stages: [String] = []
        func lap(_ stage: String) {
            let now = clock.now
            stages.append(String(format: "%@ %.1f s", stage, (now - started) / .seconds(1)))
            started = now
        }
        var reference: NeuralReference?
        if manifest == nil {
            onPhase?(.measuring)
            reference = try neuralReference()
            lap("MLX probe")
        }
        let talkerPrecision =
            manifest?.talker ?? NeuralPrecision(largestProducts: reference!.talkerProducts)
        let predictorPrecision =
            manifest?.codePredictor
            ?? NeuralPrecision(largestProducts: reference!.codePredictorProducts)
        onPhase?(.building("talker"))
        let (neuralTalker, talkerPlacement) = try await Qwen3TTSNeuralTalker.load(
            talker: talker, contextLength: contextLength, alignment: alignment,
            precision: talkerPrecision, cacheDirectory: cacheDirectory, sourceKey: source,
            computeUnits: computeUnits, requireNeuralEngine: requireNeuralEngine)
        lap("talker")
        onPhase?(.building("code predictor"))
        let (predictor, predictorPlacement) = try await Qwen3TTSNeuralCodePredictor.load(
            predictor: talker.codePredictor, topK: Qwen3TTSSampling.topK,
            precision: predictorPrecision, cacheDirectory: cacheDirectory, sourceKey: source,
            computeUnits: computeUnits, requireNeuralEngine: requireNeuralEngine)
        lap("code predictor")
        let voice = try Device.withDefaultDevice(.cpu) {
            let projection = talker.textProjection
            func dense(_ linear: Linear) -> Linear {
                Linear(weight: NeuralConstants.dense(linear), bias: linear.bias?.asType(.float32))
            }
            let (fc1, fc2) = (dense(projection.fc1), dense(projection.fc2))
            eval(fc1.parameters(), fc2.parameters())
            return Qwen3TTSNeuralVoice(
                talker: neuralTalker, codePredictor: predictor, talkerPrecision: talkerPrecision,
                codePredictorPrecision: predictorPrecision,
                codecEmbedding: NeuralConstants.dense(talker.model.codecEmbedding),
                prompts: try Qwen3TTSPromptBuilder(
                    config: config, tokenizer: tokenizer, talker: talker,
                    textEmbedding: textEmbedding,
                    projection: { fc2(silu(fc1($0.asType(.float32)))) }))
        }
        lap("host tables")
        var checked = manifest?.check ?? ""
        if let reference {
            onPhase?(.checking)
            let check = try neuralAgreement(voice, reference)
            lap("check")
            guard check.passes else {
                throw AudioGenerationError.modelNotInitialized(
                    "The Neural Engine voice differs from MLX's: talker \(talkerPrecision), "
                        + "code predictor \(predictorPrecision); \(check).")
            }
            checked = check.description
            try? JSONEncoder().encode(
                NeuralVoiceManifest(
                    talker: talkerPrecision, codePredictor: predictorPrecision, check: checked)
            ).write(to: manifestURL)
        }
        lock.withLock { neuralVoice = voice }
        return "talker: \(talkerPlacement), \(talkerPrecision); "
            + "code predictor: \(predictorPlacement), \(predictorPrecision); \(checked)"
            + (manifest == nil ? "" : " (checked before)") + "; " + stages.joined(separator: ", ")
    }

    /// Back to the talker and the code predictor in MLX, while they are
    /// still loaded (`releaseMLXVoice`).
    public func disableNeuralVoice() {
        lock.withLock { neuralVoice = nil }
    }

    /// Frees the MLX weights a prepared Neural Engine voice no longer reads:
    /// the talker's layers, norm, head and 8-bit text projection, and the
    /// whole code predictor (about 650 MB for the 0.6B). What stays is what
    /// the prompt needs, the codec embedding and the text table, and the
    /// codec. MLX can't generate after this; a new load can.
    public func releaseMLXVoice() {
        guard lock.withLock({ neuralVoice }) != nil else { return }
        Device.withDefaultDevice(.cpu) {
            let empty: (MLXArray) -> MLXArray = { _ in MLXArray.zeros([0]) }
            for layer in talker.model.layers { layer.apply(map: empty) }
            talker.model.norm.apply(map: empty)
            talker.codecHead.apply(map: empty)
            talker.textProjection.apply(map: empty)
            talker.codePredictor.apply(map: empty)
        }
        lock.withLock { mlxVoiceReleased = true }
        Memory.clearCache()
    }

    /// MLX on the CPU over a short prompt: the talker's logits and hidden
    /// state after it, and every layer's largest MLP product, the talker's
    /// over the prompt and the code predictor's over one greedy frame.
    struct NeuralReference {
        var prompt: Qwen3TTSPrompt
        var logits: [Float]
        var hidden: MLXArray
        /// The frame's first code: the best that isn't a control code.
        var first: Int
        var talkerProducts: [Float]
        var codePredictorProducts: [Float]
    }

    func neuralReference() throws -> NeuralReference {
        try Device.withDefaultDevice(.cpu) {
            let speaker = isCustomVoice ? talkerConfig.spkId?.keys.sorted().first : nil
            let prompt = try prompts.plain(
                text: "Hello there.", instruct: nil, language: "English", speaker: speaker,
                layout: .interleaved)
            let ((logits, hidden), talkerProducts) = measuringProducts(of: talker.model.layers) {
                talker.prefill(prompt.body, cache: talker.makeCache(capacity: prompt.body.dim(1)))
            }
            let values = logits.asType(.float32).asArray(Float.self)
            let control = max(0, values.count - 1024)
            let first = (0 ..< control).max { values[$0] < values[$1] }!
            // The probes evaluate every pass as it is built.
            let (_, predictorProducts) = measuringProducts(of: talker.codePredictor.model.layers) {
                talker.codePredictor.predict(
                    hidden: hidden,
                    firstEmbedding: talker.embedCodec(MLXArray([Int32(first)]).reshaped(1, 1)),
                    cache: talker.codePredictor.makeCache()
                ) { _, logits in argMax(logits, axis: -1, keepDims: true).asType(.int32) }
            }
            return NeuralReference(
                prompt: prompt, logits: values, hidden: hidden, first: first,
                talkerProducts: talkerProducts, codePredictorProducts: predictorProducts)
        }
    }

    /// How closely the Neural Engine models follow MLX on the reference's
    /// prompt: the talker's logits after it; then one greedy frame of the
    /// code predictor from MLX's hidden state, with MLX following the
    /// Neural Engine's codes, so a near tie settled the other way doesn't
    /// send the two down different frames.
    struct NeuralAgreement: CustomStringConvertible {
        var logitsSNR: Double
        /// For each code the Neural Engine took: how far MLX's logit for it
        /// falls below MLX's best; 0 when they agree.
        var shortfalls: [Float]
        var embeddingSNR: Double

        /// What a prepared voice must show: logits and embeddings within 20
        /// dB of MLX's, and no code that MLX ranks two logits below its best.
        /// Near ties go either way in fp16 (MLX itself takes Qwen's fp32 best
        /// code only 92 to 94 % of the time, ADR-0074); a broken graph lands
        /// far below both bars.
        var passes: Bool {
            logitsSNR >= 20 && embeddingSNR >= 20 && shortfalls.allSatisfy { $0 < 2 }
        }

        var description: String {
            String(
                format: "logits %.1f dB, embedding sum %.1f dB, %d of %d codes MLX's best "
                    + "(worst %.2f below)",
                logitsSNR, embeddingSNR, shortfalls.filter { $0 == 0 }.count, shortfalls.count,
                shortfalls.max() ?? 0)
        }
    }

    func neuralAgreement(_ voice: Qwen3TTSNeuralVoice, _ reference: NeuralReference) throws
        -> NeuralAgreement
    {
        let (rows, hidden) = Device.withDefaultDevice(.cpu) {
            (Self.rows(reference.prompt.body), Self.rows(reference.hidden)[0])
        }
        let session = try voice.talker.makeSession()
        var step: Qwen3TTSNeuralTalker.Step?
        for row in rows {
            step = try row.withUnsafeBufferPointer { try voice.talker.step($0, session: session) }
        }
        guard let step else { throw AudioGenerationError.invalidInput("The probe is empty.") }
        let hiddenColumn = try MLMultiArray(
            shape: [1, NSNumber(value: hidden.count), 1, 1], dataType: .float16)
        hidden.withUnsafeBufferPointer { hiddenColumn.copy(from: $0) }
        var random = NeuralRandom(seed: 0)
        let (codes, sum) = try voice.withCodecRow(reference.first) {
            try voice.codePredictor.frame(
                hidden: hiddenColumn, code0: $0, temperature: 0, random: &random)
        }
        var shortfalls: [Float] = []
        let expectedSum = Device.withDefaultDevice(.cpu) {
            talker.codePredictor.predict(
                hidden: reference.hidden,
                firstEmbedding: talker.embedCodec(MLXArray([Int32(reference.first)]).reshaped(1, 1)),
                cache: talker.codePredictor.makeCache()
            ) { group, logits in
                let values = logits.asType(.float32).asArray(Float.self)
                shortfalls.append(values.max()! - values[Int(codes[group])])
                return MLXArray([codes[group]]).reshaped(1, 1)
            }.embeddingSum.asType(.float32).asArray(Float.self)
        }
        return NeuralAgreement(
            logitsSNR: Qwen3TTSNeuralCodec.snr(step.logits, reference.logits),
            shortfalls: shortfalls,
            embeddingSNR: Qwen3TTSNeuralCodec.snr(sum.floats(), expectedSum))
    }

    /// `body` with MLX on the CPU while the Neural Engine voice is prepared:
    /// a backgrounded iPhone app may not use the GPU.
    func onVoiceDevice<R>(_ body: () throws -> R) rethrows -> R {
        guard lock.withLock({ neuralVoice }) != nil else { return try body() }
        return try Device.withDefaultDevice(.cpu, body)
    }

    /// The checkpoint's weights' identity, for the cached models' keys:
    /// file names, sizes and dates.
    func checkpointKey() -> String {
        guard let directory = speechTokenizerDirectory?.deletingLastPathComponent() else {
            return "memory"
        }
        let files = (try? Qwen3TTSWeights.safetensorsFiles(in: directory)) ?? []
        return files.map { file in
            let values = try? file.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey])
            return "\(file.lastPathComponent):\(values?.fileSize ?? 0):"
                + "\(values?.contentModificationDate?.timeIntervalSince1970 ?? 0)"
        }.joined(separator: ",")
    }

    /// `[1, n, D]` as n rows of D fp16 values.
    static func rows(_ x: MLXArray) -> [[Float16]] {
        let width = x.dim(-1)
        let values = x.asType(.float16).asArray(Float16.self)
        return (0 ..< values.count / width).map { Array(values[($0 * width) ..< (($0 + 1) * width)]) }
    }
}

/// `body`'s result, and the largest MLP product each of `layers` computed
/// while it ran.
func measuringProducts<R>(of layers: [Qwen3TTSDecoderLayer], _ body: () throws -> R) rethrows
    -> (R, [Float])
{
    var largest = [Float](repeating: 0, count: layers.count)
    for (i, layer) in layers.enumerated() {
        layer.mlp.productProbe = { largest[i] = max(largest[i], abs($0).max().item(Float.self)) }
    }
    defer { for layer in layers { layer.mlp.productProbe = nil } }
    return (try body(), largest)
}

// MARK: - The frame loop

extension Qwen3TTSModel {
    /// `run` on the Neural Engine: the prompt one position at a time, then per
    /// frame the first code drawn on the host, the code predictor's frame,
    /// and the talker's next step. Frames are decoded as in `run`; with the
    /// Neural Engine codec, on its queue. The same seed gives the same
    /// render here, though not MLX's: the draws come from `NeuralRandom`.
    func runNeural(
        _ voice: Qwen3TTSNeuralVoice, prompt: Qwen3TTSPrompt, sampling: Qwen3TTSSampling,
        seed: UInt64, chunkFrames: Int?, alignment: Qwen3TTSAlignmentHead?,
        onAudio: (@Sendable ([Float]) -> Void)?, onAlignment: (@Sendable ([Float]) -> Void)?
    ) throws -> [[Int32]] {
        generationLock.lock()
        defer { generationLock.unlock() }
        let session = try voice.talker.makeSession()
        let promptRows = (prompt.instruct.map(Self.rows) ?? []) + Self.rows(prompt.body)
        let trailing = prompt.trailingText.map(Self.rows) ?? []
        let pad = Self.rows(prompt.pad)[0]
        let room = voice.talker.contextLength - promptRows.count
        guard room > 0 else {
            throw AudioGenerationError.invalidInput(
                "The prompt is longer than the Neural Engine talker's "
                    + "\(voice.talker.contextLength) positions.")
        }
        // A frame's step comes after it, so the last frame needs no room.
        let maxFrames = min(sampling.maxTokens, max(75, prompt.textTokenCount * 6), room + 1)

        var random = NeuralRandom(seed: seed)
        var sampler = Qwen3TTSHostTalkerSampler(
            sampling: sampling, vocabSize: talkerConfig.vocabSize,
            eosTokenID: talkerConfig.codecEosTokenId)

        // Word timing: the row for frame f comes from the step that predicts
        // it, over the text track's positions written so far.
        let instructLength = prompt.instruct?.dim(1) ?? 0
        let span =
            (instructLength + prompt.textSpan.lowerBound)
            ..< (instructLength + prompt.textSpan.upperBound)
        let timing = alignment != nil && alignment == voice.talker.alignment ? onAlignment : nil
        func deliverAlignment(_ scores: [Float]?) {
            guard let timing, let scores else { return }
            let written = min(span.lowerBound, scores.count) ..< min(span.upperBound, scores.count)
            var row = Array(scores[written])
            row += [Float](repeating: -.infinity, count: span.count - row.count)
            timing(row)
        }

        var step: Qwen3TTSNeuralTalker.Step?
        for row in promptRows {
            step = try row.withUnsafeBufferPointer { try voice.talker.step($0, session: session) }
        }

        let groups = talkerConfig.numCodeGroups
        var accepted: [[Int32]] = []
        var decoder = codecDecoder.makeStream()
        var decoded = 0
        let neural = chunkFrames != nil ? lock.withLock { neuralCodec } : nil
        let neuralStream = try neural?.makeStream()
        if neural != nil { codecDecoder.releaseConvStack() }
        let neuralError = OSAllocatedUnfairLock<(any Error)?>(initialState: nil)
        let firstChunk = neural.map { $0.frames } ?? chunkFrames.map { min(3, $0) }
        let steadyChunk = neural.map { $0.frames } ?? chunkFrames

        func decodeReady(final: Bool) throws {
            guard let firstChunk, let steadyChunk else { return }
            while true {
                let ready = accepted.count - decoded
                let due = decoded == 0 ? firstChunk : steadyChunk
                guard ready > 0, final || ready >= due else { return }
                let take = neural.map { min(ready, $0.frames) } ?? ready
                let codes = MLXArray(accepted[decoded ..< (decoded + take)].flatMap { $0 })
                    .reshaped(1, take, groups)
                let latent = codecDecoder.latent(codes, stream: &decoder)
                decoded += take
                if let neural, let neuralStream {
                    let values = latent.asType(.float16).asArray(Float16.self)
                    neuralQueue.async { [onAudio] in
                        guard neuralError.withLock({ $0 == nil }) else { return }
                        do {
                            onAudio?(try neural.decode(latent: values, stream: neuralStream))
                        } catch {
                            neuralError.withLock { if $0 == nil { $0 = error } }
                        }
                    }
                } else {
                    let audio = try codecDecoder.synthesize(latent, stream: &decoder)
                    onAudio?(audio.asArray(Float.self))
                }
            }
        }

        let eos = talkerConfig.codecEosTokenId
        for frame in 0 ..< maxFrames {
            try Task.checkCancellation()
            guard let current = step else { break }
            let first = sampler(current.logits, frame: frame, random: &random)
            if first == eos { break }
            let (rest, sum) = try voice.withCodecRow(first) {
                try voice.codePredictor.frame(
                    hidden: current.hidden, code0: $0, temperature: sampling.detailTemperature,
                    random: &random)
            }
            accepted.append([Int32(first)] + rest)
            deliverAlignment(current.alignment)
            try decodeReady(final: false)
            guard frame + 1 < maxFrames else { break }

            // The talker's step for this frame: the codes' embeddings and the
            // text track's next input.
            let text = frame < trailing.count ? trailing[frame] : pad
            let embeddings = sum.floats()
            let x = (0 ..< text.count).map { Float16(embeddings[$0] + Float(text[$0])) }
            step = try x.withUnsafeBufferPointer { try voice.talker.step($0, session: session) }
        }
        try Task.checkCancellation()

        try decodeReady(final: true)
        if neural != nil {
            neuralQueue.sync {}
            if let error = neuralError.withLock({ $0 }) {
                disableNeuralEngine()
                throw error
            }
        }
        return accepted
    }
}
