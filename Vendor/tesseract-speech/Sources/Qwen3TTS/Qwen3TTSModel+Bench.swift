import CoreML
import Foundation
@preconcurrency import MLX
@preconcurrency import MLXLMCommon
import MLXNN

// Hooks for the qwen3-tts-bench tool: per-component timings with explicit
// eval barriers, and teacher-forced logits. Not used by the engine.

extension Qwen3TTSModel {

    /// The codec decoder's streaming decode, timed (`Qwen3TTSCodecDecoder`).
    package func benchStreamingDecode(
        codeFrames: [[Int32]], chunk: Int
    ) throws -> (samples: [Float], stepSeconds: [Double]) {
        try codecDecoder.benchStreamingDecode(codeFrames: codeFrames, chunk: chunk)
    }

    /// Teacher-forced logits for `codeFrames`: the talker's logits before
    /// each frame (and once more after the last), and the code predictor's
    /// logits for groups 1...15 of each frame, feeding the given codes back
    /// instead of sampling. The whole prompt is one prefill. Row-major
    /// float32: talker `[frames + 1, vocab]`, detail `[frames, groups - 1,
    /// cpVocab]`.
    package func benchTeacherForced(
        text: String, voice: String?, language: String?,
        reference: Qwen3TTSReference?, codeFrames: [[Int32]]
    ) throws -> (talker: [Float], detail: [Float]) {
        let prompt = try prompt(
            text: text, voice: voice, language: language, reference: reference, layout: .interleaved)
        let full = prompt.instruct.map { concatenated([$0, prompt.body], axis: 1) } ?? prompt.body
        let cache = talker.makeCache(capacity: full.dim(1) + codeFrames.count + 1)
        let codeCache = talker.codePredictor.makeCache()
        var (logits, hidden) = talker.prefill(full, cache: cache)
        var talkerRows: [Float] = []
        var detailRows: [Float] = []
        let predictor = talker.codePredictor
        for (t, frame) in codeFrames.enumerated() {
            talkerRows.append(contentsOf: logits.asType(.float32).asArray(Float.self))
            let codes = frame.map { MLXArray([$0]).reshaped(1, 1) }
            var groupLogits: [MLXArray] = []
            let (_, sum) = predictor.predict(
                hidden: hidden, firstEmbedding: talker.embedCodec(codes[0]), cache: codeCache
            ) { group, logits in
                groupLogits.append(logits)
                return codes[group + 1]
            }
            for l in groupLogits {
                detailRows.append(contentsOf: l.asType(.float32).asArray(Float.self))
            }
            (logits, hidden) = talker(sum + prompt.text(forFrame: t), cache: cache)
        }
        talkerRows.append(contentsOf: logits.asType(.float32).asArray(Float.self))
        return (talkerRows, detailRows)
    }
}

extension Qwen3TTSModel {
    /// Memory at each step of one generation from `prompt`: after building
    /// it, after the prefill, every `every` frames, and after decoding.
    package func benchMemoryTrace(
        text: String, voice: String?, reference: Qwen3TTSReference?, frames: Int, every: Int,
        report: (String) -> Void
    ) throws {
        func mb(_ b: Int) -> String { String(format: "%.0f", Double(b) / 1_048_576) }
        func probe(_ label: String) {
            report("\(label): active \(mb(Memory.activeMemory)) cache \(mb(Memory.cacheMemory)) peak \(mb(Memory.peakMemory))")
        }
        probe("start")
        let prompt = try prompt(text: text, voice: voice, language: nil, reference: reference, layout: .interleaved)
        eval(prompt.body)
        probe("prompt built (\(prompt.body.dim(1)) positions)")
        let cache = talker.makeCache(capacity: 1024)
        if let instruct = prompt.instruct { _ = talker.prefill(instruct, cache: cache) }
        var (logits, hidden) = talker.prefill(prompt.body, cache: cache)
        eval(logits, hidden)
        probe("prefill")
        let random = MLXRandom.RandomState(seed: 1)
        var sampler = Qwen3TTSTalkerSampler(
            sampling: Qwen3TTSSampling(), vocabSize: talkerConfig.vocabSize,
            eosTokenID: talkerConfig.codecEosTokenId, dtype: prompt.body.dtype)
        let detail = Qwen3TTSSampler(temperature: 0.5, topK: 50, topP: 1)
        let codeCache = talker.codePredictor.makeCache()
        var accepted: [MLXArray] = []
        for frame in 0 ..< frames {
            let first = sampler(logits, frame: frame, random: random)
            let (rest, sum) = talker.codePredictor.predict(
                hidden: hidden, firstEmbedding: talker.embedCodec(first), cache: codeCache
            ) { _, l in detail(l, random: random) }
            let codes = concatenated([first] + rest, axis: 1)
            (logits, hidden) = talker(sum + prompt.text(forFrame: frame), cache: cache)
            eval(codes, logits, hidden, sampler.recent)
            accepted.append(codes)
            if (frame + 1) % every == 0 { probe("frame \(frame + 1)") }
        }
        _ = try codecDecoder.benchStreamingDecode(
            codeFrames: accepted.map { $0.asArray(Int32.self) }, chunk: 5)
        probe("decoded")
    }
}

extension Qwen3TTSModel {
    /// Decodes `codeFrames` with the MLX front end and the Neural Engine
    /// conv stack, a codec chunk at a time; seconds per Neural Engine call.
    package func benchNeuralDecode(codeFrames: [[Int32]]) throws -> (samples: [Float], callSeconds: [Double]) {
        guard let codec = neuralCodecForBench else { return ([], []) }
        let all = MLXArray(codeFrames.flatMap { $0 }).reshaped(1, codeFrames.count, -1)
        var stream = codecDecoder.makeStream()
        let neuralStream = try codec.makeStream()
        var samples: [Float] = []
        var times: [Double] = []
        var start = 0
        while start < codeFrames.count {
            let end = min(start + codec.frames, codeFrames.count)
            let latent = codecDecoder.latent(all[0..., start ..< end, 0...], stream: &stream)
            let values = latent.asType(.float16).asArray(Float16.self)
            let t0 = DispatchTime.now().uptimeNanoseconds
            samples += try codec.decode(latent: values, stream: neuralStream)
            times.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9)
            start = end
        }
        return (samples, times)
    }
}

extension Qwen3TTSModel {
    /// Times a talker step and a code-predictor frame after a prompt of
    /// `promptLength` positions, split into graph construction (CPU) and
    /// evaluation (GPU, after the graph is built).
    package func benchBuildVersusRun(promptLength: Int, steps: Int) -> (talkerBuild: [Double], talkerRun: [Double], cpBuild: [Double], cpRun: [Double]) {
        let random = MLXRandom.RandomState(seed: 7)
        let prompt = (MLXRandom.normal([1, promptLength, talkerConfig.hiddenSize], key: random) * 0.02).asType(.bfloat16)
        let cache = talker.makeCache(capacity: promptLength + steps + 1)
        var (logits, hidden) = talker(prompt, cache: cache)
        eval(logits, hidden)
        var tb: [Double] = [], tr: [Double] = [], cb: [Double] = [], cr: [Double] = []
        let sampler = Qwen3TTSSampler(temperature: 0.5, topK: 50, topP: 1)
        let codeCache = talker.codePredictor.makeCache()
        let firstEmbedding = talker.embedCodec(MLXArray([Int32(5)]).reshaped(1, 1))
        eval(firstEmbedding)
        func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
        for _ in 0 ..< steps {
            var t0 = now()
            (logits, hidden) = talker(hidden, cache: cache)
            tb.append(Double(now() - t0) / 1e9)
            t0 = now()
            eval(logits, hidden)
            tr.append(Double(now() - t0) / 1e9)
            t0 = now()
            let (codes, sum) = talker.codePredictor.predict(
                hidden: hidden, firstEmbedding: firstEmbedding, cache: codeCache
            ) { _, l in sampler(l, random: random) }
            cb.append(Double(now() - t0) / 1e9)
            t0 = now()
            eval(codes + [sum])
            cr.append(Double(now() - t0) / 1e9)
        }
        return (tb, tr, cb, cr)
    }
}

extension Qwen3TTSModel {
    /// Bench: each fused Metal kernel on or off.
    package func setFusedKernels(sampler: Bool, normRoPE: Bool, addNorm: Bool) {
        Qwen3TTSKernels.sampler = sampler
        Qwen3TTSKernels.normRoPE = normRoPE
        Qwen3TTSKernels.addNorm = addNorm
    }
}

extension Qwen3TTSModel {
    /// Bench: runs `body` with every fused kernel call checked against the
    /// MLX ops it replaces, on the same inputs (the sampler on the same
    /// uniform draw): how many calls of each kernel differed.
    package func benchAudit(_ body: () async throws -> Void) async rethrows -> String {
        var calls: [Qwen3TTSKernels.Kernel: Int] = [:]
        var differed: [Qwen3TTSKernels.Kernel: Int] = [:]
        Qwen3TTSKernels.audit = { kernel, fused, reference in
            let differ = zip(fused, reference()).map { ($0 .!= $1).sum() }
                .reduce(MLXArray(Int32(0)), +)
            calls[kernel, default: 0] += 1
            if differ.item(Int32.self) > 0 { differed[kernel, default: 0] += 1 }
        }
        defer { Qwen3TTSKernels.audit = nil }
        try await body()
        return Qwen3TTSKernels.Kernel.allCases.map {
            "\($0) \(differed[$0] ?? 0)/\(calls[$0] ?? 0)"
        }.joined(separator: ", ") + " calls differ"
    }
}

extension Qwen3TTSModel {
    /// Bench: talker prefills of new lengths into one kept cache, as one
    /// graph or a layer at a time (`Qwen3TTSTalker.prefill`): the best time
    /// of `repeats` for each, and what they have left in MLX's pool so far.
    package func benchPrefill(lengths: [Int], layerByLayer: Bool, repeats: Int = 3) -> [String] {
        func mb(_ b: Int) -> String { String(format: "%.0f", Double(b) / 1_048_576) }
        let random = MLXRandom.RandomState(seed: 14)
        let cache = talker.makeCache(capacity: 1024)
        eval(cache.flatMap { $0.state })
        Memory.clearCache()
        let base = Memory.activeMemory + Memory.cacheMemory
        var lines: [String] = []
        for length in lengths {
            let embeds = (MLXRandom.normal([1, length, talkerConfig.hiddenSize], key: random) * 0.02)
                .asType(.bfloat16)
            eval(embeds)
            var best = Double.infinity
            for _ in 0 ..< repeats {
                for c in cache { c.trim(c.offset) }
                let t0 = DispatchTime.now().uptimeNanoseconds
                let (logits, _) = layerByLayer ? talker.prefill(embeds, cache: cache) : talker(embeds, cache: cache)
                eval(logits)
                best = min(best, Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6)
            }
            lines.append(String(
                format: "L %4d: %6.2f ms, pool +%@ MB", length, best,
                mb(Memory.activeMemory + Memory.cacheMemory - base)))
        }
        return lines
    }
}

extension Qwen3TTSCodecDecoder {
    /// Bench: decodes `codeFrames` (one row of `numQuantizers` codes per
    /// frame) as a stream, `chunk` frames at a time. Returns the samples and
    /// the seconds each chunk took.
    package func benchStreamingDecode(
        codeFrames: [[Int32]], chunk: Int
    ) throws -> (samples: [Float], stepSeconds: [Double]) {
        guard !codeFrames.isEmpty else { return ([], []) }
        let all = MLXArray(codeFrames.flatMap { $0 }).reshaped(1, codeFrames.count, -1)
        var stream = makeStream()
        var samples: [Float] = []
        var times: [Double] = []
        var start = 0
        while start < codeFrames.count {
            let end = min(start + chunk, codeFrames.count)
            let codes = all[0..., start ..< end, 0...]
            eval(codes)
            let t0 = DispatchTime.now().uptimeNanoseconds
            let audio = try decode(codes, stream: &stream)
            eval([audio] + stream.arrays)
            times.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9)
            samples.append(contentsOf: audio.asArray(Float.self))
            start = end
        }
        return (samples, times)
    }
}

extension Qwen3TTSModel {
    /// The Neural Engine voice teacher-forced on `codeFrames`, against MLX:
    /// the talker's logits before each frame, and the code predictor's for
    /// groups 1...15 of each frame (through its forced variant, built into
    /// `cacheDirectory`). Both sides get MLX's inputs: the prompt, each
    /// frame's code embeddings, the talker's hidden state. Reports SNR, the
    /// KL divergence of the draws (at 0.9 for the talker, over the codes it
    /// may draw; at 0.5 for the code predictor), and how often the best code
    /// agrees: ADR-0074's measures.
    package func benchNeuralTeacherForced(
        text: String, voice: String?, language: String?, codeFrames: [[Int32]],
        cacheDirectory: URL
    ) async throws -> String {
        guard let neural = lock.withLock({ neuralVoice }) else { return "no neural voice" }
        let (forced, _) = try await Qwen3TTSNeuralCodePredictor.load(
            predictor: talker.codePredictor, topK: Qwen3TTSSampling.topK,
            precision: neural.codePredictorPrecision, forced: true,
            cacheDirectory: cacheDirectory, sourceKey: checkpointKey())

        // MLX, with what the Neural Engine side is fed.
        let prompt = try self.prompt(
            text: text, voice: voice, language: language, reference: nil, layout: .interleaved)
        let full = prompt.instruct.map { concatenated([$0, prompt.body], axis: 1) } ?? prompt.body
        let cache = talker.makeCache(capacity: full.dim(1) + codeFrames.count + 1)
        let codeCache = talker.codePredictor.makeCache()
        var (logits, hidden) = talker.prefill(full, cache: cache)
        var expectedTalker: [[Float]] = []
        var expectedDetail: [[Float]] = []
        var hiddens: [[Float16]] = []
        var firstEmbeddings: [[Float16]] = []
        var steps: [[Float16]] = []
        for (t, frame) in codeFrames.enumerated() {
            expectedTalker.append(logits.asType(.float32).asArray(Float.self))
            hiddens.append(Self.rows(hidden)[0])
            let codes = frame.map { MLXArray([$0]).reshaped(1, 1) }
            let first = talker.embedCodec(codes[0])
            firstEmbeddings.append(Self.rows(first)[0])
            var groupLogits: [MLXArray] = []
            let (_, sum) = talker.codePredictor.predict(
                hidden: hidden, firstEmbedding: first, cache: codeCache
            ) { group, logits in
                groupLogits.append(logits)
                return codes[group + 1]
            }
            expectedDetail.append(
                concatenated(groupLogits, axis: 1).asType(.float32).asArray(Float.self))
            let x = sum + prompt.text(forFrame: t)
            steps.append(Self.rows(x)[0])
            (logits, hidden) = talker(x, cache: cache)
        }

        func best(_ v: [Float]) -> Int { v.indices.max { v[$0] < v[$1] }! }
        /// KL(MLX ‖ Neural Engine) of the draws at `temperature` over the
        /// codes `allowed`, in nats.
        func kl(_ expected: [Float], _ actual: [Float], temperature: Float, allowed: [Int]) -> Double {
            func probabilities(_ logits: [Float]) -> [Double] {
                let z = allowed.map { Double(logits[$0] / temperature) }
                let top = z.max()!
                let e = z.map { exp($0 - top) }
                let sum = e.reduce(0, +)
                return e.map { $0 / sum }
            }
            let (p, q) = (probabilities(expected), probabilities(actual))
            return zip(p, q).reduce(0) { $0 + ($1.0 > 0 ? $1.0 * log($1.0 / max($1.1, 1e-30)) : 0) }
        }

        // The Neural Engine talker, step by step on the same inputs.
        let session = try neural.talker.makeSession()
        var step: Qwen3TTSNeuralTalker.Step?
        for row in Self.rows(full) {
            step = try row.withUnsafeBufferPointer { try neural.talker.step($0, session: session) }
        }
        let drawable = Array(0 ..< (talkerConfig.vocabSize - 1024)) + [talkerConfig.codecEosTokenId]
        var talkerSNR: [Double] = []
        var talkerKL: [Double] = []
        var talkerAgree = 0
        for t in codeFrames.indices {
            guard let current = step else { break }
            talkerSNR.append(Qwen3TTSNeuralCodec.snr(current.logits, expectedTalker[t]))
            talkerKL.append(kl(expectedTalker[t], current.logits, temperature: 0.9, allowed: drawable))
            if best(current.logits) == best(expectedTalker[t]) { talkerAgree += 1 }
            if t + 1 < codeFrames.count {
                step = try steps[t].withUnsafeBufferPointer {
                    try neural.talker.step($0, session: session)
                }
            }
        }

        // The code predictor, forced through the same codes.
        var detailSNR: [Double] = []
        var detailKL: [Double] = []
        var detailAgree = 0
        let vocabulary = forced.vocabulary
        let codes = Array(0 ..< vocabulary)
        let hiddenArray = try MLMultiArray(
            shape: [1, NSNumber(value: hiddens[0].count), 1, 1], dataType: .float16)
        for (t, frame) in codeFrames.enumerated() {
            hiddens[t].withUnsafeBufferPointer { hiddenArray.copy(from: $0) }
            let actual = try firstEmbeddings[t].withUnsafeBufferPointer {
                try forced.logits(hidden: hiddenArray, code0: $0, codes: Array(frame.dropFirst()))
            }
            for g in 0 ..< forced.passes {
                let range = (g * vocabulary) ..< ((g + 1) * vocabulary)
                let a = Array(actual[range])
                let e = Array(expectedDetail[t][range])
                detailSNR.append(Qwen3TTSNeuralCodec.snr(a, e))
                detailKL.append(kl(e, a, temperature: 0.5, allowed: codes))
                if best(a) == best(e) { detailAgree += 1 }
            }
        }
        func summary(_ snr: [Double], _ kl: [Double], agree: Int) -> String {
            let (s, k) = (snr.sorted(), kl.sorted())
            return String(
                format: "%.1f dB mean (p10 %.1f), KL %.1e mean (p90 %.1e), best code %d/%d",
                snr.reduce(0, +) / Double(snr.count), s[s.count / 10],
                kl.reduce(0, +) / Double(kl.count), k[k.count * 9 / 10], agree, snr.count)
        }
        return "talker \(summary(talkerSNR, talkerKL, agree: talkerAgree)); "
            + "code predictor \(summary(detailSNR, detailKL, agree: detailAgree))"
    }

    /// Seconds per Neural Engine talker step, over `steps` steps after a
    /// short prompt, and per code-predictor frame (drawn at 0.5).
    package func benchNeuralCalls(steps: Int) throws -> (talker: [Double], codePredictor: [Double]) {
        guard let neural = lock.withLock({ neuralVoice }) else { return ([], []) }
        func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
        let session = try neural.talker.makeSession()
        var random = NeuralRandom(seed: 3)
        var x = [Float16](repeating: 0, count: talkerConfig.hiddenSize)
        var talkerTimes: [Double] = []
        var predictorTimes: [Double] = []
        for _ in 0 ..< min(steps, neural.talker.contextLength) {
            for i in x.indices { x[i] = Float16(random.uniform() - 0.5) }
            let t0 = now()
            let step = try x.withUnsafeBufferPointer { try neural.talker.step($0, session: session) }
            let t1 = now()
            _ = try neural.withCodecRow(Int(random.next() % 2048)) {
                try neural.codePredictor.frame(
                    hidden: step.hidden, code0: $0, temperature: 0.5, random: &random)
            }
            let t2 = now()
            talkerTimes.append(Double(t1 - t0) / 1e9)
            predictorTimes.append(Double(t2 - t1) / 1e9)
        }
        return (talkerTimes, predictorTimes)
    }
}

extension Qwen3TTSModel {
    /// The Neural Engine path's stages up to its first audio, timed one by
    /// one, as `runNeural` runs them (MLX on the CPU): the prompt, its rows,
    /// the prefill steps, three frames, the codec's front end and the
    /// Neural Engine codec's first chunk.
    package func benchNeuralFirstAudio(text: String, voice: String?) throws -> [(String, Double)] {
        guard let neural = lock.withLock({ neuralVoice }) else { return [] }
        func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
        var stages: [(String, Double)] = []
        var t = now()
        func lap(_ name: String) {
            let next = now()
            stages.append((name, Double(next - t) / 1e9))
            t = next
        }
        return try Device.withDefaultDevice(.cpu) {
            let prompt = try self.prompt(
                text: text, voice: voice, language: nil, reference: nil, layout: .interleaved)
            eval(prompt.body)
            if let trailing = prompt.trailingText { eval(trailing) }
            lap("prompt (\(prompt.textTokenCount) text tokens)")
            let promptRows = (prompt.instruct.map(Self.rows) ?? []) + Self.rows(prompt.body)
            let trailing = prompt.trailingText.map(Self.rows) ?? []
            lap("rows")
            let session = try neural.talker.makeSession()
            var step: Qwen3TTSNeuralTalker.Step?
            for row in promptRows {
                step = try row.withUnsafeBufferPointer { try neural.talker.step($0, session: session) }
            }
            lap("prefill (\(promptRows.count) steps)")
            var random = NeuralRandom(seed: 1)
            var sampler = Qwen3TTSHostTalkerSampler(
                sampling: Qwen3TTSSampling(), vocabSize: talkerConfig.vocabSize,
                eosTokenID: talkerConfig.codecEosTokenId)
            var frames: [[Int32]] = []
            for frame in 0 ..< 3 {
                let current = step!
                let first = sampler(current.logits, frame: frame, random: &random)
                let (rest, sum) = try neural.withCodecRow(first) {
                    try neural.codePredictor.frame(
                        hidden: current.hidden, code0: $0, temperature: 0.5, random: &random)
                }
                frames.append([Int32(first)] + rest)
                let embeddings = sum.floats()
                let text = trailing[min(frame, trailing.count - 1)]
                let x = (0 ..< text.count).map { Float16(embeddings[$0] + Float(text[$0])) }
                step = try x.withUnsafeBufferPointer { try neural.talker.step($0, session: session) }
            }
            lap("3 frames")
            var decoder = codecDecoder.makeStream()
            let codes = MLXArray(frames.flatMap { $0 }).reshaped(1, 3, -1)
            let latent = codecDecoder.latent(codes, stream: &decoder).asType(.float16)
                .asArray(Float16.self)
            lap("codec front end")
            if let codec = lock.withLock({ neuralCodec }) {
                _ = try codec.decode(latent: latent, stream: try codec.makeStream())
                lap("Neural Engine codec chunk")
            }
            return stages
        }
    }
}
