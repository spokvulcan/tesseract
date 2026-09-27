import Foundation
@preconcurrency import MLX
@preconcurrency import MLXLMCommon

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
