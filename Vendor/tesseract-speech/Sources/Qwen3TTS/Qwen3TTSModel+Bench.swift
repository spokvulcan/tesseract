import Foundation
@preconcurrency import MLX
@preconcurrency import MLXLMCommon

// Hooks for the qwen3-tts-bench tool: per-component timings with explicit
// eval barriers, and teacher-forced logits. Not used by the engine.

extension Qwen3TTSModel {

    /// Decodes `codeFrames` (one row of `numCodeGroups` codes per frame)
    /// with the streaming decoder, `chunk` frames per step. Returns the
    /// samples and the seconds each step took.
    package func benchStreamingDecode(
        codeFrames: [[Int32]], chunk: Int
    ) throws -> (samples: [Float], stepSeconds: [Double]) {
        guard !codeFrames.isEmpty else { return ([], []) }
        let all = MLXArray(codeFrames.flatMap { $0 }).reshaped(1, codeFrames.count, -1)
        var stream = codecDecoder.makeStream()
        var samples: [Float] = []
        var times: [Double] = []
        var start = 0
        while start < codeFrames.count {
            let end = min(start + chunk, codeFrames.count)
            let codes = all[0..., start ..< end, 0...]
            eval(codes)
            let t0 = DispatchTime.now().uptimeNanoseconds
            let audio = try codecDecoder.decode(codes, stream: &stream)
            eval([audio] + stream.arrays)
            times.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9)
            samples.append(contentsOf: audio.asArray(Float.self))
            start = end
        }
        return (samples, times)
    }

    /// Teacher-forced logits for `codeFrames`: the talker's logits before
    /// each frame (and once more after the last), and the code predictor's
    /// logits for groups 1...15 of each frame, feeding the given codes back
    /// instead of sampling. The whole prompt is one prefill. Row-major
    /// float32: talker `[frames + 1, vocab]`, detail `[frames, groups - 1,
    /// cpVocab]`.
    package func benchTeacherForced(
        text: String, voice: String?, language: String?,
        reference: Qwen3TTSReference?, codeFrames: [[Int32]],
        layout: Qwen3TTSTextLayout = .interleaved
    ) throws -> (talker: [Float], detail: [Float]) {
        let prompt = try prompt(
            text: text, voice: voice, language: language, reference: reference, layout: layout)
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
            let text =
                t < prompt.trailingCount
                ? prompt.trailingText![0..., t ..< (t + 1), 0...] : prompt.pad
            (logits, hidden) = talker(sum + text, cache: cache)
        }
        talkerRows.append(contentsOf: logits.asType(.float32).asArray(Float.self))
        return (talkerRows, detailRows)
    }

    /// Times the talker: one prefill of `promptLength` positions (each run
    /// on a fresh cache, `repeats` times), then `steps` single-position
    /// forwards, then `steps` code-predictor frames (15 sequential passes,
    /// sampling each). Seconds per call.
    package func benchTalker(
        promptLength: Int, steps: Int, repeats: Int = 3
    ) -> (prefill: [Double], talkerStep: [Double], codePredictorFrame: [Double]) {
        let hiddenSize = talkerConfig.hiddenSize
        let random = MLXRandom.RandomState(seed: 7)
        let prompt = (MLXRandom.normal([1, promptLength, hiddenSize], key: random) * 0.02)
            .asType(.bfloat16)
        eval(prompt)

        var prefill: [Double] = []
        var cache = talker.makeCache(capacity: promptLength + steps + 1)
        var (logits, hidden) = talker.prefill(prompt, cache: cache)
        for _ in 0 ..< repeats {
            cache = talker.makeCache(capacity: promptLength + steps + 1)
            let t0 = DispatchTime.now().uptimeNanoseconds
            (logits, hidden) = talker.prefill(prompt, cache: cache)
            eval(logits, hidden)
            prefill.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9)
        }

        var talkerTimes: [Double] = []
        for _ in 0 ..< steps {
            let t0 = DispatchTime.now().uptimeNanoseconds
            (logits, hidden) = talker(hidden, cache: cache)
            eval(logits, hidden)
            talkerTimes.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9)
        }

        var cpTimes: [Double] = []
        let codeCache = talker.codePredictor.makeCache()
        let sampler = Qwen3TTSSampler(temperature: 0.5, topK: 50, topP: 1)
        let firstEmbedding = talker.embedCodec(MLXArray([Int32(5)]).reshaped(1, 1))
        eval(firstEmbedding)
        for _ in 0 ..< steps {
            let t0 = DispatchTime.now().uptimeNanoseconds
            let (codes, sum) = talker.codePredictor.predict(
                hidden: hidden, firstEmbedding: firstEmbedding, cache: codeCache
            ) { _, logits in sampler(logits, random: random) }
            eval(codes + [sum])
            cpTimes.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9)
        }
        return (prefill, talkerTimes, cpTimes)
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
            let text = frame < prompt.trailingCount ? prompt.trailingText![0..., frame ..< (frame + 1), 0...] : prompt.pad
            (logits, hidden) = talker(sum + text, cache: cache)
            eval(codes, logits, hidden, sampler.recent)
            accepted.append(codes)
            if (frame + 1) % every == 0 { probe("frame \(frame + 1)") }
        }
        var stream = codecDecoder.makeStream()
        var start = 0
        while start < accepted.count {
            let end = min(start + 5, accepted.count)
            let c = concatenated(Array(accepted[start ..< end]), axis: 0).reshaped(1, end - start, -1)
            let audio = try codecDecoder.decode(c, stream: &stream)
            eval([audio] + stream.arrays)
            start = end
        }
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
        codec.reset()
        var samples: [Float] = []
        var times: [Double] = []
        let dim = codecDecoder.config.latentDim
        var start = 0
        while start < codeFrames.count {
            let end = min(start + codec.frames, codeFrames.count)
            let latent = codecDecoder.latent(all[0..., start ..< end, 0...], stream: &stream)
            var values = latent.asType(.float16).asArray(Float16.self)
            values += [Float16](repeating: 0, count: (codec.frames - (end - start)) * dim)
            let t0 = DispatchTime.now().uptimeNanoseconds
            samples += try codec.decode(latent: values, valid: end - start)
            times.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9)
            start = end
        }
        return (samples, times)
    }
}

extension Qwen3TTSModel {
    /// Splits a talker step's and a code-predictor frame's time into graph
    /// construction (CPU) and evaluation (GPU, after the graph is built).
    package func benchBuildVersusRun(steps: Int) -> (talkerBuild: [Double], talkerRun: [Double], cpBuild: [Double], cpRun: [Double]) {
        let random = MLXRandom.RandomState(seed: 7)
        let prompt = (MLXRandom.normal([1, 300, talkerConfig.hiddenSize], key: random) * 0.02).asType(.bfloat16)
        let cache = talker.makeCache(capacity: 300 + steps + 1)
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
    /// Bench: the fused Metal kernels on or off.
    package func setFusedKernels(_ enabled: Bool, sampler: Bool = true, normRoPE: Bool = true, addNorm: Bool = true) {
        Qwen3TTSKernels.enabled = enabled
        Qwen3TTSKernels.sampler = sampler
        Qwen3TTSKernels.normRoPE = normRoPE
        Qwen3TTSKernels.addNorm = addNorm
    }
}

extension Qwen3TTSModel {
    /// Bench: runs `body` with every fused kernel call checked against the
    /// MLX ops it replaces, on the same inputs (the sampler on the same
    /// uniform draw). The first mismatching q/k calls go to `dump` as JSON.
    package func benchAudit(dump: URL?, _ body: () async throws -> Void) async rethrows -> String {
        var names: [ObjectIdentifier: String] = [:]
        for (i, layer) in talker.model.layers.enumerated() {
            names[ObjectIdentifier(layer.attention)] = "talker.\(i)"
        }
        for (i, layer) in talker.codePredictor.model.layers.enumerated() {
            names[ObjectIdentifier(layer.attention)] = "predictor.\(i)"
        }
        var calls = (normRoPE: 0, addNorm: 0, sample: 0)
        var mismatched = (normRoPE: 0, addNorm: 0, sample: 0)
        var elements = 0
        var dumped = 0
        var perLayer: [String: Int] = [:]
        Qwen3TTSKernels.audit = { attention, qkv, q, k, offset in
            let h = attention.heads
            let kv = attention.kvHeads
            let d = attention.headDim
            let split = qkv.reshaped(1, 1, h + 2 * kv, d)
            let nq = attention.qNorm(split[0..., 0..., ..<h, 0...])
            let nk = attention.kNorm(split[0..., 0..., h ..< (h + kv), 0...])
            let refQ = MLXFast.RoPE(
                nq.transposed(0, 2, 1, 3), dimensions: d, traditional: false,
                base: attention.ropeBase, scale: 1, offset: offset)
            let refK = MLXFast.RoPE(
                nk.transposed(0, 2, 1, 3), dimensions: d, traditional: false,
                base: attention.ropeBase, scale: 1, offset: offset)
            let differ = (q .!= refQ).sum() + (k .!= refK).sum()
            eval(differ)
            calls.normRoPE += 1
            let n = Int(differ.item(Int32.self))
            guard n > 0 else { return }
            let name = names[ObjectIdentifier(attention)] ?? "?"
            mismatched.normRoPE += 1
            elements += n
            perLayer[name, default: 0] += 1
            guard let dump, dumped < 6 else { return }
            dumped += 1
            func floats(_ a: MLXArray) -> [Float] { a.asType(.float32).reshaped(-1).asArray(Float.self) }
            let record = NormRoPEMismatch(
                name: name, offset: offset, eps: attention.qNorm.eps, base: attention.ropeBase,
                heads: h, kvHeads: kv, headDim: d, qkv: floats(qkv),
                qw: floats(attention.qNorm.weight), kw: floats(attention.kNorm.weight),
                q: floats(q), k: floats(k), refQ: floats(refQ), refK: floats(refK),
                nq: floats(nq), nk: floats(nk))
            try? JSONEncoder().encode(record).write(
                to: dump.appendingPathComponent("mismatch-\(dumped).json"))
        }
        Qwen3TTSKernels.auditAddNorm = { x, y, weight, eps, sum, normed in
            let refSum = x + y
            let refNormed = MLXFast.rmsNorm(refSum, weight: weight, eps: eps)
            let differ = (sum .!= refSum).sum() + (normed .!= refNormed).sum()
            eval(differ)
            calls.addNorm += 1
            if differ.item(Int32.self) > 0 { mismatched.addNorm += 1 }
        }
        Qwen3TTSKernels.auditSample = { logits, noise, temperature, topK, token in
            // categorical: argmax(gumbel + logits), gumbel = -log(-log(u)).
            let filtered = Qwen3TTSSampler.filter(logits / temperature, topK: topK, topP: 1)
            let reference = argMax(-log(-log(noise)) + filtered, axis: -1).asType(.int32)
            let differ = (reference.reshaped(1, 1) .!= token).sum()
            eval(differ)
            calls.sample += 1
            if differ.item(Int32.self) > 0 { mismatched.sample += 1 }
        }
        defer {
            Qwen3TTSKernels.audit = nil
            Qwen3TTSKernels.auditAddNorm = nil
            Qwen3TTSKernels.auditSample = nil
        }
        try await body()
        let layers = perLayer.sorted { $0.key < $1.key }.map { "\($0.key): \($0.value)" }
        return "q/k norm+RoPE \(mismatched.normRoPE)/\(calls.normRoPE) calls differ (\(elements) elements)"
            + (layers.isEmpty ? "" : " [" + layers.joined(separator: ", ") + "]")
            + "; add+norm \(mismatched.addNorm)/\(calls.addNorm); sampler \(mismatched.sample)/\(calls.sample)"
    }
}

/// Bench: one fused q/k call that differed from MLX's ops, with its inputs
/// (float32), for offline analysis.
private struct NormRoPEMismatch: Encodable {
    var name: String
    var offset: Int
    var eps: Float
    var base: Float
    var heads: Int
    var kvHeads: Int
    var headDim: Int
    var qkv, qw, kw, q, k, refQ, refK, nq, nk: [Float]
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
