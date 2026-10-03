import CryptoKit
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import os

/// TurboQuant KV-cache measurement (`--turboquant-bench`), issue #603.
///
/// Measures what a live KV cache scheme buys and costs on the loaded model,
/// with the scheme as the only change between arms:
///
/// - **Quality:** decode-time KL divergence and teacher-forced top-1
///   agreement against the unquantized cache (bf16 on Qwen3.8), plus where
///   each arm's free-running greedy stream first leaves the reference's.
/// - **Speed:** decode tokens per second through the chunked Prefill Strategy
///   route (`PrefillExecutor.makeIterator` over the vendor `TokenIterator`,
///   which converts the cache itself from the `kvCache` parameter).
/// - **Memory:** KV bytes per token (live state over offset, and allocated),
///   the prefill peak, and the decode-phase peak.
///
/// **One prefill per context, shared by every arm.** Prefill runs on the
/// unquantized cache in every scheme: the vendor converts the cache after the
/// last prompt token and compresses it on the first decode step. So the
/// prefilled cache is snapshotted once (`HybridCacheSnapshot`, a device copy)
/// and every pass restores its own copy. The last prompt token is left for the
/// pass, as the production iterator leaves it.
///
/// **Quality is teacher-forced, one token per forward.** The reference arm
/// runs first: its greedy stream and per-step log probabilities are what every
/// other arm is forced along. A batched teacher-forced pass would take
/// TurboQuant's separate L>1 path, not the decode kernel. Step 0's logits come
/// from the unquantized cache in every arm, so their KL must be exactly 0; that
/// is checked, as the negative control #252 needed (a prefill-only logit vector
/// is blind to the KV scheme).
///
/// **Speed arms run round-robin, reversed every other round** (ABBA), so
/// thermal drift lands on every arm alike. A speed pass restores its copy
/// while the snapshot stays resident, so the decode-phase peak subtracts the
/// snapshot's bytes. The run peak is the larger of the shared prefill peak and
/// the arm's decode-phase peak.
///
/// **A noise-floor control** re-prefills the prompt at another chunk size and
/// teacher-forces the unquantized cache against the reference: the KL that a
/// benign numeric change costs, the scale to read the schemes' KL against.
///
/// The prompt is a chat-templated user turn (thinking on, the template's
/// default) of a natural-language corpus trimmed to the target length, then a
/// fixed question written for `docs/adr` that asks for every record in order
/// (another corpus gets the same question). Pass the corpus with
/// `--bench-corpus` (a file, or a directory whose `.md` files are joined in
/// name order) as an absolute path, since `open` starts the app in `/`; the
/// report records its SHA-256.
///
/// Flags: `--bench-model-id`, `--bench-output`, `--bench-corpus` (required),
/// `--bench-contexts` (default `8192,32768,65536`), `--bench-schemes`
/// (default `fp16,turbo8v4,turbo0v4`; `fp16` is always the reference and
/// runs first; also `turbo8v3`, `turbo0v3`, `affine8`, `affine4`),
/// `--bench-max-new` (decode tokens per pass, default 512), `--bench-runs`
/// (speed rounds, default 2), `--bench-noise-floor-step` (default 768; 0
/// skips the control), `--bench-prefill-steps` (first entry; default 1024).
@MainActor
final class TurboQuantBenchRunner {

    private let runner: BenchmarkRunner
    private let reportStamp: String

    nonisolated private static let logger = Logger(
        subsystem: "app.tesseract.agent", category: "benchmark")

    nonisolated private static let question =
        "\n\n---\n\nThe text above is a set of architecture decision records from one "
        + "software project. Go through the records above in order. For each record, give "
        + "its number, its title, and the decision it records in one sentence."

    init(runner: BenchmarkRunner) {
        self.runner = runner
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        self.reportStamp = formatter.string(from: Date())
    }

    // MARK: - Run

    func run() async throws {
        let reportDir = runner.activeConfig.outputDir.appendingPathComponent("turboquant-bench")
        try FileManager.default.createDirectory(at: reportDir, withIntermediateDirectories: true)
        let logURL = reportDir.appendingPathComponent("latest.log")
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        let handle = FileHandle(forWritingAtPath: logURL.path)
        defer { try? handle?.close() }
        // Every line is written as it is produced, so a crash mid-run keeps the
        // contexts that already finished.
        let emit: @Sendable (String) -> Void = { message in
            let line = "[\(Self.timestamp())] \(message)"
            handle?.write(Data((line + "\n").utf8))
            Self.logger.info("\(line, privacy: .public)")
        }
        let reportURL = reportDir.appendingPathComponent("turboquant_bench_\(reportStamp).json")

        let options: Options
        let corpus: Corpus
        let modelDir: URL
        do {
            options = try Options.parse(
                CommandLine.arguments,
                defaultPrefillStep: runner.activeConfig.prefillStepSizesOverride?.first ?? 1024)
            corpus = try Corpus.load(path: options.corpusPath)
            modelDir = try runner.resolveModelDirectory()
        } catch {
            emit("ERROR: \(error.localizedDescription)")
            throw error
        }
        let niceValue = Int(getpriority(PRIO_PROCESS, 0))
        emit(
            "TurboQuant bench starting: model=\(runner.resolvedModelName) "
                + "contexts=\(options.contexts) schemes=\(options.schemes.map(\.name)) "
                + "maxNew=\(options.maxNew) runs=\(options.runs) "
                + "prefillStep=\(options.prefillStep) noiseFloorStep=\(options.noiseFloorStep) "
                + "nice=\(niceValue) source=\(runner.activeConfig.sourceRevision ?? "unknown")")
        emit(
            "Corpus: \(corpus.path) files=\(corpus.fileCount) bytes=\(corpus.text.utf8.count) "
                + "sha256=\(corpus.sha256)")
        if niceValue != 0 {
            emit("WARNING: nice=\(niceValue); decode numbers are not comparable (launch via open)")
        }

        // Drafters off: the plain iterator never speculates, but an automatic
        // load keeps the DFlash2 draft resident and inflates every memory figure.
        let engine = AgentEngine(speculation: .off)
        Memory.peakMemory = 0
        let loadStart = ContinuousClock.now
        try await engine.loadModel(from: modelDir, visionMode: false)
        let loadSeconds = Self.seconds(since: loadStart)
        let load = LoadRecord(
            model: runner.resolvedModelName,
            modelID: runner.activeConfig.resolvedModelID,
            modelDirectory: modelDir.path,
            hardware: runner.resolvedHardwareDescription,
            osVersion: ProcessInfo.processInfo.operatingSystemVersionString,
            sourceRevision: runner.activeConfig.sourceRevision,
            nice: niceValue,
            loadSeconds: loadSeconds,
            activeAfterLoadGB: Self.gb(Memory.activeMemory),
            peakDuringLoadGB: Self.gb(Memory.peakMemory))
        emit(
            "Loaded in \(Self.fmt(loadSeconds))s: active=\(Self.fmt(load.activeAfterLoadGB))GB "
                + "loadPeak=\(Self.fmt(load.peakDuringLoadGB))GB")

        var contexts: [ContextRecord] = []
        for target in options.contexts {
            emit("=== context \(target) ===")
            let record = try await engine.llmActor.withModelContainer { container in
                try await container.perform { context in
                    try await Self.measureContext(
                        context: context, targetTokens: target, corpus: corpus,
                        options: options, emit: emit)
                }
            }
            contexts.append(record)
            try Self.writeReport(
                Report(
                    date: ISO8601DateFormatter().string(from: Date()),
                    load: load,
                    options: options.record,
                    corpus: CorpusRecord(
                        path: corpus.path, files: corpus.fileCount,
                        bytes: corpus.text.utf8.count, sha256: corpus.sha256),
                    question: Self.question,
                    contexts: contexts),
                to: reportURL)
        }

        Self.emitSummary(contexts, emit: emit)
        engine.unloadModel()
        await engine.awaitPendingUnload()
        emit("Report written to: \(reportURL.path)")
        let failed = contexts.flatMap(\.checks).filter { !$0.passed }
        emit("Overall: \(failed.isEmpty ? "PASS" : "FAIL (\(failed.count) checks)")")
        if !failed.isEmpty {
            throw TurboQuantBenchError.checksFailed(failed.count)
        }
    }

    // MARK: - One context

    nonisolated private static func measureContext(
        context: ModelContext,
        targetTokens: Int,
        corpus: Corpus,
        options: Options,
        emit: @Sendable (String) -> Void
    ) async throws -> ContextRecord {
        let model = context.model
        let promptText = try await buildPrompt(
            context: context, corpus: corpus.text, targetTokens: targetTokens)
        let prepared = try await context.processor.prepare(
            input: UserInput(chat: [.user(promptText)]))
        guard prepared.text.tokens.ndim == 1 else {
            throw TurboQuantBenchError.unexpectedPromptShape(prepared.text.tokens.shape)
        }
        let promptTokenIDs = prepared.text.tokens.asArray(Int.self)
        let promptDigest = SHA256.hash(
            data: promptTokenIDs.map { String($0) }.joined(separator: ",").data(using: .utf8)!
        ).map { String(format: "%02x", $0) }.joined()

        // Shared prefill, on the unquantized cache every scheme prefills on.
        Memory.clearCache()
        Memory.peakMemory = 0
        var prefillCache = try model.newCache(parameters: nil)
        let attentionLayers = prefillCache.indices.filter { !(prefillCache[$0] is ArraysCache) }
        let prefillStart = ContinuousClock.now
        let warmed = try PrefillExecutor.run(
            model: model, text: prepared.text, cache: prefillCache,
            prefillStepSize: options.prefillStep)
        let prefillSeconds = seconds(since: prefillStart)
        let prefill = PrefillRecord(
            seconds: prefillSeconds,
            tokensPerSecond: Double(promptTokenIDs.count - 1) / prefillSeconds,
            peakGB: gb(Memory.peakMemory),
            activeAfterGB: gb(Memory.activeMemory),
            chunkSize: options.prefillStep,
            carriesModelState: warmed.state != nil)
        let kvDType = prefillCache[attentionLayers[0]].state.first.map { "\($0.dtype)" } ?? "?"
        guard
            let snapshot = HybridCacheSnapshot.capture(
                cache: prefillCache, offset: promptTokenIDs.count - 1, type: .leaf)
        else {
            throw TurboQuantBenchError.snapshotUnsupported
        }
        prefillCache.removeAll()
        Memory.clearCache()
        emit(
            "prompt=\(promptTokenIDs.count) tok (attention layers=\(attentionLayers.count), "
                + "kv dtype=\(kvDType)) prefill=\(fmt(prefillSeconds))s "
                + "@ \(fmt(prefill.tokensPerSecond)) tok/s peak=\(fmt(prefill.peakGB))GB "
                + "snapshot=\(fmt(gb(snapshot.memoryBytes)))GB")

        // Quality: the reference first, then every scheme forced along it.
        var reference = Reference()
        var quality: [QualityRecord] = []
        for scheme in options.schemes {
            var cache = try snapshot.restore()
            let record = try teacherForced(
                model: model, cache: &cache, remainder: warmed.remainder, scheme: scheme,
                label: scheme.name, steps: options.maxNew, attentionLayers: attentionLayers,
                reference: &reference)
            cache.removeAll()
            Memory.clearCache()
            quality.append(record)
            emit(qualityLine(record))
        }
        let referenceText = context.tokenizer.decode(tokenIds: reference.tokens)

        // The noise floor: the same unquantized cache, prefilled in other chunks.
        var noiseFloor: QualityRecord?
        if options.noiseFloorStep > 0 {
            var cache = try model.newCache(parameters: nil)
            let alternate = try PrefillExecutor.run(
                model: model, text: prepared.text, cache: cache,
                prefillStepSize: options.noiseFloorStep)
            let record = try teacherForced(
                model: model, cache: &cache, remainder: alternate.remainder,
                scheme: options.schemes[0], label: "fp16-chunk\(options.noiseFloorStep)",
                steps: options.maxNew, attentionLayers: attentionLayers, reference: &reference)
            cache.removeAll()
            Memory.clearCache()
            noiseFloor = record
            emit(qualityLine(record))
        }
        reference.logprobs.removeAll()
        Memory.clearCache()

        // Speed and memory: the production iterator, rounds reversed every
        // other time so each arm sees the same thermal positions.
        var speed: [SpeedRecord] = []
        for round in 0..<options.runs {
            let order = round.isMultiple(of: 2) ? options.schemes : options.schemes.reversed()
            for scheme in order {
                let record = try decodePass(
                    model: model, snapshot: snapshot, fullText: prepared.text,
                    remainder: warmed.remainder, scheme: scheme, round: round,
                    steps: options.maxNew, prefillStep: options.prefillStep,
                    prefillPeakGB: prefill.peakGB, referenceTokens: reference.tokens)
                speed.append(record)
                emit(
                    "speed \(scheme.name) round=\(round): "
                        + "decode=\(fmt(record.decodeTokensPerSecond)) tok/s "
                        + "prime=\(fmt(record.primeSeconds * 1000))ms "
                        + "firstToken=\(fmt(record.firstTokenSeconds * 1000))ms "
                        + "switchOver=\(fmt(record.switchOverSeconds * 1000))ms "
                        + "decodePeak=\(fmt(record.decodePhasePeakGB))GB "
                        + "runPeak=\(fmt(record.runPeakGB))GB "
                        + "residentGrowth=\(fmt(record.residentGrowthGB))GB "
                        + "greedyDiverges=\(record.firstGreedyDivergence.map(String.init) ?? "never") "
                        + "thermal=\(record.thermalState)")
            }
        }

        // Harness checks. Step 0 is scored before any scheme engages, so a
        // nonzero KL there means the arms did not start from the same cache.
        // The unquantized speed passes run the production iterator over the
        // same restored cache the reference pass forwarded by hand, so they
        // must reproduce its greedy stream.
        var checks: [BenchmarkHarness.CheckResult] = quality.dropFirst().map { record in
            BenchmarkHarness.CheckResult(
                name: "\(record.arm) step 0 KL is 0",
                passed: record.stepZeroKL == 0,
                detail: "stepZeroKL=\(record.stepZeroKL)")
        }
        checks += speed.filter { $0.scheme == options.schemes[0].name }.map { record in
            BenchmarkHarness.CheckResult(
                name: "\(record.scheme) round \(record.round) reproduces the reference stream",
                passed: record.firstGreedyDivergence == nil
                    && record.tokenIDs.count == reference.tokens.count,
                detail: "firstDivergence=\(record.firstGreedyDivergence.map(String.init) ?? "none")"
            )
        }
        for check in checks where !check.passed {
            emit("CHECK FAILED: \(check.name) (\(check.detail))")
        }

        return ContextRecord(
            targetTokens: targetTokens,
            promptTokens: promptTokenIDs.count,
            promptTokenSHA256: promptDigest,
            kvDType: kvDType,
            attentionLayers: attentionLayers.count,
            prefill: prefill,
            snapshotGB: gb(snapshot.memoryBytes),
            referenceText: referenceText,
            referenceTokenIDs: reference.tokens,
            quality: quality,
            noiseFloor: noiseFloor,
            speed: speed,
            checks: checks)
    }

    // MARK: - Quality pass

    /// The reference stream: the unquantized arm's greedy tokens and the f32
    /// log probabilities they were chosen from.
    nonisolated private struct Reference {
        var tokens: [Int] = []
        var logprobs: [MLXArray] = []
        var margins: [Float] = []
        var isRecorded: Bool { !tokens.isEmpty }
    }

    /// Run `steps` decode steps one token per forward, as `TokenIterator.step`
    /// does: the remainder token on the unquantized cache, the scheme's
    /// conversion, then every later token through the converted cache. The
    /// first call records the reference; later calls force its tokens and score
    /// each step's distribution against it.
    nonisolated private static func teacherForced(
        model: any LanguageModel,
        cache: inout [any KVCache],
        remainder: LMInput.Text,
        scheme: KVScheme,
        label: String,
        steps: Int,
        attentionLayers: [Int],
        reference: inout Reference
    ) throws -> QualityRecord {
        let recording = !reference.isRecorded
        // The iterator built by `PrefillExecutor.makeIterator` starts from a
        // nil model state; so does this pass.
        var state: LMOutput.State?

        func forward(_ text: LMInput.Text) -> MLXArray {
            let result = withPreparedCache(cache, lengths: text.sequenceLengths) {
                model(text[text: .newAxis], cache: cache, state: state)
            }
            state = result.state
            let logits = result.logits[0, -1]
            eval([logits] + cache.flatMap { $0.state })
            return logits
        }

        var logits = forward(remainder)
        if let configuration = scheme.configuration {
            let applied = try applyKVCacheConfiguration(cache: &cache, configuration: configuration)
            guard applied.convertedLayerCount == attentionLayers.count, applied.skipped.isEmpty
            else {
                throw TurboQuantBenchError.conversion(
                    scheme: scheme.name, converted: applied.convertedLayerCount,
                    expected: attentionLayers.count, skipped: applied.skipped.count)
            }
        }

        var kl: [Float] = []
        var topOneMismatches: [Int] = []
        var referenceTokenLogprob: [Float] = []
        var absProbabilityDelta: [Float] = []
        kl.reserveCapacity(steps)

        for step in 0..<steps {
            let logprobs = logSoftmax(logits.asType(.float32), axis: -1)
            let token: Int
            if recording {
                let top = argMax(logprobs)
                let topTwo = sorted(logprobs)[(logprobs.dim(0) - 2)...]
                eval(top, topTwo)
                token = top.item(Int.self)
                let pair = topTwo.asArray(Float.self)
                reference.tokens.append(token)
                reference.logprobs.append(logprobs)
                reference.margins.append(pair[1] - pair[0])
                kl.append(0)
                referenceTokenLogprob.append(pair[1])
                absProbabilityDelta.append(0)
            } else {
                token = reference.tokens[step]
                let target = reference.logprobs[step]
                let divergence = sum(exp(target) * (target - logprobs))
                let top = argMax(logprobs)
                let forcedLogprob = logprobs[token]
                let referenceLogprob = target[token]
                eval(divergence, top, forcedLogprob, referenceLogprob)
                kl.append(divergence.item(Float.self))
                if top.item(Int.self) != token { topOneMismatches.append(step) }
                let forced = forcedLogprob.item(Float.self)
                referenceTokenLogprob.append(forced)
                let delta =
                    Foundation.exp(Double(forced))
                    - Foundation.exp(Double(referenceLogprob.item(Float.self)))
                absProbabilityDelta.append(Float(abs(delta)))
            }
            if step == steps - 1 { break }
            logits = forward(LMInput.Text(tokens: MLXArray([Int32(token)])))
        }

        let layout = try verifyAndMeasure(
            cache: cache, scheme: scheme, attentionLayers: attentionLayers)
        // Step 0 is scored before the scheme engages; the statistics cover the
        // decode steps after it.
        let decodeKL = Array(kl.dropFirst())
        let decodeMismatches = topOneMismatches.filter { $0 > 0 }
        return QualityRecord(
            arm: label,
            scheme: scheme.name,
            steps: steps,
            stepZeroKL: kl.first ?? 0,
            klMean: mean(decodeKL),
            klMedian: percentile(decodeKL, 0.5),
            klP90: percentile(decodeKL, 0.9),
            klP99: percentile(decodeKL, 0.99),
            klMax: decodeKL.max() ?? 0,
            topOneAgreement: recording
                ? 1 : 1 - Double(decodeMismatches.count) / Double(max(steps - 1, 1)),
            topOneMismatchSteps: topOneMismatches,
            referenceMarginsAtMismatches: topOneMismatches.map { reference.margins[$0] },
            meanAbsReferenceProbabilityDelta: mean(Array(absProbabilityDelta.dropFirst())),
            meanReferenceTokenLogprob: mean(Array(referenceTokenLogprob.dropFirst())),
            klPerStep: kl,
            layout: layout)
    }

    /// Check the cache is in the scheme's realized form on every attention
    /// layer, then measure it.
    nonisolated private static func verifyAndMeasure(
        cache: [any KVCache],
        scheme: KVScheme,
        attentionLayers: [Int]
    ) throws -> LayoutRecord {
        var liveBytes = 0
        var allocatedBytes = 0
        var recurrentBytes = 0
        var classes: Set<String> = []
        let attention = Set(attentionLayers)
        for (index, layer) in cache.enumerated() {
            let stateBytes = layer.state.reduce(0) { $0 + $1.nbytes }
            guard attention.contains(index) else {
                recurrentBytes += stateBytes
                continue
            }
            liveBytes += stateBytes
            classes.insert(String(describing: type(of: layer)))
            switch (scheme.expectation, layer) {
            case (.unquantized, let simple as KVCacheSimple):
                allocatedBytes += simple.innerState().reduce(0) { $0 + $1.nbytes }
            case (.affine(let bits), let quantized as QuantizedKVCache)
            where quantized.bits == bits:
                allocatedBytes += quantized.innerState().reduce(0) { $0 + $1.nbytes }
            case (.turbo(let keyBits, let valueBits), let turbo as TurboQuantKVCache)
            where turbo.isCompressed && turbo.keyBits == keyBits && turbo.valueBits == valueBits
                && turbo.affineKeyMode == (keyBits == 8) && turbo.rawKeyMode == (keyBits == 0):
                allocatedBytes += turbo.memoryBytes
            default:
                throw TurboQuantBenchError.unexpectedLayer(
                    scheme: scheme.name, layer: index, found: "\(layer)")
            }
        }
        let offset = cache[attentionLayers[0]].offset
        return LayoutRecord(
            layerClasses: classes.sorted(),
            tokens: offset,
            liveKVBytes: liveBytes,
            liveKVBytesPerToken: Double(liveBytes) / Double(max(offset, 1)),
            allocatedKVBytes: allocatedBytes,
            recurrentStateBytes: recurrentBytes)
    }

    // MARK: - Speed pass

    /// One timed decode through the chunked Prefill Strategy route: the
    /// restored cache, the remainder token, and the scheme's `kvCache`
    /// parameter, converted by the iterator as a generation would be.
    nonisolated private static func decodePass(
        model: any LanguageModel,
        snapshot: HybridCacheSnapshot,
        fullText: LMInput.Text,
        remainder: LMInput.Text,
        scheme: KVScheme,
        round: Int,
        steps: Int,
        prefillStep: Int,
        prefillPeakGB: Double,
        referenceTokens: [Int]
    ) throws -> SpeedRecord {
        var agentParameters = AgentGenerateParameters(
            maxTokens: steps, temperature: 0, topP: 1, topK: 0, minP: 0)
        agentParameters.repetitionPenalty = nil
        agentParameters.prefillStepSize = prefillStep
        var parameters = LLMActor.makeGenerateParameters(from: agentParameters)
        parameters.kvCache = scheme.configuration

        Memory.clearCache()
        Memory.peakMemory = 0
        let activeBefore = Memory.activeMemory
        let thermal = thermalStateName()
        var cache = try snapshot.restore()
        let primeStart = ContinuousClock.now
        var iterator = try PrefillExecutor.makeIterator(
            model: model, fullText: fullText, remainder: remainder, cache: &cache,
            parameters: parameters)
        // The iterator owns the cache now; a reference held here would keep the
        // unquantized layers alive after the iterator has replaced them.
        cache.removeAll()
        // Settle the remainder forward inside the prime window. The conversion
        // the iterator applied after it is lazy and not yet submitted.
        Stream.gpu.synchronize()
        let primeEnd = ContinuousClock.now

        var tokens: [Int] = []
        var stamps: [ContinuousClock.Instant] = []
        tokens.reserveCapacity(steps)
        stamps.reserveCapacity(steps)
        while let token = iterator.next() {
            tokens.append(token)
            stamps.append(.now)
        }
        // Read memory while the iterator still holds the cache.
        let (peak, activeAfter) = withExtendedLifetime(iterator) {
            Stream.gpu.synchronize()
            return (Memory.peakMemory, Memory.activeMemory)
        }

        // `next()` builds step i+1's graph, then returns token i. Building the
        // first step's graph is where TurboQuant compresses the raw cache, in
        // blocking evals, so that work lands before stamps[0]; affine's lazy
        // quantization runs with step 1 on the GPU, before stamps[1]. Every
        // scheme's one-time switch-over is therefore inside primeEnd..stamps[1],
        // and the steady state is stamps[1] onward.
        let steadyTokens = max(tokens.count - 2, 0)
        let steadySeconds =
            stamps.count > 2 ? stamps[1].duration(to: stamps[stamps.count - 1]).seconds : 0
        let firstToken = stamps.first.map { primeEnd.duration(to: $0).seconds } ?? 0
        let switchOver = stamps.count > 1 ? primeEnd.duration(to: stamps[1]).seconds : 0
        let decodePeakGB = gb(peak - snapshot.memoryBytes)
        return SpeedRecord(
            scheme: scheme.name,
            round: round,
            generatedTokens: tokens.count,
            primeSeconds: primeStart.duration(to: primeEnd).seconds,
            firstTokenSeconds: firstToken,
            switchOverSeconds: switchOver,
            decodeTokensPerSecond: steadySeconds > 0 ? Double(steadyTokens) / steadySeconds : 0,
            decodePhasePeakGB: decodePeakGB,
            runPeakGB: max(prefillPeakGB, decodePeakGB),
            residentGrowthGB: gb(activeAfter - activeBefore),
            firstGreedyDivergence: zip(tokens, referenceTokens).enumerated()
                .first { $0.element.0 != $0.element.1 }?.offset,
            thermalState: thermal,
            tokenIDs: tokens)
    }

    // MARK: - Prompt

    nonisolated private static func buildPrompt(
        context: ModelContext,
        corpus: String,
        targetTokens: Int
    ) async throws -> String {
        let overhead = try await context.processor.prepare(
            input: UserInput(chat: [.user(question)])
        ).text.tokens.dim(-1)
        let budget = targetTokens - overhead
        guard budget > 0 else { return question }
        let ids = context.tokenizer.encode(text: corpus, addSpecialTokens: false)
        guard ids.count >= budget else {
            throw TurboQuantBenchError.corpusTooShort(tokens: ids.count, needed: budget)
        }
        return context.tokenizer.decode(tokenIds: Array(ids.prefix(budget))) + question
    }

    // MARK: - Schemes and options

    nonisolated struct KVScheme: Sendable {
        enum Expectation: Sendable {
            case unquantized
            case affine(bits: Int)
            case turbo(keyBits: Int, valueBits: Int)
        }

        let name: String
        /// `nil` for the unquantized reference.
        let configuration: KVCacheConfiguration?
        let expectation: Expectation

        /// `fp16`, `affine<bits>`, or `turbo<key>v<value>` with keys fp16 (`0`)
        /// or 8-bit affine and 3- or 4-bit values: the schemes the vendor's
        /// boundary-layer protection leaves alone, so every attention layer
        /// takes the scheme and the check stays exact.
        static func parse(_ name: String) throws -> KVScheme {
            if name == "fp16" {
                return KVScheme(name: name, configuration: nil, expectation: .unquantized)
            }
            if name.hasPrefix("affine"), let bits = Int(name.dropFirst("affine".count)) {
                let configuration = KVCacheConfiguration(
                    strategy: .affine(try AffineKVCacheConfiguration(bits: bits)),
                    compatibility: .requireAllLayers)
                return KVScheme(
                    name: name, configuration: configuration, expectation: .affine(bits: bits))
            }
            if name.hasPrefix("turbo") {
                let parts = name.dropFirst("turbo".count).split(separator: "v")
                if parts.count == 2, let keyBits = Int(parts[0]), let valueBits = Int(parts[1]),
                    keyBits == 0 || keyBits == 8, valueBits == 3 || valueBits == 4
                {
                    let configuration = KVCacheConfiguration(
                        strategy: .turboQuant(
                            try TurboQuantKVCacheConfiguration(
                                keyPrecision: keyBits == 8 ? .affineEightBit : .fp16,
                                valuePrecision: valueBits == 4 ? .fourBit : .threeBit)),
                        compatibility: .requireAllLayers)
                    return KVScheme(
                        name: name, configuration: configuration,
                        expectation: .turbo(keyBits: keyBits, valueBits: valueBits))
                }
            }
            throw TurboQuantBenchError.unsupportedScheme(name)
        }
    }

    nonisolated struct Options: Sendable {
        let corpusPath: String
        let contexts: [Int]
        /// The unquantized reference first, then the schemes under test.
        let schemes: [KVScheme]
        let maxNew: Int
        let runs: Int
        let prefillStep: Int
        let noiseFloorStep: Int

        var record: OptionsRecord {
            OptionsRecord(
                contexts: contexts, schemes: schemes.map(\.name), maxNew: maxNew, runs: runs,
                prefillStep: prefillStep, noiseFloorStep: noiseFloorStep)
        }

        static func parse(_ args: [String], defaultPrefillStep: Int) throws -> Options {
            func value(_ flag: String) -> String? {
                guard let index = args.firstIndex(of: flag), index + 1 < args.count else {
                    return nil
                }
                return args[index + 1]
            }
            func intList(_ flag: String) -> [Int]? {
                let parsed = value(flag)?.split(separator: ",").compactMap { Int($0) } ?? []
                return parsed.isEmpty ? nil : parsed
            }
            guard let corpusPath = value("--bench-corpus") else {
                throw TurboQuantBenchError.corpusFlagMissing
            }
            let names = (value("--bench-schemes") ?? "fp16,turbo8v4,turbo0v4")
                .split(separator: ",").map(String.init).filter { $0 != "fp16" }
            let maxNew = value("--bench-max-new").flatMap { Int($0) } ?? 512
            guard maxNew >= 3 else { throw TurboQuantBenchError.tooFewSteps(maxNew) }
            return Options(
                corpusPath: corpusPath,
                contexts: intList("--bench-contexts") ?? [8192, 32_768, 65_536],
                schemes: try (["fp16"] + names).map(KVScheme.parse),
                maxNew: maxNew,
                runs: value("--bench-runs").flatMap { Int($0) } ?? 2,
                prefillStep: defaultPrefillStep,
                noiseFloorStep: value("--bench-noise-floor-step").flatMap { Int($0) } ?? 768)
        }
    }

    nonisolated struct Corpus: Sendable {
        let path: String
        let fileCount: Int
        let text: String
        let sha256: String

        /// A file, or a directory whose `.md` files are joined in name order.
        static func load(path: String) throws -> Corpus {
            let url = URL(fileURLWithPath: path)
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory) else {
                throw TurboQuantBenchError.corpusNotFound(path)
            }
            let files: [URL]
            if isDirectory.boolValue {
                files = try FileManager.default.contentsOfDirectory(
                    at: url, includingPropertiesForKeys: nil
                )
                .filter { $0.pathExtension == "md" }
                .sorted { $0.lastPathComponent < $1.lastPathComponent }
            } else {
                files = [url]
            }
            let text = try files.map { try String(contentsOf: $0, encoding: .utf8) }
                .joined(separator: "\n\n")
            let digest = SHA256.hash(data: Data(text.utf8))
                .map { String(format: "%02x", $0) }.joined()
            return Corpus(path: path, fileCount: files.count, text: text, sha256: digest)
        }
    }

    // MARK: - Records

    nonisolated struct LoadRecord: Codable, Sendable {
        let model: String
        let modelID: String
        let modelDirectory: String
        let hardware: String
        let osVersion: String
        let sourceRevision: String?
        let nice: Int
        let loadSeconds: Double
        let activeAfterLoadGB: Double
        let peakDuringLoadGB: Double
    }

    nonisolated struct OptionsRecord: Codable, Sendable {
        let contexts: [Int]
        let schemes: [String]
        let maxNew: Int
        let runs: Int
        let prefillStep: Int
        let noiseFloorStep: Int
    }

    nonisolated struct CorpusRecord: Codable, Sendable {
        let path: String
        let files: Int
        let bytes: Int
        let sha256: String
    }

    nonisolated struct PrefillRecord: Codable, Sendable {
        let seconds: Double
        let tokensPerSecond: Double
        let peakGB: Double
        let activeAfterGB: Double
        let chunkSize: Int
        let carriesModelState: Bool
    }

    nonisolated struct LayoutRecord: Codable, Sendable {
        let layerClasses: [String]
        /// Attention-layer offset at the end of the pass.
        let tokens: Int
        /// Attention state as the cache reports it, sliced to its offset.
        let liveKVBytes: Int
        let liveKVBytesPerToken: Double
        /// What the attention layers hold, step padding included.
        let allocatedKVBytes: Int
        /// The fixed recurrent (GatedDeltaNet) state, the same in every scheme.
        let recurrentStateBytes: Int
    }

    nonisolated struct QualityRecord: Codable, Sendable {
        let arm: String
        let scheme: String
        let steps: Int
        /// Must be 0: step 0 is scored before the scheme engages.
        let stepZeroKL: Float
        /// KL(reference ‖ arm) over steps 1..<steps, in nats.
        let klMean: Double
        let klMedian: Double
        let klP90: Double
        let klP99: Double
        let klMax: Float
        /// Share of steps 1..<steps whose argmax is the reference token.
        let topOneAgreement: Double
        let topOneMismatchSteps: [Int]
        /// The reference's top-1 minus top-2 log probability at each mismatch.
        let referenceMarginsAtMismatches: [Float]
        let meanAbsReferenceProbabilityDelta: Double
        let meanReferenceTokenLogprob: Double
        let klPerStep: [Float]
        let layout: LayoutRecord
    }

    nonisolated struct SpeedRecord: Codable, Sendable {
        let scheme: String
        let round: Int
        let generatedTokens: Int
        /// Iterator construction and the remainder token's forward, on the
        /// unquantized cache in every arm.
        let primeSeconds: Double
        /// From the prime to token 0's return: TurboQuant's compression of the
        /// raw cache (blocking, in the first step's graph build); a graph build
        /// for the other schemes.
        let firstTokenSeconds: Double
        /// From the prime to token 1's return: every scheme's one-time
        /// conversion plus the first decode step on the converted cache.
        let switchOverSeconds: Double
        /// Steady-state decode, every step after the first.
        let decodeTokensPerSecond: Double
        /// Peak while decoding, less the resident snapshot.
        let decodePhasePeakGB: Double
        /// max(shared prefill peak, decode-phase peak).
        let runPeakGB: Double
        let residentGrowthGB: Double
        /// First position where this arm's free-running greedy stream leaves
        /// the reference's; nil when it never does.
        let firstGreedyDivergence: Int?
        let thermalState: String
        let tokenIDs: [Int]
    }

    nonisolated struct ContextRecord: Codable, Sendable {
        let targetTokens: Int
        let promptTokens: Int
        let promptTokenSHA256: String
        let kvDType: String
        let attentionLayers: Int
        let prefill: PrefillRecord
        let snapshotGB: Double
        let referenceText: String
        let referenceTokenIDs: [Int]
        let quality: [QualityRecord]
        let noiseFloor: QualityRecord?
        let speed: [SpeedRecord]
        let checks: [BenchmarkHarness.CheckResult]
    }

    nonisolated struct Report: Codable, Sendable {
        let date: String
        let load: LoadRecord
        let options: OptionsRecord
        let corpus: CorpusRecord
        let question: String
        let contexts: [ContextRecord]
    }

    // MARK: - Reporting

    nonisolated private static func writeReport(_ report: Report, to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(report).write(to: url, options: .atomic)
    }

    nonisolated private static func qualityLine(_ record: QualityRecord) -> String {
        "quality \(record.arm): KL mean=\(sci(record.klMean)) median=\(sci(record.klMedian)) "
            + "p99=\(sci(record.klP99)) max=\(sci(Double(record.klMax))) "
            + "step0=\(sci(Double(record.stepZeroKL))) "
            + "top1=\(fmt(record.topOneAgreement * 100))% "
            + "(mismatches=\(record.topOneMismatchSteps.count)) "
            + "kv=\(fmt(record.layout.liveKVBytesPerToken)) B/tok "
            + "allocated=\(fmt(gb(record.layout.allocatedKVBytes)))GB "
            + "classes=\(record.layout.layerClasses)"
    }

    nonisolated private static func emitSummary(
        _ contexts: [ContextRecord], emit: @Sendable (String) -> Void
    ) {
        emit("=== summary ===")
        for context in contexts {
            let baseline = median(
                context.speed.filter { $0.scheme == "fp16" }.map(\.decodeTokensPerSecond))
            let baselineBytes = context.quality.first?.layout.liveKVBytesPerToken ?? 0
            emit(
                "ctx=\(context.targetTokens) prompt=\(context.promptTokens) "
                    + "prefillPeak=\(fmt(context.prefill.peakGB))GB "
                    + "noiseFloorKL=\(context.noiseFloor.map { sci($0.klMean) } ?? "-")")
            for record in context.quality {
                let rates = context.speed.filter { $0.scheme == record.scheme }
                let rate = median(rates.map(\.decodeTokensPerSecond))
                let runPeak = rates.map(\.runPeakGB).max() ?? 0
                let delta = baseline > 0 ? (rate / baseline - 1) * 100 : 0
                emit(
                    "  \(record.scheme): KL mean=\(sci(record.klMean)) p99=\(sci(record.klP99)) "
                        + "top1=\(fmt(record.topOneAgreement * 100))% "
                        + "decode=\(fmt(rate)) tok/s (\(fmt(delta))%) "
                        + "kv=\(fmt(record.layout.liveKVBytesPerToken)) B/tok "
                        + "(\(fmt(baselineBytes / max(record.layout.liveKVBytesPerToken, 1)))x) "
                        + "runPeak=\(fmt(runPeak))GB")
            }
        }
    }

    // MARK: - Helpers

    nonisolated private static func thermalStateName() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: "nominal"
        case .fair: "fair"
        case .serious: "serious"
        case .critical: "critical"
        @unknown default: "unknown"
        }
    }

    nonisolated private static func mean(_ values: [Float]) -> Double {
        values.isEmpty ? 0 : values.reduce(0.0) { $0 + Double($1) } / Double(values.count)
    }

    nonisolated private static func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        guard !sorted.isEmpty else { return 0 }
        return (sorted[(sorted.count - 1) / 2] + sorted[sorted.count / 2]) / 2
    }

    nonisolated private static func percentile(_ values: [Float], _ quantile: Double) -> Double {
        let sorted = values.sorted()
        guard !sorted.isEmpty else { return 0 }
        let index = Int((Double(sorted.count - 1) * quantile).rounded())
        return Double(sorted[min(max(index, 0), sorted.count - 1)])
    }

    nonisolated private static func gb(_ bytes: Int) -> Double {
        Double(bytes) / 1e9
    }

    nonisolated private static func seconds(since start: ContinuousClock.Instant) -> Double {
        start.duration(to: .now).seconds
    }

    nonisolated private static func fmt(_ value: Double) -> String {
        String(format: "%.2f", value)
    }

    nonisolated private static func sci(_ value: Double) -> String {
        String(format: "%.3e", value)
    }

    nonisolated private static func timestamp() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss.SSS"
        return formatter.string(from: Date())
    }
}

nonisolated enum TurboQuantBenchError: LocalizedError {
    case corpusFlagMissing
    case corpusNotFound(String)
    case corpusTooShort(tokens: Int, needed: Int)
    case unsupportedScheme(String)
    case tooFewSteps(Int)
    case unexpectedPromptShape([Int])
    case snapshotUnsupported
    case conversion(scheme: String, converted: Int, expected: Int, skipped: Int)
    case unexpectedLayer(scheme: String, layer: Int, found: String)
    case checksFailed(Int)

    var errorDescription: String? {
        switch self {
        case .corpusFlagMissing:
            "TurboQuant bench: pass --bench-corpus <file or directory>"
        case .corpusNotFound(let path):
            "TurboQuant bench: no corpus at \(path)"
        case .corpusTooShort(let tokens, let needed):
            "TurboQuant bench: corpus has \(tokens) tokens, the prompt needs \(needed)"
        case .unsupportedScheme(let name):
            "TurboQuant bench: unsupported scheme '\(name)' (fp16, affine<bits>, "
                + "turbo0v3, turbo0v4, turbo8v3, turbo8v4)"
        case .tooFewSteps(let steps):
            "TurboQuant bench: --bench-max-new \(steps) leaves no steady-state decode"
        case .unexpectedPromptShape(let shape):
            "TurboQuant bench: expected a 1D prompt, got \(shape)"
        case .snapshotUnsupported:
            "TurboQuant bench: the prefilled cache holds a layer the snapshot cannot copy"
        case .conversion(let scheme, let converted, let expected, let skipped):
            "TurboQuant bench: \(scheme) converted \(converted) of \(expected) attention "
                + "layers (\(skipped) skipped)"
        case .unexpectedLayer(let scheme, let layer, let found):
            "TurboQuant bench: \(scheme) left attention layer \(layer) as \(found)"
        case .checksFailed(let count):
            "TurboQuant bench: \(count) harness checks failed (see the report)"
        }
    }
}
