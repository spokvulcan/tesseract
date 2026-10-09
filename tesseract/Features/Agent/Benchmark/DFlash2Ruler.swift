import CryptoKit
import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// The speed ruler (`--dflash2-bench --bench-ruler`): one Release run that
/// measures cold prefill on the frozen 2K/8K/32K prompts and DFlash2 decode
/// on every fixture, and writes one JSON report.
///
/// Decode runs first, before prefill heats the machine: DFlash2 at the
/// production block size on each fixture, over the production prefill (the
/// app driver's pipelined chunks up to the speculative split, then the
/// iterator's capture prefill of the tail; the driver's share runs once per
/// fixture and is restored per run). Prefill is then timed on the same cold
/// path, from the first chunk to the first sampled token. `--bench-check`
/// then adds, after every timed run, the reference the DFlash2 bench has
/// always checked against (`TokenIterator` greedy decoding of the whole
/// prompt) and the same AR teacher-forced along each fixture's stream.
///
/// Flags: `--bench-fixture-dir DIR` (required), `--bench-prefill a,b,c`
/// (files in DIR, default the 2K/8K/32K prompts, `none` skips),
/// `--bench-fixtures a,b` (default travel,summary,math,code, `none` skips),
/// `--bench-max-tokens` (default 512), `--bench-runs` (DFlash2 runs per
/// fixture, default 1), `--bench-prefill-runs` (default 1), `--bench-check`,
/// `--bench-round-timings` (each round's milliseconds and accepted drafts),
/// `--bench-cooldown S` (idle seconds before every timed run, default 30:
/// sustained load throttles the GPU, and a run's speed otherwise depends on
/// what ran before it), `--bench-kv-scheme turbo8v4|turbo0v4` (the app's KV
/// Cache Compression: attention layers compress once prefill ends; default
/// bf16), `--bench-json PATH`.
nonisolated enum DFlash2Ruler {

    /// The production chunk of the app driver and the iterator's tail
    /// (`AgentGenerateParameters.prefillStepSize`).
    static let prefillStepSize = 1024

    struct GPUReference: Encodable {
        let label: String
        let teraflops: Double
    }

    struct PrefillRecord: Encodable {
        let file: String
        let sha256: String
        let promptTokens: Int
        let split: Int
        let run: Int
        let seconds: Double
        let tokensPerSecond: Double
        let peakGB: Double
        /// Checksum of every cache array's bits after the prefill: two
        /// builds with equal digests prefilled bitwise identically.
        let stateDigest: String
    }

    struct ARRecord: Encodable {
        let tokens: Int
        let decodeSeconds: Double
        let tokensPerSecond: Double
        let stream: [Int]
    }

    struct DecodeRecord: Encodable {
        let run: Int
        let tokens: Int
        let decodeSeconds: Double
        let tokensPerSecond: Double
        let accepted: Int
        let proposed: Int
        let rounds: Int
        let tokensPerRound: Double
        let msPerRound: Double
        /// `MATCH` or `DIVERGED at +N` against the AR reference; nil unchecked.
        var identity: String?
        let stream: [Int]
        /// Per round, with `--bench-round-timings`: milliseconds since the
        /// previous round's tokens arrived, and drafts accepted.
        let roundMilliseconds: [Double]?
        let roundAccepted: [Int]?
    }

    /// The target teacher-forced along a DFlash2 stream with AR decoding:
    /// every position where the stream's token is not the target's argmax
    /// given the stream's own prefix.
    struct ForcedRecord: Encodable {
        struct Departure: Encodable {
            let position: Int
            let token: Int
            let argmax: Int
            let tokenLogit: Float
            let argmaxLogit: Float
            /// The logit gap in bf16 ulps at the argmax logit's magnitude.
            let gapUlps: Double
        }
        let departures: [Departure]
    }

    struct FixtureRecord: Encodable {
        let name: String
        let sha256: String
        let promptTokens: Int
        let split: Int
        var ar: ARRecord?
        /// With `--bench-check`: run 0's stream, teacher-forced.
        var forced: ForcedRecord?
        var runs: [DecodeRecord]
    }

    struct Report: Encodable {
        let sourceRevision: String?
        let model: String
        let maxNewTokens: Int
        let blockSize: Int
        let prefillStepSize: Int
        let cooldownSeconds: Int
        /// `--bench-kv-scheme`; nil for bf16 attention caches.
        let kvScheme: String?
        /// The run's MLX_* and DFLASH2_* knobs.
        let environment: [String: String]
        let thermalStart: String
        let thermalEnd: String
        let gpuReference: [GPUReference]
        let prefill: [PrefillRecord]
        let fixtures: [FixtureRecord]
    }

    private static var arguments: [String] { ProcessInfo.processInfo.arguments }

    private static func option(_ name: String) -> String? {
        guard let i = arguments.firstIndex(of: name), i + 1 < arguments.count else { return nil }
        return arguments[i + 1]
    }

    private static func list(_ name: String, default fallback: [String]) -> [String] {
        guard let raw = option(name) else { return fallback }
        return raw == "none" ? [] : raw.split(separator: ",").map(String.init)
    }

    private static func positive(_ name: String, default fallback: Int) -> Int {
        option(name).flatMap(Int.init).flatMap { $0 > 0 ? $0 : nil } ?? fallback
    }

    private static var cooldownSeconds: Int {
        option("--bench-cooldown").flatMap(Int.init).map { max(0, $0) } ?? 30
    }

    /// Idle the GPU before a timed run.
    private static func cooldown() {
        Stream.gpu.synchronize()
        if cooldownSeconds > 0 { Thread.sleep(forTimeInterval: Double(cooldownSeconds)) }
    }

    static func run(
        context: ModelContext, draft: any DFlash2DrafterModel, emit: (String) -> Void
    ) async throws {
        guard let directory = option("--bench-fixture-dir") else {
            throw RulerError.missingFixtureDirectory
        }
        let check = arguments.contains("--bench-check")
        let maxNewTokens = positive("--bench-max-tokens", default: 512)
        let runs = positive("--bench-runs", default: 1)
        let prefillRuns = positive("--bench-prefill-runs", default: 1)
        let prefillFiles = list(
            "--bench-prefill", default: ["prefill-2k.txt", "prefill-8k.txt", "prefill-32k.txt"])
        let fixtures = list("--bench-fixtures", default: ["travel", "summary", "math", "code"])
        let thermalStart = thermalName()
        var gpu: [GPUReference] = [gpuReference("start")]
        emit(
            "[ruler] blocks \(DFlash2Support.blockSize) max tokens \(maxNewTokens) "
                + "prefill step \(prefillStepSize) thermal \(thermalStart) "
                + String(format: "gpu %.2f TFLOP/s", gpu[0].teraflops))

        // Pipelines and traces compile outside every timed region: every
        // block width a capped generation ends on, on a short prompt (the
        // repeated text accepts whole blocks, so 9 + k tokens end on a
        // k + 1 wide round); then, on ~3.3K tokens, the driver's and the
        // iterator's full 1024-token chunks, verify passes past the 1024-key
        // SDPA switch, and the drafter's context-cache compaction (~40
        // rounds over a full window).
        let sentence = "The quick brown fox jumps over the lazy dog. "
        for maxTokens in (1...7).map({ 9 + $0 }) {
            _ = try await decode(
                "warmup", text: String(repeating: sentence, count: 10), context: context,
                draft: draft, maxNewTokens: maxTokens, runs: 1, emit: { _ in })
        }
        _ = try await decode(
            "warmup", text: String(repeating: sentence, count: 330), context: context,
            draft: draft, maxNewTokens: 320, runs: 1, emit: { _ in })

        var records: [FixtureRecord] = []
        var texts: [String] = []
        for name in fixtures {
            let text = try String(
                contentsOfFile: URL(fileURLWithPath: directory)
                    .appendingPathComponent("\(name).txt").path,
                encoding: .utf8)
            texts.append(text)
            records.append(
                try await decode(
                    name, text: text, context: context, draft: draft,
                    maxNewTokens: maxNewTokens, runs: runs, emit: emit))
        }
        if !fixtures.isEmpty { gpu.append(gpuReference("after-decode")) }

        var prefill: [PrefillRecord] = []
        for file in prefillFiles {
            let text = try String(
                contentsOfFile: URL(fileURLWithPath: directory).appendingPathComponent(file).path,
                encoding: .utf8)
            for run in 0..<prefillRuns {
                let record = try await measurePrefill(
                    file: file, text: text, run: run, context: context, draft: draft)
                prefill.append(record)
                emit(
                    String(
                        format: "[ruler] prefill %@ run%d: %d tokens in %.2f s = %.1f tok/s "
                            + "(split %d, peak %.1f GB)",
                        file, run, record.promptTokens, record.seconds, record.tokensPerSecond,
                        record.split, record.peakGB))
            }
        }
        gpu.append(gpuReference("end"))
        emit(
            "[ruler] gpu reference "
                + gpu.map { String(format: "%@ %.2f", $0.label, $0.teraflops) }
                .joined(separator: ", ") + " TFLOP/s")

        // The references run after every timed run: AR leaves the GPU slower
        // for minutes (a decode run after one measured ~30% slower).
        if check {
            for i in records.indices {
                try await reference(
                    &records[i], text: texts[i], context: context, maxNewTokens: maxNewTokens,
                    emit: emit)
            }
        }

        if let path = option("--bench-json") {
            let report = Report(
                sourceRevision: option("--bench-source-revision"),
                model: context.configuration.name, maxNewTokens: maxNewTokens,
                blockSize: DFlash2Support.blockSize, prefillStepSize: prefillStepSize,
                cooldownSeconds: cooldownSeconds, kvScheme: kvScheme?.rawValue,
                environment: ProcessInfo.processInfo.environment.filter {
                    $0.key.hasPrefix("MLX_") || $0.key.hasPrefix("DFLASH2_")
                },
                thermalStart: thermalStart, thermalEnd: thermalName(), gpuReference: gpu,
                prefill: prefill, fixtures: records)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(report).write(to: URL(fileURLWithPath: path), options: .atomic)
            emit("[ruler] report \(path)")
        }
        if check, records.contains(where: { $0.runs.contains { $0.identity != "MATCH" } }) {
            emit("[ruler] identity DIVERGED on at least one fixture")
        }
    }

    // MARK: Prefill

    /// The speculative split of a cold request with no checkpoints
    /// (`SpeculationPlan`): the driver prefills up to the drafter's context
    /// window, the iterator captures the rest.
    private static func split(promptTokens: Int, draft: any DFlash2DrafterModel) -> Int {
        min(max(0, promptTokens - 1 - draft.contextWindow), promptTokens - 1)
    }

    private static func parameters(maxTokens: Int) -> GenerateParameters {
        var parameters = GenerateParameters(maxTokens: maxTokens)
        parameters.temperature = 0
        parameters.prefill = PrefillParameters(stepSize: prefillStepSize)
        parameters.kvScheme = kvScheme?.rawValue
        return parameters
    }

    private static var kvScheme: KVScheme? {
        option("--bench-kv-scheme").flatMap(KVScheme.init(rawValue:))
    }

    /// The app driver's pipelined prefill of `[0, split)` into `cache`.
    private static func driverPrefill(
        _ prepared: LMInput, split: Int, cache: [KVCache], model: any LanguageModel
    ) throws {
        guard split > 0 else { return }
        let tokens = prepared.text.tokens
        let head = LMInput.Text(
            tokens: tokens.ndim <= 1 ? tokens[..<(split + 1)] : tokens[0..., ..<(split + 1)],
            mask: nil)
        _ = try PrefillExecutor.run(
            model: model, text: head, cache: cache, prefillStepSize: prefillStepSize,
            consumeAll: false, evalPolicy: .pipelined)
    }

    private static func measurePrefill(
        file: String, text: String, run: Int, context: ModelContext,
        draft: any DFlash2DrafterModel
    ) async throws -> PrefillRecord {
        let prepared = try await context.processor.prepare(input: UserInput(chat: [.user(text)]))
        let promptTokens = prepared.text.tokens.dim(-1)
        let split = split(promptTokens: promptTokens, draft: draft)
        // One new token: the iterator samples it and builds no round.
        let parameters = parameters(maxTokens: 1)
        Memory.clearCache()
        let cache = try context.model.newCache(parameters: parameters)
        cooldown()
        Memory.peakMemory = 0
        // Diagnostic window for the mlx fork's MLX_KERNEL_PROFILE probe.
        setenv("MLX_KERNEL_PROFILE_ACTIVE", "prefill", 1)
        let start = ContinuousClock.now
        try driverPrefill(prepared, split: split, cache: cache, model: context.model)
        var iterator = try DFlash2SpeculativeTokenIterator(
            input: LMInput(text: prepared.text), mainModel: context.model, drafter: draft,
            mainCache: cache, prefilledPrefixTokens: split, parameters: parameters,
            blockSize: DFlash2Support.blockSize)
        let seconds = elapsed(since: start)
        unsetenv("MLX_KERNEL_PROFILE_ACTIVE")
        let peak = Double(Memory.peakMemory) / 1e9
        let first = iterator.next() ?? -1
        iterator.finalizeGeneration()
        Stream.gpu.synchronize()
        let digest = stateDigest(iterator.cache) + "-t\(first)"
        Memory.clearCache()
        return PrefillRecord(
            file: file, sha256: sha256(text), promptTokens: promptTokens, split: split,
            run: run, seconds: seconds, tokensPerSecond: Double(promptTokens) / seconds,
            peakGB: peak, stateDigest: digest)
    }

    // MARK: Decode

    private static func decode(
        _ name: String, text: String, context: ModelContext, draft: any DFlash2DrafterModel,
        maxNewTokens: Int, runs: Int, emit: (String) -> Void
    ) async throws -> FixtureRecord {
        let prepared = try await context.processor.prepare(input: UserInput(chat: [.user(text)]))
        let promptTokens = prepared.text.tokens.dim(-1)
        let split = split(promptTokens: promptTokens, draft: draft)
        let parameters = parameters(maxTokens: maxNewTokens)

        // The driver's share is identical for every arm: prefill it once.
        var snapshot: HybridCacheSnapshot?
        if split > 0 {
            let cache = try context.model.newCache(parameters: parameters)
            try driverPrefill(prepared, split: split, cache: cache, model: context.model)
            snapshot = HybridCacheSnapshot.capture(cache: cache, offset: split, type: .leaf)
            guard snapshot != nil else { throw RulerError.snapshotUnsupported }
        }
        Memory.clearCache()
        func freshCache() throws -> [KVCache] {
            try snapshot?.restore() ?? context.model.newCache(parameters: parameters)
        }

        var records: [DecodeRecord] = []
        for run in 0..<runs {
            let cache = try freshCache()
            if name != "warmup" { cooldown() }
            var iterator = try DFlash2SpeculativeTokenIterator(
                input: LMInput(text: prepared.text), mainModel: context.model, drafter: draft,
                mainCache: cache, prefilledPrefixTokens: split, parameters: parameters,
                blockSize: DFlash2Support.blockSize)
            let timeRounds = arguments.contains("--bench-round-timings")
            var roundMilliseconds: [Double] = []
            var roundAccepted: [Int] = []
            var lastProposed = 0
            var lastAccepted = 0
            if name != "warmup" { setenv("MLX_KERNEL_PROFILE_ACTIVE", "spec", 1) }
            let start = ContinuousClock.now
            var lastRound = start
            var stream: [Int] = []
            while let token = iterator.next() {
                stream.append(token)
                if timeRounds, iterator.proposedCount != lastProposed {
                    let now = ContinuousClock.now
                    roundMilliseconds.append(elapsed(since: lastRound) * 1000)
                    roundAccepted.append(iterator.acceptedCount - lastAccepted)
                    lastRound = now
                    lastProposed = iterator.proposedCount
                    lastAccepted = iterator.acceptedCount
                }
            }
            let seconds = elapsed(since: start)
            unsetenv("MLX_KERNEL_PROFILE_ACTIVE")
            let rounds = iterator.speculativeDecodingTelemetry?.roundCount ?? 0
            iterator.finalizeGeneration()
            Stream.gpu.synchronize()
            let record = DecodeRecord(
                run: run, tokens: stream.count, decodeSeconds: seconds,
                tokensPerSecond: Double(stream.count) / seconds,
                accepted: iterator.acceptedCount, proposed: iterator.proposedCount,
                rounds: rounds,
                tokensPerRound: Double(iterator.acceptedCount + rounds) / Double(max(1, rounds)),
                msPerRound: seconds * 1000 / Double(max(1, rounds)), identity: nil,
                stream: stream, roundMilliseconds: timeRounds ? roundMilliseconds : nil,
                roundAccepted: timeRounds ? roundAccepted : nil)
            records.append(record)
            emit(
                String(
                    format: "[ruler] %@ run%d: %.1f tok/s, %d/%d accepted, %.2f tok/round, "
                        + "%.1f ms/round", name, run, record.tokensPerSecond, record.accepted,
                    record.proposed, record.tokensPerRound, record.msPerRound))
            Memory.clearCache()
        }

        return FixtureRecord(
            name: name, sha256: sha256(text), promptTokens: promptTokens, split: split, ar: nil,
            forced: nil, runs: records)
    }

    /// `--bench-check` for one fixture: `TokenIterator` greedy decoding of the
    /// whole prompt, and the same AR teacher-forced along run 0's stream (past
    /// the first tie the free AR stream is another continuation, so the
    /// stream is also checked position by position, on its own prefix).
    private static func reference(
        _ record: inout FixtureRecord, text: String, context: ModelContext, maxNewTokens: Int,
        emit: (String) -> Void
    ) async throws {
        let prepared = try await context.processor.prepare(input: UserInput(chat: [.user(text)]))
        let ar = try autoregressive(prepared, context: context, maxNewTokens: maxNewTokens)
        record.ar = ar
        for i in record.runs.indices {
            record.runs[i].identity = identityLabel(record.runs[i].stream, ar.stream)
        }
        emit(
            String(
                format: "[ruler] %@ ar: %.1f tok/s (%d tokens in %.2f s), identity %@",
                record.name, ar.tokensPerSecond, ar.tokens, ar.decodeSeconds,
                record.runs.map { $0.identity ?? "?" }.joined(separator: ", ")))
        guard let stream = record.runs.first?.stream else { return }
        let forced = try forcedAutoregressive(prepared, stream: stream, context: context)
        record.forced = forced
        emit(
            "[ruler] \(record.name) forced AR: \(forced.departures.count) departure(s)"
                + forced.departures.map {
                    String(
                        format: " +%d (%d vs argmax %d, %.2f ulp)", $0.position, $0.token,
                        $0.argmax, $0.gapUlps)
                }.joined())
    }

    /// Picks the stream's tokens and records where the target's argmax
    /// differs.
    private final class ForcedSampler: LogitSampler {
        let stream: [Int]
        var position = 0
        var departures: [ForcedRecord.Departure] = []

        init(stream: [Int]) { self.stream = stream }

        func sample(logits: MLXArray) -> MLXArray {
            defer { position += 1 }
            guard position < stream.count else { return argMax(logits, axis: -1) }
            let row = logits.reshaped([-1]).asType(.float32)
            let token = stream[position]
            let best = argMax(row).item(Int.self)
            if best != token {
                let top = row[best].item(Float.self)
                let mine = row[token].item(Float.self)
                // A bf16 ulp at |top|: 2^(exponent - 7).
                let ulp = pow(
                    2.0, Double(Int(log2(Double(max(abs(top), 1e-30))).rounded(.down)) - 7))
                departures.append(
                    .init(
                        position: position, token: token, argmax: best, tokenLogit: mine,
                        argmaxLogit: top, gapUlps: Double(top - mine) / ulp))
            }
            return MLXArray([Int32(token)])
        }
    }

    /// AR decoding as ``autoregressive(_:context:maxNewTokens:)`` runs it,
    /// fed `stream` instead of its own choices.
    private static func forcedAutoregressive(
        _ prepared: LMInput, stream: [Int], context: ModelContext
    ) throws -> ForcedRecord {
        var parameters = GenerateParameters(maxTokens: stream.count)
        parameters.temperature = 0
        let sampler = ForcedSampler(stream: stream)
        var iterator = try TokenIterator(
            input: prepared, model: context.model,
            cache: try context.model.newCache(parameters: parameters), processor: nil,
            sampler: sampler, prefill: parameters.prefill, maxTokens: stream.count)
        while iterator.next() != nil {}
        Stream.gpu.synchronize()
        Memory.clearCache()
        return ForcedRecord(departures: sampler.departures)
    }

    /// The reference `--bench-check` has always used: `TokenIterator`
    /// greedy decoding of the whole prompt, prefilled its own way.
    private static func autoregressive(
        _ prepared: LMInput, context: ModelContext, maxNewTokens: Int
    ) throws -> ARRecord {
        var parameters = GenerateParameters(maxTokens: maxNewTokens)
        parameters.temperature = 0
        var iterator = try TokenIterator(
            input: prepared, model: context.model, cache: nil, parameters: parameters)
        var stream: [Int] = []
        let decodeStart = ContinuousClock.now
        while let token = iterator.next() { stream.append(token) }
        let seconds = elapsed(since: decodeStart)
        Memory.clearCache()
        return ARRecord(
            tokens: stream.count, decodeSeconds: seconds,
            tokensPerSecond: Double(stream.count) / seconds, stream: stream)
    }

    // MARK: Lattice sweep

    /// `--bench-lattice DIR`: per fixture, the greedy stream, then the
    /// drafter's whole lattice at every anchor along it, its context taken
    /// from the target teacher-forced over the stream in 8-token blocks, as
    /// verify passes run. Writes `DIR/<fixture>.lattice.json`; drafting
    /// policies (trees, context drafts) replay offline against it.
    static func sweepLattices(
        context: ModelContext, draft: any DFlash2DrafterModel, modelDirectory: URL,
        emit: (String) -> Void
    ) async throws {
        guard let directory = option("--bench-fixture-dir"),
            let outDirectory = option("--bench-lattice")
        else { throw RulerError.missingFixtureDirectory }
        guard let drafter = draft as? DFlash2DraftModel,
            let target = context.model as? any DFlash2TargetModel
        else { throw RulerError.unsupportedModels }
        let maxNewTokens = positive("--bench-max-tokens", default: 512)
        let fixtures = list("--bench-fixtures", default: ["travel", "summary", "math", "code"])
        let blockSize = DFlash2Support.blockSize
        // `--bench-lattice-mtp`: also the target's MTP head's greedy guess
        // for each anchor's first draft position.
        let mtp: (any StatefulMTPDrafterModel)? =
            arguments.contains("--bench-lattice-mtp")
            ? try await MTPDrafterSupport.loadDrafter(directory: modelDirectory, pairing: .text)
                .model as? any StatefulMTPDrafterModel
            : nil
        for name in fixtures {
            let text = try String(
                contentsOfFile: URL(fileURLWithPath: directory)
                    .appendingPathComponent("\(name).txt").path,
                encoding: .utf8)
            let prepared = try await context.processor.prepare(
                input: UserInput(chat: [.user(text)]))
            let promptTokens = prepared.text.tokens.dim(-1)
            // The DFlash2 stream by default: every verify row runs the same
            // M = 8 arithmetic wherever its block starts, so any lossless
            // drafting policy reproduces it; `ar` replays the AR stream.
            let stream =
                option("--bench-lattice-stream") == "ar"
                ? try autoregressive(prepared, context: context, maxNewTokens: maxNewTokens)
                    .stream
                : try await decode(
                    name, text: text, context: context, draft: draft,
                    maxNewTokens: maxNewTokens, runs: 1, emit: { _ in }
                ).runs[0].stream

            // Production prefill, capturing the tail's rows for the drafter.
            let cache = try context.model.newCache(parameters: parameters(maxTokens: 1))
            let split = split(promptTokens: promptTokens, draft: draft)
            try driverPrefill(prepared, split: split, cache: cache, model: context.model)
            let prompt = prepared.text.tokens.reshaped(-1)
            var rows: [MLXArray] = []
            var start = split
            while start < promptTokens {
                let remaining = promptTokens - start
                let end = start + (remaining == 1 ? 1 : min(prefillStepSize, remaining - 1))
                let result = target.dflash2Prefill(
                    prompt[start..<end].expandedDimensions(axis: 0), cache: cache,
                    captureLayers: draft.targetLayerIds, positionDelta: 0)
                rows.append(concatenated(result.hidden, axis: -1))
                eval(cache.flatMap { $0.innerState() } + [rows.last!])
                start = end
            }
            var promptHidden = concatenated(rows, axis: 1)
            let keep = min(draft.contextWindow, promptHidden.dim(1))
            promptHidden = promptHidden[0..., (promptHidden.dim(1) - keep)..., 0...]
            eval(promptHidden)

            // The stream teacher-forced in verify-sized blocks: row i is
            // the target at position promptTokens + i.
            var streamRows: [MLXArray] = []
            var j = 0
            while j < stream.count - 1 {
                let end = min(j + blockSize, stream.count - 1)
                let tokens = MLXArray(stream[j..<end].map { Int32($0) }).reshaped(1, -1)
                let result = target.dflash2Prefill(
                    tokens, cache: cache, captureLayers: draft.targetLayerIds, positionDelta: 0)
                streamRows.append(concatenated(result.hidden, axis: -1))
                eval(cache.flatMap { $0.innerState() } + [streamRows.last!])
                j = end
            }
            let streamHidden = concatenated(streamRows, axis: 1)
            eval(streamHidden)

            var state = drafter.makeState()
            var anchors: [[String: Any]] = []
            let drafts = blockSize - 1
            for j in 0..<(stream.count - drafts) {
                let newRows =
                    j == 0 ? promptHidden : streamHidden[0..., (j - 1)..<j, 0...]
                let contextPosition =
                    j == 0 ? promptTokens - promptHidden.dim(1) : promptTokens + j - 1
                let block = MLXArray(
                    [Int32(stream[j])]
                        + Array(repeating: Int32(drafter.maskTokenId), count: drafts)
                ).reshaped(1, blockSize)
                let lattice = drafter.proposeLattice(
                    block: block, targetHidden: newRows, contextPosition: contextPosition,
                    validRows: MLXArray(Int32(newRows.dim(1))), target: target, state: &state)
                eval(
                    lattice.candidates, lattice.unary, lattice.edges, lattice.anchorEdges,
                    lattice.tokens)
                for contextCache in state.contextCaches {
                    contextCache.resolve(newest: newRows.dim(1), valid: newRows.dim(1))
                }
                func rounded(_ a: MLXArray) -> [Double] {
                    a.asType(.float32).asArray(Float.self).map {
                        (Double($0) * 1e4).rounded() / 1e4
                    }
                }
                anchors.append([
                    "j": j,
                    "candidates": lattice.candidates.asType(.int32).asArray(Int32.self).map(
                        Int.init),
                    "unary": rounded(lattice.unary),
                    "edges": rounded(lattice.edges),
                    "anchorEdges": rounded(lattice.anchorEdges),
                    "greedy": lattice.tokens.asType(.int32).asArray(Int32.self).map(Int.init),
                ])
            }
            var record: [String: Any] = [
                "name": name, "promptTokens": promptTokens, "stream": stream,
                "promptIds": prompt.asArray(Int32.self).map(Int.init), "anchors": anchors,
            ]
            if let mtp {
                record["mtpTop1"] = try mtpGuesses(
                    prepared, stream: stream, anchors: anchors.count, context: context, mtp: mtp)
            }
            let url = URL(fileURLWithPath: outDirectory)
                .appendingPathComponent("\(name).lattice.json")
            try JSONSerialization.data(withJSONObject: record).write(to: url, options: .atomic)
            emit("[ruler] lattice \(name): \(anchors.count) anchors -> \(url.path)")
            Memory.clearCache()
        }
    }

    /// The MTP head's greedy guess for position `j + 1` at every anchor
    /// `j`, from the target's normed hidden states teacher-forced over the
    /// prompt and the stream (the pairs the MTP iterator commits).
    private static func mtpGuesses(
        _ prepared: LMInput, stream: [Int], anchors: Int, context: ModelContext,
        mtp: any StatefulMTPDrafterModel
    ) throws -> [Int] {
        let cache = try context.model.newCache(parameters: parameters(maxTokens: 1))
        var emitState = LMOutput.State()
        emitState[mtpEmitFlagKey] = true
        func hidden(_ tokens: MLXArray) -> MLXArray {
            let output = context.model(
                LMInput.Text(tokens: tokens.expandedDimensions(axis: 0)), cache: cache,
                state: emitState)
            let rows = output.state![mtpLastHiddenStatesKey]!
            eval(cache.flatMap { $0.innerState() } + [rows])
            return rows
        }
        let prompt = prepared.text.tokens.reshaped(-1)
        var promptRows: [MLXArray] = []
        var start = 0
        while start < prompt.dim(0) {
            let end = min(start + prefillStepSize, prompt.dim(0))
            promptRows.append(hidden(prompt[start..<end]))
            start = end
        }
        var streamRows: [MLXArray] = []
        var j = 0
        while j < anchors {
            let end = min(j + DFlash2Support.blockSize, anchors)
            streamRows.append(hidden(MLXArray(stream[j..<end].map { Int32($0) })))
            j = end
        }
        let streamHidden = concatenated(streamRows, axis: 1)
        var state = mtp.makeState(parameters: nil)
        let sampler = ArgMaxSampler()
        mtp.prepareDrafterState(
            target: context.model, promptTokens: prompt.expandedDimensions(axis: 0),
            targetHidden: concatenated(promptRows, axis: 1),
            firstBonus: MLXArray([Int32(stream[0])]), positionDeltas: nil, state: &state,
            sampler: sampler)
        var guesses = [state.seedToken!.item(Int.self)]
        for j in 1..<anchors {
            mtp.commitDrafterState(
                target: context.model, targetHidden: streamHidden[0..., (j - 1)..<j, 0...],
                draftTokens: MLXArray.zeros([1, 1], dtype: .int32), acceptedCount: 0,
                finalToken: MLXArray([Int32(stream[j])]), positionDeltas: nil, state: &state,
                sampler: sampler)
            guesses.append(state.seedToken!.item(Int.self))
        }
        Memory.clearCache()
        return guesses
    }

    // MARK: Helpers

    /// Order-free checksums of the cache's bits (sum and sum of squares of
    /// its 16-bit words, mod 2^32): any differing bit almost surely moves one.
    private static func stateDigest(_ cache: [KVCache]) -> String {
        var sum: UInt64 = 0
        var squares: UInt64 = 0
        for array in cache.flatMap({ $0.state }) {
            let words = contiguous(array).flattened().view(dtype: .uint16).asType(.uint32)
            sum &+= UInt64(words.sum().item(UInt32.self))
            squares &+= UInt64((words * words).sum().item(UInt32.self))
        }
        return String(
            format: "%08x%08x", UInt32(truncatingIfNeeded: sum),
            UInt32(truncatingIfNeeded: squares))
    }

    /// A fixed bf16 GEMM, for GPU clock drift between and within runs.
    private static func gpuReference(_ label: String) -> GPUReference {
        MLXRandom.seed(3)
        let a = MLXRandom.normal([4096, 4096], dtype: .bfloat16)
        let b = MLXRandom.normal([4096, 4096], dtype: .bfloat16)
        eval(a, b)
        for _ in 0..<3 { eval(matmul(a, b)) }
        let iterations = 20
        let start = ContinuousClock.now
        for _ in 0..<iterations { eval(matmul(a, b)) }
        let seconds = elapsed(since: start)
        Memory.clearCache()
        return GPUReference(
            label: label, teraflops: 2 * pow(4096.0, 3) * Double(iterations) / seconds / 1e12)
    }

    private static func identityLabel(_ stream: [Int], _ reference: [Int]) -> String {
        if stream == reference { return "MATCH" }
        let shared = min(stream.count, reference.count)
        let first = (0..<shared).first { stream[$0] != reference[$0] } ?? shared
        return "DIVERGED at +\(first)"
    }

    private static func thermalName() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: "nominal"
        case .fair: "fair"
        case .serious: "serious"
        case .critical: "critical"
        @unknown default: "unknown"
        }
    }

    private static func sha256(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    private static func elapsed(since start: ContinuousClock.Instant) -> Double {
        let elapsed = ContinuousClock.now - start
        return Double(elapsed.components.seconds)
            + Double(elapsed.components.attoseconds) / 1e18
    }

    enum RulerError: Error {
        case missingFixtureDirectory
        case snapshotUnsupported
        case unsupportedModels
    }
}
