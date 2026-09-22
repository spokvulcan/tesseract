import CryptoKit
import Foundation
import MLX
import MLXLMCommon

/// Owner-run #528 parity gate: the three-arm loaded campaign the
/// [pre-registration](../../../../benchmarks/warm-body-parity/2026-09-19/README.md)
/// defines. Every arm drives the same in-process **Server Completion** path
/// the `prefix-cache-e2e` gate uses; the resident form of the hit body is
/// selected through `PrefixCacheAdmin`'s measurement overrides (Warm Body
/// compression at 8 or 4 bits, or **Leaf Checkout** switched off so the fp16
/// control restores by copy). The dequantization allowance is timed at the
/// Model Session seam on the actual checked-in body. The runner decides
/// nothing: it records observations and the per-case verdicts the
/// pre-registration's rules produce; the default flag stays where it is.
@MainActor
final class WarmBodyParityBenchRunner {
    private let runner: BenchmarkRunner

    init(runner: BenchmarkRunner) { self.runner = runner }

    // MARK: - Manifest

    nonisolated struct Manifest: Codable, Sendable {
        struct Host: Codable, Sendable {
            let identifier: String
            let ramBytes: Int
            let osVersion: String
            let mlxVersion: String
        }
        struct Model: Codable, Sendable {
            let id: String
            let directory: String
            let configSHA256: String
            let tokenizerSHA256: String
            let templateSHA256: String
        }
        struct Corpus: Codable, Sendable {
            let fillerFile: String
            let fillerSHA256: String
            let orderedCaseIDs: [String]
            let unsupportedCases: [String]
        }
        struct Settings: Codable, Sendable {
            let liveKVDtype: String
            let warmBits: [Int]
            let groupSize: Int
            let mode: String
            let temperature: Double
            let speculationPolicy: String
            let seed: Int?
            let maximumPromptTokens: Int
            let maximumOutputTokens: Int
            let timedOutputTokens: Int
        }
        struct Resources: Codable, Sendable {
            let minimumInitialAvailableBytes: Int
            let expectedModelAndCachePeakBytes: Int
            let footprintStopBytes: Int
            let minimumAvailableStopBytes: Int
            let swapGrowthStopBytes: Int
            let pressureStopLevel: Int
            let maximumRequestSeconds: Int
            let maximumCampaignSeconds: Int
            let sampleIntervalMilliseconds: Int
            let cancelGraceSeconds: Int
        }
        struct Case: Codable, Sendable {
            enum Kind: String, Codable, Sendable {
                /// Direct leaf continuation: the hit restores the whole leaf.
                case directLeaf
                /// A fork off a planned branch point: the hit restores a
                /// Prefix-View Checkpoint through its Backing Leaf.
                case plannedView
                /// A think-stripping template's stop turn: the timed hit is
                /// a direct leaf hit on the canonical leaf; the transient
                /// boundary view resolves against the arm's form at the end
                /// of the timed turn.
                case thinkStrippingBoundary
            }
            let id: String
            let kind: Kind
            let prefixTokens: Int
            let setupOutputTokens: Int
            let reasoningEffort: String?
            let description: String
        }

        let status: String
        let ownerApproval: String
        let preRegistrationCommit: String
        let appCommit: String
        let vendorCommit: String
        let instrumentationCommit: String
        let releaseBinarySHA256: String
        let host: Host
        let model: Model
        let corpus: Corpus
        let settings: Settings
        let resources: Resources
        let cases: [Case]
        let orders: [[String]]
        let privateOutputDirectory: String
    }

    nonisolated enum Arm: String, Codable, CaseIterable, Sendable {
        case fp16
        case warm8 = "8"
        case warm4 = "4"

        var bits: Int? {
            switch self {
            case .fp16: nil
            case .warm8: 8
            case .warm4: 4
            }
        }
    }

    // MARK: - Records

    /// One timed hit. Written to `observations.jsonl` the moment it exists.
    nonisolated struct Observation: Codable, Sendable {
        let caseID: String
        let block: Int
        let arm: Arm
        let armIndexInBlock: Int
        /// The setup turns' restore facts, for the invalidation rules.
        let setupLookups: [[String: String]]
        let warmCompressEvents: [[String: String]]
        let residentFormBeforeHit: String
        let residentBitsBeforeHit: Int?
        let residentBodyBytesBeforeHit: Int
        let residentLeafOffsetBeforeHit: Int
        /// The timed hit.
        let ttftSeconds: Double
        let hitLookups: [[String: String]]
        let hitLeafStore: [String: String]?
        let serverLookupMs: Double
        let serverRestoreMs: Double
        let serverPrefillMs: Double
        let cachedTokenCount: Int
        let promptTokenCount: Int
        let generationTokenCount: Int
        let generatedTextSHA256: String
        let generatedText: String
        /// The token ids the live turn fed past the prompt (nil where the
        /// leaf path is a canonical re-render rather than the fed ids).
        let generatedTokenIDs: [Int]?
        let mlxActiveBeforeHit: Int
        let mlxActiveAfterHit: Int
        let mlxPeakDuringHit: Int
        let mlxCacheAfterHit: Int
        let footprintBeforeHit: Int?
        let footprintAfterHit: Int?
        let requestMemoryPeakFootprint: Int?
        let requestMemoryPeakActiveMlx: Int?
        let followUpLookups: [[String: String]]
        let invalid: [String]
    }

    nonisolated struct DequantizeMeasurement: Codable, Sendable {
        let caseID: String
        let block: Int
        let arm: Arm
        let bits: Int
        let prefixOffset: Int
        let layers: Int
        let quantizedBytes: Int
        let dequantizedBytes: Int
        let warmupSeconds: Double
        let seconds: [Double]
        let observerOverheadSeconds: [Double]
        let mlxPeakBytes: Int
    }

    nonisolated struct CopyMeasurement: Codable, Sendable {
        let caseID: String
        let block: Int
        let prefixOffset: Int
        let bytes: Int
        let warmupSeconds: Double
        let seconds: [Double]
    }

    nonisolated struct FidelitySession: Codable, Sendable {
        let caseID: String
        let block: Int
        let arm: Arm
        let boundaries: Int
        let boundaryKinds: [String]
        let mismatches: Int
        let skipped: [String]
        let report: String
    }

    nonisolated final class EventBuffer: @unchecked Sendable {
        private let lock = NSLock()
        private var events: [PromptCacheTelemetryEvent] = []
        func append(_ event: PromptCacheTelemetryEvent) {
            lock.lock(); defer { lock.unlock() }
            events.append(event)
        }
        func take() -> [PromptCacheTelemetryEvent] {
            lock.lock(); defer { lock.unlock() }
            defer { events.removeAll() }
            return events
        }
    }

    nonisolated private struct TurnResult {
        let ttftSeconds: Double
        let text: String
        let reasoning: String
        let promptTokenCount: Int
        let generationTokenCount: Int
        let cachedTokenCount: Int
        let diagnostics: HTTPServerGenerationStart.Diagnostics
        let events: [PromptCacheTelemetryEvent]
        let mlxPeak: Int
        let mlxActiveAfter: Int
        let mlxCacheAfter: Int
        let footprintAfter: Int?
    }

    nonisolated private struct CaseCorpus {
        let systemPrompt: String
        let setupUser: String
        let forkUsers: [String]
        let timedUser: String
        let followUpUser: String
        let noiseSystem: String
        let noiseUser: String
    }

    private var manifest: Manifest!
    private var reportDir: URL!
    private var privateDir: URL!
    private let events = EventBuffer()
    private var observations: [Observation] = []
    private var dequantizations: [DequantizeMeasurement] = []
    private var copies: [CopyMeasurement] = []
    private var fidelity: [FidelitySession] = []
    private var invalidations: [String] = []
    private var viewOffsets: [String: Int] = [:]

    // MARK: - Entry

    func run() async throws {
        let args = CommandLine.arguments
        guard let flag = args.firstIndex(of: "--warm-parity-plan"), flag + 1 < args.count else {
            throw failure("Pass --warm-parity-plan with the owner-approved manifest")
        }
        let planData = try Data(contentsOf: URL(fileURLWithPath: args[flag + 1]))
        let manifest = try JSONDecoder().decode(Manifest.self, from: planData)
        try Self.validate(manifest)
        self.manifest = manifest

        let modelDir = try runner.resolveModelDirectory()
        try Self.verifyModelFiles(manifest.model, modelDir: modelDir)
        let filler = try Self.loadFiller(manifest.corpus)

        reportDir = runner.activeConfig.outputDir
            .appendingPathComponent("warm-parity-\(UUID().uuidString)", isDirectory: true)
        privateDir = URL(fileURLWithPath: manifest.privateOutputDirectory, isDirectory: true)
            .appendingPathComponent(reportDir.lastPathComponent, isDirectory: true)
        try FileManager.default.createDirectory(at: reportDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: privateDir, withIntermediateDirectories: true)
        try planData.write(to: reportDir.appendingPathComponent("approved-plan.json"))
        log("warm-parity report dir: \(reportDir.path)")
        log("private dir: \(privateDir.path)")

        let sink = PrefixCacheDiagnostics.addTelemetrySink { [events] event in
            events.append(event)
        }
        defer { PrefixCacheDiagnostics.removeTelemetrySink(sink) }

        let engine = AgentEngine(speculation: .off)
        log("loading \(modelDir.path)")
        try await engine.loadModel(from: modelDir, visionMode: false)
        log("model loaded")
        let tokenizer = try await AppTokenizerLoader().load(from: modelDir)
        // The Server Completion module and its cache manager are created on
        // the first request; the admin overrides are no-ops until then.
        let priming = try await runTurn(
            [
                OpenAI.ChatMessage(role: .system, content: .text("You are a terse assistant.")),
                OpenAI.ChatMessage(
                    role: .user, content: .text("Reply with the single word ready.")),
            ],
            parameters: AgentGenerateParameters(
                maxTokens: 4, temperature: 0, topP: 1, topK: 1, minP: 0),
            renderContext: .canonical, engine: engine, modelID: runner.activeConfig.resolvedModelID,
            deadline: manifest.resources.maximumRequestSeconds)
        guard engine.llmActor.prefixCacheAdmin.stats != nil else {
            throw failure("no live prefix cache after the priming request")
        }
        log("primed: \(priming.generationTokenCount) tokens, cache live")
        _ = events.take()

        do {
            for caseSpec in manifest.cases {
                let corpus = Self.makeCorpus(
                    for: caseSpec, filler: filler, tokenizer: tokenizer)
                try await runCase(caseSpec, corpus: corpus, engine: engine, tokenizer: tokenizer)
            }
            try writeReport()
        } catch {
            invalidations.append("campaign error: \(error)")
            try? writeReport()
            engine.unloadModel()
            await engine.awaitPendingUnload()
            throw error
        }
        engine.unloadModel()
        await engine.awaitPendingUnload()
    }

    // MARK: - Validation

    nonisolated private static func validate(_ manifest: Manifest) throws {
        guard manifest.status == "APPROVED" else {
            throw failure("Manifest status must be APPROVED (is \(manifest.status))")
        }
        for (name, value) in [
            ("ownerApproval", manifest.ownerApproval),
            ("preRegistrationCommit", manifest.preRegistrationCommit),
            ("appCommit", manifest.appCommit), ("vendorCommit", manifest.vendorCommit),
            ("instrumentationCommit", manifest.instrumentationCommit),
            ("releaseBinarySHA256", manifest.releaseBinarySHA256),
            ("privateOutputDirectory", manifest.privateOutputDirectory),
        ] where value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            throw failure("Manifest field \(name) is empty: not executable")
        }
        let resources = manifest.resources
        for (name, value) in [
            ("minimumInitialAvailableBytes", resources.minimumInitialAvailableBytes),
            ("expectedModelAndCachePeakBytes", resources.expectedModelAndCachePeakBytes),
            ("footprintStopBytes", resources.footprintStopBytes),
            ("minimumAvailableStopBytes", resources.minimumAvailableStopBytes),
            ("swapGrowthStopBytes", resources.swapGrowthStopBytes),
            ("pressureStopLevel", resources.pressureStopLevel),
            ("maximumRequestSeconds", resources.maximumRequestSeconds),
            ("maximumCampaignSeconds", resources.maximumCampaignSeconds),
            ("sampleIntervalMilliseconds", resources.sampleIntervalMilliseconds),
            ("cancelGraceSeconds", resources.cancelGraceSeconds),
            ("maximumPromptTokens", manifest.settings.maximumPromptTokens),
            ("maximumOutputTokens", manifest.settings.maximumOutputTokens),
            ("timedOutputTokens", manifest.settings.timedOutputTokens),
        ] where value <= 0 {
            throw failure("Manifest resource \(name) must be positive: not executable")
        }
        guard manifest.settings.temperature == 0, manifest.settings.liveKVDtype == "float16",
            manifest.settings.groupSize == HybridCacheSnapshot.warmCompressionGroupSize,
            manifest.settings.mode == "affine", manifest.settings.speculationPolicy == "off",
            Set(manifest.settings.warmBits) == [8, 4]
        else { throw failure("Manifest settings differ from the pre-registration") }
        let expectedOrders = [
            ["fp16", "8", "4"], ["fp16", "4", "8"], ["8", "fp16", "4"],
            ["8", "4", "fp16"], ["4", "fp16", "8"], ["4", "8", "fp16"],
        ]
        guard manifest.orders == expectedOrders else {
            throw failure("Manifest orders differ from the pre-registered six")
        }
        guard !manifest.cases.isEmpty,
            manifest.cases.map(\.id) == manifest.corpus.orderedCaseIDs,
            Set(manifest.cases.map(\.id)).count == manifest.cases.count
        else { throw failure("Manifest cases must be nonempty and match orderedCaseIDs") }
        for spec in manifest.cases {
            guard spec.prefixTokens > 0, spec.prefixTokens <= manifest.settings.maximumPromptTokens,
                spec.setupOutputTokens > 0,
                spec.setupOutputTokens <= manifest.settings.maximumOutputTokens
            else { throw failure("Case \(spec.id) exceeds the manifest's caps") }
        }
    }

    nonisolated private static func verifyModelFiles(_ model: Manifest.Model, modelDir: URL) throws
    {
        guard
            modelDir.standardizedFileURL.path
                == URL(fileURLWithPath: model.directory)
                .standardizedFileURL.path
        else { throw failure("Model directory differs from the manifest") }
        for (name, expected) in [
            ("config.json", model.configSHA256), ("tokenizer.json", model.tokenizerSHA256),
            ("chat_template.jinja", model.templateSHA256),
        ] {
            let digest = try fileSHA256(modelDir.appendingPathComponent(name))
            guard digest == expected else {
                throw failure("\(name) SHA-256 differs from the manifest")
            }
        }
    }

    nonisolated private static func loadFiller(_ corpus: Manifest.Corpus) throws -> String {
        let url = URL(fileURLWithPath: corpus.fillerFile)
        guard try fileSHA256(url) == corpus.fillerSHA256 else {
            throw failure("Filler corpus SHA-256 differs from the manifest")
        }
        return try String(contentsOf: url, encoding: .utf8)
    }

    // MARK: - Corpus

    /// Deterministic prompts sized by the real tokenizer: the system prompt
    /// carries the case's prefix so the leaf the timed hit restores has the
    /// registered size. Same bytes for every arm and block of a case.
    nonisolated private static func makeCorpus(
        for spec: Manifest.Case, filler: String, tokenizer: any Tokenizer
    ) -> CaseCorpus {
        let header = """
            You are a careful, methodical assistant answering questions about the \
            design document below. Answer briefly and precisely. Case \(spec.id).

            """
        let headerTokens = tokenizer.encode(text: header, addSpecialTokens: false).count
        var body = filler
        while tokenizer.encode(text: body, addSpecialTokens: false).count
            < spec.prefixTokens - headerTokens
        {
            body += "\n\n" + filler
        }
        let ids = tokenizer.encode(text: body, addSpecialTokens: false)
        let trimmed = tokenizer.decode(
            tokenIds: Array(ids.prefix(max(1, spec.prefixTokens - headerTokens))),
            skipSpecialTokens: false)
        let sharedUser = "Summarize the three most important terms defined in the document, "
        return CaseCorpus(
            systemPrompt: header + trimmed,
            setupUser: sharedUser + "one sentence each.",
            forkUsers: [sharedUser + "alpha order.", sharedUser + "beta order."],
            timedUser: "Now name one open risk in that design and explain it in two sentences.",
            followUpUser: "Thanks.",
            noiseSystem: "You are a terse assistant.",
            noiseUser: "Reply with the single word ready.")
    }

    // MARK: - Campaign

    private func runCase(
        _ spec: Manifest.Case, corpus: CaseCorpus, engine: AgentEngine, tokenizer: any Tokenizer
    ) async throws {
        log("── case \(spec.id) (\(spec.kind.rawValue), prefix \(spec.prefixTokens))")
        for block in -1..<manifest.orders.count {
            let order = block < 0 ? ["fp16", "8", "4"] : manifest.orders[block]
            for (index, raw) in order.enumerated() {
                guard let arm = Arm(rawValue: raw) else { throw failure("Unknown arm \(raw)") }
                let observation = try await observe(
                    spec, corpus: corpus, block: block, arm: arm, index: index, engine: engine,
                    tokenizer: tokenizer)
                if block >= 0 { observations.append(observation) }
                try append(
                    observation, to: block < 0 ? "warmup-observations.jsonl" : "observations.jsonl")
                log(
                    "  block \(block) \(arm.rawValue): ttft=\(String(format: "%.1f", observation.ttftSeconds * 1000))ms "
                        + "restore=\(observation.serverRestoreMs)ms form=\(observation.residentFormBeforeHit) "
                        + "tokens=\(observation.generationTokenCount) invalid=\(observation.invalid)"
                )
                if !observation.invalid.isEmpty {
                    invalidations.append(
                        "\(spec.id) block \(block) \(arm.rawValue): \(observation.invalid.joined(separator: "; "))"
                    )
                }
            }
        }
    }

    // swiftlint:disable:next function_body_length
    private func observe(
        _ spec: Manifest.Case, corpus: CaseCorpus, block: Int, arm: Arm, index: Int,
        engine: AgentEngine, tokenizer: any Tokenizer
    ) async throws -> Observation {
        let admin = engine.llmActor.prefixCacheAdmin
        var invalid: [String] = []
        let sessionAffinity = "\(spec.id)-b\(block)-\(arm.rawValue)"

        // Reset: every resident body goes; the request path stays warm.
        _ = admin.clearRAMTier()
        await admin.awaitPendingDrain()
        Memory.clearCache()
        try await Task.sleep(for: .milliseconds(500))
        if let stats = admin.stats, stats.totalSnapshotBytes != 0 {
            invalid.append("RAM tier not empty after clear: \(stats.totalSnapshotBytes) bytes")
        }

        // Arm form.
        switch arm {
        case .fp16:
            admin.setWarmCompression(
                enabled: false, bits: 8, hotLeafPathLimit: 0, opportunisticFraction: 0)
            admin.setLeafCheckoutDisabled(true)
        case .warm8, .warm4:
            admin.setWarmCompression(
                enabled: true, bits: arm.bits!, hotLeafPathLimit: 0, opportunisticFraction: 0)
            admin.setLeafCheckoutDisabled(false)
        }
        _ = events.take()

        let renderContext = try await makeRenderContext(spec, engine: engine)
        var params = AgentGenerateParameters(
            maxTokens: spec.setupOutputTokens, temperature: 0, topP: 1, topK: 1, minP: 0)
        params.reasoningEffort = renderContext.reasoningEffort
        let modelID = runner.activeConfig.resolvedModelID

        // Setup turns: build the conversation whose leaf the timed hit restores.
        var recorded: [CanonicalEchoFidelity.RecordedRequest] = []
        var recordings: [OpenAI.ChatCompletionRequest] = []
        var history: [OpenAI.ChatMessage] = [
            OpenAI.ChatMessage(role: .system, content: .text(corpus.systemPrompt))
        ]
        var setupLookups: [[String: String]] = []
        var warmCompressEvents: [[String: String]] = []
        func noiseTurn() async throws {
            let turn = try await noise(corpus, engine: engine, modelID: modelID, params: params)
            warmCompressEvents.append(
                contentsOf: turn.events.filter { $0.eventName == "warmCompress" }.map(Self.fieldMap)
            )
        }
        func send(
            _ messages: [OpenAI.ChatMessage], maxTokens: Int, affinity: String, record: Bool
        ) async throws -> TurnResult {
            var turnParams = params
            turnParams.maxTokens = maxTokens
            let request = OpenAI.ChatCompletionRequest(
                model: modelID, messages: messages, max_tokens: maxTokens, temperature: 0,
                chat_template_kwargs: renderContext.kwargs.isEmpty
                    && renderContext.reasoningEffort == nil
                    ? nil
                    : OpenAI.ChatTemplateKwargs(
                        booleanFlags: Dictionary(
                            uniqueKeysWithValues: renderContext.kwargs.map {
                                ($0.key.rawValue, $0.value)
                            }),
                        stringValues: renderContext.reasoningEffort.map {
                            [TemplateRenderContext.reasoningEffortKwargName: $0.rawValue]
                        } ?? [:]))
            if record {
                recordings.append(request)
                recorded.append(
                    CanonicalEchoFidelity.RecordedRequest(
                        messages: messages, tools: nil, renderContext: renderContext))
            }
            return try await runTurn(
                messages, parameters: turnParams, renderContext: renderContext, engine: engine,
                modelID: modelID, deadline: manifest.resources.maximumRequestSeconds)
        }
        func assistantMessage(_ turn: TurnResult) -> OpenAI.ChatMessage {
            OpenAI.ChatMessage(
                role: .assistant, content: .text(turn.text),
                reasoning_content: turn.reasoning.isEmpty ? nil : turn.reasoning)
        }
        func expectCold(_ turn: TurnResult, _ label: String) {
            let lookups = turn.events.filter { $0.eventName == "lookup" }
            setupLookups.append(contentsOf: lookups.map(Self.fieldMap))
            guard let first = lookups.first else {
                invalid.append("\(label): no lookup event")
                return
            }
            if first.field("restoreMode") != "cold" || turn.cachedTokenCount != 0 {
                invalid.append(
                    "\(label): expected a cold miss, got \(first.field("reason") ?? "?") "
                        + "restoreMode=\(first.field("restoreMode") ?? "?") cached=\(turn.cachedTokenCount)"
                )
            }
        }
        func settleCompression(_ label: String) async {
            await admin.awaitPendingDrain()
        }

        switch spec.kind {
        case .directLeaf, .thinkStrippingBoundary:
            history.append(OpenAI.ChatMessage(role: .user, content: .text(corpus.setupUser)))
            let turn = try await send(
                history, maxTokens: spec.setupOutputTokens, affinity: sessionAffinity, record: true)
            expectCold(turn, "T1")
            history.append(assistantMessage(turn))
        case .plannedView:
            for (forkIndex, forkUser) in corpus.forkUsers.enumerated() {
                let messages = history + [OpenAI.ChatMessage(role: .user, content: .text(forkUser))]
                let turn = try await send(
                    messages, maxTokens: spec.setupOutputTokens, affinity: sessionAffinity,
                    record: false)
                if forkIndex == 0 {
                    expectCold(turn, "T1a")
                } else {
                    setupLookups.append(
                        contentsOf: turn.events.filter { $0.eventName == "lookup" }.map(
                            Self.fieldMap))
                }
                try await noiseTurn()
                await settleCompression("fork \(forkIndex)")
                warmCompressEvents.append(
                    contentsOf: events.take().filter { $0.eventName == "warmCompress" }.map(
                        Self.fieldMap))
            }
            history.append(OpenAI.ChatMessage(role: .user, content: .text(corpus.setupUser)))
        }
        // A short unrelated turn makes the conversation's leaf older than the
        // Budget Floor's freshest leaf, so the opportunistic pass may
        // compress it (the floor's most-recently-extended leaf is exempt).
        try await noiseTurn()
        await settleCompression("setup")
        warmCompressEvents.append(
            contentsOf: events.take().filter { $0.eventName == "warmCompress" }.map(Self.fieldMap))

        // The resident form the hit will restore: the case conversation's
        // leaf, told apart from the noise conversation's by its size.
        guard let freshest = admin.freshestLeaf(minimumOffset: spec.prefixTokens) else {
            throw failure("no resident leaf after setup")
        }
        let leafForm =
            freshest.body.isWarm ? "warm" : (freshest.body.isPrefixView ? "view" : "ownedBody")
        switch arm {
        case .fp16:
            if freshest.body.isWarm { invalid.append("control leaf is warm") }
        case .warm8, .warm4:
            if !freshest.body.isWarm || freshest.body.warmBits != arm.bits {
                invalid.append(
                    "warm leaf form=\(leafForm) bits=\(freshest.body.warmBits.map(String.init) ?? "nil"), expected \(arm.bits!)"
                )
            }
            if !warmCompressEvents.contains(where: { $0["bits"] == "\(arm.bits!)" }) {
                invalid.append("no warmCompress event at \(arm.bits!) bits")
            }
        }

        // Dequantization allowance (warm arms) and the control's copy timing,
        // at the Model Session seam, on the body the hit restores; released
        // before the timed observation.
        let prefixOffset =
            spec.kind == .plannedView
            ? (viewOffsets[spec.id] ?? freshest.body.tokenOffset) : freshest.body.tokenOffset
        if let bits = arm.bits, freshest.body.isWarm {
            let measurement = try await engine.llmActor.withModelContainer {
                [body = freshest.body] container in
                try await container.perform { _ in
                    try Self.measureDequantize(
                        body, offset: prefixOffset, bits: bits, caseID: spec.id, block: block,
                        arm: arm)
                }
            }
            if block >= 0 { dequantizations.append(measurement) }
            try append(measurement, to: block < 0 ? "warmup-dequantize.jsonl" : "dequantize.jsonl")
        } else if arm == .fp16, !freshest.body.isWarm {
            let measurement = try await engine.llmActor.withModelContainer {
                [body = freshest.body] container in
                try await container.perform { _ in
                    Self.measureCopy(body, offset: prefixOffset, caseID: spec.id, block: block)
                }
            }
            if block >= 0 { copies.append(measurement) }
            try append(measurement, to: block < 0 ? "warmup-copy.jsonl" : "copy.jsonl")
        }
        Memory.clearCache()
        try await Task.sleep(for: .seconds(1))
        _ = events.take()

        // The timed hit.
        let before = RequestMemoryTelemetry.Sample.current()
        history.append(OpenAI.ChatMessage(role: .user, content: .text(corpus.timedUser)))
        let hit = try await send(
            history, maxTokens: manifest.settings.timedOutputTokens, affinity: sessionAffinity,
            record: true)
        history.append(assistantMessage(hit))
        let hitLookups = hit.events.filter { $0.eventName == "lookup" }.map(Self.fieldMap)
        let leafStore = hit.events.first { $0.eventName == "leafStore" }.map(Self.fieldMap)
        let requestMemory = hit.events.filter { $0.eventName == "requestMemory" }.map(Self.fieldMap)
        invalid.append(
            contentsOf: Self.checkHitPath(spec, arm: arm, lookups: hitLookups, leafStore: leafStore)
        )
        if spec.kind == .plannedView,
            let offset = hitLookups.first.flatMap({ Int($0["snapshotOffset"] ?? "") })
        {
            viewOffsets[spec.id] = offset
        }
        if hit.generationTokenCount != manifest.settings.timedOutputTokens {
            invalid.append(
                "timed turn generated \(hit.generationTokenCount) tokens, expected \(manifest.settings.timedOutputTokens)"
            )
        }
        var generatedIDs: [Int]?
        if spec.kind != .thinkStrippingBoundary,
            let leaf = admin.freshestLeaf(minimumOffset: spec.prefixTokens),
            leaf.tokens.count > hit.promptTokenCount
        {
            generatedIDs = Array(leaf.tokens[hit.promptTokenCount...])
        }

        // Follow-up: another unrelated turn, then a one-token continuation so
        // the fidelity walk has the boundary that crosses the timed turn.
        try await noiseTurn()
        await settleCompression("follow-up")
        _ = events.take()
        history.append(OpenAI.ChatMessage(role: .user, content: .text(corpus.followUpUser)))
        let followUp = try await send(
            history, maxTokens: 1, affinity: sessionAffinity, record: true)
        let followUpLookups = followUp.events.filter { $0.eventName == "lookup" }.map(Self.fieldMap)

        // Fidelity over this arm's own recordings.
        let report = await CanonicalEchoFidelity.walkSession(
            requests: recorded, sessionAffinity: sessionAffinity, modelID: modelID,
            tokenizer: tokenizer)
        let session = FidelitySession(
            caseID: spec.id, block: block, arm: arm, boundaries: report.boundaries.count,
            boundaryKinds: report.boundaries.map(\.boundary.kind.rawValue).sorted(),
            mismatches: report.mismatchCount,
            skipped: report.skipped.map { "\($0.requestIndex): \($0.reason)" },
            report: CanonicalEchoFidelity.renderText(report))
        if block >= 0 { fidelity.append(session) }
        try append(session, to: block < 0 ? "warmup-fidelity.jsonl" : "fidelity.jsonl")
        try writeRecordings(
            recordings, affinity: sessionAffinity, caseID: spec.id, block: block, arm: arm)
        try appendPrivate(hit.events, name: "hit-events-\(sessionAffinity).jsonl")

        admin.setLeafCheckoutDisabled(false)
        return Observation(
            caseID: spec.id, block: block, arm: arm, armIndexInBlock: index,
            setupLookups: setupLookups, warmCompressEvents: warmCompressEvents,
            residentFormBeforeHit: leafForm, residentBitsBeforeHit: freshest.body.warmBits,
            residentBodyBytesBeforeHit: freshest.body.memoryBytes,
            residentLeafOffsetBeforeHit: freshest.body.tokenOffset,
            ttftSeconds: hit.ttftSeconds, hitLookups: hitLookups, hitLeafStore: leafStore,
            serverLookupMs: hit.diagnostics.lookupMs, serverRestoreMs: hit.diagnostics.restoreMs,
            serverPrefillMs: hit.diagnostics.prefillMs, cachedTokenCount: hit.cachedTokenCount,
            promptTokenCount: hit.promptTokenCount, generationTokenCount: hit.generationTokenCount,
            generatedTextSHA256: Self.sha256(Data((hit.reasoning + "\u{1F}" + hit.text).utf8)),
            generatedText: hit.reasoning + "\u{1F}" + hit.text, generatedTokenIDs: generatedIDs,
            mlxActiveBeforeHit: before.activeBytes, mlxActiveAfterHit: hit.mlxActiveAfter,
            mlxPeakDuringHit: hit.mlxPeak, mlxCacheAfterHit: hit.mlxCacheAfter,
            footprintBeforeHit: before.footprintBytes, footprintAfterHit: hit.footprintAfter,
            requestMemoryPeakFootprint: requestMemory.compactMap {
                Int($0["sampledRequestPeakFootprintBytes"] ?? "")
            }.max(),
            requestMemoryPeakActiveMlx: requestMemory.compactMap {
                Int($0["sampledRequestPeakActiveMlxBytes"] ?? "")
            }.max(),
            followUpLookups: followUpLookups, invalid: invalid)
    }

    /// The pre-registration's "intended restore path" rule, read from the
    /// hit's own diagnostics.
    nonisolated private static func checkHitPath(
        _ spec: Manifest.Case, arm: Arm, lookups: [[String: String]], leafStore: [String: String]?
    ) -> [String] {
        var problems: [String] = []
        guard let restore = lookups.first else { return ["timed hit: no lookup event"] }
        if restore["reason"] != "hit" {
            problems.append("timed hit: reason=\(restore["reason"] ?? "?")")
        }
        if restore["hydratedFromSSD"] == "true" { problems.append("timed hit: hydrated from SSD") }
        if restore["chainPrefixRestore"] == "true" {
            problems.append("timed hit: chain-prefix restore")
        }
        if restore["restoreMode"] == "handoff" { problems.append("timed hit: handoff, not a copy") }
        let source = restore["source"]
        switch (spec.kind, arm) {
        case (.directLeaf, .fp16), (.thinkStrippingBoundary, .fp16):
            if source != nil { problems.append("control: unexpected source=\(source!)") }
            if restore["copyReason"] != "checkoutDisabled" {
                problems.append("control: copyReason=\(restore["copyReason"] ?? "nil")")
            }
        case (.directLeaf, _), (.thinkStrippingBoundary, _):
            if source != "warm" { problems.append("warm: source=\(source ?? "nil")") }
            if restore["copyReason"] != "warmBody" {
                problems.append("warm: copyReason=\(restore["copyReason"] ?? "nil")")
            }
        case (.plannedView, .fp16):
            if source != "view" { problems.append("view control: source=\(source ?? "nil")") }
            if restore["backingLeafForm"] != "ownedBody" {
                problems.append(
                    "view control: backingLeafForm=\(restore["backingLeafForm"] ?? "nil")")
            }
            if restore["copyReason"] != "checkpoint" {
                problems.append("view control: copyReason=\(restore["copyReason"] ?? "nil")")
            }
        case (.plannedView, _):
            if source != "view" { problems.append("view warm: source=\(source ?? "nil")") }
            if restore["backingLeafForm"] != "warm" {
                problems.append("view warm: backingLeafForm=\(restore["backingLeafForm"] ?? "nil")")
            }
            if restore["copyReason"] != "checkpoint" {
                problems.append("view warm: copyReason=\(restore["copyReason"] ?? "nil")")
            }
        }
        if spec.kind == .thinkStrippingBoundary {
            guard let leafStore, leafStore["path"] == "boundary" else {
                problems.append("boundary: leafStore path=\(leafStore?["path"] ?? "nil")")
                return problems
            }
            // The transient boundary view is resolved inside the leaf store,
            // which reports the restore it made from the arm's resident form.
            if leafStore["boundary"] == nil {
                problems.append("boundary: leafStore carries no boundary kind")
            }
            switch arm {
            case .fp16:
                if leafStore["restoreMode"] != "copy"
                    || leafStore["copyReason"] != "checkoutDisabled"
                {
                    problems.append(
                        "boundary control: restoreMode=\(leafStore["restoreMode"] ?? "nil") copyReason=\(leafStore["copyReason"] ?? "nil")"
                    )
                }
            case .warm8, .warm4:
                if leafStore["restoreMode"] != "copy" || leafStore["copyReason"] != "warmBody" {
                    problems.append(
                        "boundary warm: restoreMode=\(leafStore["restoreMode"] ?? "nil") copyReason=\(leafStore["copyReason"] ?? "nil")"
                    )
                }
            }
        }
        return problems
    }

    private func makeRenderContext(
        _ spec: Manifest.Case, engine: AgentEngine
    ) async throws -> TemplateRenderContext {
        guard spec.kind == .thinkStrippingBoundary || spec.reasoningEffort != nil else {
            return .canonical
        }
        let effort = try spec.reasoningEffort.map { raw -> ReasoningEffort in
            guard let effort = ReasoningEffort(rawValue: raw) else {
                throw failure("Unknown reasoning effort \(raw)")
            }
            return effort
        }
        return TemplateRenderContext.resolve(
            requestKwargs: spec.kind == .thinkStrippingBoundary
                ? ["preserve_thinking": false] : nil,
            appDesired: [:],
            declaredFlags: await engine.llmActor.loadedDeclaredTemplateFlags(),
            templateDefaults: await engine.llmActor.loadedTemplateFlagDefaults(),
            requestedReasoningEffort: effort,
            declaresReasoningEffort: await engine.llmActor.loadedDeclaresReasoningEffort(),
            reasoningEffortTemplateDefault: await engine.llmActor
                .loadedReasoningEffortTemplateDefault())
    }

    private func noise(
        _ corpus: CaseCorpus, engine: AgentEngine, modelID: String, params: AgentGenerateParameters
    ) async throws -> TurnResult {
        var noiseParams = params
        noiseParams.maxTokens = 4
        noiseParams.reasoningEffort = nil
        return try await runTurn(
            [
                OpenAI.ChatMessage(role: .system, content: .text(corpus.noiseSystem)),
                OpenAI.ChatMessage(role: .user, content: .text(corpus.noiseUser)),
            ], parameters: noiseParams, renderContext: .canonical, engine: engine,
            modelID: modelID, deadline: manifest.resources.maximumRequestSeconds)
    }

    /// One request through the in-process Server Completion path, the shape
    /// the e2e gate uses. TTFT is the monotonic span from submission to the
    /// first text or thinking delta; the handle's diagnostics carry the
    /// server's own lookup/restore/prefill intervals.
    private func runTurn(
        _ messages: [OpenAI.ChatMessage], parameters: AgentGenerateParameters,
        renderContext: TemplateRenderContext, engine: AgentEngine, modelID: String,
        deadline: Int
    ) async throws -> TurnResult {
        let normalized = MessageConverter.normalizeRequest(
            messages, tools: nil, templateContextDigest: renderContext.digest)
        guard let conversation = normalized.prefixCacheEligibility.conversation else {
            throw failure("conversation ineligible for the prefix cache")
        }
        _ = events.take()
        Memory.peakMemory = 0
        let start = ContinuousClock.now
        let handle = try await engine.llmActor.startServerCompletion(
            modelID: modelID, conversation: conversation, toolSpecs: nil,
            parameters: parameters, renderContext: renderContext)
        var ttft: Double = 0
        var text = ""
        var reasoning = ""
        var info: AgentGeneration.Info?
        var first = false
        let watchdog = Task { [cancel = handle.cancel] in
            try await Task.sleep(for: .seconds(deadline))
            cancel()
        }
        defer { watchdog.cancel() }
        for try await event in handle.stream {
            switch event {
            case .text(let chunk):
                if !first { ttft = start.duration(to: .now).seconds; first = true }
                text += chunk
            case .thinking(let chunk):
                if !first { ttft = start.duration(to: .now).seconds; first = true }
                reasoning += chunk
            case .thinkReclassify:
                text += reasoning
                reasoning = ""
            case .info(let completion):
                info = completion
            default:
                break
            }
        }
        await handle.waitForCompletion()
        if !first { ttft = start.duration(to: .now).seconds }
        guard let info else { throw failure("stream ended without a completion info event") }
        let peak = Memory.peakMemory
        let after = RequestMemoryTelemetry.Sample.current()
        return TurnResult(
            ttftSeconds: ttft, text: text,
            reasoning: reasoning.trimmingCharacters(in: .whitespacesAndNewlines),
            promptTokenCount: info.promptTokenCount,
            generationTokenCount: info.generationTokenCount,
            cachedTokenCount: handle.cachedTokenCount, diagnostics: handle.diagnostics,
            events: events.take(), mlxPeak: peak, mlxActiveAfter: after.activeBytes,
            mlxCacheAfter: after.cacheBytes, footprintAfter: after.footprintBytes)
    }

    // MARK: - Seam measurements

    /// The dequantization allowance: the vendor `toUnquantized` conversion
    /// of the Warm Body's packed attention rows, sliced to `offset` (the
    /// view offset for a view case), evaluated through fresh fp16 outputs
    /// and device completion. One warmup, six timed. Inputs are settled
    /// before each timing; the observer overhead is the same convention
    /// over a scalar.
    nonisolated private static func measureDequantize(
        _ body: HybridCacheSnapshot, offset: Int, bits: Int, caseID: String, block: Int, arm: Arm
    ) throws -> DequantizeMeasurement {
        let layers = body.layers.filter {
            $0.kind == .sliceableAttention && $0.className == "QuantizedKVCache"
        }
        guard !layers.isEmpty, offset > 0, offset <= body.tokenOffset else {
            throw failure("dequantize measurement: no quantized attention at offset \(offset)")
        }
        let sources: [QuantizedKVCache] = layers.map { layer in
            var metaState = layer.metaState
            metaState[1] = String(offset)
            let cache = QuantizedKVCache(groupSize: Int(metaState[2])!, bits: Int(metaState[3])!)
            cache.state = layer.state.map { $0[.ellipsis, 0..<offset, 0...] }
            cache.metaState = metaState
            return cache
        }
        guard sources.allSatisfy({ $0.bits == bits }) else {
            throw failure("dequantize measurement: body bits differ from the arm")
        }
        eval(sources.flatMap(\.state))
        Stream.gpu.synchronize()
        let quantizedBytes = sources.flatMap(\.state).reduce(0) { $0 + $1.nbytes }
        var seconds: [Double] = []
        var overhead: [Double] = []
        var dequantizedBytes = 0
        Memory.peakMemory = 0
        for repetition in 0..<7 {
            let scalar = MLXArray(Float(repetition))
            eval(scalar)
            Stream.gpu.synchronize()
            let overheadStart = ContinuousClock.now
            let probe = scalar + 1
            eval(probe)
            Stream.gpu.synchronize()
            overhead.append(overheadStart.duration(to: .now).seconds)
            let start = ContinuousClock.now
            let outputs = sources.map { $0.toUnquantized() }
            eval(outputs.flatMap(\.state))
            Stream.gpu.synchronize()
            seconds.append(start.duration(to: .now).seconds)
            dequantizedBytes = outputs.flatMap(\.state).reduce(0) { $0 + $1.nbytes }
        }
        return DequantizeMeasurement(
            caseID: caseID, block: block, arm: arm, bits: bits, prefixOffset: offset,
            layers: sources.count, quantizedBytes: quantizedBytes,
            dequantizedBytes: dequantizedBytes, warmupSeconds: seconds[0],
            seconds: Array(seconds.dropFirst()), observerOverheadSeconds: overhead,
            mlxPeakBytes: Memory.peakMemory)
    }

    /// The control's counterpart under the same convention: the fp16 deep
    /// copy of the same prefix. Context only; the gate never uses it.
    nonisolated private static func measureCopy(
        _ body: HybridCacheSnapshot, offset: Int, caseID: String, block: Int
    ) -> CopyMeasurement {
        let sources = body.layers.filter { $0.kind == .sliceableAttention }
            .flatMap { $0.state.map { $0[.ellipsis, 0..<offset, 0...] } }
        eval(sources)
        Stream.gpu.synchronize()
        var seconds: [Double] = []
        var bytes = 0
        for _ in 0..<7 {
            let start = ContinuousClock.now
            let copies = sources.map { HybridCacheSnapshot.deepCopyState($0) }
            eval(copies)
            Stream.gpu.synchronize()
            seconds.append(start.duration(to: .now).seconds)
            bytes = copies.reduce(0) { $0 + $1.nbytes }
        }
        return CopyMeasurement(
            caseID: caseID, block: block, prefixOffset: offset, bytes: bytes,
            warmupSeconds: seconds[0], seconds: Array(seconds.dropFirst()))
    }

    // MARK: - Report

    nonisolated struct CaseVerdict: Codable, Sendable {
        let caseID: String
        let arm: Arm
        let samples: Int
        let fidelityMismatches: Int
        let fidelityBoundaries: Int
        let boundaryKindCoverageMatchesControl: Bool
        let tokenMismatches: Int
        let promptTokenMismatches: Int
        let ttftMedianMs: Double?
        let ttftP95Ms: Double?
        let controlTtftMedianMs: Double?
        let pairedExcessMs: [Double]
        let pairedExcessMedianMs: Double?
        let dequantizeMedianMs: Double?
        let dequantizeSamplesMs: [Double]
        let timingPasses: Bool?
        let invalidObservations: Int
        let residentBodyBytesMedian: Int
        let mlxPeakDuringHitMax: Int
        let footprintAfterHitMax: Int?
        let verdict: String
    }

    private func verdicts() -> [CaseVerdict] {
        var result: [CaseVerdict] = []
        for spec in manifest.cases {
            let caseObservations = observations.filter { $0.caseID == spec.id }
            let control = caseObservations.filter { $0.arm == .fp16 }
            let controlFidelity = fidelity.filter { $0.caseID == spec.id && $0.arm == .fp16 }
            for arm in Arm.allCases {
                let armObservations = caseObservations.filter { $0.arm == arm }
                let armFidelity = fidelity.filter { $0.caseID == spec.id && $0.arm == arm }
                let paired: [(Observation, Observation)] = armObservations.compactMap { warm in
                    control.first { $0.block == warm.block }.map { (warm, $0) }
                }
                let excess =
                    arm == .fp16 ? [] : paired.map { ($0.0.ttftSeconds - $0.1.ttftSeconds) * 1000 }
                let tokenMismatches =
                    arm == .fp16
                    ? 0
                    : paired.count { warm, ctrl in
                        if let a = warm.generatedTokenIDs, let b = ctrl.generatedTokenIDs {
                            return a != b || warm.generatedText != ctrl.generatedText
                        }
                        return warm.generatedText != ctrl.generatedText
                            || warm.generationTokenCount != ctrl.generationTokenCount
                    }
                let promptMismatches =
                    arm == .fp16
                    ? 0 : paired.count { $0.0.promptTokenCount != $0.1.promptTokenCount }
                let dequant = dequantizations.filter { $0.caseID == spec.id && $0.arm == arm }
                    .flatMap { $0.seconds.map { $0 * 1000 } }
                let ttfts = armObservations.map { $0.ttftSeconds * 1000 }
                let coverage = zip(
                    armFidelity.sorted { $0.block < $1.block },
                    controlFidelity.sorted { $0.block < $1.block }
                ).allSatisfy { $0.boundaryKinds == $1.boundaryKinds && $0.boundaries > 0 }
                let invalidCount = armObservations.count { !$0.invalid.isEmpty }
                let mismatches = armFidelity.reduce(0) { $0 + $1.mismatches }
                let boundaries = armFidelity.reduce(0) { $0 + $1.boundaries }
                let excessMedian = excess.isEmpty ? nil : Self.median(excess)
                let dequantMedian = dequant.isEmpty ? nil : Self.median(dequant)
                let timing: Bool? =
                    arm == .fp16 ? nil : dequantMedian.flatMap { d in excessMedian.map { $0 <= d } }
                let verdict: String
                if arm == .fp16 {
                    verdict =
                        invalidCount == 0 && mismatches == 0 && boundaries > 0
                        ? "REFERENCE" : "INVALID"
                } else if armObservations.count != manifest.orders.count
                    || paired.count != manifest.orders.count
                    || dequant.count != manifest.orders.count * 6 || invalidCount > 0
                    || control.count { !$0.invalid.isEmpty } > 0
                {
                    verdict = "INCONCLUSIVE"
                } else if mismatches == 0, boundaries > 0, coverage, tokenMismatches == 0,
                    promptMismatches == 0, timing == true
                {
                    verdict = "PASS"
                } else {
                    verdict = "FAIL"
                }
                result.append(
                    CaseVerdict(
                        caseID: spec.id, arm: arm, samples: armObservations.count,
                        fidelityMismatches: mismatches, fidelityBoundaries: boundaries,
                        boundaryKindCoverageMatchesControl: coverage,
                        tokenMismatches: tokenMismatches, promptTokenMismatches: promptMismatches,
                        ttftMedianMs: ttfts.isEmpty ? nil : Self.median(ttfts),
                        ttftP95Ms: ttfts.isEmpty ? nil : Self.percentile(ttfts, 0.95),
                        controlTtftMedianMs: control.isEmpty
                            ? nil : Self.median(control.map { $0.ttftSeconds * 1000 }),
                        pairedExcessMs: excess, pairedExcessMedianMs: excessMedian,
                        dequantizeMedianMs: dequantMedian, dequantizeSamplesMs: dequant,
                        timingPasses: timing, invalidObservations: invalidCount,
                        residentBodyBytesMedian: armObservations.isEmpty
                            ? 0
                            : Int(
                                Self.median(
                                    armObservations.map { Double($0.residentBodyBytesBeforeHit) })),
                        mlxPeakDuringHitMax: armObservations.map(\.mlxPeakDuringHit).max() ?? 0,
                        footprintAfterHitMax: armObservations.compactMap(\.footprintAfterHit).max(),
                        verdict: verdict))
            }
        }
        return result
    }

    private func writeReport() throws {
        let verdicts = verdicts()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(verdicts).write(to: reportDir.appendingPathComponent("verdicts.json"))
        try encoder.encode(invalidations).write(
            to: reportDir.appendingPathComponent("invalidations.json"))
        var text = "# Warm Body parity gate — runner output\n\n"
        text +=
            "Model: \(runner.activeConfig.resolvedModelID). Hardware: \(runner.resolvedHardwareDescription).\n"
        text +=
            "Revision: \(runner.activeConfig.sourceRevision ?? "unrecorded"). Plan: approved-plan.json.\n\n"
        text +=
            "| Case | Arm | n | Fidelity mismatches / boundaries | Kind coverage | Token mismatches | TTFT median / p95 (ms) | Dequantize median (ms) | Paired excess median (ms) | Body bytes | Peak MLX during hit | Invalid | Verdict |\n"
        text +=
            "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n"
        for v in verdicts {
            let ms: (Double?) -> String = { $0.map { String(format: "%.1f", $0) } ?? "n/a" }
            text +=
                "| \(v.caseID) | \(v.arm.rawValue) | \(v.samples) | \(v.fidelityMismatches) / \(v.fidelityBoundaries) "
            text +=
                "| \(v.boundaryKindCoverageMatchesControl ? "same" : "differs") | \(v.tokenMismatches) "
            text +=
                "| \(ms(v.ttftMedianMs)) / \(ms(v.ttftP95Ms)) | \(v.dequantizeMedianMs.map { String(format: "%.2f", $0) } ?? "n/a") "
            text +=
                "| \(v.arm == .fp16 ? "reference" : ms(v.pairedExcessMedianMs)) | \(v.residentBodyBytesMedian) "
            text += "| \(v.mlxPeakDuringHitMax) | \(v.invalidObservations) | \(v.verdict) |\n"
        }
        text += "\nInvalidations: \(invalidations.count)\n"
        for line in invalidations { text += "- \(line)\n" }
        text +=
            "\nThe runner records; the owner decides per the pre-registration. No flag was changed.\n"
        try text.write(
            to: reportDir.appendingPathComponent("README.md"), atomically: true, encoding: .utf8)
        Log.agent.notice("warm parity results: \(self.reportDir.path)")
    }

    // MARK: - Files

    private func append(_ value: some Encodable, to name: String) throws {
        let url = reportDir.appendingPathComponent(name)
        var line = try JSONEncoder().encode(value)
        line.append(0x0A)
        if FileManager.default.fileExists(atPath: url.path) {
            let handle = try FileHandle(forWritingTo: url)
            defer { try? handle.close() }
            try handle.seekToEnd()
            try handle.write(contentsOf: line)
        } else {
            try line.write(to: url)
        }
    }

    private func appendPrivate(_ events: [PromptCacheTelemetryEvent], name: String) throws {
        let url = privateDir.appendingPathComponent(name)
        var data = Data()
        for event in events {
            data.append(try JSONEncoder().encode(event))
            data.append(0x0A)
        }
        try data.write(to: url)
    }

    /// `HTTPRequestLogger`-shaped recordings, so the corpus gate can replay
    /// this arm offline (`docs/testing.md`, Canonical-echo fidelity gate).
    private func writeRecordings(
        _ requests: [OpenAI.ChatCompletionRequest], affinity: String, caseID: String, block: Int,
        arm: Arm
    ) throws {
        let dir = privateDir.appendingPathComponent(
            "recordings/\(caseID)/block-\(block)/\(arm.rawValue)/http-completions",
            isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        for (index, request) in requests.enumerated() {
            var data = Data("// session=\(affinity)\n".utf8)
            data.append(try encoder.encode(request))
            try data.write(
                to: dir.appendingPathComponent(
                    String(format: "00-00-00-%04d-request.json", index + 1)))
        }
    }

    // MARK: - Helpers

    nonisolated private static func fieldMap(_ event: PromptCacheTelemetryEvent) -> [String: String]
    {
        var map: [String: String] = ["eventName": event.eventName]
        for field in event.fields { map[field.key] = field.value }
        return map
    }

    nonisolated private static func median(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return .nan }
        let sorted = values.sorted()
        return sorted.count % 2 == 1
            ? sorted[sorted.count / 2]
            : (sorted[sorted.count / 2 - 1] + sorted[sorted.count / 2]) / 2
    }

    nonisolated private static func percentile(_ values: [Double], _ p: Double) -> Double {
        guard !values.isEmpty else { return .nan }
        let sorted = values.sorted()
        let rank = Int((Double(sorted.count - 1) * p).rounded(.up))
        return sorted[min(max(rank, 0), sorted.count - 1)]
    }

    nonisolated private static func fileSHA256(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hash = SHA256()
        while let chunk = try handle.read(upToCount: 8 * 1_024 * 1_024), !chunk.isEmpty {
            hash.update(data: chunk)
        }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }

    nonisolated private static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private func log(_ message: String) {
        Log.agent.notice("[warm-parity] \(message)")
        print("[warm-parity] \(message)")
        fflush(stdout)
    }

    nonisolated private static func failure(_ message: String) -> NSError {
        NSError(
            domain: "WarmBodyParityBench", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
    }
    private func failure(_ message: String) -> NSError { Self.failure(message) }
}
