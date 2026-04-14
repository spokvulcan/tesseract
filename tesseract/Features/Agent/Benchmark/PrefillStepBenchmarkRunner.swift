import Foundation
import MLX
import MLXLMCommon
import os

nonisolated enum PrefillStepBenchmarkMode: String, Codable, CaseIterable, Equatable, Sendable {
    case cold
    case warm
}

nonisolated struct PrefillStepBenchmarkCase: Codable, Equatable, Sendable {
    let mode: PrefillStepBenchmarkMode
    let prefillStepSize: Int

    var id: String {
        "\(mode.rawValue)-\(prefillStepSize)"
    }
}

nonisolated struct PrefillStepBenchmarkMeasurement: Codable, Equatable, Sendable {
    let caseID: String
    let mode: PrefillStepBenchmarkMode
    let prefillStepSize: Int
    let passed: Bool
    let error: String?
    let promptTimeSeconds: Double?
    let externalTTFTSeconds: Double?
    let promptTokenCount: Int?
    let cachedTokenCount: Int?
    let prefilledTokenCount: Int?
    let peakMemoryMB: Float?
    let activeMemoryBeforeMB: Float?
    let activeMemoryAfterMB: Float?
}

nonisolated struct PrefillStepBenchmarkSummary: Codable, Equatable, Sendable {
    nonisolated struct SuggestedAdaptivePair: Codable, Equatable, Sendable {
        let coldStepSize: Int
        let warmStepSize: Int
    }

    let fastestColdStepSize: Int?
    let lowestPeakColdStepSize: Int?
    let fastestWarmStepSize: Int?
    let lowestPeakWarmStepSize: Int?
    let suggestedAdaptivePair: SuggestedAdaptivePair?
}

nonisolated struct PrefillStepBenchmarkFixture: @unchecked Sendable {
    let systemPrompt: String
    let coldUserMessage: String
    let warmUserMessage: String
    let toolSpecs: [ToolSpec]

    func promptTokenCounts(
        measurePromptTokens: (String, String, [ToolSpec]) -> Int
    ) -> (cold: Int, warm: Int) {
        (
            cold: measurePromptTokens(systemPrompt, coldUserMessage, toolSpecs),
            warm: measurePromptTokens(systemPrompt, warmUserMessage, toolSpecs)
        )
    }
}

nonisolated enum PrefillStepBenchmarkSupport {
    nonisolated static let defaultStepSizes = [256, 512, 1024, 2048, 4096]
    nonisolated static let targetStablePrefixTokens = 12_000
    nonisolated static let targetUserTokens = 4_000

    nonisolated static let markerCandidates: [(String, String)] = [
        ("alpha", "bravo"),
        ("A", "B"),
        ("left", "right"),
        ("one", "two"),
        ("cat", "dog"),
    ]

    nonisolated static func benchmarkMatrix(
        stepSizes: [Int] = defaultStepSizes
    ) -> [PrefillStepBenchmarkCase] {
        stepSizes.flatMap { step in
            [
                PrefillStepBenchmarkCase(mode: .cold, prefillStepSize: step),
                PrefillStepBenchmarkCase(mode: .warm, prefillStepSize: step),
            ]
        }
    }

    nonisolated static func validationFailure(
        mode: PrefillStepBenchmarkMode,
        promptTokenCount: Int,
        cachedTokenCount: Int
    ) -> String? {
        switch mode {
        case .cold:
            guard cachedTokenCount == 0 else {
                return "cold case expected cachedTokenCount == 0, got \(cachedTokenCount)"
            }
            return nil
        case .warm:
            guard cachedTokenCount > 0 else {
                return "warm case expected cachedTokenCount > 0, got \(cachedTokenCount)"
            }
            guard cachedTokenCount < promptTokenCount else {
                return "warm case expected non-zero suffix prefill, got cachedTokenCount=\(cachedTokenCount) promptTokenCount=\(promptTokenCount)"
            }
            return nil
        }
    }

    nonisolated static func summarize(
        measurements: [PrefillStepBenchmarkMeasurement]
    ) -> PrefillStepBenchmarkSummary {
        let passed = measurements.filter(\.passed)
        let cold = passed.filter { $0.mode == .cold }
        let warm = passed.filter { $0.mode == .warm }

        func fastest(
            _ values: [PrefillStepBenchmarkMeasurement]
        ) -> PrefillStepBenchmarkMeasurement? {
            values
                .filter { $0.promptTimeSeconds != nil }
                .min {
                    if $0.promptTimeSeconds == $1.promptTimeSeconds {
                        return $0.prefillStepSize < $1.prefillStepSize
                    }
                    return ($0.promptTimeSeconds ?? .infinity) < ($1.promptTimeSeconds ?? .infinity)
                }
        }

        func lowestPeak(
            _ values: [PrefillStepBenchmarkMeasurement]
        ) -> PrefillStepBenchmarkMeasurement? {
            values
                .filter { $0.peakMemoryMB != nil }
                .min {
                    if $0.peakMemoryMB == $1.peakMemoryMB {
                        return $0.prefillStepSize < $1.prefillStepSize
                    }
                    return ($0.peakMemoryMB ?? .infinity) < ($1.peakMemoryMB ?? .infinity)
                }
        }

        let fastestCold = fastest(cold)
        let lowestPeakCold = lowestPeak(cold)
        let fastestWarm = fastest(warm)
        let lowestPeakWarm = lowestPeak(warm)
        let adaptivePair: PrefillStepBenchmarkSummary.SuggestedAdaptivePair?
        if let coldStep = lowestPeakCold?.prefillStepSize,
           let warmStep = fastestWarm?.prefillStepSize
        {
            adaptivePair = .init(coldStepSize: coldStep, warmStepSize: warmStep)
        } else {
            adaptivePair = nil
        }

        return PrefillStepBenchmarkSummary(
            fastestColdStepSize: fastestCold?.prefillStepSize,
            lowestPeakColdStepSize: lowestPeakCold?.prefillStepSize,
            fastestWarmStepSize: fastestWarm?.prefillStepSize,
            lowestPeakWarmStepSize: lowestPeakWarm?.prefillStepSize,
            suggestedAdaptivePair: adaptivePair
        )
    }

    nonisolated static func buildFixture(
        markerPair: (String, String),
        targetStablePrefixTokens: Int = targetStablePrefixTokens,
        targetUserTokens: Int = targetUserTokens,
        measureTextTokens: (String) -> Int
    ) -> PrefillStepBenchmarkFixture {
        let systemPrompt = buildText(
            targetTokens: targetStablePrefixTokens,
            measureTextTokens: measureTextTokens,
            seedParagraph: systemSeedParagraph
        )
        let separator = "\n\nRequest label: "
        let coldSuffix = separator + markerPair.0
        let warmSuffix = separator + markerPair.1
        let suffixBudget = max(
            measureTextTokens(coldSuffix),
            measureTextTokens(warmSuffix)
        )
        let userBody = buildText(
            targetTokens: max(32, targetUserTokens - suffixBudget),
            measureTextTokens: measureTextTokens,
            seedParagraph: userSeedParagraph
        )

        return PrefillStepBenchmarkFixture(
            systemPrompt: systemPrompt,
            coldUserMessage: userBody + coldSuffix,
            warmUserMessage: userBody + warmSuffix,
            toolSpecs: deterministicToolSpecs
        )
    }

    nonisolated static func buildText(
        targetTokens: Int,
        measureTextTokens: (String) -> Int,
        seedParagraph: String
    ) -> String {
        var text = seedParagraph.trimmingCharacters(in: .whitespacesAndNewlines)
        while measureTextTokens(text) < targetTokens {
            text += "\n\n" + seedParagraph
        }
        return text
    }

    nonisolated static let deterministicToolSpecs: [ToolSpec] = [
        [
            "type": "function" as any Sendable,
            "function": [
                "name": "read",
                "description": "Read a file from disk.",
                "parameters": [
                    "type": "object" as any Sendable,
                    "required": ["path"],
                    "properties": [
                        "path": [
                            "type": "string" as any Sendable,
                            "description": "Absolute path to the file.",
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ],
        [
            "type": "function" as any Sendable,
            "function": [
                "name": "ls",
                "description": "List entries in a directory.",
                "parameters": [
                    "type": "object" as any Sendable,
                    "required": ["path"],
                    "properties": [
                        "path": [
                            "type": "string" as any Sendable,
                            "description": "Absolute path to the directory.",
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ],
    ]

    private nonisolated static let systemSeedParagraph = """
        You are running a deterministic prefill benchmark against a coding-agent prompt path. \
        The benchmark intentionally uses a very long, repeated system instruction block so the \
        prompt contains a large stable shared prefix. The content should remain plain text, \
        predictable, and semantically harmless. Prefer concise answers. Never call tools unless \
        the user explicitly asks to inspect a path. When you do mention files, use absolute paths. \
        Keep the style factual, avoid greetings, and avoid long explanations. This paragraph is \
        repeated to build a large stable prefix for cache-hit benchmarking, not to change the task.
        """

    private nonisolated static let userSeedParagraph = """
        Inspect the benchmark fixture request carefully. The goal is to keep the user message \
        deterministic and long enough to force a non-trivial suffix prefill while still sharing \
        the same large stable system and tool prefix across requests. Summarize the request in one \
        short sentence and do not call any tools. This repeated paragraph exists only to pad the \
        user turn to the desired token budget for the prefill-step-size sweep.
        """
}

@MainActor
final class PrefillStepBenchmarkRunner {

    private let runner: BenchmarkRunner
    private let logger = Logger(subsystem: "app.tesseract.agent", category: "benchmark")
    private var logFileHandle: FileHandle?
    private let reportStamp: String

    init(runner: BenchmarkRunner) {
        self.runner = runner
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        self.reportStamp = formatter.string(from: Date())
    }

    func run() async throws {
        setupLogging()
        log("PrefillStepBenchmark starting — model=\(runner.resolvedModelName)")

        let engine = AgentEngine()
        let modelDir = try runner.resolveModelDirectory()
        let modelID = runner.activeConfig.resolvedModelID
        log("Loading model from: \(modelDir.path)")
        try await engine.loadModel(from: modelDir, visionMode: false)
        log("Model loaded.")

        let fixtureData = try await engine.withModelContainer { container in
            try await container.perform { context in
                try await Self.buildBalancedFixture(context: context)
            }
        }
        let fixture = fixtureData.fixture
        log(
            "Fixture calibrated — coldPromptTokens=\(fixtureData.coldPromptTokens) "
                + "warmPromptTokens=\(fixtureData.warmPromptTokens) "
                + "toolCount=\(fixture.toolSpecs.count)"
        )

        var measurements: [PrefillStepBenchmarkMeasurement] = []
        let cases = PrefillStepBenchmarkSupport.benchmarkMatrix()

        for spec in cases {
            log("\n── \(spec.id) ──")
            let measurement = await runCase(
                spec,
                fixture: fixture,
                modelDir: modelDir,
                modelID: modelID,
                engine: engine
            )
            measurements.append(measurement)
            try writeReport(
                fixtureColdPromptTokens: fixtureData.coldPromptTokens,
                fixtureWarmPromptTokens: fixtureData.warmPromptTokens,
                measurements: measurements
            )

            if measurement.passed {
                log(
                    "  ✅ promptTime=\(Self.format(measurement.promptTimeSeconds))s "
                        + "ttft=\(Self.format(measurement.externalTTFTSeconds))s "
                        + "cached=\(measurement.cachedTokenCount ?? -1) "
                        + "prefilled=\(measurement.prefilledTokenCount ?? -1) "
                        + "peakMB=\(Self.format(measurement.peakMemoryMB))"
                )
            } else {
                log("  ❌ \(measurement.error ?? "unknown failure")")
            }
        }

        let summary = PrefillStepBenchmarkSupport.summarize(measurements: measurements)
        log("\n── Summary ──")
        log("  fastestColdStepSize=\(summary.fastestColdStepSize.map(String.init) ?? "n/a")")
        log("  lowestPeakColdStepSize=\(summary.lowestPeakColdStepSize.map(String.init) ?? "n/a")")
        log("  fastestWarmStepSize=\(summary.fastestWarmStepSize.map(String.init) ?? "n/a")")
        log("  lowestPeakWarmStepSize=\(summary.lowestPeakWarmStepSize.map(String.init) ?? "n/a")")
        if let pair = summary.suggestedAdaptivePair {
            log("  suggestedAdaptivePair=cold:\(pair.coldStepSize) warm:\(pair.warmStepSize)")
        } else {
            log("  suggestedAdaptivePair=n/a")
        }

        let failures = measurements.filter { !$0.passed }
        let allPassed = failures.isEmpty && measurements.count == cases.count
        log("\nOverall: \(allPassed ? "PASS" : "FAIL")")
        log("Report written to: \(reportURL.path)")
        logFileHandle?.closeFile()

        if !allPassed {
            throw PrefillStepBenchmarkError.verificationFailed(failedCases: failures.map(\.caseID))
        }
    }

    private struct FixtureData: Sendable {
        let fixture: PrefillStepBenchmarkFixture
        let coldPromptTokens: Int
        let warmPromptTokens: Int
    }

    private struct RequestMeasurement {
        let promptTimeSeconds: Double
        let externalTTFTSeconds: Double
        let promptTokenCount: Int
        let cachedTokenCount: Int
    }

    private func runCase(
        _ spec: PrefillStepBenchmarkCase,
        fixture: PrefillStepBenchmarkFixture,
        modelDir: URL,
        modelID: String,
        engine: AgentEngine
    ) async -> PrefillStepBenchmarkMeasurement {
        do {
            engine.unloadModel()
            try await engine.loadModel(from: modelDir, visionMode: false)

            let parameters = AgentGenerateParameters(
                maxTokens: 1,
                temperature: 0.0,
                topP: 1.0,
                topK: 1,
                minP: 0.0,
                prefillStepSize: spec.prefillStepSize
            )

            if spec.mode == .warm {
                _ = try await runRequest(
                    caseID: "\(spec.id)-prime",
                    engine: engine,
                    modelID: modelID,
                    systemPrompt: fixture.systemPrompt,
                    userMessage: fixture.coldUserMessage,
                    toolSpecs: fixture.toolSpecs,
                    parameters: parameters
                )
            }

            await engine.clearMemoryCache()
            Memory.peakMemory = 0
            let activeBefore = await engine.memoryStats().activeMB
            let measuredUserMessage = spec.mode == .cold
                ? fixture.coldUserMessage
                : fixture.warmUserMessage
            let request = try await runRequest(
                caseID: spec.id,
                engine: engine,
                modelID: modelID,
                systemPrompt: fixture.systemPrompt,
                userMessage: measuredUserMessage,
                toolSpecs: fixture.toolSpecs,
                parameters: parameters
            )
            let activeAfter = await engine.memoryStats().activeMB
            let peakMB = await engine.memoryStats().peakMB
            let prefilledTokens = max(0, request.promptTokenCount - request.cachedTokenCount)

            if let failure = PrefillStepBenchmarkSupport.validationFailure(
                mode: spec.mode,
                promptTokenCount: request.promptTokenCount,
                cachedTokenCount: request.cachedTokenCount
            ) {
                return PrefillStepBenchmarkMeasurement(
                    caseID: spec.id,
                    mode: spec.mode,
                    prefillStepSize: spec.prefillStepSize,
                    passed: false,
                    error: failure,
                    promptTimeSeconds: request.promptTimeSeconds,
                    externalTTFTSeconds: request.externalTTFTSeconds,
                    promptTokenCount: request.promptTokenCount,
                    cachedTokenCount: request.cachedTokenCount,
                    prefilledTokenCount: prefilledTokens,
                    peakMemoryMB: peakMB,
                    activeMemoryBeforeMB: activeBefore,
                    activeMemoryAfterMB: activeAfter
                )
            }

            return PrefillStepBenchmarkMeasurement(
                caseID: spec.id,
                mode: spec.mode,
                prefillStepSize: spec.prefillStepSize,
                passed: true,
                error: nil,
                promptTimeSeconds: request.promptTimeSeconds,
                externalTTFTSeconds: request.externalTTFTSeconds,
                promptTokenCount: request.promptTokenCount,
                cachedTokenCount: request.cachedTokenCount,
                prefilledTokenCount: prefilledTokens,
                peakMemoryMB: peakMB,
                activeMemoryBeforeMB: activeBefore,
                activeMemoryAfterMB: activeAfter
            )
        } catch {
            return PrefillStepBenchmarkMeasurement(
                caseID: spec.id,
                mode: spec.mode,
                prefillStepSize: spec.prefillStepSize,
                passed: false,
                error: error.localizedDescription,
                promptTimeSeconds: nil,
                externalTTFTSeconds: nil,
                promptTokenCount: nil,
                cachedTokenCount: nil,
                prefilledTokenCount: nil,
                peakMemoryMB: nil,
                activeMemoryBeforeMB: nil,
                activeMemoryAfterMB: nil
            )
        }
    }

    private func runRequest(
        caseID: String,
        engine: AgentEngine,
        modelID: String,
        systemPrompt: String,
        userMessage: String,
        toolSpecs: [ToolSpec],
        parameters: AgentGenerateParameters
    ) async throws -> RequestMeasurement {
        let prefixCacheConversation = HTTPPrefixCacheConversation(
            systemPrompt: systemPrompt,
            messages: [HTTPPrefixCacheMessage(role: .user, content: userMessage)]
        )
        let llmMessages: [LLMMessage] = [.user(content: userMessage, images: [])]

        let startInstant = ContinuousClock.now
        let start = try await engine.generateServerTextCompletion(
            modelID: modelID,
            systemPrompt: systemPrompt,
            messages: llmMessages,
            toolSpecs: toolSpecs,
            prefixCacheConversation: prefixCacheConversation,
            sessionAffinity: nil,
            parameters: parameters
        )

        var firstEventSeen = false
        var ttftSeconds = 0.0
        var completionInfo: AgentGeneration.Info?

        for try await event in start.stream {
            if !firstEventSeen {
                ttftSeconds = Self.elapsedSeconds(from: startInstant)
                firstEventSeen = true
            }
            if case .info(let info) = event {
                completionInfo = info
            }
        }

        if !firstEventSeen {
            ttftSeconds = Self.elapsedSeconds(from: startInstant)
        }

        guard let completionInfo else {
            throw PrefillStepBenchmarkError.missingCompletionInfo(caseID: caseID)
        }

        return RequestMeasurement(
            promptTimeSeconds: completionInfo.promptTime,
            externalTTFTSeconds: ttftSeconds,
            promptTokenCount: completionInfo.promptTokenCount,
            cachedTokenCount: start.cachedTokenCount
        )
    }

    private nonisolated static func buildBalancedFixture(
        context: ModelContext
    ) async throws -> FixtureData {
        let measureTextTokens: (String) -> Int = { text in
            context.tokenizer.encode(text: text, addSpecialTokens: false).count
        }

        for pair in PrefillStepBenchmarkSupport.markerCandidates {
            let prefixedCold = "\n\nRequest label: " + pair.0
            let prefixedWarm = "\n\nRequest label: " + pair.1
            guard measureTextTokens(prefixedCold) == measureTextTokens(prefixedWarm) else {
                continue
            }

            let fixture = PrefillStepBenchmarkSupport.buildFixture(
                markerPair: pair,
                measureTextTokens: measureTextTokens
            )
            let coldPromptTokens = try await measurePromptTokens(
                context: context,
                systemPrompt: fixture.systemPrompt,
                userMessage: fixture.coldUserMessage,
                toolSpecs: fixture.toolSpecs
            )
            let warmPromptTokens = try await measurePromptTokens(
                context: context,
                systemPrompt: fixture.systemPrompt,
                userMessage: fixture.warmUserMessage,
                toolSpecs: fixture.toolSpecs
            )
            if coldPromptTokens == warmPromptTokens {
                return FixtureData(
                    fixture: fixture,
                    coldPromptTokens: coldPromptTokens,
                    warmPromptTokens: warmPromptTokens
                )
            }
        }

        throw PrefillStepBenchmarkError.unbalancedFixture
    }

    private nonisolated static func measurePromptTokens(
        context: ModelContext,
        systemPrompt: String,
        userMessage: String,
        toolSpecs: [ToolSpec]
    ) async throws -> Int {
        let prepared = try await context.processor.prepare(
            input: UserInput(
                chat: [
                    .system(systemPrompt),
                    .user(userMessage),
                ],
                tools: toolSpecs
            )
        )
        return prepared.text.tokens.size
    }

    private func writeReport(
        fixtureColdPromptTokens: Int,
        fixtureWarmPromptTokens: Int,
        measurements: [PrefillStepBenchmarkMeasurement]
    ) throws {
        struct Metadata: Codable {
            let date: String
            let model: String
            let hardware: String
            let targetStablePrefixTokens: Int
            let targetUserTokens: Int
            let stepSizes: [Int]
        }

        struct FixtureInfo: Codable {
            let coldPromptTokens: Int
            let warmPromptTokens: Int
            let toolCount: Int
        }

        struct Report: Codable {
            let metadata: Metadata
            let fixture: FixtureInfo
            let measurements: [PrefillStepBenchmarkMeasurement]
            let summary: PrefillStepBenchmarkSummary
            let passed: Bool
        }

        try FileManager.default.createDirectory(at: reportDir, withIntermediateDirectories: true)
        let summary = PrefillStepBenchmarkSupport.summarize(measurements: measurements)
        let report = Report(
            metadata: Metadata(
                date: ISO8601DateFormatter().string(from: Date()),
                model: runner.resolvedModelName,
                hardware: runner.resolvedHardwareDescription,
                targetStablePrefixTokens: PrefillStepBenchmarkSupport.targetStablePrefixTokens,
                targetUserTokens: PrefillStepBenchmarkSupport.targetUserTokens,
                stepSizes: PrefillStepBenchmarkSupport.defaultStepSizes
            ),
            fixture: FixtureInfo(
                coldPromptTokens: fixtureColdPromptTokens,
                warmPromptTokens: fixtureWarmPromptTokens,
                toolCount: PrefillStepBenchmarkSupport.deterministicToolSpecs.count
            ),
            measurements: measurements,
            summary: summary,
            passed: measurements.count == PrefillStepBenchmarkSupport.defaultStepSizes.count * 2
                && measurements.allSatisfy(\.passed)
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(report).write(to: reportURL, options: .atomic)
    }

    private var reportDir: URL {
        runner.activeConfig.outputDir.appendingPathComponent("prefill-step-benchmark")
    }

    private var reportURL: URL {
        reportDir.appendingPathComponent("prefill_step_benchmark_\(reportStamp).json")
    }

    private func setupLogging() {
        try? FileManager.default.createDirectory(at: reportDir, withIntermediateDirectories: true)
        let logURL = reportDir.appendingPathComponent("latest.log")
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        logFileHandle = FileHandle(forWritingAtPath: logURL.path)
    }

    private func log(_ message: String) {
        let line = "[\(Self.timestamp())] \(message)"
        logger.info("\(line, privacy: .public)")
        if let data = (line + "\n").data(using: .utf8) {
            logFileHandle?.write(data)
        }
    }

    private static func timestamp() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss.SSS"
        return formatter.string(from: Date())
    }

    private static func elapsedSeconds(from start: ContinuousClock.Instant) -> Double {
        start.duration(to: .now).seconds
    }

    private static func format(_ value: Double?) -> String {
        guard let value else { return "n/a" }
        return String(format: "%.3f", value)
    }

    private static func format(_ value: Float) -> String {
        String(format: "%.0f", value)
    }

    private static func format(_ value: Float?) -> String {
        guard let value else { return "n/a" }
        return format(value)
    }
}

nonisolated enum PrefillStepBenchmarkError: LocalizedError {
    case unbalancedFixture
    case missingCompletionInfo(caseID: String)
    case verificationFailed(failedCases: [String])

    var errorDescription: String? {
        switch self {
        case .unbalancedFixture:
            "Failed to build a cold/warm fixture pair with equal prompt token counts"
        case .missingCompletionInfo(let caseID):
            "Missing completion info while measuring \(caseID)"
        case .verificationFailed(let failedCases):
            "PrefillStepBenchmark failed cases: \(failedCases.joined(separator: ", "))"
        }
    }
}
