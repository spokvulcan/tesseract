import Foundation
import MLX
import MLXLMCommon
import os

/// Orchestrates the full benchmark run: load model, run scenarios, write reports.
///
/// **Deprecated**: This benchmark runner uses the old domain-tool architecture
/// (AgentRunner, LegacyToolRegistry, AgentDataStore). It will be rewritten in
/// Epic 7 to test the new file-based tool workflows.
@MainActor
final class BenchmarkRunner {

    private let config: BenchmarkConfig
    private let logger = Logger(subsystem: "com.tesseract.app", category: "benchmark")
    private var logFileHandle: FileHandle?

    init(config: BenchmarkConfig = .fromCommandLine()) {
        self.config = config
    }

    func run() async throws {
        setupLogging()
        log("Benchmark starting — sweep: \(config.sweep.rawValue)")
        log("Output dir: \(config.outputDir.path)")

        // Load model
        let engine = AgentEngine()
        let modelDir = try resolveModelDirectory()
        log("Loading model from: \(modelDir.path)")
        try await engine.loadModel(from: modelDir)
        log("Model loaded successfully")

        let scenarios = BenchmarkScenarios.filtered(by: config.scenarioIDs)
        log("Running \(scenarios.count) scenario(s)")

        let paramConfigs = config.parameterConfigs
        log("Parameter configs: \(paramConfigs.count)")

        // Ensure results directory exists
        let resultsDir = resolveResultsDirectory()
        try FileManager.default.createDirectory(at: resultsDir, withIntermediateDirectories: true)

        // Determine model-specific base parameters
        let targetID = config.resolvedModelID
        let modelBaseParams = AgentGenerateParameters.forModel(targetID)

        for (configIdx, baseParams) in paramConfigs.enumerated() {
            // Cap maxTokens per round to prevent runaway generation
            // For quick sweep, use model-specific params; for full sweep, use the sweep config
            var params = config.sweep == .quick ? modelBaseParams : baseParams
            params.maxTokens = min(params.maxTokens, config.maxTokensPerRound)

            let paramLabel = BenchmarkConfig.label(for: params)
            log("\n━━━ Config \(configIdx + 1)/\(paramConfigs.count): \(paramLabel) (maxTokens=\(params.maxTokens)) ━━━")

            var scenarioResults: [BenchmarkScenarioResult] = []
            var peakMemoryMB: Float = 0

            for scenario in scenarios {
                // Skip slow scenarios on non-default configs during full sweep
                if config.sweep == .full && configIdx > 0 &&
                   BenchmarkConfig.slowScenarioIDs.contains(scenario.id) {
                    log("  Skipping \(scenario.id) (slow scenario, non-default config)")
                    continue
                }

                let result = try await runScenario(
                    scenario, engine: engine, parameters: params
                )
                scenarioResults.append(result)

                let mem = await engine.memoryStats()
                peakMemoryMB = max(peakMemoryMB, mem.peakMB)

                let status = result.passed ? "PASS" : "FAIL"
                log("  [\(status)] \(scenario.id): \(scenario.description) — " +
                    "tools: \(String(format: "%.0f%%", result.summary.toolAccuracy * 100)), " +
                    "dups: \(String(format: "%.0f%%", result.summary.duplicateRate * 100)), " +
                    "\(String(format: "%.1f", result.summary.avgTokPerSec)) tok/s")
            }

            let aggregate = BenchmarkEvaluator.computeAggregate(
                scenarios: scenarioResults, peakMemoryMB: peakMemoryMB
            )

            let report = BenchmarkReport(
                metadata: BenchmarkMetadata(
                    date: ISO8601DateFormatter().string(from: Date()),
                    modelName: resolvedModelName,
                    hardware: hardwareString(),
                    parameters: params,
                    contextLimit: 20,
                    maxToolRounds: 3,
                    sweepLabel: paramLabel
                ),
                scenarios: scenarioResults,
                aggregate: aggregate
            )

            // Write report
            let paramHash = BenchmarkConfig.paramHash(for: params)
            let dateStr = dateString()
            let filename = "bench_\(dateStr)_\(config.sweep.rawValue)_\(paramHash).json"
            let reportURL = resultsDir.appendingPathComponent(filename)
            try report.write(to: reportURL)
            log("  Report written: \(reportURL.path)")

            // Print summary
            log("\n\(report.summaryString())")
        }

        log("\nBenchmark complete.")
        logFileHandle?.closeFile()
    }

    // MARK: - Run a Single Scenario

    private func runScenario(
        _ scenario: any BenchmarkScenario,
        engine: AgentEngine,
        parameters: AgentGenerateParameters
    ) async throws -> BenchmarkScenarioResult {
        log("  Running \(scenario.id): \(scenario.description) (\(scenario.turns.count) turns)")

        // Fresh temp directory for benchmark data
        let tempDir = config.outputDir
            .appendingPathComponent("data_\(scenario.id)_\(UUID().uuidString.prefix(8))")

        // Deprecated: uses empty LegacyToolRegistry. Epic 7 will rewrite benchmarks.
        let registry = LegacyToolRegistry(tools: [])
        let runner = AgentRunner(engine: engine, toolRegistry: registry, maxToolRounds: 3)

        // Transcript for full I/O debugging
        let transcript = BenchmarkTranscript()
        transcript.writeHeader(
            scenarioID: scenario.id,
            description: scenario.description,
            parameters: parameters
        )

        var messages: [AgentChatMessage] = []
        var turnResults: [BenchmarkTurnResult] = []
        var conversationToolHistory: [(name: String, arguments: [String: JSONValue], result: String)] = []

        for (turnIdx, expectation) in scenario.turns.enumerated() {
            let turnStart = CFAbsoluteTimeGetCurrent()

            // Append user message
            messages.append(.user(expectation.userMessage))

            // Build prompt with system prompt (memories are now read via file tools)
            let targetModelID = config.resolvedModelID
            let systemPrompt = SystemPromptBuilder.build(modelID: targetModelID)
            let contextLimit = 60
            let recentMessages = Array(messages.suffix(contextLimit))
            let maskedMessages = AgentChatMessage.withObservationMasking(recentMessages)
            let prompt: [AgentChatMessage] = [.system(systemPrompt)] + maskedMessages

            // Write turn start to transcript (skip raw prompt formatting — it's expensive)
            transcript.writeTurnStart(
                index: turnIdx, total: scenario.turns.count,
                userMessage: expectation.userMessage
            )

            // Run agent — track per-round output for transcript
            var responseText = ""
            var toolsCalled: [(name: String, arguments: [String: JSONValue])] = []
            var toolResults: [String] = []
            var info: AgentGeneration.Info?
            var toolRounds = 0

            // Per-round accumulation for transcript
            var roundText = ""
            var roundThinking: String? = nil
            var roundToolCalls: [(name: String, arguments: [String: JSONValue])] = []
            var currentRound = 1
            var inThinking = false
            var roundPromptTokens: Int? = nil
            var roundGenTokens: Int? = nil

            let stream = try runner.run(messages: prompt, parameters: parameters, emitRawPrompts: true)
            for try await event in stream {
                switch event {
                case .text(let chunk):
                    responseText += chunk
                    roundText += chunk

                case .thinkStart:
                    inThinking = true
                    roundThinking = ""

                case .thinking(let chunk):
                    roundThinking = (roundThinking ?? "") + chunk

                case .thinkEnd:
                    inThinking = false

                case .toolStart(let name, let arguments):
                    toolRounds += 1
                    toolsCalled.append((name: name, arguments: arguments))
                    roundToolCalls.append((name: name, arguments: arguments))

                case .toolResult(let name, let result):
                    toolResults.append(result)

                    // Write the completed round to transcript
                    transcript.writeRoundOutput(
                        round: currentRound,
                        rawOutput: roundText,
                        thinkingContent: roundThinking,
                        promptTokens: roundPromptTokens,
                        genTokens: roundGenTokens
                    )
                    if !roundToolCalls.isEmpty {
                        transcript.writeToolCalls(calls: roundToolCalls)
                    }

                    // Write tool execution
                    let matchingCall = roundToolCalls.last { $0.name == name }
                    transcript.writeToolExecution(
                        name: name,
                        arguments: matchingCall?.arguments ?? [:],
                        result: result
                    )

                    // Reset for next round
                    currentRound += 1
                    roundText = ""
                    roundThinking = nil
                    roundToolCalls = []
                    roundPromptTokens = nil
                    roundGenTokens = nil

                case .roundStart(let round, let rawPrompt, let msgCount):
                    transcript.writeRawPrompt(round: round, rawPrompt: rawPrompt, messageCount: msgCount)

                case .info(let i):
                    info = i
                    roundPromptTokens = i.promptTokenCount
                    roundGenTokens = i.generationTokenCount

                case .completed(let newMessages):
                    // Get the final assistant text (last assistant message)
                    if let lastAssistant = newMessages.last(where: { $0.role == .assistant }) {
                        responseText = lastAssistant.content
                    }
                    messages.append(contentsOf: newMessages)

                    // Always write final round output — even for text-only responses
                    // (an empty response is itself diagnostic)
                    transcript.writeRoundOutput(
                        round: currentRound,
                        rawOutput: roundText,
                        thinkingContent: roundThinking,
                        promptTokens: roundPromptTokens,
                        genTokens: roundGenTokens
                    )

                default:
                    break
                }
            }

            let latencyMs = (CFAbsoluteTimeGetCurrent() - turnStart) * 1000

            // Evaluate turn
            let turnResult = BenchmarkEvaluator.evaluate(
                turnIndex: turnIdx,
                expectation: expectation,
                toolsCalled: toolsCalled,
                toolResults: toolResults,
                assistantResponse: responseText,
                info: info,
                toolRounds: toolRounds,
                latencyMs: latencyMs,
                conversationToolHistory: conversationToolHistory
            )
            turnResults.append(turnResult)

            // Write turn result to transcript (exclude `respond` — it's infrastructure)
            let dataToolsCalled = toolsCalled.filter { $0.name != "respond" }
            transcript.writeTurnResult(
                passed: turnResult.passed,
                toolsCalled: dataToolsCalled.map(\.name),
                expectedTools: expectation.expectedTools,
                tokPerSec: info?.tokensPerSecond,
                latencyMs: latencyMs,
                checks: turnResult.checks
            )

            // Update conversation tool history (exclude respond — it's infrastructure)
            for (i, call) in dataToolsCalled.enumerated() {
                let result = i < toolResults.count ? toolResults[i] : ""
                conversationToolHistory.append((name: call.name, arguments: call.arguments, result: result))
            }

            let status = turnResult.passed ? "ok" : "FAIL"
            let tokSec = info.map { String(format: "%.1f", $0.tokensPerSecond) } ?? "?"
            log("    Turn \(turnIdx + 1)/\(scenario.turns.count) [\(status)] — " +
                "tools: \(dataToolsCalled.map(\.name)) — \(tokSec) tok/s — \(String(format: "%.0f", latencyMs))ms")
        }

        // Write scenario footer and save transcript
        let summary = BenchmarkEvaluator.computeScenarioSummary(turns: turnResults)
        let allPassed = turnResults.allSatisfy(\.passed)

        transcript.writeFooter(
            passed: allPassed,
            toolAccuracy: summary.toolAccuracy,
            duplicateRate: summary.duplicateRate
        )

        let transcriptDir = config.outputDir.appendingPathComponent("transcripts")
        try? FileManager.default.createDirectory(at: transcriptDir, withIntermediateDirectories: true)
        let transcriptURL = transcriptDir.appendingPathComponent("\(scenario.id).transcript.txt")
        try? transcript.write(to: transcriptURL)
        log("    Transcript: \(transcriptURL.path)")

        // Cleanup temp directory
        try? FileManager.default.removeItem(at: tempDir)

        return BenchmarkScenarioResult(
            id: scenario.id,
            description: scenario.description,
            turns: turnResults,
            passed: allPassed,
            summary: summary
        )
    }

    // MARK: - Model Resolution

    private func resolveModelDirectory() throws -> URL {
        if let dir = config.modelDir {
            return dir
        }

        // Resolve model ID to cache subdirectory
        let targetID = config.resolvedModelID
        guard let definition = ModelDefinition.all.first(where: { $0.id == targetID }),
              let cacheSub = definition.cacheSubdirectory else {
            throw BenchmarkError.modelNotFound(
                "Unknown model ID '\(targetID)'. Available: \(ModelDefinition.all.filter { $0.category == .agent }.map(\.id))"
            )
        }

        let cacheBase = URL.cachesDirectory.appendingPathComponent("mlx-audio")
        let modelDir = cacheBase.appendingPathComponent(cacheSub)

        guard FileManager.default.fileExists(atPath: modelDir.path) else {
            throw BenchmarkError.modelNotFound(
                "Model not found at \(modelDir.path). Download \(definition.displayName) from the Models page first."
            )
        }
        return modelDir
    }

    private var resolvedModelName: String {
        let targetID = config.resolvedModelID
        return ModelDefinition.all.first(where: { $0.id == targetID })?.displayName ?? targetID
    }

    // MARK: - Logging

    private func setupLogging() {
        try? FileManager.default.createDirectory(at: config.outputDir, withIntermediateDirectories: true)
        let logURL = config.outputDir.appendingPathComponent("latest.log")
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        logFileHandle = FileHandle(forWritingAtPath: logURL.path)
    }

    private func log(_ message: String) {
        let timestamp = ISO8601DateFormatter().string(from: Date())
        let line = "[\(timestamp)] \(message)\n"
        logger.info("\(message)")
        logFileHandle?.write(Data(line.utf8))
    }

    // MARK: - Helpers

    private func resolveResultsDirectory() -> URL {
        // Write results under the sandbox-accessible output dir.
        // bench.sh copies them to benchmarks/results/ in the repo afterwards.
        config.outputDir.appendingPathComponent("results")
    }

    private func hardwareString() -> String {
        var size: size_t = 0
        sysctlbyname("hw.model", nil, &size, nil, 0)
        var model = [CChar](repeating: 0, count: size)
        sysctlbyname("hw.model", &model, &size, nil, 0)
        let hwModel = String(cString: model)

        let memGB = ProcessInfo.processInfo.physicalMemory / (1024 * 1024 * 1024)
        return "\(hwModel), \(memGB)GB"
    }

    private func dateString() -> String {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd_HHmmss"
        return f.string(from: Date())
    }
}

enum BenchmarkError: LocalizedError {
    case modelNotFound(String)

    var errorDescription: String? {
        switch self {
        case .modelNotFound(let msg): return msg
        }
    }
}
