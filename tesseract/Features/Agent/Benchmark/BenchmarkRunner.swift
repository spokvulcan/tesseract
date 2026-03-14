import Foundation
import MLX
import MLXLMCommon
import os

/// Orchestrates the full benchmark run: load model, run scenarios, write reports.
///
/// Uses the new `Agent` class with built-in file tools (read, write, edit, ls)
/// and a sandbox-isolated temporary directory per scenario.
@MainActor
final class BenchmarkRunner {

    private let config: BenchmarkConfig
    private let logger = Logger(subsystem: "app.tesseract.agent", category: "benchmark")
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

        for (configIdx, baseParams) in paramConfigs.enumerated() {
            // Cap maxTokens per round to prevent runaway generation
            var params = baseParams
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
                    date: Self.iso8601.string(from: Date()),
                    modelName: resolvedModelName,
                    hardware: hardwareString(),
                    parameters: params,
                    promptProfile: config.promptProfile,
                    contextLimit: 0,  // No fixed limit — managed by compaction
                    maxToolRounds: 0,  // No cap — loop runs until no more tool calls
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

        // 1. Create isolated sandbox directory for this scenario
        let tempDir = config.outputDir
            .appendingPathComponent("data_\(scenario.id)_\(UUID().uuidString.prefix(8))")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)

        // Seed initial data files
        seedBenchmarkData(at: tempDir)
        seedScenarioFiles(at: tempDir, files: scenario.benchmarkFiles)

        // Seed skill files for this scenario
        let skillMetadata = seedSkillFiles(at: tempDir, skills: scenario.benchmarkSkills)

        let sandbox = PathSandbox(root: tempDir)
        let tools = BuiltInToolFactory.createAll(sandbox: sandbox)

        // 2. Build system prompt with file-workflow instructions
        let systemPrompt = buildBenchmarkSystemPrompt(
            tools: tools, agentRoot: tempDir.path, skills: skillMetadata
        )

        // 3. Create performance metrics collector
        let metrics = BenchmarkMetrics()

        // 4. Create Agent with benchmarking generate function
        let loopConfig = AgentLoopConfig(
            model: AgentModelRef(id: config.resolvedModelID),
            convertToLlm: { msgs in msgs.compactMap { $0.toLLMMessage() } },
            contextTransform: nil,
            getSteeringMessages: nil,
            getFollowUpMessages: nil
        )

        let generateFn = makeBenchmarkGenerate(
            engine: engine, parameters: parameters, metrics: metrics
        )

        let agent = Agent(
            config: loopConfig,
            systemPrompt: systemPrompt,
            tools: tools,
            generate: generateFn
        )

        // 5. Transcript for full I/O debugging
        let transcript = BenchmarkTranscript()
        transcript.writeHeader(
            scenarioID: scenario.id,
            description: scenario.description,
            parameters: parameters
        )

        // 6. Run each turn
        var turnResults: [BenchmarkTurnResult] = []
        var conversationToolHistory: [(name: String, arguments: [String: JSONValue], result: String)] = []

        for (turnIdx, expectation) in scenario.turns.enumerated() {
            let turnStart = CFAbsoluteTimeGetCurrent()

            // Collect events for this turn
            let turnCollector = TurnEventCollector()
            let unsubscribe = agent.subscribe { event in
                turnCollector.handleEvent(event)
            }

            // Write turn start to transcript
            transcript.writeTurnStart(
                index: turnIdx, total: scenario.turns.count,
                userMessage: expectation.userMessage
            )

            for file in expectation.preTurnFiles {
                seedScenarioFiles(at: tempDir, files: [file])
                transcript.writePreTurnMutation(path: file.relativePath)
            }

            // Send user message and wait for agent to finish
            let userMsg = UserMessage.create(expectation.userMessage)
            agent.prompt(userMsg)
            await agent.waitForIdle()

            // Unsubscribe
            unsubscribe()

            // Drain metrics
            let infos = metrics.drain()
            let aggregateInfo = Self.aggregateInfos(infos)

            let latencyMs = (CFAbsoluteTimeGetCurrent() - turnStart) * 1000

            // Extract collected data from events
            let attemptedToolCalls = turnCollector.attemptedToolCalls
            let toolsCalled = turnCollector.toolsCalled
            let toolResults = turnCollector.toolResults
            let responseText = turnCollector.assistantText
            let toolRounds = turnCollector.toolRoundCount

            // Write tool executions to transcript
            for (round, roundData) in turnCollector.rounds.enumerated() {
                // Write round output (assistant text for this round)
                transcript.writeRoundOutput(
                    round: round + 1,
                    rawOutput: roundData.text,
                    thinkingContent: roundData.thinking,
                    promptTokens: nil,
                    genTokens: nil
                )

                // Write tool calls and results
                for execution in roundData.toolExecutions {
                    transcript.writeToolExecution(
                        name: execution.name,
                        arguments: execution.arguments,
                        result: execution.result
                    )
                }
            }

            // Write final round if there's trailing text after last tool call
            if let trailingText = turnCollector.trailingText, !trailingText.isEmpty {
                transcript.writeRoundOutput(
                    round: turnCollector.rounds.count + 1,
                    rawOutput: trailingText,
                    thinkingContent: nil,
                    promptTokens: nil,
                    genTokens: nil
                )
            }

            // Evaluate turn
            let turnResult = BenchmarkEvaluator.evaluate(
                turnIndex: turnIdx,
                expectation: expectation,
                attemptedToolCalls: attemptedToolCalls,
                executedToolCalls: toolsCalled,
                toolResults: toolResults.map(\.result),
                assistantResponse: responseText,
                info: aggregateInfo,
                toolRounds: toolRounds,
                latencyMs: latencyMs,
                conversationToolHistory: conversationToolHistory,
                malformedToolCalls: turnCollector.malformedToolCalls,
                sandboxRoot: tempDir
            )
            turnResults.append(turnResult)

            // Write turn result to transcript
            transcript.writeTurnResult(
                passed: turnResult.passed,
                attemptedTools: turnResult.attemptedTools,
                toolsCalled: toolsCalled.map(\.name),
                expectedTools: expectation.expectedToolCalls.map(\.name),
                tokPerSec: aggregateInfo?.tokensPerSecond,
                latencyMs: latencyMs,
                checks: turnResult.checks
            )

            // Update conversation tool history
            for (i, call) in toolsCalled.enumerated() {
                let result = i < toolResults.count ? toolResults[i].result : ""
                conversationToolHistory.append(
                    (name: call.name, arguments: call.arguments, result: result)
                )
            }

            let status = turnResult.passed ? "ok" : "FAIL"
            let tokSec = aggregateInfo.map { String(format: "%.1f", $0.tokensPerSecond) } ?? "?"
            log("    Turn \(turnIdx + 1)/\(scenario.turns.count) [\(status)] — " +
                "tools: \(toolsCalled.map(\.name)) — \(tokSec) tok/s — \(String(format: "%.0f", latencyMs))ms")
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

    // MARK: - Benchmark Setup

    /// Seeds the benchmark sandbox with initial data files.
    private func seedBenchmarkData(at directory: URL) {
        let memoriesContent = """
            # Memories

            - Alex prefers dark mode for all apps
            - Alex's birthday is March 15th
            """

        let tasksContent = """
            # Tasks

            - [ ] Buy groceries
            - [ ] Schedule dentist appointment
            - [ ] Call mom
            """

        do {
            try memoriesContent.write(
                to: directory.appendingPathComponent("memories.md"),
                atomically: true, encoding: .utf8
            )
            try tasksContent.write(
                to: directory.appendingPathComponent("tasks.md"),
                atomically: true, encoding: .utf8
            )
        } catch {
            log("WARNING: Failed to seed benchmark data: \(error.localizedDescription)")
        }
    }

    /// Writes scenario-specific files into the sandbox, overwriting existing files when needed.
    private func seedScenarioFiles(at directory: URL, files: [BenchmarkSeedFile]) {
        guard !files.isEmpty else { return }

        for file in files {
            let url = directory.appendingPathComponent(file.relativePath)
            let parent = url.deletingLastPathComponent()
            do {
                try FileManager.default.createDirectory(
                    at: parent, withIntermediateDirectories: true
                )
                try file.content.write(to: url, atomically: true, encoding: .utf8)
            } catch {
                log("WARNING: Failed to seed benchmark file '\(file.relativePath)': \(error.localizedDescription)")
            }
        }
    }

    /// Seeds skill files into the sandbox and returns SkillMetadata for prompt assembly.
    private func seedSkillFiles(at directory: URL, skills: [BenchmarkSkill]) -> [SkillMetadata] {
        skills.compactMap { skill in
            let fileURL = directory.appendingPathComponent(skill.relativePath)
            let parentDir = fileURL.deletingLastPathComponent()
            do {
                try FileManager.default.createDirectory(
                    at: parentDir, withIntermediateDirectories: true
                )
                try skill.content.write(to: fileURL, atomically: true, encoding: .utf8)
                return SkillMetadata(
                    name: skill.name,
                    description: skill.description,
                    filePath: fileURL.path,
                    disableModelInvocation: false
                )
            } catch {
                log("WARNING: Failed to seed skill '\(skill.name)': \(error.localizedDescription)")
                return nil
            }
        }
    }

    /// Builds a system prompt tailored for benchmarks with file-workflow instructions.
    private func buildBenchmarkSystemPrompt(
        tools: [AgentToolDefinition],
        agentRoot: String,
        skills: [SkillMetadata] = []
    ) -> String {
        let contextLoader = ContextLoader(agentRoot: URL(fileURLWithPath: agentRoot))
        let loadedContext = contextLoader.load()
        let defaultPrompt: String = switch config.promptProfile {
        case .benchmark:
            Self.benchmarkCorePrompt
        case .production:
            SystemPromptAssembler.defaultCorePrompt
        }

        return SystemPromptAssembler.assemble(
            defaultPrompt: defaultPrompt,
            loadedContext: loadedContext,
            skills: skills,
            tools: tools,
            agentRoot: agentRoot
        )
    }

    /// Benchmark-specific core prompt that includes file-workflow instructions.
    private static let benchmarkCorePrompt = """
        You are an expert local assistant operating inside Tesseract, a tool-calling agent harness.
        You help users by reading files, editing files, writing files, and using other tools provided by the current package or project.

        Available tools:
        - read: Read file contents
        - write: Create or overwrite files
        - edit: Make surgical edits to files (find exact text and replace)
        - ls: List files and directories

        Guidelines:
        - Use ls to discover files and directories
        - Use read to examine files before editing and writing
        - Use write only if you read the file first and it is empty or does not exist, otherwise use edit
        - Use edit for precise changes (old_text must match exactly)
        - Be concise in your responses

        You manage the user's personal data using files:
        - memories.md: User's memories and important facts (append new entries as bullet points)
        - tasks.md: User's task list (markdown checkboxes: - [ ] for pending, - [x] for done)

        When asked to remember something: read memories.md, then edit to add the new memory as a bullet point.
        When asked to create a task: read tasks.md, then edit to add a new checkbox item (- [ ] task).
        When asked to complete a task: read tasks.md, then edit to change [ ] to [x] for that task.
        When asked about memories or tasks: read the appropriate file.
        """

    // MARK: - Generate Function

    /// Creates an instrumented generate function that captures performance metrics.
    private func makeBenchmarkGenerate(
        engine: AgentEngine,
        parameters: AgentGenerateParameters,
        metrics: BenchmarkMetrics
    ) -> LLMGenerateFunction {
        // Capture parameters for all generations in this scenario
        return { [weak engine] systemPrompt, messages, tools, _ in
            let (stream, continuation) = AsyncThrowingStream.makeStream(of: AgentGeneration.self)
            let task = Task { @MainActor in
                guard let engine else {
                    continuation.finish()
                    return
                }
                do {
                    let engineStream = try engine.generate(
                        systemPrompt: systemPrompt,
                        messages: messages,
                        tools: tools,
                        parameters: parameters
                    )
                    for try await generation in engineStream {
                        // Capture info events for performance metrics
                        if case .info(let info) = generation {
                            metrics.append(info)
                        }
                        continuation.yield(generation)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
            return stream
        }
    }

    /// Aggregates multiple generation info events into a single summary.
    private static func aggregateInfos(_ infos: [AgentGeneration.Info]) -> AgentGeneration.Info? {
        guard !infos.isEmpty else { return nil }
        let totalPromptTokens = infos.reduce(0) { $0 + $1.promptTokenCount }
        let totalGenTokens = infos.reduce(0) { $0 + $1.generationTokenCount }
        let totalPromptTime = infos.reduce(0.0) { $0 + $1.promptTime }
        let totalGenTime = infos.reduce(0.0) { $0 + $1.generateTime }
        return AgentGeneration.Info(
            promptTokenCount: totalPromptTokens,
            generationTokenCount: totalGenTokens,
            promptTime: totalPromptTime,
            generateTime: totalGenTime
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

        let cacheBase = ModelDownloadManager.modelStorageURL
        var modelDir = cacheBase.appendingPathComponent(cacheSub)
        if let prefix = definition.pathPrefix {
            modelDir = modelDir.appendingPathComponent(prefix)
        }

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

    private static let iso8601 = ISO8601DateFormatter()

    private func log(_ message: String) {
        let timestamp = Self.iso8601.string(from: Date())
        let line = "[\(timestamp)] \(message)\n"
        logger.info("\(message)")
        logFileHandle?.write(Data(line.utf8))
    }

    // MARK: - Helpers

    private func resolveResultsDirectory() -> URL {
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
        DebugPaths.timestamp()
    }
}

// MARK: - BenchmarkMetrics

/// Thread-safe performance metrics collector for benchmark generation.
/// Written from the generate closure (Task context), read on MainActor.
@MainActor
final class BenchmarkMetrics {
    private var infos: [AgentGeneration.Info] = []

    func append(_ info: AgentGeneration.Info) {
        infos.append(info)
    }

    func drain() -> [AgentGeneration.Info] {
        let result = infos
        infos.removeAll()
        return result
    }

}

// MARK: - TurnEventCollector

/// Collects agent events for a single benchmark turn to extract tool calls,
/// results, assistant text, and round structure.
@MainActor
final class TurnEventCollector {

    struct ToolExecution {
        let name: String
        let arguments: [String: JSONValue]
        let result: String
    }

    struct RoundData {
        var text: String = ""
        var thinking: String?
        var toolExecutions: [ToolExecution] = []
    }

    private(set) var toolsCalled: [(name: String, arguments: [String: JSONValue])] = []
    private(set) var attemptedToolCalls: [(name: String, arguments: [String: JSONValue])] = []
    private(set) var toolResults: [(name: String, result: String)] = []
    private(set) var assistantText: String = ""
    private(set) var rounds: [RoundData] = []
    private(set) var trailingText: String?
    private(set) var toolRoundCount: Int = 0
    private(set) var malformedToolCalls: [String] = []

    private var currentRound = RoundData()
    private var pendingToolArgs: [String: [String: JSONValue]] = [:]  // toolCallId -> args

    func handleEvent(_ event: AgentEvent) {
        switch event {
        case .messageUpdate(let message, let delta):
            if let textDelta = delta.textDelta {
                currentRound.text += textDelta
            }
            if let thinkingDelta = delta.thinkingDelta {
                currentRound.thinking = (currentRound.thinking ?? "") + thinkingDelta
            }
            // Update full assistant text from the latest message
            assistantText = message.content
            attemptedToolCalls = message.toolCalls.map { call in
                (
                    name: call.name,
                    arguments: Self.parseArguments(from: call.argumentsJSON)
                )
            }

        case .toolExecutionStart(let toolCallId, let name, let argsJSON):
            toolRoundCount += 1
            // Parse arguments
            let args = Self.parseArguments(from: argsJSON)
            toolsCalled.append((name: name, arguments: args))
            pendingToolArgs[toolCallId] = args

        case .toolExecutionEnd(let toolCallId, let name, let result, _):
            let resultText = result.content.textContent
            toolResults.append((name: name, result: resultText))

            let args = pendingToolArgs.removeValue(forKey: toolCallId) ?? [:]
            currentRound.toolExecutions.append(
                ToolExecution(name: name, arguments: args, result: resultText)
            )

        case .turnEnd(let message, _, _):
            attemptedToolCalls = message.toolCalls.map { call in
                (
                    name: call.name,
                    arguments: Self.parseArguments(from: call.argumentsJSON)
                )
            }
            // Finalize the current round
            if !currentRound.text.isEmpty || !currentRound.toolExecutions.isEmpty {
                rounds.append(currentRound)
            }
            currentRound = RoundData()

        case .agentEnd:
            // Capture any trailing text
            if !currentRound.text.isEmpty {
                trailingText = currentRound.text
            }

        case .malformedToolCall(let raw):
            malformedToolCalls.append(raw)

        default:
            break
        }
    }

    private static func parseArguments(from json: String) -> [String: JSONValue] {
        guard let data = json.data(using: .utf8),
              let parsed = try? JSONDecoder().decode([String: JSONValue].self, from: data) else {
            return [:]
        }
        return parsed
    }
}

// MARK: - BenchmarkError

enum BenchmarkError: LocalizedError {
    case modelNotFound(String)

    var errorDescription: String? {
        switch self {
        case .modelNotFound(let msg): return msg
        }
    }
}
