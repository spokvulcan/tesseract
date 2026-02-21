import Foundation
import MLXLMCommon

/// Evaluates a single turn's model output against expectations.
enum BenchmarkEvaluator {

    /// Evaluates one turn and returns the result.
    static func evaluate(
        turnIndex: Int,
        expectation: TurnExpectation,
        toolsCalled: [(name: String, arguments: [String: JSONValue])],
        toolResults: [String],
        assistantResponse: String,
        info: AgentGeneration.Info?,
        toolRounds: Int,
        latencyMs: Double,
        conversationToolHistory: [(name: String, arguments: [String: JSONValue], result: String)]
    ) -> BenchmarkTurnResult {
        // Filter out `respond` — it's infrastructure, not a data tool.
        // Benchmark expectations only care about data tools.
        let calledNames = toolsCalled.map(\.name).filter { $0 != "respond" }
        var details: [String] = []

        // 1. Tool correctness (skip if tools are optional for this turn)
        let toolsCorrect: Bool
        if expectation.toolsOptional {
            toolsCorrect = true
        } else {
            toolsCorrect = evaluateToolCorrectness(
                expected: expectation.expectedTools,
                called: calledNames,
                details: &details
            )
        }

        // 2. Duplicate detection (within this turn)
        let withinTurnDuplicates = countWithinTurnDuplicates(toolsCalled)

        // 3. Cross-turn duplicates: tool called when conversation already has identical successful result
        let crossTurnDuplicates = countCrossTurnDuplicates(
            toolsCalled, conversationHistory: conversationToolHistory
        )

        let totalDuplicates = withinTurnDuplicates + crossTurnDuplicates
        if totalDuplicates > 0 {
            details.append("\(totalDuplicates) duplicate tool call(s) (within-turn: \(withinTurnDuplicates), cross-turn: \(crossTurnDuplicates))")
        }

        // 4. Argument correctness
        let argsCorrect = evaluateArguments(
            expected: expectation.expectedArguments,
            toolsCalled: toolsCalled,
            details: &details
        )

        // 5. Response relevance
        let responseRelevant = evaluateResponseRelevance(
            expected: expectation.expectedSubstrings,
            response: assistantResponse,
            expectsClarification: expectation.expectsClarification,
            details: &details
        )

        // 6. Forbidden tools
        let noForbidden = evaluateForbiddenTools(
            forbidden: expectation.forbiddenTools,
            called: calledNames,
            details: &details
        )

        // 7. Hallucinated actions — model claims it did something without calling a tool
        let noHallucinated = evaluateNoHallucinatedActions(
            response: assistantResponse,
            toolsCalled: calledNames,
            details: &details
        )

        let checks = BenchmarkTurnChecks(
            toolsCorrect: toolsCorrect,
            duplicateToolCalls: totalDuplicates,
            argumentsCorrect: argsCorrect,
            responseRelevant: responseRelevant,
            noForbiddenTools: noForbidden,
            noHallucinatedActions: noHallucinated,
            details: details.isEmpty ? nil : details.joined(separator: "; ")
        )

        let performance = BenchmarkTurnPerformance(
            promptTokens: info?.promptTokenCount ?? 0,
            genTokens: info?.generationTokenCount ?? 0,
            tokPerSec: info?.tokensPerSecond ?? 0,
            latencyMs: latencyMs,
            toolRoundsUsed: toolRounds
        )

        let passed = toolsCorrect && argsCorrect && noForbidden && noHallucinated && withinTurnDuplicates == 0

        return BenchmarkTurnResult(
            turnIndex: turnIndex,
            userMessage: expectation.userMessage,
            assistantResponse: String(assistantResponse.prefix(500)),
            toolsCalled: calledNames,
            toolResults: toolResults.map { String($0.prefix(200)) },
            passed: passed,
            checks: checks,
            performance: performance
        )
    }

    // MARK: - Tool Correctness

    /// Checks that expected tools were called (order-independent, allows multi-call).
    private static func evaluateToolCorrectness(
        expected: [String],
        called: [String],
        details: inout [String]
    ) -> Bool {
        if expected.isEmpty {
            // No tools expected — model should not have called any
            if !called.isEmpty {
                details.append("Expected no tools, but called: \(called.joined(separator: ", "))")
                return false
            }
            return true
        }

        // Count expected occurrences
        var expectedCounts: [String: Int] = [:]
        for name in expected { expectedCounts[name, default: 0] += 1 }

        var calledCounts: [String: Int] = [:]
        for name in called { calledCounts[name, default: 0] += 1 }

        var correct = true
        for (name, count) in expectedCounts {
            let actualCount = calledCounts[name] ?? 0
            if actualCount < count {
                details.append("Expected \(count)x \(name), got \(actualCount)")
                correct = false
            }
        }

        return correct
    }

    // MARK: - Duplicate Detection

    /// Counts same-tool-same-args duplicates within a single turn.
    private static func countWithinTurnDuplicates(
        _ calls: [(name: String, arguments: [String: JSONValue])]
    ) -> Int {
        var seen: [String] = []
        var duplicates = 0

        for call in calls {
            let key = canonicalKey(name: call.name, arguments: call.arguments)
            if seen.contains(key) {
                duplicates += 1
            } else {
                seen.append(key)
            }
        }
        return duplicates
    }

    /// Read-only tools whose results change as data is mutated.
    /// Re-calling these is expected behavior, not a duplicate.
    private static let readOnlyTools: Set<String> = [
        "goal_list", "task_list", "mood_list", "habit_status",
    ]

    /// Counts tool calls that duplicate a previous successful call in conversation history.
    /// Read-only listing/status tools are excluded — their results change after mutations,
    /// so re-calling them is correct behavior, not waste.
    private static func countCrossTurnDuplicates(
        _ calls: [(name: String, arguments: [String: JSONValue])],
        conversationHistory: [(name: String, arguments: [String: JSONValue], result: String)]
    ) -> Int {
        let historyKeys = Set(conversationHistory.compactMap { entry -> String? in
            // Only count successful past calls (not errors)
            guard !entry.result.hasPrefix("Error:") else { return nil }
            // Skip read-only tools — re-querying is expected
            guard !readOnlyTools.contains(entry.name) else { return nil }
            return canonicalKey(name: entry.name, arguments: entry.arguments)
        })

        var duplicates = 0
        for call in calls {
            guard !readOnlyTools.contains(call.name) else { continue }
            let key = canonicalKey(name: call.name, arguments: call.arguments)
            if historyKeys.contains(key) {
                duplicates += 1
            }
        }
        return duplicates
    }

    private static func canonicalKey(name: String, arguments: [String: JSONValue]) -> String {
        let sortedArgs = arguments.sorted { $0.key < $1.key }
            .map { "\($0.key)=\(stringValue($0.value))" }
            .joined(separator: ",")
        return "\(name)(\(sortedArgs))"
    }

    private static func stringValue(_ value: JSONValue) -> String {
        switch value {
        case .string(let s): return s.lowercased()
        case .int(let i): return String(i)
        case .double(let d): return String(d)
        case .bool(let b): return String(b)
        case .null: return "null"
        default: return "?"
        }
    }

    // MARK: - Argument Validation

    private static func evaluateArguments(
        expected: [String: [String: String]],
        toolsCalled: [(name: String, arguments: [String: JSONValue])],
        details: inout [String]
    ) -> Bool {
        guard !expected.isEmpty else { return true }

        var correct = true
        for (toolName, expectedArgs) in expected {
            guard let call = toolsCalled.first(where: { $0.name == toolName }) else {
                // Tool wasn't called — already flagged in tool correctness
                continue
            }

            for (key, expectedSubstring) in expectedArgs {
                let actualValue = call.arguments.string(for: key) ?? ""
                if !actualValue.lowercased().contains(expectedSubstring.lowercased()) {
                    details.append("\(toolName).\(key): expected '\(expectedSubstring)' in '\(actualValue)'")
                    correct = false
                }
            }
        }
        return correct
    }

    // MARK: - Response Relevance

    private static func evaluateResponseRelevance(
        expected: [String],
        response: String,
        expectsClarification: Bool,
        details: inout [String]
    ) -> Bool {
        let lower = response.lowercased()
        var relevant = true

        for substring in expected {
            if !lower.contains(substring.lowercased()) {
                details.append("Response missing expected substring: '\(substring)'")
                relevant = false
            }
        }

        if expectsClarification {
            let clarificationSignals = ["?", "what", "when", "which", "could you", "can you",
                                        "please", "more details", "specify", "clarify",
                                        "let me know", "tell me"]
            let hasClarification = clarificationSignals.contains { lower.contains($0) }
            if !hasClarification {
                details.append("Expected clarification question, but response seems definitive")
                // Don't fail — this is a soft check
            }
        }

        return relevant
    }

    // MARK: - Forbidden Tools

    private static func evaluateForbiddenTools(
        forbidden: [String],
        called: [String],
        details: inout [String]
    ) -> Bool {
        guard !forbidden.isEmpty else { return true }

        let forbiddenSet = Set(forbidden)
        let violating = called.filter { forbiddenSet.contains($0) }
        if !violating.isEmpty {
            details.append("Forbidden tool(s) called: \(violating.joined(separator: ", "))")
            return false
        }
        return true
    }

    // MARK: - Hallucinated Action Detection

    /// Past-tense action phrases that indicate the model claims it completed a tool action.
    /// Patterns must be specific enough to avoid matching offers ("Would you like me to set a reminder?").
    private static let actionClaims: [(pattern: String, tool: String)] = [
        // habit_create — past-tense claims
        ("habit created", "habit_create"),
        ("created a habit", "habit_create"),
        ("created habit", "habit_create"),
        // goal_create
        ("goal created", "goal_create"),
        ("created a goal", "goal_create"),
        ("created goal", "goal_create"),
        // task_create
        ("task created", "task_create"),
        ("created a task", "task_create"),
        ("created task", "task_create"),
        // reminder_set — only past-tense; "set a reminder" is ambiguous (offer vs claim)
        ("reminder is set", "reminder_set"),
        ("reminder has been set", "reminder_set"),
        ("i've set a reminder", "reminder_set"),
        ("i set a reminder", "reminder_set"),
        // memory_save
        ("saved to memory", "memory_save"),
        ("i've saved", "memory_save"),
        ("i saved", "memory_save"),
        // mood_log
        ("logged your mood", "mood_log"),
        ("mood logged", "mood_log"),
        // habit_log
        ("logged your habit", "habit_log"),
        ("habit logged", "habit_log"),
        // task_complete
        ("marked as complete", "task_complete"),
        ("task completed", "task_complete"),
    ]

    /// Phrases that signal the model is offering/suggesting rather than claiming completion.
    private static let offerPrefixes = [
        "would you like me to",
        "want me to",
        "shall i",
        "i can ",
        "i could ",
        "like me to",
    ]

    /// Detects when the model claims it performed an action without calling the tool.
    private static func evaluateNoHallucinatedActions(
        response: String,
        toolsCalled: [String],
        details: inout [String]
    ) -> Bool {
        let lower = response.lowercased()
        let calledSet = Set(toolsCalled)

        for claim in actionClaims {
            guard lower.contains(claim.pattern) && !calledSet.contains(claim.tool) else { continue }

            // Check if the pattern appears inside an offer/suggestion context
            if let range = lower.range(of: claim.pattern) {
                let lineStart = lower[..<range.lowerBound].lastIndex(of: "\n").map { lower.index(after: $0) } ?? lower.startIndex
                let prefix = String(lower[lineStart..<range.lowerBound])
                if offerPrefixes.contains(where: { prefix.contains($0) }) {
                    continue  // This is an offer, not a claim
                }
            }

            details.append("Hallucinated action: response says '\(claim.pattern)' but \(claim.tool) was never called")
            return false
        }
        return true
    }

    // MARK: - Aggregate Scoring

    static func computeScenarioSummary(turns: [BenchmarkTurnResult]) -> BenchmarkScenarioSummary {
        let totalTools = turns.reduce(0) { $0 + max($1.toolsCalled.count, 1) }
        let correctTools = turns.filter { $0.checks.toolsCorrect }.count
        let toolAccuracy = turns.isEmpty ? 1.0 : Double(correctTools) / Double(turns.count)

        let totalDuplicates = turns.reduce(0) { $0 + $1.checks.duplicateToolCalls }
        let totalCalls = turns.reduce(0) { $0 + $1.toolsCalled.count }
        let duplicateRate = totalCalls == 0 ? 0.0 : Double(totalDuplicates) / Double(totalCalls)

        let tokSpeeds = turns.compactMap { $0.performance.tokPerSec > 0 ? $0.performance.tokPerSec : nil }
        let avgTokPerSec = tokSpeeds.isEmpty ? 0 : tokSpeeds.reduce(0, +) / Double(tokSpeeds.count)

        let latencies = turns.map(\.performance.latencyMs)
        let avgLatency = latencies.isEmpty ? 0 : latencies.reduce(0, +) / Double(latencies.count)

        return BenchmarkScenarioSummary(
            toolAccuracy: toolAccuracy,
            duplicateRate: duplicateRate,
            avgTokPerSec: avgTokPerSec,
            avgLatencyMs: avgLatency
        )
    }

    static func computeAggregate(
        scenarios: [BenchmarkScenarioResult],
        peakMemoryMB: Float
    ) -> BenchmarkAggregate {
        let passed = scenarios.filter(\.passed).count

        let allTurns = scenarios.flatMap(\.turns)
        let toolTurns = allTurns.filter { !$0.toolsCalled.isEmpty || !$0.checks.toolsCorrect }
        let correctToolTurns = toolTurns.filter { $0.checks.toolsCorrect }
        let overallToolAccuracy = toolTurns.isEmpty ? 1.0 : Double(correctToolTurns.count) / Double(toolTurns.count)

        let totalDuplicates = allTurns.reduce(0) { $0 + $1.checks.duplicateToolCalls }
        let totalCalls = allTurns.reduce(0) { $0 + $1.toolsCalled.count }
        let duplicateRate = totalCalls == 0 ? 0.0 : Double(totalDuplicates) / Double(totalCalls)

        let tokSpeeds = allTurns.compactMap { $0.performance.tokPerSec > 0 ? $0.performance.tokPerSec : nil }
        let avgTokPerSec = tokSpeeds.isEmpty ? 0 : tokSpeeds.reduce(0, +) / Double(tokSpeeds.count)

        let latencies = allTurns.map(\.performance.latencyMs).sorted()
        let p50 = percentile(latencies, p: 0.50)
        let p95 = percentile(latencies, p: 0.95)

        return BenchmarkAggregate(
            passedScenarios: passed,
            totalScenarios: scenarios.count,
            overallToolAccuracy: overallToolAccuracy,
            duplicateRate: duplicateRate,
            avgTokPerSec: avgTokPerSec,
            p50LatencyMs: p50,
            p95LatencyMs: p95,
            peakMemoryMB: peakMemoryMB
        )
    }

    private static func percentile(_ sorted: [Double], p: Double) -> Double {
        guard !sorted.isEmpty else { return 0 }
        let index = p * Double(sorted.count - 1)
        let lower = Int(index)
        let upper = min(lower + 1, sorted.count - 1)
        let fraction = index - Double(lower)
        return sorted[lower] + fraction * (sorted[upper] - sorted[lower])
    }
}
