import Foundation
import MLXLMCommon

/// Evaluates a single turn's model output against expectations.
enum BenchmarkEvaluator {

    static func evaluate(
        turnIndex: Int,
        expectation: TurnExpectation,
        attemptedToolCalls: [(name: String, arguments: [String: JSONValue])],
        executedToolCalls: [(name: String, arguments: [String: JSONValue])],
        toolResults: [String],
        assistantResponse: String,
        info: AgentGeneration.Info?,
        toolRounds: Int,
        latencyMs: Double,
        conversationToolHistory: [(name: String, arguments: [String: JSONValue], result: String)],
        malformedToolCalls: [String],
        sandboxRoot: URL
    ) -> BenchmarkTurnResult {
        let calledNames = executedToolCalls.map(\.name)
        let attemptedNames = attemptedToolCalls.map(\.name)
        var details: [String] = []

        let toolsCorrect = evaluateToolCorrectness(
            expectation: expectation,
            executed: executedToolCalls,
            details: &details
        )

        let argsCorrect = evaluateArguments(
            expectation: expectation,
            executed: executedToolCalls,
            details: &details
        )

        let withinTurnDuplicates = countWithinTurnDuplicates(executedToolCalls)
        let crossTurnDuplicates = countCrossTurnDuplicates(
            executedToolCalls, conversationHistory: conversationToolHistory
        )
        let totalDuplicates = withinTurnDuplicates + crossTurnDuplicates
        if totalDuplicates > 0 {
            details.append(
                "\(totalDuplicates) duplicate tool call(s) (within-turn: \(withinTurnDuplicates), cross-turn: \(crossTurnDuplicates))"
            )
        }

        let responseRelevant = evaluateResponseRelevance(
            expected: expectation.expectedSubstrings,
            response: assistantResponse,
            details: &details
        )

        let clarificationAsked = evaluateClarification(
            required: expectation.expectsClarification,
            response: assistantResponse,
            details: &details
        )

        let noForbidden = evaluateForbiddenTools(
            forbidden: expectation.forbiddenTools,
            attempted: attemptedNames,
            executed: calledNames,
            details: &details
        )

        let noInvalidToolCalls = evaluateInvalidToolCalls(
            attempted: attemptedToolCalls,
            executed: executedToolCalls,
            malformedToolCalls: malformedToolCalls,
            details: &details
        )

        let fileAssertionsPassed = evaluateFileAssertions(
            assertions: expectation.fileAssertions,
            sandboxRoot: sandboxRoot,
            details: &details
        )

        let checks = BenchmarkTurnChecks(
            toolsCorrect: toolsCorrect,
            duplicateToolCalls: totalDuplicates,
            argumentsCorrect: argsCorrect,
            responseRelevant: responseRelevant,
            clarificationAsked: clarificationAsked,
            noForbiddenTools: noForbidden,
            noInvalidToolCalls: noInvalidToolCalls,
            malformedToolCallCount: malformedToolCalls.count,
            fileAssertionsPassed: fileAssertionsPassed,
            details: details.isEmpty ? nil : details.joined(separator: "; ")
        )

        let performance = BenchmarkTurnPerformance(
            promptTokens: info?.promptTokenCount ?? 0,
            genTokens: info?.generationTokenCount ?? 0,
            tokPerSec: info?.tokensPerSecond ?? 0,
            latencyMs: latencyMs,
            toolRoundsUsed: toolRounds
        )

        let passed = toolsCorrect
            && argsCorrect
            && responseRelevant
            && clarificationAsked
            && noForbidden
            && noInvalidToolCalls
            && fileAssertionsPassed
            && withinTurnDuplicates == 0

        return BenchmarkTurnResult(
            turnIndex: turnIndex,
            userMessage: expectation.userMessage,
            assistantResponse: String(assistantResponse.prefix(500)),
            attemptedTools: attemptedNames,
            toolsCalled: calledNames,
            toolResults: toolResults.map { String($0.prefix(200)) },
            passed: passed,
            checks: checks,
            performance: performance
        )
    }

    // MARK: - Tool Correctness

    private static func evaluateToolCorrectness(
        expectation: TurnExpectation,
        executed: [(name: String, arguments: [String: JSONValue])],
        details: inout [String]
    ) -> Bool {
        switch expectation.toolMatchMode {
        case .noTools:
            if !executed.isEmpty {
                details.append("Expected no executed tools, but called: \(executed.map(\.name).joined(separator: ", "))")
                return false
            }
            return true

        case .containsSequence:
            return sequenceMatch(
                expected: expectation.expectedToolCalls,
                actual: executed,
                requireExactLength: false,
                validateArguments: false,
                details: &details
            )

        case .exactSequence:
            return sequenceMatch(
                expected: expectation.expectedToolCalls,
                actual: executed,
                requireExactLength: true,
                validateArguments: false,
                details: &details
            )
        }
    }

    // MARK: - Duplicate Detection

    private static func countWithinTurnDuplicates(
        _ calls: [(name: String, arguments: [String: JSONValue])]
    ) -> Int {
        var seen = Set<String>()
        var duplicates = 0

        for call in calls {
            let key = canonicalKey(name: call.name, arguments: call.arguments)
            if !seen.insert(key).inserted {
                duplicates += 1
            }
        }
        return duplicates
    }

    private static let readOnlyTools: Set<String> = [
        "read", "ls",
    ]

    private static func countCrossTurnDuplicates(
        _ calls: [(name: String, arguments: [String: JSONValue])],
        conversationHistory: [(name: String, arguments: [String: JSONValue], result: String)]
    ) -> Int {
        let historyKeys = Set(conversationHistory.compactMap { entry -> String? in
            guard !entry.result.hasPrefix("Error:") else { return nil }
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
        expectation: TurnExpectation,
        executed: [(name: String, arguments: [String: JSONValue])],
        details: inout [String]
    ) -> Bool {
        switch expectation.toolMatchMode {
        case .noTools:
            return true
        case .containsSequence:
            return sequenceMatch(
                expected: expectation.expectedToolCalls,
                actual: executed,
                requireExactLength: false,
                validateArguments: true,
                details: &details
            )
        case .exactSequence:
            return sequenceMatch(
                expected: expectation.expectedToolCalls,
                actual: executed,
                requireExactLength: true,
                validateArguments: true,
                details: &details
            )
        }
    }

    private static func sequenceMatch(
        expected: [ExpectedToolCall],
        actual: [(name: String, arguments: [String: JSONValue])],
        requireExactLength: Bool,
        validateArguments: Bool,
        details: inout [String]
    ) -> Bool {
        if requireExactLength && expected.count != actual.count {
            details.append("Expected exactly \(expected.count) tool call(s), got \(actual.count)")
        }

        guard !expected.isEmpty else { return requireExactLength ? actual.isEmpty : true }

        var actualIndex = 0
        var matched = 0

        for (expectedIndex, expectedCall) in expected.enumerated() {
            var foundIndex: Int?
            while actualIndex < actual.count {
                let actualCall = actual[actualIndex]
                if actualCall.name == expectedCall.name {
                    foundIndex = actualIndex
                    break
                }
                if requireExactLength {
                    details.append(
                        "Tool order mismatch at position \(expectedIndex + 1): expected \(expectedCall.name), got \(actualCall.name)"
                    )
                    return false
                }
                actualIndex += 1
            }

            guard let matchIndex = foundIndex else {
                details.append("Missing expected tool call \(expectedIndex + 1): \(expectedCall.name)")
                return false
            }

            if validateArguments {
                let actualCall = actual[matchIndex]
                for (key, expectedSubstring) in expectedCall.arguments {
                    let actualValue = actualCall.arguments.string(for: key) ?? ""
                    if !actualValue.lowercased().contains(expectedSubstring.lowercased()) {
                        details.append(
                            "\(expectedCall.name).\(key) at call \(expectedIndex + 1): expected '\(expectedSubstring)' in '\(actualValue)'"
                        )
                        return false
                    }
                }
            }

            matched += 1
            actualIndex = matchIndex + 1
        }

        if requireExactLength && matched == expected.count && actual.count > expected.count {
            details.append("Unexpected extra tool call(s): \(actual.dropFirst(expected.count).map(\.name))")
            return false
        }

        return true
    }

    // MARK: - Response Relevance

    private static func evaluateResponseRelevance(
        expected: [String],
        response: String,
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

        return relevant
    }

    private static func evaluateClarification(
        required: Bool,
        response: String,
        details: inout [String]
    ) -> Bool {
        guard required else { return true }

        let lower = response.lowercased()
        let clarificationSignals = [
            "?",
            "which",
            "what",
            "could you clarify",
            "can you clarify",
            "which one",
            "do you mean",
            "please clarify",
        ]
        let hasClarification = clarificationSignals.contains { lower.contains($0) }
        if !hasClarification {
            details.append("Expected clarification question, but response seems definitive")
        }
        return hasClarification
    }

    // MARK: - Forbidden Tools

    private static func evaluateForbiddenTools(
        forbidden: [String],
        attempted: [String],
        executed: [String],
        details: inout [String]
    ) -> Bool {
        guard !forbidden.isEmpty else { return true }

        let forbiddenSet = Set(forbidden)
        let violating = (attempted + executed).filter { forbiddenSet.contains($0) }
        if !violating.isEmpty {
            details.append("Forbidden tool(s) called: \(violating.joined(separator: ", "))")
            return false
        }
        return true
    }

    // MARK: - Invalid Tool Calls

    private static func evaluateInvalidToolCalls(
        attempted: [(name: String, arguments: [String: JSONValue])],
        executed: [(name: String, arguments: [String: JSONValue])],
        malformedToolCalls: [String],
        details: inout [String]
    ) -> Bool {
        var valid = true

        if !malformedToolCalls.isEmpty {
            details.append("Malformed tool call(s): \(malformedToolCalls.count)")
            valid = false
        }

        var executedCounts: [String: Int] = [:]
        for call in executed {
            executedCounts[canonicalKey(name: call.name, arguments: call.arguments), default: 0] += 1
        }

        var invalidAttempts: [String] = []
        for call in attempted {
            let key = canonicalKey(name: call.name, arguments: call.arguments)
            if let remaining = executedCounts[key], remaining > 0 {
                executedCounts[key] = remaining - 1
            } else {
                invalidAttempts.append(call.name)
            }
        }

        if !invalidAttempts.isEmpty {
            details.append("Attempted tool call(s) that never executed: \(invalidAttempts.joined(separator: ", "))")
            valid = false
        }

        return valid
    }

    // MARK: - File Assertions

    private static func evaluateFileAssertions(
        assertions: [FileAssertion],
        sandboxRoot: URL,
        details: inout [String]
    ) -> Bool {
        guard !assertions.isEmpty else { return true }

        var passed = true

        for assertion in assertions {
            let fileURL = sandboxRoot.appendingPathComponent(assertion.path)
            guard let data = try? Data(contentsOf: fileURL),
                  let text = String(data: data, encoding: .utf8) else {
                details.append("File assertion failed: could not read \(assertion.path)")
                passed = false
                continue
            }

            for substring in assertion.mustContain {
                if !text.localizedCaseInsensitiveContains(substring) {
                    details.append("File assertion failed: \(assertion.path) missing '\(substring)'")
                    passed = false
                }
            }

            for substring in assertion.mustNotContain {
                if text.localizedCaseInsensitiveContains(substring) {
                    details.append("File assertion failed: \(assertion.path) unexpectedly contains '\(substring)'")
                    passed = false
                }
            }
        }

        return passed
    }

    // MARK: - Aggregate Scoring

    static func computeScenarioSummary(turns: [BenchmarkTurnResult]) -> BenchmarkScenarioSummary {
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
