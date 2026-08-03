import Foundation

// MARK: - Turn Result

struct BenchmarkTurnResult: Codable {
    let turnIndex: Int
    let userMessage: String
    let assistantResponse: String
    let attemptedTools: [String]
    let toolsCalled: [String]
    let toolResults: [String]
    let passed: Bool
    let checks: BenchmarkTurnChecks
    let performance: BenchmarkTurnPerformance
}

struct BenchmarkTurnChecks: Codable {
    let toolsCorrect: Bool
    let duplicateToolCalls: Int
    let argumentsCorrect: Bool
    let responseRelevant: Bool
    let clarificationAsked: Bool
    let noForbiddenTools: Bool
    let noInvalidToolCalls: Bool
    let malformedToolCallCount: Int
    let fileAssertionsPassed: Bool
    let details: String?
}

struct BenchmarkTurnPerformance: Codable {
    let promptTokens: Int
    let genTokens: Int
    let tokPerSec: Double
    let latencyMs: Double
    let toolRoundsUsed: Int
}

// MARK: - Scenario Result

struct BenchmarkScenarioResult: Codable {
    let id: String
    let description: String
    let turns: [BenchmarkTurnResult]
    let passed: Bool
    let summary: BenchmarkScenarioSummary
}

struct BenchmarkScenarioSummary: Codable {
    let toolAccuracy: Double
    let duplicateRate: Double
    let avgTokPerSec: Double
    let avgLatencyMs: Double
}

// MARK: - Aggregate

struct BenchmarkAggregate: Codable {
    let passedScenarios: Int
    let totalScenarios: Int
    let overallToolAccuracy: Double
    let duplicateRate: Double
    let avgTokPerSec: Double
    let p50LatencyMs: Double
    let p95LatencyMs: Double
    let peakMemoryMB: Float
}

// MARK: - Full Report

struct BenchmarkReport: Codable {
    let metadata: BenchmarkMetadata
    let scenarios: [BenchmarkScenarioResult]
    let aggregate: BenchmarkAggregate
    /// Present only on results annotated after the fact — see `BenchmarkValidity`.
    /// The runner never sets this; a fresh report is valid by construction.
    var validity: BenchmarkValidity?
}

struct BenchmarkMetadata: Codable {
    let date: String
    let modelName: String
    let hardware: String
    let parameters: AgentGenerateParameters
    let promptProfile: BenchmarkConfig.PromptProfile
    let contextLimit: Int
    let maxToolRounds: Int
    let sweepLabel: String

    /// Git revision of the tree this build came from (`a73a7aa5`, or
    /// `a73a7aa5-dirty` when the working copy had uncommitted changes).
    /// Supplied by `scripts/bench.sh` via `--bench-source-revision`.
    ///
    /// Optional because every report written before 2026-07-27 predates the
    /// field, and because a hand-launched run has no revision to record. When
    /// it is nil, a result cannot be attributed to a commit — treat any
    /// before/after comparison involving it as correlational only.
    var sourceRevision: String?
}

/// Post-hoc annotation marking a stored result as not-comparable.
///
/// Results are immutable records of what the machine did, so a run is never
/// deleted or edited when it turns out to have measured the wrong thing —
/// it is annotated, and analyses filter on this. Written by hand (or by a
/// migration), never by the runner.
struct BenchmarkValidity: Codable {
    /// `invalid` = do not include in any comparison or trend.
    let status: String
    /// Why, in one line.
    let reason: String
    /// Where the full story lives — an issue, PR, commit, or doc.
    let reference: String
}

// MARK: - Report Writing

extension BenchmarkReport {
    func write(to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(self)
        try data.write(to: url, options: .atomic)
    }

    /// Prints a summary table to the log file.
    func summaryString() -> String {
        var lines: [String] = []
        lines.append("═══════════════════════════════════════════════════════")
        lines.append("  BENCHMARK REPORT — \(metadata.date)")
        lines.append("  Model: \(metadata.modelName) | \(metadata.hardware)")
        lines.append("  Params: \(metadata.sweepLabel)")
        lines.append("  Prompt: \(metadata.promptProfile.rawValue)")
        // Surfaced in the log, not just the JSON: an unattributable run should be
        // obvious while you are still looking at it.
        let revision = metadata.sourceRevision ?? "UNKNOWN (not launched via bench.sh)"
        lines.append("  Source: \(revision)")
        lines.append("═══════════════════════════════════════════════════════")
        lines.append("")

        for scenario in scenarios {
            let status = scenario.passed ? "PASS" : "FAIL"
            let toolAcc = String(format: "%.0f%%", scenario.summary.toolAccuracy * 100)
            let dupRate = String(format: "%.0f%%", scenario.summary.duplicateRate * 100)
            let tokSec = String(format: "%.1f", scenario.summary.avgTokPerSec)
            lines.append("  [\(status)] \(scenario.id): \(scenario.description)")
            lines.append("         Tools: \(toolAcc) | Dups: \(dupRate) | \(tokSec) tok/s")
        }

        lines.append("")
        lines.append("  ─── Aggregate ───")
        lines.append("  Passed: \(aggregate.passedScenarios)/\(aggregate.totalScenarios)")
        lines.append(
            "  Tool Accuracy: \(String(format: "%.1f%%", aggregate.overallToolAccuracy * 100))")
        lines.append("  Duplicate Rate: \(String(format: "%.1f%%", aggregate.duplicateRate * 100))")
        lines.append("  Avg tok/s: \(String(format: "%.1f", aggregate.avgTokPerSec))")
        lines.append(
            "  Latency p50: \(String(format: "%.0f", aggregate.p50LatencyMs))ms  p95: \(String(format: "%.0f", aggregate.p95LatencyMs))ms"
        )
        lines.append("  Peak Memory: \(String(format: "%.0f", aggregate.peakMemoryMB))MB")
        lines.append("═══════════════════════════════════════════════════════")

        return lines.joined(separator: "\n")
    }
}
