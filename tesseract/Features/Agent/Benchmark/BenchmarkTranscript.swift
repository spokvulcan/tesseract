import Foundation
import MLXLMCommon

/// Writes a human-readable transcript of a benchmark scenario to a plain text file.
///
/// Each scenario gets its own `.transcript.txt` file showing the full model I/O:
/// input messages, raw output (including `<think>` and `<tool_call>` tags), tool
/// executions, and per-turn evaluation results. This makes it straightforward to
/// debug why the model made specific decisions (e.g. duplicate tool calls).
@MainActor
final class BenchmarkTranscript {

    private var lines: [String] = []

    // MARK: - Header

    func writeHeader(scenarioID: String, description: String, parameters: AgentGenerateParameters) {
        lines.append(String(repeating: "═", count: 72))
        lines.append("SCENARIO: \(scenarioID) — \(description)")
        lines.append("Parameters: temp=\(parameters.temperature), topP=\(parameters.topP)"
            + (parameters.repetitionPenalty.map { ", repPenalty=\($0)" } ?? ""))
        lines.append("Started: \(ISO8601DateFormatter().string(from: Date()))")
        lines.append(String(repeating: "═", count: 72))
        lines.append("")
    }

    // MARK: - Turn Start

    func writeTurnStart(index: Int, total: Int, userMessage: String) {
        lines.append(String(repeating: "━", count: 60))
        lines.append("TURN \(index + 1)/\(total)")
        lines.append(String(repeating: "━", count: 60))
        lines.append("")
        lines.append("User: \"\(userMessage)\"")
        lines.append("")
    }

    // MARK: - Model Input

    /// Writes the exact raw ChatML prompt the model sees, rendered through the Jinja
    /// template (includes `<|im_start|>`, `<|im_end|>`, tool definitions, etc.).
    func writeRawPrompt(round: Int, rawPrompt: String, messageCount: Int) {
        lines.append("╌╌╌ ROUND \(round) — MODEL INPUT (\(messageCount) messages, \(rawPrompt.count) chars) ╌╌╌")
        lines.append("")
        lines.append(rawPrompt)
        lines.append("")
    }

    // MARK: - Round Output

    /// Writes the reconstructed raw model output for one generation round.
    func writeRoundOutput(round: Int, rawOutput: String, thinkingContent: String?,
                          promptTokens: Int? = nil, genTokens: Int? = nil) {
        var header = "╌╌╌ ROUND \(round) — RAW OUTPUT"
        if let pt = promptTokens, let gt = genTokens {
            header += " (\(pt) prompt → \(gt) gen tokens)"
        }
        header += " ╌╌╌"
        lines.append(header)
        lines.append("")

        var reconstructed = ""
        if let thinking = thinkingContent, !thinking.isEmpty {
            reconstructed += "<think>\n\(thinking)\n</think>\n"
        }
        reconstructed += rawOutput

        if reconstructed.isEmpty {
            lines.append("(empty)")
        } else {
            lines.append(reconstructed)
        }
        lines.append("")
    }

    /// Appends reconstructed `<tool_call>` tags to the round output.
    func writeToolCalls(calls: [(name: String, arguments: [String: JSONValue])]) {
        for call in calls {
            let argsJSON = formatArguments(call.arguments)
            lines.append("<tool_call>")
            lines.append("{\"name\": \"\(call.name)\", \"arguments\": \(argsJSON)}")
            lines.append("</tool_call>")
        }
        lines.append("")
    }

    // MARK: - Tool Execution

    func writeToolExecution(name: String, arguments: [String: JSONValue], result: String) {
        let argsJSON = formatArguments(arguments)
        lines.append("╌╌╌ TOOL: \(name) ╌╌╌")
        lines.append("→ \(name)(\(argsJSON))")
        lines.append("← \(result)")
        lines.append("")
    }

    // MARK: - Turn Result

    func writeTurnResult(
        passed: Bool,
        toolsCalled: [String],
        expectedTools: [String],
        tokPerSec: Double?,
        latencyMs: Double,
        duplicateCount: Int
    ) {
        let status = passed ? "PASS" : "FAIL"
        lines.append("── RESULT: [\(status)] ──")
        lines.append("  Tools called:  \(toolsCalled)")
        lines.append("  Expected:      \(expectedTools)")
        if duplicateCount > 0 {
            lines.append("  Duplicates:    \(duplicateCount)")
        }
        if let tps = tokPerSec {
            lines.append("  Token/s:       \(String(format: "%.1f", tps))")
        }
        lines.append("  Latency:       \(String(format: "%.0f", latencyMs))ms")
        lines.append("")
        lines.append("")
    }

    // MARK: - Footer

    func writeFooter(passed: Bool, toolAccuracy: Double, duplicateRate: Double) {
        lines.append(String(repeating: "═", count: 72))
        lines.append("SCENARIO RESULT: \(passed ? "PASS" : "FAIL")")
        lines.append("  Tool Accuracy:  \(String(format: "%.0f%%", toolAccuracy * 100))")
        lines.append("  Duplicate Rate: \(String(format: "%.0f%%", duplicateRate * 100))")
        lines.append(String(repeating: "═", count: 72))
    }

    // MARK: - Write to Disk

    func write(to url: URL) throws {
        let content = lines.joined(separator: "\n")
        try content.write(to: url, atomically: true, encoding: .utf8)
    }

    // MARK: - Private

    private func formatArguments(_ arguments: [String: JSONValue]) -> String {
        guard !arguments.isEmpty else { return "{}" }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        if let data = try? encoder.encode(arguments),
           let json = String(data: data, encoding: .utf8) {
            return json
        }
        // Fallback: manual formatting
        let pairs = arguments.map { "\"\($0.key)\": \(describeValue($0.value))" }
        return "{\(pairs.joined(separator: ", "))}"
    }

    private func describeValue(_ value: JSONValue) -> String {
        switch value {
        case .null: return "null"
        case .bool(let b): return b ? "true" : "false"
        case .int(let i): return "\(i)"
        case .double(let d): return "\(d)"
        case .string(let s): return "\"\(s)\""
        case .array(let arr): return "[\(arr.map { describeValue($0) }.joined(separator: ", "))]"
        case .object(let obj):
            let pairs = obj.map { "\"\($0.key)\": \(describeValue($0.value))" }
            return "{\(pairs.joined(separator: ", "))}"
        }
    }
}
