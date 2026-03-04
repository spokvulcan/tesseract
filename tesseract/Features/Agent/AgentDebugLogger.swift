import Foundation
import os

/// Writes agent conversation turns to disk as JSON for post-hoc debugging.
///
/// Each conversation gets a timestamped directory under the sandbox-safe temp dir.
/// Each turn writes a JSON file with assistant content, thinking, tool calls,
/// and context size.
///
/// Follows the same pattern as `AudioPlaybackManager`'s debug dump.
@MainActor
final class AgentDebugLogger {

    private var sessionDir: URL?
    private var turnIndex: Int = 0
    private static let iso8601 = ISO8601DateFormatter()

    /// Starts a new debug session with a timestamped directory.
    func startSession() {
        let dir = DebugPaths.agent
            .appendingPathComponent(DebugPaths.timestamp())

        do {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            sessionDir = dir
            turnIndex = 0
            Log.agent.info("Debug session started → \(dir.path)")
        } catch {
            Log.agent.error("Failed to create debug dir: \(error)")
            sessionDir = nil
        }
    }

    /// Logs the system prompt and tool definitions at session start.
    func logSystemPrompt(_ prompt: String, tools: [AgentToolDefinition]) {
        guard let dir = sessionDir else { return }

        // Write full system prompt as text
        let promptFile = dir.appendingPathComponent("system_prompt.txt")
        try? prompt.write(to: promptFile, atomically: true, encoding: .utf8)

        // Write tool names + descriptions as JSON
        let toolEntries: [[String: String]] = tools.map {
            ["name": $0.name, "description": String($0.description.prefix(200))]
        }
        if let data = try? JSONSerialization.data(
            withJSONObject: toolEntries,
            options: [.prettyPrinted, .sortedKeys]
        ) {
            try? data.write(to: dir.appendingPathComponent("tools.json"))
        }

        Log.agent.info("System prompt: \(prompt.count) chars, \(tools.count) tools [\(tools.map(\.name).joined(separator: ", "))]")
    }

    /// Logs a complete turn: assistant content, thinking, tool calls + results, and message count.
    func logTurn(
        message: AssistantMessage,
        toolResults: [ToolResultMessage],
        messageCount: Int
    ) {
        guard let dir = sessionDir else { return }

        var entry: [String: Any] = [
            "type": "turn",
            "turn": turnIndex,
            "timestamp": Self.iso8601.string(from: Date()),
            "assistantContent": message.content,
            "messageCount": messageCount,
        ]

        if let thinking = message.thinking, !thinking.isEmpty {
            entry["thinking"] = thinking
        }

        if !toolResults.isEmpty {
            entry["toolCalls"] = toolResults.map { [
                "name": $0.toolName,
                "result": String($0.content.textContent.prefix(500)),
            ] }
        }

        write(entry, filename: String(format: "turn_%03d.json", turnIndex), to: dir)
        turnIndex += 1
    }

    /// Resets the session (called on conversation clear).
    func reset() {
        sessionDir = nil
        turnIndex = 0
    }

    // MARK: - Private

    private func write(_ dict: [String: Any], filename: String, to dir: URL) {
        do {
            let data = try JSONSerialization.data(
                withJSONObject: dict,
                options: [.prettyPrinted, .sortedKeys]
            )
            try data.write(to: dir.appendingPathComponent(filename))
        } catch {
            Log.agent.error("Failed to write debug log \(filename): \(error)")
        }
    }
}
