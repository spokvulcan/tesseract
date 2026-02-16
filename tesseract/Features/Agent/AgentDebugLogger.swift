import Foundation
import os

/// Writes agent conversation turns to disk as JSON for post-hoc debugging.
///
/// Each conversation gets a timestamped directory under `/tmp/tesseract-debug/agent/`.
/// Each generation turn writes a JSON file with the full prompt, parameters,
/// raw output (including think blocks), and performance metrics.
///
/// Only works in Debug builds (Release entitlements block `/tmp` writes).
/// Follows the same pattern as `AudioPlaybackManager`'s debug dump.
@MainActor
final class AgentDebugLogger {

    private var sessionDir: URL?
    private var turnIndex: Int = 0

    /// Starts a new debug session with a timestamped directory.
    func startSession() {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd_HHmmss"
        let dir = URL(fileURLWithPath: "/tmp/tesseract-debug/agent")
            .appendingPathComponent(formatter.string(from: Date()))

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

    /// Logs the full input messages sent to the model.
    func logPrompt(messages: [AgentChatMessage], parameters: AgentGenerateParameters) {
        guard let dir = sessionDir else { return }

        let entry: [String: Any] = [
            "type": "prompt",
            "turn": turnIndex,
            "timestamp": ISO8601DateFormatter().string(from: Date()),
            "parameters": [
                "maxTokens": parameters.maxTokens,
                "temperature": parameters.temperature,
                "topP": parameters.topP,
                "repetitionPenalty": parameters.repetitionPenalty as Any,
                "repetitionContextSize": parameters.repetitionContextSize,
            ],
            "messages": messages.map { [
                "role": $0.role.rawValue,
                "content": $0.content,
            ] },
        ]

        write(entry, filename: String(format: "turn_%03d_prompt.json", turnIndex), to: dir)
    }

    /// Logs the complete raw generation output (including think blocks).
    func logResponse(
        rawOutput: String,
        displayOutput: String,
        info: AgentGeneration.Info?
    ) {
        guard let dir = sessionDir else { return }

        var entry: [String: Any] = [
            "type": "response",
            "turn": turnIndex,
            "timestamp": ISO8601DateFormatter().string(from: Date()),
            "rawOutput": rawOutput,
            "displayOutput": displayOutput,
        ]

        if let info {
            entry["metrics"] = [
                "promptTokenCount": info.promptTokenCount,
                "generationTokenCount": info.generationTokenCount,
                "promptTime": info.promptTime,
                "generateTime": info.generateTime,
                "tokensPerSecond": info.tokensPerSecond,
            ]
        }

        write(entry, filename: String(format: "turn_%03d_response.json", turnIndex), to: dir)
        turnIndex += 1
    }

    /// Logs an error that occurred during generation.
    func logError(_ error: String) {
        guard let dir = sessionDir else { return }

        let entry: [String: Any] = [
            "type": "error",
            "turn": turnIndex,
            "timestamp": ISO8601DateFormatter().string(from: Date()),
            "error": error,
        ]

        write(entry, filename: String(format: "turn_%03d_error.json", turnIndex), to: dir)
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
