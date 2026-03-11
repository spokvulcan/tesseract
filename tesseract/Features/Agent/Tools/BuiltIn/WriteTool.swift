import Foundation
import MLXLMCommon

// MARK: - WriteToolError

nonisolated struct WriteToolError: LocalizedError {
    let message: String

    var errorDescription: String? { message }
}

// MARK: - WriteTool Factory

nonisolated func createWriteTool(sandbox: PathSandbox, readTracker: FileReadTracker) -> AgentToolDefinition {
    AgentToolDefinition(
        name: "write",
        label: "Write File",
        description: "Write content to a file. Creates parent directories if needed. Overwrites existing files.",
        parameterSchema: JSONSchema(
            type: "object",
            properties: [
                "path": PropertySchema(
                    type: "string",
                    description: "File path to write"
                ),
                "content": PropertySchema(
                    type: "string",
                    description: "Content to write"
                ),
            ],
            required: ["path", "content"]
        ),
        execute: { _, argsJSON, signal, _ in
            guard let path = ToolArgExtractor.string(argsJSON, key: "path") else {
                return .error("Missing required argument: path")
            }
            guard let content = ToolArgExtractor.string(argsJSON, key: "content") else {
                return .error("Missing required argument: content")
            }

            let url = try sandbox.resolveForWrite(path)

            if FileManager.default.fileExists(atPath: url.path) && !readTracker.hasRead(url.path) {
                return .error("You must read a file before overwriting it. Use the read tool on '\(path)' first.")
            }

            if signal?.isCancelled == true {
                throw WriteToolError(message: "Operation aborted")
            }

            let fileManager = FileManager.default

            // Create parent directories if needed
            let parent = url.deletingLastPathComponent()
            if !fileManager.fileExists(atPath: parent.path) {
                try fileManager.createDirectory(
                    at: parent, withIntermediateDirectories: true)
            }

            // Write content
            guard let data = content.data(using: .utf8) else {
                return .error("Content could not be encoded as UTF-8")
            }
            try data.write(to: url, options: .atomic)

            if signal?.isCancelled == true {
                throw WriteToolError(message: "Operation aborted")
            }

            let reportedByteCount = content.utf16.count
            return AgentToolResult(
                content: [.text("Successfully wrote \(reportedByteCount) bytes to \(path)")]
            )
        }
    )
}
