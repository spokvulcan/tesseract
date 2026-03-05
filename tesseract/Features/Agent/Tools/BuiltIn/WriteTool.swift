import Foundation
import MLXLMCommon

// MARK: - WriteToolDetails

nonisolated struct WriteToolDetails: Sendable, Hashable {
    let path: String
    let byteCount: Int
    let created: Bool
}

// MARK: - WriteTool Factory

nonisolated func createWriteTool(sandbox: PathSandbox) -> AgentToolDefinition {
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
            let displayName = sandbox.displayPath(url)

            if signal?.isCancelled == true {
                return .error("Cancelled")
            }

            let fileManager = FileManager.default
            let isNew = !fileManager.fileExists(atPath: url.path)

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
                return .error("Cancelled")
            }

            let byteCount = data.count
            return AgentToolResult(
                content: [.text("Wrote \(byteCount) bytes to \(displayName)")],
                details: WriteToolDetails(
                    path: displayName,
                    byteCount: byteCount,
                    created: isNew
                )
            )
        }
    )
}

