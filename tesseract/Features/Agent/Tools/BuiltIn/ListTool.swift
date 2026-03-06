import Foundation
import MLXLMCommon

// MARK: - LsToolDetails

nonisolated struct LsToolTruncationDetails: Sendable, Hashable {
    let truncated: Bool
    let truncatedBy: String?
    let totalLines: Int
    let totalBytes: Int
    let outputLines: Int
    let outputBytes: Int
    let lastLinePartial: Bool
    let firstLineExceedsLimit: Bool
    let maxLines: Int
    let maxBytes: Int
}

nonisolated struct LsToolDetails: Sendable, Hashable {
    let truncation: LsToolTruncationDetails?
    let entryLimitReached: Int?
}

// MARK: - LsToolError

nonisolated struct LsToolError: LocalizedError {
    let message: String

    var errorDescription: String? { message }
}

// MARK: - LsTool Factory

private nonisolated enum Defaults {
    static let defaultLimit = 500
    static let maxBytes = 50 * 1_024
}

nonisolated func createLsTool(sandbox: PathSandbox) -> AgentToolDefinition {
    AgentToolDefinition(
        name: "ls",
        label: "ls",
        description: "List directory contents. Returns entries sorted alphabetically, with '/' suffix for directories. Includes dotfiles. Output is truncated to 500 entries or 50KB (whichever is hit first).",
        parameterSchema: JSONSchema(
            type: "object",
            properties: [
                "path": PropertySchema(
                    type: "string",
                    description: "Directory to list (default: current directory)"
                ),
                "limit": PropertySchema(
                    type: "integer",
                    description: "Maximum number of entries to return (default: 500)"
                ),
            ],
            required: []
        ),
        execute: { _, argsJSON, signal, _ in
            if signal?.isCancelled == true {
                throw LsToolError(message: "Operation aborted")
            }

            let pathArg = ToolArgExtractor.string(argsJSON, key: "path")
            let limit = ToolArgExtractor.int(argsJSON, key: "limit") ?? Defaults.defaultLimit

            let url: URL
            if let pathArg {
                url = try sandbox.resolve(pathArg)
            } else {
                url = sandbox.root.standardizedFileURL
            }

            var isDir: ObjCBool = false
            let fm = FileManager.default
            guard fm.fileExists(atPath: url.path, isDirectory: &isDir) else {
                throw LsToolError(message: "Path not found: \(url.path)")
            }
            guard isDir.boolValue else {
                throw LsToolError(message: "Not a directory: \(url.path)")
            }

            return try LsToolHelper.listDirectory(
                at: url,
                limit: limit,
                fileManager: fm,
                signal: signal
            )
        }
    )
}

nonisolated func createListTool(sandbox: PathSandbox) -> AgentToolDefinition {
    createLsTool(sandbox: sandbox)
}

// MARK: - Helper

private nonisolated enum LsToolHelper: Sendable {
    static func listDirectory(
        at url: URL,
        limit: Int,
        fileManager: FileManager,
        signal: CancellationToken?
    ) throws -> AgentToolResult {
        let entries: [String]
        do {
            entries = try fileManager.contentsOfDirectory(atPath: url.path)
        } catch {
            throw LsToolError(message: "Cannot read directory: \(error.localizedDescription)")
        }

        let sortedEntries = entries.sorted(by: caseInsensitiveAscending)
        var results: [String] = []
        var entryLimitReached: Int?

        for entry in sortedEntries {
            if signal?.isCancelled == true {
                throw LsToolError(message: "Operation aborted")
            }

            if results.count >= limit {
                entryLimitReached = limit
                break
            }

            let entryURL = url.appendingPathComponent(entry)
            var isDirectory: ObjCBool = false
            guard fileManager.fileExists(atPath: entryURL.path, isDirectory: &isDirectory) else {
                continue
            }

            results.append(entry + (isDirectory.boolValue ? "/" : ""))
        }

        if results.isEmpty {
            return .text("(empty directory)")
        }

        let rawOutput = results.joined(separator: "\n")
        let truncation = truncateHead(rawOutput, maxBytes: Defaults.maxBytes)
        var output = truncation.content
        var notices: [String] = []

        if let entryLimitReached {
            notices.append("\(entryLimitReached) entries limit reached. Use limit=\(entryLimitReached * 2) for more")
        }

        var truncationDetails: LsToolTruncationDetails?
        if truncation.truncated {
            notices.append("\(formatSize(Defaults.maxBytes)) limit reached")
            truncationDetails = LsToolTruncationDetails(
                truncated: true,
                truncatedBy: truncation.truncatedBy,
                totalLines: truncation.totalLines,
                totalBytes: truncation.totalBytes,
                outputLines: truncation.outputLines,
                outputBytes: truncation.outputBytes,
                lastLinePartial: false,
                firstLineExceedsLimit: truncation.firstLineExceedsLimit,
                maxLines: Int.max,
                maxBytes: Defaults.maxBytes
            )
        }

        if !notices.isEmpty {
            output += "\n\n[\(notices.joined(separator: ". "))]"
        }

        let details: LsToolDetails? =
            if truncationDetails != nil || entryLimitReached != nil {
                LsToolDetails(truncation: truncationDetails, entryLimitReached: entryLimitReached)
            } else {
                nil
            }

        return AgentToolResult(content: [.text(output)], details: details)
    }

    static func caseInsensitiveAscending(_ lhs: String, _ rhs: String) -> Bool {
        let lowerCompare = lhs.lowercased().compare(rhs.lowercased())
        if lowerCompare == .orderedSame {
            return lhs < rhs
        }
        return lowerCompare == .orderedAscending
    }

    static func truncateHead(_ content: String, maxBytes: Int) -> (
        content: String,
        truncated: Bool,
        truncatedBy: String?,
        totalLines: Int,
        totalBytes: Int,
        outputLines: Int,
        outputBytes: Int,
        firstLineExceedsLimit: Bool
    ) {
        let totalBytes = content.utf8.count
        let lines = content.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        let totalLines = lines.count

        if totalBytes <= maxBytes {
            return (
                content: content,
                truncated: false,
                truncatedBy: nil,
                totalLines: totalLines,
                totalBytes: totalBytes,
                outputLines: totalLines,
                outputBytes: totalBytes,
                firstLineExceedsLimit: false
            )
        }

        if let firstLine = lines.first, firstLine.utf8.count > maxBytes {
            return (
                content: "",
                truncated: true,
                truncatedBy: "bytes",
                totalLines: totalLines,
                totalBytes: totalBytes,
                outputLines: 0,
                outputBytes: 0,
                firstLineExceedsLimit: true
            )
        }

        var outputLines: [String] = []
        var outputBytes = 0

        for line in lines {
            let lineBytes = line.utf8.count + (outputLines.isEmpty ? 0 : 1)
            if outputBytes + lineBytes > maxBytes {
                break
            }
            outputLines.append(line)
            outputBytes += lineBytes
        }

        let output = outputLines.joined(separator: "\n")
        return (
            content: output,
            truncated: true,
            truncatedBy: "bytes",
            totalLines: totalLines,
            totalBytes: totalBytes,
            outputLines: outputLines.count,
            outputBytes: output.utf8.count,
            firstLineExceedsLimit: false
        )
    }

    static func formatSize(_ bytes: Int) -> String {
        if bytes < 1_024 {
            return "\(bytes)B"
        }
        if bytes < 1_024 * 1_024 {
            return String(format: "%.1fKB", Double(bytes) / 1_024.0)
        }
        return String(format: "%.1fMB", Double(bytes) / (1_024.0 * 1_024.0))
    }
}
