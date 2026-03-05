import Foundation
import MLXLMCommon
import UniformTypeIdentifiers

// MARK: - ReadToolDetails

nonisolated struct ReadToolDetails: Sendable, Hashable {
    let path: String
    let lineCount: Int
    let wasTruncated: Bool
    let totalLines: Int
}

// MARK: - ReadTool Factory

nonisolated func createReadTool(sandbox: PathSandbox) -> AgentToolDefinition {
    AgentToolDefinition(
        name: "read",
        label: "Read File",
        description: "Read the contents of a file. Returns numbered lines (cat -n style). For images, returns the image data directly.",
        parameterSchema: JSONSchema(
            type: "object",
            properties: [
                "path": PropertySchema(
                    type: "string",
                    description: "File path relative to working directory"
                ),
                "offset": PropertySchema(
                    type: "integer",
                    description: "Line number to start from (1-indexed)"
                ),
                "limit": PropertySchema(
                    type: "integer",
                    description: "Maximum number of lines to return"
                ),
            ],
            required: ["path"]
        ),
        execute: { _, argsJSON, _, _ in
            guard let path = ToolArgExtractor.string(argsJSON, key: "path") else {
                return .error("Missing required argument: path")
            }
            let offset = ToolArgExtractor.int(argsJSON, key: "offset")
            let limit = ToolArgExtractor.int(argsJSON, key: "limit")

            let url = try sandbox.resolveExisting(path)
            let displayName = sandbox.displayPath(url)

            // Check if this is an image file
            if let utType = UTType(filenameExtension: url.pathExtension),
               ReadToolHelper.isImage(utType)
            {
                let data = try Data(contentsOf: url)
                let mimeType = utType.preferredMIMEType ?? "application/octet-stream"
                return AgentToolResult(
                    content: [.image(data: data, mimeType: mimeType)],
                    details: ReadToolDetails(
                        path: displayName, lineCount: 0,
                        wasTruncated: false, totalLines: 0
                    )
                )
            }

            // Read as text
            let content = try String(contentsOf: url, encoding: .utf8)
            return ReadToolHelper.formatOutput(
                content: content, offset: offset, limit: limit,
                displayName: displayName
            )
        }
    )
}

// MARK: - Helper (nonisolated)

private nonisolated enum ReadToolHelper: Sendable {
    static let maxLines = 2_000
    static let maxBytes = 30_000
    static let maxSingleLineBytes = 30_000

    static func isImage(_ utType: UTType) -> Bool {
        utType.conforms(to: .png) || utType.conforms(to: .jpeg)
            || utType.conforms(to: .gif) || utType.conforms(to: .webP)
            || utType.conforms(to: .tiff)
    }

    static func formatLine(number: Int, content: String) -> String {
        let pad = String(repeating: " ", count: max(0, 6 - String(number).count))
        return "\(pad)\(number)\t\(content)"
    }

    static func formatOutput(
        content: String, offset: Int?, limit: Int?,
        displayName: String
    ) -> AgentToolResult {
        if content.isEmpty {
            return AgentToolResult(
                content: [.text("(empty file)")],
                details: ReadToolDetails(
                    path: displayName, lineCount: 0,
                    wasTruncated: false, totalLines: 0
                )
            )
        }

        let allLines = content.split(separator: "\n", omittingEmptySubsequences: false)
        let totalLines = allLines.count

        // Apply offset (1-indexed → 0-indexed)
        let startIndex: Int
        if let offset, offset > 1 {
            startIndex = min(offset - 1, totalLines)
        } else {
            startIndex = 0
        }

        let remainingLines = allLines[startIndex...]
        let lineLimit = max(0, min(limit ?? maxLines, maxLines))

        // Reserve exact budget for the worst-case truncation notice.
        // The largest possible notice uses totalLines for all numeric fields
        // (startLine, endLine, totalLines, nextOffset are all <= totalLines + 1).
        let maxNum = String(totalLines + 1)
        let worstNotice = "\n[Showing lines \(maxNum)-\(maxNum) of \(maxNum). Use offset=\(maxNum) to continue.]"
        let effectiveMax = maxBytes - worstNotice.utf8.count

        // Collect lines up to both line and byte limits
        var outputLines: [String] = []
        var byteCount = 0
        var hitByteLimit = false
        var linesProcessed = 0

        for (i, line) in remainingLines.prefix(lineLimit).enumerated() {
            let lineNumber = startIndex + i + 1
            let lineStr = String(line)

            // Truncate individual lines that exceed the single-line cap
            if lineStr.utf8.count > maxSingleLineBytes {
                let sizeKB = lineStr.utf8.count / 1_000
                let trimmed = String(lineStr.prefix(maxSingleLineBytes))
                let formatted = formatLine(number: lineNumber, content: trimmed)
                let notice = "[Line \(lineNumber) is \(sizeKB)KB, truncated to \(maxSingleLineBytes / 1_000)KB. Content may be incomplete.]"
                let addedBytes = formatted.utf8.count + notice.utf8.count + 2
                // Always include at least the first line, even if it exceeds the
                // byte budget on its own (the per-line truncation already caps it).
                if !outputLines.isEmpty, byteCount + addedBytes > effectiveMax {
                    hitByteLimit = true
                    break
                }
                outputLines.append(formatted)
                outputLines.append(notice)
                byteCount += addedBytes
                linesProcessed += 1
                continue
            }

            let formatted = formatLine(number: lineNumber, content: lineStr)
            let addedBytes = formatted.utf8.count + 1

            if !outputLines.isEmpty, byteCount + addedBytes > effectiveMax {
                hitByteLimit = true
                break
            }

            outputLines.append(formatted)
            byteCount += addedBytes
            linesProcessed += 1
        }

        let endLine = startIndex + linesProcessed
        let wasTruncated = hitByteLimit || endLine < totalLines

        var result = outputLines.joined(separator: "\n")

        if wasTruncated {
            let nextOffset = endLine + 1
            result += "\n[Showing lines \(startIndex + 1)-\(endLine) of \(totalLines). Use offset=\(nextOffset) to continue.]"
        }

        return AgentToolResult(
            content: [.text(result)],
            details: ReadToolDetails(
                path: displayName,
                lineCount: linesProcessed,
                wasTruncated: wasTruncated,
                totalLines: totalLines
            )
        )
    }
}
