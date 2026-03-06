import Foundation
import ImageIO
import MLXLMCommon
import UniformTypeIdentifiers

// MARK: - ReadToolDetails

nonisolated struct ReadToolDetails: Sendable, Hashable {
    let path: String
    let lineCount: Int
    let wasTruncated: Bool
    let totalLines: Int
}

// MARK: - ReadToolError

nonisolated enum ReadToolError: LocalizedError {
    case offsetBeyondEnd(offset: Int, totalLines: Int)

    var errorDescription: String? {
        switch self {
        case .offsetBeyondEnd(let offset, let totalLines):
            "Offset \(offset) is beyond end of file (\(totalLines) lines total)"
        }
    }
}

// MARK: - ReadTool Factory

nonisolated func createReadTool(sandbox: PathSandbox) -> AgentToolDefinition {
    AgentToolDefinition(
        name: "read",
        label: "Read File",
        description: "Read the contents of a file. Returns numbered lines (cat -n style). For images, returns a note and the image data directly.",
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
            let data = try Data(contentsOf: url)

            if let mimeType = ReadToolHelper.detectSupportedImageMimeType(from: data) {
                return AgentToolResult(
                    content: [
                        .text("Read image file [\(mimeType)]"),
                        .image(data: data, mimeType: mimeType),
                    ],
                    details: ReadToolDetails(
                        path: displayName, lineCount: 0,
                        wasTruncated: false, totalLines: 0
                    )
                )
            }

            let content = String(decoding: data, as: UTF8.self)
            return try ReadToolHelper.formatOutput(
                content: content, offset: offset, limit: limit,
                displayName: displayName
            )
        }
    )
}

// MARK: - Helper (nonisolated)

private nonisolated enum ReadToolHelper: Sendable {
    static let maxLines = 2_000
    static let maxBytes = 50 * 1_024
    static let maxSingleLineBytes = maxBytes

    static func isSupportedImage(_ utType: UTType) -> Bool {
        utType.conforms(to: .png) || utType.conforms(to: .jpeg)
            || utType.conforms(to: .gif) || utType.conforms(to: .webP)
            || utType.conforms(to: .tiff)
    }

    static func detectSupportedImageMimeType(from data: Data) -> String? {
        guard
            let imageSource = CGImageSourceCreateWithData(data as CFData, nil),
            let typeIdentifier = CGImageSourceGetType(imageSource) as String?,
            let utType = UTType(typeIdentifier),
            isSupportedImage(utType)
        else {
            return nil
        }

        return utType.preferredMIMEType
    }

    static func formatLine(number: Int, content: String) -> String {
        let pad = String(repeating: " ", count: max(0, 6 - String(number).count))
        return "\(pad)\(number)\t\(content)"
    }

    static func truncateToByteLimit(_ string: String, maxBytes: Int) -> String {
        var byteCount = 0
        var result = ""

        for character in string {
            let charString = String(character)
            let charBytes = charString.utf8.count
            if byteCount + charBytes > maxBytes {
                break
            }
            result.append(charString)
            byteCount += charBytes
        }

        return result
    }

    static func formatSize(_ byteCount: Int) -> String {
        if byteCount < 1_024 {
            return "\(byteCount)B"
        }
        if byteCount < 1_024 * 1_024 {
            return String(format: "%.1fKB", Double(byteCount) / 1_024.0)
        }
        return String(format: "%.1fMB", Double(byteCount) / (1_024.0 * 1_024.0))
    }

    static func formatOutput(
        content: String, offset: Int?, limit: Int?,
        displayName: String
    ) throws -> AgentToolResult {
        if content.isEmpty {
            if let offset, offset > 1 {
                throw ReadToolError.offsetBeyondEnd(offset: offset, totalLines: 0)
            }

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

        let startIndex = max(0, (offset ?? 1) - 1)
        guard startIndex < totalLines else {
            throw ReadToolError.offsetBeyondEnd(
                offset: offset ?? (startIndex + 1),
                totalLines: totalLines
            )
        }

        let remainingLines = Array(allLines[startIndex...])
        let userLineLimit = limit.map { max(0, $0) }
        let selectedLines = userLineLimit.map { Array(remainingLines.prefix($0)) } ?? remainingLines
        let internalLineLimit = min(selectedLines.count, maxLines)

        // Reserve exact budget for the worst-case truncation notice.
        // The largest possible notice uses totalLines for all numeric fields
        // (startLine, endLine, totalLines, nextOffset are all <= totalLines + 1).
        let maxNum = String(totalLines + 1)
        let worstNotice =
            "\n\n[Showing lines \(maxNum)-\(maxNum) of \(maxNum) (\(formatSize(maxBytes)) limit). Use offset=\(maxNum) to continue.]"
        let effectiveMax = maxBytes - worstNotice.utf8.count

        // Collect lines up to both line and byte limits
        var outputLines: [String] = []
        var byteCount = 0
        var hitByteLimit = false
        var hitInternalLineLimit = false
        var truncatedOversizedLine = false
        var linesProcessed = 0

        for (i, line) in selectedLines.prefix(internalLineLimit).enumerated() {
            let lineNumber = startIndex + i + 1
            let lineStr = String(line)

            // Truncate individual lines that exceed the single-line cap
            if lineStr.utf8.count > maxSingleLineBytes {
                let trimmed = truncateToByteLimit(lineStr, maxBytes: maxSingleLineBytes)
                let formatted = formatLine(number: lineNumber, content: trimmed)
                let notice =
                    "[Line \(lineNumber) is \(formatSize(lineStr.utf8.count)), truncated to \(formatSize(maxSingleLineBytes)). Content may be incomplete.]"
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
                truncatedOversizedLine = true
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

        if linesProcessed == internalLineLimit && selectedLines.count > internalLineLimit {
            hitInternalLineLimit = true
        }

        let endLine = startIndex + linesProcessed
        let hitToolTruncation = hitByteLimit || hitInternalLineLimit || truncatedOversizedLine

        var result = outputLines.joined(separator: "\n")

        if hitByteLimit || hitInternalLineLimit {
            let nextOffset = endLine + 1
            if !result.isEmpty {
                result += "\n\n"
            }

            if hitByteLimit {
                result +=
                    "[Showing lines \(startIndex + 1)-\(endLine) of \(totalLines) (\(formatSize(maxBytes)) limit). Use offset=\(nextOffset) to continue.]"
            } else {
                result +=
                    "[Showing lines \(startIndex + 1)-\(endLine) of \(totalLines). Use offset=\(nextOffset) to continue.]"
            }
        } else if let userLineLimit, linesProcessed == selectedLines.count,
            startIndex + userLineLimit < totalLines
        {
            let remainingCount = totalLines - (startIndex + userLineLimit)
            let nextOffset = startIndex + userLineLimit + 1
            if !result.isEmpty {
                result += "\n\n"
            }
            result += "[\(remainingCount) more lines in file. Use offset=\(nextOffset) to continue.]"
        }

        return AgentToolResult(
            content: [.text(result)],
            details: ReadToolDetails(
                path: displayName,
                lineCount: linesProcessed,
                wasTruncated: hitToolTruncation,
                totalLines: totalLines
            )
        )
    }
}
