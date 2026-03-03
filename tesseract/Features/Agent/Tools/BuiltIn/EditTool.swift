import Foundation
import MLXLMCommon

// MARK: - EditToolDetails

nonisolated struct EditToolDetails: Sendable, Hashable {
    let path: String
    let diff: String
    let firstChangedLine: Int
}

// MARK: - EditTool Factory

nonisolated func createEditTool(sandbox: PathSandbox) -> AgentToolDefinition {
    AgentToolDefinition(
        name: "edit",
        label: "Edit File",
        description: "Edit a file by replacing an exact text match. old_text must match exactly once in the file. Add surrounding context lines to old_text to disambiguate multiple matches.",
        parameterSchema: JSONSchema(
            type: "object",
            properties: [
                "path": PropertySchema(
                    type: "string",
                    description: "File path to edit"
                ),
                "old_text": PropertySchema(
                    type: "string",
                    description: "Exact text to find and replace"
                ),
                "new_text": PropertySchema(
                    type: "string",
                    description: "Replacement text"
                ),
            ],
            required: ["path", "old_text", "new_text"]
        ),
        execute: { _, argsJSON, _, _ in
            guard let path = EditToolHelper.extractString(argsJSON, key: "path") else {
                return .error("Missing required argument: path")
            }
            guard let oldText = EditToolHelper.extractString(argsJSON, key: "old_text") else {
                return .error("Missing required argument: old_text")
            }
            guard let newText = EditToolHelper.extractString(argsJSON, key: "new_text") else {
                return .error("Missing required argument: new_text")
            }

            let url = try sandbox.resolveExisting(path)
            let displayName = sandbox.displayPath(url)

            // Read raw bytes to detect BOM and line endings
            let rawData = try Data(contentsOf: url)
            let (content, hasBOM) = EditToolHelper.stripBOM(rawData)
            let hasCRLF = content.contains("\r\n")

            // Normalize to LF for matching
            let normalized = hasCRLF ? content.replacingOccurrences(of: "\r\n", with: "\n") : content
            let normalizedOld = oldText.replacingOccurrences(of: "\r\n", with: "\n")
            let normalizedNew = newText.replacingOccurrences(of: "\r\n", with: "\n")

            // Count occurrences
            let count = EditToolHelper.countOccurrences(of: normalizedOld, in: normalized)

            if count == 0 {
                return EditToolHelper.handleZeroMatches(
                    normalizedOld: normalizedOld, normalized: normalized)
            }

            if count > 1 {
                return .error("Found \(count) matches. old_text must match exactly once. Add surrounding context to make it unique.")
            }

            // Exactly one match — perform the replacement
            let replaced = normalized.replacingOccurrences(of: normalizedOld, with: normalizedNew)

            // Restore original line endings and BOM
            var output = hasCRLF ? replaced.replacingOccurrences(of: "\n", with: "\r\n") : replaced
            if hasBOM {
                output = "\u{FEFF}" + output
            }

            guard let outputData = output.data(using: .utf8) else {
                return .error("Edited content could not be encoded as UTF-8")
            }
            try outputData.write(to: url, options: .atomic)

            // Compute first changed line (1-indexed) and diff
            let firstChangedLine = EditToolHelper.lineNumber(
                of: normalizedOld, in: normalized)
            let diff = EditToolHelper.unifiedDiff(
                path: displayName,
                oldText: normalizedOld,
                newText: normalizedNew,
                firstChangedLine: firstChangedLine
            )

            let oldPreview = EditToolHelper.preview(normalizedOld)
            let newPreview = EditToolHelper.preview(normalizedNew)

            return AgentToolResult(
                content: [.text("Edited \(displayName): replaced \(oldPreview) with \(newPreview)")],
                details: EditToolDetails(
                    path: displayName,
                    diff: diff,
                    firstChangedLine: firstChangedLine
                )
            )
        }
    )
}

// MARK: - Helper (nonisolated)

private nonisolated enum EditToolHelper: Sendable {

    // MARK: Arg extraction

    static func extractString(_ args: [String: JSONValue], key: String) -> String? {
        guard let value = args[key] else { return nil }
        switch value {
        case .string(let s): return s
        case .int(let i): return String(i)
        case .double(let d): return String(d)
        default: return nil
        }
    }

    // MARK: BOM handling

    static func stripBOM(_ data: Data) -> (content: String, hasBOM: Bool) {
        let bom: [UInt8] = [0xEF, 0xBB, 0xBF]
        if data.count >= 3, data.prefix(3).elementsEqual(bom) {
            let stripped = String(data: data.dropFirst(3), encoding: .utf8) ?? ""
            return (stripped, true)
        }
        let content = String(data: data, encoding: .utf8) ?? ""
        return (content, false)
    }

    // MARK: Occurrence counting

    static func countOccurrences(of needle: String, in haystack: String) -> Int {
        guard !needle.isEmpty else { return 0 }
        var count = 0
        var searchRange = haystack.startIndex..<haystack.endIndex
        while let range = haystack.range(of: needle, range: searchRange) {
            count += 1
            searchRange = range.upperBound..<haystack.endIndex
        }
        return count
    }

    // MARK: Zero-match handling with fuzzy fallback

    static func handleZeroMatches(
        normalizedOld: String, normalized: String
    ) -> AgentToolResult {
        let fuzzyNeedle = collapseWhitespace(normalizedOld)
        // Scan the file content for a fuzzy match
        let lines = normalized.split(separator: "\n", omittingEmptySubsequences: false)

        // Build a sliding window of the same number of lines as old_text
        let needleLineCount = normalizedOld.split(
            separator: "\n", omittingEmptySubsequences: false
        ).count

        for windowStart in 0...max(0, lines.count - needleLineCount) {
            let windowEnd = min(windowStart + needleLineCount, lines.count)
            let windowText = lines[windowStart..<windowEnd].joined(separator: "\n")
            if collapseWhitespace(windowText) == fuzzyNeedle {
                return .error(
                    "No exact match found. Did you mean:\n\n\(windowText)\n\n(Copy the exact text above into old_text)"
                )
            }
        }

        return .error("No match found for the specified old_text")
    }

    /// Collapse runs of whitespace (spaces, tabs, etc.) to a single space and trim each line.
    static func collapseWhitespace(_ text: String) -> String {
        text.split(separator: "\n", omittingEmptySubsequences: false)
            .map { line in
                line.split(omittingEmptySubsequences: true, whereSeparator: \.isWhitespace)
                    .joined(separator: " ")
            }
            .joined(separator: "\n")
    }

    // MARK: Line number

    /// Find the 1-indexed line number where `needle` first appears in `haystack`.
    static func lineNumber(of needle: String, in haystack: String) -> Int {
        guard let range = haystack.range(of: needle) else { return 1 }
        let prefix = haystack[haystack.startIndex..<range.lowerBound]
        return prefix.filter({ $0 == "\n" }).count + 1
    }

    // MARK: Unified diff

    static func unifiedDiff(
        path: String,
        oldText: String,
        newText: String,
        firstChangedLine: Int
    ) -> String {
        let oldLines = oldText.split(separator: "\n", omittingEmptySubsequences: false)
            .map(String.init)
        let newLines = newText.split(separator: "\n", omittingEmptySubsequences: false)
            .map(String.init)

        var diff = "--- a/\(path)\n+++ b/\(path)\n"
        diff += "@@ -\(firstChangedLine),\(oldLines.count) +\(firstChangedLine),\(newLines.count) @@\n"
        for line in oldLines {
            diff += "-\(line)\n"
        }
        for line in newLines {
            diff += "+\(line)\n"
        }
        return diff
    }

    // MARK: Preview

    /// Short preview of text for the success message.
    static func preview(_ text: String, maxLength: Int = 40) -> String {
        let oneLine = text.replacingOccurrences(of: "\n", with: "\\n")
        if oneLine.count <= maxLength {
            return "\"\(oneLine)\""
        }
        return "\"\(oneLine.prefix(maxLength))…\""
    }
}
