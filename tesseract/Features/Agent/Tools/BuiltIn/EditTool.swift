import Foundation
import MLXLMCommon

// MARK: - EditToolDetails

nonisolated struct EditToolDetails: Sendable, Hashable {
    let path: String
    let diff: String
    let firstChangedLine: Int
}

nonisolated struct EditToolError: LocalizedError {
    let message: String

    var errorDescription: String? { message }
}

// MARK: - EditTool Factory

nonisolated func createEditTool(sandbox: PathSandbox, readTracker: FileReadTracker) -> AgentToolDefinition {
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
            guard let path = ToolArgExtractor.string(argsJSON, key: "path") else {
                return .error("Missing required argument: path")
            }
            guard let oldText = ToolArgExtractor.string(argsJSON, key: "old_text") else {
                return .error("Missing required argument: old_text")
            }
            guard let newText = ToolArgExtractor.string(argsJSON, key: "new_text") else {
                return .error("Missing required argument: new_text")
            }

            let url = try sandbox.resolveExisting(path)

            if !readTracker.hasRead(url.path) {
                return .error("You must read a file before editing it. Use the read tool on '\(path)' first.")
            }

            let displayName = sandbox.displayPath(url)

            let rawData = try Data(contentsOf: url)
            let (content, hasBOM) = EditToolHelper.stripBOM(rawData)
            let normalizedContent = EditToolHelper.normalizeToLF(content)
            let normalizedOld = EditToolHelper.normalizeToLF(oldText)
            let normalizedNew = EditToolHelper.normalizeToLF(newText)
            let match = EditToolHelper.fuzzyFindText(content: normalizedContent, oldText: normalizedOld)

            guard match.found else {
                throw EditToolError(
                    message:
                        "Could not find the exact text in \(path). The old text must match exactly including all whitespace and newlines."
                )
            }

            let fuzzyContent = EditToolHelper.normalizeForFuzzyMatch(normalizedContent)
            let fuzzyOld = EditToolHelper.normalizeForFuzzyMatch(normalizedOld)
            let occurrences = EditToolHelper.countOccurrences(of: fuzzyOld, in: fuzzyContent)

            if occurrences > 1 {
                throw EditToolError(
                    message:
                        "Found \(occurrences) occurrences of the text in \(path). The text must be unique. Please provide more context to make it unique."
                )
            }

            let baseContent = match.contentForReplacement
            let matchedText = EditToolHelper.substring(
                of: baseContent,
                start: match.index,
                length: match.matchLength
            )
            let replaced = EditToolHelper.replacing(
                in: baseContent,
                start: match.index,
                length: match.matchLength,
                with: normalizedNew
            )

            if baseContent == replaced {
                throw EditToolError(
                    message:
                        "No changes made to \(path). The replacement produced identical content. This might indicate an issue with special characters or the text not existing as expected."
                )
            }

            var output = EditToolHelper.restoreLineEndings(
                replaced,
                ending: EditToolHelper.detectLineEnding(in: rawData)
            )
            if hasBOM {
                output = "\u{FEFF}" + output
            }

            guard let outputData = output.data(using: .utf8) else {
                throw EditToolError(message: "Edited content could not be encoded as UTF-8")
            }
            try outputData.write(to: url, options: .atomic)

            let firstChangedLine = EditToolHelper.lineNumber(
                forCharacterOffset: match.index,
                in: baseContent
            )
            let diff = EditToolHelper.unifiedDiff(
                path: displayName,
                oldText: matchedText,
                newText: normalizedNew,
                firstChangedLine: firstChangedLine
            )

            return AgentToolResult(
                content: [.text("Successfully replaced text in \(path).")],
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

    enum LineEnding {
        case lf
        case crlf
    }

    struct MatchResult: Sendable {
        let found: Bool
        let index: Int
        let matchLength: Int
        let contentForReplacement: String
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

    // MARK: Line endings

    static func detectLineEnding(in data: Data) -> LineEnding {
        let bytes = Array(data)
        guard !bytes.isEmpty else { return .lf }

        for index in bytes.indices {
            switch bytes[index] {
            case 0x0D:
                if index + 1 < bytes.count, bytes[index + 1] == 0x0A {
                    return .crlf
                }
                return .lf
            case 0x0A:
                return .lf
            default:
                continue
            }
        }

        return .lf
    }

    static func normalizeToLF(_ text: String) -> String {
        text
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
    }

    static func restoreLineEndings(_ text: String, ending: LineEnding) -> String {
        switch ending {
        case .lf:
            text
        case .crlf:
            text
                .split(separator: "\n", omittingEmptySubsequences: false)
                .joined(separator: "\r\n")
        }
    }

    // MARK: Fuzzy matching

    static func normalizeForFuzzyMatch(_ text: String) -> String {
        text
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map { trimTrailingWhitespace(String($0)) }
            .joined(separator: "\n")
            .replacingOccurrences(
                of: #"[‘’‚‛]"#,
                with: "'",
                options: .regularExpression
            )
            .replacingOccurrences(
                of: #"[“”„‟]"#,
                with: "\"",
                options: .regularExpression
            )
            .replacingOccurrences(
                of: #"[‐‑‒–—―−]"#,
                with: "-",
                options: .regularExpression
            )
            .replacingOccurrences(
                of: #"[  -   　]"#,
                with: " ",
                options: .regularExpression
            )
    }

    static func fuzzyFindText(content: String, oldText: String) -> MatchResult {
        if let exactRange = content.range(of: oldText) {
            return MatchResult(
                found: true,
                index: content.distance(from: content.startIndex, to: exactRange.lowerBound),
                matchLength: oldText.count,
                contentForReplacement: content
            )
        }

        let fuzzyContent = normalizeForFuzzyMatch(content)
        let fuzzyOld = normalizeForFuzzyMatch(oldText)

        guard let fuzzyRange = fuzzyContent.range(of: fuzzyOld) else {
            return MatchResult(found: false, index: -1, matchLength: 0, contentForReplacement: content)
        }

        return MatchResult(
            found: true,
            index: fuzzyContent.distance(from: fuzzyContent.startIndex, to: fuzzyRange.lowerBound),
            matchLength: fuzzyOld.count,
            contentForReplacement: fuzzyContent
        )
    }

    private static func trimTrailingWhitespace(_ text: String) -> String {
        var result = text
        while let last = result.last, last.isWhitespace {
            result.removeLast()
        }
        return result
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

    // MARK: Line number

    static func lineNumber(forCharacterOffset offset: Int, in content: String) -> Int {
        guard offset > 0 else { return 1 }
        let clampedOffset = min(offset, content.count)
        let boundary = content.index(content.startIndex, offsetBy: clampedOffset)
        let prefix = content[content.startIndex..<boundary]
        return prefix.filter({ $0 == "\n" }).count + 1
    }

    static func substring(of text: String, start: Int, length: Int) -> String {
        let startIndex = text.index(text.startIndex, offsetBy: start)
        let endIndex = text.index(startIndex, offsetBy: length)
        return String(text[startIndex..<endIndex])
    }

    static func replacing(
        in text: String,
        start: Int,
        length: Int,
        with replacement: String
    ) -> String {
        let startIndex = text.index(text.startIndex, offsetBy: start)
        let endIndex = text.index(startIndex, offsetBy: length)
        return String(text[..<startIndex]) + replacement + String(text[endIndex...])
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
}
