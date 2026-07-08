import Foundation
import MLXLMCommon
import TesseractHighlight
import os

// MARK: - Panel value types

/// A named color role of the Code Accent Palette — the app-side mirror of the
/// Rust engine's `TokenRole`, so no FFI type leaks past the projection.
nonisolated enum CodeTokenRole: Sendable, Hashable {
    case plain, keyword, string, number, constant, comment, type, function, attribute, variable
}

/// One run of same-styled text within a panel row. `emphasized` marks the
/// word-level changed range inside a modified diff line.
nonisolated struct PanelSpan: Sendable, Hashable {
    let text: String
    let role: CodeTokenRole
    var emphasized = false
}

/// One row of code-like panel content: an optional line-number gutter pair
/// plus highlighted spans. Diff panels use `kind` for the ± tint; code panels
/// are all `.context` with only `newLine` set.
nonisolated struct PanelCodeRow: Sendable, Hashable, Identifiable {
    enum Kind: Sendable, Hashable {
        case context, added, removed
    }

    /// Position in the panel — stable because a committed result is frozen.
    let id: Int
    let kind: Kind
    let oldLine: Int?
    let newLine: Int?
    let spans: [PanelSpan]
}

/// One structured web-search result row.
nonisolated struct PanelSearchHit: Sendable, Hashable, Identifiable {
    let id: Int
    let title: String
    let url: String
    let snippet: String
}

// MARK: - ToolPanel

/// The Tool Panel model (PRD #200): a typed, render-ready description of one
/// tool call's expanded body, projected from the exact call + result. Views
/// are thin renderers over this — every decision (panel kind, rows, tints,
/// line numbers) is made here, in pure value logic.
nonisolated enum ToolPanel: Sendable, Hashable {
    /// The edit tool's syntax-highlighted diff with ± rows and inline emphasis.
    case diff(path: String, rows: [PanelCodeRow])
    /// A line-numbered, highlighted file slice (read / write).
    case code(path: String?, rows: [PanelCodeRow], footnote: String?)
    /// Monospaced text as returned (ls, page_map, find, evaluate, use_skill…).
    case text(String)
    /// Structured web-search results; `fallbackText` when the engine returned
    /// prose instead of result rows.
    case search(results: [PanelSearchHit], fallbackText: String?)
    /// A fetched page: rendered markdown by default, exact raw text behind
    /// the panel header's Rendered · Raw switch.
    case page(title: String?, url: String?, body: String, raw: String)
    /// Compact status lines (navigate, back, click, type, tabs, screenshot).
    case status(lines: [String])
    /// A failed call: the error text, rendered with the semantic error tint.
    case error(String)
    /// Unknown/external tools and detail-less legacy messages: pretty-printed
    /// arguments plus result text in the new visual language.
    case generic(resultText: String?)
}

// MARK: - Panel Cap

/// The Panel Cap (PRD #200): long content shows its first ~40 lines with a
/// "Show N more lines" row expanding the rest inline. Never a nested scroll
/// view — the transcript remains the only scroller.
nonisolated enum PanelCap {
    static let visibleLines = 40
    /// Rows within this margin over the cap render in full — "Show 2 more
    /// lines" is worse than the two lines.
    static let slack = 8

    /// The rows to show and how many stay hidden. `expanded` is the user's
    /// per-panel "Show more" state.
    static func split<Row>(_ rows: [Row], expanded: Bool) -> (visible: [Row], hidden: Int) {
        guard !expanded, rows.count > visibleLines + slack else { return (rows, 0) }
        return (Array(rows.prefix(visibleLines)), rows.count - visibleLines)
    }

    /// The cap for markdown source (rendered pages): cutting at a raw line 40
    /// can bisect a code fence and render broken markdown, so the cut point
    /// extends to the closing fence when line 40 falls inside one.
    static func splitMarkdown(
        _ lines: [Substring], expanded: Bool
    ) -> (visible: [Substring], hidden: Int) {
        guard !expanded, lines.count > visibleLines + slack else { return (lines, 0) }

        var cut = visibleLines
        if fenceCount(in: lines[..<cut]).isMultiple(of: 2) == false {
            // Inside a fence — extend to just past its closing line.
            if let closing = lines[cut...].firstIndex(where: isFenceLine) {
                cut = closing + 1
            } else {
                return (lines, 0)  // Unterminated fence: show everything.
            }
        }
        guard lines.count > cut + slack else { return (lines, 0) }
        return (Array(lines.prefix(cut)), lines.count - cut)
    }

    private static func fenceCount(in lines: ArraySlice<Substring>) -> Int {
        lines.count(where: isFenceLine)
    }

    private static func isFenceLine(_ line: Substring) -> Bool {
        let trimmed = line.drop(while: { $0 == " " })
        return trimmed.hasPrefix("```") || trimmed.hasPrefix("~~~")
    }
}

// MARK: - ToolPanelBuilder

/// Builds the Tool Panel for a committed tool call. Pure value logic over
/// (call, result) — memoized like `ToolDisplayHelpers.displayProps`, because
/// panels can highlight thousands of lines and the streaming tail-patch
/// reprojects the active turn ~20×/sec.
nonisolated enum ToolPanelBuilder {

    static func panel(for info: ToolCallInfo, result: ToolResultMessage) -> ToolPanel {
        let key = CacheKey(info: info, resultID: result.id)
        return cache.withLock { cache in
            if let cached = cache[key] { return cached }
            let panel = build(info: info, result: result)
            if cache.count >= cacheCap { cache.removeAll(keepingCapacity: true) }
            cache[key] = panel
            return panel
        }
    }

    // MARK: Dispatch

    private static func build(info: ToolCallInfo, result: ToolResultMessage) -> ToolPanel {
        let text = result.content.textContent
        if result.isError {
            return .error(text)
        }

        let args = info.parsedArguments
        switch info.name.lowercased() {
        case "edit":
            return editPanel(details: result.details?.editDetails, resultText: text)
        case "read":
            return readPanel(args: args, resultText: text)
        case "write":
            return writePanel(args: args)
        case "ls", "use_skill", "browser.page_map", "browser.find", "browser.evaluate":
            return .text(text)
        case "browser.search":
            return searchPanel(resultText: text)
        case "browser.fetch", "browser.read_page":
            return pagePanel(resultText: text)
        case "browser.navigate", "browser.back", "browser.click", "browser.type",
            "browser.tabs", "browser.screenshot":
            return .status(lines: statusLines(text))
        default:
            return .generic(resultText: text.isEmpty ? nil : text)
        }
    }

    // MARK: Edit → diff

    private static func editPanel(details: EditToolDetails?, resultText: String) -> ToolPanel {
        guard let details else {
            // Legacy conversations: no typed details → generic transparency.
            return .generic(resultText: resultText.isEmpty ? nil : resultText)
        }
        // Recompute from the exact replaced text via the Rust engine — the
        // stored unified diff has no word-level runs. Both fragments start at
        // `firstChangedLine`, so both gutters offset by it.
        let offset = details.firstChangedLine - 1
        let rows = diffHighlighted(
            old: details.oldText, new: details.newText, languageHint: details.path
        )
        .enumerated().map { index, row in
            PanelCodeRow(
                id: index,
                kind: rowKind(row.kind),
                oldLine: row.oldLine.map { Int($0) + offset },
                newLine: row.newLine.map { Int($0) + offset },
                spans: row.segments.map {
                    PanelSpan(text: $0.text, role: tokenRole($0.role), emphasized: $0.emphasized)
                }
            )
        }
        return .diff(path: details.path, rows: rows)
    }

    // MARK: Read → numbered slice

    /// The read tool returns `cat -n` numbered lines with occasional bracketed
    /// notice lines. Parsing the numbers back out (rather than trusting a
    /// side channel) keeps the panel faithful to what the model received and
    /// works for legacy messages too.
    private static func readPanel(args: [String: JSONValue]?, resultText: String) -> ToolPanel {
        let path = args?.string(for: "path")
        var numbered: [(number: Int, content: String)] = []
        var footnotes: [String] = []

        for line in resultText.split(separator: "\n", omittingEmptySubsequences: false) {
            if let parsed = parseNumberedLine(line) {
                numbered.append(parsed)
            } else if !line.trimmingCharacters(in: .whitespaces).isEmpty {
                footnotes.append(String(line))
            }
        }
        guard !numbered.isEmpty else {
            // "(empty file)", image reads, unexpected shapes.
            return .text(resultText)
        }

        let code = numbered.map(\.content).joined(separator: "\n")
        let highlighted = highlightRows(
            code: code, languageHint: path ?? "", numbers: numbered.map(\.number))
        return .code(
            path: path, rows: highlighted,
            footnote: footnotes.isEmpty ? nil : footnotes.joined(separator: "\n"))
    }

    /// `"    12\tcontent"` → `(12, "content")`.
    private static func parseNumberedLine(_ line: Substring) -> (number: Int, content: String)? {
        guard let tab = line.firstIndex(of: "\t") else { return nil }
        let prefix = line[..<tab].trimmingCharacters(in: .whitespaces)
        guard !prefix.isEmpty, prefix.allSatisfy(\.isNumber), let number = Int(prefix) else {
            return nil
        }
        return (number, String(line[line.index(after: tab)...]))
    }

    // MARK: Write → full content

    private static func writePanel(args: [String: JSONValue]?) -> ToolPanel {
        let path = args?.string(for: "path")
        guard let content = args?.string(for: "content"), !content.isEmpty else {
            return .code(path: path, rows: [], footnote: nil)
        }
        let lineCount = content.split(separator: "\n", omittingEmptySubsequences: false).count
        let rows = highlightRows(
            code: content, languageHint: path ?? "", numbers: Array(1...lineCount))
        return .code(path: path, rows: rows, footnote: nil)
    }

    // MARK: Search → structured hits

    /// Parses the executor's own rendering: `"N. Title\n   url\n   snippet"`
    /// blocks separated by blank lines. Anything else (the "No structured
    /// results" fallback, challenge pages) passes through as text.
    private static func searchPanel(resultText: String) -> ToolPanel {
        var hits: [PanelSearchHit] = []
        for block in resultText.components(separatedBy: "\n\n") {
            let lines = block.split(separator: "\n").map {
                $0.trimmingCharacters(in: .whitespaces)
            }
            guard lines.count >= 2,
                let head = lines.first,
                let dot = head.firstIndex(of: "."),
                let index = Int(head[..<dot])
            else {
                return .search(results: [], fallbackText: resultText)
            }
            hits.append(
                PanelSearchHit(
                    id: index,
                    title: head[head.index(after: dot)...].trimmingCharacters(in: .whitespaces),
                    url: lines[1],
                    snippet: lines.dropFirst(2).joined(separator: " ")
                ))
        }
        guard !hits.isEmpty else {
            return .search(results: [], fallbackText: resultText)
        }
        return .search(results: hits, fallbackText: nil)
    }

    // MARK: Fetch / read_page → rendered page

    /// Splits the executor's `"Title: …\nURL: …\n\n<body>"` header off; the
    /// body renders as markdown, the exact text stays behind the Raw switch.
    private static func pagePanel(resultText: String) -> ToolPanel {
        var title: String?
        var url: String?
        var body = resultText

        let lines = resultText.split(separator: "\n", omittingEmptySubsequences: false)
        if lines.count >= 2, lines[0].hasPrefix("Title: "), lines[1].hasPrefix("URL: ") {
            title = String(lines[0].dropFirst("Title: ".count))
            url = String(lines[1].dropFirst("URL: ".count))
            body = lines.dropFirst(2).joined(separator: "\n")
                .trimmingCharacters(in: .newlines)
        }
        return .page(title: title, url: url, body: body, raw: resultText)
    }

    // MARK: Status

    private static func statusLines(_ text: String) -> [String] {
        text.split(separator: "\n").map(String.init)
            .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
    }

    // MARK: Highlight bridging

    /// Highlight `code` and zip the result with per-row gutter numbers.
    private static func highlightRows(
        code: String, languageHint: String, numbers: [Int]
    ) -> [PanelCodeRow] {
        let lines = TesseractHighlight.highlight(code: code, languageHint: languageHint)
        return zip(numbers, lines).enumerated().map { index, pair in
            let (number, line) = pair
            return PanelCodeRow(
                id: index,
                kind: .context,
                oldLine: nil,
                newLine: number,
                spans: line.spans.map { PanelSpan(text: $0.text, role: tokenRole($0.role)) }
            )
        }
    }

    private static func tokenRole(_ role: TokenRole) -> CodeTokenRole {
        switch role {
        case .plain: .plain
        case .keyword: .keyword
        case .stringLit: .string
        case .number: .number
        case .constant: .constant
        case .comment: .comment
        case .typeName: .type
        case .functionName: .function
        case .attribute: .attribute
        case .variableName: .variable
        }
    }

    private static func rowKind(_ kind: DiffRowKind) -> PanelCodeRow.Kind {
        switch kind {
        case .context: .context
        case .added: .added
        case .removed: .removed
        }
    }

    // MARK: Memoization

    private struct CacheKey: Hashable {
        let info: ToolCallInfo
        let resultID: UUID
    }

    private static let cache = OSAllocatedUnfairLock(initialState: [CacheKey: ToolPanel]())
    private static let cacheCap = 64
}
