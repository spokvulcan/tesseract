//
//  ToolPanelBuilderTests.swift
//  tesseractTests
//
//  The panel-projection seam of PRD #200: given a committed (tool call,
//  result) pair, the Tool Panel model says exactly what a user would see —
//  which panel kind, which rows, which tints, which line numbers, where the
//  Panel Cap falls. Prior art: ToolDisplayHelpersTests.
//

import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

struct ToolPanelBuilderTests {

    // MARK: - Fixtures

    private func call(_ name: String, args: [String: String] = [:]) -> ToolCallInfo {
        let json =
            (try? JSONEncoder().encode(args))
            .flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
        return ToolCallInfo(id: "call-\(name)", name: name, argumentsJSON: json)
    }

    private func result(
        _ tool: String, text: String, isError: Bool = false, details: ToolResultDetails? = nil
    ) -> ToolResultMessage {
        ToolResultMessage(
            toolCallId: "call-\(tool)", toolName: tool,
            content: [.text(text)], isError: isError, details: details
        )
    }

    // MARK: - Edit → diff panel

    @Test func editWithDetailsProjectsHighlightedDiffRows() throws {
        let details = ToolResultDetails.edit(
            EditToolDetails(
                path: "Sources/App.swift",
                diff: "irrelevant here",
                firstChangedLine: 10,
                oldText: "let count = 1\nlet keep = true\n",
                newText: "let count = 2\nlet keep = true\n"
            ))
        let panel = ToolPanelBuilder.panel(
            for: call("edit", args: ["path": "Sources/App.swift"]),
            result: result("edit", text: "Successfully replaced text.", details: details)
        )

        guard case .diff(let path, let rows) = panel else {
            Issue.record("Expected diff panel, got \(panel)")
            return
        }
        #expect(path == "Sources/App.swift")

        // Removed then added for the changed line, context for the kept line.
        let removed = try #require(rows.first { $0.kind == .removed })
        let added = try #require(rows.first { $0.kind == .added })
        #expect(rows.contains { $0.kind == .context })

        // Gutter numbers offset by firstChangedLine (fragment line 1 → file line 10).
        #expect(removed.oldLine == 10)
        #expect(removed.newLine == nil)
        #expect(added.newLine == 10)
        #expect(added.oldLine == nil)

        // Row text is lossless, word-level emphasis marks the changed token,
        // and the Swift keyword got a syntax role.
        #expect(added.spans.map(\.text).joined() == "let count = 2")
        #expect(added.spans.contains { $0.emphasized && $0.text.contains("2") })
        #expect(added.spans.contains { $0.role == .keyword })
    }

    @Test func editWithoutDetailsFallsBackToGeneric() {
        let panel = ToolPanelBuilder.panel(
            for: call("edit", args: ["path": "a.txt"]),
            result: result("edit", text: "Successfully replaced text in a.txt.")
        )
        #expect(panel == .generic(resultText: "Successfully replaced text in a.txt."))
    }

    // MARK: - Read → numbered slice

    @Test func readProjectsLineNumbersFromTheActualSlice() throws {
        let text = "    41\tfunc run() {\n    42\t    start()\n    43\t}"
        let panel = ToolPanelBuilder.panel(
            for: call("read", args: ["path": "src/main.swift"]),
            result: result("read", text: text)
        )

        guard case .code(let path, let rows, let footnote) = panel else {
            Issue.record("Expected code panel, got \(panel)")
            return
        }
        #expect(path == "src/main.swift")
        #expect(footnote == nil)
        #expect(rows.map(\.newLine) == [41, 42, 43])
        #expect(rows.allSatisfy { $0.kind == .context })
        #expect(rows[0].spans.map(\.text).joined() == "func run() {")
        #expect(rows[0].spans.contains { $0.role == .keyword })
    }

    @Test func readTruncationNoticeBecomesFootnote() throws {
        let text = "     1\ta\n     2\tb\n\n[Showing lines 1-2 of 900. Use offset=3 to continue.]"
        let panel = ToolPanelBuilder.panel(
            for: call("read", args: ["path": "big.txt"]),
            result: result("read", text: text)
        )
        guard case .code(_, let rows, let footnote) = panel else {
            Issue.record("Expected code panel, got \(panel)")
            return
        }
        #expect(rows.count == 2)
        #expect(footnote == "[Showing lines 1-2 of 900. Use offset=3 to continue.]")
    }

    @Test func emptyFileReadStaysTextual() {
        let panel = ToolPanelBuilder.panel(
            for: call("read", args: ["path": "empty.txt"]),
            result: result("read", text: "(empty file)")
        )
        #expect(panel == .text("(empty file)"))
    }

    // MARK: - Write → full content

    @Test func writeProjectsItsArgumentContentNumberedFromOne() throws {
        let panel = ToolPanelBuilder.panel(
            for: call("write", args: ["path": "hello.py", "content": "def hi():\n    pass"]),
            result: result("write", text: "Wrote 2 lines to hello.py.")
        )
        guard case .code(let path, let rows, _) = panel else {
            Issue.record("Expected code panel, got \(panel)")
            return
        }
        #expect(path == "hello.py")
        #expect(rows.map(\.newLine) == [1, 2])
        #expect(rows[0].spans.contains { $0.role == .keyword })  // def
    }

    // MARK: - Listing tools stay monospaced text

    @Test func lsAndSkillAndPageMapAreTextPanels() {
        for tool in ["ls", "use_skill", "browser.page_map", "browser.find", "browser.evaluate"] {
            let panel = ToolPanelBuilder.panel(
                for: call(tool), result: result(tool, text: "line one\nline two"))
            #expect(panel == .text("line one\nline two"), "tool: \(tool)")
        }
    }

    // MARK: - Search → structured hits

    @Test func searchResultTextParsesIntoStructuredHits() throws {
        let text = """
            1. Swift 6 announced
               https://swift.org/blog/swift-6
               The next major release brings full data-race safety.

            2. Migration guide
               https://swift.org/migration
               How to adopt. Strict concurrency. Today.
            """
        let panel = ToolPanelBuilder.panel(
            for: call("browser.search", args: ["query": "swift 6"]),
            result: result("browser.search", text: text)
        )
        guard case .search(let hits, let fallback) = panel else {
            Issue.record("Expected search panel, got \(panel)")
            return
        }
        #expect(fallback == nil)
        #expect(hits.count == 2)
        #expect(hits[0].title == "Swift 6 announced")
        #expect(hits[0].url == "https://swift.org/blog/swift-6")
        #expect(hits[0].snippet == "The next major release brings full data-race safety.")
        #expect(hits[1].id == 2)
    }

    @Test func unstructuredSearchTextPassesThroughAsFallback() {
        let text = "No structured results for \"x\". Showing the results page text:\n\nSome page."
        let panel = ToolPanelBuilder.panel(
            for: call("browser.search"), result: result("browser.search", text: text))
        #expect(panel == .search(results: [], fallbackText: text))
    }

    // MARK: - Fetch / read_page → page panel

    @Test func fetchSplitsHeaderAndKeepsRawIntact() throws {
        let text = "Title: Example\nURL: https://example.com/\n\n# Heading\n\nBody text."
        let panel = ToolPanelBuilder.panel(
            for: call("browser.fetch", args: ["url": "example.com"]),
            result: result("browser.fetch", text: text)
        )
        guard case .page(let title, let url, let body, let raw) = panel else {
            Issue.record("Expected page panel, got \(panel)")
            return
        }
        #expect(title == "Example")
        #expect(url == "https://example.com/")
        #expect(body == "# Heading\n\nBody text.")
        #expect(raw == text)
    }

    @Test func headerlessFetchBodyIsItsOwnRaw() {
        let panel = ToolPanelBuilder.panel(
            for: call("browser.read_page"), result: result("browser.read_page", text: "just text"))
        #expect(panel == .page(title: nil, url: nil, body: "just text", raw: "just text"))
    }

    // MARK: - Navigation → status panel

    @Test func navigationToolsProjectCompactStatusLines() {
        let text =
            "Loaded https://example.com\nTitle: Example\n\nUse read_page to read the content."
        let panel = ToolPanelBuilder.panel(
            for: call("browser.navigate", args: ["url": "example.com"]),
            result: result("browser.navigate", text: text)
        )
        #expect(
            panel
                == .status(lines: [
                    "Loaded https://example.com",
                    "Title: Example",
                    "Use read_page to read the content.",
                ]))
    }

    // MARK: - Errors and unknowns

    @Test func errorResultsProjectTheErrorPanelRegardlessOfTool() {
        let panel = ToolPanelBuilder.panel(
            for: call("edit", args: ["path": "a.txt"]),
            result: result("edit", text: "Could not find the exact text in a.txt.", isError: true)
        )
        #expect(panel == .error("Could not find the exact text in a.txt."))
    }

    @Test func unknownToolsProjectTheGenericPanel() {
        let panel = ToolPanelBuilder.panel(
            for: call("context7.lookup", args: ["library": "swiftui"]),
            result: result("context7.lookup", text: "docs…")
        )
        #expect(panel == .generic(resultText: "docs…"))
    }

    // MARK: - Panel Cap

    @Test func panelCapSplitsAtFortyWithSlack() {
        let rows = Array(0..<200)
        let (visible, hidden) = PanelCap.split(rows, expanded: false)
        #expect(visible.count == 40)
        #expect(hidden == 160)

        // Expanded shows everything.
        let (all, none) = PanelCap.split(rows, expanded: true)
        #expect(all.count == 200)
        #expect(none == 0)

        // Within the slack margin nothing is hidden — "Show 3 more lines"
        // costs more than the three lines.
        let (nearCap, nearHidden) = PanelCap.split(Array(0..<45), expanded: false)
        #expect(nearCap.count == 45)
        #expect(nearHidden == 0)
    }

    @Test func markdownCapNeverBisectsACodeFence() {
        // A fence opens at line 39 (0-indexed 38) and closes at line 50 —
        // the raw cut at 40 would land inside it.
        var lines: [String] = Array(repeating: "prose", count: 38)
        lines.append("```swift")
        lines.append(contentsOf: Array(repeating: "let x = 1", count: 11))
        lines.append("```")
        lines.append(contentsOf: Array(repeating: "after", count: 60))
        let substrings = lines.joined(separator: "\n")
            .split(separator: "\n", omittingEmptySubsequences: false)

        let (visible, hidden) = PanelCap.splitMarkdown(Array(substrings), expanded: false)
        // The cut extended to just past the closing fence (line 51).
        #expect(visible.count == 51)
        #expect(visible.last == "```")
        #expect(hidden == substrings.count - 51)

        // Fence-free markdown behaves like the plain cap.
        let plain = Array(
            Array(repeating: "p" as Substring, count: 100))
        let (plainVisible, plainHidden) = PanelCap.splitMarkdown(plain, expanded: false)
        #expect(plainVisible.count == 40)
        #expect(plainHidden == 60)

        // An unterminated fence shows everything rather than broken markdown.
        var open: [String] = Array(repeating: "prose", count: 39)
        open.append("```")
        open.append(contentsOf: Array(repeating: "code", count: 30))
        let openSub = open.joined(separator: "\n")
            .split(separator: "\n", omittingEmptySubsequences: false)
        let (allVisible, noneHidden) = PanelCap.splitMarkdown(Array(openSub), expanded: false)
        #expect(allVisible.count == openSub.count)
        #expect(noneHidden == 0)
    }
}
