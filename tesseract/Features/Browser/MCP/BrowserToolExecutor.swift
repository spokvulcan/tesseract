import Foundation
import MLXLMCommon

// MARK: - BrowserToolExecutor

/// Executes a browser tool call against a caller's **Browser Session**. The one
/// place tool names map to browser operations; the MCP server owns the wire,
/// this owns behavior. Tool-level failures come back as `isError` results
/// (never thrown to the protocol layer).
@MainActor
final class BrowserToolExecutor {

    private let browser: AgentBrowser

    /// Default character budget for a single `read_page` / `fetch` chunk — well
    /// under a typical agent's MCP response cap, paginated beyond it.
    static let defaultReadChars = 20_000
    private static let maxEvaluateChars = 20_000

    init(browser: AgentBrowser) {
        self.browser = browser
    }

    /// Names this executor can handle (mirrors the catalog).
    var toolNames: [String] { BrowserToolCatalog.all.map(\.name) }

    /// Dispatch a call. Runs serialized within the session so a single client's
    /// pipelined calls don't interleave.
    func call(
        _ name: String,
        session: BrowserSession,
        arguments: [String: JSONValue]
    ) async -> BrowserToolResult {
        await session.serialized {
            await self.run(name, session: session, arguments: arguments)
        }
    }

    // MARK: - Dispatch

    private func run(
        _ name: String,
        session: BrowserSession,
        arguments args: [String: JSONValue]
    ) async -> BrowserToolResult {
        do {
            switch name {
            case "navigate": return try await navigate(session, args)
            case "back": return try await back(session)
            case "read_page": return try await readPage(session, args)
            case "page_map": return try await pageMap(session, args)
            case "click": return try await click(session, args)
            case "type": return try await type(session, args)
            case "find": return try await find(session, args)
            case "tabs": return await tabs(session, args)
            case "evaluate": return try await evaluate(session, args)
            case "screenshot": return try await screenshot(session)
            case "search": return try await search(args)
            case "fetch": return try await fetch(args)
            default:
                return .error("Unknown browser tool: \(name)")
            }
        } catch {
            return .error(Self.describe(error))
        }
    }

    // MARK: - Tools

    private func navigate(_ session: BrowserSession, _ args: [String: JSONValue]) async throws
        -> BrowserToolResult
    {
        guard let raw = args.string(for: "url"), let url = Self.normalizeURL(raw) else {
            return .error("navigate requires a valid `url`.")
        }
        let tab = session.requireActiveTab()
        let status = try await tab.navigate(to: url)
        session.reflect()
        return .text(
            "Loaded \(status.url)\nTitle: \(status.title)\n\n"
                + "Use read_page to read the content, or page_map to interact.")
    }

    private func back(_ session: BrowserSession) async throws -> BrowserToolResult {
        guard let tab = session.activeTab else { return .error("No page open.") }
        let status = try await tab.goBack()
        session.reflect()
        return .text("Back to \(status.url)\nTitle: \(status.title)")
    }

    private func readPage(_ session: BrowserSession, _ args: [String: JSONValue]) async throws
        -> BrowserToolResult
    {
        guard let tab = session.activeTab else {
            return .error("No page open. Use navigate first.")
        }
        let content = try await tab.pageContent()
        let cursor = args.int(for: "cursor") ?? 0
        let maxChars = args.int(for: "max_chars") ?? Self.defaultReadChars
        let chunk = PageReadPaginator.paginate(content.content, cursor: cursor, maxChars: maxChars)

        var out = ""
        if cursor == 0 {
            out += "Title: \(content.title)\nURL: \(content.url.absoluteString)\n\n"
        }
        out += chunk.text
        if let next = chunk.nextCursor {
            out +=
                "\n\n[\(chunk.totalChars) chars total — call read_page again with "
                + "cursor=\(next) for more.]"
        }
        return .text(out)
    }

    private func pageMap(_ session: BrowserSession, _ args: [String: JSONValue]) async throws
        -> BrowserToolResult
    {
        guard let tab = session.activeTab else {
            return .error("No page open. Use navigate first.")
        }
        let elements = try await tab.pageMap()
        let maxElements = args.int(for: "max_elements") ?? 200
        let outline = PageMapFormatter.format(elements, maxElements: maxElements)
        return .text("URL: \(tab.status.url)\n\n\(outline)")
    }

    private func click(_ session: BrowserSession, _ args: [String: JSONValue]) async throws
        -> BrowserToolResult
    {
        guard let tab = session.activeTab else { return .error("No page open.") }
        guard let ref = args.int(for: "ref") else { return .error("click requires `ref`.") }
        let status = try await tab.click(ref: ref)
        session.reflect()
        return .text("Clicked [\(ref)]. Now at \(status.url)\nTitle: \(status.title)")
    }

    private func type(_ session: BrowserSession, _ args: [String: JSONValue]) async throws
        -> BrowserToolResult
    {
        guard let tab = session.activeTab else { return .error("No page open.") }
        guard let ref = args.int(for: "ref"), let text = args.string(for: "text") else {
            return .error("type requires `ref` and `text`.")
        }
        let submit = args.bool(for: "submit") ?? false
        let status = try await tab.type(ref: ref, text: text, submit: submit)
        session.reflect()
        let suffix = submit ? " Now at \(status.url)\nTitle: \(status.title)" : ""
        return .text("Typed into [\(ref)].\(suffix)")
    }

    private func find(_ session: BrowserSession, _ args: [String: JSONValue]) async throws
        -> BrowserToolResult
    {
        guard let tab = session.activeTab else { return .error("No page open.") }
        guard let query = args.string(for: "query") else { return .error("find requires `query`.") }
        let maxResults = min(max(args.int(for: "max_results") ?? 10, 1), 50)
        let (count, hits) = try await tab.find(query, maxResults: maxResults)
        if count == 0 { return .text("No matches for \"\(query)\".") }
        var out = "\(count) match(es) for \"\(query)\":\n"
        out += hits.map { "- \($0)" }.joined(separator: "\n")
        if count > hits.count {
            out += "\n(\(count - hits.count) more not shown.)"
        }
        return .text(out)
    }

    private func tabs(_ session: BrowserSession, _ args: [String: JSONValue]) async
        -> BrowserToolResult
    {
        let action = args.string(for: "action") ?? "list"
        switch action {
        case "open":
            let tab = session.openTab()
            if let raw = args.string(for: "url"), let url = Self.normalizeURL(raw) {
                do { try await tab.navigate(to: url) } catch {
                    return .error("Opened a tab but navigation failed: \(Self.describe(error))")
                }
                session.reflect()
            }
            return .text("Opened tab \(tab.id).\n\n" + Self.renderTabs(session.tabSummaries()))
        case "select":
            guard let id = args.string(for: "tab_id"), session.selectTab(id: id) else {
                return .error("select requires a valid `tab_id` (see tabs action=list).")
            }
            return .text("Selected tab \(id).\n\n" + Self.renderTabs(session.tabSummaries()))
        case "close":
            guard let id = args.string(for: "tab_id"), session.closeTab(id: id) else {
                return .error("close requires a valid `tab_id` (see tabs action=list).")
            }
            return .text("Closed tab \(id).\n\n" + Self.renderTabs(session.tabSummaries()))
        default:
            return .text(Self.renderTabs(session.tabSummaries()))
        }
    }

    private func evaluate(_ session: BrowserSession, _ args: [String: JSONValue]) async throws
        -> BrowserToolResult
    {
        guard let tab = session.activeTab else { return .error("No page open.") }
        guard let script = args.string(for: "script") else {
            return .error("evaluate requires `script`.")
        }
        var result = try await tab.evaluate(script)
        if result.count > Self.maxEvaluateChars {
            result = String(result.prefix(Self.maxEvaluateChars)) + "\n[result truncated]"
        }
        return .text(result)
    }

    private func screenshot(_ session: BrowserSession) async throws -> BrowserToolResult {
        guard let tab = session.activeTab else { return .error("No page open.") }
        let data = try await tab.screenshot()
        return .blocks([
            .image(data: data, mimeType: "image/png"),
            .text("Screenshot of \(tab.status.url)"),
        ])
    }

    private func search(_ args: [String: JSONValue]) async throws -> BrowserToolResult {
        guard
            let query = args.string(for: "query")?.trimmingCharacters(in: .whitespacesAndNewlines),
            !query.isEmpty
        else { return .error("search requires a non-empty `query`.") }
        let maxResults = min(max(args.int(for: "max_results") ?? 5, 1), 10)
        let results = try await DuckDuckGoClient.search(query: query, maxResults: maxResults)
        if results.isEmpty { return .text("No results for \"\(query)\".") }
        let rendered = results.enumerated().map { index, result in
            "\(index + 1). \(result.title)\n   \(result.url)\n   \(result.snippet)"
        }.joined(separator: "\n\n")
        return .text(rendered)
    }

    private func fetch(_ args: [String: JSONValue]) async throws -> BrowserToolResult {
        guard let raw = args.string(for: "url"), let url = Self.normalizeURL(raw) else {
            return .error("fetch requires a valid `url`.")
        }
        let content = try await EphemeralPageReader.read(url: url)
        let maxChars = args.int(for: "max_chars") ?? Self.defaultReadChars
        let chunk = PageReadPaginator.paginate(content.content, cursor: 0, maxChars: maxChars)
        var out = "Title: \(content.title)\nURL: \(content.url.absoluteString)\n\n\(chunk.text)"
        if chunk.nextCursor != nil {
            out += "\n\n[Content truncated at \(maxChars) characters.]"
        }
        return .text(out)
    }

    // MARK: - Helpers

    private static func renderTabs(_ tabs: [TabSummary]) -> String {
        guard !tabs.isEmpty else { return "No tabs open." }
        return tabs.map { tab in
            let marker = tab.active ? "*" : " "
            let title = tab.title.isEmpty ? tab.url : tab.title
            return "\(marker) \(tab.id)  \(title)  —  \(tab.url)"
        }.joined(separator: "\n")
    }

    private static func normalizeURL(_ raw: String) -> URL? {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        if trimmed.contains("://") { return URL(string: trimmed) }
        return URL(string: "https://\(trimmed)")
    }

    private static func describe(_ error: Error) -> String {
        (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
    }
}
