import Foundation
import AppKit
import WebKit

// MARK: - Value types

/// Where a tab currently is.
nonisolated struct PageStatus: Sendable, Equatable {
    let url: String
    let title: String
}

/// Errors surfaced by tab operations. Mapped to tool-result errors (never
/// thrown across the MCP wire as protocol errors).
nonisolated enum BrowserTabError: LocalizedError, Sendable {
    case timeout
    case navigationFailed(String)
    case noHistory
    case elementNotFound(Int)
    case scriptFailed(String)
    case emptyResult

    var errorDescription: String? {
        switch self {
        case .timeout: "Navigation timed out"
        case .navigationFailed(let m): "Navigation failed: \(m)"
        case .noHistory: "No page to go back to"
        case .elementNotFound(let ref):
            "No element with ref [\(ref)] on this page — the page may have changed; call page_map again"
        case .scriptFailed(let m): "Script failed: \(m)"
        case .emptyResult: "The page produced no content"
        }
    }
}

private nonisolated struct ClickResult: Codable {
    let ok: Bool; let tag: String?; let error: String?
}
private nonisolated struct TypeResult: Codable { let ok: Bool; let error: String? }
private nonisolated struct FindResult: Codable { let count: Int; let hits: [String] }

// MARK: - BrowserTab

/// One tab in a **Browser Session**: a single `WebPage` over the shared **Agent
/// Profile**, plus the operations the browser tools drive. Owns no window — the
/// `AgentBrowserPresenting` port renders it — so the whole tab is exercisable
/// headlessly in tests via a no-op presenter.
///
/// Reads and DOM scripting run in an isolated `WKContentWorld`
/// (`.defaultClient`): full DOM access, invisible to and shielded from the
/// page's own JavaScript.
@MainActor
final class BrowserTab {

    let id: String
    let page: WebPage

    /// Distilled Page Read cache, keyed by the URL it was extracted from, so
    /// paginated `read_page` calls don't re-run extraction (and don't drift).
    private var readCache: (url: String, content: WebContentExtractor.ExtractedContent)?

    private let navigationTimeout: Duration
    private let hydrationSettle: Duration

    init(
        id: String = UUID().uuidString,
        configuration: WebPage.Configuration,
        navigationTimeout: Duration = .seconds(30),
        hydrationSettle: Duration = .milliseconds(700)
    ) {
        self.id = id
        self.navigationTimeout = navigationTimeout
        self.hydrationSettle = hydrationSettle
        let page = WebPage(configuration: configuration)
        page.customUserAgent = AgentProfile.userAgent
        self.page = page
    }

    var status: PageStatus {
        PageStatus(url: page.url?.absoluteString ?? "about:blank", title: page.title)
    }

    // MARK: - Navigation

    /// Load a URL and wait for the page to finish (plus a short hydration
    /// settle for SPA frameworks). Unlike anonymous `fetch`, the Agent Browser
    /// applies no SSRF filtering — it is the user's own visible browser and may
    /// reach anything the user could, localhost included.
    @discardableResult
    func navigate(to url: URL) async throws -> PageStatus {
        readCache = nil
        try await runNavigation(page.load(URLRequest(url: url)))
        return status
    }

    /// Go back one entry in this tab's history.
    @discardableResult
    func goBack() async throws -> PageStatus {
        guard let previous = page.backForwardList.backList.last else {
            throw BrowserTabError.noHistory
        }
        readCache = nil
        try await runNavigation(page.load(previous))
        return status
    }

    private func runNavigation(
        _ events: some AsyncSequence<WebPage.NavigationEvent, any Error>
    ) async throws {
        let timeoutTask = Task { try await Task.sleep(for: navigationTimeout) }
        defer { timeoutTask.cancel() }

        do {
            for try await event in events where event == .finished {
                // Allow SPA frameworks a moment to hydrate before we read.
                try? await Task.sleep(for: hydrationSettle)
                return
            }
            // Sequence ended without an explicit `.finished` — treat the
            // final committed state as loaded rather than failing. Navigation
            // failures arrive as a thrown `NavigationError`, handled below.
        } catch is CancellationError {
            throw BrowserTabError.timeout
        } catch let error as WebPage.NavigationError {
            throw BrowserTabError.navigationFailed(error.localizedDescription)
        }
    }

    // MARK: - Page Read

    /// Distilled Markdown for the current page (cached per URL). The executor
    /// paginates the result via ``PageReadPaginator``.
    func pageContent() async throws -> WebContentExtractor.ExtractedContent {
        let currentURL = page.url?.absoluteString ?? "about:blank"
        if let cached = readCache, cached.url == currentURL {
            return cached.content
        }

        let html = try await currentHTML()
        let base = page.url ?? URL(string: "about:blank")!
        let content = await WebContentExtractor.extract(html: html, url: base)
        readCache = (currentURL, content)
        return content
    }

    /// Raw outer HTML of the live DOM, read in the isolated world.
    func currentHTML() async throws -> String {
        guard
            let html = try await page.callJavaScript(
                "return document.documentElement.outerHTML",
                contentWorld: .defaultClient
            ) as? String, !html.isEmpty
        else {
            throw BrowserTabError.emptyResult
        }
        return html
    }

    // MARK: - Page Map

    /// Build the interaction map: assign refs onto the live DOM and return the
    /// decoded elements. `PageMapFormatter` renders them for the agent.
    func pageMap() async throws -> [PageMapElement] {
        guard
            let json = try await page.callJavaScript(
                PageMapScript.source,
                contentWorld: .defaultClient
            ) as? String,
            let data = json.data(using: .utf8)
        else {
            throw BrowserTabError.scriptFailed("page map returned no data")
        }
        do {
            return try JSONDecoder().decode([PageMapElement].self, from: data)
        } catch {
            throw BrowserTabError.scriptFailed("could not decode page map: \(error)")
        }
    }

    // MARK: - Interaction

    /// Click the element carrying `data-tesseract-ref="ref"`, then wait for any
    /// resulting navigation to settle.
    @discardableResult
    func click(ref: Int) async throws -> PageStatus {
        let script = #"""
            const el = document.querySelector('[data-tesseract-ref="' + ref + '"]');
            if (!el) { return JSON.stringify({ ok: false, error: 'not-found' }); }
            el.scrollIntoView({ block: 'center' });
            el.focus({ preventScroll: true });
            el.click();
            return JSON.stringify({ ok: true, tag: el.tagName.toLowerCase() });
            """#
        let result = try await callDecoding(
            ClickResult.self, script, arguments: ["ref": ref])
        guard result.ok else { throw BrowserTabError.elementNotFound(ref) }
        readCache = nil
        await settleAfterInteraction()
        return status
    }

    /// Type `text` into the element, firing `input`/`change` so reactive
    /// frameworks observe it. When `submit` is set, dispatch an Enter keydown
    /// and submit the enclosing form.
    @discardableResult
    func type(ref: Int, text: String, submit: Bool) async throws -> PageStatus {
        let script = #"""
            const el = document.querySelector('[data-tesseract-ref="' + ref + '"]');
            if (!el) { return JSON.stringify({ ok: false, error: 'not-found' }); }
            el.focus({ preventScroll: true });
            if (el.isContentEditable) {
              el.textContent = text;
            } else {
              el.value = text;
            }
            el.dispatchEvent(new Event('input', { bubbles: true }));
            el.dispatchEvent(new Event('change', { bubbles: true }));
            if (submit) {
              el.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', code: 'Enter', bubbles: true }));
              if (el.form && typeof el.form.requestSubmit === 'function') {
                el.form.requestSubmit();
              } else if (el.form && typeof el.form.submit === 'function') {
                el.form.submit();
              }
            }
            return JSON.stringify({ ok: true });
            """#
        let result = try await callDecoding(
            TypeResult.self, script,
            arguments: ["ref": ref, "text": text, "submit": submit])
        guard result.ok else { throw BrowserTabError.elementNotFound(ref) }
        if submit {
            readCache = nil
            await settleAfterInteraction()
        }
        return status
    }

    // MARK: - Find

    /// Locate up to `maxResults` occurrences of `query` in the visible text,
    /// returning short surrounding snippets — a cheap locate aid that avoids
    /// re-reading the whole page.
    func find(_ query: String, maxResults: Int) async throws -> (count: Int, hits: [String]) {
        let script = #"""
            const q = String(query).toLowerCase();
            if (!q) { return JSON.stringify({ count: 0, hits: [] }); }
            // innerText is layout-dependent (empty on a page that hasn't laid out);
            // fall back to textContent so find works regardless of a host view.
            const body = document.body;
            const text = (body && (body.innerText || body.textContent)) || '';
            const hay = text.toLowerCase();
            const hits = [];
            let from = 0, count = 0;
            while (true) {
              const idx = hay.indexOf(q, from);
              if (idx < 0) break;
              count += 1;
              if (hits.length < maxResults) {
                const start = Math.max(0, idx - 60);
                const end = Math.min(text.length, idx + q.length + 60);
                let snip = text.slice(start, end).replace(/\s+/g, ' ').trim();
                if (start > 0) snip = '…' + snip;
                if (end < text.length) snip = snip + '…';
                hits.push(snip);
              }
              from = idx + q.length;
              if (count > 5000) break;
            }
            return JSON.stringify({ count: count, hits: hits });
            """#
        let result = try await callDecoding(
            FindResult.self, script,
            arguments: ["query": query, "maxResults": maxResults])
        return (result.count, result.hits)
    }

    // MARK: - Evaluate

    /// Run agent-supplied JavaScript in the isolated world and return a
    /// stringified result. The code-execution-over-tools lever: the agent can
    /// extract exactly what it needs instead of re-reading a rendered page.
    func evaluate(_ script: String) async throws -> String {
        do {
            let value = try await page.callJavaScript(
                script, contentWorld: .defaultClient)
            return Self.stringify(value)
        } catch {
            throw BrowserTabError.scriptFailed(error.localizedDescription)
        }
    }

    // MARK: - Screenshot

    /// PNG capture of the current page contents, re-encoded to guarantee the
    /// declared MIME. Width-capped to bound token cost when a VLM consumes it.
    func screenshot(maxWidth: CGFloat = 1200) async throws -> Data {
        let raw = try await page.exported(
            as: .image(region: .contents, snapshotWidth: maxWidth))
        if let rep = NSBitmapImageRep(data: raw),
            let png = rep.representation(using: .png, properties: [:])
        {
            return png
        }
        return raw
    }

    // MARK: - Private

    /// Wait briefly for an interaction-triggered navigation to begin and
    /// settle, so the returned status reflects where the click landed.
    private func settleAfterInteraction() async {
        try? await Task.sleep(for: .milliseconds(400))
        // If a navigation is in flight, let it finish (bounded).
        var waited: Duration = .zero
        let cap: Duration = .seconds(8)
        while page.isLoading && waited < cap {
            try? await Task.sleep(for: .milliseconds(100))
            waited += .milliseconds(100)
        }
    }

    private func callDecoding<T: Decodable>(
        _ type: T.Type,
        _ script: String,
        arguments: [String: Any]
    ) async throws -> T {
        let value = try await page.callJavaScript(
            script, arguments: arguments, contentWorld: .defaultClient)
        guard let json = value as? String, let data = json.data(using: .utf8) else {
            throw BrowserTabError.scriptFailed("script returned no JSON")
        }
        return try JSONDecoder().decode(T.self, from: data)
    }

    /// Best-effort stringification of a JS return value for `evaluate`.
    private static func stringify(_ value: Any?) -> String {
        switch value {
        case nil, is NSNull:
            return "undefined"
        case let s as String:
            return s
        case let n as NSNumber:
            return n.stringValue
        case let b as Bool:
            return b ? "true" : "false"
        default:
            if let value,
                JSONSerialization.isValidJSONObject(value),
                let data = try? JSONSerialization.data(
                    withJSONObject: value, options: [.sortedKeys]),
                let json = String(data: data, encoding: .utf8)
            {
                return json
            }
            return String(describing: value ?? "undefined")
        }
    }
}
