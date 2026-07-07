import Foundation

// MARK: - WebSearchError

/// Errors from the WebKit-rendered search path (ADR-0028). The old HTTP-scrape
/// failure modes (rate-limit, network) are gone — search now renders in an
/// **Ephemeral Page**, so the only failures here are a bad query or an engine
/// whose results URL cannot be built.
nonisolated enum WebSearchError: LocalizedError, Sendable {
    case emptyQuery
    case invalidEngineURL

    var errorDescription: String? {
        switch self {
        case .emptyQuery: "Search query cannot be empty"
        case .invalidEngineURL: "Could not build a search URL for this engine"
        }
    }
}

// MARK: - SearchEngine

/// A web search engine the **Ephemeral Page** renders to run a query: where its
/// results live and how to lift structured `{title, url, snippet}` out of the
/// rendered results page. A value type so the engine is swappable (ADR-0028) and
/// injectable — tests point `base` at a local fixture results page and drive the
/// real render-and-extract path offline.
///
/// `browser.search` renders `resultsURL(for:)` in a cookieless, real-User-Agent
/// Ephemeral Page — not a fingerprintable bot HTTP client — then runs
/// ``extractionScript`` against the live DOM.
nonisolated struct SearchEngine: Sendable {

    /// The results endpoint. Only the query string varies (appended as
    /// `queryItem`), so this is a trusted, fixed host — not an SSRF surface, and
    /// therefore not subject to ``WebAddressGuard``.
    let base: URL

    /// Query parameter name the engine reads the search terms from.
    let queryItem: String

    /// A `callJavaScript` function body, run in the rendered results page's
    /// isolated content world, that returns a JSON array string of
    /// `{title, url, snippet}` objects. Engine-specific because result-page
    /// selectors differ per engine.
    let extractionScript: String

    /// The results URL for `query`, preserving any query items already on `base`.
    func resultsURL(for query: String) -> URL? {
        guard var components = URLComponents(url: base, resolvingAgainstBaseURL: false) else {
            return nil
        }
        var items = components.queryItems ?? []
        items.append(URLQueryItem(name: queryItem, value: query))
        components.queryItems = items
        return components.url
    }

    /// The same engine pointed at a different `base` — the injection seam that
    /// lets tests drive the real render-and-extract path against a local fixture
    /// results page.
    func pointedAt(_ base: URL) -> SearchEngine {
        SearchEngine(base: base, queryItem: queryItem, extractionScript: extractionScript)
    }
}

// MARK: - Default engine

extension SearchEngine {

    /// The default engine (ADR-0028). DuckDuckGo's no-JS HTML results page is
    /// rendered server-side with a stable `.result__a` / `.result__snippet`
    /// structure, so a real Ephemeral Page should reach it without the 202/403
    /// bot challenge the old HTTP client hit. That a cookieless real-WebKit page
    /// renders here un-challenged is **unverified from code** (ADR-0028) — which
    /// is why the engine is swappable and search falls back to page text.
    static let duckDuckGo = SearchEngine(
        base: URL(string: "https://html.duckduckgo.com/html/")!,
        queryItem: "q",
        extractionScript: ddgExtractionScript)

    /// DOM query over DuckDuckGo's results page. Unwraps DDG's `/l/?uddg=`
    /// redirect links to the real destination and de-dupes by URL. Runs in the
    /// isolated content world (invisible to the page's own scripts).
    private static let ddgExtractionScript = #"""
        const out = [];
        const seen = new Set();
        const nodes = document.querySelectorAll('.result, .web-result, .results_links');
        for (const el of nodes) {
          const a = el.querySelector('a.result__a');
          if (!a) continue;
          let href = a.getAttribute('href') || '';
          try {
            const u = new URL(href, location.href);
            const uddg = u.searchParams.get('uddg');
            if (/(^|\.)duckduckgo\.com$/i.test(u.hostname) && uddg) {
              href = uddg;
            } else {
              href = u.href;
            }
          } catch (e) { /* keep the raw href */ }
          const title = (a.textContent || '').replace(/\s+/g, ' ').trim();
          const snipEl = el.querySelector('.result__snippet');
          const snippet = ((snipEl && snipEl.textContent) || '').replace(/\s+/g, ' ').trim();
          if (!title || !href || seen.has(href)) continue;
          seen.add(href);
          out.push({ title: title, url: href, snippet: snippet });
        }
        return JSON.stringify(out);
        """#
}
