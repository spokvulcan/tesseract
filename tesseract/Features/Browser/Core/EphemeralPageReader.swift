import Foundation
import WebKit

// MARK: - WebAddressGuard

/// SSRF guard for the *anonymous* web paths — the single home for this policy,
/// applied by the browser `fetch` tool (which routes through
/// ``EphemeralPageReader/read(url:)``). The authenticated **Agent Browser**
/// deliberately omits it — it is the user's own browser and may reach localhost
/// — but the anonymous reader takes an untrusted URL as a convenience and must
/// not be turned into a private-network probe. `search` is exempt: its engine
/// host is a trusted constant and only the query string varies.
nonisolated enum WebAddressGuard {

    enum GuardError: LocalizedError, Sendable {
        case invalidURL
        case privateAddress
        var errorDescription: String? {
            switch self {
            case .invalidURL: "Invalid URL — must be an http:// or https:// address"
            case .privateAddress: "Refusing to fetch a private/internal network address"
            }
        }
    }

    static func validate(_ url: URL) throws {
        guard let scheme = url.scheme?.lowercased(), scheme == "http" || scheme == "https",
            url.user == nil, url.password == nil
        else { throw GuardError.invalidURL }
        guard let host = url.host?.lowercased() else { throw GuardError.invalidURL }
        guard !isPrivate(host: host) else { throw GuardError.privateAddress }
    }

    static func isPrivate(host: String) -> Bool {
        if ["localhost", "127.0.0.1", "::1", "[::1]", "0.0.0.0"].contains(host) { return true }
        if host.hasPrefix("10.") || host.hasPrefix("192.168.") || host.hasPrefix("169.254.") {
            return true
        }
        if host.hasPrefix("172.") {
            let parts = host.split(separator: ".")
            if parts.count >= 2, let second = Int(parts[1]), (16...31).contains(second) {
                return true
            }
        }
        if host.hasPrefix("fc") || host.hasPrefix("fd") || host.hasPrefix("[fc")
            || host.hasPrefix("[fd")
        {
            return true
        }
        if host.hasSuffix(".local") || host.hasSuffix(".internal") { return true }
        return false
    }
}

// MARK: - EphemeralPageReader

/// The **Ephemeral Page**: a cookieless, profile-less render backing anonymous
/// `browser.fetch` and `browser.search`. A throwaway non-persistent `WebPage`
/// executes the page's JavaScript (so JS-heavy sites read correctly); `fetch`
/// distills the live DOM to Markdown, `search` runs the engine's DOM query —
/// then the page is discarded. Nothing touches the Agent Profile, no window
/// shows, and the real desktop-Safari User-Agent (not a bot token) is presented.
@MainActor
enum EphemeralPageReader {

    static func read(url: URL) async throws -> WebContentExtractor.ExtractedContent {
        try WebAddressGuard.validate(url)
        let tab = makeEphemeralTab()
        try await tab.navigate(to: url)
        return try await tab.pageContent()
    }

    // MARK: - Search

    /// One search result lifted from a rendered results page. `Decodable` so the
    /// engine extraction script's JSON payload decodes straight into it (keys
    /// match the property names) — no intermediate transfer type.
    struct SearchResult: Sendable, Equatable, Decodable {
        let title: String
        let url: String
        let snippet: String
    }

    /// The outcome of a `browser.search`: structured results, or — when the
    /// engine's results page yielded none — the page's readable text, so the
    /// caller can still recover from a "no results" or challenge page instead of
    /// getting nothing back.
    enum SearchOutcome: Sendable {
        case results([SearchResult])
        case fallbackText(WebContentExtractor.ExtractedContent)
    }

    /// Render `engine`'s results page for `query` in a cookieless Ephemeral Page
    /// (real browser UA and JavaScript — not a fingerprintable bot client),
    /// extract up to `maxResults` structured results via the engine's DOM query,
    /// and fall back to the page's readable text on zero results.
    ///
    /// No ``WebAddressGuard``: the engine host is a trusted constant and only the
    /// query string varies, so this takes no untrusted URL and is not an SSRF
    /// vector (unlike ``read(url:)``).
    static func search(
        query: String, engine: SearchEngine, maxResults: Int
    ) async throws -> SearchOutcome {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { throw WebSearchError.emptyQuery }
        guard let url = engine.resultsURL(for: trimmed) else {
            throw WebSearchError.invalidEngineURL
        }

        let tab = makeEphemeralTab()
        try await tab.navigate(to: url)

        let raw = try await tab.evaluate(engine.extractionScript)
        let results = Self.decodeResults(raw)
        guard !results.isEmpty else {
            return .fallbackText(try await tab.pageContent())
        }
        return .results(Array(results.prefix(max(maxResults, 1))))
    }

    /// Decode the engine extraction script's JSON payload. A malformed payload
    /// (script error, unexpected shape) decodes to no results, which the caller
    /// treats as the fallback-to-text case.
    private static func decodeResults(_ json: String) -> [SearchResult] {
        guard let data = json.data(using: .utf8),
            let decoded = try? JSONDecoder().decode([SearchResult].self, from: data)
        else { return [] }
        return decoded
    }

    // MARK: - Render harness

    /// A throwaway cookieless `BrowserTab` reused purely as a render harness — no
    /// presenter, so no window; the ephemeral data store keeps it outside the
    /// **Agent Profile**. Callers navigate it themselves.
    private static func makeEphemeralTab() -> BrowserTab {
        var config = WebPage.Configuration()
        config.websiteDataStore = .nonPersistent()
        return BrowserTab(configuration: config)
    }
}
