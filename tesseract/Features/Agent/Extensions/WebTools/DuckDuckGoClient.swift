import Foundation

// MARK: - WebSearchError

nonisolated enum WebSearchError: LocalizedError, Sendable {
    case networkError(String)
    case parseError(String)
    case rateLimited
    case emptyQuery
    case timeout

    var errorDescription: String? {
        switch self {
        case .networkError(let msg): "Network error: \(msg)"
        case .parseError(let msg): "Failed to parse search results: \(msg)"
        case .rateLimited: "Rate limited by DuckDuckGo. Wait a moment before searching again."
        case .emptyQuery: "Search query cannot be empty"
        case .timeout: "Search request timed out"
        }
    }
}

// MARK: - RateLimiter

/// Enforces minimum interval between requests.
private actor RateLimiter {
    private var lastRequestTime: ContinuousClock.Instant?
    private let minInterval: Duration = .seconds(1)

    func waitIfNeeded() async throws {
        if let last = lastRequestTime {
            let elapsed = ContinuousClock.now - last
            if elapsed < minInterval {
                try await Task.sleep(for: minInterval - elapsed)
            }
        }
        lastRequestTime = .now
    }
}

// MARK: - DuckDuckGoClient

/// Zero-config web search via DuckDuckGo HTML scraping.
/// No API key required. Rate limited to ~1 req/sec.
nonisolated enum DuckDuckGoClient: Sendable {

    struct SearchResult: Sendable {
        let title: String
        let url: String
        let snippet: String
    }

    private static let rateLimiter = RateLimiter()

    // MARK: - Static Regex Patterns (compiled once)

    /// Matches <a class="result__a" href="URL">TITLE</a>
    private static let titleLinkRegex = try! NSRegularExpression(
        pattern: #"<a[^>]+class="result__a"[^>]*href="([^"]*)"[^>]*>([\s\S]*?)</a>"#,
        options: []
    )

    /// Matches class="result__snippet"...>SNIPPET</a|td>
    private static let snippetRegex = try! NSRegularExpression(
        pattern: #"class="result__snippet"[^>]*>([\s\S]*?)</(?:a|td|div)>"#,
        options: []
    )

    /// Matches any HTML tag for stripping.
    private static let htmlTagRegex = try! NSRegularExpression(
        pattern: #"<[^>]+>"#,
        options: []
    )

    /// Matches &#NNN; decimal numeric entities.
    private static let numericEntityRegex = try! NSRegularExpression(
        pattern: #"&#(\d+);"#,
        options: []
    )

    /// Matches &#xHHH; hex numeric entities.
    private static let hexEntityRegex = try! NSRegularExpression(
        pattern: #"&#x([0-9a-fA-F]+);"#,
        options: .caseInsensitive
    )

    /// Named HTML entities lookup (static to avoid recreating on every decode call).
    private static let namedEntities: [String: String] = [
        "&amp;": "&", "&lt;": "<", "&gt;": ">",
        "&quot;": "\"", "&apos;": "'", "&#39;": "'",
        "&nbsp;": " ", "&mdash;": "—", "&ndash;": "–",
        "&laquo;": "«", "&raquo;": "»", "&hellip;": "…",
        "&copy;": "©", "&reg;": "®", "&trade;": "™",
    ]

    /// Ephemeral session — no persistent cookies or cache.
    private static let session: URLSession = {
        let config = URLSessionConfiguration.ephemeral
        config.waitsForConnectivity = false
        return URLSession(configuration: config)
    }()

    // MARK: - Public API

    /// Search DuckDuckGo and return structured results.
    /// - Precondition: `query` should be pre-trimmed by the caller.
    static func search(
        query: String,
        maxResults: Int = 5,
        timeout: TimeInterval = 15
    ) async throws -> [SearchResult] {
        guard !query.isEmpty else { throw WebSearchError.emptyQuery }

        try await rateLimiter.waitIfNeeded()

        let html = try await fetchHTML(query: query, timeout: timeout)
        return parseResults(html: html, maxResults: maxResults)
    }

    // MARK: - HTTP

    private static func fetchHTML(query: String, timeout: TimeInterval) async throws -> String {
        guard let url = URL(string: "https://html.duckduckgo.com/html/") else {
            throw WebSearchError.networkError("Invalid DuckDuckGo URL")
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
        request.setValue("TesseractAgent/1.0 (macOS)", forHTTPHeaderField: "User-Agent")
        request.timeoutInterval = timeout

        let formBody = "q=\(formEncode(query))&kl="
        request.httpBody = formBody.data(using: .utf8)

        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.data(for: request)
        } catch let error as URLError where error.code == .timedOut {
            throw WebSearchError.timeout
        } catch {
            throw WebSearchError.networkError(error.localizedDescription)
        }

        if let httpResponse = response as? HTTPURLResponse {
            if httpResponse.statusCode == 429 || httpResponse.statusCode == 202 {
                throw WebSearchError.rateLimited
            }
            guard (200...299).contains(httpResponse.statusCode) else {
                throw WebSearchError.networkError("HTTP \(httpResponse.statusCode)")
            }
        }

        guard let html = String(data: data, encoding: .utf8) else {
            throw WebSearchError.parseError("Response is not valid UTF-8")
        }
        return html
    }

    /// Percent-encode a string for application/x-www-form-urlencoded.
    private static func formEncode(_ string: String) -> String {
        // URLQueryAllowed doesn't encode +, &, = which are significant in form data.
        // Use a custom character set that encodes everything except unreserved chars.
        var allowed = CharacterSet.alphanumerics
        allowed.insert(charactersIn: "-._~")
        return string.addingPercentEncoding(withAllowedCharacters: allowed) ?? string
    }

    // MARK: - HTML Parsing (internal for testability)

    static func parseResults(html: String, maxResults: Int) -> [SearchResult] {
        let nsHTML = html as NSString
        let fullRange = NSRange(location: 0, length: nsHTML.length)

        // Extract title+URL pairs
        let titleMatches = titleLinkRegex.matches(in: html, range: fullRange)
        // Extract snippets
        let snippetMatches = snippetRegex.matches(in: html, range: fullRange)

        let count = min(min(titleMatches.count, snippetMatches.count), maxResults)
        var results: [SearchResult] = []

        for i in 0..<count {
            let titleMatch = titleMatches[i]
            let snippetMatch = snippetMatches[i]

            // Group 1: href, Group 2: title HTML
            guard titleMatch.numberOfRanges >= 3,
                  let hrefRange = Range(titleMatch.range(at: 1), in: html),
                  let titleHTMLRange = Range(titleMatch.range(at: 2), in: html)
            else { continue }

            // Group 1: snippet HTML
            guard snippetMatch.numberOfRanges >= 2,
                  let snippetHTMLRange = Range(snippetMatch.range(at: 1), in: html)
            else { continue }

            let href = String(html[hrefRange])
            let titleHTML = String(html[titleHTMLRange])
            let snippetHTML = String(html[snippetHTMLRange])

            guard let realURL = extractRealURL(from: href), !realURL.isEmpty else { continue }
            let title = cleanHTMLText(titleHTML)
            let snippet = cleanHTMLText(snippetHTML)

            guard !title.isEmpty else { continue }

            results.append(SearchResult(title: title, url: realURL, snippet: snippet))
        }

        return results
    }

    /// Extract the actual URL from a DDG redirect link.
    /// DDG wraps results as `//duckduckgo.com/l/?uddg=ENCODED_URL&...`
    static func extractRealURL(from ddgHref: String) -> String? {
        let fullURL: String
        if ddgHref.hasPrefix("//") {
            fullURL = "https:" + ddgHref
        } else {
            fullURL = ddgHref
        }

        // If it's a DDG redirect, extract the uddg parameter
        if let components = URLComponents(string: fullURL),
           let host = components.host,
           host.contains("duckduckgo.com"),
           let queryItems = components.queryItems,
           let uddg = queryItems.first(where: { $0.name == "uddg" })?.value,
           !uddg.isEmpty
        {
            return uddg
        }

        // Not a redirect — return as-is
        return fullURL.isEmpty ? nil : fullURL
    }

    // MARK: - Text Cleaning (internal for testability)

    /// Strip HTML tags and decode entities from extracted text.
    static func cleanHTMLText(_ html: String) -> String {
        let nsString = html as NSString
        let range = NSRange(location: 0, length: nsString.length)

        // Strip HTML tags
        let stripped = htmlTagRegex.stringByReplacingMatches(
            in: html, range: range, withTemplate: ""
        )

        // Decode HTML entities
        let decoded = decodeHTMLEntities(stripped)

        // Collapse whitespace
        return decoded
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Decode common HTML entities and numeric character references.
    static func decodeHTMLEntities(_ string: String) -> String {
        var result = string
        for (entity, replacement) in namedEntities {
            result = result.replacingOccurrences(of: entity, with: replacement)
        }

        // Decimal numeric entities: &#NNN;
        result = replaceNumericEntities(in: result, regex: numericEntityRegex, radix: 10)

        // Hex numeric entities: &#xHHH;
        result = replaceNumericEntities(in: result, regex: hexEntityRegex, radix: 16)

        return result
    }

    /// Replace numeric character references (decimal or hex) with their Unicode characters.
    private static func replaceNumericEntities(
        in string: String, regex: NSRegularExpression, radix: Int
    ) -> String {
        let nsString = string as NSString
        let range = NSRange(location: 0, length: nsString.length)
        let matches = regex.matches(in: string, range: range)

        // Process matches in reverse to preserve ranges
        var result = string
        for match in matches.reversed() {
            guard match.numberOfRanges >= 2,
                  let numberRange = Range(match.range(at: 1), in: result),
                  let fullRange = Range(match.range(at: 0), in: result),
                  let codePoint = UInt32(result[numberRange], radix: radix),
                  let scalar = Unicode.Scalar(codePoint)
            else { continue }

            result.replaceSubrange(fullRange, with: String(scalar))
        }

        return result
    }
}
