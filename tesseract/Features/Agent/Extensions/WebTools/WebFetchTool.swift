import Foundation
import MLXLMCommon

// MARK: - WebFetchError

nonisolated enum WebFetchError: LocalizedError, Sendable {
    case invalidURL
    case urlTooLong
    case credentialsInURL

    var errorDescription: String? {
        switch self {
        case .invalidURL: "Invalid URL. Must be an http:// or https:// URL."
        case .urlTooLong: "URL exceeds maximum length (2000 characters)"
        case .credentialsInURL: "URLs with embedded credentials are not allowed"
        }
    }
}

// MARK: - URL Validation

/// Validates URLs before fetching — scheme, length, credentials, SSRF prevention.
private nonisolated func validateURL(_ url: URL) throws {
    // Scheme: only http/https
    guard let scheme = url.scheme?.lowercased(),
        scheme == "http" || scheme == "https"
    else {
        throw WebFetchError.invalidURL
    }

    // Length: max 2,000 characters
    guard url.absoluteString.count <= 2_000 else {
        throw WebFetchError.urlTooLong
    }

    // No credentials in URL
    guard url.user == nil && url.password == nil else {
        throw WebFetchError.credentialsInURL
    }

    // Host must be present. Private/internal-address (SSRF) filtering now lives
    // in one place — WebAddressGuard, applied inside EphemeralPageReader.read —
    // shared by web_fetch and the browser `fetch` tool.
    guard let host = url.host, !host.isEmpty else {
        throw WebFetchError.invalidURL
    }
}

/// Upgrade HTTP to HTTPS (best-effort).
private nonisolated func upgradeToHTTPS(_ url: URL) -> URL {
    guard url.scheme?.lowercased() == "http" else { return url }
    var components = URLComponents(url: url, resolvingAgainstBaseURL: false)
    components?.scheme = "https"
    return components?.url ?? url
}

// MARK: - Output

/// Render a fetched page for the agent: a `Title:/URL:` header followed by the
/// distilled body, clamped to `maxChars` at a paragraph/line boundary. The clamp
/// runs through ``PageReadPaginator`` — the browser core's single truncation
/// home, shared with the `read_page`/`fetch` tools — so the fetch paths and the
/// MCP reader can't drift.
private nonisolated func renderFetchOutput(
    _ content: WebContentExtractor.ExtractedContent, maxChars: Int
) -> String {
    let chunk = PageReadPaginator.paginate(content.content, cursor: 0, maxChars: maxChars)
    var output = "Title: \(content.title)\nURL: \(content.url.absoluteString)\n\n"
    output += chunk.text
    if chunk.nextCursor != nil {
        output += "\n\n[Content truncated at \(maxChars) characters]"
    }
    return output
}

// MARK: - WebFetchTool Factory

nonisolated func createWebFetchTool() -> AgentToolDefinition {
    AgentToolDefinition(
        name: "web_fetch",
        label: "Fetch Web Page",
        description:
            "Fetch a web page and extract its text content. Use this after web_search to read full page content from a URL, or to fetch documentation, articles, and other web pages.",
        parameterSchema: JSONSchema(
            type: "object",
            properties: [
                "url": PropertySchema(
                    type: "string",
                    description: "The URL to fetch (must be http or https)"
                ),
                "max_chars": PropertySchema(
                    type: "integer",
                    description: "Maximum characters to return (default: 50000)"
                ),
            ],
            required: ["url"]
        ),
        execute: { _, argsJSON, _, _ in
            guard let urlString = ToolArgExtractor.string(argsJSON, key: "url") else {
                return .error("Missing required argument: url")
            }

            guard let url = URL(string: urlString) else {
                return .error("Invalid URL. Must be an http:// or https:// URL.")
            }

            // Validate URL (scheme, length, credentials, SSRF)
            do {
                try validateURL(url)
            } catch {
                return .error(error.localizedDescription)
            }

            // Upgrade HTTP → HTTPS
            let fetchURL = upgradeToHTTPS(url)
            let maxChars = ToolArgExtractor.int(argsJSON, key: "max_chars") ?? 50_000

            // Check cache first
            let cacheKey = fetchURL.absoluteString
            if let cached = await WebFetchCache.shared.get(url: cacheKey) {
                return .text(renderFetchOutput(cached, maxChars: maxChars))
            }

            do {
                // Re-backed on the browser core (ADR-0026): render the page as an
                // Ephemeral Page (a cookieless WebPage) and distill it — one WebKit
                // owner for both authenticated browsing and anonymous reads.
                let extracted = try await EphemeralPageReader.read(url: fetchURL)

                // Cache the result
                await WebFetchCache.shared.set(url: cacheKey, content: extracted)

                return .text(renderFetchOutput(extracted, maxChars: maxChars))
            } catch {
                Log.agent.warning("[WebFetch] Fetch failed for '\(urlString)': \(error)")
                return .error("Web fetch failed: \(error.localizedDescription)")
            }
        }
    )
}
