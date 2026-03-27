import Foundation
import MLXLMCommon

// MARK: - WebFetchError

nonisolated enum WebFetchError: LocalizedError, Sendable {
    case invalidURL
    case urlTooLong
    case credentialsInURL
    case privateAddress
    case httpError(statusCode: Int)
    case decodingFailed
    case networkError(String)
    case timeout
    case responseTooLarge(Int)

    var errorDescription: String? {
        switch self {
        case .invalidURL: "Invalid URL. Must be an http:// or https:// URL."
        case .urlTooLong: "URL exceeds maximum length (2000 characters)"
        case .credentialsInURL: "URLs with embedded credentials are not allowed"
        case .privateAddress: "Cannot fetch private/internal network addresses"
        case .httpError(let code): "HTTP error \(code)"
        case .decodingFailed: "Failed to decode response as text"
        case .networkError(let msg): "Network error: \(msg)"
        case .timeout: "Request timed out"
        case .responseTooLarge(let bytes):
            "Response too large (\(ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file)))"
        }
    }
}

// MARK: - URL Validation

/// Validates URLs before fetching — scheme, length, credentials, SSRF prevention.
private nonisolated func validateURL(_ url: URL) throws {
    // Scheme: only http/https
    guard let scheme = url.scheme?.lowercased(),
          scheme == "http" || scheme == "https" else {
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

    // Block private/internal addresses (SSRF prevention)
    guard let host = url.host?.lowercased() else {
        throw WebFetchError.invalidURL
    }
    guard !isPrivateAddress(host) else {
        throw WebFetchError.privateAddress
    }
}

/// Check if a hostname resolves to a private/internal address.
private nonisolated func isPrivateAddress(_ host: String) -> Bool {
    // Localhost variants
    if host == "localhost" || host == "127.0.0.1" || host == "::1" || host == "[::1]" {
        return true
    }
    // 0.0.0.0
    if host == "0.0.0.0" {
        return true
    }
    // IPv4 private ranges
    if host.hasPrefix("10.") || host.hasPrefix("192.168.") || host.hasPrefix("169.254.") {
        return true
    }
    // 172.16.0.0/12
    if host.hasPrefix("172.") {
        let parts = host.split(separator: ".")
        if parts.count >= 2, let second = Int(parts[1]), (16...31).contains(second) {
            return true
        }
    }
    // IPv6 private (fc00::/7 — starts with fc or fd)
    if host.hasPrefix("fc") || host.hasPrefix("fd") || host.hasPrefix("[fc") || host.hasPrefix("[fd") {
        return true
    }
    // .local domains
    if host.hasSuffix(".local") || host.hasSuffix(".internal") {
        return true
    }
    return false
}

/// Upgrade HTTP to HTTPS (best-effort).
private nonisolated func upgradeToHTTPS(_ url: URL) -> URL {
    guard url.scheme?.lowercased() == "http" else { return url }
    var components = URLComponents(url: url, resolvingAgainstBaseURL: false)
    components?.scheme = "https"
    return components?.url ?? url
}

// MARK: - HTTP Fetching

/// Ephemeral session — no persistent cookies, cache, or credentials.
private let fetchSession: URLSession = {
    let config = URLSessionConfiguration.ephemeral
    config.httpAdditionalHeaders = [
        "User-Agent": "TesseractAgent/1.0 (macOS; Apple Silicon)",
        "Accept": "text/html, text/plain, */*",
    ]
    config.httpCookieAcceptPolicy = .never
    config.httpShouldSetCookies = false
    config.urlCredentialStorage = nil
    config.waitsForConnectivity = false
    return URLSession(configuration: config)
}()

private let maxResponseBytes = 5_000_000 // 5 MB

/// Fetch a URL and return the HTML string + final URL after redirects.
private func fetchHTML(url: URL, timeout: TimeInterval = 15) async throws -> (html: String, finalURL: URL) {
    var request = URLRequest(url: url)
    request.timeoutInterval = timeout

    let data: Data
    let response: URLResponse
    do {
        (data, response) = try await fetchSession.data(for: request)
    } catch let error as URLError where error.code == .timedOut {
        throw WebFetchError.timeout
    } catch {
        throw WebFetchError.networkError(error.localizedDescription)
    }

    guard data.count <= maxResponseBytes else {
        throw WebFetchError.responseTooLarge(data.count)
    }

    guard let httpResponse = response as? HTTPURLResponse else {
        throw WebFetchError.networkError("Non-HTTP response")
    }
    guard (200...299).contains(httpResponse.statusCode) else {
        throw WebFetchError.httpError(statusCode: httpResponse.statusCode)
    }

    // Decode with charset from response, falling back to UTF-8 then Latin-1
    let html: String
    if let encoding = httpResponse.textEncodingName.flatMap({ encodingFromIANA($0) }),
       let decoded = String(data: data, encoding: encoding)
    {
        html = decoded
    } else if let decoded = String(data: data, encoding: .utf8) {
        html = decoded
    } else if let decoded = String(data: data, encoding: .isoLatin1) {
        html = decoded
    } else {
        throw WebFetchError.decodingFailed
    }

    let finalURL = httpResponse.url ?? url
    return (html, finalURL)
}

/// Map common IANA charset names to Swift String.Encoding.
private func encodingFromIANA(_ name: String) -> String.Encoding? {
    switch name.lowercased() {
    case "utf-8", "utf8": return .utf8
    case "iso-8859-1", "latin1", "iso_8859-1": return .isoLatin1
    case "iso-8859-2", "latin2", "iso_8859-2": return .isoLatin2
    case "ascii", "us-ascii": return .ascii
    case "utf-16", "utf16": return .utf16
    case "utf-16be": return .utf16BigEndian
    case "utf-16le": return .utf16LittleEndian
    case "windows-1252", "cp1252": return .windowsCP1252
    case "windows-1251", "cp1251": return .windowsCP1251
    case "windows-1250", "cp1250": return .windowsCP1250
    case "euc-jp": return .japaneseEUC
    case "shift_jis", "shift-jis", "sjis": return .shiftJIS
    case "euc-kr": return .init(rawValue: CFStringConvertEncodingToNSStringEncoding(CFStringEncoding(CFStringEncodings.EUC_KR.rawValue)))
    case "gb2312", "gbk", "gb18030": return .init(rawValue: CFStringConvertEncodingToNSStringEncoding(CFStringEncoding(CFStringEncodings.GB_18030_2000.rawValue)))
    default: return nil
    }
}

// MARK: - Truncation

/// Truncate text at a paragraph or line boundary.
nonisolated func truncateAtBoundary(_ text: String, maxChars: Int) -> String {
    guard text.count > maxChars else { return text }

    let prefix = String(text.prefix(maxChars))

    if let range = prefix.range(of: "\n\n", options: .backwards) {
        return String(prefix[..<range.lowerBound])
    }
    if let range = prefix.range(of: "\n", options: .backwards) {
        return String(prefix[..<range.lowerBound])
    }
    return prefix
}

// MARK: - WebFetchTool Factory

nonisolated func createWebFetchTool() -> AgentToolDefinition {
    AgentToolDefinition(
        name: "web_fetch",
        label: "Fetch Web Page",
        description: "Fetch a web page and extract its text content. Use this after web_search to read full page content from a URL, or to fetch documentation, articles, and other web pages.",
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
                var content = cached.content
                var wasTruncated = false
                if content.count > maxChars {
                    content = truncateAtBoundary(content, maxChars: maxChars)
                    wasTruncated = true
                }
                var output = "Title: \(cached.title)\nURL: \(cached.url.absoluteString)\n\n"
                output += content
                if wasTruncated {
                    output += "\n\n[Content truncated at \(maxChars) characters]"
                }
                return .text(output)
            }

            do {
                let (html, finalURL) = try await fetchHTML(url: fetchURL)
                let extracted = await WebContentExtractor.extract(html: html, url: finalURL)

                // Cache the result
                await WebFetchCache.shared.set(url: cacheKey, content: extracted)

                var content = extracted.content
                var wasTruncated = false
                if content.count > maxChars {
                    content = truncateAtBoundary(content, maxChars: maxChars)
                    wasTruncated = true
                }

                var output = "Title: \(extracted.title)\nURL: \(finalURL.absoluteString)\n\n"
                output += content
                if wasTruncated {
                    output += "\n\n[Content truncated at \(maxChars) characters]"
                }

                return .text(output)
            } catch {
                Log.agent.warning("[WebFetch] Fetch failed for '\(urlString)': \(error)")
                return .error("Web fetch failed: \(error.localizedDescription)")
            }
        }
    )
}
