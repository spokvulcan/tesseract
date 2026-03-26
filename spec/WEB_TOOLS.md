# Web Search & Web Fetch — Implementation Spec

**Status**: Ready for implementation
**Date**: 2026-03-26
**Research**: `docs/web-search-and-fetch-research.md`

---

## Overview

Add two new agent tools — `web_search` and `web_fetch` — implemented as an **extension** (not built-in tools). This keeps web access opt-in and avoids bloating the default tool set for users who don't need it.

**Architecture**:
- `WebToolsExtension: AgentExtension` — registered via `PackageBootstrap` when web access is enabled
- `web_search` — DuckDuckGo HTML scraping, zero config, no API key
- `web_fetch` — URLSession + swift-readability + Demark (fast path), WebPage API fallback (JS-rendered pages)

**Key decisions**:
- Extension-based, not built-in — conditionally registered based on `SettingsManager.webAccessEnabled`
- DuckDuckGo search (Unsloth Studio pattern) — free, no API key, zero friction
- macOS 26 `WebPage` API for JS rendering — no WKWebView visibility hack needed
- Privacy-first: ephemeral sessions, no persistent cookies, honest User-Agent

**Dependencies (SPM)**:
- `SwiftSoup` — already used transitively; needed for DuckDuckGo HTML parsing
- `swift-readability` (lake-of-fire) — pure Swift Readability port, BSD-3, uses SwiftSoup
- `Demark` (steipete) — HTML→Markdown via JavaScriptCore, MIT

---

## Epic 1: Web Search (DuckDuckGo)

### Task 1.1: WebToolsExtension + web_search tool

**Goal**: Agent can search the web via DuckDuckGo. Working end-to-end from first task.

**Files to create**:
- `Features/Agent/Extensions/WebTools/WebToolsExtension.swift`
- `Features/Agent/Extensions/WebTools/WebSearchTool.swift`
- `Features/Agent/Extensions/WebTools/DuckDuckGoClient.swift`

#### WebToolsExtension

```swift
/// Extension providing web_search and web_fetch tools.
/// Registered only when SettingsManager.webAccessEnabled is true.
final class WebToolsExtension: AgentExtension, @unchecked Sendable {
    let path = "web-tools"
    let commands: [String: RegisteredCommand] = [:]
    let handlers: [ExtensionEventType: [ExtensionEventHandler]] = [:]

    let tools: [String: AgentToolDefinition]

    init() {
        let searchTool = createWebSearchTool()
        tools = [
            searchTool.name: searchTool,
        ]
    }
}
```

#### DuckDuckGoClient

Implements DuckDuckGo HTML scraping in Swift. This is the core networking + parsing layer.

**How DuckDuckGo HTML search works**:
1. GET `https://html.duckduckgo.com/html/` with form-encoded `q=<query>`
2. Response is a plain HTML page (no JS required — this is DDG's HTML-only interface)
3. Parse result blocks from the HTML using SwiftSoup

```swift
/// Zero-config web search via DuckDuckGo HTML scraping.
/// No API key required. Rate limited to ~1 req/sec.
nonisolated enum DuckDuckGoClient: Sendable {

    struct SearchResult: Sendable {
        let title: String
        let url: String
        let snippet: String
    }

    /// Search DuckDuckGo and return structured results.
    static func search(
        query: String,
        maxResults: Int = 5,
        timeout: TimeInterval = 15
    ) async throws -> [SearchResult]
}
```

**Implementation details**:
- Use `URLSession.shared` with ephemeral config for the request
- URL: `https://html.duckduckgo.com/html/` with POST body `q=<query>&kl=`
- User-Agent: `"TesseractAgent/1.0 (macOS)"` — honest, not spoofing a browser
- Parse with SwiftSoup:
  - Result containers: `.result` class elements
  - Title: `.result__title` → inner `a` tag text
  - URL: `.result__url` href attribute (DDG wraps these in redirect URLs — extract the actual URL from the `uddg` query parameter)
  - Snippet: `.result__snippet` text content
- Return up to `maxResults` results
- Throw typed errors: `networkError`, `parseError`, `rateLimited`, `noResults`

#### WebSearchTool

```swift
nonisolated func createWebSearchTool() -> AgentToolDefinition {
    AgentToolDefinition(
        name: "web_search",
        label: "Web Search",
        description: "Search the web for current information using DuckDuckGo. Returns titles, URLs, and snippets. Use this when you need up-to-date information, facts, documentation, or anything beyond your training data.",
        parameterSchema: JSONSchema(
            type: "object",
            properties: [
                "query": PropertySchema(
                    type: "string",
                    description: "The search query"
                ),
                "max_results": PropertySchema(
                    type: "integer",
                    description: "Maximum number of results to return (default: 5, range: 1-10)"
                ),
            ],
            required: ["query"]
        ),
        execute: { _, argsJSON, signal, _ in
            guard let query = ToolArgExtractor.string(argsJSON, key: "query") else {
                return .error("Missing required argument: query")
            }
            let maxResults = min(max(ToolArgExtractor.int(argsJSON, key: "max_results") ?? 5, 1), 10)

            let results = try await DuckDuckGoClient.search(
                query: query,
                maxResults: maxResults
            )

            // Format results as text (Unsloth Studio pattern)
            let formatted = results.enumerated().map { i, r in
                "[\(i + 1)] \(r.title)\n    URL: \(r.url)\n    \(r.snippet)"
            }.joined(separator: "\n\n")

            let header = "Search results for: \(query)\n\n"
            return AgentToolResult(
                content: [.text(header + formatted)],
                details: nil
            )
        }
    )
}
```

#### Registration in PackageBootstrap

**File to modify**: `Features/Agent/Packages/PackageBootstrap.swift`

Add a new registration path alongside `PersonalAssistantExtension`. The web tools extension is registered unconditionally during bootstrap — it's always available once the package is loaded. The on/off toggle is handled at the settings/UI level (Task 1.3), not at the registration level, so the user can toggle without requiring an agent restart.

Actually, cleaner approach: Register `WebToolsExtension` **directly** in `PackageBootstrap.bootstrap()` alongside PersonalAssistantExtension, guarded by a static flag or SettingsManager check. This avoids needing a package.json for built-in extensions.

```swift
// In PackageBootstrap.bootstrap():
// After existing extension registration loop...

// Register standalone extensions (not package-managed)
let webExt = WebToolsExtension()
extensionHost.register(webExt)
registeredPaths.append(webExt.path)
```

For Task 1.1 (MVP), register unconditionally. Task 1.3 gates it behind a setting.

**Verify**:
1. Build succeeds
2. `scripts/dev.sh dev` — launch app
3. In agent chat, prompt: "Search the web for Swift 6.2 new features"
4. Agent invokes `web_search` tool, returns real DuckDuckGo results with titles, URLs, snippets
5. Results are formatted and readable in the conversation

**Acceptance criteria**:
- `web_search` appears in agent tool specs
- DuckDuckGo returns real results for arbitrary queries
- Results contain title, URL, and snippet for each result
- Build + run succeeds

---

### Task 1.2: Error handling, rate limiting, User-Agent

**Goal**: web_search handles failures gracefully — network errors, rate limits, timeouts, empty results all produce useful error messages to the agent instead of crashes.

**File to modify**: `Features/Agent/Extensions/WebTools/DuckDuckGoClient.swift`

#### Error types

```swift
nonisolated enum WebSearchError: LocalizedError {
    case networkError(underlying: Error)
    case parseError(String)
    case rateLimited
    case emptyQuery
    case timeout

    var errorDescription: String? {
        switch self {
        case .networkError(let e): "Network error: \(e.localizedDescription)"
        case .parseError(let msg): "Failed to parse search results: \(msg)"
        case .rateLimited: "Rate limited by DuckDuckGo. Wait a moment before searching again."
        case .emptyQuery: "Search query cannot be empty"
        case .timeout: "Search request timed out"
        }
    }
}
```

#### Rate limiting

Simple token bucket: 1 request per second, enforced via actor-isolated timestamp.

```swift
/// Tracks last request time to enforce rate limiting.
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
```

#### Timeout

Use `URLRequest.timeoutInterval` set to 15 seconds (configurable). Also wrap the entire search call in a `Task` with `withThrowingTaskGroup` timeout pattern if CancellationToken is signaled.

#### Error handling in execute closure

Catch all errors and return them as `.error()` results so the agent sees them as tool output, not crashes:

```swift
execute: { _, argsJSON, signal, _ in
    // ... parameter extraction ...
    do {
        let results = try await DuckDuckGoClient.search(query: query, maxResults: maxResults)
        if results.isEmpty {
            return AgentToolResult(
                content: [.text("No results found for: \(query)")],
                details: nil
            )
        }
        // ... format results ...
    } catch {
        return .error("Web search failed: \(error.localizedDescription)")
    }
}
```

**Verify**:
1. Build succeeds
2. Search with valid query returns results (unchanged from Task 1.1)
3. Search with empty/whitespace query returns clear error
4. Disable network (airplane mode or firewall) → search returns "Network error" message, agent can reason about it
5. Rapid sequential searches don't trigger DDG rate limiting (rate limiter spaces them 1s apart)

**Acceptance criteria**:
- No crashes on any failure path
- Agent receives useful error text for all failure modes
- Rate limiter prevents exceeding 1 req/s
- 15-second timeout prevents hanging

---

### Task 1.3: Settings UI integration (web access toggle)

**Goal**: Web access is off by default. User enables it in Settings. The web tools extension is only registered when enabled.

**Files to modify**:
- `Features/Settings/SettingsManager.swift` — add `webAccessEnabled` property
- `Features/Agent/Packages/PackageBootstrap.swift` — gate registration
- Settings UI view (wherever the agent settings panel lives) — add toggle

#### SettingsManager addition

```swift
// In Key enum:
static let webAccessEnabled = "webAccessEnabled"

// Property:
var webAccessEnabled = false {
    didSet { UserDefaults.standard.set(webAccessEnabled, forKey: Key.webAccessEnabled) }
}

// In init register(defaults:):
Key.webAccessEnabled: false,
```

#### Conditional registration

```swift
// In PackageBootstrap.bootstrap():
if settingsManager.webAccessEnabled {
    let webExt = WebToolsExtension()
    extensionHost.register(webExt)
    registeredPaths.append(webExt.path)
}
```

This requires threading `SettingsManager` into `PackageBootstrap.bootstrap()`. Add it as a parameter:

```swift
static func bootstrap(
    packageRegistry: PackageRegistry,
    extensionHost: ExtensionHost,
    agentRoot: URL,
    settings: SettingsManager  // NEW
)
```

Update call sites in `AgentFactory.makeAgent()` and `BackgroundAgentFactory`.

#### Settings UI

Add a toggle in the Agent section of Settings:

```swift
Toggle("Web Access", isOn: $settings.webAccessEnabled)
    .help("Allow the agent to search the web and fetch pages. Queries are sent to DuckDuckGo.")
```

Include a brief privacy note below the toggle: "Search queries are sent to DuckDuckGo. No conversation content or personal data leaves your device."

**Verify**:
1. Fresh install: web access is OFF by default
2. `web_search` tool does NOT appear in agent tool specs when disabled
3. Toggle ON → restart agent conversation → `web_search` appears
4. Toggle OFF → restart agent conversation → `web_search` disappears
5. Privacy note is visible in Settings

**Acceptance criteria**:
- Default is OFF
- Toggle controls extension registration
- Setting persists across app restarts
- Call sites updated (AgentFactory, BackgroundAgentFactory)

---

## Epic 2: Web Fetch

### Task 2.1: Core web_fetch tool (URLSession + SwiftSoup text extraction)

**Goal**: Agent can fetch any URL and get readable text content. Working end-to-end from first task.

**Files to create**:
- `Features/Agent/Extensions/WebTools/WebFetchTool.swift`
- `Features/Agent/Extensions/WebTools/WebContentExtractor.swift`

**File to modify**:
- `Features/Agent/Extensions/WebTools/WebToolsExtension.swift` — add `web_fetch` to tools dict

#### WebContentExtractor (basic — SwiftSoup only)

First pass uses SwiftSoup directly for HTML cleaning and text extraction. This is intentionally simple — Task 2.2 upgrades to swift-readability + Demark.

```swift
/// Extracts readable text content from HTML.
nonisolated enum WebContentExtractor: Sendable {

    struct ExtractedContent: Sendable {
        let title: String
        let text: String
        let byline: String?
    }

    /// Extract main text content from HTML, stripping boilerplate.
    static func extract(html: String, url: URL) throws -> ExtractedContent
}
```

**Implementation**:
1. Parse HTML with SwiftSoup
2. Extract `<title>` text
3. Remove noise elements: `script`, `style`, `nav`, `footer`, `header`, `aside`, `noscript`, `svg`, `iframe`, elements with classes matching `ad|banner|cookie|popup|modal|sidebar|comment|share|social|login|signup`
4. Extract `body` text content (`.text()` on remaining body)
5. Collapse whitespace (multiple newlines → double newline, multiple spaces → single space)
6. Return `ExtractedContent` with title and cleaned text

#### HTTP Fetching

In `WebFetchTool.swift`:

```swift
/// Fetch a URL using ephemeral URLSession.
private static func fetchHTML(url: URL, timeout: TimeInterval = 15) async throws -> (html: String, finalURL: URL) {
    let config = URLSessionConfiguration.ephemeral
    config.httpAdditionalHeaders = [
        "User-Agent": "TesseractAgent/1.0 (macOS; Apple Silicon)",
        "Accept": "text/html, text/plain, */*",
    ]
    config.timeoutIntervalForRequest = timeout
    config.timeoutIntervalForResource = timeout * 2

    let session = URLSession(configuration: config)
    defer { session.invalidateAndCancel() }

    let (data, response) = try await session.data(from: url)

    guard let httpResponse = response as? HTTPURLResponse else {
        throw WebFetchError.invalidResponse
    }
    guard (200...299).contains(httpResponse.statusCode) else {
        throw WebFetchError.httpError(statusCode: httpResponse.statusCode)
    }

    let encoding = httpResponse.textEncodingName.flatMap { String.Encoding(ianaCharsetName: $0) } ?? .utf8
    guard let html = String(data: data, encoding: encoding) ?? String(data: data, encoding: .utf8) else {
        throw WebFetchError.decodingFailed
    }

    let finalURL = httpResponse.url ?? url
    return (html, finalURL)
}
```

#### WebFetchTool

```swift
nonisolated func createWebFetchTool() -> AgentToolDefinition {
    AgentToolDefinition(
        name: "web_fetch",
        label: "Fetch Web Page",
        description: "Fetch a web page and extract its text content. Use this after web_search to read full page content, or to fetch any URL (documentation, articles, etc.).",
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
        execute: { _, argsJSON, signal, _ in
            guard let urlString = ToolArgExtractor.string(argsJSON, key: "url"),
                  let url = URL(string: urlString),
                  let scheme = url.scheme?.lowercased(),
                  scheme == "http" || scheme == "https"
            else {
                return .error("Invalid or missing URL. Must be an http:// or https:// URL.")
            }

            let maxChars = ToolArgExtractor.int(argsJSON, key: "max_chars") ?? 50_000

            let (html, finalURL) = try await fetchHTML(url: url)
            let extracted = try WebContentExtractor.extract(html: html, url: finalURL)

            var content = extracted.text
            var wasTruncated = false
            if content.count > maxChars {
                // Truncate at paragraph boundary
                content = truncateAtBoundary(content, maxChars: maxChars)
                wasTruncated = true
            }

            var output = "Title: \(extracted.title)\nURL: \(finalURL.absoluteString)\n\n"
            output += content
            if wasTruncated {
                output += "\n\n[Content truncated at \(maxChars) characters]"
            }

            return AgentToolResult(content: [.text(output)], details: nil)
        }
    )
}
```

**Truncation helper**: Cut at the last double-newline (paragraph break) before `maxChars`. If no paragraph break found, cut at last single newline. Last resort: hard cut at `maxChars`.

#### Register in WebToolsExtension

```swift
init() {
    let searchTool = createWebSearchTool()
    let fetchTool = createWebFetchTool()
    tools = [
        searchTool.name: searchTool,
        fetchTool.name: fetchTool,
    ]
}
```

**Verify**:
1. Build succeeds
2. Agent can fetch a known static page (e.g., a Wikipedia article URL)
3. Returned content is readable text without HTML tags
4. Title and final URL are included in output
5. Large pages are truncated with a notice
6. Invalid URLs return clear error
7. Non-200 responses return HTTP error with status code

**Acceptance criteria**:
- `web_fetch` appears in agent tool specs (when web access enabled)
- Fetches real pages and returns readable text
- HTML noise (scripts, styles, nav, ads) is stripped
- Truncation at ~50K chars with boundary-aware cutting
- Error handling for network failures, bad URLs, non-200 responses

---

### Task 2.2: High-quality extraction (swift-readability + Demark)

**Goal**: Upgrade from basic SwiftSoup text extraction to Readability-based content extraction + Markdown conversion. Output quality jumps dramatically — headings, lists, code blocks, tables, links all preserved.

**SPM dependencies to add**:
- `swift-readability` — `https://github.com/lake-of-fire/swift-readability` (BSD-3)
- `Demark` — `https://github.com/steipete/Demark` (MIT)

**File to modify**: `Features/Agent/Extensions/WebTools/WebContentExtractor.swift`

#### Updated extraction pipeline

```
Raw HTML
    ↓
swift-readability (content extraction, pure Swift)
    ↓
Clean HTML (article body only, no nav/ads/sidebar)
    ↓
Demark html-to-md engine (JSC, ~5ms, off main thread)
    ↓
Clean Markdown (headings, lists, code, tables preserved)
    ↓
Token-aware truncation (~50KB / ~12K tokens)
    ↓
Return to agent
```

#### Implementation

```swift
nonisolated enum WebContentExtractor: Sendable {

    struct ExtractedContent: Sendable {
        let title: String
        let content: String       // Markdown-formatted
        let byline: String?
        let excerpt: String?
        let isReaderable: Bool    // Whether Readability found article content
    }

    static func extract(html: String, url: URL) throws -> ExtractedContent {
        // 1. Try swift-readability first
        let isReaderable = Readability.isProbablyReaderable(html: html)

        if isReaderable {
            let reader = Readability(html: html, url: url)
            if let result = try reader.parse() {
                // result.content is clean HTML — convert to markdown
                let markdown = try DemarkConverter.htmlToMarkdown(result.content)
                return ExtractedContent(
                    title: result.title ?? "",
                    content: markdown,
                    byline: result.byline,
                    excerpt: result.excerpt,
                    isReaderable: true
                )
            }
        }

        // 2. Fallback: basic SwiftSoup extraction (from Task 2.1)
        return try extractBasic(html: html, url: url)
    }
}
```

#### Demark integration

Use Demark's `html-to-md` engine (JavaScriptCore, not WKWebView):

```swift
/// Wrapper for Demark HTML→Markdown conversion.
private nonisolated enum DemarkConverter: Sendable {
    /// Convert clean HTML to markdown using JSC engine (~5ms).
    static func htmlToMarkdown(_ html: String) throws -> String {
        // Use Demark's JSC-based html-to-md engine
        // Runs off main thread, ~5ms per conversion
        try Demark.convert(html, engine: .htmlToMd)
    }
}
```

If Demark's API differs from this pseudocode, adapt accordingly — the key is using the JSC engine, not the WKWebView engine.

#### Output format

The agent now receives markdown with structure preserved:

```
Title: How to Use Swift Concurrency
URL: https://example.com/swift-concurrency

# How to Use Swift Concurrency

Swift concurrency provides structured, safe concurrent code...

## async/await

The `async` keyword marks a function that can be suspended...

### Example

```swift
func fetchData() async throws -> Data {
    ...
}
```

[Content truncated at 50000 characters]
```

**Verify**:
1. Build succeeds (SPM dependencies resolve)
2. Fetch a news article → get clean markdown with headings, paragraphs
3. Fetch a documentation page → code blocks and lists preserved
4. Fetch a page Readability can't parse → fallback to basic extraction still works
5. Markdown output is well-formatted (no raw HTML tags leaking through)

**Acceptance criteria**:
- swift-readability and Demark dependencies added to Package.swift / Xcode project
- Readability extraction produces clean article content
- Demark converts to proper markdown with headings, lists, code, tables, links
- Fallback to basic extraction when Readability fails
- Output is markdown, not plain text

---

### Task 2.3: WebPage API fallback for JS-rendered pages

**Goal**: Pages that require JavaScript rendering (React, Next.js, Vue SPAs) now work. When the fast path extracts suspiciously little content, the WebPage API renders the page headlessly and re-extracts.

**File to create**:
- `Features/Agent/Extensions/WebTools/HeadlessRenderer.swift`

**File to modify**:
- `Features/Agent/Extensions/WebTools/WebContentExtractor.swift` — add SPA detection + fallback

#### SPA detection heuristic

After the fast-path extraction (URLSession + Readability), check if the result is suspicious:

```swift
/// Detect if the page is likely a JavaScript-rendered SPA that needs browser rendering.
private static func isSuspectedSPA(extractedText: String, rawHTML: String) -> Bool {
    let textLength = extractedText.trimmingCharacters(in: .whitespacesAndNewlines).count
    let htmlLength = rawHTML.utf8.count

    // Strong signal: very little text from large HTML
    if textLength < 200 && htmlLength > 10_000 {
        return true
    }

    // SPA markers in HTML
    let spaMarkers = [
        "<div id=\"root\"></div>",
        "<div id=\"app\"></div>",
        "<div id=\"__next\"></div>",
        "<div id=\"__nuxt\"></div>",
    ]
    let htmlLower = rawHTML.lowercased()
    if textLength < 500 && spaMarkers.contains(where: { htmlLower.contains($0) }) {
        return true
    }

    return false
}
```

#### HeadlessRenderer (macOS 26 WebPage API)

```swift
import WebKit

/// Headless page renderer using macOS 26 WebPage API.
/// No view hierarchy needed — WebPage works standalone.
@MainActor
final class HeadlessRenderer {
    /// Render a URL with JavaScript execution and return the fully rendered HTML.
    /// Uses a single reusable WebPage instance with nonPersistent data store.
    func render(url: URL, timeout: Duration = .seconds(20)) async throws -> String
}
```

**Implementation**:
1. Create `WebPage` with `nonPersistent()` data store and custom User-Agent
2. `page.load(URLRequest(url: url))`
3. Observe `currentNavigationEvent` via `Observations`:
   - Wait for `.finished` event
   - After `.finished`, inject MutationObserver via `callJavaScript` to detect when SPA rendering stabilizes (1.5s idle = done)
   - Alternatively: simple 1.5s sleep after `.finished` for MVP, upgrade to MutationObserver later
4. Extract full rendered DOM: `page.callJavaScript("document.documentElement.outerHTML")`
5. Return rendered HTML string
6. Hard timeout via `withThrowingTaskGroup` — cancel if exceeds `timeout`

**Memory note**: Each WebPage spawns a ~50-150 MB WebContent process. Create a **single reusable instance** — don't create one per fetch. Use a serial queue pattern (one render at a time).

**Privacy**: `config.websiteDataStore = .nonPersistent()` — no cookies, caches, localStorage persisted.

#### Updated extraction pipeline

```swift
static func extract(html: String, url: URL) async throws -> ExtractedContent {
    // 1. Fast path: swift-readability
    let fastResult = try extractWithReadability(html: html, url: url)

    // 2. Check if SPA fallback needed
    if isSuspectedSPA(extractedText: fastResult.content, rawHTML: html) {
        // 3. Render with WebPage API
        let renderer = HeadlessRenderer.shared
        let renderedHTML = try await renderer.render(url: url)

        // 4. Re-extract from rendered HTML
        let spaResult = try extractWithReadability(html: renderedHTML, url: url)
        if spaResult.content.count > fastResult.content.count {
            return spaResult
        }
    }

    return fastResult
}
```

Note: `extract` becomes `async` — update the call site in `WebFetchTool`.

#### NavigationDeciding (optional, recommended)

Restrict the WebPage to only load from the target domain — block cross-domain navigations for security:

```swift
class SingleDomainNavigationDecider: WebPage.NavigationDeciding {
    let allowedHost: String

    func decidePolicy(
        for action: WebPage.NavigationAction,
        preferences: inout WebPage.NavigationPreferences
    ) async -> WKNavigationActionPolicy {
        guard let host = action.request.url?.host, host == allowedHost else {
            return .cancel
        }
        return .allow
    }
}
```

**Verify**:
1. Build succeeds
2. Fetch a standard static page → fast path works, no WebPage spawned
3. Fetch a known React SPA (e.g., a React docs page or any SPA URL) → SPA detected, WebPage renders, full content extracted
4. Timeout handling: unreachable/slow SPA doesn't hang forever
5. Memory: WebPage process cleans up after render

**Acceptance criteria**:
- SPA detection triggers on JS-rendered pages
- WebPage API renders the page headlessly (no window, no view hack)
- Re-extraction from rendered HTML produces full article content
- Falls back gracefully if WebPage render fails
- Single reusable WebPage instance (not one per fetch)
- Privacy: nonPersistent data store, no cookies/cache persisted

---

### Task 2.4: URL validation, privacy hardening, caching

**Goal**: Production-quality URL handling, privacy guarantees, and a simple cache to avoid redundant fetches.

**Files to modify**:
- `Features/Agent/Extensions/WebTools/WebFetchTool.swift`

**File to create**:
- `Features/Agent/Extensions/WebTools/WebFetchCache.swift`

#### URL validation

Before fetching, validate the URL:

```swift
nonisolated enum URLValidator: Sendable {
    enum ValidationError: LocalizedError {
        case tooLong
        case unsupportedScheme
        case privateAddress
        case credentialsInURL

        var errorDescription: String? { ... }
    }

    static func validate(_ url: URL) throws {
        // 1. Length: max 2,000 characters
        guard url.absoluteString.count <= 2_000 else { throw .tooLong }

        // 2. Scheme: only http/https
        guard let scheme = url.scheme?.lowercased(),
              scheme == "http" || scheme == "https" else {
            throw .unsupportedScheme
        }

        // 3. No credentials in URL
        guard url.user == nil && url.password == nil else {
            throw .credentialsInURL
        }

        // 4. Block private/internal addresses (SSRF prevention)
        // Block: 127.0.0.0/8, 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16,
        //        169.254.0.0/16, ::1, fc00::/7, localhost
        guard let host = url.host?.lowercased() else { throw .unsupportedScheme }
        guard !isPrivateAddress(host) else { throw .privateAddress }
    }
}
```

#### Privacy hardening

Ensure the fetch configuration strips identifying information:

```swift
let config = URLSessionConfiguration.ephemeral
config.httpAdditionalHeaders = [
    "User-Agent": "TesseractAgent/1.0 (macOS; Apple Silicon)",
    "Accept": "text/html, text/plain, */*",
    // Explicitly omit Referer
]
config.httpCookieAcceptPolicy = .never
config.httpShouldSetCookies = false
config.urlCredentialStorage = nil
```

Key privacy guarantees:
- Ephemeral session: no persistent cookies, cache, or credentials
- No Referer header
- No cookies accepted or sent
- Honest User-Agent (not spoofing a browser)
- Credentials stripped from URLs
- Private addresses blocked (SSRF prevention)

#### HTTP→HTTPS upgrade

```swift
// Upgrade http to https (best-effort)
var fetchURL = validatedURL
if fetchURL.scheme == "http" {
    var components = URLComponents(url: fetchURL, resolvingAgainstBaseURL: false)!
    components.scheme = "https"
    if let upgraded = components.url {
        fetchURL = upgraded
    }
}
```

#### Response size limit

Cap the response body at 5 MB to prevent memory issues:

```swift
guard data.count <= 5_000_000 else {
    throw WebFetchError.responseTooLarge(data.count)
}
```

#### Cache

Simple in-memory cache with 15-minute TTL:

```swift
/// Simple TTL cache for web fetch results.
/// Prevents redundant fetches when the agent re-reads a page.
actor WebFetchCache {
    private struct Entry {
        let content: WebContentExtractor.ExtractedContent
        let fetchedAt: ContinuousClock.Instant
    }

    private var entries: [String: Entry] = [:]
    private let ttl: Duration = .seconds(15 * 60)  // 15 minutes
    private let maxEntries = 50

    func get(url: String) -> WebContentExtractor.ExtractedContent? {
        guard let entry = entries[url] else { return nil }
        if ContinuousClock.now - entry.fetchedAt > ttl {
            entries.removeValue(forKey: url)
            return nil
        }
        return entry.content
    }

    func set(url: String, content: WebContentExtractor.ExtractedContent) {
        // Evict oldest if at capacity
        if entries.count >= maxEntries {
            if let oldest = entries.min(by: { $0.value.fetchedAt < $1.value.fetchedAt }) {
                entries.removeValue(forKey: oldest.key)
            }
        }
        entries[url] = Entry(content: content, fetchedAt: .now)
    }
}
```

**Verify**:
1. Build succeeds
2. Fetch with valid HTTPS URL → works as before
3. Fetch with HTTP URL → auto-upgraded to HTTPS
4. Fetch with private IP (127.0.0.1, 192.168.x.x) → blocked with clear error
5. Fetch with credentials in URL → stripped/blocked
6. Fetch same URL twice within 15 minutes → second fetch returns cached result (fast)
7. Fetch same URL after 15 minutes → re-fetches
8. Response > 5MB → error returned

**Acceptance criteria**:
- URL validation rejects invalid/dangerous URLs
- SSRF prevention blocks private addresses
- HTTP→HTTPS upgrade
- Ephemeral sessions with no tracking headers
- 15-minute TTL cache reduces redundant fetches
- 5MB response size limit

---

## Integration Notes

### Tool interaction pattern

The two tools are designed to work together in the agent loop:

1. User asks: "What are the new features in Swift 6.2?"
2. Agent calls `web_search(query: "Swift 6.2 new features 2026")`
3. Agent sees results with titles and URLs
4. Agent calls `web_fetch(url: "https://...")` on the most relevant result
5. Agent reads the full article content
6. Agent synthesizes an answer citing the source

This multi-step pattern emerges naturally from the agent loop — no special orchestration needed.

### Token budget

With 120K context window, 16K reserve, 20K recent tokens:
- `web_search` returns ~500-1,000 tokens (5 results with snippets)
- `web_fetch` returns ~3,000-12,000 tokens (50KB text ≈ 12K tokens)
- Room for 4-8 pages in context simultaneously
- Compaction handles cleanup when context fills up

### Dependencies summary

| Dependency | License | Purpose | Size |
|-----------|---------|---------|------|
| SwiftSoup | MIT | HTML parsing (already used) | — |
| swift-readability | BSD-3 | Content extraction (Readability port) | Pure Swift |
| Demark | MIT | HTML→Markdown (JSC engine) | ~30KB JS |

### Files created/modified

**New files** (6):
- `Features/Agent/Extensions/WebTools/WebToolsExtension.swift`
- `Features/Agent/Extensions/WebTools/WebSearchTool.swift`
- `Features/Agent/Extensions/WebTools/DuckDuckGoClient.swift`
- `Features/Agent/Extensions/WebTools/WebFetchTool.swift`
- `Features/Agent/Extensions/WebTools/WebContentExtractor.swift`
- `Features/Agent/Extensions/WebTools/HeadlessRenderer.swift`
- `Features/Agent/Extensions/WebTools/WebFetchCache.swift`

**Modified files** (4):
- `Features/Settings/SettingsManager.swift` — add `webAccessEnabled`
- `Features/Agent/Packages/PackageBootstrap.swift` — register WebToolsExtension
- `Features/Agent/AgentFactory.swift` — pass settings to bootstrap
- `Features/Agent/BackgroundAgentFactory.swift` — pass settings to bootstrap
- Settings UI view — add web access toggle

### Task dependency graph

```
Epic 1 (Web Search)           Epic 2 (Web Fetch)

Task 1.1 ──→ Task 1.2         Task 2.1 ──→ Task 2.2 ──→ Task 2.3
    │            │                │                         │
    └──→ Task 1.3 ←──────────────┘                         │
                                  Task 2.4 ←───────────────┘
```

- Task 1.1 is the starting point — creates the extension skeleton
- Tasks 1.2 and 2.1 can start in parallel after 1.1
- Task 1.3 (settings) depends on 1.1 and should account for 2.1's tool
- Tasks 2.2, 2.3, 2.4 are sequential (each builds on prior)
- Task 2.4 (validation/caching) can also be done after 2.2 if 2.3 is deferred
