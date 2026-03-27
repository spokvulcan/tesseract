import Foundation
import Testing
import MLXLMCommon
@testable import tesseract

// MARK: - WebContentExtractor Tests

@MainActor
struct WebContentExtractorTests {

    private static let sampleURL = URL(string: "https://example.com/article")!

    /// Realistic HTML page with boilerplate.
    private static let fullPageHTML = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Swift 6.2 Released &mdash; The Swift Blog</title>
        <style>body { font-family: sans-serif; }</style>
        <script>var analytics = true;</script>
    </head>
    <body>
        <nav class="site-nav"><a href="/">Home</a> | <a href="/blog">Blog</a></nav>
        <header><h1>Swift Blog</h1></header>
        <main>
            <article>
                <h2>Swift 6.2 Released</h2>
                <p>We&#x27;re excited to announce <b>Swift 6.2</b>, a major release.</p>
                <p>This release includes improved concurrency &amp; performance.</p>
            </article>
        </main>
        <aside class="sidebar"><h3>Popular Posts</h3><ul><li>Post 1</li></ul></aside>
        <footer><p>&copy; 2026 Apple Inc.</p></footer>
        <script>trackPageView();</script>
    </body>
    </html>
    """

    @Test func extractsTitleFromHTML() {
        let result = WebContentExtractor.extractBasic(html: Self.fullPageHTML, url: Self.sampleURL)
        #expect(result.title == "Swift 6.2 Released — The Swift Blog")
    }

    @Test func stripsScriptAndStyleContent() {
        let result = WebContentExtractor.extractBasic(html: Self.fullPageHTML, url: Self.sampleURL)
        #expect(!result.content.contains("analytics"))
        #expect(!result.content.contains("trackPageView"))
        #expect(!result.content.contains("font-family"))
    }

    @Test func stripsNavAndFooter() {
        let result = WebContentExtractor.extractBasic(html: Self.fullPageHTML, url: Self.sampleURL)
        // Nav links should be removed
        #expect(!result.content.contains("Home | Blog"))
        // Footer should be removed
        #expect(!result.content.contains("2026 Apple Inc"))
    }

    @Test func preservesArticleContent() {
        let result = WebContentExtractor.extractBasic(html: Self.fullPageHTML, url: Self.sampleURL)
        #expect(result.content.contains("Swift 6.2 Released"))
        #expect(result.content.contains("excited to announce"))
        #expect(result.content.contains("improved concurrency & performance"))
    }

    @Test func decodesHTMLEntities() {
        let result = WebContentExtractor.extractBasic(html: Self.fullPageHTML, url: Self.sampleURL)
        // &#x27; should become '
        #expect(result.content.contains("We're excited"))
        // &amp; should become &
        #expect(result.content.contains("concurrency & performance"))
    }

    @Test func stripsAsideSidebar() {
        let result = WebContentExtractor.extractBasic(html: Self.fullPageHTML, url: Self.sampleURL)
        #expect(!result.content.contains("Popular Posts"))
    }

    @Test func preservesURLInResult() {
        let result = WebContentExtractor.extractBasic(html: Self.fullPageHTML, url: Self.sampleURL)
        #expect(result.url == Self.sampleURL)
    }

    @Test func handlesEmptyHTML() {
        let result = WebContentExtractor.extractBasic(html: "", url: Self.sampleURL)
        #expect(result.title == "")
        #expect(result.content == "")
    }

    @Test func handlesHTMLWithNoTitle() {
        let html = "<html><body><p>Just some text.</p></body></html>"
        let result = WebContentExtractor.extractBasic(html: html, url: Self.sampleURL)
        #expect(result.title == "")
        #expect(result.content.contains("Just some text"))
    }

    @Test func handlesHTMLWithOnlyBoilerplate() {
        let html = """
        <html><body>
        <script>lots of js code here</script>
        <style>.class { color: red; }</style>
        <nav>Navigation links</nav>
        </body></html>
        """
        let result = WebContentExtractor.extractBasic(html: html, url: Self.sampleURL)
        #expect(!result.content.contains("lots of js"))
        #expect(!result.content.contains("color: red"))
    }

    @Test func collapsesExcessiveWhitespace() {
        let html = "<html><body><p>Line 1</p>\n\n\n\n\n<p>Line 2</p></body></html>"
        let result = WebContentExtractor.extractBasic(html: html, url: Self.sampleURL)
        // Should not have more than double newline
        #expect(!result.content.contains("\n\n\n"))
    }

    @Test func stripsHTMLComments() {
        let html = "<html><body><!-- This is a comment --><p>Visible text</p></body></html>"
        let result = WebContentExtractor.extractBasic(html: html, url: Self.sampleURL)
        #expect(!result.content.contains("This is a comment"))
        #expect(result.content.contains("Visible text"))
    }
}

// MARK: - Truncation Tests

@MainActor
struct TruncationTests {

    @Test func truncatesAtParagraphBoundary() {
        let text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph that is longer."
        let result = truncateAtBoundary(text, maxChars: 40)
        #expect(result == "First paragraph.\n\nSecond paragraph.")
    }

    @Test func truncatesAtLineBoundary() {
        let text = "Line one.\nLine two.\nLine three is long."
        let result = truncateAtBoundary(text, maxChars: 25)
        #expect(result == "Line one.\nLine two.")
    }

    @Test func hardCutsWhenNoBoundary() {
        let text = "One very long line without any breaks at all in this text."
        let result = truncateAtBoundary(text, maxChars: 20)
        #expect(result == "One very long line w")
    }

    @Test func returnsFullTextWhenUnderLimit() {
        let text = "Short text."
        let result = truncateAtBoundary(text, maxChars: 1000)
        #expect(result == text)
    }

    @Test func handlesEmptyText() {
        let result = truncateAtBoundary("", maxChars: 100)
        #expect(result == "")
    }
}

// MARK: - WebFetchTool Definition Tests

@MainActor
struct WebFetchToolTests {

    @Test func toolHasCorrectNameAndSchema() {
        let tool = createWebFetchTool()
        #expect(tool.name == "web_fetch")
        #expect(tool.label == "Fetch Web Page")
        #expect(tool.parameterSchema.required == ["url"])
        #expect(tool.parameterSchema.properties["url"] != nil)
        #expect(tool.parameterSchema.properties["max_chars"] != nil)
    }

    @Test func returnsErrorForMissingURL() async throws {
        let tool = createWebFetchTool()
        let result = try await tool.execute("t1", [:], nil, nil)
        let text = result.content.textContent
        #expect(text.contains("Missing required argument: url"))
    }

    @Test func returnsErrorForInvalidURL() async throws {
        let tool = createWebFetchTool()
        let result = try await tool.execute("t2", ["url": .string("not-a-url")], nil, nil)
        let text = result.content.textContent
        #expect(text.contains("Invalid URL"))
    }

    @Test func returnsErrorForFTPURL() async throws {
        let tool = createWebFetchTool()
        let result = try await tool.execute("t3", ["url": .string("ftp://files.example.com/doc")], nil, nil)
        let text = result.content.textContent
        #expect(text.contains("Invalid URL"))
    }

    @Test func returnsErrorForFileURL() async throws {
        let tool = createWebFetchTool()
        let result = try await tool.execute("t4", ["url": .string("file:///etc/passwd")], nil, nil)
        let text = result.content.textContent
        #expect(text.contains("Invalid URL"))
    }

    @Test func returnsErrorForLocalhostURL() async throws {
        let tool = createWebFetchTool()
        let result = try await tool.execute("t5", ["url": .string("http://localhost:8080/admin")], nil, nil)
        let text = result.content.textContent
        #expect(text.contains("private"))
    }

    @Test func returnsErrorForPrivateIPURL() async throws {
        let tool = createWebFetchTool()
        let result = try await tool.execute("t6", ["url": .string("http://192.168.1.1/config")], nil, nil)
        let text = result.content.textContent
        #expect(text.contains("private"))
    }

    @Test func returnsErrorFor10xPrivateIP() async throws {
        let tool = createWebFetchTool()
        let result = try await tool.execute("t7", ["url": .string("http://10.0.0.1/internal")], nil, nil)
        let text = result.content.textContent
        #expect(text.contains("private"))
    }

    @Test func returnsErrorForCredentialsInURL() async throws {
        let tool = createWebFetchTool()
        let result = try await tool.execute("t8", ["url": .string("https://user:pass@example.com/")], nil, nil)
        let text = result.content.textContent
        #expect(text.contains("credentials"))
    }

    @Test func returnsErrorForVeryLongURL() async throws {
        let tool = createWebFetchTool()
        let longURL = "https://example.com/" + String(repeating: "a", count: 2000)
        let result = try await tool.execute("t9", ["url": .string(longURL)], nil, nil)
        let text = result.content.textContent
        #expect(text.contains("length"))
    }
}

// MARK: - SSRF Prevention Tests

@MainActor
struct SSRFPreventionTests {

    @Test func blocksLoopbackIPv4() async throws {
        let tool = createWebFetchTool()
        let result = try await tool.execute("s1", ["url": .string("http://127.0.0.1/")], nil, nil)
        #expect(result.content.textContent.contains("private"))
    }

    @Test func blocksLoopbackIPv6() async throws {
        let tool = createWebFetchTool()
        let result = try await tool.execute("s2", ["url": .string("http://[::1]/")], nil, nil)
        #expect(result.content.textContent.contains("private"))
    }

    @Test func blocks172PrivateRange() async throws {
        let tool = createWebFetchTool()
        let result = try await tool.execute("s3", ["url": .string("http://172.16.0.1/")], nil, nil)
        #expect(result.content.textContent.contains("private"))
    }

    @Test func blocks169LinkLocal() async throws {
        let tool = createWebFetchTool()
        let result = try await tool.execute("s4", ["url": .string("http://169.254.1.1/")], nil, nil)
        #expect(result.content.textContent.contains("private"))
    }

    @Test func blocksLocalDomain() async throws {
        let tool = createWebFetchTool()
        let result = try await tool.execute("s5", ["url": .string("http://myserver.local/")], nil, nil)
        #expect(result.content.textContent.contains("private"))
    }

    @Test func allowsPublicURL() async throws {
        // This shouldn't be blocked by SSRF check (it will fail on network, not validation)
        let tool = createWebFetchTool()
        let result = try await tool.execute("s6", ["url": .string("https://93.184.216.34/")], nil, nil)
        // Should NOT contain "private" — should fail with network error instead
        #expect(!result.content.textContent.contains("private"))
    }
}

// MARK: - WebToolsExtension Updated Tests

@MainActor
struct WebToolsExtensionFetchTests {

    @Test func extensionRegistersBothTools() {
        let ext = WebToolsExtension()
        #expect(ext.tools.count == 2)
        #expect(ext.tools["web_search"] != nil)
        #expect(ext.tools["web_fetch"] != nil)
    }
}

// MARK: - Whitespace Collapse Tests

@MainActor
struct WhitespaceCollapseTests {

    @Test func collapsesMultipleNewlines() {
        let input = "Hello\n\n\n\n\nWorld"
        let result = WebContentExtractor.collapseWhitespace(input)
        #expect(result == "Hello\n\nWorld")
    }

    @Test func collapsesMultipleSpaces() {
        let input = "Hello    World     Foo"
        let result = WebContentExtractor.collapseWhitespace(input)
        #expect(result == "Hello World Foo")
    }

    @Test func preservesDoubleNewline() {
        let input = "Paragraph 1\n\nParagraph 2"
        let result = WebContentExtractor.collapseWhitespace(input)
        #expect(result == "Paragraph 1\n\nParagraph 2")
    }

    @Test func trimsLeadingAndTrailingWhitespace() {
        let input = "\n\n  Hello World  \n\n"
        let result = WebContentExtractor.collapseWhitespace(input)
        #expect(result == "Hello World")
    }

    @Test func normalizesCRLF() {
        let input = "Line 1\r\nLine 2\rLine 3"
        let result = WebContentExtractor.collapseWhitespace(input)
        #expect(result == "Line 1\nLine 2\nLine 3")
    }
}

// MARK: - Readability + Demark Integration Tests

@MainActor
struct ReadabilityExtractionTests {

    private static let sampleURL = URL(string: "https://example.com/article")!

    /// Well-structured article HTML that Readability should handle.
    private static let articleHTML = """
    <!DOCTYPE html>
    <html>
    <head><title>Swift Concurrency Guide</title></head>
    <body>
        <nav><a href="/">Home</a></nav>
        <article>
            <h1>Swift Concurrency Guide</h1>
            <p>Swift's concurrency model provides <strong>structured concurrency</strong> with async/await.</p>
            <h2>async/await</h2>
            <p>The <code>async</code> keyword marks a function that can suspend:</p>
            <pre><code>func fetchData() async throws -&gt; Data {
        let (data, _) = try await URLSession.shared.data(from: url)
        return data
    }</code></pre>
            <h2>Task Groups</h2>
            <p>Use <a href="https://docs.swift.org">task groups</a> for parallel work.</p>
            <ul>
                <li>Structured concurrency</li>
                <li>Automatic cancellation</li>
            </ul>
        </article>
        <footer><p>Copyright 2026</p></footer>
    </body>
    </html>
    """

    @Test func readabilityExtractsTitle() async {
        let result = await WebContentExtractor.extract(html: Self.articleHTML, url: Self.sampleURL)
        #expect(result.title.contains("Swift Concurrency"))
    }

    @Test func readabilityProducesMarkdown() async {
        let result = await WebContentExtractor.extract(html: Self.articleHTML, url: Self.sampleURL)
        // Should contain markdown headings (from Demark)
        #expect(result.content.contains("# ") || result.content.contains("## "))
    }

    @Test func readabilityPreservesCodeBlocks() async {
        let result = await WebContentExtractor.extract(html: Self.articleHTML, url: Self.sampleURL)
        #expect(result.content.contains("fetchData"))
        #expect(result.content.contains("async"))
    }

    @Test func readabilityPreservesBoldText() async {
        let result = await WebContentExtractor.extract(html: Self.articleHTML, url: Self.sampleURL)
        // Demark should convert <strong> to **bold**
        #expect(result.content.contains("**structured concurrency**") || result.content.contains("structured concurrency"))
    }

    @Test func readabilityPreservesList() async {
        let result = await WebContentExtractor.extract(html: Self.articleHTML, url: Self.sampleURL)
        #expect(result.content.contains("Structured concurrency"))
        #expect(result.content.contains("Automatic cancellation"))
    }

    @Test func fallsBackForMinimalHTML() async {
        // HTML too minimal for Readability — should fall back to regex
        let html = "<html><body><p>Just a short paragraph.</p></body></html>"
        let result = await WebContentExtractor.extract(html: html, url: Self.sampleURL)
        #expect(result.content.contains("Just a short paragraph"))
    }

    @Test func preservesURL() async {
        let result = await WebContentExtractor.extract(html: Self.articleHTML, url: Self.sampleURL)
        #expect(result.url == Self.sampleURL)
    }
}

// MARK: - SPA Detection Tests

@MainActor
struct SPADetectionTests {

    @Test func detectsReactSPA() {
        let html = """
        <!DOCTYPE html><html><head><title>My App</title></head>
        <body><div id="root"></div>
        <script src="/static/js/main.abc123.js"></script>
        </body></html>
        """
        // Large HTML (padded), minimal text, React marker
        let padded = html + String(repeating: " ", count: 10_000)
        #expect(WebContentExtractor.isSuspectedSPA(extractedContent: "", rawHTML: padded))
    }

    @Test func detectsNextJSSPA() {
        let html = """
        <!DOCTYPE html><html><head></head>
        <body><div id="__next"></div></body></html>
        """
        let padded = html + String(repeating: " ", count: 10_000)
        #expect(WebContentExtractor.isSuspectedSPA(extractedContent: "Loading...", rawHTML: padded))
    }

    @Test func detectsVueSPA() {
        let html = """
        <!DOCTYPE html><html><head></head>
        <body><div id="app"></div></body></html>
        """
        let padded = html + String(repeating: " ", count: 10_000)
        #expect(WebContentExtractor.isSuspectedSPA(extractedContent: "", rawHTML: padded))
    }

    @Test func detectsLargeHTMLWithNoText() {
        // No SPA markers, but large HTML with very little extracted text
        let html = String(repeating: "<div class=\"wrapper\">", count: 500) + String(repeating: "</div>", count: 500)
        #expect(WebContentExtractor.isSuspectedSPA(extractedContent: "Loading", rawHTML: html))
    }

    @Test func doesNotTriggerForNormalPage() {
        let html = "<html><body><p>Normal content here.</p></body></html>"
        let content = "Normal content here. This is a well-structured article with enough text to be considered real content by the heuristic."
        // 100+ chars of content from small HTML — not an SPA
        #expect(!WebContentExtractor.isSuspectedSPA(extractedContent: content, rawHTML: html))
    }

    @Test func doesNotTriggerForSmallHTML() {
        // Small HTML even with no text — not an SPA, just a small page
        let html = "<html><body></body></html>"
        #expect(!WebContentExtractor.isSuspectedSPA(extractedContent: "", rawHTML: html))
    }

    @Test func doesNotTriggerForContentRichPage() {
        // Large HTML but also lots of extracted content — not an SPA
        let html = String(repeating: "<p>paragraph</p>", count: 1000)
        let content = String(repeating: "paragraph ", count: 1000)
        #expect(!WebContentExtractor.isSuspectedSPA(extractedContent: content, rawHTML: html))
    }

    @Test func markerDetectionIsCaseInsensitive() {
        let html = """
        <html><body><DIV ID="root"></DIV></body></html>
        """
        let padded = html + String(repeating: " ", count: 10_000)
        #expect(WebContentExtractor.isSuspectedSPA(extractedContent: "", rawHTML: padded))
    }
}

// MARK: - Dynamic Tool Toggle Tests

@MainActor
struct DynamicToolToggleTests {

    /// Helper: create a minimal set of mock tool definitions for testing.
    private static func mockTools() -> [AgentToolDefinition] {
        let readTool = AgentToolDefinition(
            name: "read",
            label: "Read",
            description: "Read a file",
            parameterSchema: JSONSchema(type: "object", properties: [:], required: []),
            execute: { _, _, _, _ in .text("ok") }
        )
        let searchTool = createWebSearchTool()
        let fetchTool = createWebFetchTool()
        return [readTool, searchTool, fetchTool]
    }

    private static let webToolNames: Set<String> = ["web_search", "web_fetch"]

    @Test func filteringExcludesWebToolsWhenDisabled() {
        let allTools = Self.mockTools()
        let filtered = allTools.filter { !Self.webToolNames.contains($0.name) }
        #expect(filtered.count == 1)
        #expect(filtered[0].name == "read")
    }

    @Test func filteringIncludesAllToolsWhenEnabled() {
        let allTools = Self.mockTools()
        // When enabled, pass all tools through
        #expect(allTools.count == 3)
        #expect(allTools.contains(where: { $0.name == "web_search" }))
        #expect(allTools.contains(where: { $0.name == "web_fetch" }))
        #expect(allTools.contains(where: { $0.name == "read" }))
    }

    @Test func webToolNamesMatchExtensionTools() {
        let ext = WebToolsExtension()
        // The names we filter on must match the actual tool names
        #expect(ext.tools["web_search"]?.name == "web_search")
        #expect(ext.tools["web_fetch"]?.name == "web_fetch")
        // Verify no unexpected tools
        #expect(ext.tools.count == 2)
    }

    @Test func extensionAlwaysRegistersBothTools() {
        // WebToolsExtension should always provide both tools regardless of settings
        let ext = WebToolsExtension()
        #expect(ext.tools.keys.contains("web_search"))
        #expect(ext.tools.keys.contains("web_fetch"))
    }
}
