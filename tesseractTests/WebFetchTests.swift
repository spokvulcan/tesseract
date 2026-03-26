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
        let result = WebContentExtractor.extract(html: Self.fullPageHTML, url: Self.sampleURL)
        #expect(result.title == "Swift 6.2 Released — The Swift Blog")
    }

    @Test func stripsScriptAndStyleContent() {
        let result = WebContentExtractor.extract(html: Self.fullPageHTML, url: Self.sampleURL)
        #expect(!result.text.contains("analytics"))
        #expect(!result.text.contains("trackPageView"))
        #expect(!result.text.contains("font-family"))
    }

    @Test func stripsNavAndFooter() {
        let result = WebContentExtractor.extract(html: Self.fullPageHTML, url: Self.sampleURL)
        // Nav links should be removed
        #expect(!result.text.contains("Home | Blog"))
        // Footer should be removed
        #expect(!result.text.contains("2026 Apple Inc"))
    }

    @Test func preservesArticleContent() {
        let result = WebContentExtractor.extract(html: Self.fullPageHTML, url: Self.sampleURL)
        #expect(result.text.contains("Swift 6.2 Released"))
        #expect(result.text.contains("excited to announce"))
        #expect(result.text.contains("improved concurrency & performance"))
    }

    @Test func decodesHTMLEntities() {
        let result = WebContentExtractor.extract(html: Self.fullPageHTML, url: Self.sampleURL)
        // &#x27; should become '
        #expect(result.text.contains("We're excited"))
        // &amp; should become &
        #expect(result.text.contains("concurrency & performance"))
    }

    @Test func stripsAsideSidebar() {
        let result = WebContentExtractor.extract(html: Self.fullPageHTML, url: Self.sampleURL)
        #expect(!result.text.contains("Popular Posts"))
    }

    @Test func preservesURLInResult() {
        let result = WebContentExtractor.extract(html: Self.fullPageHTML, url: Self.sampleURL)
        #expect(result.url == Self.sampleURL)
    }

    @Test func handlesEmptyHTML() {
        let result = WebContentExtractor.extract(html: "", url: Self.sampleURL)
        #expect(result.title == "")
        #expect(result.text == "")
    }

    @Test func handlesHTMLWithNoTitle() {
        let html = "<html><body><p>Just some text.</p></body></html>"
        let result = WebContentExtractor.extract(html: html, url: Self.sampleURL)
        #expect(result.title == "")
        #expect(result.text.contains("Just some text"))
    }

    @Test func handlesHTMLWithOnlyBoilerplate() {
        let html = """
        <html><body>
        <script>lots of js code here</script>
        <style>.class { color: red; }</style>
        <nav>Navigation links</nav>
        </body></html>
        """
        let result = WebContentExtractor.extract(html: html, url: Self.sampleURL)
        #expect(!result.text.contains("lots of js"))
        #expect(!result.text.contains("color: red"))
    }

    @Test func collapsesExcessiveWhitespace() {
        let html = "<html><body><p>Line 1</p>\n\n\n\n\n<p>Line 2</p></body></html>"
        let result = WebContentExtractor.extract(html: html, url: Self.sampleURL)
        // Should not have more than double newline
        #expect(!result.text.contains("\n\n\n"))
    }

    @Test func stripsHTMLComments() {
        let html = "<html><body><!-- This is a comment --><p>Visible text</p></body></html>"
        let result = WebContentExtractor.extract(html: html, url: Self.sampleURL)
        #expect(!result.text.contains("This is a comment"))
        #expect(result.text.contains("Visible text"))
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
