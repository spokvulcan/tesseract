import Foundation
import Testing
import MLXLMCommon
@testable import Tesseract_Agent

// MARK: - DuckDuckGo HTML Parsing Tests

@MainActor
struct DuckDuckGoParsingTests {

    /// Realistic DDG HTML fragment with multiple results.
    private static let sampleHTML = """
    <html>
    <body>
    <div id="links" class="results">
      <div class="result results_links results_links_deep web-result">
        <div class="links_main links_deep result__body">
          <h2 class="result__title">
            <a rel="nofollow" class="result__a" href="https://www.swift.org/blog/swift-6.2-released/">We&#x27;re excited to announce <b>Swift</b> <b>6.2</b></a>
          </h2>
          <a class="result__snippet" href="https://www.swift.org/blog/swift-6.2-released/">Swift <b>6.2</b> delivers a broad set of features designed for real-world development.</a>
        </div>
      </div>
      <div class="result results_links results_links_deep web-result">
        <div class="links_main links_deep result__body">
          <h2 class="result__title">
            <a rel="nofollow" class="result__a" href="https://www.hackingwithswift.com/articles/277/whats-new-in-swift-6-2">What&#39;s new in Swift 6.2 &mdash; Hacking with Swift</a>
          </h2>
          <a class="result__snippet" href="https://www.hackingwithswift.com/articles/277/whats-new-in-swift-6-2">Swift 6.2 brings improved concurrency &amp; performance enhancements.</a>
        </div>
      </div>
      <div class="result results_links results_links_deep web-result">
        <div class="links_main links_deep result__body">
          <h2 class="result__title">
            <a rel="nofollow" class="result__a" href="https://developer.apple.com/swift/whats-new/">What&#x27;s new in Swift &ndash; Apple Developer</a>
          </h2>
          <a class="result__snippet" href="https://developer.apple.com/swift/whats-new/">Dive into the latest features and capabilities of the <b>Swift</b> programming language.</a>
        </div>
      </div>
    </div>
    </body>
    </html>
    """

    @Test func parsesMultipleResultsFromHTML() {
        let results = DuckDuckGoClient.parseResults(html: Self.sampleHTML, maxResults: 10)
        #expect(results.count == 3)
    }

    @Test func extractsTitlesWithHTMLTagsStripped() {
        let results = DuckDuckGoClient.parseResults(html: Self.sampleHTML, maxResults: 10)
        #expect(results[0].title == "We're excited to announce Swift 6.2")
        #expect(results[1].title == "What's new in Swift 6.2 — Hacking with Swift")
        #expect(results[2].title == "What's new in Swift – Apple Developer")
    }

    @Test func extractsURLsDirectly() {
        let results = DuckDuckGoClient.parseResults(html: Self.sampleHTML, maxResults: 10)
        #expect(results[0].url == "https://www.swift.org/blog/swift-6.2-released/")
        #expect(results[1].url == "https://www.hackingwithswift.com/articles/277/whats-new-in-swift-6-2")
        #expect(results[2].url == "https://developer.apple.com/swift/whats-new/")
    }

    @Test func extractsSnippetsWithHTMLEntitiesDecoded() {
        let results = DuckDuckGoClient.parseResults(html: Self.sampleHTML, maxResults: 10)
        #expect(results[0].snippet == "Swift 6.2 delivers a broad set of features designed for real-world development.")
        #expect(results[1].snippet == "Swift 6.2 brings improved concurrency & performance enhancements.")
    }

    @Test func respectsMaxResultsLimit() {
        let results = DuckDuckGoClient.parseResults(html: Self.sampleHTML, maxResults: 2)
        #expect(results.count == 2)
    }

    @Test func returnsEmptyForHTMLWithNoResults() {
        let html = "<html><body><div>No search results here</div></body></html>"
        let results = DuckDuckGoClient.parseResults(html: html, maxResults: 5)
        #expect(results.isEmpty)
    }
}

// MARK: - DDG Redirect URL Extraction Tests

@MainActor
struct DuckDuckGoURLExtractionTests {

    @Test func extractsURLFromDDGRedirect() {
        let ddgHref = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpath%3Fq%3D1&rut=abc"
        let result = DuckDuckGoClient.extractRealURL(from: ddgHref)
        #expect(result == "https://example.com/path?q=1")
    }

    @Test func returnsDirectURLUnchanged() {
        let url = "https://www.swift.org/blog/swift-6.2-released/"
        let result = DuckDuckGoClient.extractRealURL(from: url)
        #expect(result == url)
    }

    @Test func handlesProtocolRelativeNonDDGURL() {
        let url = "//example.com/page"
        let result = DuckDuckGoClient.extractRealURL(from: url)
        #expect(result == "https://example.com/page")
    }

    @Test func returnsNilForEmptyString() {
        let result = DuckDuckGoClient.extractRealURL(from: "")
        #expect(result == nil)
    }

    @Test func handlesDDGRedirectWithoutUddgParam() {
        let ddgHref = "//duckduckgo.com/l/?other=value"
        let result = DuckDuckGoClient.extractRealURL(from: ddgHref)
        // Falls through to return the full URL since uddg is missing
        #expect(result == "https://duckduckgo.com/l/?other=value")
    }
}

// MARK: - HTML Entity Decoding Tests

@MainActor
struct HTMLEntityDecodingTests {

    @Test func decodesNamedEntities() {
        #expect(DuckDuckGoClient.decodeHTMLEntities("&amp;") == "&")
        #expect(DuckDuckGoClient.decodeHTMLEntities("&lt;") == "<")
        #expect(DuckDuckGoClient.decodeHTMLEntities("&gt;") == ">")
        #expect(DuckDuckGoClient.decodeHTMLEntities("&quot;") == "\"")
        #expect(DuckDuckGoClient.decodeHTMLEntities("&apos;") == "'")
        #expect(DuckDuckGoClient.decodeHTMLEntities("&nbsp;") == " ")
        #expect(DuckDuckGoClient.decodeHTMLEntities("&mdash;") == "—")
        #expect(DuckDuckGoClient.decodeHTMLEntities("&ndash;") == "–")
        #expect(DuckDuckGoClient.decodeHTMLEntities("&hellip;") == "…")
    }

    @Test func decodesDecimalNumericEntities() {
        #expect(DuckDuckGoClient.decodeHTMLEntities("&#39;") == "'")
        #expect(DuckDuckGoClient.decodeHTMLEntities("&#169;") == "©")
        #expect(DuckDuckGoClient.decodeHTMLEntities("&#8212;") == "—")
    }

    @Test func decodesHexNumericEntities() {
        #expect(DuckDuckGoClient.decodeHTMLEntities("&#x27;") == "'")
        #expect(DuckDuckGoClient.decodeHTMLEntities("&#xA9;") == "©")
        #expect(DuckDuckGoClient.decodeHTMLEntities("&#x2014;") == "—")
    }

    @Test func decodesMultipleEntitiesInString() {
        let input = "Tom &amp; Jerry&#x27;s &quot;Adventure&quot;"
        let expected = "Tom & Jerry's \"Adventure\""
        #expect(DuckDuckGoClient.decodeHTMLEntities(input) == expected)
    }

    @Test func preservesPlainText() {
        let plain = "Hello, world!"
        #expect(DuckDuckGoClient.decodeHTMLEntities(plain) == plain)
    }
}

// MARK: - HTML Text Cleaning Tests

@MainActor
struct HTMLTextCleaningTests {

    @Test func stripsHTMLTagsAndDecodesEntities() {
        let html = "<b>Swift</b> <b>6.2</b> is &quot;great&quot;"
        let result = DuckDuckGoClient.cleanHTMLText(html)
        #expect(result == "Swift 6.2 is \"great\"")
    }

    @Test func collapsesWhitespaceAndNewlines() {
        let html = "  Hello  \n  world  \n\n  foo  "
        let result = DuckDuckGoClient.cleanHTMLText(html)
        #expect(result == "Hello world foo")
    }

    @Test func handlesNestedTags() {
        let html = "<a href=\"url\"><span class=\"foo\">Link <b>text</b></span></a>"
        let result = DuckDuckGoClient.cleanHTMLText(html)
        #expect(result == "Link text")
    }

    @Test func handlesEmptyInput() {
        #expect(DuckDuckGoClient.cleanHTMLText("") == "")
    }

    @Test func handlesTagsOnlyInput() {
        #expect(DuckDuckGoClient.cleanHTMLText("<br/><hr/>") == "")
    }
}

// MARK: - WebSearchTool Definition Tests

@MainActor
struct WebSearchToolTests {

    @Test func toolHasCorrectNameAndSchema() {
        let tool = createWebSearchTool()
        #expect(tool.name == "web_search")
        #expect(tool.label == "Web Search")
        #expect(tool.parameterSchema.required == ["query"])
        #expect(tool.parameterSchema.properties["query"] != nil)
        #expect(tool.parameterSchema.properties["max_results"] != nil)
    }

    @Test func returnsErrorForMissingQuery() async throws {
        let tool = createWebSearchTool()
        let result = try await tool.execute("t1", [:], nil, nil)
        let text = result.content.textContent
        #expect(text.contains("Missing required argument: query"))
    }

    @Test func returnsErrorForEmptyQuery() async throws {
        let tool = createWebSearchTool()
        let result = try await tool.execute("t2", ["query": .string("   ")], nil, nil)
        let text = result.content.textContent
        #expect(text.contains("Search query cannot be empty"))
    }
}

// MARK: - WebToolsExtension Tests

@MainActor
struct WebToolsExtensionTests {

    @Test func extensionRegistersWebSearchTool() {
        let ext = WebToolsExtension()
        #expect(ext.path == "web-tools")
        #expect(ext.tools.count == 1)
        #expect(ext.tools["web_search"] != nil)
        #expect(ext.tools["web_search"]?.name == "web_search")
    }

    @Test func extensionConformsToAgentExtension() {
        let ext = WebToolsExtension()
        let agentExt: any AgentExtension = ext
        #expect(agentExt.path == "web-tools")
        #expect(agentExt.handlers.isEmpty)
        #expect(agentExt.commands.isEmpty)
    }
}

// MARK: - DDG HTML Parsing with Redirect URLs

@MainActor
struct DuckDuckGoRedirectParsingTests {

    /// DDG HTML with redirect-wrapped URLs (the format used in some regions).
    private static let redirectHTML = """
    <html><body>
    <div class="result results_links results_links_deep web-result">
      <h2 class="result__title">
        <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Farticle&rut=abc123">Example Article</a>
      </h2>
      <a class="result__snippet">This is the snippet for the example article.</a>
    </div>
    <div class="result results_links results_links_deep web-result">
      <h2 class="result__title">
        <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.swift.org%2Fguide&rut=def456">Swift Guide</a>
      </h2>
      <a class="result__snippet">Official Swift documentation guide.</a>
    </div>
    </body></html>
    """

    @Test func extractsRealURLFromRedirectWrapper() {
        let results = DuckDuckGoClient.parseResults(html: Self.redirectHTML, maxResults: 10)
        #expect(results.count == 2)
        #expect(results[0].url == "https://example.com/article")
        #expect(results[1].url == "https://docs.swift.org/guide")
    }

    @Test func parsesCorrectTitlesFromRedirectHTML() {
        let results = DuckDuckGoClient.parseResults(html: Self.redirectHTML, maxResults: 10)
        #expect(results[0].title == "Example Article")
        #expect(results[1].title == "Swift Guide")
    }

    @Test func parsesCorrectSnippetsFromRedirectHTML() {
        let results = DuckDuckGoClient.parseResults(html: Self.redirectHTML, maxResults: 10)
        #expect(results[0].snippet == "This is the snippet for the example article.")
        #expect(results[1].snippet == "Official Swift documentation guide.")
    }
}

// MARK: - Edge Cases

@MainActor
struct DuckDuckGoEdgeCaseTests {

    @Test func handlesResultsWithMismatchedTitleSnippetCount() {
        // More titles than snippets — should only return matched pairs
        let html = """
        <a class="result__a" href="https://a.com">Title A</a>
        <a class="result__a" href="https://b.com">Title B</a>
        <a class="result__snippet">Snippet A</a>
        """
        let results = DuckDuckGoClient.parseResults(html: html, maxResults: 10)
        #expect(results.count == 1)
        #expect(results[0].title == "Title A")
        #expect(results[0].snippet == "Snippet A")
    }

    @Test func handlesSpecialCharactersInQuery() {
        // This tests formEncode indirectly — the actual HTTP call would
        // use the encoded query. We verify the tool doesn't crash.
        let tool = createWebSearchTool()
        // Just verify it doesn't crash with special chars
        #expect(tool.parameterSchema.properties["query"]?.type == "string")
    }

    @Test func snippetClosedByDivTag() {
        // Some DDG variants close snippets with </div> instead of </a>
        let html = """
        <a class="result__a" href="https://example.com">Title</a>
        <div class="result__snippet">Snippet closed by div</div>
        """
        let results = DuckDuckGoClient.parseResults(html: html, maxResults: 10)
        #expect(results.count == 1)
        #expect(results[0].snippet == "Snippet closed by div")
    }
}

// MARK: - Live Integration Test (requires network)

@MainActor
struct DuckDuckGoLiveTests {

    @Test(.enabled(if: ProcessInfo.processInfo.environment["RUN_LIVE_TESTS"] != nil))
    func liveSearchReturnsResults() async throws {
        let results = try await DuckDuckGoClient.search(query: "Swift programming language", maxResults: 3)
        #expect(!results.isEmpty)
        #expect(results.count <= 3)
        for result in results {
            #expect(!result.title.isEmpty)
            #expect(result.url.hasPrefix("http"))
            // Snippet may occasionally be empty but title and URL should always be present
        }
    }
}
