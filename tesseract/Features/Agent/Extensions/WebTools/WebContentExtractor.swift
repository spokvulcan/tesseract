import Foundation
import SwiftReadability
import Demark

// MARK: - WebContentExtractor

/// Extracts readable content from raw HTML.
///
/// Primary path: swift-readability (Mozilla Readability port) extracts the article,
/// then Demark converts the clean HTML to markdown.
/// Fallback: regex-based noise removal + tag stripping when Readability returns nil.
nonisolated enum WebContentExtractor: Sendable {

    struct ExtractedContent: Sendable {
        let title: String
        let content: String
        let url: URL
    }

    // MARK: - Public API

    /// Extract readable content from HTML as markdown.
    /// Pipeline: Readability+Demark → SPA detection+WebPage render → regex fallback.
    static func extract(html: String, url: URL) async -> ExtractedContent {
        // 1. Try Readability + Demark on the raw HTML (fast path)
        var fastResult: ExtractedContent?
        do {
            fastResult = try await extractWithReadability(html: html, url: url)
        } catch {
            Log.agent.debug("[WebContentExtractor] Readability/Demark failed: \(error)")
        }

        // 2. If we got content, check if it looks suspiciously thin (possible SPA)
        if let result = fastResult {
            if isSuspectedSPA(extractedContent: result.content, rawHTML: html) {
                if let rendered = await renderAndExtract(url: url) {
                    return rendered
                }
            }
            return result
        }

        // 3. Readability returned nil — check if SPA before falling back to regex
        if isSuspectedSPA(extractedContent: "", rawHTML: html) {
            if let rendered = await renderAndExtract(url: url) {
                return rendered
            }
        }

        // 4. Final fallback: regex-based extraction
        return extractBasic(html: html, url: url)
    }

    // MARK: - SPA Detection

    private static let spaMarkers = [
        "<div id=\"root\"></div>",
        "<div id=\"app\"></div>",
        "<div id=\"__next\"></div>",
        "<div id=\"__nuxt\"></div>",
    ]

    /// Heuristic: large HTML but minimal extracted text suggests JS-rendered content.
    static func isSuspectedSPA(extractedContent: String, rawHTML: String) -> Bool {
        let textLen = extractedContent.trimmingCharacters(in: .whitespacesAndNewlines).count
        let htmlLen = rawHTML.utf8.count

        // Strong signal: very little text from large HTML
        if textLen < 200 && htmlLen > 10_000 {
            return true
        }

        // SPA framework markers in HTML with little text
        if textLen < 500 {
            if spaMarkers.contains(where: { rawHTML.range(of: $0, options: .caseInsensitive) != nil }) {
                return true
            }
        }

        return false
    }

    // MARK: - WebPage Render Fallback

    /// Render the page with WebPage API and re-extract from the rendered DOM.
    private static func renderAndExtract(url: URL) async -> ExtractedContent? {
        do {
            let renderedHTML = try await HeadlessRenderer.shared.render(url: url)
            if let result = try await extractWithReadability(html: renderedHTML, url: url) {
                return result
            }
            // Readability failed on rendered HTML too — try regex on it
            return extractBasic(html: renderedHTML, url: url)
        } catch {
            Log.agent.debug("[WebContentExtractor] WebPage render failed for \(url): \(error)")
            return nil
        }
    }

    // MARK: - Readability + Demark Pipeline

    private static func extractWithReadability(html: String, url: URL) async throws -> ExtractedContent? {
        let reader = Readability(html: html, url: url)
        guard let article = try reader.parse() else { return nil }

        let title = article.title ?? ""
        let contentHTML = article.contentHTML

        // Convert clean HTML to markdown via Demark (JSC engine, fast).
        // Demark must be created on MainActor (ConversionRuntime is @MainActor).
        let markdown = try await demarkConvert(contentHTML)

        return ExtractedContent(title: title, content: markdown, url: url)
    }

    /// Singleton Demark instance on MainActor — avoids recreating JSC context per call.
    @MainActor
    private static let sharedDemark = Demark()

    @MainActor
    private static func demarkConvert(_ html: String) async throws -> String {
        try await sharedDemark.convertToMarkdown(
            html,
            options: DemarkOptions(engine: .htmlToMd)
        )
    }

    // MARK: - Regex Fallback Pipeline

    /// Extracts <title>...</title> content.
    private static let titleRegex = try! NSRegularExpression(
        pattern: #"<title[^>]*>([\s\S]*?)</title>"#,
        options: .caseInsensitive
    )

    /// Matches entire noise block elements (tag + content) with backreference.
    private static let noiseBlockRegex = try! NSRegularExpression(
        pattern: #"<(script|style|nav|footer|header|aside|noscript|svg|iframe|form)\b[^>]*>[\s\S]*?</\1>"#,
        options: .caseInsensitive
    )

    private static let noiseSelfClosingRegex = try! NSRegularExpression(
        pattern: #"<(script|style|link|meta|svg|iframe)\b[^>]*/\s*>"#,
        options: .caseInsensitive
    )

    /// Matches elements with noise classes, backreference for closing tag.
    private static let noiseClassRegex = try! NSRegularExpression(
        pattern: #"<(\w+)[^>]+class="[^"]*\b(ad|ads|advert|banner|cookie|popup|modal|sidebar|comment|share|social|login|signup|newsletter|promo)\b[^"]*"[^>]*>[\s\S]*?</\1>"#,
        options: .caseInsensitive
    )

    private static let commentRegex = try! NSRegularExpression(
        pattern: #"<!--[\s\S]*?-->"#,
        options: []
    )

    private static let excessNewlinesRegex = try! NSRegularExpression(
        pattern: #"\n{3,}"#,
        options: []
    )

    private static let horizontalSpaceRegex = try! NSRegularExpression(
        pattern: #"[^\S\n]{2,}"#,
        options: []
    )

    /// Regex-based fallback when Readability can't extract article content.
    static func extractBasic(html: String, url: URL) -> ExtractedContent {
        let title = extractTitle(from: html)
        let cleaned = removeNoise(from: html)
        let text = HTMLUtilities.stripHTMLTags(cleaned)
        let decoded = HTMLUtilities.decodeHTMLEntities(text)
        let collapsed = collapseWhitespace(decoded)

        return ExtractedContent(title: title, content: collapsed, url: url)
    }

    private static func extractTitle(from html: String) -> String {
        let nsHTML = html as NSString
        let range = NSRange(location: 0, length: nsHTML.length)

        guard let match = titleRegex.firstMatch(in: html, range: range),
              match.numberOfRanges >= 2,
              let titleRange = Range(match.range(at: 1), in: html)
        else { return "" }

        let raw = String(html[titleRange])
        return HTMLUtilities.decodeHTMLEntities(raw).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func removeNoise(from html: String) -> String {
        var result = html
        let regexes = [commentRegex, noiseBlockRegex, noiseSelfClosingRegex, noiseClassRegex]

        for regex in regexes {
            let nsResult = result as NSString
            let range = NSRange(location: 0, length: nsResult.length)
            result = regex.stringByReplacingMatches(in: result, range: range, withTemplate: " ")
        }

        return result
    }

    /// Collapse runs of whitespace into readable formatting.
    static func collapseWhitespace(_ text: String) -> String {
        var result = text
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")

        let nsResult1 = result as NSString
        result = excessNewlinesRegex.stringByReplacingMatches(
            in: result,
            range: NSRange(location: 0, length: nsResult1.length),
            withTemplate: "\n\n"
        )

        let nsResult2 = result as NSString
        result = horizontalSpaceRegex.stringByReplacingMatches(
            in: result,
            range: NSRange(location: 0, length: nsResult2.length),
            withTemplate: " "
        )

        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
