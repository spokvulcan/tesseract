import Foundation

// MARK: - WebContentExtractor

/// Extracts readable text content from raw HTML using Foundation regex.
/// Strips boilerplate (nav, ads, scripts, styles) and decodes entities.
nonisolated enum WebContentExtractor: Sendable {

    struct ExtractedContent: Sendable {
        let title: String
        let text: String
        let url: URL
    }

    // MARK: - Static Regex Patterns

    /// Extracts <title>...</title> content.
    private static let titleRegex = try! NSRegularExpression(
        pattern: #"<title[^>]*>([\s\S]*?)</title>"#,
        options: .caseInsensitive
    )

    /// Matches entire block elements that should be removed (tag + content).
    /// Uses backreference `\1` to ensure closing tag matches opening tag name.
    private static let noiseBlockRegex = try! NSRegularExpression(
        pattern: #"<(script|style|nav|footer|header|aside|noscript|svg|iframe|form)\b[^>]*>[\s\S]*?</\1>"#,
        options: .caseInsensitive
    )

    /// Matches self-closing or void noise elements (e.g., <script src="..."/>).
    private static let noiseSelfClosingRegex = try! NSRegularExpression(
        pattern: #"<(script|style|link|meta|svg|iframe)\b[^>]*/\s*>"#,
        options: .caseInsensitive
    )

    /// Matches elements with noise classes. Uses backreference to match closing tag.
    private static let noiseClassRegex = try! NSRegularExpression(
        pattern: #"<(\w+)[^>]+class="[^"]*\b(ad|ads|advert|banner|cookie|popup|modal|sidebar|comment|share|social|login|signup|newsletter|promo)\b[^"]*"[^>]*>[\s\S]*?</\1>"#,
        options: .caseInsensitive
    )

    /// Matches HTML comments.
    private static let commentRegex = try! NSRegularExpression(
        pattern: #"<!--[\s\S]*?-->"#,
        options: []
    )

    /// Collapses 3+ newlines to double newline in a single pass.
    private static let excessNewlinesRegex = try! NSRegularExpression(
        pattern: #"\n{3,}"#,
        options: []
    )

    /// Collapses runs of horizontal whitespace (spaces/tabs) to a single space.
    private static let horizontalSpaceRegex = try! NSRegularExpression(
        pattern: #"[^\S\n]{2,}"#,
        options: []
    )

    // MARK: - Public API

    /// Extract readable text from HTML, stripping boilerplate.
    static func extract(html: String, url: URL) -> ExtractedContent {
        let title = extractTitle(from: html)
        let cleaned = removeNoise(from: html)
        let text = HTMLUtilities.stripHTMLTags(cleaned)
        let decoded = HTMLUtilities.decodeHTMLEntities(text)
        let collapsed = collapseWhitespace(decoded)

        return ExtractedContent(title: title, text: collapsed, url: url)
    }

    // MARK: - Extraction Steps (internal for testability)

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

        // Collapse 3+ newlines → double newline (single regex pass)
        let nsResult1 = result as NSString
        result = excessNewlinesRegex.stringByReplacingMatches(
            in: result,
            range: NSRange(location: 0, length: nsResult1.length),
            withTemplate: "\n\n"
        )

        // Collapse runs of horizontal whitespace → single space (single regex pass)
        let nsResult2 = result as NSString
        result = horizontalSpaceRegex.stringByReplacingMatches(
            in: result,
            range: NSRange(location: 0, length: nsResult2.length),
            withTemplate: " "
        )

        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
