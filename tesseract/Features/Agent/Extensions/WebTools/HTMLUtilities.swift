import Foundation

// MARK: - HTMLUtilities

/// Shared HTML entity decoding and tag stripping used by
/// ``WebContentExtractor`` when distilling fetched/read pages.
nonisolated enum HTMLUtilities: Sendable {

    // MARK: - Regex Patterns

    // Static literal pattern — compilation cannot fail (matches ToolCallParser precedent).
    // swiftlint:disable force_try
    /// Matches any HTML tag.
    static let htmlTagRegex = try! NSRegularExpression(
        pattern: #"<[^>]+>"#,
        options: []
    )
    // swiftlint:enable force_try

    /// Named HTML entities lookup.
    static let namedEntities: [String: String] = [
        "&amp;": "&", "&lt;": "<", "&gt;": ">",
        "&quot;": "\"", "&apos;": "'", "&#39;": "'",
        "&nbsp;": " ", "&mdash;": "—", "&ndash;": "–",
        "&laquo;": "«", "&raquo;": "»", "&hellip;": "…",
        "&copy;": "©", "&reg;": "®", "&trade;": "™",
    ]

    // Static literal pattern — compilation cannot fail (matches ToolCallParser precedent).
    // swiftlint:disable:next force_try
    private static let numericEntityRegex = try! NSRegularExpression(
        pattern: #"&#(\d+);"#,
        options: []
    )

    // Static literal pattern — compilation cannot fail (matches ToolCallParser precedent).
    // swiftlint:disable:next force_try
    private static let hexEntityRegex = try! NSRegularExpression(
        pattern: #"&#x([0-9a-fA-F]+);"#,
        options: .caseInsensitive
    )

    // MARK: - Public API

    /// Decode HTML entities (named + numeric/hex).
    static func decodeHTMLEntities(_ string: String) -> String {
        var result = string
        for (entity, replacement) in namedEntities {
            result = result.replacingOccurrences(of: entity, with: replacement)
        }
        result = replaceNumericEntities(in: result, regex: numericEntityRegex, radix: 10)
        result = replaceNumericEntities(in: result, regex: hexEntityRegex, radix: 16)
        return result
    }

    /// Strip all HTML tags from a string.
    static func stripHTMLTags(_ html: String) -> String {
        let nsString = html as NSString
        let range = NSRange(location: 0, length: nsString.length)
        return htmlTagRegex.stringByReplacingMatches(in: html, range: range, withTemplate: " ")
    }

    // MARK: - Private

    private static func replaceNumericEntities(
        in string: String, regex: NSRegularExpression, radix: Int
    ) -> String {
        let nsString = string as NSString
        let range = NSRange(location: 0, length: nsString.length)
        let matches = regex.matches(in: string, range: range)

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
