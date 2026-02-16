/// Extracts quoted text from prompts for glyph injection.
/// Matches text between both single and double quotes.
public enum GlyphTextExtractor {

    /// Extract all quoted text from a prompt (both 'single' and "double" quotes).
    public nonisolated static func extractQuotedText(from prompt: String) -> [String] {
        var results = [String]()
        var remaining = prompt[...]

        while !remaining.isEmpty {
            // Find the next opening quote (single or double)
            guard let nextQuote = remaining.firstIndex(where: { $0 == "'" || $0 == "\"" || $0 == "\u{201C}" || $0 == "\u{2018}" }) else { break }

            let quoteChar = remaining[nextQuote]
            // Determine matching close quote
            let closeChar: Character
            switch quoteChar {
            case "'": closeChar = "'"
            case "\"": closeChar = "\""
            case "\u{201C}": closeChar = "\u{201D}"  // " → "
            case "\u{2018}": closeChar = "\u{2019}"  // ' → '
            default: closeChar = quoteChar
            }

            let afterOpen = remaining.index(after: nextQuote)
            guard afterOpen < remaining.endIndex else { break }

            let searchRange = afterOpen..<remaining.endIndex
            guard let closeQuote = remaining[searchRange].firstIndex(of: closeChar) else {
                // No matching close quote — skip this char and continue
                remaining = remaining[afterOpen...]
                continue
            }

            let extracted = String(remaining[afterOpen..<closeQuote])
            // Filter out very long strings (likely descriptions, not display text)
            // and very short ones (likely punctuation artifacts)
            if extracted.count >= 2 && extracted.count <= 200 {
                results.append(extracted)
            }
            remaining = remaining[remaining.index(after: closeQuote)...]
        }

        return results
    }

    /// Returns the first quoted text, or nil if none found.
    public nonisolated static func firstQuotedText(from prompt: String) -> String? {
        extractQuotedText(from: prompt).first
    }
}
