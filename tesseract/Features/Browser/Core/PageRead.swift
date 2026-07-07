import Foundation

// MARK: - PageReadPaginator

/// Pure transform: paginate a distilled **Page Read** (Markdown) under a hard
/// character budget, breaking on a paragraph/line boundary near the cap so a
/// chunk never splits mid-sentence. Stateless — the cursor is a character
/// offset into deterministically re-derived content, so pagination needs no
/// server-side bookmark. Unit-tested as the lower browser test seam.
nonisolated enum PageReadPaginator {

    struct Chunk: Sendable, Equatable {
        /// The slice of content for this page.
        let text: String
        /// Character offset to pass as `cursor` for the next page, or nil when
        /// the end has been reached.
        let nextCursor: Int?
        /// Total character length of the full content (for steering text).
        let totalChars: Int
    }

    /// - Parameters:
    ///   - content: the full distilled Markdown.
    ///   - cursor: character offset to start from (clamped to `[0, count]`).
    ///   - maxChars: soft cap; the actual break lands on the nearest earlier
    ///     paragraph/line boundary, else a hard cut at `maxChars`.
    static func paginate(_ content: String, cursor: Int = 0, maxChars: Int = 20_000) -> Chunk {
        let total = content.count
        let safeCap = max(1, maxChars)
        let start = min(max(0, cursor), total)

        guard start < total else {
            return Chunk(text: "", nextCursor: nil, totalChars: total)
        }

        let startIndex = content.index(content.startIndex, offsetBy: start)
        let remaining = total - start

        if remaining <= safeCap {
            return Chunk(
                text: String(content[startIndex...]),
                nextCursor: nil,
                totalChars: total
            )
        }

        // More than a page remains — find a clean break within the window.
        let windowEnd = content.index(startIndex, offsetBy: safeCap)
        let window = content[startIndex..<windowEnd]

        let breakOffset = boundaryOffset(in: window, cap: safeCap)
        let cutIndex = content.index(startIndex, offsetBy: breakOffset)
        let chunkText = String(content[startIndex..<cutIndex])

        return Chunk(
            text: chunkText,
            nextCursor: start + breakOffset,
            totalChars: total
        )
    }

    /// Character offset (from the window start) of the best break: last
    /// paragraph break, else last newline, else the hard cap. Never returns 0
    /// (guarantees forward progress).
    private static func boundaryOffset(in window: Substring, cap: Int) -> Int {
        if let range = window.range(of: "\n\n", options: .backwards) {
            let offset = window.distance(from: window.startIndex, to: range.lowerBound)
            if offset > 0 { return offset }
        }
        if let range = window.range(of: "\n", options: .backwards) {
            let offset = window.distance(from: window.startIndex, to: range.lowerBound)
            if offset > 0 { return offset }
        }
        return cap
    }
}
