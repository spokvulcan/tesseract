import SwiftUI

// MARK: - Chunked streaming text

/// Plain live-streaming text whose layout cost is O(new tail), not
/// O(accumulated text): completed paragraphs freeze into chunk `Text`s that
/// SwiftUI never re-lays-out (their strings are append-only and compare by
/// shared storage), so each publish re-measures only the short live tail.
/// This is the expanded thinking row's analog of the raw span list's
/// `LiveStreamingText` split — without it, a long reasoning stream re-lays
/// the entire accumulated thought on every publish, which grows until the
/// main thread misses frames wholesale.
///
/// Edge trim matches `chatDisplayTrimmed`: leading whitespace falls off the
/// first chunk, trailing whitespace off the live tail — interior formatting
/// is untouched. Text selection cannot cross chunk boundaries while the
/// stream is live; the committed row (one `Text`) restores full selection.
struct ChunkedStreamingText: View {
    let text: String

    /// Matches the interline gap a single `Text` would draw at the chat's
    /// `lineSpacing`, so the chunk seams are invisible.
    var spacing: CGFloat = chatLineSpacing

    @State private var frozen = ChunkedTextAccumulator()

    var body: some View {
        let tail = frozen.liveTail(of: text)
        VStack(alignment: .leading, spacing: spacing) {
            ForEach(frozen.chunks.indices, id: \.self) { index in
                Text(frozen.chunks[index])
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            Text(displayTail(tail))
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .onChange(of: text, initial: true) { _, newValue in
            frozen.freezeCompletedParagraphs(of: newValue)
        }
    }

    private func displayTail(_ tail: Substring) -> String {
        let trimmed =
            frozen.chunks.isEmpty
            ? tail.trimmingCharacters(in: .whitespacesAndNewlines)
            : String(tail).trimmingTrailingWhitespace()
        return trimmed
    }
}

extension String {
    fileprivate func trimmingTrailingWhitespace() -> String {
        guard let last = lastIndex(where: { !$0.isWhitespace }) else { return "" }
        return String(self[...last])
    }
}

// MARK: - Accumulator

/// The chunking state behind `ChunkedStreamingText`, kept separately so the
/// cut rules are unit-testable. Contract with the producers: the source text
/// only ever *grows* by appending; every non-append rewrite upstream (the
/// span cap's elision re-slice, the thinking-loop safeguard's truncate)
/// *shrinks* it, so a shrink is the reset signal.
nonisolated struct ChunkedTextAccumulator: Equatable {
    /// Frozen paragraph chunks — append-only, never mutated afterward, so a
    /// chunk `Text` keyed by its index is stable for the life of the stream.
    private(set) var chunks: [String] = []
    /// UTF-8 length of the consumed source prefix (frozen chunks plus each
    /// cut newline). Always lands just past an ASCII `\n`, so re-deriving
    /// the index is grapheme-safe.
    private(set) var consumedUTF8: Int = 0
    private var lastSeenUTF8: Int = 0

    /// Freeze once the live tail outgrows this many UTF-8 bytes and contains
    /// a newline to cut at. A pathological single-line stream never freezes,
    /// but the thinking-loop safeguard bounds that case at the thinking
    /// budget.
    static let freezeThresholdUTF8 = 4096

    /// Everything past the frozen prefix — what the live `Text` renders.
    /// Non-mutating so `body` can call it; a text that shrank since the last
    /// freeze renders whole until `freezeCompletedParagraphs` resets.
    func liveTail(of text: String) -> Substring {
        guard consumedUTF8 > 0, text.utf8.count >= consumedUTF8 else {
            return text[...]
        }
        return text[Self.index(text, atUTF8Offset: consumedUTF8)...]
    }

    /// Fold the latest full text: reset if it shrank (rewrite upstream),
    /// then move completed paragraphs from the tail into frozen chunks,
    /// cutting at the tail's last newline. The first chunk sheds leading
    /// whitespace so the frozen render matches `chatDisplayTrimmed`.
    mutating func freezeCompletedParagraphs(of text: String) {
        let total = text.utf8.count
        if total < lastSeenUTF8 || total < consumedUTF8 { self = .init() }
        lastSeenUTF8 = total

        let tailStart =
            consumedUTF8 > 0 ? Self.index(text, atUTF8Offset: consumedUTF8) : text.startIndex
        let tail = text[tailStart...]
        guard tail.utf8.count > Self.freezeThresholdUTF8,
            let cut = tail.lastIndex(of: "\n")
        else { return }

        var chunk = tail[..<cut]
        if chunks.isEmpty {
            chunk = chunk.drop(while: { $0.isWhitespace || $0.isNewline })
        }
        if !chunk.isEmpty {
            chunks.append(String(chunk))
        }
        consumedUTF8 += text.utf8.distance(from: tailStart, to: text.index(after: cut))
    }

    private static func index(_ text: String, atUTF8Offset offset: Int) -> String.Index {
        text.utf8.index(text.utf8.startIndex, offsetBy: offset)
    }
}
