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

    /// Reference box, not a `@State` struct: the accumulator records scan
    /// bookkeeping on every publish, and a `@State` write would schedule a
    /// second body pass per publish just to note the watermark. A freeze
    /// landing one render late is invisible — chunks + tail always
    /// reassemble to the same text.
    @State private var frozen = Box()

    private final class Box {
        var accumulator = ChunkedTextAccumulator()
    }

    var body: some View {
        let accumulator = frozen.accumulator
        let tail = accumulator.liveTail(of: text)
        // Matches the interline gap a single `Text` would draw at the chat's
        // `lineSpacing`, so the chunk seams are invisible.
        VStack(alignment: .leading, spacing: chatLineSpacing) {
            ForEach(accumulator.chunks.indices, id: \.self) { index in
                Text(accumulator.chunks[index])
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            Text(displayTail(tail, isFirstChunk: accumulator.chunks.isEmpty))
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .onChange(of: text, initial: true) { _, newValue in
            frozen.accumulator.freezeCompletedParagraphs(of: newValue)
        }
    }

    private func displayTail(_ tail: Substring, isFirstChunk: Bool) -> String {
        isFirstChunk
            ? tail.chatDisplayTrimmed
            : tail.trimmingTrailingWhitespace()
    }
}

extension StringProtocol {
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
nonisolated struct ChunkedTextAccumulator {
    /// Frozen paragraph chunks — append-only, never mutated afterward, so a
    /// chunk `Text` keyed by its index is stable for the life of the stream.
    private(set) var chunks: [String] = []
    /// UTF-8 length of the consumed source prefix (frozen chunks plus each
    /// cut newline). Always lands just past a `\n` byte, so re-deriving
    /// the index is grapheme-safe.
    private(set) var consumedUTF8: Int = 0
    private var lastSeenUTF8: Int = 0
    /// UTF-8 offset just past the last newline byte seen so far (0 = none).
    /// The newline search covers only bytes appended since the previous
    /// pass, so a publish costs O(delta) even when a giant single paragraph
    /// keeps the tail from ever freezing.
    private var newlineEndUTF8: Int = 0

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
    /// scan the freshly appended bytes for the newest newline, then move
    /// completed paragraphs from the tail into frozen chunks, cutting there.
    /// The first chunk sheds leading whitespace so the frozen render matches
    /// `chatDisplayTrimmed`.
    mutating func freezeCompletedParagraphs(of text: String) {
        let total = text.utf8.count
        if total < lastSeenUTF8 { self = .init() }

        if lastSeenUTF8 < total {
            let utf8 = text.utf8
            var offset = lastSeenUTF8
            for byte in utf8[Self.index(text, atUTF8Offset: lastSeenUTF8)...] {
                offset += 1
                if byte == 0x0A { newlineEndUTF8 = offset }
            }
        }
        lastSeenUTF8 = total

        guard total - consumedUTF8 > Self.freezeThresholdUTF8,
            newlineEndUTF8 > consumedUTF8
        else { return }

        let tailStart = Self.index(text, atUTF8Offset: consumedUTF8)
        let cutEnd = Self.index(text, atUTF8Offset: newlineEndUTF8)
        // `index(before:)` steps back one grapheme, so a CRLF pair drops
        // from the chunk whole.
        var chunk = text[tailStart..<text.index(before: cutEnd)]
        if chunks.isEmpty {
            chunk = chunk.drop(while: { $0.isWhitespace || $0.isNewline })
        }
        if !chunk.isEmpty {
            chunks.append(String(chunk))
        }
        consumedUTF8 = newlineEndUTF8
    }

    private static func index(_ text: String, atUTF8Offset offset: Int) -> String.Index {
        text.utf8.index(text.utf8.startIndex, offsetBy: offset)
    }
}
