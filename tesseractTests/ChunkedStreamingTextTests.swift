import Foundation
import Testing

@testable import Tesseract_Agent

/// The chunking rules behind `ChunkedStreamingText`: paragraphs freeze out
/// of the live tail at newline cuts, the frozen prefix is append-only, and
/// any shrink of the source text (elision re-slice, safeguard truncate)
/// resets the state.
struct ChunkedTextAccumulatorTests {

    private func makeText(paragraphs: Int, paragraphLength: Int = 600) -> String {
        (0..<paragraphs).map { index in
            String(repeating: "p\(index) word ", count: paragraphLength / 8)
        }.joined(separator: "\n")
    }

    @Test func shortTextNeverFreezes() {
        var acc = ChunkedTextAccumulator()
        let text = "a short thought\nwith two lines"
        acc.freezeCompletedParagraphs(of: text)
        #expect(acc.chunks.isEmpty)
        #expect(String(acc.liveTail(of: text)) == text)
    }

    @Test func longTextFreezesAtLastNewlineAndTailStaysSmall() {
        var acc = ChunkedTextAccumulator()
        let text = makeText(paragraphs: 12)
        acc.freezeCompletedParagraphs(of: text)

        #expect(!acc.chunks.isEmpty)
        let tail = acc.liveTail(of: text)
        // The cut is at the LAST newline: the tail is the final line only.
        #expect(!tail.contains("\n"))
        #expect(text.hasSuffix(tail))
        // Frozen chunks + the cut newline + tail reassemble the source
        // (modulo the first chunk's leading-edge trim, absent here).
        let reassembled = acc.chunks.joined(separator: "\n") + "\n" + tail
        #expect(reassembled == text)
    }

    @Test func frozenChunksAreAppendOnlyAsTextGrows() {
        var acc = ChunkedTextAccumulator()
        var text = makeText(paragraphs: 12)
        acc.freezeCompletedParagraphs(of: text)
        let before = acc.chunks

        text += "\n" + makeText(paragraphs: 12)
        acc.freezeCompletedParagraphs(of: text)

        #expect(acc.chunks.count > before.count)
        #expect(Array(acc.chunks.prefix(before.count)) == before)
    }

    @Test func growthBelowThresholdLeavesStateUntouched() {
        var acc = ChunkedTextAccumulator()
        var text = makeText(paragraphs: 12)
        acc.freezeCompletedParagraphs(of: text)
        let frozen = acc

        text += " more words on the same line"
        acc.freezeCompletedParagraphs(of: text)
        #expect(acc.chunks == frozen.chunks)
        #expect(acc.consumedUTF8 == frozen.consumedUTF8)
        #expect(text.hasSuffix(acc.liveTail(of: text)))
    }

    @Test func shrinkResetsAndRechunksFromScratch() {
        var acc = ChunkedTextAccumulator()
        let long = makeText(paragraphs: 30)
        acc.freezeCompletedParagraphs(of: long)
        #expect(!acc.chunks.isEmpty)

        // The safeguard truncate / elision re-slice always shrinks.
        let replacement = "safe prefix only"
        acc.freezeCompletedParagraphs(of: replacement)
        #expect(acc.chunks.isEmpty)
        #expect(String(acc.liveTail(of: replacement)) == replacement)
    }

    @Test func liveTailIsWholeTextWhenStaleStateExceedsIt() {
        var acc = ChunkedTextAccumulator()
        let long = makeText(paragraphs: 30)
        acc.freezeCompletedParagraphs(of: long)

        // Between the shrink and the next freeze pass, rendering must not
        // slice with a stale offset.
        let shorter = "tiny"
        #expect(String(acc.liveTail(of: shorter)) == shorter)
    }

    @Test func firstChunkShedsLeadingWhitespace() {
        var acc = ChunkedTextAccumulator()
        let text = "\n\n  " + makeText(paragraphs: 12)
        acc.freezeCompletedParagraphs(of: text)
        #expect(acc.chunks.first?.first?.isWhitespace == false)
        // Offsets still track the ORIGINAL text, so the tail is unaffected.
        #expect(text.hasSuffix(acc.liveTail(of: text)))
    }

    @Test func multiByteContentCutsOnCharacterBoundaries() {
        var acc = ChunkedTextAccumulator()
        var text = String(repeating: "héllo wörld 🌍✨\n", count: 400)
        acc.freezeCompletedParagraphs(of: text)
        #expect(!acc.chunks.isEmpty)
        #expect(text.hasSuffix(acc.liveTail(of: text)))

        text += "más🎈"
        acc.freezeCompletedParagraphs(of: text)
        #expect(acc.liveTail(of: text).hasSuffix("más🎈"))
    }

    @Test func singleGiantLineNeverFreezesButStaysRenderable() {
        var acc = ChunkedTextAccumulator()
        let text = String(repeating: "x", count: 20_000)
        acc.freezeCompletedParagraphs(of: text)
        #expect(acc.chunks.isEmpty)
        #expect(String(acc.liveTail(of: text)) == text)
    }
}
