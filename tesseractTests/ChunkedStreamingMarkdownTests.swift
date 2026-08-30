import Foundation
import Testing

@testable import Tesseract_Agent

/// The chunking rules behind `ChunkedStreamingMarkdown`: block runs freeze
/// out of the live tail only at blank-line boundaries outside fenced code
/// blocks and never before an indented continuation, the frozen prefix is
/// append-only, and any shrink of the source text resets the state.
struct ChunkedMarkdownAccumulatorTests {

    private func makeMarkdown(paragraphs: Int) -> String {
        (0..<paragraphs).map { index in
            String(repeating: "p\(index) word ", count: 60)
        }.joined(separator: "\n\n")
    }

    /// Frozen chunks plus the live tail must always reassemble the source
    /// text exactly — chunking may never drop or duplicate a byte.
    private func expectReassembles(_ acc: ChunkedMarkdownAccumulator, _ text: String) {
        #expect(acc.chunks.map(\.text).joined() + acc.liveTail(of: text) == text)
    }

    @Test func shortTextNeverFreezes() {
        var acc = ChunkedMarkdownAccumulator()
        let text = "a short answer\n\nwith two paragraphs"
        acc.freezeCompletedBlocks(of: text)
        #expect(acc.chunks.isEmpty)
        #expect(String(acc.liveTail(of: text)) == text)
    }

    @Test func longTextFreezesAtBlankLineBoundary() {
        var acc = ChunkedMarkdownAccumulator()
        let text = makeMarkdown(paragraphs: 8)
        acc.freezeCompletedBlocks(of: text)

        #expect(!acc.chunks.isEmpty)
        // The cut lands at the start of a paragraph: the frozen prefix ends
        // with the blank-line separator, the tail starts with content.
        #expect(acc.chunks.last?.text.hasSuffix("\n\n") == true)
        #expect(acc.chunks.first?.topSpacing == 0)
        #expect(acc.liveTail(of: text).first?.isWhitespace == false)
        expectReassembles(acc, text)
    }

    @Test func frozenChunksAreAppendOnlyAsTextGrows() {
        var acc = ChunkedMarkdownAccumulator()
        var text = makeMarkdown(paragraphs: 8)
        acc.freezeCompletedBlocks(of: text)
        let before = acc.chunks

        text += "\n\n" + makeMarkdown(paragraphs: 8)
        acc.freezeCompletedBlocks(of: text)

        #expect(acc.chunks.count > before.count)
        #expect(Array(acc.chunks.prefix(before.count)) == before)
        expectReassembles(acc, text)
    }

    @Test func growthBelowThresholdLeavesStateUntouched() {
        var acc = ChunkedMarkdownAccumulator()
        var text = makeMarkdown(paragraphs: 8)
        acc.freezeCompletedBlocks(of: text)
        let frozen = acc

        text += " more words in the same paragraph"
        acc.freezeCompletedBlocks(of: text)
        #expect(acc.chunks == frozen.chunks)
        #expect(acc.consumedUTF8 == frozen.consumedUTF8)
        expectReassembles(acc, text)
    }

    @Test func blankLinesInsideFencedCodeNeverCut() {
        var acc = ChunkedMarkdownAccumulator()
        let fenced =
            "```swift\n"
            + String(
                repeating: "let x = 1  // filler line\n\nlet y = 2  // more\n",
                count: 60)
        acc.freezeCompletedBlocks(of: fenced)
        // Well past the threshold, but every blank line is inside the open
        // fence — nothing may freeze.
        #expect(acc.chunks.isEmpty)
        #expect(String(acc.liveTail(of: fenced)) == fenced)
    }

    @Test func cutsResumeAfterFenceCloses() {
        var acc = ChunkedMarkdownAccumulator()
        var text =
            "```swift\n"
            + String(repeating: "let x = 1\n\nlet y = 2\n", count: 150)
            + "```\n\nafter the fence"
        acc.freezeCompletedBlocks(of: text)
        #expect(acc.chunks.count == 1)
        #expect(String(acc.liveTail(of: text)) == "after the fence")

        text += " keeps growing"
        acc.freezeCompletedBlocks(of: text)
        expectReassembles(acc, text)
    }

    @Test func tildeFenceClosesOnlyOnTilde() {
        var acc = ChunkedMarkdownAccumulator()
        // A ``` line inside a ~~~ fence is fence *content*, not a close.
        let text =
            "~~~\n```\n\n"
            + String(repeating: "code line\n\n", count: 200)
        acc.freezeCompletedBlocks(of: text)
        #expect(acc.chunks.isEmpty)
    }

    @Test func backtickRunWithTrailingBacktickIsNotAFence() {
        var acc = ChunkedMarkdownAccumulator()
        // "```x```" is inline code in a paragraph, not an open fence —
        // blank lines after it must still cut.
        let text =
            "```inline```\n\n"
            + makeMarkdown(paragraphs: 8)
        acc.freezeCompletedBlocks(of: text)
        #expect(!acc.chunks.isEmpty)
        expectReassembles(acc, text)
    }

    @Test func indentedContinuationIsNotACutPoint() {
        var acc = ChunkedMarkdownAccumulator()
        // An indented code block spans blank lines; the cut may not land on
        // the indented continuation, only on the later unindented paragraph.
        let indented = String(repeating: "    indented code\n\n", count: 150)
        let text = "intro paragraph\n\n" + indented + "closing paragraph"
        acc.freezeCompletedBlocks(of: text)
        #expect(String(acc.liveTail(of: text)) == "closing paragraph")
        expectReassembles(acc, text)
    }

    @Test func shrinkResetsAndRechunksFromScratch() {
        var acc = ChunkedMarkdownAccumulator()
        let long = makeMarkdown(paragraphs: 20)
        acc.freezeCompletedBlocks(of: long)
        #expect(!acc.chunks.isEmpty)

        let replacement = "safe prefix only"
        acc.freezeCompletedBlocks(of: replacement)
        #expect(acc.chunks.isEmpty)
        #expect(String(acc.liveTail(of: replacement)) == replacement)
    }

    @Test func liveTailIsWholeTextWhenStaleStateExceedsIt() {
        var acc = ChunkedMarkdownAccumulator()
        let long = makeMarkdown(paragraphs: 20)
        acc.freezeCompletedBlocks(of: long)

        let shorter = "tiny"
        #expect(String(acc.liveTail(of: shorter)) == shorter)
    }

    @Test func multiByteContentCutsOnCharacterBoundaries() {
        var acc = ChunkedMarkdownAccumulator()
        var text = String(repeating: "héllo wörld 🌍✨\n\n", count: 200)
        acc.freezeCompletedBlocks(of: text)
        #expect(!acc.chunks.isEmpty)
        expectReassembles(acc, text)

        text += "más🎈"
        acc.freezeCompletedBlocks(of: text)
        #expect(acc.liveTail(of: text).hasSuffix("más🎈"))
    }

    @Test func singleGiantParagraphNeverFreezesButStaysRenderable() {
        var acc = ChunkedMarkdownAccumulator()
        let text = String(repeating: "x", count: 20_000)
        acc.freezeCompletedBlocks(of: text)
        #expect(acc.chunks.isEmpty)
        #expect(String(acc.liveTail(of: text)) == text)
    }

    // MARK: Seam spacing

    @Test func frozenChunkRecordsSeamSpacingForItsLeadBlock() {
        var acc = ChunkedMarkdownAccumulator()
        var text = makeMarkdown(paragraphs: 8) + "\n\n# Title"
        acc.freezeCompletedBlocks(of: text)
        #expect(acc.chunks.count == 1)
        #expect(acc.chunks.first?.topSpacing == 0)
        #expect(String(acc.liveTail(of: text)) == "# Title")

        text += "\n\n" + makeMarkdown(paragraphs: 8)
        acc.freezeCompletedBlocks(of: text)
        #expect(acc.chunks.count == 2)
        #expect(acc.chunks[1].text.hasPrefix("# Title"))
        #expect(acc.chunks[1].topSpacing == ChatMarkdownBlockSpacing.beforeHeading)
        expectReassembles(acc, text)
    }

    @Test func seamSpacingMatchesBlockTopSpacing() {
        let heading = ChatMarkdownBlockSpacing.beforeHeading
        let base = ChatMarkdownBlockSpacing.betweenBlocks

        #expect(ChunkedMarkdownAccumulator.seamSpacing(before: "# Title") == heading)
        #expect(ChunkedMarkdownAccumulator.seamSpacing(before: "###### h6\nbody") == heading)
        #expect(ChunkedMarkdownAccumulator.seamSpacing(before: "---") == heading)
        #expect(ChunkedMarkdownAccumulator.seamSpacing(before: "- - -") == heading)
        #expect(ChunkedMarkdownAccumulator.seamSpacing(before: "***") == heading)

        #expect(ChunkedMarkdownAccumulator.seamSpacing(before: "plain paragraph") == base)
        #expect(ChunkedMarkdownAccumulator.seamSpacing(before: "####### seven") == base)
        #expect(ChunkedMarkdownAccumulator.seamSpacing(before: "#tag no space") == base)
        #expect(ChunkedMarkdownAccumulator.seamSpacing(before: "- list item") == base)
        #expect(ChunkedMarkdownAccumulator.seamSpacing(before: "    # indented code") == base)
        #expect(ChunkedMarkdownAccumulator.seamSpacing(before: "***bold*** text") == base)
    }
}
