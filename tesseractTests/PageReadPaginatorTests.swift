import Foundation
import Testing

@testable import Tesseract_Agent

/// Lower browser test seam: **Page Read** pagination is a pure transform.
struct PageReadPaginatorTests {

    @Test
    func shortContentReturnsWholeThingWithNoCursor() {
        let content = "Just a short page."
        let chunk = PageReadPaginator.paginate(content, cursor: 0, maxChars: 1000)
        #expect(chunk.text == content)
        #expect(chunk.nextCursor == nil)
        #expect(chunk.totalChars == content.count)
    }

    @Test
    func longContentPaginatesAndAdvancesCursor() {
        let content = String(repeating: "abcde ", count: 1000)  // 6000 chars
        let first = PageReadPaginator.paginate(content, cursor: 0, maxChars: 2000)
        #expect(first.text.count <= 2000)
        #expect(first.nextCursor != nil)
        #expect(first.totalChars == content.count)
    }

    @Test
    func cursorPastEndYieldsEmptyTerminalChunk() {
        let content = "Some content here."
        let chunk = PageReadPaginator.paginate(content, cursor: 9999, maxChars: 100)
        #expect(chunk.text.isEmpty)
        #expect(chunk.nextCursor == nil)
    }

    @Test
    func breaksOnParagraphBoundaryWhenAvailable() {
        // First paragraph is well under the cap; the break should land at the
        // blank line, not mid-second-paragraph.
        let para1 = "First paragraph."
        let content = para1 + "\n\n" + String(repeating: "x", count: 5000)
        let chunk = PageReadPaginator.paginate(content, cursor: 0, maxChars: 200)
        #expect(chunk.text == para1)
        #expect(chunk.nextCursor == para1.count)
    }

    @Test
    func concatenatingAllChunksReconstructsContentExactly() {
        let content =
            (1...40)
            .map { "Paragraph number \($0) with a little bit of filler text.\n\n" }
            .joined()

        var reassembled = ""
        var cursor: Int? = 0
        var guardCounter = 0
        while let current = cursor, guardCounter < 1000 {
            let chunk = PageReadPaginator.paginate(content, cursor: current, maxChars: 300)
            reassembled += chunk.text
            cursor = chunk.nextCursor
            guardCounter += 1
        }
        #expect(reassembled == content)
        #expect(guardCounter < 1000)  // terminated via nil cursor, not the guard
    }
}
