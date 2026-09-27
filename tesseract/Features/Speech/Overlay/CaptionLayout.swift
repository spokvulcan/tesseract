//
//  CaptionLayout.swift
//  tesseract
//
//  How the Speech Overlay pages a segment: words fill lines up to a width,
//  and the overlay shows the two lines that hold the word being heard.
//  Measured once per segment and size, never per frame. Pure.
//

import CoreGraphics

nonisolated enum CaptionLayout {
    /// The words on one line, as indices into the segment's words.
    typealias Line = Range<Int>

    /// Word indices per line: each line takes words while their widths (plus
    /// a space between) fit `width`. A word wider than the line gets a line
    /// to itself.
    static func lines(wordWidths: [CGFloat], spaceWidth: CGFloat, width: CGFloat) -> [Line] {
        guard !wordWidths.isEmpty else { return [] }
        var lines: [Line] = []
        var start = 0
        var used: CGFloat = 0
        for (index, wordWidth) in wordWidths.enumerated() {
            let needed = used == 0 ? wordWidth : used + spaceWidth + wordWidth
            if used > 0, needed > width {
                lines.append(start..<index)
                start = index
                used = wordWidth
            } else {
                used = needed
            }
        }
        lines.append(start..<wordWidths.count)
        return lines
    }

    /// The lines to show for `word`: the page of `linesPerPage` lines that
    /// holds it. Before the first word, the first page.
    static func page(for word: Int, in lines: [Line], linesPerPage: Int = 2) -> ArraySlice<Line> {
        guard !lines.isEmpty else { return [] }
        let line =
            lines.firstIndex { $0.contains(max(word, 0)) } ?? (word < 0 ? 0 : lines.count - 1)
        let first = line / linesPerPage * linesPerPage
        return lines[first..<min(first + linesPerPage, lines.count)]
    }
}
