//
//  CaptionLayout.swift
//  tesseract
//
//  How the Speech Overlay lays out what it reads: words fill lines up to a
//  width, a paragraph always starts a line, and the lines of a reading form
//  one continuous feed that grows as its passages arrive (ADR-0077). Each
//  passage is measured once, when it arrives; nothing is measured per frame
//  or per word. Pure.
//

import CoreGraphics

nonisolated enum CaptionLayout {
    /// The words on one line, as indices into the words measured.
    typealias Line = Range<Int>

    /// Word indices per line: each line takes words while their widths (plus
    /// a space between) fit `width`, and ends after a word in `breaksAfter`
    /// (a paragraph's last). A word wider than the line gets a line to
    /// itself.
    static func lines(
        wordWidths: [CGFloat], spaceWidth: CGFloat, width: CGFloat, breaksAfter: Set<Int> = []
    ) -> [Line] {
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
            if breaksAfter.contains(index), index + 1 < wordWidths.count {
                lines.append(start..<(index + 1))
                start = index + 1
                used = 0
            }
        }
        lines.append(start..<wordWidths.count)
        return lines
    }
}

/// The lines of one reading, in order, as the Speech Overlay scrolls through
/// them. A line keeps its number for the whole reading, so the overlay moves
/// lines, never replaces them; lines of passages already read drop out.
nonisolated struct CaptionFeed: Equatable {
    struct Line: Equatable, Identifiable {
        /// The line's number in the reading, from 0: also where it sits.
        let id: Int
        let passage: Int
        /// The utterance's index of the line's first word.
        let firstWord: Int
        let words: [String]

        var wordRange: Range<Int> { firstWord..<(firstWord + words.count) }
    }

    private(set) var lines: [Line] = []
    /// The last passage laid out.
    private(set) var lastPassage = -1
    private var nextID = 0

    /// Adds a passage's lines, from `ReadAlongPassage` words measured once.
    /// A passage laid out already, or out of order, is ignored.
    mutating func append(
        _ passage: ReadAlongPassage, wordWidths: [CGFloat], spaceWidth: CGFloat, width: CGFloat
    ) {
        guard passage.index > lastPassage, wordWidths.count == passage.words.count else { return }
        lastPassage = passage.index
        for line in CaptionLayout.lines(
            wordWidths: wordWidths, spaceWidth: spaceWidth, width: width,
            breaksAfter: passage.paragraphEnds)
        {
            lines.append(
                Line(
                    id: nextID, passage: passage.index,
                    firstWord: passage.firstWord + line.lowerBound,
                    words: Array(passage.words[line])))
            nextID += 1
        }
    }

    /// Lets go of the lines of passages before `index`.
    mutating func drop(passagesBefore index: Int) {
        lines.removeAll { $0.passage < index }
    }

    /// The position in `lines` of the line holding `word`: the first line
    /// before the reading's first word, the last past its end.
    func line(holding word: Int) -> Int? {
        guard !lines.isEmpty else { return nil }
        var low = 0
        var high = lines.count - 1
        while low < high {
            let mid = (low + high + 1) / 2
            if lines[mid].firstWord <= word { low = mid } else { high = mid - 1 }
        }
        return low
    }
}
