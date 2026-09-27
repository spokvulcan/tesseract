//
//  ReaderText.swift
//  tesseract
//
//  Text geometry for the Reader, on the document's UTF-16 offsets (the text
//  view's own). Everything here looks at a window around an offset, never
//  the whole document, so a whole book costs the same as a page. Pure.
//

import Foundation
import NaturalLanguage

nonisolated enum ReaderText {
    /// A window wide enough to see a sentence boundary on either side.
    static let sentenceWindow = 4_000

    /// The ranges of the `count` whitespace-separated words starting at
    /// `offset` — the engine's and the Word Timeline's definition of a word,
    /// so word `i` of a spoken segment is range `i` here.
    static func words(in text: NSString, from offset: Int, count: Int) -> [NSRange] {
        var ranges: [NSRange] = []
        ranges.reserveCapacity(count)
        var index = offset
        let length = text.length
        while ranges.count < count, index < length {
            while index < length, isSeparator(text.character(at: index)) { index += 1 }
            guard index < length else { break }
            let start = index
            while index < length, !isSeparator(text.character(at: index)) { index += 1 }
            ranges.append(NSRange(location: start, length: index - start))
        }
        return ranges
    }

    /// The first word's range at or after `offset`, if any.
    static func firstWord(in text: NSString, from offset: Int) -> NSRange? {
        words(in: text, from: offset, count: 1).first
    }

    /// Whether `text` holds anything to read in `range`.
    static func hasWords(_ text: NSString, in range: NSRange) -> Bool {
        let end = min(range.upperBound, text.length)
        var index = range.location
        while index < end {
            if !isSeparator(text.character(at: index)) { return true }
            index += 1
        }
        return false
    }

    /// Sentence ranges within `range`.
    static func sentences(in text: NSString, range: NSRange) -> [NSRange] {
        let clamped = NSIntersectionRange(range, NSRange(location: 0, length: text.length))
        guard clamped.length > 0 else { return [] }
        let substring = text.substring(with: clamped)
        let tokenizer = NLTokenizer(unit: .sentence)
        tokenizer.string = substring
        var ranges: [NSRange] = []
        tokenizer.enumerateTokens(in: substring.startIndex..<substring.endIndex) { tokenRange, _ in
            let local = NSRange(tokenRange, in: substring)
            ranges.append(
                NSRange(location: clamped.location + local.location, length: local.length))
            return true
        }
        return ranges
    }

    /// The start of the sentence holding `offset` (so reading from a
    /// clicked word begins at its sentence), or `offset` at a boundary.
    static func sentenceStart(containing offset: Int, in text: NSString) -> Int {
        let target = min(max(offset, 0), max(text.length - 1, 0))
        let window = around(target, in: text)
        let sentences = sentences(in: text, range: window)
        let holding = sentences.last { $0.location <= target }
        // A boundary seen only because the window cut a sentence is not a start.
        if let holding, holding.location > window.location || window.location == 0 {
            return firstReadable(in: text, from: holding.location)
        }
        return firstReadable(in: text, from: target)
    }

    /// The start of the sentence after the one holding `offset`, or nil at the end.
    static func nextSentenceStart(after offset: Int, in text: NSString) -> Int? {
        // From the current sentence's start, so a window opening mid-sentence
        // doesn't pass the rest of it off as the next one.
        let current = sentenceStart(containing: offset, in: text)
        let window = NSRange(location: current, length: min(sentenceWindow, text.length - current))
        let sentences = sentences(in: text, range: window)
        guard let next = sentences.first(where: { $0.location > current }),
            hasWords(
                text, in: NSRange(location: next.location, length: text.length - next.location))
        else { return nil }
        return firstReadable(in: text, from: next.location)
    }

    /// The start of the sentence before the one holding `offset`, or the
    /// start of the text.
    static func previousSentenceStart(before offset: Int, in text: NSString) -> Int {
        let current = sentenceStart(containing: offset, in: text)
        // Back over the space before it: the tokenizer can count a paragraph
        // break as the start of the next sentence.
        var index = current - 1
        while index > 0, isSeparator(text.character(at: index)) { index -= 1 }
        guard index >= 0 else { return 0 }
        return sentenceStart(containing: index, in: text)
    }

    /// The first non-separator at or after `offset` (clamped to the text).
    static func firstReadable(in text: NSString, from offset: Int) -> Int {
        var index = max(offset, 0)
        while index < text.length, isSeparator(text.character(at: index)) { index += 1 }
        return min(index, text.length)
    }

    private static func around(_ offset: Int, in text: NSString) -> NSRange {
        let lower = max(0, offset - sentenceWindow)
        let upper = min(text.length, offset + sentenceWindow / 4)
        return NSRange(location: lower, length: upper - lower)
    }

    /// Whitespace and newlines: what splits words for the engine, the Word
    /// Timeline and here.
    static func isSeparator(_ unit: unichar) -> Bool {
        guard let scalar = Unicode.Scalar(unit) else { return false }
        return CharacterSet.whitespacesAndNewlines.contains(scalar)
    }
}
