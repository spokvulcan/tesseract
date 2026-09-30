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
import TesseractSpeech

nonisolated enum ReaderText {
    /// A window wide enough to see a sentence boundary on either side.
    static let sentenceWindow = 4_000

    /// How much text a walk over Characters reads at a time.
    static let characterWindow = 1_024

    /// The ranges of the `count` words starting at `offset`, split as the
    /// engine splits them: by Character, with `Character.separatesWords`. So
    /// word `i` of a spoken segment is range `i` here, and a mark joined to
    /// a space is part of the space, not a word.
    static func words(in text: NSString, from offset: Int, count: Int) -> [NSRange] {
        guard count > 0 else { return [] }
        var ranges: [NSRange] = []
        ranges.reserveCapacity(count)
        var wordStart: Int?
        var wordEnd = offset
        forEachCharacter(in: text, from: offset) { character, range in
            guard character.separatesWords else {
                if wordStart == nil { wordStart = range.location }
                wordEnd = range.upperBound
                return true
            }
            if let start = wordStart {
                ranges.append(NSRange(location: start, length: wordEnd - start))
                wordStart = nil
            }
            return ranges.count < count
        }
        if let start = wordStart, ranges.count < count {
            ranges.append(NSRange(location: start, length: wordEnd - start))
        }
        return ranges
    }

    /// The first word's range at or after `offset`, if any.
    static func firstWord(in text: NSString, from offset: Int) -> NSRange? {
        words(in: text, from: offset, count: 1).first
    }

    /// Whether `text` holds anything to read in `range`.
    static func hasWords(_ text: NSString, in range: NSRange) -> Bool {
        var found = false
        forEachCharacter(in: text, from: range.location) { character, characterRange in
            guard characterRange.location < range.upperBound else { return false }
            found = !character.separatesWords
            return !found
        }
        return found
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
        guard current > 0 else { return 0 }
        // Back over the space before it: the tokenizer can count a paragraph
        // break as the start of the next sentence.
        let before = lastReadable(before: current, in: text) ?? 0
        return sentenceStart(containing: before, in: text)
    }

    /// The first readable Character at or after `offset` (clamped to the
    /// text).
    static func firstReadable(in text: NSString, from offset: Int) -> Int {
        var readable = text.length
        forEachCharacter(in: text, from: max(offset, 0)) { character, range in
            guard character.separatesWords else {
                readable = range.location
                return false
            }
            return true
        }
        return readable
    }

    /// The start of the last readable Character before `offset`, or nil
    /// when only separators come before it.
    private static func lastReadable(before offset: Int, in text: NSString) -> Int? {
        var lower = min(offset, text.length)
        while lower > 0 {
            lower = max(0, lower - sentenceWindow)
            var last: Int?
            forEachCharacter(in: text, from: lower) { character, range in
                guard range.location < offset else { return false }
                if !character.separatesWords { last = range.location }
                return true
            }
            if let last { return last }
        }
        return nil
    }

    /// Visit `text`'s Characters from `offset` on, each with its UTF-16
    /// range, until `body` returns false or the text ends. Characters are
    /// Swift's, as the engine splits them, and the walk reads a window at a
    /// time, so it costs what it reads.
    private static func forEachCharacter(
        in text: NSString, from offset: Int, _ body: (Character, NSRange) -> Bool
    ) {
        let length = text.length
        var location = max(offset, 0)
        var window = characterWindow
        while location < length {
            let end = min(length, location + window)
            let slice = text.substring(with: NSRange(location: location, length: end - location))
            var characters = slice.makeIterator()
            var current = characters.next()
            var visited = 0
            while let character = current {
                let next = characters.next()
                // The window may cut its last Character short: read it
                // again, whole, at the start of the next window.
                if next == nil, end < length { break }
                let size = character.utf16.count
                guard body(character, NSRange(location: location, length: size)) else { return }
                location += size
                visited += 1
                current = next
            }
            guard end < length else { return }
            // A single Character wider than the window.
            if visited == 0 { window *= 2 }
        }
    }

    private static func around(_ offset: Int, in text: NSString) -> NSRange {
        let lower = max(0, offset - sentenceWindow)
        let upper = min(text.length, offset + sentenceWindow / 4)
        return NSRange(location: lower, length: upper - lower)
    }
}
