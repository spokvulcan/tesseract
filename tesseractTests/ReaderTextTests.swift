//
//  ReaderTextTests.swift
//  tesseractTests
//
//  The Reader's text geometry and its store: pure functions on UTF-16
//  offsets, and the text and Bookmark on disk.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct ReaderTextTests {

    private let text: NSString =
        "The first sentence is here. A second one follows!\n\nThen a new paragraph starts. And ends."

    private func substring(_ range: NSRange) -> String { text.substring(with: range) }

    @Test func wordsSplitOnWhitespaceAndNewlinesLikeTheEngine() {
        let words = ReaderText.words(in: text, from: 28, count: 6)
        #expect(words.map(substring) == ["A", "second", "one", "follows!", "Then", "a"])
    }

    @Test func wordsStopAtTheEndOfTheText() {
        let words = ReaderText.words(in: text, from: 83, count: 10)
        #expect(words.map(substring) == ["ends."])
    }

    @Test func hasWordsIgnoresWhitespace() {
        #expect(!ReaderText.hasWords(" \n\t " as NSString, in: NSRange(location: 0, length: 4)))
        #expect(ReaderText.hasWords(text, in: NSRange(location: 49, length: 10)))
    }

    @Test func aSentenceStartsWhereItsFirstWordDoes() {
        // An offset inside "second" belongs to the sentence starting at "A".
        #expect(ReaderText.sentenceStart(containing: 32, in: text) == 28)
        #expect(ReaderText.sentenceStart(containing: 0, in: text) == 0)
        // A paragraph break leads to the next sentence's first word.
        let paragraph = text.range(of: "Then").location
        #expect(ReaderText.sentenceStart(containing: paragraph + 2, in: text) == paragraph)
    }

    @Test func nextAndPreviousSentences() {
        let then = text.range(of: "Then").location
        let and = text.range(of: "And").location
        #expect(ReaderText.nextSentenceStart(after: 0, in: text) == 28)
        // From a space inside a sentence, the next is still the next sentence.
        #expect(ReaderText.nextSentenceStart(after: 3, in: text) == 28)
        #expect(ReaderText.nextSentenceStart(after: 30, in: text) == then)
        #expect(ReaderText.nextSentenceStart(after: and, in: text) == nil)
        #expect(ReaderText.previousSentenceStart(before: then + 3, in: text) == 28)
        #expect(ReaderText.previousSentenceStart(before: 5, in: text) == 0)
    }

    @Test func firstReadableSkipsLeadingWhitespace() {
        #expect(ReaderText.firstReadable(in: "  \n word" as NSString, from: 0) == 4)
        #expect(ReaderText.firstReadable(in: "word  " as NSString, from: 4) == 6)
    }

    // MARK: - A mark after a space (#581)

    /// A combining mark, a zero-width joiner or a variation selector right
    /// after a space joins the space's Character, so the engine and the Word
    /// Timeline count no word there. The Reader finds the same words.
    @Test(arguments: [
        "one \u{301} two three", "one \u{200D} two three", "one\u{00A0}\u{FE0F} two three",
        "one\u{3000}\u{301}two three",
    ])
    func aMarkAfterASpaceIsPartOfTheSpace(_ text: String) {
        let ns = text as NSString
        let words = ReaderText.words(in: ns, from: 0, count: 10).map(ns.substring(with:))
        #expect(words == text.splitIntoWords().map(String.init))
        #expect(words == ["one", "two", "three"])
    }

    @Test func aMarkAfterASpaceIsNothingToRead() {
        let text = "one. \u{301}" as NSString
        #expect(!ReaderText.hasWords(text, in: NSRange(location: 4, length: 2)))
        let next = "one \u{301} two" as NSString
        #expect(ReaderText.firstReadable(in: next, from: 3) == next.range(of: "two").location)
    }

    /// Going back a sentence steps over the space and its mark, back into the
    /// sentence before.
    @Test func thePreviousSentenceIsFoundPastAMarkAfterASpace() {
        let text = "First one here. \u{301}Second one here." as NSString
        let second = text.range(of: "Second").location
        #expect(ReaderText.previousSentenceStart(before: second + 2, in: text) == 0)
    }

    /// Deep in a book-length text the answer comes from a window around the
    /// offset, and it is the same as near the start.
    @Test func sentencesDeepInABookAreFoundLocally() {
        let sentence = "The keeper climbed the ninety-nine steps again. "
        let book = String(repeating: sentence, count: 20_000) as NSString
        let length = (sentence as NSString).length
        let deep = length * 15_000 + 10
        #expect(ReaderText.sentenceStart(containing: deep, in: book) == length * 15_000)
        #expect(ReaderText.nextSentenceStart(after: deep, in: book) == length * 15_001)
    }
}

struct ReaderDocumentStoreTests {

    @Test func textAndBookmarkSurviveARelaunch() {
        let directory = makeTempDir("reader-store")
        let store = ReaderDocumentStore(directory: directory)
        store.save(text: "Hello there. Read me.")
        store.save(bookmark: 13, length: 21)

        let loaded = ReaderDocumentStore(directory: directory).load()
        #expect(loaded.text == "Hello there. Read me.")
        #expect(loaded.bookmark == 13)
    }

    @Test func aBookmarkForOtherTextIsDropped() {
        let directory = makeTempDir("reader-store")
        let store = ReaderDocumentStore(directory: directory)
        store.save(text: "Short.")
        store.save(bookmark: 40, length: 90)
        #expect(store.load().bookmark == 0)
    }

    @Test func nothingSavedLoadsEmpty() {
        let loaded = ReaderDocumentStore(directory: makeTempDir("reader-store")).load()
        #expect(loaded.text.isEmpty)
        #expect(loaded.bookmark == 0)
    }
}

struct CaptionLayoutTests {

    @Test func wordsFillLinesUpToTheWidth() {
        // 30 + 5 + 30 = 65 fits 70; the third word does not.
        let lines = CaptionLayout.lines(wordWidths: [30, 30, 30, 30], spaceWidth: 5, width: 70)
        #expect(lines == [0..<2, 2..<4])
    }

    @Test func aWordWiderThanTheLineGetsItsOwn() {
        let lines = CaptionLayout.lines(wordWidths: [20, 200, 20], spaceWidth: 5, width: 100)
        #expect(lines == [0..<1, 1..<2, 2..<3])
    }

    @Test func aParagraphStartsALine() {
        let lines = CaptionLayout.lines(
            wordWidths: [10, 10, 10, 10, 10], spaceWidth: 5, width: 100, breaksAfter: [1, 4])
        #expect(lines == [0..<2, 2..<5], "a break after the last word adds no empty line")
    }
}

/// The Speech Overlay's feed (ADR-0077): one reading's lines in order, grown
/// passage by passage, each line keeping its number.
struct CaptionFeedTests {

    private static func passage(_ index: Int, first: Int, _ text: String) -> ReadAlongPassage {
        ReadAlongPassage(index: index, firstWord: first, text: text)
    }

    private static func add(_ passage: ReadAlongPassage, to feed: inout CaptionFeed) {
        // Every word 30 wide, a space 5, lines of 70: two words a line.
        feed.append(
            passage, wordWidths: passage.words.map { _ in 30 }, spaceWidth: 5, width: 70)
    }

    @Test func linesKeepTheirNumbersAsPassagesArriveAndLeave() {
        var feed = CaptionFeed()
        Self.add(Self.passage(0, first: 0, "a b c"), to: &feed)
        Self.add(Self.passage(1, first: 3, "d e f g"), to: &feed)
        #expect(feed.lines.map(\.id) == [0, 1, 2, 3])
        #expect(feed.lines.map(\.wordRange) == [0..<2, 2..<3, 3..<5, 5..<7])

        // The same passage again, or an older one, changes nothing.
        Self.add(Self.passage(1, first: 3, "d e f g"), to: &feed)
        #expect(feed.lines.count == 4)

        feed.drop(passagesBefore: 1)
        Self.add(Self.passage(2, first: 7, "h"), to: &feed)
        #expect(feed.lines.map(\.id) == [2, 3, 4])
        #expect(feed.lines.first?.words == ["d", "e"])
    }

    @Test func theHeardWordFindsItsLine() {
        var feed = CaptionFeed()
        #expect(feed.line(holding: 0) == nil)
        Self.add(Self.passage(0, first: 0, "a b c d e"), to: &feed)
        #expect(feed.line(holding: 0) == 0)
        #expect(feed.line(holding: 3) == 1)
        #expect(feed.line(holding: 4) == 2)
        #expect(feed.line(holding: 99) == 2, "past the end: the last line")
    }

    @Test func paragraphsStartLines() {
        var feed = CaptionFeed()
        Self.add(Self.passage(0, first: 0, "a\n\nb c"), to: &feed)
        #expect(feed.lines.map(\.words) == [["a"], ["b", "c"]])
    }
}
