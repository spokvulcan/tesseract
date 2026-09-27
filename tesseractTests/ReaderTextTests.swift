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

    @Test func thePageHoldsTheHeardWord() {
        let lines: [Range<Int>] = [0..<3, 3..<6, 6..<9, 9..<10]
        #expect(Array(CaptionLayout.page(for: 0, in: lines)) == [0..<3, 3..<6])
        #expect(Array(CaptionLayout.page(for: 7, in: lines)) == [6..<9, 9..<10])
        #expect(Array(CaptionLayout.page(for: -1, in: lines)) == [0..<3, 3..<6])
        #expect(CaptionLayout.page(for: 0, in: []).isEmpty)
    }
}
