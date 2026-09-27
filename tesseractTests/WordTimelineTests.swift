//
//  WordTimelineTests.swift
//  tesseractTests
//
//  Pure-value tests for the Word Timeline: one segment's words on a
//  character line, and the word a character count falls in — the half of the
//  Read-Along that turns the heard character into a word. Input -> output,
//  no clock, no MainActor.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct WordTimelineTests {

    @Test func buildsWordCharRangesFromText() {
        let timeline = WordTimeline(text: "hello world foo")
        #expect(timeline.words.map(\.text) == ["hello", "world", "foo"])
        // charOffset is the cumulative start: offset += word.count + 1 (one space).
        #expect(timeline.words.map(\.charOffset) == [0, 6, 12])
        // joined-with-spaces length: 5 + 1 + 5 + 1 + 3.
        #expect(timeline.totalCharCount == 15)
    }

    @Test func normalizesNewlinesAndCollapsesWhitespace() {
        // Shared word split (StringWordSplitting): whitespace OR newline separators,
        // empty runs omitted — the same words the engine speaks and the Reader maps.
        let timeline = WordTimeline(text: "a\nb  c")
        #expect(timeline.words.map(\.text) == ["a", "b", "c"])
        #expect(timeline.words.map(\.charOffset) == [0, 2, 4])
        #expect(timeline.totalCharCount == 5)
    }

    @Test func emptyTextHasNoWords() {
        let timeline = WordTimeline(text: "   ")
        #expect(timeline.words.isEmpty)
        #expect(timeline.totalCharCount == 0)
        #expect(timeline.activeWordIndex(highlightedCharCount: 3) == 0)
    }

    @Test func activeWordIndexReportsActiveWordByEndOffset() {
        // ends: hello→5, world→11, foo→15 (charOffset + word.count).
        let timeline = WordTimeline(text: "hello world foo")
        #expect(timeline.activeWordIndex(highlightedCharCount: 0) == 0)
        #expect(timeline.activeWordIndex(highlightedCharCount: 5) == 0)  // <= end is still this word
        #expect(timeline.activeWordIndex(highlightedCharCount: 6) == 1)
        #expect(timeline.activeWordIndex(highlightedCharCount: 9) == 1)
        // Past the end clamps to the last word.
        #expect(timeline.activeWordIndex(highlightedCharCount: 100) == 2)
    }
}
