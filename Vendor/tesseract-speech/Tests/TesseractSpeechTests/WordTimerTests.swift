// The word timer (ADR-0077) on synthetic attention rows and audio levels:
// where the path steps, how pauses move a start, what the silence cap's
// dropped frames do to the count, and when starts are sent.

import Foundation
import Testing
@testable import TesseractSpeech

@Suite struct WordTimerTests {

    private static let voiced = SilenceCap.Frame(loudness: -20, kept: true)
    private static let silent = SilenceCap.Frame(loudness: -90, kept: true)
    private static let dropped = SilenceCap.Frame(loudness: -90, kept: false)

    /// A row of `width` logits looking at `position`.
    private static func row(_ position: Int, width: Int) -> [Float] {
        (0 ..< width).map { $0 == position ? 12 : 0 }
    }

    /// Feeds rows (one per frame, each the position looked at) and levels,
    /// then finishes: every start, in order.
    private static func time(
        _ text: String, offsets: [Int], reference: Int = 0, looks: [Int],
        levels: [SilenceCap.Frame]
    ) -> [WordStart] {
        var timer = WordTimer(text: text, tokenOffsets: offsets, referenceTokens: reference)
        let width = reference + offsets.count + 1
        var starts: [WordStart] = []
        for look in looks { timer.appendAttention(row(look, width: width)) }
        timer.appendAudio(levels)
        starts += timer.takeStarts()
        timer.finish()
        return starts + timer.takeStarts()
    }

    private static func frames(_ level: SilenceCap.Frame, _ count: Int) -> [SilenceCap.Frame] {
        [SilenceCap.Frame](repeating: level, count: count)
    }

    // MARK: - Words

    @Test func tokensBelongToTheWordOfTheirFirstLetter() {
        // "Hello" "," " world" "." " Hi"
        let (words, count) = WordTimer.tokenWords(
            offsets: [0, 5, 6, 12, 13], text: "Hello, world. Hi")
        #expect(words == [0, 0, 1, 1, 2])
        #expect(count == 3)
    }

    @Test func aTokenOfSpacesGoesWithTheNextWord() {
        // "Hi" " " " there" "\nyou"
        let (words, count) = WordTimer.tokenWords(offsets: [0, 2, 3, 9], text: "Hi  there\nyou")
        #expect(words == [0, 1, 1, 2])
        #expect(count == 3, "newlines split words too")
    }

    // MARK: - The path

    /// Each word starts where the head steps onto it.
    @Test func wordsStartWhereTheHeadStepsOntoThem() {
        let looks = [Int](repeating: 0, count: 5) + [Int](repeating: 1, count: 5)
            + [Int](repeating: 2, count: 5) + [3, 3, 3]
        let starts = Self.time(
            "one two three", offsets: [0, 3, 7], looks: looks,
            levels: Self.frames(Self.voiced, 15) + Self.frames(Self.silent, 3))
        #expect(starts == [
            WordStart(word: 0, frame: 0), WordStart(word: 1, frame: 5),
            WordStart(word: 2, frame: 10),
        ])
    }

    /// A glance back doesn't move the path back.
    @Test func aGlanceBackDoesNotMoveAWordBack() {
        let looks = [0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 2, 2, 2, 2, 3]
        let starts = Self.time(
            "one two three", offsets: [0, 3, 7], looks: looks,
            levels: Self.frames(Self.voiced, 14) + [Self.silent])
        #expect(starts.map(\.frame) == [0, 4, 10])
    }

    // MARK: - Pauses

    /// The head moves on during the pause; the voice speaks when it ends.
    @Test func aStartInsideAPauseMovesToWhereTheSoundResumes() {
        // "one" "," " two": the head reaches "two" at frame 6, mid-pause.
        let looks = [0, 0, 0, 0, 1, 1] + [Int](repeating: 2, count: 7) + [3, 3]
        let levels =
            Self.frames(Self.voiced, 6) + Self.frames(Self.silent, 5) + Self.frames(Self.voiced, 2)
            + Self.frames(Self.silent, 2)
        let starts = Self.time("one, two", offsets: [0, 3, 4], looks: looks, levels: levels)
        #expect(starts == [WordStart(word: 0, frame: 0), WordStart(word: 1, frame: 11)])
    }

    /// Stepping on in the last sound before a pause counts too.
    @Test func aStartJustBeforeAPauseMovesPastIt() {
        let looks = [0, 0, 0, 0, 1] + [Int](repeating: 2, count: 8) + [3, 3]
        let levels =
            Self.frames(Self.voiced, 6) + Self.frames(Self.silent, 5) + Self.frames(Self.voiced, 2)
            + Self.frames(Self.silent, 2)
        let starts = Self.time("one, two", offsets: [0, 3, 4], looks: looks, levels: levels)
        #expect(starts.map(\.frame) == [0, 11])
    }

    /// A one-frame dip is a stop consonant, not a pause.
    @Test func aOneFrameDipIsNotAPause() {
        let looks = [0, 0, 0, 1, 1, 1, 1, 1, 2]
        let levels = Self.frames(Self.voiced, 4) + [Self.silent] + Self.frames(Self.voiced, 3)
            + [Self.silent]
        let starts = Self.time("one two", offsets: [0, 3], looks: looks, levels: levels)
        #expect(starts.map(\.frame) == [0, 3])
    }

    /// The last word keeps its start when the trailing silence begins as the
    /// head reaches EOS.
    @Test func theTrailingSilenceDoesNotMoveTheLastWord() {
        let looks = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2]
        let levels = Self.frames(Self.voiced, 7) + Self.frames(Self.silent, 5)
        let starts = Self.time("Maybe so.", offsets: [0, 5], looks: looks, levels: levels)
        #expect(starts.map(\.frame) == [0, 4])
    }

    // MARK: - A take ahead of the text

    /// While the head looks at the take's text, the new text hasn't begun.
    @Test func theTakesTextComesBeforeTheFirstWord() {
        // Two take tokens, then "one" " two", then EOS.
        let looks = [1, 1, 1, 1] + [2, 2, 2, 2] + [3, 3, 3, 3] + [4]
        let levels = Self.frames(Self.silent, 4) + Self.frames(Self.voiced, 8) + [Self.silent]
        let starts = Self.time(
            "one two", offsets: [0, 3], reference: 2, looks: looks, levels: levels)
        #expect(starts.map(\.frame) == [4, 8])
    }

    // MARK: - Counting frames

    /// Frames the silence cap dropped don't count: a start is a frame of the
    /// audio that plays.
    @Test func droppedFramesAreNotCounted() {
        let looks = [0, 0, 0] + [Int](repeating: 1, count: 23) + [2]
        let levels =
            Self.frames(Self.voiced, 3) + Self.frames(Self.silent, 15)
            + Self.frames(Self.dropped, 5) + Self.frames(Self.voiced, 3) + [Self.silent]
        let starts = Self.time("one two", offsets: [0, 3], looks: looks, levels: levels)
        #expect(starts.map(\.frame) == [0, 18], "frame 23 is the 19th kept")
    }

    // MARK: - When starts are sent

    /// Starts go out as the path and the audio settle them, before the end.
    @Test func startsAreSentBeforeTheSegmentEnds() {
        var timer = WordTimer(
            text: "one two three four", tokenOffsets: [0, 3, 7, 13], referenceTokens: 0)
        for look in [Int](repeating: 0, count: 6) + [Int](repeating: 1, count: 6)
            + [Int](repeating: 2, count: 6)
        {
            timer.appendAttention(Self.row(look, width: 5))
        }
        #expect(timer.takeStarts().isEmpty, "no audio yet")
        timer.appendAudio(Self.frames(Self.voiced, 18))
        let early = timer.takeStarts()
        #expect(early.map(\.word) == [0], "word 1's next step isn't decided yet")
        for look in [Int](repeating: 3, count: 10) { timer.appendAttention(Self.row(look, width: 5)) }
        timer.appendAudio(Self.frames(Self.voiced, 10))
        #expect(timer.takeStarts().map(\.frame) == [6, 12])
        timer.finish()
        #expect(timer.takeStarts().map(\.word) == [3])
    }

    @Test func noRowsNoStarts() {
        var timer = WordTimer(text: "one two", tokenOffsets: [0, 3], referenceTokens: 0)
        timer.appendAudio(Self.frames(Self.voiced, 6))
        timer.finish()
        #expect(timer.takeStarts().isEmpty)
    }
}
