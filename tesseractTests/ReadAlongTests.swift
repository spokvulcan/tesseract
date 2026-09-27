//
//  ReadAlongTests.swift
//  tesseractTests
//
//  The Read-Along (ADR-0076): which word of the speech now playing is heard.
//  `ReadAlongTimeline` is pure and tested as input -> output; the
//  `SpeechReadAlong` clock is driven by a test clock and `tick()`, the same
//  call its timer makes.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct ReadAlongTimelineTests {

    @Test func aSegmentStartsWhenThePlaybackHeadReachesItNotWhenItArrives() {
        var timeline = ReadAlongTimeline(charsPerSecond: 10)
        timeline.append(text: "one two three four five", start: 0)
        timeline.setEnd(2, at: 0)
        // Under lookahead the next script arrives seconds before its audio.
        timeline.append(text: "six seven eight nine ten", start: 2)

        #expect(timeline.position(at: 1.9)?.segment == 0)
        #expect(timeline.position(at: 2.0)?.segment == 1)
        #expect(timeline.position(at: 2.0)?.word == 0)
    }

    @Test func wordsArePlacedInProportionToTheirCharacters() {
        var timeline = ReadAlongTimeline(charsPerSecond: 10)
        // Word ends at characters 4, 9, 14, 19.
        timeline.append(text: "aaaa bbbb cccc dddd", start: 0)
        timeline.setEnd(4, at: 0)

        #expect(timeline.position(at: 0.1)?.word == 0)
        #expect(timeline.position(at: 1.5)?.word == 1)  // 7 of 19 characters
        #expect(timeline.position(at: 3.9)?.word == 3)
    }

    @Test func aSegmentStillGeneratingFollowsTheLearnedPace() {
        var timeline = ReadAlongTimeline(charsPerSecond: 10)
        timeline.append(text: "aaaa bbbb cccc dddd", start: 0)

        #expect(timeline.position(at: 0.7)?.word == 1)  // 7 characters in
        #expect(timeline.position(at: 1.2)?.word == 2)  // 12 characters in
    }

    /// The first segment always plays while it generates: when its real end
    /// arrives the word carries on from where it was, instead of snapping
    /// back to where the end alone would put it.
    @Test func learningTheEndMidSegmentNeverMovesTheWordBack() {
        var timeline = ReadAlongTimeline(charsPerSecond: 10)
        timeline.append(text: "aaaa bbbb cccc dddd", start: 0)
        let heard = timeline.position(at: 1.2)?.word
        #expect(heard == 2)

        // It runs 8 s, not the 1.9 s the pace estimated. Mapped over the end
        // alone, 1.2 s would be 2.85 characters: back to the first word.
        timeline.setEnd(8, at: 1.2)

        #expect(timeline.position(at: 1.2)?.word == heard)
        #expect(timeline.position(at: 4.0)?.word == 2)
        #expect(timeline.position(at: 7.9)?.word == 3)
    }

    @Test func finishedSegmentsTeachThePace() {
        var timeline = ReadAlongTimeline(charsPerSecond: 10)
        timeline.append(text: String(repeating: "word ", count: 20), start: 0)  // 99 characters
        timeline.setEnd(5, at: 0)
        #expect(timeline.charsPerSecond > 15)

        let learned = timeline.charsPerSecond
        // The same end again (segment done, then finished) teaches nothing.
        timeline.setEnd(5, at: 1)
        #expect(timeline.charsPerSecond == learned)
    }

    @Test func finishedOnlyOnceTheHeadPassesTheLastEnd() {
        var timeline = ReadAlongTimeline(charsPerSecond: 10)
        timeline.append(text: "one two", start: 0)
        #expect(!timeline.isFinished(at: 100), "no end is known yet")
        timeline.setEnd(2, at: 0)
        #expect(!timeline.isFinished(at: 1.5))
        #expect(timeline.isFinished(at: 2))
    }

    @Test func pruningKeepsTheSegmentBeforeTheOneHeard() {
        var timeline = ReadAlongTimeline(charsPerSecond: 10)
        for index in 0..<4 {
            timeline.append(text: "segment \(index)", start: Double(index))
            timeline.setEnd(Double(index + 1), at: 0)
        }
        timeline.prune(keeping: 3)
        #expect(timeline.segments.map(\.index) == [2, 3])
        #expect(timeline.segments.first?.firstWord == 4, "word counts survive pruning")
    }
}

@MainActor
struct SpeechReadAlongTests {

    /// The playback head, moved by the test.
    final class TestClock {
        var now: TimeInterval = 0
    }

    private func show(
        _ readAlong: SpeechReadAlong, _ text: String, clock: TestClock
    ) {
        readAlong.show(text: text, playbackTimeProvider: { clock.now })
    }

    @Test func eachUtteranceShownIsNewAndStartsOnItsFirstWord() {
        let readAlong = SpeechReadAlong()
        let clock = TestClock()
        let before = readAlong.utteranceID

        show(readAlong, "one two three", clock: clock)
        #expect(readAlong.isActive)
        #expect(readAlong.utteranceID != before)
        #expect(readAlong.segment?.index == 0)
        #expect(readAlong.word == 0)

        readAlong.dismiss()
        #expect(!readAlong.isActive)
    }

    @Test func switchesSegmentAtThePlaybackHead() {
        let readAlong = SpeechReadAlong()
        let clock = TestClock()
        show(readAlong, "one two three four", clock: clock)
        readAlong.updateTotalDuration(2)
        readAlong.switchText("five six seven eight", segmentBase: 2)

        clock.now = 1.9
        readAlong.tick()
        #expect(readAlong.segment?.index == 0)
        #expect(readAlong.word == 3)

        clock.now = 2.05
        readAlong.tick()
        #expect(readAlong.segment?.index == 1)
        #expect(readAlong.word == 0)
        readAlong.dismiss()
    }

    /// A time-left label follows the pace, so it has to be published.
    @Test func thePaceLearnedFromAFinishedSegmentIsPublished() {
        let readAlong = SpeechReadAlong()
        let clock = TestClock()
        let before = readAlong.charsPerSecond
        show(readAlong, "one two three four", clock: clock)  // 18 characters
        readAlong.updateTotalDuration(2)
        #expect(readAlong.charsPerSecond < before)
        #expect(abs(readAlong.charsPerSecond - (0.7 * 9 + 0.3 * before)) < 0.001)
        readAlong.dismiss()
    }

    @Test func theWordOnlyMovesForwardWithinASegment() {
        let readAlong = SpeechReadAlong()
        let clock = TestClock()
        show(readAlong, "one two three four", clock: clock)
        readAlong.updateTotalDuration(4)

        clock.now = 3
        readAlong.tick()
        let heard = readAlong.word
        clock.now = 1
        readAlong.tick()
        #expect(readAlong.word == heard)
        readAlong.dismiss()
    }

    @Test func letsGoShortlyAfterTheLastAudioOnceGenerationIsComplete() async {
        let readAlong = SpeechReadAlong()
        let clock = TestClock()
        show(readAlong, "one two", clock: clock)
        readAlong.updateTotalDuration(1)

        clock.now = 1
        readAlong.tick()
        #expect(readAlong.isActive, "more may still be generating")

        readAlong.markGenerationComplete()
        readAlong.tick()
        #expect(readAlong.isActive, "the last word stays lit a moment")
        let deadline = ContinuousClock.now + .seconds(3)
        while readAlong.isActive, ContinuousClock.now < deadline {
            try? await Task.sleep(for: .milliseconds(20))
        }
        #expect(!readAlong.isActive)
    }
}
