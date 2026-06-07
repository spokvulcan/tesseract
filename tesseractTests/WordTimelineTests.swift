//
//  WordTimelineTests.swift
//  tesseractTests
//
//  Pure-value tests for the Word Timeline (#55) — the single home for the
//  spoken-text token→char→word model and the elapsed→position pacing fold the TTS
//  notch overlay renders. Like `ChatTranscriptTests`, every test is an input ->
//  output assertion with no `Timer`, no clock, no MainActor; the interface is the
//  test surface and nothing reaches into private storage.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct WordTimelineTests {

    // MARK: - Construction: the word char-range model, computed once

    @Test func buildsWordCharRangesFromText() {
        let timeline = WordTimeline(text: "hello world foo")
        #expect(timeline.words.map(\.text) == ["hello", "world", "foo"])
        // charOffset is the cumulative start: offset += word.count + 1 (one space).
        #expect(timeline.words.map(\.charOffset) == [0, 6, 12])
        // joined-with-spaces length: 5 + 1 + 5 + 1 + 3.
        #expect(timeline.totalCharCount == 15)
        #expect(timeline.words.allSatisfy { !$0.isAnnotation })
    }

    @Test func normalizesNewlinesAndCollapsesWhitespace() {
        // Matches the tracker/view: "\n" → space, then split on whitespace omitting
        // empties, so runs of separators don't yield empty words.
        let timeline = WordTimeline(text: "a\nb  c")
        #expect(timeline.words.map(\.text) == ["a", "b", "c"])
        #expect(timeline.words.map(\.charOffset) == [0, 2, 4])
        #expect(timeline.totalCharCount == 5)
    }

    // MARK: - Annotation classification (moved out of the view)

    @Test func classifiesBracketedAndPunctuationOnlyWordsAsAnnotations() {
        let timeline = WordTimeline(text: "hello [laughs] world --- 42")
        let byText = Dictionary(uniqueKeysWithValues: timeline.words.map { ($0.text, $0.isAnnotation) })
        #expect(byText["hello"] == false)
        #expect(byText["[laughs]"] == true)     // bracketed
        #expect(byText["world"] == false)
        #expect(byText["---"] == true)          // no letters or numbers
        #expect(byText["42"] == false)          // digits count as content
    }

    // MARK: - cursor: char → active word + in-word lit fraction

    @Test func cursorReportsActiveWordByEndOffset() {
        // ends: hello→5, world→11, foo→15 (charOffset + word.count).
        let timeline = WordTimeline(text: "hello world foo")
        #expect(timeline.cursor(highlightedCharCount: 0).activeWordIndex == 0)
        #expect(timeline.cursor(highlightedCharCount: 5).activeWordIndex == 0)   // <= end is still this word
        #expect(timeline.cursor(highlightedCharCount: 6).activeWordIndex == 1)
        #expect(timeline.cursor(highlightedCharCount: 9).activeWordIndex == 1)
        // Past the end clamps to the last word (matches the view's fallback).
        #expect(timeline.cursor(highlightedCharCount: 100).activeWordIndex == 2)
    }

    @Test func cursorInWordLitFractionTracksLetterProgress() {
        let timeline = WordTimeline(text: "hello world foo")
        // 3 chars into "world" (5 letters) → 0.6.
        #expect(abs(timeline.cursor(highlightedCharCount: 9).inWordLitFraction - 0.6) < 1e-9)
        // Start of a word → nothing lit yet.
        #expect(timeline.cursor(highlightedCharCount: 6).inWordLitFraction == 0.0)
    }

    // MARK: - litFraction: the per-word value the view renders

    @Test func litFractionIsLitCountOverLetterCountClampedAtBothEnds() {
        let timeline = WordTimeline(text: "hello world foo")
        #expect(abs(timeline.litFraction(wordIndex: 0, charCount: 3) - 0.6) < 1e-9)  // 3/5
        #expect(timeline.litFraction(wordIndex: 0, charCount: 100) == 1.0)           // saturates at fully lit
        #expect(timeline.litFraction(wordIndex: 0, charCount: -5) == 0.0)            // never negative
        #expect(timeline.litFraction(wordIndex: 99, charCount: 0) == 0.0)            // out of range is safe
    }

    // MARK: - advance: the single pacing fold (replaces tickToken / tickProportional)

    /// A 100-char single "word" makes the proportional math exact: charCount ==
    /// floor(progress * 100).
    private func hundredCharTimeline(tokenCharOffsets: [Int] = []) -> WordTimeline {
        let t = WordTimeline(text: String(repeating: "a", count: 100), tokenCharOffsets: tokenCharOffsets)
        #expect(t.totalCharCount == 100)
        return t
    }

    @Test func advanceMapsElapsedProportionallyAcrossSmoothedDuration() {
        let timeline = hundredCharTimeline()
        // seed == target ⇒ smoothing is a no-op, progress = 5/10 = 0.5 ⇒ 50 chars.
        let (count, pacing) = timeline.advance(
            elapsed: 5, totalDuration: 10, estimatedFinalDuration: 10,
            isGenerationComplete: false, pacing: .init(seed: 10)
        )
        #expect(count == 50)
        #expect(pacing.smoothedEffDuration == 10)
    }

    @Test func advanceThreadsTheSmoothingCarryInAndOut() {
        let timeline = hundredCharTimeline()
        // smoothed' = 10 + (20 - 10) * 0.08 = 10.8; progress = 5 / 10.8 ⇒ floor(46.29) = 46.
        let (count, pacing) = timeline.advance(
            elapsed: 5, totalDuration: 20, estimatedFinalDuration: 20,
            isGenerationComplete: false, pacing: .init(seed: 10)
        )
        #expect(count == 46)
        #expect(abs(pacing.smoothedEffDuration - 10.8) < 1e-9)
    }

    @Test func advanceClampsProgressToOne() {
        let timeline = hundredCharTimeline()
        let (count, _) = timeline.advance(
            elapsed: 10_000, totalDuration: 10, estimatedFinalDuration: 10,
            isGenerationComplete: false, pacing: .init(seed: 10)
        )
        #expect(count == 100)
    }

    @Test func advanceUsesActualDurationOnceGenerationIsComplete() {
        let timeline = hundredCharTimeline()
        // With a huge estimate but isGenerationComplete, the target is totalDuration
        // (10), not max(10, 1000) — so progress = 5/10 = 0.5 ⇒ 50, not a tiny count.
        let (count, _) = timeline.advance(
            elapsed: 5, totalDuration: 10, estimatedFinalDuration: 1000,
            isGenerationComplete: true, pacing: .init(seed: 10)
        )
        #expect(count == 50)
    }

    @Test func advanceHandlesDegenerateZeroDurationAndEmptyText() {
        let empty = WordTimeline(text: "   ")
        #expect(empty.totalCharCount == 0)
        let (emptyCount, _) = empty.advance(
            elapsed: 5, totalDuration: 10, estimatedFinalDuration: 10,
            isGenerationComplete: false, pacing: .init(seed: 10)
        )
        #expect(emptyCount == 0)

        // Zero smoothed duration ⇒ no division by zero, just nothing lit yet.
        let timeline = hundredCharTimeline()
        let (zeroCount, _) = timeline.advance(
            elapsed: 5, totalDuration: 0, estimatedFinalDuration: 0,
            isGenerationComplete: false, pacing: .init(seed: 0)
        )
        #expect(zeroCount == 0)
    }

    /// Pins the *preserved* behavior: the token-offset snap currently coincides with
    /// the proportional target (the old `min(max(charPos, targetChars), total)` is a
    /// latent no-op). Token-present must equal token-absent — neither a floor-snap
    /// (would give 40) nor a ceil-snap (would give 60) is active. See WordTimeline.snap.
    @Test func advanceTokenPathCurrentlyEqualsProportional() {
        let proportional = hundredCharTimeline()
        let tokenAligned = hundredCharTimeline(tokenCharOffsets: [0, 20, 40, 60, 80])
        let inputs = (elapsed: 5.0, totalDuration: 10.0, est: 10.0)

        let (pCount, _) = proportional.advance(
            elapsed: inputs.elapsed, totalDuration: inputs.totalDuration,
            estimatedFinalDuration: inputs.est, isGenerationComplete: false, pacing: .init(seed: 10)
        )
        let (tCount, _) = tokenAligned.advance(
            elapsed: inputs.elapsed, totalDuration: inputs.totalDuration,
            estimatedFinalDuration: inputs.est, isGenerationComplete: false, pacing: .init(seed: 10)
        )
        #expect(pCount == 50)
        #expect(tCount == pCount)
    }

    /// `advance` is a pure fold with no memory; the monotonic "never regress" clamp
    /// lives in the driver (TTS Word Tracker), not here (#55 US#8).
    @Test func advanceIsNotMonotonic() {
        let timeline = hundredCharTimeline()
        let (high, _) = timeline.advance(
            elapsed: 5, totalDuration: 10, estimatedFinalDuration: 10,
            isGenerationComplete: false, pacing: .init(seed: 10)
        )
        let (low, _) = timeline.advance(
            elapsed: 1, totalDuration: 10, estimatedFinalDuration: 10,
            isGenerationComplete: false, pacing: .init(seed: 10)
        )
        #expect(high == 50)
        #expect(low == 10)   // a smaller elapsed yields a smaller count — no internal floor
    }
}
