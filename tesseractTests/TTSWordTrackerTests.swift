//
//  TTSWordTrackerTests.swift
//  tesseractTests
//
//  Hermetic tests for the TTS Word Tracker (#140) — the stateful driver over the
//  pure Word Timeline. Every test constructs the tracker with a *scripted* playback
//  clock (the already-injected `playbackTimeProvider` seam; no wall clock, in the
//  style of `InMemoryAudioPlayback`'s virtual clock), drives the public API, and
//  pumps the frame fold directly instead of waiting on the production Timer. Only
//  published state is asserted — the recognized character count's progression and
//  monotonicity, activity, completion, and fade state; private carries (pacing,
//  estimates) are asserted through their visible effect on progression.
//
//  The panel controller's NSPanel lifecycle is the accepted untested sliver
//  (platform-adapter island); the dismissal convergence is pinned here via the
//  tracker's fade-state transitions.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct TTSWordTrackerTests {

    /// Mutable box the scripted `playbackTimeProvider` closure reads — the test's
    /// virtual playback head.
    final class ScriptedClock {
        var now: TimeInterval = 0
    }

    /// 30 × "word" joined by single spaces → `totalCharCount` 149; at the default
    /// learned rate of 15 chars/sec the seeded duration estimate is ~9.93 s.
    static let longText = Array(repeating: "word", count: 30).joined(separator: " ")

    /// Drive the frame fold directly — the production 60 fps Timer never runs here.
    private func pump(_ tracker: TTSWordTracker, frames: Int) {
        for _ in 0..<frames { tracker.tick() }
    }

    // MARK: - Fade state: one owned transition, read-only for consumers

    @Test func fadeStateIsSetByItsOwnTransitionAndClearedByLifecycle() {
        let tracker = TTSWordTracker()
        #expect(!tracker.isFadingOut)

        tracker.beginFadeOut()
        #expect(tracker.isFadingOut)

        // stop() — the teardown path — clears the fade for the next show.
        tracker.stop()
        #expect(!tracker.isFadingOut)

        // A fresh start() also clears a stale fade.
        tracker.beginFadeOut()
        tracker.start(text: "hello world", tokenCharOffsets: [], playbackTimeProvider: { 0 })
        #expect(!tracker.isFadingOut)
        tracker.stop()
    }

    // MARK: - Progression against the scripted clock

    @Test func highlightProgressesWithTheScriptedClockAndReachesFullOnCompletion() {
        let tracker = TTSWordTracker()
        let clock = ScriptedClock()
        tracker.start(
            text: Self.longText, tokenCharOffsets: [], playbackTimeProvider: { clock.now })
        defer { tracker.stop() }
        let total = tracker.timeline.totalCharCount
        #expect(total == 149)
        #expect(tracker.isActive)

        // No scheduled duration yet → the fold is gated, nothing highlights.
        pump(tracker, frames: 10)
        #expect(tracker.recognizedCharCount == 0)

        tracker.updateTotalDuration(10)  // 10 s of scheduled audio for this segment

        // Head at 5 s of a ~10 s effective duration → roughly half the text lit
        // (independent worked example: 5 s × ~15 chars/sec ≈ 75 of 149).
        clock.now = 5
        pump(tracker, frames: 120)  // let the estimate→actual smoothing converge
        let midCount = tracker.recognizedCharCount
        #expect((60...90).contains(midCount))

        // Generation complete and the head past the scheduled end → fully lit.
        tracker.markGenerationComplete()
        #expect(tracker.isGenerationComplete)
        clock.now = 10.5
        pump(tracker, frames: 120)
        #expect(tracker.recognizedCharCount == total)
    }

    // MARK: - Segment Window: head and duration measured against one shared base

    @Test func segmentWindowMeasuresHeadAndDurationAgainstOneSharedBase() {
        let tracker = TTSWordTracker()
        let clock = ScriptedClock()
        tracker.start(
            text: "intro segment text", tokenCharOffsets: [], playbackTimeProvider: { clock.now })
        defer { tracker.stop() }
        tracker.updateTotalDuration(10)  // segment 1: cumulative scheduled duration 10 s

        // Cross into segment 2 at the Segment Window: base = 10 s of prior audio.
        tracker.updateText(Self.longText, tokenCharOffsets: [], segmentBase: 10)
        tracker.updateTotalDuration(20)  // cumulative 20 s → this segment spans 10 s
        let total = tracker.timeline.totalCharCount

        // Head exactly at the base → zero elapsed inside this segment, nothing lit.
        clock.now = 10
        pump(tracker, frames: 10)
        #expect(tracker.recognizedCharCount == 0)

        // Head 5 s past the base of this 10 s segment → roughly half. A head not
        // rebased would read 15 s (fully lit); a duration not rebased would read
        // 20 s (about a quarter) — either fails this window.
        clock.now = 15
        pump(tracker, frames: 120)
        #expect((60...90).contains(tracker.recognizedCharCount))

        // Head past the cumulative end → the full segment.
        tracker.markGenerationComplete()
        clock.now = 20.5
        pump(tracker, frames: 120)
        #expect(tracker.recognizedCharCount == total)
    }

    // MARK: - Estimate alignment on segment completion

    @Test func markSegmentCompleteAlignsTheEstimateSoTheHighlightConverges() {
        let tracker = TTSWordTracker()
        let clock = ScriptedClock()
        // Double the text → the seeded estimate (~19.9 s) far exceeds the actual 10 s.
        let text = Self.longText + " " + Self.longText
        tracker.start(text: text, tokenCharOffsets: [], playbackTimeProvider: { clock.now })
        defer { tracker.stop() }
        let total = tracker.timeline.totalCharCount
        tracker.updateTotalDuration(10)

        // Head at the segment's actual end, but the oversized estimate still paces
        // the fold → visibly short of full.
        clock.now = 10
        pump(tracker, frames: 120)
        #expect(tracker.recognizedCharCount < total)

        // Aligning the estimate to actual converges the highlight to 100% for this
        // segment — without flipping generation-complete (more segments remain).
        tracker.markSegmentComplete()
        clock.now = 10.5
        pump(tracker, frames: 240)
        #expect(tracker.recognizedCharCount == total)
        #expect(!tracker.isGenerationComplete)
        #expect(tracker.isActive)
    }

    // MARK: - Monotonic highlight guarantee

    @Test func recognizedCharCountNeverMovesBackward() {
        let tracker = TTSWordTracker()
        let clock = ScriptedClock()
        tracker.start(
            text: Self.longText, tokenCharOffsets: [], playbackTimeProvider: { clock.now })
        defer { tracker.stop() }
        tracker.updateTotalDuration(10)

        clock.now = 5
        pump(tracker, frames: 120)
        let advanced = tracker.recognizedCharCount
        #expect(advanced > 0)

        // The playback head jumping backward must not pull the highlight back, nor
        // may new audio stretching the schedule (the progress fraction drops).
        clock.now = 2
        var previous = advanced
        for _ in 0..<60 {
            tracker.tick()
            #expect(tracker.recognizedCharCount >= previous)
            previous = tracker.recognizedCharCount
        }
        clock.now = 5
        tracker.updateTotalDuration(30)
        for _ in 0..<120 {
            tracker.tick()
            #expect(tracker.recognizedCharCount >= previous)
            previous = tracker.recognizedCharCount
        }
    }

    // MARK: - EWMA rate learning: shifts the next seed, owned per instance

    @Test func ewmaRateLearningShiftsTheNextSegmentsPacingSeedPerInstance() {
        let clock = ScriptedClock()
        let learner = TTSWordTracker()
        // Generation 1: 149 chars actually spoken in 5 s → 29.8 chars/sec, far above
        // the 15 chars/sec default. EWMA (0.7 × actual + 0.3 × prior) → ~25.4.
        learner.start(
            text: Self.longText, tokenCharOffsets: [], playbackTimeProvider: { clock.now })
        defer { learner.stop() }
        learner.updateTotalDuration(5)
        learner.markGenerationComplete()

        // Generation 2 on the same tracker: the seeded estimate is now 149/25.4 ≈
        // 5.9 s. With barely any audio scheduled the seed alone paces the first
        // frame: a 2 s head yields ~2 × 25.4 ≈ 51 chars, where the default seed
        // would give ~2 × 15 ≈ 30.
        clock.now = 0
        learner.start(
            text: Self.longText, tokenCharOffsets: [], playbackTimeProvider: { clock.now })
        learner.updateTotalDuration(0.1)
        clock.now = 2
        pump(learner, frames: 1)  // one frame — the seed, before smoothing moves it
        #expect((45...56).contains(learner.recognizedCharCount))

        // A fresh tracker still paces at the default: the learned rate is instance
        // state, not a process-global steering every tracker in the process.
        let freshClock = ScriptedClock()
        let fresh = TTSWordTracker()
        fresh.start(
            text: Self.longText, tokenCharOffsets: [], playbackTimeProvider: { freshClock.now })
        defer { fresh.stop() }
        fresh.updateTotalDuration(0.1)
        freshClock.now = 2
        pump(fresh, frames: 1)
        #expect((25...35).contains(fresh.recognizedCharCount))
    }
}
