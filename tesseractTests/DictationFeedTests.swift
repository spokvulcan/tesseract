//
//  DictationFeedTests.swift
//  tesseractTests
//
//  Exercises the **Overlay Feed** as a pure value surface: phase transitions
//  and the recording timestamp, beat identity (equal outcomes still read as
//  distinct beats), meter clamping and the change dead-band, and the
//  attach-once meter pump. No audio engine — frames are hand-built.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct DictationFeedTests {

    // MARK: - Phase

    @Test
    func recordingPhaseStampsTheStartAndLeavingItClearsIt() {
        let feed = DictationFeed()
        #expect(feed.recordingStarted == nil)

        feed.setPhase(.recording)
        let started = feed.recordingStarted
        #expect(started != nil)

        // Re-asserting .recording keeps the original timestamp — elapsed-time
        // displays must not reset mid-recording.
        feed.setPhase(.recording)
        #expect(feed.recordingStarted == started)

        feed.setPhase(.processing)
        #expect(feed.recordingStarted == nil)
    }

    @Test
    func isActiveCoversExactlyTheLivePhases() {
        #expect(DictationFeed.Phase.recording.isActive)
        #expect(DictationFeed.Phase.processing.isActive)
        #expect(!DictationFeed.Phase.idle.isActive)
        #expect(!DictationFeed.Phase.error(.noSpeechDetected).isActive)
    }

    // MARK: - Beats

    @Test
    func equalOutcomesInARowStillReadAsTwoBeats() {
        let feed = DictationFeed()
        #expect(feed.beat == nil)

        feed.emit(.empty)
        let first = feed.beat
        feed.emit(.empty)
        let second = feed.beat

        // Same outcome, distinct beats — the id is what lets a variant react
        // to every ending, not just to outcome changes.
        #expect(first?.outcome == .empty)
        #expect(second?.outcome == .empty)
        #expect(first != second)
    }

    // MARK: - Meters

    @Test
    func meterFrameIsClampedAndApplied() {
        let feed = DictationFeed()
        var bands = MeterFrame.zeroBands
        bands[0] = 0.75

        feed.apply(MeterFrame(level: 1.7, bands: bands))
        #expect(feed.level == 1)
        #expect(feed.spectrum == bands)

        feed.apply(MeterFrame(level: -0.3, bands: MeterFrame.zeroBands))
        #expect(feed.level == 0)
        #expect(feed.spectrum == MeterFrame.zeroBands)
    }

    @Test
    func subThresholdLevelChangeWithUnchangedBandsIsDropped() {
        let feed = DictationFeed()
        feed.apply(MeterFrame(level: 0.5, bands: MeterFrame.zeroBands))
        #expect(feed.level == 0.5)

        // Inside the 0.001 dead-band with identical bands: no re-publish, so
        // meter-reading subtrees are not invalidated by float jitter.
        feed.apply(MeterFrame(level: 0.5005, bands: MeterFrame.zeroBands))
        #expect(feed.level == 0.5)
    }

    @Test
    func attachedStreamDrivesTheMeter() async {
        let feed = DictationFeed()
        let (stream, continuation) = AsyncStream.makeStream(of: MeterFrame.self)
        feed.attachMeters(stream)

        continuation.yield(MeterFrame(level: 0.42, bands: MeterFrame.zeroBands))

        var n = 0
        while feed.level != 0.42 && n < 100_000 {
            n += 1
            await Task.yield()
        }
        #expect(feed.level == 0.42)
        continuation.finish()
    }
}
