//
//  VoiceEndpointerTests.swift
//  tesseractTests
//
//  The voice session's ears (#310): speech-start debounce, trailing-silence
//  end-of-speech, and the barge-in profile's blip rejection — pure state
//  machine, driven with synthetic level samples at meter cadence.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@Suite struct VoiceEndpointerTests {

    /// Feed a constant level for a duration at 50 ms cadence.
    private func feed(
        _ endpointer: inout VoiceEndpointer, level: Float, from start: TimeInterval,
        for duration: TimeInterval
    ) -> (events: [VoiceEndpointer.Event], end: TimeInterval) {
        var events: [VoiceEndpointer.Event] = []
        var time = start
        while time <= start + duration {
            if let event = endpointer.ingest(level: level, at: time) { events.append(event) }
            time += 0.05
        }
        return (events, time)
    }

    @Test func sustainedSpeechStartsAfterDebounce() {
        var endpointer = VoiceEndpointer(config: .listening())
        let (events, _) = feed(&endpointer, level: 0.5, from: 0, for: 0.5)
        #expect(events == [.speechStarted])
        #expect(endpointer.isInSpeech)
    }

    @Test func aShortBlipNeverStartsSpeech() {
        var endpointer = VoiceEndpointer(config: .listening())
        // One loud frame (a keyboard thump), then silence.
        _ = endpointer.ingest(level: 0.9, at: 0)
        let (events, _) = feed(&endpointer, level: 0.02, from: 0.05, for: 2.0)
        #expect(events.isEmpty)
        #expect(!endpointer.isInSpeech)
    }

    @Test func trailingSilenceEndsTheTurn() {
        var endpointer = VoiceEndpointer(config: .listening(trailingSilence: 1.8))
        var (events, time) = feed(&endpointer, level: 0.5, from: 0, for: 1.0)
        #expect(events == [.speechStarted])

        // Silence shorter than the trailing window — still his turn.
        (events, time) = feed(&endpointer, level: 0.02, from: time, for: 1.0)
        #expect(events.isEmpty)

        // He resumes; the silence clock resets.
        (events, time) = feed(&endpointer, level: 0.5, from: time, for: 0.4)
        #expect(events.isEmpty)
        #expect(endpointer.isInSpeech)

        // Now a full trailing window of silence closes it.
        (events, _) = feed(&endpointer, level: 0.02, from: time, for: 2.0)
        #expect(events == [.endOfSpeech])
        #expect(!endpointer.isInSpeech)
    }

    @Test func quietLevelsBelowThresholdNeverTrigger() {
        var endpointer = VoiceEndpointer(config: .listening(speechLevel: 0.22))
        let (events, _) = feed(&endpointer, level: 0.15, from: 0, for: 5.0)
        #expect(events.isEmpty)
    }

    @Test func bargeInProfileNeedsLongerCommitment() {
        var endpointer = VoiceEndpointer(config: .bargeIn())
        // 0.3 s of speech energy — under the 0.45 s barge debounce: rejected.
        var (events, time) = feed(&endpointer, level: 0.6, from: 0, for: 0.3)
        #expect(events.isEmpty)
        (events, time) = feed(&endpointer, level: 0.02, from: time, for: 0.5)
        #expect(events.isEmpty)

        // A deliberate interruption sustains: fires.
        (events, _) = feed(&endpointer, level: 0.6, from: time, for: 0.7)
        #expect(events == [.speechStarted])
    }

    @Test func voicedSecondsAccumulateOnlyLoudTime() {
        var endpointer = VoiceEndpointer(config: .listening())
        var (_, time) = feed(&endpointer, level: 0.5, from: 0, for: 1.0)
        (_, time) = feed(&endpointer, level: 0.02, from: time, for: 1.0)
        _ = feed(&endpointer, level: 0.5, from: time, for: 0.5)
        // ~1.5 s of loud samples; the silent second contributes nothing.
        #expect(endpointer.voicedSeconds > 1.2)
        #expect(endpointer.voicedSeconds < 1.8)
    }

    @Test func voicedSecondsResetWithTheWatch() {
        var endpointer = VoiceEndpointer(config: .listening())
        _ = feed(&endpointer, level: 0.5, from: 0, for: 1.0)
        #expect(endpointer.voicedSeconds > 0)
        endpointer.reset()
        #expect(endpointer.voicedSeconds == 0)
    }

    @Test func aStalledTickerCreditsNoPhantomSpeech() {
        var endpointer = VoiceEndpointer(config: .listening())
        _ = endpointer.ingest(level: 0.5, at: 0)
        // The ticker stalled for 5 s between loud samples — credit at most
        // the clamp, never the whole gap.
        _ = endpointer.ingest(level: 0.5, at: 5.0)
        #expect(endpointer.voicedSeconds <= 0.25)
    }

    // MARK: The Echo Floor input (ADR-0041)

    @Test func speechFloorRaisesTheEffectiveThreshold() {
        var endpointer = VoiceEndpointer(config: .bargeIn(speechLevel: 0.25))
        // Residual reading 0.5 under a floor of 0.6: never speech.
        var time: TimeInterval = 0
        while time <= 3.0 {
            let event = endpointer.ingest(level: 0.5, at: time, speechFloor: 0.6)
            #expect(event == nil)
            time += 0.05
        }
        #expect(!endpointer.isInSpeech)
        // Same level, floor lifted (playback ended): the owner fires.
        var events: [VoiceEndpointer.Event] = []
        while time <= 4.0 {
            if let event = endpointer.ingest(level: 0.5, at: time) { events.append(event) }
            time += 0.05
        }
        #expect(events == [.speechStarted])
    }

    @Test func levelsUnderTheFloorBankNoVoicedTime() {
        var endpointer = VoiceEndpointer(config: .bargeIn(speechLevel: 0.25))
        var time: TimeInterval = 0
        while time <= 2.0 {
            _ = endpointer.ingest(level: 0.5, at: time, speechFloor: 0.6)
            time += 0.05
        }
        #expect(endpointer.voicedSeconds == 0)
    }

    @Test func nilFloorKeepsTheStaticBehavior() {
        var withNil = VoiceEndpointer(config: .listening())
        var without = VoiceEndpointer(config: .listening())
        var time: TimeInterval = 0
        while time <= 1.0 {
            let a = withNil.ingest(level: 0.5, at: time, speechFloor: nil)
            let b = without.ingest(level: 0.5, at: time)
            #expect(a == b)
            time += 0.05
        }
        #expect(withNil.isInSpeech == without.isInSpeech)
        #expect(withNil.voicedSeconds == without.voicedSeconds)
    }

    @Test func resetClearsMidSpeechState() {
        var endpointer = VoiceEndpointer(config: .listening())
        _ = feed(&endpointer, level: 0.5, from: 0, for: 1.0)
        #expect(endpointer.isInSpeech)
        endpointer.reset(config: .bargeIn())
        #expect(!endpointer.isInSpeech)
        let (events, _) = feed(&endpointer, level: 0.02, from: 10, for: 3.0)
        #expect(events.isEmpty)
    }
}
