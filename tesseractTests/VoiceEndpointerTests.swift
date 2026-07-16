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
