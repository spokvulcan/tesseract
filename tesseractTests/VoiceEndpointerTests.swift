//
//  VoiceEndpointerTests.swift
//  tesseractTests
//
//  The voice session's ears (#310): speech-start debounce, blip rejection,
//  and trailing-silence end-of-speech — pure state machine, driven with
//  synthetic level samples at meter cadence.
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
        // A long turn: past the patience ramp, the Setting's silence applies.
        var (events, time) = feed(&endpointer, level: 0.5, from: 0, for: 4.0)
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

    /// ADR-0097: a short turn closes after a second of silence; the close
    /// grows to the Setting's value as the turn keeps going.
    @Test func aShortTurnClosesSoonerThanALongOne() {
        var short = VoiceEndpointer(config: .listening(trailingSilence: 1.8))
        var (events, time) = feed(&short, level: 0.5, from: 0, for: 1.0)
        #expect(events == [.speechStarted])
        // 0.8 s of silence: still the owner's turn.
        (events, time) = feed(&short, level: 0.02, from: time, for: 0.75)
        #expect(events.isEmpty)
        // Closed by 1.25 s, well before the 1.8 s a long turn gets.
        (events, _) = feed(&short, level: 0.02, from: time, for: 0.4)
        #expect(events == [.endOfSpeech])

        var long = VoiceEndpointer(config: .listening(trailingSilence: 1.8))
        (events, time) = feed(&long, level: 0.5, from: 0, for: 4.0)
        (events, _) = feed(&long, level: 0.02, from: time, for: 1.6)
        #expect(events.isEmpty)
        #expect(long.isInSpeech)
    }

    @Test func theCloseGrowsAcrossTheRamp() {
        let config = VoiceEndpointer.Config.listening(trailingSilence: 1.8)
        #expect(config.closingSilence(afterSpeaking: 0.5) == 1.0)
        #expect(config.closingSilence(afterSpeaking: 1.5) == 1.0)
        #expect(abs(config.closingSilence(afterSpeaking: 2.5) - 1.4) < 1e-9)
        #expect(config.closingSilence(afterSpeaking: 3.5) == 1.8)
        #expect(config.closingSilence(afterSpeaking: 20) == 1.8)
        // A Setting under the short close is the close for every turn.
        let quick = VoiceEndpointer.Config.listening(trailingSilence: 0.8)
        #expect(quick.closingSilence(afterSpeaking: 0.5) == 0.8)
        #expect(quick.closingSilence(afterSpeaking: 10) == 0.8)
    }

    @Test func quietLevelsBelowThresholdNeverTrigger() {
        var endpointer = VoiceEndpointer(config: .listening(speechLevel: 0.22))
        let (events, _) = feed(&endpointer, level: 0.15, from: 0, for: 5.0)
        #expect(events.isEmpty)
    }

    @Test func resetClearsMidSpeechState() {
        var endpointer = VoiceEndpointer(config: .listening())
        _ = feed(&endpointer, level: 0.5, from: 0, for: 1.0)
        #expect(endpointer.isInSpeech)
        endpointer.reset(config: .listening())
        #expect(!endpointer.isInSpeech)
        let (events, _) = feed(&endpointer, level: 0.02, from: 10, for: 3.0)
        #expect(events.isEmpty)
    }
}
