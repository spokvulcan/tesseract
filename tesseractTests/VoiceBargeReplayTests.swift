//
//  VoiceBargeReplayTests.swift
//  tesseractTests
//
//  The calibration lock (ADR-0041): real-hardware traces recorded by
//  tools/voice-hold-lab replayed through the SHIPPED path — the Voice
//  Session Machine in `.speaking`, driving the real `VoiceEndpointer` +
//  `EchoResidualFloor` (ADR-0042 ended the hand-copied mirror of the
//  `.speaking` tick this suite used to carry). The zero-false-positive goal
//  is pinned here: a clean reply trace must produce ZERO energy onsets; a
//  sustained owner-level excursion must still fire inside the barge
//  debounce. Retune any constant and these tests say what it does to the
//  field traces. Regenerate fixtures: tools/voice-hold-lab/RUNBOOK.md.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@Suite struct VoiceBargeReplayTests {

    /// Replays a lab trace through the machine while it speaks: floor
    /// ingest → threshold → endpointer → soft-barge reaction, exactly as
    /// shipped. An onset is the machine's own `voice.barge-soft-onset`
    /// record; the harness's tunables carry the shipped static threshold
    /// (0.25, the Settings default).
    private func replay(mic: [Float], playback: [Float]) -> (
        onsetIndices: [Int], floorPeak: Float
    ) {
        var harness = VoiceSessionMachineHarness()
        harness.startSpeaking()
        var onsetIndices: [Int] = []
        var floorPeak: Float = 0
        for (index, level) in mic.enumerated() {
            let playbackLevel = index < playback.count ? playback[index] : 0
            let effects = harness.tick(level: level, playback: playbackLevel)
            floorPeak = max(floorPeak, harness.machine.echoFloorLevel)
            if effects.contains(record: "voice.barge-soft-onset") {
                onsetIndices.append(index)
            }
        }
        return (onsetIndices, floorPeak)
    }

    // MARK: Zero false onsets on clean replies (the owner's hard goal)

    @Test func hostedReplyTraceNeverFires() {
        let result = replay(
            mic: VoiceLabFixtures.hostedReply,
            playback: VoiceLabFixtures.e2SignalEnvelope)
        #expect(result.onsetIndices.isEmpty)
    }

    @Test func dedicatedMinTraceNeverFires() {
        let result = replay(
            mic: VoiceLabFixtures.dedicatedMin,
            playback: VoiceLabFixtures.e2SignalEnvelope)
        #expect(result.onsetIndices.isEmpty)
    }

    @Test func dedicatedDefaultTraceNeverFires() {
        let result = replay(
            mic: VoiceLabFixtures.dedicatedDefault,
            playback: VoiceLabFixtures.e2SignalEnvelope)
        #expect(result.onsetIndices.isEmpty)
    }

    @Test func duckAndResumeTransientsNeverFire() {
        let result = replay(
            mic: VoiceLabFixtures.resumeTransient,
            playback: VoiceLabFixtures.e5SignalEnvelope)
        #expect(result.onsetIndices.isEmpty)
    }

    @Test func roomNoiseAloneNeverFires() {
        // No playback at all: the floor stays zero and the static gate
        // stands — the room's noise floor must sit under it.
        let result = replay(
            mic: VoiceLabFixtures.noiseFloor,
            playback: [Float](repeating: 0, count: VoiceLabFixtures.noiseFloor.count))
        #expect(result.onsetIndices.isEmpty)
        #expect(result.floorPeak == 0)
    }

    // MARK: A real interruption still fires

    @Test func sustainedOwnerLevelSplicedIntoTheReplyFiresInsideTheDebounce() {
        // The owner talks over the reply: a sustained near-close-mic
        // excursion (0.75 for 1.5 s) laid over the recorded residual, in
        // the speech-shaped region of the trace. The path-loss cap keeps
        // the threshold below it, and the 0.45 s debounce fires within
        // 600 ms of the excursion's start.
        var mic = VoiceLabFixtures.hostedReply
        let spliceStart = 80  // t = 4.0 s
        let spliceEnd = min(spliceStart + 30, mic.count)
        for index in spliceStart..<spliceEnd {
            mic[index] = max(mic[index], 0.75)
        }
        let result = replay(mic: mic, playback: VoiceLabFixtures.e2SignalEnvelope)
        let fired = result.onsetIndices.first { $0 >= spliceStart }
        #expect(fired != nil)
        if let fired {
            #expect(Double(fired - spliceStart) * 0.05 <= 0.6)
        }
        // And nothing before the splice — the clean prefix stays clean.
        #expect(result.onsetIndices.allSatisfy { $0 >= spliceStart })
    }
}
