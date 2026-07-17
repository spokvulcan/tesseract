//
//  VoiceBargeReplayTests.swift
//  tesseractTests
//
//  The calibration lock (ADR-0041): real-hardware traces recorded by
//  tools/voice-hold-lab replayed through the REAL detector — the shipped
//  `VoiceEndpointer` + `EchoResidualFloor` constants — mirroring the
//  session controller's `.speaking` tick. The zero-false-positive goal is
//  pinned here: a clean reply trace must produce ZERO energy onsets; a
//  sustained owner-level excursion must still fire inside the barge
//  debounce. Retune any constant and these tests say what it does to the
//  field traces. Regenerate fixtures: tools/voice-hold-lab/RUNBOOK.md.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@Suite struct VoiceBargeReplayTests {

    /// The session's static barge threshold (the shipped Settings default).
    private let staticLevel: Float = 0.25

    /// Mirrors `CompanionVoiceSessionController.tick()` in `.speaking`:
    /// floor ingest, then endpointer ingest with the floor threshold. On an
    /// onset the endpointer re-arms (as the session does around a barge) so
    /// every distinct onset in the trace is counted.
    private func replay(mic: [Float], playback: [Float]) -> (
        onsets: [TimeInterval], floorPeak: Float
    ) {
        var endpointer = VoiceEndpointer(config: .bargeIn(speechLevel: staticLevel))
        var floor = EchoResidualFloor()
        var onsets: [TimeInterval] = []
        var floorPeak: Float = 0
        for (index, level) in mic.enumerated() {
            let time = Double(index) * 0.05
            let playbackLevel = index < playback.count ? playback[index] : 0
            floor.ingest(micLevel: level, playbackLevel: playbackLevel, at: time)
            floorPeak = max(floorPeak, floor.floor)
            let event = endpointer.ingest(
                level: level, at: time,
                speechFloor: floor.threshold(atLeast: staticLevel))
            if event == .speechStarted {
                onsets.append(time)
                endpointer.reset(config: .bargeIn(speechLevel: staticLevel))
            }
        }
        return (onsets, floorPeak)
    }

    // MARK: Zero false onsets on clean replies (the owner's hard goal)

    @Test func hostedReplyTraceNeverFires() {
        let result = replay(
            mic: VoiceLabFixtures.hostedReply,
            playback: VoiceLabFixtures.e2SignalEnvelope)
        #expect(result.onsets.isEmpty)
    }

    @Test func dedicatedMinTraceNeverFires() {
        let result = replay(
            mic: VoiceLabFixtures.dedicatedMin,
            playback: VoiceLabFixtures.e2SignalEnvelope)
        #expect(result.onsets.isEmpty)
    }

    @Test func dedicatedDefaultTraceNeverFires() {
        let result = replay(
            mic: VoiceLabFixtures.dedicatedDefault,
            playback: VoiceLabFixtures.e2SignalEnvelope)
        #expect(result.onsets.isEmpty)
    }

    @Test func duckAndResumeTransientsNeverFire() {
        let result = replay(
            mic: VoiceLabFixtures.resumeTransient,
            playback: VoiceLabFixtures.e5SignalEnvelope)
        #expect(result.onsets.isEmpty)
    }

    @Test func roomNoiseAloneNeverFires() {
        // No playback at all: the floor stays zero and the static gate
        // stands — the room's noise floor must sit under it.
        let result = replay(
            mic: VoiceLabFixtures.noiseFloor,
            playback: [Float](repeating: 0, count: VoiceLabFixtures.noiseFloor.count))
        #expect(result.onsets.isEmpty)
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
        let spliceTime = Double(spliceStart) * 0.05
        let fired = result.onsets.first { $0 >= spliceTime }
        #expect(fired != nil)
        if let fired {
            #expect(fired - spliceTime <= 0.6)
        }
        // And nothing before the splice — the clean prefix stays clean.
        #expect(result.onsets.allSatisfy { $0 >= spliceTime })
    }
}
