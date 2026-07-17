//
//  EchoResidualFloorTests.swift
//  tesseractTests
//
//  The Echo Floor (ADR-0041): the self-calibrating residual tracker that
//  keeps the barge threshold above the reply's own read-back at the mic.
//  Pinned pure, at ticker cadence: the echo-path-loss cap (owner speech
//  must not drag the floor up under its own onset), the fast attack, the
//  slow decay, the playback-loud trailing hold, and the never-below-static
//  threshold. Real-hardware traces are pinned in VoiceBargeReplayTests.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@Suite struct EchoResidualFloorTests {

    /// Feed constant mic + playback levels at 50 ms cadence.
    private func feed(
        _ floor: inout EchoResidualFloor, mic: Float, playback: Float,
        from start: TimeInterval, for duration: TimeInterval
    ) -> TimeInterval {
        var time = start
        while time <= start + duration {
            floor.ingest(micLevel: mic, playbackLevel: playback, at: time)
            time += 0.05
        }
        return time
    }

    @Test func floorConvergesOnResidualInsideTheDebounce() {
        var floor = EchoResidualFloor()
        // Sustained residual at 0.3 under a loud reply (0.9 envelope): the
        // fast attack (2.0/s) converges well inside the 0.45 s debounce, so
        // sustained echo can never outrun the threshold.
        _ = feed(&floor, mic: 0.3, playback: 0.9, from: 0, for: 0.4)
        #expect(floor.floor > 0.25)
        #expect(floor.floor <= 0.3)
    }

    @Test func pathLossCapStopsTheFloorUnderOwnerSpeech() {
        var floor = EchoResidualFloor()
        var time = feed(&floor, mic: 0.2, playback: 0.9, from: 0, for: 1.0)
        // The owner talks over the reply (mic 0.9, playback 0.9): the floor
        // may only believe playback − pathLoss (0.7) — the chase stops
        // there, and the owner's level stays above floor + margin.
        time = feed(&floor, mic: 0.9, playback: 0.9, from: time, for: 2.0)
        #expect(floor.floor <= 0.7)
        #expect(floor.threshold(atLeast: 0.25) < 0.9)
    }

    @Test func floorDecaysWhenPlaybackGoesQuiet() {
        var floor = EchoResidualFloor()
        var time = feed(&floor, mic: 0.3, playback: 0.9, from: 0, for: 1.0)
        let settled = floor.floor
        // Playback silent past the trailing hold: even with the mic still
        // reading (room noise), the floor decays — quiet-gap energy is not
        // residual and must not be banked as such.
        time = feed(&floor, mic: 0.3, playback: 0, from: time, for: 2.0)
        #expect(floor.floor < settled)
    }

    @Test func trailingHoldBridgesShortPlaybackGaps() {
        var floor = EchoResidualFloor()
        var time = feed(&floor, mic: 0.3, playback: 0.9, from: 0, for: 1.0)
        let settled = floor.floor
        // A 0.2 s dip in the envelope (inter-word gap, alignment skew) is
        // inside the 0.3 s hold: the slow in-playback decay applies, never
        // the fast quiet decay — the floor sheds a few hundredths at most.
        time = feed(&floor, mic: 0.3, playback: 0, from: time, for: 0.2)
        #expect(floor.floor >= settled - 0.05)
        // Past the hold the fast quiet decay takes over.
        let held = floor.floor
        _ = feed(&floor, mic: 0.3, playback: 0, from: time, for: 0.5)
        #expect(floor.floor < held - 0.2)
    }

    @Test func thresholdNeverFallsBelowTheStaticLevel() {
        var floor = EchoResidualFloor()
        // No playback heard yet: the static threshold stands alone.
        #expect(floor.threshold(atLeast: 0.25) == 0.25)
        // A quiet residual (0.05): floor + margin sits under the static
        // level, which still wins.
        _ = feed(&floor, mic: 0.05, playback: 0.9, from: 0, for: 1.0)
        #expect(floor.threshold(atLeast: 0.25) == 0.25)
    }

    @Test func thresholdRidesFloorPlusMargin() {
        var floor = EchoResidualFloor()
        _ = feed(&floor, mic: 0.3, playback: 0.9, from: 0, for: 1.0)
        let base = floor.threshold(atLeast: 0.25)
        #expect(base > 0.3)
        #expect(base <= 0.3 + 0.09)
        // Escalation widens the margin, never the floor itself.
        let escalated = floor.threshold(atLeast: 0.25, marginScale: 1.5)
        #expect(escalated > base)
    }

    @Test func resetForgetsEverything() {
        var floor = EchoResidualFloor()
        _ = feed(&floor, mic: 0.3, playback: 0.9, from: 0, for: 1.0)
        #expect(floor.floor > 0)
        floor.reset()
        #expect(floor.floor == 0)
        #expect(floor.threshold(atLeast: 0.25) == 0.25)
    }
}
