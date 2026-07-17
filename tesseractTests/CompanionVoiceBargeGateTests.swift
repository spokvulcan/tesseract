//
//  CompanionVoiceBargeGateTests.swift
//  tesseractTests
//
//  The Soft Barge confirm window (ADR-0041) — the pure decision that turns
//  an energy onset's duck into a real pause, or fades the reply back.
//  Pinned as a decision table because a wrong verdict either interrupts the
//  reply on its own echo or swallows a real interruption. The former word
//  gates (Substance Gate, Session Directives) were removed 2026-07-18 by
//  owner decision: the barge decision is purely acoustic.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@Suite struct CompanionVoiceBargeGateTests {

    private typealias Gate = CompanionVoiceSessionController

    // Shipped constants: window 0.8 s, confirm at 0.3 s of voicing.

    @Test func sustainedVoicingConfirmsEarly() {
        // A real interruption should not wait out the window: the moment
        // voicing accumulates, the duck becomes the pause.
        #expect(
            Gate.resolveSoftBarge(
                voicedSeconds: 0.3, elapsed: 0.35, confirmWindow: 0.8, confirmVoiced: 0.3)
                == .confirm)
    }

    @Test func silenceInsideTheWindowKeepsWaiting() {
        #expect(
            Gate.resolveSoftBarge(
                voicedSeconds: 0.1, elapsed: 0.4, confirmWindow: 0.8, confirmVoiced: 0.3)
                == .keepWaiting)
        #expect(
            Gate.resolveSoftBarge(
                voicedSeconds: 0, elapsed: 0.79, confirmWindow: 0.8, confirmVoiced: 0.3)
                == .keepWaiting)
    }

    @Test func aClosedWindowWithoutVoicingFadesBack() {
        // The false-fire cost: a ~1 s murmur, then the reply comes back —
        // never the 2–3 s dead pause of the 2026-07-17 flap storms.
        #expect(
            Gate.resolveSoftBarge(
                voicedSeconds: 0.29, elapsed: 0.8, confirmWindow: 0.8, confirmVoiced: 0.3)
                == .fadeBack)
        #expect(
            Gate.resolveSoftBarge(
                voicedSeconds: 0, elapsed: 2.0, confirmWindow: 0.8, confirmVoiced: 0.3)
                == .fadeBack)
    }

    @Test func voicingAtTheWindowEdgeStillConfirms() {
        #expect(
            Gate.resolveSoftBarge(
                voicedSeconds: 0.3, elapsed: 0.8, confirmWindow: 0.8, confirmVoiced: 0.3)
                == .confirm)
    }
}
