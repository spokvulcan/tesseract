//
//  SpeedCheckTests.swift
//  tesseractTests
//
//  The Speed Check (#515): a measured real-time factor in; whether the
//  neural voice reads, and which of the speed menu's rates it can read at.
//

import Testing

@testable import Tesseract_Agent

struct SpeedCheckTests {

    /// Fast enough for every rate: the whole menu.
    @Test func aFastVoiceKeepsTheWholeMenu() {
        let check = SpeedCheck(realTimeFactor: 0.3, firstAudio: 0.2)
        #expect(check.keepsUpAtNormalSpeed)
        #expect(check.playableRates() == PlaybackRate.menu)
        #expect(check.fastestRate() == 2.0)
        #expect(check.verdict == nil)
    }

    /// The budget's own speed (RTF 0.67 keeps 1.5×) with headroom: capped
    /// below 1.5×, every slower rate kept.
    @Test func aVoiceAtTheBudgetIsCappedWithHeadroom() {
        let check = SpeedCheck(realTimeFactor: 0.6, firstAudio: 0.3)
        // 0.6 × 1.25 × 1.2 = 0.9 keeps 1.25×; 0.6 × 1.5 × 1.2 = 1.08 doesn't.
        #expect(check.fastestRate() == 1.25)
        #expect(check.playableRates() == [0.75, 0.9, 1.0, 1.15, 1.25])
    }

    /// Too slow for 1×: the system voice reads, and the owner is told why.
    @Test func aSlowVoiceHandsOverToTheSystemVoice() {
        let check = SpeedCheck(realTimeFactor: 0.9, firstAudio: 0.5)
        #expect(!check.keepsUpAtNormalSpeed)
        #expect(check.fastestRate() == nil)
        #expect(check.verdict?.contains("system voice") == true)
    }
}
