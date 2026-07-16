//
//  CompanionVoiceBargeGateTests.swift
//  tesseractTests
//
//  The Substance Gate + Session Directive resolution for a take captured
//  under a barged (paused) reply — the pure decision that separates a real
//  interruption from echo residual, a thump, or a control word. Pinned as a
//  decision table because a wrong verdict either feeds the agent garbage
//  (the 2026-07-16 4-char self-echo turn) or swallows a real turn.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@Suite struct CompanionVoiceBargeGateTests {

    private typealias Gate = CompanionVoiceSessionController

    // MARK: Session Directives — playback control, never sent to the agent

    @Test func terseControlWordsResolveAsDirectives() {
        #expect(Gate.resolveBargeTake(text: "Stop.", voicedSeconds: 0.3) == .directive("stop"))
        #expect(Gate.resolveBargeTake(text: "wait", voicedSeconds: 0.2) == .directive("wait"))
        #expect(Gate.resolveBargeTake(text: "Pause!", voicedSeconds: 0.4) == .directive("pause"))
        #expect(Gate.resolveBargeTake(text: "quiet", voicedSeconds: 0.5) == .directive("quiet"))
    }

    @Test func directivesWinRegardlessOfVoicedTime() {
        // Even a long-held "stooooop" that measured well over the gate is a
        // directive, not a turn.
        #expect(Gate.resolveBargeTake(text: "stop", voicedSeconds: 2.0) == .directive("stop"))
    }

    @Test func contentAmbiguousWordsAreNotDirectives() {
        // "no" / "yes" / "okay" are answers; they stay off the allowlist —
        // and alone they lack substance, so a paused reply resumes.
        #expect(Gate.resolveBargeTake(text: "No.", voicedSeconds: 1.0) == .falseBarge)
        #expect(Gate.resolveBargeTake(text: "yes", voicedSeconds: 1.0) == .falseBarge)
        #expect(Gate.resolveBargeTake(text: "okay", voicedSeconds: 1.0) == .falseBarge)
    }

    @Test func directiveInsideASentenceIsNotADirective() {
        #expect(
            Gate.resolveBargeTake(text: "please stop talking", voicedSeconds: 1.0) == .turn)
    }

    // MARK: The Substance Gate

    @Test func sustainedMultiWordSpeechIsATurn() {
        #expect(
            Gate.resolveBargeTake(
                text: "what about the second option", voicedSeconds: 1.6) == .turn)
        #expect(Gate.resolveBargeTake(text: "hang on", voicedSeconds: 0.6) == .turn)
    }

    @Test func aShortScrapBelowTheGateIsFalse() {
        // The recorded failure class: a 4-char scrap after an energy barge.
        #expect(Gate.resolveBargeTake(text: "Yeah", voicedSeconds: 0.5) == .falseBarge)
        #expect(Gate.resolveBargeTake(text: "the", voicedSeconds: 3.0) == .falseBarge)
    }

    @Test func multiWordTextWithoutVoicedEnergyIsFalse() {
        // Whisper hallucinating a sentence out of near-silence: words alone
        // don't pass — the energy has to have been there too.
        #expect(
            Gate.resolveBargeTake(text: "thank you for watching", voicedSeconds: 0.2)
                == .falseBarge)
    }

    @Test func emptyAndPunctuationOnlyTakesAreFalse() {
        #expect(Gate.resolveBargeTake(text: "", voicedSeconds: 2.0) == .falseBarge)
        #expect(Gate.resolveBargeTake(text: "…", voicedSeconds: 2.0) == .falseBarge)
    }

    @Test func normalizationIsCaseAndPunctuationBlind() {
        #expect(Gate.resolveBargeTake(text: "  STOP!!  ", voicedSeconds: 0.3) == .directive("stop"))
    }
}
