//
//  CompanionVoiceNativeTurnTests.swift
//  tesseractTests
//
//  The Native Audio Turn decision (ADR-0042) — the pure resolution that
//  sends a closed take to the model as audio, falls back to the ASR path,
//  or abandons it as acoustically empty. Pinned as a decision table like
//  the Soft Barge: a wrong verdict either feeds the model a cough or
//  silently strands a spoken turn.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@Suite struct CompanionVoiceNativeTurnTests {

    private typealias Controller = CompanionVoiceSessionController

    // Shipped constants: 30 s clip window, 0.35 s voicing floor.

    private func resolve(
        enabled: Bool = true, modelAvailable: Bool = true, autoSend: Bool = true,
        duration: TimeInterval = 3.0, voicedSeconds: TimeInterval = 1.2
    ) -> Controller.NativeTurnDecision {
        Controller.resolveNativeTurn(
            enabled: enabled, modelAvailable: modelAvailable, autoSend: autoSend,
            duration: duration, voicedSeconds: voicedSeconds,
            maxDuration: 30, minVoiced: 0.35)
    }

    @Test func aVoicedTakeGoesNative() {
        #expect(resolve() == .sendNative)
    }

    @Test func theToggleOffKeepsTheProvenPath() {
        #expect(resolve(enabled: false) == .transcribe)
    }

    @Test func aDeafModelKeepsTheProvenPath() {
        #expect(resolve(modelAvailable: false) == .transcribe)
    }

    /// Staging (Auto-Send off) is inherently text — the composer cannot
    /// hold audio, so the take transcribes.
    @Test func stagingFallsBackToTranscription() {
        #expect(resolve(autoSend: false) == .transcribe)
    }

    /// The model's audio window is 750 tokens × 40 ms = 30 s; a longer take
    /// would be truncated silently, so it falls back to the ASR path whole.
    @Test func overlongTakesFallBackToTranscription() {
        #expect(resolve(duration: 30.1) == .transcribe)
        #expect(resolve(duration: 30.0) == .sendNative)
    }

    /// The acoustic empty-take gate: under the voicing floor there is no
    /// turn — the barged reply resumes, nothing is sent (purely acoustic,
    /// like every barge decision since ADR-0041).
    @Test func underTheVoicingFloorIsNoTurn() {
        #expect(resolve(voicedSeconds: 0.34) == .abandon)
        #expect(resolve(voicedSeconds: 0.35) == .sendNative)
    }

    /// Gate order: an unavailable native path never abandons — the ASR
    /// path's transcript gate owns emptiness there.
    @Test func unavailabilityWinsOverEmptiness() {
        #expect(resolve(enabled: false, voicedSeconds: 0.1) == .transcribe)
        #expect(resolve(duration: 31, voicedSeconds: 0.1) == .transcribe)
    }
}
