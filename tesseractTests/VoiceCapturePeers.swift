//
//  VoiceCapturePeers.swift
//  tesseractTests
//
//  **Voice Capture** for the tests: the module built over the peers that sit
//  below it — a fake microphone, a recognizer peer, a recording Capture Dump,
//  Learned Words that know nothing, and a Proofread Pass that skips — each
//  replaceable by the stage a test exercises. Production builds the module
//  once in the composition root with every stage; so does every test.
//

import Foundation

@testable import Tesseract_Agent

/// Records what a session hands to the **Capture Dump**.
@MainActor
final class FakeCaptureDump: CaptureDumpStoring {
    private(set) var saved: [RawCapture] = []

    @discardableResult
    func save(_ capture: RawCapture) -> String? {
        saved.append(capture)
        return "capture-\(saved.count).wav"
    }
    func deleteAll() {}
}

/// Learned Words that know no word: every take reads as cleaned.
@MainActor
final class NoLearnedWords: LearnedWordApplying {
    func apply(to text: String, appBundleID: String?) -> LearnedWordsApplication {
        .unchanged(text)
    }
    func recordCatches(_ catches: [LearnedWordCatch]) {}
}

/// The module over the given stages; the rest stay out of the way.
@MainActor
func makeVoiceCapture(
    microphone: any AudioCapturing,
    recognizer: any Transcribing,
    captureDump: any CaptureDumpStoring = FakeCaptureDump(),
    isCaptureDumpEnabled: @escaping @MainActor () -> Bool = { true },
    learnedWords: any LearnedWordApplying = NoLearnedWords(),
    proofread: @escaping @MainActor (String) async -> ProofreadVerdict? = { _ in nil },
    frontmostApp: @escaping @MainActor () -> TargetApp? = { nil }
) -> VoiceCapture {
    VoiceCapture(
        microphone: microphone, recognizer: recognizer, captureDump: captureDump,
        isCaptureDumpEnabled: isCaptureDumpEnabled, learnedWords: learnedWords,
        proofread: proofread, language: { "en" }, frontmostApp: frontmostApp)
}
