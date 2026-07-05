//
//  OnboardingTryItTests.swift
//  tesseractTests
//
//  Try-it availability (PRD #171): a Chapter's live demo slot activates only
//  when its preconditions are met — model on disk, permission granted — and
//  otherwise the chapter shows its scripted animation. Pure, ModelCatalog
//  style; prior art: `OnboardingModelPickTests`.
//

import Testing

@testable import Tesseract_Agent

@MainActor
struct OnboardingTryItTests {

    @Test func dictationIsLiveOnlyWithMicGrantedAndSpeechModelOnDisk() {
        #expect(
            OnboardingTryIt.dictationIsLive(microphone: .granted, speechModelDownloaded: true))
        #expect(
            !OnboardingTryIt.dictationIsLive(microphone: .granted, speechModelDownloaded: false))
        #expect(
            !OnboardingTryIt.dictationIsLive(microphone: .denied, speechModelDownloaded: true))

        for state in [PermissionState.unknown, .requesting, .denied, .restricted] {
            #expect(
                !OnboardingTryIt.dictationIsLive(microphone: state, speechModelDownloaded: true),
                "\(state) must not activate the dictation Try-it")
        }
    }

    @Test func voiceIsLiveOnceTheVoiceModelIsOnDisk() {
        #expect(OnboardingTryIt.voiceIsLive(voiceModelDownloaded: true))
        #expect(!OnboardingTryIt.voiceIsLive(voiceModelDownloaded: false))
    }
}
