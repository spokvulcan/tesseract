//
//  PhoneSettingsTests.swift
//  tesseractTests
//
//  The phone's Settings Facade follows the Mac's rules (ADR-0002): loading
//  writes nothing, a change persists, and every default is the catalogue's.
//

import Foundation
import Testing
import TesseractSpeech

@testable import Tesseract_Agent

@MainActor
struct PhoneSettingsTests {

    @Test func loadingWritesNothing() {
        let store = InMemorySettingsStore()
        _ = PhoneSettings(store: store)
        #expect(store.writes.isEmpty)
    }

    @Test func aFreshPhoneReadsTheCataloguesDefaults() {
        let settings = PhoneSettings(store: InMemorySettingsStore())
        #expect(settings.phoneVoice == SettingsCatalogue.phoneVoice.default)
        #expect(PresetVoice.named(settings.phoneVoice) != nil, "the default is one of the speakers")
        #expect(settings.ttsPlaybackRate == 1.0)
        #expect(settings.ttsParameters == TTSParameters())
        #expect(settings.readerHighlight == .both)
    }

    @Test func aChangeSurvivesARelaunch() {
        let store = InMemorySettingsStore()
        let settings = PhoneSettings(store: store)
        settings.phoneVoice = "aiden"
        settings.ttsPlaybackRate = 1.25
        #expect(store.writes == ["phoneVoice", "ttsPlaybackRate"])

        let relaunched = PhoneSettings(store: store)
        #expect(relaunched.phoneVoice == "aiden")
        #expect(relaunched.ttsPlaybackRate == 1.25)
    }

    /// The Mac and the phone persist the speech settings they share under
    /// the same keys.
    @Test func theSharedSpeechSettingsUseTheMacsKeys() {
        let store = InMemorySettingsStore()
        let mac = SettingsManager(store: store)
        mac.ttsPlaybackRate = 1.5
        #expect(PhoneSettings(store: store).ttsPlaybackRate == 1.5)
    }
}
