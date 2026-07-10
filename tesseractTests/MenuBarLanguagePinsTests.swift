//
//  MenuBarLanguagePinsTests.swift
//  tesseractTests
//
//  The status-bar menu's pinned language entries are pure derivations
//  (`MenuBarLanguagePins`) plus one small piece of settings state (the
//  recents list). These pin the ordering, deduplication, filtering, and
//  caps — the menu rendering itself is dumb over these values.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct MenuBarLanguagePinsTests {

    // MARK: - Dictation pins

    @Test
    func currentComesFirstThenRecentsThenSystemPreferred() {
        let pins = MenuBarLanguagePins.pinnedCodes(
            current: "de",
            recents: ["uk", "fr"],
            preferredLanguageIdentifiers: ["en-US", "ja-JP"]
        )
        #expect(pins == ["de", "uk", "fr", "en", "ja"])
    }

    @Test
    func duplicatesCollapseToTheirFirstMention() {
        let pins = MenuBarLanguagePins.pinnedCodes(
            current: "en",
            recents: ["uk", "en"],
            preferredLanguageIdentifiers: ["uk-UA", "en-GB"]
        )
        #expect(pins == ["en", "uk"])
    }

    @Test
    func autoAndUnknownCodesNeverPin() {
        let pins = MenuBarLanguagePins.pinnedCodes(
            current: "auto",
            recents: ["xx", "uk"],
            preferredLanguageIdentifiers: ["en-US"]
        )
        #expect(pins == ["uk", "en"])
    }

    @Test
    func pinsCapAtSix() {
        let pins = MenuBarLanguagePins.pinnedCodes(
            current: "en",
            recents: ["uk", "de", "fr", "es", "it", "pl", "nl"],
            preferredLanguageIdentifiers: []
        )
        #expect(pins.count == 6)
        #expect(pins == ["en", "uk", "de", "fr", "es", "it"])
    }

    // MARK: - Translate pins

    @Test
    func translatePinsMapPreferredIdentifiersToCatalogueNames() {
        let options = SupportedLanguage.translateTargetOptions(current: "English")
        let pins = MenuBarLanguagePins.pinnedTranslateNames(
            current: "Ukrainian",
            preferredLanguageIdentifiers: ["uk-UA", "de-DE"],
            options: options
        )
        #expect(pins == ["Ukrainian", "German", "English"])
    }

    @Test
    func translatePinsDropNamesOutsideTheOptionList() {
        let pins = MenuBarLanguagePins.pinnedTranslateNames(
            current: "Klingon",
            preferredLanguageIdentifiers: ["en-US"],
            options: ["English", "Ukrainian"]
        )
        #expect(pins == ["English"])
    }

    // MARK: - Shared translate option list

    @Test
    func translateTargetOptionsExcludeAutoDetectAndKeepTheCurrentValue() {
        let options = SupportedLanguage.translateTargetOptions(current: "Klingon")
        #expect(!options.contains("Auto-detect"))
        #expect(options.contains("Klingon"))
        #expect(options == options.sorted())
    }

    // MARK: - Recents recording

    @Test
    @MainActor
    func recordingRecentsMovesRepeatsToTheFrontAndCapsAtFive() {
        let settings = SettingsManager(store: InMemorySettingsStore())
        for code in ["uk", "de", "fr", "es", "it", "pl"] {
            settings.recordRecentDictationLanguage(code)
        }
        #expect(settings.recentDictationLanguages == "pl,it,es,fr,de")

        settings.recordRecentDictationLanguage("fr")
        #expect(settings.recentDictationLanguages == "fr,pl,it,es,de")
    }

    @Test
    @MainActor
    func autoIsNeverRecordedAsARecent() {
        let settings = SettingsManager(store: InMemorySettingsStore())
        settings.recordRecentDictationLanguage("auto")
        #expect(settings.recentDictationLanguages.isEmpty)
    }
}
