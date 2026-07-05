//
//  SettingsCatalogueTests.swift
//  tesseractTests
//
//  The Settings Catalogue is the single source of truth for each setting's
//  default. These pin the values that the old triplicated design got wrong or
//  put at risk: the SSD-budget drift bug, and the unset true-defaults that
//  removing `register(defaults:)` could silently flip to `false`.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct SettingsCatalogueTests {

    @Test
    func budgetCapsDefaultToAutomatic() {
        // ADR-0018: both prefix-cache budgets are measured; the user
        // settings are caps whose default is "Automatic" (nil). The old
        // fixed 20 GiB SSD budget survives only as
        // `SSDBudgetPolicy.floorBytes` (the drift bug of issue #16 stays
        // single-sourced there).
        let store = InMemorySettingsStore()
        #expect(SettingsCatalogue.prefixCacheRAMBudgetCapBytes.default == nil)
        #expect(SettingsCatalogue.prefixCacheRAMBudgetCapBytes.load(from: store) == nil)
        #expect(SettingsCatalogue.prefixCacheSSDBudgetCapBytes.default == nil)
        #expect(SettingsCatalogue.prefixCacheSSDBudgetCapBytes.load(from: store) == nil)
        #expect(SSDBudgetPolicy.floorBytes == 20 * 1024 * 1024 * 1024)
    }

    @Test
    func freshStoreYieldsCorrectTrueDefaults() {
        // Removing `register(defaults:)` must not silently flip these unset
        // true-defaults to `false` (a fresh install keeps dock visible, SSD
        // cache on, web access on, TTS streaming on, …). Each reads its
        // catalogue default through default-on-read.
        let store = InMemorySettingsStore()
        #expect(SettingsCatalogue.showInDock.load(from: store) == true)
        #expect(SettingsCatalogue.showInMenuBar.load(from: store) == true)
        #expect(SettingsCatalogue.autoInsertText.load(from: store) == true)
        #expect(SettingsCatalogue.restoreClipboard.load(from: store) == true)
        #expect(SettingsCatalogue.ttsStreamingEnabled.load(from: store) == true)
        #expect(SettingsCatalogue.playSounds.load(from: store) == true)
        #expect(SettingsCatalogue.webAccessEnabled.load(from: store) == true)
        #expect(SettingsCatalogue.prefixCacheSSDEnabled.load(from: store) == true)
        // Vision-by-default (ADR-0013, PRD #112): the global opt-out defaults
        // on, so vision-capable models load their image-aware container from a
        // fresh install.
        #expect(SettingsCatalogue.useVisionWhenAvailable.load(from: store) == true)
        // Skill Pills (PRD #174): the row shows by default on a fresh install.
        #expect(SettingsCatalogue.showSkillPills.load(from: store) == true)
    }

    @Test
    func translateTargetLanguageDefaultsFromSystemPreferredLanguages() {
        // The catalogue default is the launch-time derivation from the macOS
        // preferred languages — non-empty, and exactly what the pure helper
        // produces for this process (PRD #174).
        let store = InMemorySettingsStore()
        let expected = TranslateLanguageDefault.derive(from: Locale.preferredLanguages)
        #expect(!expected.isEmpty)
        #expect(SettingsCatalogue.translateTargetLanguage.load(from: store) == expected)
    }

    @Test
    func skillUsageCountsDefaultToZeroAndResetSweepsThem() {
        // Dynamic per-skill keys mint on demand (default 0) and share a prefix
        // so resetToDefaults can sweep them back to the curated pill order.
        let store = InMemorySettingsStore()
        #expect(SettingsCatalogue.skillUsageCount(skillName: "proofread").load(from: store) == 0)

        let settings = SettingsManager(store: store)
        settings.incrementSkillUsage(skillName: "proofread")
        settings.incrementSkillUsage(skillName: "proofread")
        #expect(settings.skillUsageCount(skillName: "proofread") == 2)

        settings.resetToDefaults()
        #expect(settings.skillUsageCount(skillName: "proofread") == 0)
        #expect(settings.showSkillPills == true)
    }
}

// MARK: - TranslateLanguageDefault

struct TranslateLanguageDefaultTests {

    @Test
    func firstNonEnglishPreferredLanguageWins() {
        #expect(TranslateLanguageDefault.derive(from: ["uk-UA", "en-US"]) == "Ukrainian")
        #expect(TranslateLanguageDefault.derive(from: ["en-US", "de-DE", "fr-FR"]) == "German")
    }

    @Test
    func allEnglishFallsBackToEnglish() {
        // Target = English collapses the translate direction flip (PRD #174):
        // everything simply translates to English.
        #expect(TranslateLanguageDefault.derive(from: ["en-US", "en-GB"]) == "English")
        #expect(TranslateLanguageDefault.derive(from: []) == "English")
    }
}
