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
    func ssdBudgetDefaultIsSingleSourcedAtTwentyGiB() {
        // The motivating drift bug: 50 GiB in the property literal/doc vs 20 GiB
        // in register/reset, with 20 GiB winning at runtime. Now declared once.
        let store = InMemorySettingsStore()
        #expect(SettingsCatalogue.prefixCacheSSDBudgetBytes.default == 20 * 1024 * 1024 * 1024)
        #expect(
            SettingsCatalogue.prefixCacheSSDBudgetBytes.load(from: store) == 20 * 1024 * 1024 * 1024
        )
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
    }
}
