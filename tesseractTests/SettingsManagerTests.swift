//
//  SettingsManagerTests.swift
//  tesseractTests
//
//  The Settings Facade, exercised through the in-memory adapter — no global
//  state, fully hermetic. Asserts observable behaviour through the public
//  interface (values read back, values surviving a simulated relaunch, keys
//  removed), never the private property layout or which store method ran.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct SettingsManagerTests {

    // MARK: - Default-on-read through the facade

    @Test
    func freshManagerReadsCatalogueDefaults() {
        // Pins default-on-read at the seam (story 12) and that removing
        // `register(defaults:)` keeps a fresh install's true-defaults on
        // (story 25).
        let settings = SettingsManager(store: InMemorySettingsStore())
        #expect(settings.showInDock == true)
        #expect(settings.webAccessEnabled == true)
        #expect(settings.prefixCacheSSDEnabled == true)
        #expect(settings.ttsStreamingEnabled == true)
        #expect(settings.playSounds == true)
        #expect(settings.prefixCacheSSDBudgetBytes == 20 * 1024 * 1024 * 1024)
        #expect(settings.prefixCacheSSDDirectoryOverride == nil)
        #expect(settings.selectedAgentModelID == ModelDefinition.defaultAgentModelID)
    }

    // MARK: - Persistence across a simulated relaunch

    @Test
    func flippingPropertyPersistsAndSurvivesRelaunch() {
        let store = InMemorySettingsStore()
        let first = SettingsManager(store: store)
        first.playSounds = false
        first.serverPort = 9000
        first.prefixCacheSSDBudgetBytes = 12 * 1024 * 1024 * 1024
        first.prefixCacheSSDDirectoryOverride = "/tmp/roundtrip"

        // A fresh facade on the same store is the relaunch.
        let second = SettingsManager(store: store)
        #expect(second.playSounds == false)
        #expect(second.serverPort == 9000)
        #expect(second.prefixCacheSSDBudgetBytes == 12 * 1024 * 1024 * 1024)
        #expect(second.prefixCacheSSDDirectoryOverride == "/tmp/roundtrip")
    }

    @Test
    func clearingOptionalSettingRemovesItAcrossRelaunch() {
        let store = InMemorySettingsStore()
        let first = SettingsManager(store: store)
        first.prefixCacheSSDDirectoryOverride = "/tmp/x"
        first.prefixCacheSSDDirectoryOverride = nil

        let second = SettingsManager(store: store)
        #expect(second.prefixCacheSSDDirectoryOverride == nil)
    }

    // MARK: - Reset

    @Test
    func resetToDefaultsRestoresCatalogueDefaultsAndPersists() {
        let store = InMemorySettingsStore()
        let settings = SettingsManager(store: store)
        settings.showInDock = false
        settings.serverPort = 9000
        settings.prefixCacheSSDBudgetBytes = 1
        settings.prefixCacheSSDDirectoryOverride = "/tmp/x"
        settings.triattentionEnabled = true

        settings.resetToDefaults()
        #expect(settings.showInDock == true)
        #expect(settings.serverPort == 8321)
        #expect(settings.prefixCacheSSDBudgetBytes == 20 * 1024 * 1024 * 1024)
        #expect(settings.prefixCacheSSDDirectoryOverride == nil)
        #expect(settings.triattentionEnabled == false)

        // Reset persists: a relaunch on the same store sees the defaults, never
        // the stale values (i.e. reset wrote through the store).
        let relaunched = SettingsManager(store: store)
        #expect(relaunched.showInDock == true)
        #expect(relaunched.serverPort == 8321)
        #expect(relaunched.prefixCacheSSDBudgetBytes == 20 * 1024 * 1024 * 1024)
        #expect(relaunched.prefixCacheSSDDirectoryOverride == nil)
        #expect(relaunched.triattentionEnabled == false)
    }

    @Test
    func resetToDefaultsReFiresThroughStore() {
        // Reset runs *after* init, so each assignment fires `didSet` and writes
        // through the store (and re-applies side effects) — exactly as today.
        let store = InMemorySettingsStore()
        let settings = SettingsManager(store: store)
        store.resetWriteRecording()
        settings.resetToDefaults()
        #expect(store.writes.contains("showInDock"))
        #expect(store.writes.contains("prefixCacheSSDBudgetBytes"))
        #expect(store.writes.contains("serverPort"))
    }

    // MARK: - Stale-value migration (the deliberate exception)

    @Test
    func staleAgentModelIdNormalisesOnLaunchAndPersists() {
        let store = InMemorySettingsStore()
        store.set("nonexistent-model-id", for: "selectedAgentModelID")

        let settings = SettingsManager(store: store)
        // Normalized in-memory…
        #expect(settings.selectedAgentModelID == ModelDefinition.defaultAgentModelID)
        // …and persisted through the store (survives the next relaunch).
        #expect(
            store.string(for: "selectedAgentModelID", default: "")
            == ModelDefinition.defaultAgentModelID
        )
    }

    // MARK: - Hydration ≠ mutation boundary

    @Test
    func constructingFromValidValuesPerformsNoWrites() {
        let store = InMemorySettingsStore()
        // A prior session persisted some valid non-default values.
        let first = SettingsManager(store: store)
        first.playSounds = false
        first.serverPort = 9000
        first.prefixCacheSSDDirectoryOverride = "/tmp/x"
        store.resetWriteRecording()

        // Hydrating those valid values must perform zero store writes and run no
        // side effects — the direct-first-assignment `init` skips `didSet`.
        let second = SettingsManager(store: store)
        _ = second
        #expect(store.writes.isEmpty)
    }

    @Test
    func constructingFromStaleModelIdWritesExactlyTheNormalizedKey() {
        let store = InMemorySettingsStore()
        store.set("nonexistent-model-id", for: "selectedAgentModelID")
        store.resetWriteRecording()

        let settings = SettingsManager(store: store)
        _ = settings
        // The single deliberate migration write — and nothing else.
        #expect(store.writes == ["selectedAgentModelID"])
    }

    @Test
    func postConstructionFlipIsAlwaysObserved() {
        let store = InMemorySettingsStore()
        let settings = SettingsManager(store: store)
        store.resetWriteRecording()
        settings.playSounds = false
        #expect(store.writes == ["playSounds"])
    }
}
