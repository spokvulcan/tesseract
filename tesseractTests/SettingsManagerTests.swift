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
import TesseractSpeech
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
        #expect(settings.speechOverlayShowsControls == true)
        #expect(settings.ttsPlaybackRate == 1.0)
        #expect(settings.readerTypeface == .serif)
        #expect(settings.speechOverlayStyle == .island)
        #expect(settings.savedVoices.isEmpty)
        #expect(settings.playSounds == true)
        #expect(settings.prefixCacheRAMBudgetCapBytes == nil)
        #expect(settings.prefixCacheSSDBudgetCapBytes == nil)
        #expect(settings.prefixCacheSSDDirectoryOverride == nil)
        #expect(settings.selectedAgentModelID == ModelDefinition.defaultAgentModelID)
        #expect(settings.useVisionWhenAvailable == true)
    }

    /// ADR-0072: values tuned for the old TTS sampler (temperature after
    /// top-p, a whole-chunk repetition penalty) sit under abandoned keys and
    /// are never read, so the new defaults apply and nothing is written.
    @Test
    func oldSamplerValuesAreLeftUnread() {
        let store = InMemorySettingsStore()
        store.set(0.6, for: "ttsTemperature")
        store.set(0.8, for: "ttsTopP")
        store.set(1.3, for: "ttsRepetitionPenalty")
        store.resetWriteRecording()

        let settings = SettingsManager(store: store)
        #expect(settings.ttsParameters == TTSParameters())
        #expect(store.writes.isEmpty)

        settings.ttsTemperature = 0.8
        #expect(SettingsManager(store: store).ttsTemperature == 0.8)
    }

    /// The vision opt-out persists across a relaunch and writes exactly its own
    /// key (the hydration≠mutation boundary) — pins ADR-0013's global setting.
    @Test
    func visionOptOutPersistsAndWritesOnlyItsKey() {
        let store = InMemorySettingsStore()
        let first = SettingsManager(store: store)
        store.resetWriteRecording()
        first.useVisionWhenAvailable = false
        #expect(store.writes == ["useVisionWhenAvailable"])

        let second = SettingsManager(store: store)
        #expect(second.useVisionWhenAvailable == false)
    }

    /// The Markdown render toggle graduated from a loose `@AppStorage` into the
    /// catalogue (map #211 cutover): default on, persists across a relaunch,
    /// and a pre-existing store value (what `@AppStorage` wrote under the same
    /// key) hydrates instead of the default — the migration is read-compatible.
    @Test
    func agentUseMarkdownDefaultsOnPersistsAndReadsLegacyKey() {
        let store = InMemorySettingsStore()
        let first = SettingsManager(store: store)
        #expect(first.agentUseMarkdown == true)

        first.agentUseMarkdown = false
        let second = SettingsManager(store: store)
        #expect(second.agentUseMarkdown == false)

        second.resetToDefaults()
        #expect(second.agentUseMarkdown == true)
    }

    /// The speculation-mode picker replaced the `mtpSpeculationEnabled` bool.
    /// A fresh install reads `.automatic`; a persisted legacy opt-out
    /// migrates to `.off` once; an explicit later choice survives a relaunch
    /// even with the legacy opt-out still in the store.
    @Test
    func speculationModeDefaultsAutomaticAndMigratesLegacyOptOut() {
        let fresh = SettingsManager(store: InMemorySettingsStore())
        #expect(fresh.speculationMode == .automatic)

        let store = InMemorySettingsStore()
        store.set(false, for: "mtpSpeculationEnabled")
        let migrated = SettingsManager(store: store)
        #expect(migrated.speculationMode == .off)

        migrated.speculationMode = .dflash2
        let relaunched = SettingsManager(store: store)
        #expect(relaunched.speculationMode == .dflash2)

        relaunched.resetToDefaults()
        #expect(relaunched.speculationMode == .automatic)
    }

    /// A legacy `true` (or an absent legacy key) must NOT write the new key —
    /// the catalogue default stays live so a future default change reaches
    /// users who never chose explicitly.
    @Test
    func speculationModeLegacyOnDoesNotPinTheNewKey() {
        let store = InMemorySettingsStore()
        store.set(true, for: "mtpSpeculationEnabled")
        _ = SettingsManager(store: store)
        #expect(store.optionalString(for: "speculationMode") == nil)
    }

    /// An unrecognized persisted raw value degrades to `.automatic` instead
    /// of crashing the facade load.
    @Test
    func speculationModeUnknownRawValueDegradesToAutomatic() {
        let store = InMemorySettingsStore()
        store.set("warp-drive", for: "speculationMode")
        let settings = SettingsManager(store: store)
        #expect(settings.speculationMode == .automatic)
    }

    // MARK: - Persistence across a simulated relaunch

    @Test
    func flippingPropertyPersistsAndSurvivesRelaunch() {
        let store = InMemorySettingsStore()
        let first = SettingsManager(store: store)
        first.playSounds = false
        first.serverPort = 9000
        first.prefixCacheSSDBudgetCapBytes = 12 * 1024 * 1024 * 1024
        first.prefixCacheSSDDirectoryOverride = "/tmp/roundtrip"

        // A fresh facade on the same store is the relaunch.
        let second = SettingsManager(store: store)
        #expect(second.playSounds == false)
        #expect(second.serverPort == 9000)
        #expect(second.prefixCacheSSDBudgetCapBytes == 12 * 1024 * 1024 * 1024)
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
        settings.prefixCacheSSDBudgetCapBytes = 1
        settings.prefixCacheSSDDirectoryOverride = "/tmp/x"

        settings.resetToDefaults()
        #expect(settings.showInDock == true)
        #expect(settings.serverPort == 8321)
        #expect(settings.prefixCacheSSDBudgetCapBytes == nil)
        #expect(settings.prefixCacheSSDDirectoryOverride == nil)

        // Reset persists: a relaunch on the same store sees the defaults, never
        // the stale values (i.e. reset wrote through the store).
        let relaunched = SettingsManager(store: store)
        #expect(relaunched.showInDock == true)
        #expect(relaunched.serverPort == 8321)
        #expect(relaunched.prefixCacheSSDBudgetCapBytes == nil)
        #expect(relaunched.prefixCacheSSDDirectoryOverride == nil)
    }

    @Test
    func fixHotkeyAndProofreadPassResetToTheirDefaults() {
        // PRD #612: the fix hotkey persists like the other hotkeys, and Reset
        // to Defaults brings back ⌃⌥Space with the Proofread Pass off.
        let store = InMemorySettingsStore()
        let settings = SettingsManager(store: store)
        settings.fixHotkey = .f5
        settings.proofreadDictation = true
        #expect(SettingsManager(store: store).fixHotkey == .f5)
        #expect(SettingsManager(store: store).proofreadDictation == true)

        settings.resetToDefaults()
        #expect(settings.fixHotkey == .controlOptionSpace)
        #expect(settings.proofreadDictation == false)

        let relaunched = SettingsManager(store: store)
        #expect(relaunched.fixHotkey == .controlOptionSpace)
        #expect(relaunched.proofreadDictation == false)
    }

    @Test
    func checkBeforePastingPersistsAndResetsToWhenShiftTapped() {
        // PRD #612: the choice survives a relaunch, writes only its own key,
        // and Reset to Defaults brings back "When I tap ⇧".
        let store = InMemorySettingsStore()
        let settings = SettingsManager(store: store)
        store.resetWriteRecording()
        settings.checkBeforePasting = .always
        #expect(store.writes == ["checkBeforePasting"])
        #expect(SettingsManager(store: store).checkBeforePasting == .always)

        settings.checkBeforePasting = .never
        #expect(SettingsManager(store: store).checkBeforePasting == .never)

        settings.resetToDefaults()
        #expect(settings.checkBeforePasting == .whenShiftTapped)
        #expect(SettingsManager(store: store).checkBeforePasting == .whenShiftTapped)
    }

    @Test
    func checkBeforePastingUnknownRawValueDegradesToWhenShiftTapped() {
        let store = InMemorySettingsStore()
        store.set("sometimes", for: "checkBeforePasting")
        let settings = SettingsManager(store: store)
        #expect(settings.checkBeforePasting == .whenShiftTapped)
    }

    @Test
    func checkBeforePastingDecidesWhetherATakeWaits() {
        #expect(CheckBeforePasting.whenShiftTapped.takeWaits(shiftTapped: true))
        #expect(!CheckBeforePasting.whenShiftTapped.takeWaits(shiftTapped: false))
        #expect(CheckBeforePasting.always.takeWaits(shiftTapped: false))
        #expect(CheckBeforePasting.always.takeWaits(shiftTapped: true))
        #expect(!CheckBeforePasting.never.takeWaits(shiftTapped: true))
        #expect(!CheckBeforePasting.never.takeWaits(shiftTapped: false))
    }

    @Test
    func aStoredProofreadPassIsTurnedOffOnceAndAnOptInStays() {
        // PRD #612: installs that stored the old default (`true`, written by
        // the toggle or by Reset to Defaults) are switched off once.
        let store = InMemorySettingsStore()
        SettingsCatalogue.proofreadDictation.write(true, to: store)
        let settings = SettingsManager(store: store)
        #expect(settings.proofreadDictation == false)

        // A later opt-in survives every relaunch.
        settings.proofreadDictation = true
        #expect(SettingsManager(store: store).proofreadDictation == true)
        #expect(SettingsManager(store: store).proofreadDictation == true)
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
        #expect(store.writes.contains("prefixCacheSSDBudgetCapBytes"))
        #expect(store.writes.contains("serverPort"))
        #expect(store.writes.contains("fixHotkeyKeyCode"))
        #expect(store.writes.contains("fixHotkeyModifiers"))
        #expect(store.writes.contains("checkBeforePasting"))
    }

    @Test
    func resetToDefaultsLeavesOnboardingCompletionIntact() {
        // The deliberate exception to the catalogue-reset contract: "Reset to
        // Defaults" must not resurface onboarding for an existing user.
        // `hasCompletedOnboarding` is catalogued (hydrated + persisted) but not
        // reset — onboarding completion is app-lifecycle state, not a preference.
        let store = InMemorySettingsStore()
        let settings = SettingsManager(store: store)
        settings.hasCompletedOnboarding = true

        settings.resetToDefaults()
        #expect(settings.hasCompletedOnboarding == true)

        // And it survives the next relaunch — never silently re-onboards.
        let relaunched = SettingsManager(store: store)
        #expect(relaunched.hasCompletedOnboarding == true)
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
