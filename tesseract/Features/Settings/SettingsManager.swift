//
//  SettingsManager.swift
//  tesseract
//

import Foundation
import Observation
import ServiceManagement
import AppKit
import MLXLMCommon

/// The Settings Facade: an `@Observable` `@MainActor` class that keeps one
/// bindable stored property per setting (so SwiftUI `$settings.foo` bindings and
/// per-property Observation are preserved) and forwards each `didSet` to an
/// injected `SettingsStore` via the property's `Setting` in the
/// `SettingsCatalogue`. Persistence plumbing lives below the facade in the store;
/// only the two genuine side effects (launch-at-login via `SMAppService`, dock
/// visibility via `NSApp`) stay here, above the store.
///
/// **Construction is hydration, not mutation.** Stored properties are declared
/// *without* a default value, so the direct, property-named assignment in `init`
/// is the genuine first write that routes through the synthesized
/// storage-restrictions init accessor and **skips `didSet`** — no write back to
/// the store, no side effects, during a clean load. (Under `@Observable` a
/// *re-assignment* in `init` would fire `didSet`; keeping a declaration default
/// would make the `init` line a re-assignment. See ADR-0002 and CONTEXT.md.)
/// The one deliberate exception is stale-value migration
/// (`normalizePersistedSelectionsIfNeeded`), which runs *after* hydration and so
/// fires `didSet` to persist the normalized value.
@Observable @MainActor
final class SettingsManager {

    /// The persistence seam. `UserDefaultsSettingsStore` in the app, an
    /// in-memory adapter in tests. The default keeps existing call sites
    /// (`SettingsManager()`) unchanged.
    @ObservationIgnored private let store: any SettingsStore

    // MARK: - General Settings

    var launchAtLogin: Bool {
        didSet {
            SettingsCatalogue.launchAtLogin.write(launchAtLogin, to: store)
            updateLaunchAtLogin()
        }
    }

    var showInDock: Bool {
        didSet {
            SettingsCatalogue.showInDock.write(showInDock, to: store)
            applyDockVisibility()
        }
    }

    var showInMenuBar: Bool {
        didSet { SettingsCatalogue.showInMenuBar.write(showInMenuBar, to: store) }
    }

    var autoInsertText: Bool {
        didSet { SettingsCatalogue.autoInsertText.write(autoInsertText, to: store) }
    }

    var restoreClipboard: Bool {
        didSet { SettingsCatalogue.restoreClipboard.write(restoreClipboard, to: store) }
    }

    var overlayStyleRaw: String {
        didSet { SettingsCatalogue.overlayStyleRaw.write(overlayStyleRaw, to: store) }
    }

    var overlayStyle: OverlayStyle {
        get { OverlayStyle(rawValue: overlayStyleRaw) ?? .pill }
        set { overlayStyleRaw = newValue.rawValue }
    }

    var samplingPresetRaw: String {
        didSet { SettingsCatalogue.samplingPresetRaw.write(samplingPresetRaw, to: store) }
    }

    var samplingPreset: SamplingPreset {
        get { SamplingPreset(rawValue: samplingPresetRaw) ?? .automatic }
        set { samplingPresetRaw = newValue.rawValue }
    }

    var glowThemeRaw: String {
        didSet { SettingsCatalogue.glowThemeRaw.write(glowThemeRaw, to: store) }
    }

    var glowTheme: GlowTheme {
        get { GlowTheme(rawValue: glowThemeRaw) ?? .appleIntelligence }
        set { glowThemeRaw = newValue.rawValue }
    }

    // MARK: - Audio Settings

    var selectedMicrophoneUID: String {
        didSet { SettingsCatalogue.selectedMicrophoneUID.write(selectedMicrophoneUID, to: store) }
    }

    // MARK: - Language Settings

    var language: String {
        didSet { SettingsCatalogue.language.write(language, to: store) }
    }

    var selectedLanguage: SupportedLanguage {
        SupportedLanguage.language(forCode: language) ?? .auto
    }

    // MARK: - Hotkey Settings

    var hotkeyKeyCode: Int {
        didSet { SettingsCatalogue.hotkeyKeyCode.write(hotkeyKeyCode, to: store) }
    }

    var hotkeyModifiers: Int {
        didSet { SettingsCatalogue.hotkeyModifiers.write(hotkeyModifiers, to: store) }
    }

    var hotkey: KeyCombo {
        get {
            KeyCombo(
                keyCode: UInt16(hotkeyKeyCode),
                modifiers: NSEvent.ModifierFlags(rawValue: UInt(hotkeyModifiers))
            )
        }
        set {
            hotkeyKeyCode = Int(newValue.keyCode)
            hotkeyModifiers = Int(newValue.modifiers)
        }
    }

    // MARK: - TTS Hotkey

    var ttsHotkeyKeyCode: Int {
        didSet { SettingsCatalogue.ttsHotkeyKeyCode.write(ttsHotkeyKeyCode, to: store) }
    }

    var ttsHotkeyModifiers: Int {
        didSet { SettingsCatalogue.ttsHotkeyModifiers.write(ttsHotkeyModifiers, to: store) }
    }

    var ttsHotkey: KeyCombo {
        get {
            KeyCombo(
                keyCode: UInt16(ttsHotkeyKeyCode),
                modifiers: NSEvent.ModifierFlags(rawValue: UInt(ttsHotkeyModifiers))
            )
        }
        set {
            ttsHotkeyKeyCode = Int(newValue.keyCode)
            ttsHotkeyModifiers = Int(newValue.modifiers)
        }
    }

    // MARK: - Agent Hotkey

    var agentHotkeyKeyCode: Int {
        didSet { SettingsCatalogue.agentHotkeyKeyCode.write(agentHotkeyKeyCode, to: store) }
    }

    var agentHotkeyModifiers: Int {
        didSet { SettingsCatalogue.agentHotkeyModifiers.write(agentHotkeyModifiers, to: store) }
    }

    var agentHotkey: KeyCombo {
        get {
            KeyCombo(
                keyCode: UInt16(agentHotkeyKeyCode),
                modifiers: NSEvent.ModifierFlags(rawValue: UInt(agentHotkeyModifiers))
            )
        }
        set {
            agentHotkeyKeyCode = Int(newValue.keyCode)
            agentHotkeyModifiers = Int(newValue.modifiers)
        }
    }

    // MARK: - TTS Settings

    var ttsTemperature: Double {
        didSet { SettingsCatalogue.ttsTemperature.write(ttsTemperature, to: store) }
    }

    var ttsTopP: Double {
        didSet { SettingsCatalogue.ttsTopP.write(ttsTopP, to: store) }
    }

    var ttsRepetitionPenalty: Double {
        didSet { SettingsCatalogue.ttsRepetitionPenalty.write(ttsRepetitionPenalty, to: store) }
    }

    var ttsMaxTokens: Int {
        didSet { SettingsCatalogue.ttsMaxTokens.write(ttsMaxTokens, to: store) }
    }

    var ttsSeed: Int {
        didSet { SettingsCatalogue.ttsSeed.write(ttsSeed, to: store) }
    }

    var ttsVoiceDescription: String {
        didSet { SettingsCatalogue.ttsVoiceDescription.write(ttsVoiceDescription, to: store) }
    }

    var ttsLanguage: String {
        didSet { SettingsCatalogue.ttsLanguage.write(ttsLanguage, to: store) }
    }

    var ttsStreamingEnabled: Bool {
        didSet { SettingsCatalogue.ttsStreamingEnabled.write(ttsStreamingEnabled, to: store) }
    }

    var agentAutoSpeak: Bool {
        didSet { SettingsCatalogue.agentAutoSpeak.write(agentAutoSpeak, to: store) }
    }

    var selectedAgentModelID: String {
        didSet { SettingsCatalogue.selectedAgentModelID.write(selectedAgentModelID, to: store) }
    }

    var ttsParameters: TTSParameters {
        get {
            TTSParameters(
                temperature: Float(ttsTemperature),
                topP: Float(ttsTopP),
                repetitionPenalty: Float(ttsRepetitionPenalty),
                maxTokens: ttsMaxTokens,
                seed: UInt64(ttsSeed)
            )
        }
        set {
            ttsTemperature = Double(newValue.temperature)
            ttsTopP = Double(newValue.topP)
            ttsRepetitionPenalty = Double(newValue.repetitionPenalty)
            ttsMaxTokens = newValue.maxTokens
            ttsSeed = Int(newValue.seed)
        }
    }

    // MARK: - Advanced Settings

    var maxRecordingDuration: Double {
        didSet { SettingsCatalogue.maxRecordingDuration.write(maxRecordingDuration, to: store) }
    }

    var playSounds: Bool {
        didSet { SettingsCatalogue.playSounds.write(playSounds, to: store) }
    }

    // MARK: - Agent Web Access

    var webAccessEnabled: Bool {
        didSet { SettingsCatalogue.webAccessEnabled.write(webAccessEnabled, to: store) }
    }

    // MARK: - Agent Vision Mode

    /// When true, the agent loads the VLM Qwen3.5 container which supports image
    /// attachments but has ~3.4× slower prefill on long text prompts. Default false
    /// — users opt-in via the composer toggle when they need to attach an image.
    /// Changing this triggers a model reload via `InferenceArbiter.ensureLoaded(.llm)`.
    var visionModeEnabled: Bool {
        didSet { SettingsCatalogue.visionModeEnabled.write(visionModeEnabled, to: store) }
    }

    /// Runtime gate for TriAttention sparse attention on Qwen3.5 PARO text
    /// models. Defaults to `false`. Surfaced in `ServerConfigurationView` as a plain
    /// toggle; plumbing flows through `makeTriAttentionConfig()` →
    /// `AgentGenerateParameters` → server core. Flipping this while an LLM is
    /// loaded kicks off an eager reload via `InferenceArbiter.reloadLLMIfNeeded()`
    /// (observed in `DependencyContainer`). Flipping before any LLM is loaded
    /// is picked up at the next lazy-load `ensureLoaded(.llm)` call. The view
    /// reads `arbiter.loadedLLMState.triAttentionFallbackReason` to surface
    /// dense fallback reasons (non-PARO model, vision mode, missing artifact).
    var triattentionEnabled: Bool {
        didSet { SettingsCatalogue.triattentionEnabled.write(triattentionEnabled, to: store) }
    }

    // MARK: - Server Settings

    var isServerEnabled: Bool {
        didSet { SettingsCatalogue.isServerEnabled.write(isServerEnabled, to: store) }
    }

    var serverPort: Int {
        didSet { SettingsCatalogue.serverPort.write(serverPort, to: store) }
    }

    // MARK: - SSD Prefix Cache

    // Changes to these settings take effect on the next model unload/reload.
    // `LLMActor` snapshots the effective config at load time — the hot path
    // inside `container.perform` cannot await MainActor mid-inference.

    var prefixCacheSSDEnabled: Bool {
        didSet { SettingsCatalogue.prefixCacheSSDEnabled.write(prefixCacheSSDEnabled, to: store) }
    }

    /// Hard top-level byte budget for the SSD tier. Default single-sourced in
    /// `SettingsCatalogue.prefixCacheSSDBudgetBytes` (20 GiB).
    var prefixCacheSSDBudgetBytes: Int {
        didSet { SettingsCatalogue.prefixCacheSSDBudgetBytes.write(prefixCacheSSDBudgetBytes, to: store) }
    }

    /// Optional override for the SSD root directory. When `nil`, the config
    /// falls back to the sandbox Caches directory. Accepts either a file
    /// URL string or a plain filesystem path. Writing `nil` removes the key.
    var prefixCacheSSDDirectoryOverride: String? {
        didSet { SettingsCatalogue.prefixCacheSSDDirectoryOverride.write(prefixCacheSSDDirectoryOverride, to: store) }
    }

    // MARK: - Onboarding

    var hasCompletedOnboarding: Bool {
        didSet { SettingsCatalogue.hasCompletedOnboarding.write(hasCompletedOnboarding, to: store) }
    }

    // MARK: - Init

    /// Hydrate every property from the injected store via a direct, property-named
    /// first assignment fed by the catalogue — `self.foo = Catalogue.foo.load(...)`
    /// — which skips `didSet`, so construction performs no store writes and runs
    /// no side effects. `normalizePersistedSelectionsIfNeeded()` runs last, after
    /// the clean load, so its (rare) re-assignment fires `didSet` and persists.
    init(store: any SettingsStore = UserDefaultsSettingsStore()) {
        self.store = store

        self.launchAtLogin = SettingsCatalogue.launchAtLogin.load(from: store)
        self.showInDock = SettingsCatalogue.showInDock.load(from: store)
        self.showInMenuBar = SettingsCatalogue.showInMenuBar.load(from: store)
        self.autoInsertText = SettingsCatalogue.autoInsertText.load(from: store)
        self.restoreClipboard = SettingsCatalogue.restoreClipboard.load(from: store)
        self.overlayStyleRaw = SettingsCatalogue.overlayStyleRaw.load(from: store)
        self.glowThemeRaw = SettingsCatalogue.glowThemeRaw.load(from: store)
        self.samplingPresetRaw = SettingsCatalogue.samplingPresetRaw.load(from: store)
        self.selectedMicrophoneUID = SettingsCatalogue.selectedMicrophoneUID.load(from: store)
        self.language = SettingsCatalogue.language.load(from: store)
        self.hotkeyKeyCode = SettingsCatalogue.hotkeyKeyCode.load(from: store)
        self.hotkeyModifiers = SettingsCatalogue.hotkeyModifiers.load(from: store)
        self.ttsHotkeyKeyCode = SettingsCatalogue.ttsHotkeyKeyCode.load(from: store)
        self.ttsHotkeyModifiers = SettingsCatalogue.ttsHotkeyModifiers.load(from: store)
        self.agentHotkeyKeyCode = SettingsCatalogue.agentHotkeyKeyCode.load(from: store)
        self.agentHotkeyModifiers = SettingsCatalogue.agentHotkeyModifiers.load(from: store)
        self.ttsTemperature = SettingsCatalogue.ttsTemperature.load(from: store)
        self.ttsTopP = SettingsCatalogue.ttsTopP.load(from: store)
        self.ttsRepetitionPenalty = SettingsCatalogue.ttsRepetitionPenalty.load(from: store)
        self.ttsMaxTokens = SettingsCatalogue.ttsMaxTokens.load(from: store)
        self.ttsSeed = SettingsCatalogue.ttsSeed.load(from: store)
        self.ttsVoiceDescription = SettingsCatalogue.ttsVoiceDescription.load(from: store)
        self.ttsLanguage = SettingsCatalogue.ttsLanguage.load(from: store)
        self.ttsStreamingEnabled = SettingsCatalogue.ttsStreamingEnabled.load(from: store)
        self.agentAutoSpeak = SettingsCatalogue.agentAutoSpeak.load(from: store)
        self.selectedAgentModelID = SettingsCatalogue.selectedAgentModelID.load(from: store)
        self.maxRecordingDuration = SettingsCatalogue.maxRecordingDuration.load(from: store)
        self.playSounds = SettingsCatalogue.playSounds.load(from: store)
        self.webAccessEnabled = SettingsCatalogue.webAccessEnabled.load(from: store)
        self.visionModeEnabled = SettingsCatalogue.visionModeEnabled.load(from: store)
        self.triattentionEnabled = SettingsCatalogue.triattentionEnabled.load(from: store)
        self.isServerEnabled = SettingsCatalogue.isServerEnabled.load(from: store)
        self.serverPort = SettingsCatalogue.serverPort.load(from: store)
        self.prefixCacheSSDEnabled = SettingsCatalogue.prefixCacheSSDEnabled.load(from: store)
        self.prefixCacheSSDBudgetBytes = SettingsCatalogue.prefixCacheSSDBudgetBytes.load(from: store)
        self.prefixCacheSSDDirectoryOverride = SettingsCatalogue.prefixCacheSSDDirectoryOverride.load(from: store)
        self.hasCompletedOnboarding = SettingsCatalogue.hasCompletedOnboarding.load(from: store)

        normalizePersistedSelectionsIfNeeded()
    }

    // MARK: - Methods

    /// Capture an immutable `SSDPrefixCacheConfig` from the current settings,
    /// or `nil` if the SSD tier is disabled. Called on MainActor at model
    /// load time; the result is held by `LLMActor` for the lifetime of the
    /// load. Settings mutated after this call take effect on the next
    /// unload/reload cycle — the hot prefix-cache path cannot await
    /// MainActor for mid-run config reads.
    func makeSSDPrefixCacheConfig() -> SSDPrefixCacheConfig? {
        guard prefixCacheSSDEnabled else { return nil }
        return .withAutoPendingCap(
            rootURL: resolvedSSDPrefixCacheRootURL(),
            budgetBytes: prefixCacheSSDBudgetBytes
        )
    }

    /// Build the agent generation parameters implied by the current settings:
    /// model-derived preset + user sampling override + TriAttention toggle.
    /// Live-reads all three so a settings change takes effect on the very
    /// next call. Factories should prefer this over assembling the pieces
    /// inline to keep the ordering and sources canonical.
    func makeAgentGenerateParameters() -> AgentGenerateParameters {
        var parameters = AgentGenerateParameters.forModel(selectedAgentModelID)
        parameters.triAttention = makeTriAttentionConfig()
        parameters = samplingPreset.apply(to: parameters)
        return parameters
    }

    func makeTriAttentionConfig() -> TriAttentionConfiguration {
        TriAttentionConfiguration(
            enabled: triattentionEnabled,
            budgetTokens: TriAttentionConfiguration.v1BudgetTokens,
            calibrationArtifactIdentity: nil,
            implementationVersion: .v1,
            prefixProtectionMode: .protectStablePrefixOnly
        )
    }

    private func resolvedSSDPrefixCacheRootURL() -> URL {
        if let override = prefixCacheSSDDirectoryOverride, !override.isEmpty {
            // Accept either a file URL string or a plain path.
            if let url = URL(string: override), url.isFileURL {
                return url
            }
            return URL(fileURLWithPath: override, isDirectory: true)
        }
        return FileManager.default
            .urls(for: .cachesDirectory, in: .userDomainMask)
            .first!
            .appendingPathComponent("prefix-cache", isDirectory: true)
    }

    /// Restore every *preference* to its single-sourced catalogue default. Runs
    /// *after* `init`, so each assignment fires `didSet` — the value persists
    /// through the store and side effects (launch-at-login, dock visibility)
    /// re-apply, exactly as reset did before the seam.
    ///
    /// Deliberate exception: `hasCompletedOnboarding` is *not* reset. "Reset to
    /// Defaults" is a Settings action and must never resurface the onboarding
    /// flow for an existing user — onboarding completion is app-lifecycle state,
    /// not a preference. It stays catalogued (still hydrated and persisted), just
    /// outside the reset contract. (Matches the pre-seam behaviour, which also
    /// omitted it.) Pinned by
    /// `SettingsManagerTests.resetToDefaultsLeavesOnboardingCompletionIntact`.
    func resetToDefaults() {
        launchAtLogin = SettingsCatalogue.launchAtLogin.default
        showInDock = SettingsCatalogue.showInDock.default
        showInMenuBar = SettingsCatalogue.showInMenuBar.default
        autoInsertText = SettingsCatalogue.autoInsertText.default
        restoreClipboard = SettingsCatalogue.restoreClipboard.default
        overlayStyleRaw = SettingsCatalogue.overlayStyleRaw.default
        glowThemeRaw = SettingsCatalogue.glowThemeRaw.default
        selectedMicrophoneUID = SettingsCatalogue.selectedMicrophoneUID.default
        language = SettingsCatalogue.language.default
        hotkeyKeyCode = SettingsCatalogue.hotkeyKeyCode.default
        hotkeyModifiers = SettingsCatalogue.hotkeyModifiers.default
        maxRecordingDuration = SettingsCatalogue.maxRecordingDuration.default
        playSounds = SettingsCatalogue.playSounds.default
        ttsHotkeyKeyCode = SettingsCatalogue.ttsHotkeyKeyCode.default
        ttsHotkeyModifiers = SettingsCatalogue.ttsHotkeyModifiers.default
        agentHotkeyKeyCode = SettingsCatalogue.agentHotkeyKeyCode.default
        agentHotkeyModifiers = SettingsCatalogue.agentHotkeyModifiers.default
        ttsTemperature = SettingsCatalogue.ttsTemperature.default
        ttsTopP = SettingsCatalogue.ttsTopP.default
        ttsRepetitionPenalty = SettingsCatalogue.ttsRepetitionPenalty.default
        ttsMaxTokens = SettingsCatalogue.ttsMaxTokens.default
        ttsSeed = SettingsCatalogue.ttsSeed.default
        ttsVoiceDescription = SettingsCatalogue.ttsVoiceDescription.default
        ttsLanguage = SettingsCatalogue.ttsLanguage.default
        ttsStreamingEnabled = SettingsCatalogue.ttsStreamingEnabled.default
        agentAutoSpeak = SettingsCatalogue.agentAutoSpeak.default
        selectedAgentModelID = SettingsCatalogue.selectedAgentModelID.default
        webAccessEnabled = SettingsCatalogue.webAccessEnabled.default
        visionModeEnabled = SettingsCatalogue.visionModeEnabled.default
        triattentionEnabled = SettingsCatalogue.triattentionEnabled.default
        samplingPresetRaw = SettingsCatalogue.samplingPresetRaw.default
        isServerEnabled = SettingsCatalogue.isServerEnabled.default
        serverPort = SettingsCatalogue.serverPort.default
        prefixCacheSSDEnabled = SettingsCatalogue.prefixCacheSSDEnabled.default
        prefixCacheSSDBudgetBytes = SettingsCatalogue.prefixCacheSSDBudgetBytes.default
        prefixCacheSSDDirectoryOverride = SettingsCatalogue.prefixCacheSSDDirectoryOverride.default
        // hasCompletedOnboarding is intentionally omitted — see the doc comment.
    }

    // MARK: - Private

    private func updateLaunchAtLogin() {
        do {
            if launchAtLogin {
                try SMAppService.mainApp.register()
            } else {
                try SMAppService.mainApp.unregister()
            }
        } catch {
            Log.general.error("Failed to update launch at login: \(error)")
        }
    }

    /// Stale-value migration (the one deliberate non-hydration step). When the
    /// persisted agent model id no longer maps to a known agent model, normalise
    /// it to the default. Runs after hydration, so the re-assignment fires
    /// `didSet` and persists through the store for free.
    private func normalizePersistedSelectionsIfNeeded() {
        let normalizedModelID = Self.normalizedAgentModelID(selectedAgentModelID)
        guard normalizedModelID != selectedAgentModelID else { return }
        selectedAgentModelID = normalizedModelID
    }

    private static func normalizedAgentModelID(_ candidate: String) -> String {
        let agentModelIDs = Set(
            ModelDefinition.all
                .filter { $0.category == .agent }
                .map(\.id)
        )
        if agentModelIDs.contains(candidate) {
            return candidate
        }
        if agentModelIDs.contains(ModelDefinition.defaultAgentModelID) {
            return ModelDefinition.defaultAgentModelID
        }
        return ModelDefinition.all.first(where: { $0.category == .agent })?.id ?? candidate
    }

    func applyDockVisibility() {
        if showInDock {
            NSApp.setActivationPolicy(.regular)
        } else {
            NSApp.setActivationPolicy(.accessory)
        }
    }
}
