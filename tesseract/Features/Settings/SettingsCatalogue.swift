//
//  SettingsCatalogue.swift
//  tesseract
//

import Foundation

/// The table of all `Setting` declarations — one per persisted primitive, the
/// single source of truth for each setting's key and default. Replaces the
/// former triplication (stored-property literal + `register(defaults:)` +
/// `resetToDefaults`) so a default has exactly one home; the 50-vs-20-GiB SSD
/// budget drift becomes unrepresentable.
///
/// Composite/derived members (`hotkey`, `ttsHotkey`, `agentHotkey`,
/// `ttsParameters`, `selectedLanguage`, the enum-over-raw pairs) stay computed
/// over these primitives in the `SettingsManager` facade, so the catalogue only
/// declares the primitives that actually own a key.
enum SettingsCatalogue {

    // MARK: - General

    static let launchAtLogin = Setting.bool("launchAtLogin", default: false)
    static let showInDock = Setting.bool("showInDock", default: true)
    static let showInMenuBar = Setting.bool("showInMenuBar", default: true)
    static let autoInsertText = Setting.bool("autoInsertText", default: true)
    static let restoreClipboard = Setting.bool("restoreClipboard", default: true)
    static let overlayStyleRaw = Setting.string("overlayStyle", default: OverlayStyle.pill.rawValue)
    static let glowThemeRaw = Setting.string("glowTheme", default: GlowTheme.appleIntelligence.rawValue)
    static let samplingPresetRaw = Setting.string("samplingPreset", default: SamplingPreset.automatic.rawValue)

    // MARK: - Audio

    static let selectedMicrophoneUID = Setting.string("selectedMicrophoneUID", default: "")

    // MARK: - Language

    static let language = Setting.string("language", default: "en")

    // MARK: - Hotkeys

    static let hotkeyKeyCode = Setting.int("hotkeyKeyCode", default: Int(KeyCombo.optionSpace.keyCode))
    static let hotkeyModifiers = Setting.int("hotkeyModifiers", default: Int(KeyCombo.optionSpace.modifiers))
    static let ttsHotkeyKeyCode = Setting.int("ttsHotkeyKeyCode", default: Int(KeyCombo.functionSpace.keyCode))
    static let ttsHotkeyModifiers = Setting.int("ttsHotkeyModifiers", default: Int(KeyCombo.functionSpace.modifiers))
    static let agentHotkeyKeyCode = Setting.int("agentHotkeyKeyCode", default: Int(KeyCombo.controlSpace.keyCode))
    static let agentHotkeyModifiers = Setting.int("agentHotkeyModifiers", default: Int(KeyCombo.controlSpace.modifiers))

    // MARK: - TTS

    static let ttsTemperature = Setting.double("ttsTemperature", default: 0.6)
    static let ttsTopP = Setting.double("ttsTopP", default: 0.8)
    static let ttsRepetitionPenalty = Setting.double("ttsRepetitionPenalty", default: 1.3)
    static let ttsMaxTokens = Setting.int("ttsMaxTokens", default: 4096)
    static let ttsSeed = Setting.int("ttsSeed", default: 0)
    static let ttsVoiceDescription = Setting.string("ttsVoiceDescription", default: "")
    static let ttsLanguage = Setting.string("ttsLanguage", default: "English")
    static let ttsStreamingEnabled = Setting.bool("ttsStreamingEnabled", default: true)
    static let agentAutoSpeak = Setting.bool("agentAutoSpeak", default: false)
    static let selectedAgentModelID = Setting.string("selectedAgentModelID", default: ModelDefinition.defaultAgentModelID)
    static let selectedSpeechToTextModelID = Setting.string("selectedSpeechToTextModelID", default: ModelDefinition.defaultSpeechToTextModelID)

    // MARK: - Advanced

    static let maxRecordingDuration = Setting.double("maxRecordingDuration", default: 300.0)
    static let playSounds = Setting.bool("playSounds", default: true)

    // MARK: - Agent

    static let webAccessEnabled = Setting.bool("webAccessEnabled", default: true)
    static let visionModeEnabled = Setting.bool("visionModeEnabled", default: false)

    /// Per-model opt-in for the **Preserve-Thinking Render** (issue #98).
    /// Keyed by model ID because the capability is per chat template; the UI
    /// surfaces the toggle only for models whose template declares the flag
    /// (`ModelIdentity.declaredTemplateFlags`). The one dynamic-key setting in
    /// the catalogue — a fixed-key declaration cannot enumerate model IDs.
    /// Shared key prefix for the dynamic per-model keys, so `resetToDefaults`
    /// can sweep them without re-deriving the literal.
    static let preserveThinkingRenderKeyPrefix = "preserveThinkingRender."

    static func preserveThinkingRender(modelID: String) -> Setting<Bool> {
        Setting.bool(preserveThinkingRenderKeyPrefix + modelID, default: false)
    }

    // MARK: - Server

    static let isServerEnabled = Setting.bool("isServerEnabled", default: false)
    static let serverPort = Setting.int("serverPort", default: 8321)

    // MARK: - SSD Prefix Cache

    static let prefixCacheSSDEnabled = Setting.bool("prefixCacheSSDEnabled", default: true)
    /// Hard top-level byte budget for the SSD tier. Single-sourced to **20 GiB**
    /// — the value the old `register(defaults:)` made effective at runtime.
    /// (The pre-refactor property literal and doc comment said 50 GiB, but the
    /// registered 20 GiB won on read; resolved here to preserve observed
    /// behaviour. See issue #16.)
    static let prefixCacheSSDBudgetBytes = Setting.int("prefixCacheSSDBudgetBytes", default: 20 * 1024 * 1024 * 1024)
    static let prefixCacheSSDDirectoryOverride = Setting.optionalString("prefixCacheSSDDirectoryOverride")

    // MARK: - Onboarding

    static let hasCompletedOnboarding = Setting.bool("hasCompletedOnboarding", default: false)
}
