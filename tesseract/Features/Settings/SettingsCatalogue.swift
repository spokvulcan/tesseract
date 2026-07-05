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
    static let glowThemeRaw = Setting.string(
        "glowTheme", default: GlowTheme.appleIntelligence.rawValue)
    static let samplingPresetRaw = Setting.string(
        "samplingPreset", default: SamplingPreset.automatic.rawValue)

    // MARK: - Audio

    static let selectedMicrophoneUID = Setting.string("selectedMicrophoneUID", default: "")

    // MARK: - Language

    static let language = Setting.string("language", default: "en")

    // MARK: - Hotkeys

    static let hotkeyKeyCode = Setting.int(
        "hotkeyKeyCode", default: Int(KeyCombo.optionSpace.keyCode))
    static let hotkeyModifiers = Setting.int(
        "hotkeyModifiers", default: Int(KeyCombo.optionSpace.modifiers))
    static let ttsHotkeyKeyCode = Setting.int(
        "ttsHotkeyKeyCode", default: Int(KeyCombo.functionSpace.keyCode))
    static let ttsHotkeyModifiers = Setting.int(
        "ttsHotkeyModifiers", default: Int(KeyCombo.functionSpace.modifiers))
    static let agentHotkeyKeyCode = Setting.int(
        "agentHotkeyKeyCode", default: Int(KeyCombo.controlSpace.keyCode))
    static let agentHotkeyModifiers = Setting.int(
        "agentHotkeyModifiers", default: Int(KeyCombo.controlSpace.modifiers))
    static let appshotHotkeyKeyCode = Setting.int(
        "appshotHotkeyKeyCode", default: Int(KeyCombo.doubleCommand.keyCode))
    static let appshotHotkeyModifiers = Setting.int(
        "appshotHotkeyModifiers", default: Int(KeyCombo.doubleCommand.modifiers))

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
    static let selectedAgentModelID = Setting.string(
        "selectedAgentModelID", default: ModelDefinition.defaultAgentModelID)
    static let selectedSpeechToTextModelID = Setting.string(
        "selectedSpeechToTextModelID", default: ModelDefinition.defaultSpeechToTextModelID)

    // MARK: - Advanced

    static let maxRecordingDuration = Setting.double("maxRecordingDuration", default: 300.0)
    static let playSounds = Setting.bool("playSounds", default: true)

    // MARK: - Agent

    static let webAccessEnabled = Setting.bool("webAccessEnabled", default: true)
    /// Global opt-out governing chat-initiated vision loads (ADR-0013, PRD #112).
    /// When on (default), the chat send path requests `.visionIfCapable`, so a
    /// vision-capable model loads its VLM container from turn one and image
    /// affordances appear in the composer. When off, the send path resolves
    /// `.fromSettings`, which gates vision on this opt-out (→ text-only). The
    /// HTTP server ignores this (ADR-0008).
    static let useVisionWhenAvailable = Setting.bool("useVisionWhenAvailable", default: true)

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

    // MARK: - Skill Pills (PRD #174)

    /// The "Show skill pills" opt-out for the pill row above the agent
    /// composer. Default on; the row also hides itself when no skill declares
    /// pill membership.
    static let showSkillPills = Setting.bool("showSkillPills", default: true)

    /// The `translate` skill's default target language, stored as an English
    /// display name ("Ukrainian"). Pre-filled once per launch from the first
    /// non-English macOS preferred language; the picker in agent settings
    /// overrides it.
    static let translateTargetLanguage = Setting.string(
        "translateTargetLanguage", default: TranslateLanguageDefault.systemDefault())

    /// Shared key prefix for the dynamic per-skill usage counters (the Skill
    /// Usage Ranking), so `resetToDefaults` can sweep them — same pattern as
    /// `preserveThinkingRenderKeyPrefix`.
    static let skillUsageCountKeyPrefix = "skillUsageCount."

    static func skillUsageCount(skillName: String) -> Setting<Int> {
        Setting.int(skillUsageCountKeyPrefix + skillName, default: 0)
    }

    // MARK: - Server

    static let isServerEnabled = Setting.bool("isServerEnabled", default: false)
    static let serverPort = Setting.int("serverPort", default: 8321)

    // MARK: - Prefix Cache

    static let prefixCacheSSDEnabled = Setting.bool("prefixCacheSSDEnabled", default: true)
    /// User cap on the RAM-tier cache budget (ADR-0018). `nil` =
    /// "Automatic (recommended)": the ceiling tracks measured headroom.
    /// A custom value only ever lowers the effective ceiling — caps,
    /// never floors; pressure retreat always wins.
    static let prefixCacheRAMBudgetCapBytes = Setting.optionalInt(
        "prefixCacheRAMBudgetCapBytes")
    /// User cap on the SSD-tier budget (ADR-0018). `nil` = "Automatic
    /// (recommended)": the budget tracks measured free disk space
    /// (`SSDBudgetPolicy` — fraction, absolute cap, floored at the old
    /// 20 GiB default, which replaced the retired fixed
    /// `prefixCacheSSDBudgetBytes` setting).
    static let prefixCacheSSDBudgetCapBytes = Setting.optionalInt(
        "prefixCacheSSDBudgetCapBytes")
    static let prefixCacheSSDDirectoryOverride = Setting.optionalString(
        "prefixCacheSSDDirectoryOverride")

    // MARK: - Onboarding

    static let hasCompletedOnboarding = Setting.bool("hasCompletedOnboarding", default: false)
}
