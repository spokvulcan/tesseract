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
///
/// Deliberately *not* catalogued — the blessed UI-local `@AppStorage` keys
/// (Settings IA, issue #213): pure view state (panel visibility, disclosure)
/// that belongs to its surface, is never shown in the Settings window, and is
/// never swept by Reset to Defaults: `ttsParametersPanelVisible`,
/// `toolPanelPageShowsRaw`, and `server.cache.mode`/`.window`/`.events.open`.
enum SettingsCatalogue {

    // MARK: - General

    static let launchAtLogin = Setting.bool("launchAtLogin", default: false)
    static let showInDock = Setting.bool("showInDock", default: true)
    static let showInMenuBar = Setting.bool("showInMenuBar", default: true)
    static let autoInsertText = Setting.bool("autoInsertText", default: true)
    static let restoreClipboard = Setting.bool("restoreClipboard", default: true)
    /// The dictation **Proofread Pass** (ADR-0034): the small co-resident
    /// model polishes each transcription when the GPU is free. On by
    /// default; the pass silently skips until its model is downloaded.
    static let proofreadDictation = Setting.bool("proofreadDictation", default: true)
    /// Exploration scaffolding (map #283): selects the live Overlay Variant;
    /// deleted when the redesign prunes to one winner.
    static let overlayVariantRaw = Setting.string("overlayVariant", default: "classic")
    static let samplingPresetRaw = Setting.string(
        "samplingPreset", default: SamplingPreset.automatic.rawValue)

    // MARK: - Audio

    static let selectedMicrophoneUID = Setting.string("selectedMicrophoneUID", default: "")
    // Voice Processing (PRD #175) graduated from a toggle to the standard
    // capture mode (PRD #188) — the `voiceProcessingEnabled` key is abandoned,
    // not migrated: reading simply stopped.
    static let captureDumpEnabled = Setting.bool("captureDumpEnabled", default: true)

    // MARK: - Language

    static let language = Setting.string("language", default: "en")
    /// Recently picked dictation languages, newest first, as a
    /// comma-joined code list (e.g. `"uk,de"`). Feeds the status-bar
    /// menu's pinned Language entries so switching back is one click;
    /// `"auto"` and the current selection are pinned separately.
    static let recentDictationLanguages = Setting.string(
        "recentDictationLanguages", default: "")

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
    /// Walking-skeleton scaffolding (map #301, ticket #303): the lived-with
    /// Companion heartbeat prototype. Absorbed or retired by the exit PRDs.
    static let companionHeartbeatEnabled = Setting.bool(
        "companionHeartbeatEnabled", default: false)
    static let companionHeartbeatSpeaks = Setting.bool(
        "companionHeartbeatSpeaks", default: false)
    /// Companion voice-overlay concept picker (ticket #328). Exploration
    /// scaffolding: deleted when the concepts prune to one winner.
    static let companionVoiceConcept = Setting.string(
        "companionVoiceConcept", default: "emissary")
    /// #328 wearing instrument: heartbeat beats summon the picked overlay
    /// concept instead of a banner (banner stays the unanswered fallback).
    static let companionBeatsUseOverlay = Setting.bool(
        "companionBeatsUseOverlay", default: false)

    // MARK: - Memory (ADR-0035, map #314)

    /// The master switch for the living memory system. Off disables capture,
    /// retrieval, and consolidation alike.
    static let memoryEnabled = Setting.bool("memoryEnabled", default: true)
    /// Whether dictated content becomes memory. The owner's explicit call
    /// ("dictated content is your life too", map #314 final grill) — but it is
    /// the one capture source whose text is usually addressed to *other* apps,
    /// so it gets its own switch.
    static let memoryCaptureDictation = Setting.bool("memoryCaptureDictation", default: true)
    /// Whether sleep consolidation may run when the Mac goes idle.
    static let memorySleepEnabled = Setting.bool("memorySleepEnabled", default: true)

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

    /// Render assistant prose as Markdown. Surfaced only by the agent
    /// toolbar's in-context toggle — a mid-conversation mode switch, never
    /// mirrored in the Settings window (#213). Same key the former loose
    /// `@AppStorage` wrote, so existing choices carry over.
    static let agentUseMarkdown = Setting.bool("agentUseMarkdown", default: true)

    /// Per-model setting for the **Preserve-Thinking Render** (issue #98).
    /// Keyed by model ID because the capability is per chat template; the UI
    /// surfaces the toggle only for models whose template declares the flag
    /// (`ModelIdentity.declaredTemplateFlags`). The one dynamic-key setting in
    /// the catalogue — a fixed-key declaration cannot enumerate model IDs.
    /// Shared key prefix for the dynamic per-model keys, so `resetToDefaults`
    /// can sweep them without re-deriving the literal.
    static let preserveThinkingRenderKeyPrefix = "preserveThinkingRender."

    /// Default **on** (#237): a declaring model — Qwen3.6-35B-A3B MoE and its
    /// siblings — is only worth running with preserved thinking, because the
    /// append-stable render lets the coding-agent loop reuse the growing prefix
    /// across turns and auto-disables the expensive per-turn speculative
    /// double-prefill (`speculativeSeedPlan` returns nil under preserve). The
    /// blanket `true` is safe for non-declaring models: `TemplateRenderContext
    /// .resolve` only enables flags the template declares, and the UI toggle is
    /// shown only for declaring models — so a dense model reads `true` here but
    /// renders canonically. The per-model toggle still turns it OFF explicitly.
    static func preserveThinkingRender(modelID: String) -> Setting<Bool> {
        Setting.bool(preserveThinkingRenderKeyPrefix + modelID, default: true)
    }

    // MARK: - Skill Pills (PRD #174)

    /// The "Show skill button" opt-out for the Skill Cluster above the agent
    /// composer (ADR-0030; same stored key as the retired pill row). Default
    /// on; the cluster also hides itself when no skill declares pill
    /// membership.
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

    /// Exposes the **Browser MCP Server** (`/mcp`) on the running HTTP server so
    /// agents can drive the **Agent Browser**. This is the *HTTP exposure* switch
    /// — it gates only the loopback `/mcp` listener that admits outside clients
    /// (Claude Code, OpenCode). The in-app agent's own browser-use is governed by
    /// the separate *Web Access* switch (`webAccessEnabled`) over the in-process
    /// transport, so the two are independent (ADR-0028). On by default; the origin
    /// guard fails closed on non-loopback requests.
    static let browserMCPServerEnabled = Setting.bool("browserMCPServerEnabled", default: true)

    /// Local-only usage telemetry for the Browser MCP tools (ADR-0031): one
    /// JSONL event per tool call (arguments, latency, outcome, result shape,
    /// screenshot dimensions) under Application Support, for offline analysis
    /// that improves the tools. Nothing ever leaves the Mac; on by default,
    /// bounded by rotation + 30-day retention.
    static let browserMCPTelemetryEnabled = Setting.bool(
        "browserMCPTelemetryEnabled", default: true)

    /// User-configured MCP servers the in-app agent connects to as an MCP client
    /// (#190). The built-in Browser server is synthesized separately (always
    /// connected in-process) and never stored here. Persisted as JSON;
    /// header values live in the app-sandbox settings for v1 (a Keychain move for
    /// secret headers is the recorded follow-up).
    static let mcpServers = Setting.json("mcpServers", default: [MCPServerConfig]())

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
