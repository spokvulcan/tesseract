//
//  SettingsCatalogue+Mac.swift
//  tesseract
//
//  The Mac's own settings: everything the iPhone app doesn't share. The
//  shared ones, and the catalogue's rules, are in `SettingsCatalogue.swift`.
//

import Foundation

extension SettingsCatalogue {

    // MARK: - General

    static let launchAtLogin = Setting.bool("launchAtLogin", default: false)
    static let showInDock = Setting.bool("showInDock", default: true)
    static let showInMenuBar = Setting.bool("showInMenuBar", default: true)
    static let autoInsertText = Setting.bool("autoInsertText", default: true)
    static let restoreClipboard = Setting.bool("restoreClipboard", default: true)
    /// The dictation **Proofread Pass** (ADR-0034): the small co-resident
    /// model polishes each transcription when the GPU is free. Opt-in since
    /// PRD #612: in 1,000 takes it fixed 4 misheard words and changed the
    /// meaning of 13, so **Learned Words** are the corrector now. When on,
    /// the pass silently skips until its model is downloaded.
    static let proofreadDictation = Setting.bool("proofreadDictation", default: false)
    /// Marks the one-time switch-off of the Proofread Pass (PRD #612) as
    /// done, or a choice made since, so a later opt-in is never undone.
    static let proofreadDefaultOffApplied = Setting.bool(
        "proofreadDefaultOffApplied", default: false)
    /// Whether a finished take waits in the **Lens** for ↩ before it pastes
    /// (``CheckBeforePasting``, PRD #612). By default only a take where ⇧
    /// was tapped while talking waits.
    static let checkBeforePastingRaw = Setting.string(
        "checkBeforePasting", default: CheckBeforePasting.whenShiftTapped.rawValue)
    // The Lens replaced the Overlay Variant registry (PRD #612, map #283),
    // so the `overlayVariant` key is abandoned, not migrated: reading
    // simply stopped.
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
    /// The fix hotkey: reopens the last take in the **Lens** to fix a word
    /// by typing the one meant (PRD #612).
    static let fixHotkeyKeyCode = Setting.int(
        "fixHotkeyKeyCode", default: Int(KeyCombo.controlOptionSpace.keyCode))
    static let fixHotkeyModifiers = Setting.int(
        "fixHotkeyModifiers", default: Int(KeyCombo.controlOptionSpace.modifiers))

    // MARK: - Speech Overlay

    static let speechOverlayStyleRaw = Setting.string(
        "speechOverlayStyle", default: SpeechOverlayStyle.island.rawValue)
    static let speechOverlaySizeRaw = Setting.string(
        "speechOverlaySize", default: SpeechOverlaySize.medium.rawValue)
    static let speechOverlayTintRaw = Setting.string(
        "speechOverlayTint", default: SpeechOverlayTint.accent.rawValue)
    static let speechOverlayShowsControls = Setting.bool(
        "speechOverlayShowsControls", default: true)
    static let speechOverlayScopeRaw = Setting.string(
        "speechOverlayScope", default: SpeechOverlayScope.automatic.rawValue)
    static let agentAutoSpeak = Setting.bool("agentAutoSpeak", default: false)
    /// The Companion master switch. The key keeps its skeleton-era name so the
    /// owner's existing opt-in survives each redesign.
    static let companionHeartbeatEnabled = Setting.bool(
        "companionHeartbeatEnabled", default: false)
    /// The one-time launch-at-login ask (ADR-0040 §3): asked when the
    /// Companion is first enabled; a silent login-item flip is a trust
    /// violation, so the answer is always the owner's.
    static let companionLaunchAtLoginAsked = Setting.bool(
        "companionLaunchAtLoginAsked", default: false)
    /// Companion voice-overlay concept picker (ticket #328). Exploration
    /// scaffolding: deleted when the concepts prune to one winner.
    static let companionVoiceConcept = Setting.string(
        "companionVoiceConcept", default: "emissary")
    /// The voice session's taste ledger (#310) — every disputed call ships as
    /// a Setting the owner tunes on the wearing build.
    static let companionVoiceAutoSend = Setting.bool(
        "companionVoiceAutoSend", default: true)
    static let companionVoiceTrailingSilence = Setting.double(
        "companionVoiceTrailingSilence", default: 1.8)
    static let companionVoiceSessionTimeout = Setting.double(
        "companionVoiceSessionTimeout", default: 30)

    // MARK: - Companion (the day)

    /// The owner's Areas: which Reminders lists count as Areas, their names,
    /// and the Inbox list (`AreaMap`, JSON). `{}` means every list is an Area.
    static let companionAreasJSON = Setting.string("companionAreasJSON", default: "{}")
    /// The calendar new events go to; nil is the system default calendar.
    static let companionDefaultCalendarID = Setting.optionalString("companionDefaultCalendarID")
    /// How long before an event its nudge fires.
    static let companionNudgeLeadMinutes = Setting.int("companionNudgeLeadMinutes", default: 10)
    /// The Morning Plan's window: it may run from this hour…
    static let companionMorningStartHour = Setting.int("companionMorningStartHour", default: 4)
    /// …until this hour, local time.
    static let companionMorningEndHour = Setting.int("companionMorningEndHour", default: 12)
    /// When the Evening Wrap-up is due, in minutes after midnight.
    static let companionEveningMinutes = Setting.int("companionEveningMinutes", default: 21 * 60)
    /// How long the owner must be away before coming back is a Breakpoint.
    static let companionBreakpointAwayMinutes = Setting.int(
        "companionBreakpointAwayMinutes", default: 10)
    /// Whether Jarvis may speak urgent lines aloud (the voice rung).
    static let companionSpeaks = Setting.bool("companionSpeaks", default: true)
    /// Quiet hours start (minutes after midnight): Jarvis's own deliveries stop;
    /// the owner's reminders and event nudges still fire.
    static let companionQuietStartMinutes = Setting.int(
        "companionQuietStartMinutes", default: 23 * 60)
    /// Quiet hours end (minutes after midnight).
    static let companionQuietEndMinutes = Setting.int("companionQuietEndMinutes", default: 8 * 60)
    /// Whether Jarvis says when tomorrow starts, once, as quiet hours begin
    /// with the owner still at the Mac.
    static let companionWindDown = Setting.bool("companionWindDown", default: true)
    /// Whether a planned step comes to the owner on the Jarvis Panel when its
    /// slot starts, and checks in at its end once started (Step Cues).
    static let companionStepCues = Setting.bool("companionStepCues", default: true)
    /// Whether Jarvis suggests a break on the Jarvis Panel after two hours at
    /// the Mac without one (Break Cues).
    static let companionBreakCues = Setting.bool("companionBreakCues", default: true)
    /// The owner's notification rules (`[TriageRule]`, JSON), newest first.
    static let companionTriageRulesJSON = Setting.string("companionTriageRulesJSON", default: "[]")
    /// The Day Thread's compaction ceiling, in tokens.
    static let companionThreadCeilingTokens = Setting.int(
        "companionThreadCeilingTokens", default: 64_000)
    /// The capture hotkey: a thought into Reminders from any app. One key —
    /// Right Option alone: tap to type, hold to speak.
    static let captureHotkeyKeyCode = Setting.int(
        "captureHotkeyKeyCode", default: Int(KeyCombo.rightOption.keyCode))
    static let captureHotkeyModifiers = Setting.int(
        "captureHotkeyModifiers", default: Int(KeyCombo.rightOption.modifiers))

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

    /// Which speculative-decoding drafters a model load may attach
    /// (``SpeculationMode``). Replaces the retired `mtpSpeculationEnabled`
    /// bool; `SettingsManager.init` migrates a stored `false` to `.off` once.
    static let speculationModeRaw = Setting.string(
        "speculationMode", default: SpeculationMode.automatic.rawValue)

    /// The **KV Cache Compression** setting (``KVCacheCompression``): the
    /// TurboQuant KV Scheme requests use on a model that supports one.
    /// Defaults to Smallest (`turbo8v4`) while its quality is checked in
    /// long sessions.
    static let kvCacheCompressionRaw = Setting.string(
        "kvCacheCompression", default: KVCacheCompression.turbo8v4.rawValue)

    /// Retired predecessor of ``speculationModeRaw`` — read only by the
    /// one-time migration, never surfaced.
    static let legacyMTPSpeculationEnabled = Setting.bool("mtpSpeculationEnabled", default: true)

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

    // MARK: - Reasoning Effort (ADR-0060)

    /// The agent's **Reasoning Effort** level, as the raw picker value:
    /// `"automatic"` (inject nothing — the model template's own default
    /// applies) or a native level (`low`/`medium`/`xhigh`). Applies only to
    /// models whose template declares the kwarg; changing it re-renders the
    /// first system block, so the next turn re-prefills from token 0.
    static let agentReasoningEffortRaw = Setting.string(
        "agentReasoningEffort", default: "automatic")

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
