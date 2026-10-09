//
//  SettingsManager.swift
//  tesseract
//

import Foundation
import Observation
import ServiceManagement
import AppKit
import MLXLMCommon
import TesseractSpeech

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

    var proofreadDictation: Bool {
        didSet {
            SettingsCatalogue.proofreadDictation.write(proofreadDictation, to: store)
            // A choice made now is never undone by the one-time switch-off.
            SettingsCatalogue.proofreadDefaultOffApplied.write(true, to: store)
        }
    }

    /// The **check-before-pasting** setting (``CheckBeforePasting``). Read
    /// when a take finishes, so a change applies to the next take. Stored
    /// raw so an unrecognized value degrades to `.whenShiftTapped`.
    var checkBeforePastingRaw: String {
        didSet { SettingsCatalogue.checkBeforePastingRaw.write(checkBeforePastingRaw, to: store) }
    }

    var checkBeforePasting: CheckBeforePasting {
        get { CheckBeforePasting(rawValue: checkBeforePastingRaw) ?? .whenShiftTapped }
        set { checkBeforePastingRaw = newValue.rawValue }
    }

    var samplingPresetRaw: String {
        didSet { SettingsCatalogue.samplingPresetRaw.write(samplingPresetRaw, to: store) }
    }

    var samplingPreset: SamplingPreset {
        get { SamplingPreset(rawValue: samplingPresetRaw) ?? .automatic }
        set { samplingPresetRaw = newValue.rawValue }
    }

    // MARK: - Audio Settings

    var selectedMicrophoneUID: String {
        didSet { SettingsCatalogue.selectedMicrophoneUID.write(selectedMicrophoneUID, to: store) }
    }

    /// **Capture Dump** (PRD #175): keep recent dictation recordings on disk
    /// for diagnosing bad transcriptions.
    var captureDumpEnabled: Bool {
        didSet { SettingsCatalogue.captureDumpEnabled.write(captureDumpEnabled, to: store) }
    }

    // MARK: - Language Settings

    var language: String {
        didSet { SettingsCatalogue.language.write(language, to: store) }
    }

    var selectedLanguage: SupportedLanguage {
        SupportedLanguage.language(forCode: language) ?? .auto
    }

    /// Recently picked dictation languages, newest first (comma-joined
    /// codes; see the catalogue entry). Maintained by
    /// `recordRecentDictationLanguage`.
    var recentDictationLanguages: String {
        didSet {
            SettingsCatalogue.recentDictationLanguages.write(
                recentDictationLanguages, to: store)
        }
    }

    /// Remember `code` as the most recent dictation-language pick.
    /// `"auto"` is never recorded — it has a permanent pinned slot.
    func recordRecentDictationLanguage(_ code: String) {
        guard code != SupportedLanguage.auto.code else { return }
        var codes =
            recentDictationLanguages
            .split(separator: ",")
            .map(String.init)
        codes.removeAll { $0 == code }
        codes.insert(code, at: 0)
        recentDictationLanguages = codes.prefix(5).joined(separator: ",")
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

    // MARK: - Appshot Hotkey

    var appshotHotkeyKeyCode: Int {
        didSet { SettingsCatalogue.appshotHotkeyKeyCode.write(appshotHotkeyKeyCode, to: store) }
    }

    var appshotHotkeyModifiers: Int {
        didSet {
            SettingsCatalogue.appshotHotkeyModifiers.write(appshotHotkeyModifiers, to: store)
        }
    }

    var appshotHotkey: KeyCombo {
        get {
            KeyCombo(
                keyCode: UInt16(appshotHotkeyKeyCode),
                modifiers: NSEvent.ModifierFlags(rawValue: UInt(appshotHotkeyModifiers))
            )
        }
        set {
            appshotHotkeyKeyCode = Int(newValue.keyCode)
            appshotHotkeyModifiers = Int(newValue.modifiers)
        }
    }

    // MARK: - Fix Hotkey

    var fixHotkeyKeyCode: Int {
        didSet { SettingsCatalogue.fixHotkeyKeyCode.write(fixHotkeyKeyCode, to: store) }
    }

    var fixHotkeyModifiers: Int {
        didSet { SettingsCatalogue.fixHotkeyModifiers.write(fixHotkeyModifiers, to: store) }
    }

    /// Reopens the last take in the **Lens** to fix a word (⌃⌥Space by default).
    var fixHotkey: KeyCombo {
        get {
            KeyCombo(
                keyCode: UInt16(fixHotkeyKeyCode),
                modifiers: NSEvent.ModifierFlags(rawValue: UInt(fixHotkeyModifiers))
            )
        }
        set {
            fixHotkeyKeyCode = Int(newValue.keyCode)
            fixHotkeyModifiers = Int(newValue.modifiers)
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

    var ttsDetailTemperature: Double {
        didSet { SettingsCatalogue.ttsDetailTemperature.write(ttsDetailTemperature, to: store) }
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

    var ttsPlaybackRate: Double {
        didSet { SettingsCatalogue.ttsPlaybackRate.write(ttsPlaybackRate, to: store) }
    }

    var savedVoices: [SavedVoice] {
        didSet { SettingsCatalogue.savedVoices.write(savedVoices, to: store) }
    }

    // MARK: - Speech Page Settings

    var readerTextSize: Double {
        didSet { SettingsCatalogue.readerTextSize.write(readerTextSize, to: store) }
    }

    var readerTypefaceRaw: String {
        didSet { SettingsCatalogue.readerTypefaceRaw.write(readerTypefaceRaw, to: store) }
    }

    var readerTypeface: ReaderTypeface {
        get { ReaderTypeface(rawValue: readerTypefaceRaw) ?? .serif }
        set { readerTypefaceRaw = newValue.rawValue }
    }

    var readerHighlightRaw: String {
        didSet { SettingsCatalogue.readerHighlightRaw.write(readerHighlightRaw, to: store) }
    }

    var readerHighlight: ReadAlongHighlight {
        get { ReadAlongHighlight(rawValue: readerHighlightRaw) ?? .both }
        set { readerHighlightRaw = newValue.rawValue }
    }

    var speechOverlayStyleRaw: String {
        didSet { SettingsCatalogue.speechOverlayStyleRaw.write(speechOverlayStyleRaw, to: store) }
    }

    var speechOverlayStyle: SpeechOverlayStyle {
        get { SpeechOverlayStyle(rawValue: speechOverlayStyleRaw) ?? .island }
        set { speechOverlayStyleRaw = newValue.rawValue }
    }

    var speechOverlaySizeRaw: String {
        didSet { SettingsCatalogue.speechOverlaySizeRaw.write(speechOverlaySizeRaw, to: store) }
    }

    var speechOverlaySize: SpeechOverlaySize {
        get { SpeechOverlaySize(rawValue: speechOverlaySizeRaw) ?? .medium }
        set { speechOverlaySizeRaw = newValue.rawValue }
    }

    var speechOverlayTintRaw: String {
        didSet { SettingsCatalogue.speechOverlayTintRaw.write(speechOverlayTintRaw, to: store) }
    }

    var speechOverlayTint: SpeechOverlayTint {
        get { SpeechOverlayTint(rawValue: speechOverlayTintRaw) ?? .accent }
        set { speechOverlayTintRaw = newValue.rawValue }
    }

    var speechOverlayShowsControls: Bool {
        didSet {
            SettingsCatalogue.speechOverlayShowsControls.write(
                speechOverlayShowsControls, to: store)
        }
    }

    var speechOverlayScopeRaw: String {
        didSet { SettingsCatalogue.speechOverlayScopeRaw.write(speechOverlayScopeRaw, to: store) }
    }

    var speechOverlayScope: SpeechOverlayScope {
        get { SpeechOverlayScope(rawValue: speechOverlayScopeRaw) ?? .automatic }
        set { speechOverlayScopeRaw = newValue.rawValue }
    }

    var agentAutoSpeak: Bool {
        didSet { SettingsCatalogue.agentAutoSpeak.write(agentAutoSpeak, to: store) }
    }

    var companionHeartbeatEnabled: Bool {
        didSet {
            SettingsCatalogue.companionHeartbeatEnabled.write(
                companionHeartbeatEnabled, to: store)
        }
    }

    var companionLaunchAtLoginAsked: Bool {
        didSet {
            SettingsCatalogue.companionLaunchAtLoginAsked.write(
                companionLaunchAtLoginAsked, to: store)
        }
    }

    var companionVoiceConceptRaw: String {
        didSet {
            SettingsCatalogue.companionVoiceConcept.write(companionVoiceConceptRaw, to: store)
        }
    }

    var companionVoiceAutoSend: Bool {
        didSet {
            SettingsCatalogue.companionVoiceAutoSend.write(companionVoiceAutoSend, to: store)
        }
    }

    var companionVoiceTrailingSilence: Double {
        didSet {
            SettingsCatalogue.companionVoiceTrailingSilence.write(
                companionVoiceTrailingSilence, to: store)
        }
    }

    var companionVoiceSessionTimeout: Double {
        didSet {
            SettingsCatalogue.companionVoiceSessionTimeout.write(
                companionVoiceSessionTimeout, to: store)
        }
    }

    // MARK: - Companion (the day)

    var companionAreasJSON: String {
        didSet { SettingsCatalogue.companionAreasJSON.write(companionAreasJSON, to: store) }
    }

    var companionDefaultCalendarID: String? {
        didSet {
            SettingsCatalogue.companionDefaultCalendarID.write(
                companionDefaultCalendarID, to: store)
        }
    }

    var companionNudgeLeadMinutes: Int {
        didSet {
            SettingsCatalogue.companionNudgeLeadMinutes.write(companionNudgeLeadMinutes, to: store)
        }
    }

    var companionMorningStartHour: Int {
        didSet {
            SettingsCatalogue.companionMorningStartHour.write(companionMorningStartHour, to: store)
        }
    }

    var companionMorningEndHour: Int {
        didSet {
            SettingsCatalogue.companionMorningEndHour.write(companionMorningEndHour, to: store)
        }
    }

    var companionEveningMinutes: Int {
        didSet {
            SettingsCatalogue.companionEveningMinutes.write(companionEveningMinutes, to: store)
        }
    }

    var companionBreakpointAwayMinutes: Int {
        didSet {
            SettingsCatalogue.companionBreakpointAwayMinutes.write(
                companionBreakpointAwayMinutes, to: store)
        }
    }

    var companionSpeaks: Bool {
        didSet { SettingsCatalogue.companionSpeaks.write(companionSpeaks, to: store) }
    }

    var companionQuietStartMinutes: Int {
        didSet {
            SettingsCatalogue.companionQuietStartMinutes.write(
                companionQuietStartMinutes, to: store)
        }
    }

    var companionQuietEndMinutes: Int {
        didSet {
            SettingsCatalogue.companionQuietEndMinutes.write(companionQuietEndMinutes, to: store)
        }
    }

    var companionWindDown: Bool {
        didSet { SettingsCatalogue.companionWindDown.write(companionWindDown, to: store) }
    }

    var companionTriageRulesJSON: String {
        didSet {
            SettingsCatalogue.companionTriageRulesJSON.write(companionTriageRulesJSON, to: store)
        }
    }

    var companionThreadCeilingTokens: Int {
        didSet {
            SettingsCatalogue.companionThreadCeilingTokens.write(
                companionThreadCeilingTokens, to: store)
        }
    }

    var captureHotkeyKeyCode: Int {
        didSet { SettingsCatalogue.captureHotkeyKeyCode.write(captureHotkeyKeyCode, to: store) }
    }

    var captureHotkeyModifiers: Int {
        didSet { SettingsCatalogue.captureHotkeyModifiers.write(captureHotkeyModifiers, to: store) }
    }

    var captureHotkey: KeyCombo {
        get {
            KeyCombo(
                keyCode: UInt16(captureHotkeyKeyCode),
                modifiers: NSEvent.ModifierFlags(rawValue: UInt(captureHotkeyModifiers))
            )
        }
        set {
            captureHotkeyKeyCode = Int(newValue.keyCode)
            captureHotkeyModifiers = Int(newValue.modifiers)
        }
    }

    var selectedAgentModelID: String {
        didSet { SettingsCatalogue.selectedAgentModelID.write(selectedAgentModelID, to: store) }
    }

    var selectedSpeechToTextModelID: String {
        didSet {
            SettingsCatalogue.selectedSpeechToTextModelID.write(
                selectedSpeechToTextModelID, to: store)
        }
    }

    /// The v2 engine's sampler parameters (TesseractSpeech). Deliberately
    /// seed-free: the seed is a per-utterance reproducibility knob
    /// (`SpeechOptions`), never part of the sampler (ADR-0038) — the
    /// coordinator reads `ttsSeed` separately.
    var ttsParameters: TTSParameters {
        get {
            TTSParameters(
                temperature: Float(ttsTemperature),
                topP: Float(ttsTopP),
                repetitionPenalty: Float(ttsRepetitionPenalty),
                detailTemperature: Float(ttsDetailTemperature),
                maxTokens: ttsMaxTokens
            )
        }
        set {
            ttsTemperature = Double(newValue.temperature)
            ttsTopP = Double(newValue.topP)
            ttsRepetitionPenalty = Double(newValue.repetitionPenalty)
            ttsDetailTemperature = Double(newValue.detailTemperature)
            ttsMaxTokens = newValue.maxTokens
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

    // MARK: - Agent Markdown Render

    /// Render assistant prose as Markdown (default on). Toggled only from the
    /// agent toolbar — an in-context mode switch, never mirrored in the
    /// Settings window (#213).
    var agentUseMarkdown: Bool {
        didSet { SettingsCatalogue.agentUseMarkdown.write(agentUseMarkdown, to: store) }
    }

    // MARK: - Agent Vision Mode

    /// Global opt-out "Use vision models when available" (default on, ADR-0013).
    /// Governs chat-initiated loads only: when on, the chat send path requests
    /// `.visionIfCapable` so a vision-capable model loads its VLM container from
    /// turn one; when off, it falls back to `.fromSettings`, which gates vision
    /// on this opt-out (→ text-only). The HTTP server ignores it (ADR-0008).
    /// Load-state upgrades but never downgrades, so flipping this mid-session
    /// takes effect on the next (re)load, not eagerly. The VLM and LLM containers
    /// wrap the same language-model weights and chunk text prefill identically —
    /// measured parity on Qwen3.6-27B PARO (cold 79.3 s vision vs 79.9 s text
    /// over 16,413 tokens; warm 20.9 s vs 21.2 s), so vision's only standing cost
    /// is the resident vision tower (~+1 GB RAM), not prefill speed (ADR-0013).
    var useVisionWhenAvailable: Bool {
        didSet { SettingsCatalogue.useVisionWhenAvailable.write(useVisionWhenAvailable, to: store) }
    }

    /// Speculative-decoding drafter policy (``SpeculationMode``). Gates
    /// *drafter loading*, so changing it takes effect on the next model
    /// (re)load — matching the vision toggle's semantics above. Inert on
    /// checkpoints with no pairing drafter (the common case). Stored raw so
    /// an unrecognized persisted value degrades to `.automatic` instead of
    /// crashing the facade load.
    var speculationModeRaw: String {
        didSet { SettingsCatalogue.speculationModeRaw.write(speculationModeRaw, to: store) }
    }

    var speculationMode: SpeculationMode {
        get { SpeculationMode(rawValue: speculationModeRaw) ?? .automatic }
        set { speculationModeRaw = newValue.rawValue }
    }

    /// The **KV Cache Compression** setting. Read per request, so a change
    /// takes effect on the next turn (in a new cache partition). Stored raw
    /// so an unrecognized value degrades to `.off`.
    var kvCacheCompressionRaw: String {
        didSet { SettingsCatalogue.kvCacheCompressionRaw.write(kvCacheCompressionRaw, to: store) }
    }

    var kvCacheCompression: KVCacheCompression {
        get { KVCacheCompression(rawValue: kvCacheCompressionRaw) ?? .off }
        set { kvCacheCompressionRaw = newValue.rawValue }
    }

    // MARK: - Reasoning Effort (ADR-0060)

    /// Raw picker value for the agent's **Reasoning Effort** —
    /// `"automatic"` or a native level. See `agentReasoningEffort`.
    var agentReasoningEffortRaw: String {
        didSet {
            SettingsCatalogue.agentReasoningEffortRaw.write(agentReasoningEffortRaw, to: store)
        }
    }

    /// Typed view of `agentReasoningEffortRaw`: `nil` means Automatic — no
    /// kwarg is injected and the model template's own default applies.
    var agentReasoningEffort: ReasoningEffort? {
        get { ReasoningEffort(rawValue: agentReasoningEffortRaw) }
        set {
            agentReasoningEffortRaw =
                newValue?.rawValue ?? SettingsCatalogue.agentReasoningEffortRaw.default
        }
    }

    // MARK: - Preserve-Thinking Render (issue #98)

    /// Per-model **Preserve-Thinking Render** opt-in. Method-based rather
    /// than a stored facade property because the key is dynamic (one per
    /// model ID); reads/writes go straight through to the store. The
    /// `preserveThinkingRenderRevision` counter gives SwiftUI something to
    /// observe so a toggle bound through these methods re-renders.
    private(set) var preserveThinkingRenderRevision = 0

    func preserveThinkingRender(modelID: String) -> Bool {
        _ = preserveThinkingRenderRevision
        return SettingsCatalogue.preserveThinkingRender(modelID: modelID).load(from: store)
    }

    func setPreserveThinkingRender(_ enabled: Bool, modelID: String) {
        let setting = SettingsCatalogue.preserveThinkingRender(modelID: modelID)
        guard setting.load(from: store) != enabled else { return }
        setting.write(enabled, to: store)
        preserveThinkingRenderRevision += 1
    }

    // MARK: - Skill Pills (PRD #174)

    /// "Show skill button" — the Skill Cluster's single opt-out.
    var showSkillPills: Bool {
        didSet { SettingsCatalogue.showSkillPills.write(showSkillPills, to: store) }
    }

    /// The `translate` skill's default target language (English display name).
    var translateTargetLanguage: String {
        didSet {
            SettingsCatalogue.translateTargetLanguage.write(translateTargetLanguage, to: store)
        }
    }

    /// Per-skill usage counters for the **Skill Usage Ranking**. Method-based
    /// (dynamic key — one per skill name), reads/writes straight through to the
    /// store; the revision counter gives Observation something to track, same
    /// as the Preserve-Thinking Render pattern.
    private(set) var skillUsageRevision = 0

    func skillUsageCount(skillName: String) -> Int {
        _ = skillUsageRevision
        return SettingsCatalogue.skillUsageCount(skillName: skillName).load(from: store)
    }

    func incrementSkillUsage(skillName: String) {
        let setting = SettingsCatalogue.skillUsageCount(skillName: skillName)
        setting.write(setting.load(from: store) + 1, to: store)
        skillUsageRevision += 1
    }

    // MARK: - Server Settings

    var isServerEnabled: Bool {
        didSet { SettingsCatalogue.isServerEnabled.write(isServerEnabled, to: store) }
    }

    var serverPort: Int {
        didSet { SettingsCatalogue.serverPort.write(serverPort, to: store) }
    }

    /// Whether the **Browser MCP Server** is exposed on the HTTP server. Also
    /// requires the local server itself to be running (they share the one
    /// loopback listener).
    var browserMCPServerEnabled: Bool {
        didSet {
            SettingsCatalogue.browserMCPServerEnabled.write(browserMCPServerEnabled, to: store)
        }
    }

    /// Local-only Browser MCP tool telemetry (ADR-0031). Applies to both entry
    /// paths — the in-app agent and external HTTP clients.
    var browserMCPTelemetryEnabled: Bool {
        didSet {
            SettingsCatalogue.browserMCPTelemetryEnabled.write(
                browserMCPTelemetryEnabled, to: store)
        }
    }

    /// User-configured MCP servers the agent connects to as a client (#190). The
    /// built-in Browser server is synthesized from `browserMCPServerEnabled`, so
    /// it never appears in this list.
    var mcpServers: [MCPServerConfig] {
        didSet { SettingsCatalogue.mcpServers.write(mcpServers, to: store) }
    }

    // MARK: - SSD Prefix Cache

    // Changes to these settings take effect on the next model unload/reload.
    // `LLMActor` snapshots the effective config at load time — the hot path
    // inside `container.perform` cannot await MainActor mid-inference.

    var prefixCacheSSDEnabled: Bool {
        didSet { SettingsCatalogue.prefixCacheSSDEnabled.write(prefixCacheSSDEnabled, to: store) }
    }

    /// User cap on the RAM-tier cache budget (ADR-0018). `nil` =
    /// "Automatic (recommended)". Caps only — a value above the measured
    /// ceiling changes nothing, and pressure retreat always wins.
    var prefixCacheRAMBudgetCapBytes: Int? {
        didSet {
            SettingsCatalogue.prefixCacheRAMBudgetCapBytes.write(
                prefixCacheRAMBudgetCapBytes, to: store)
        }
    }

    /// User cap on the SSD-tier budget (ADR-0018). `nil` = "Automatic
    /// (recommended)": the budget tracks measured free disk space.
    var prefixCacheSSDBudgetCapBytes: Int? {
        didSet {
            SettingsCatalogue.prefixCacheSSDBudgetCapBytes.write(
                prefixCacheSSDBudgetCapBytes, to: store)
        }
    }

    /// Optional override for the SSD root directory. When `nil`, the config
    /// falls back to the sandbox Caches directory. Accepts either a file
    /// URL string or a plain filesystem path. Writing `nil` removes the key.
    var prefixCacheSSDDirectoryOverride: String? {
        didSet {
            SettingsCatalogue.prefixCacheSSDDirectoryOverride.write(
                prefixCacheSSDDirectoryOverride, to: store)
        }
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
        // One-time migration (PRD #612): the Proofread Pass used to be on by
        // default, and Reset to Defaults wrote that `true` down. A stored
        // `true` from before is turned off once; a choice made since (the
        // toggle writes the marker too) stands.
        if !SettingsCatalogue.proofreadDefaultOffApplied.load(from: store),
            SettingsCatalogue.proofreadDictation.load(from: store)
        {
            SettingsCatalogue.proofreadDictation.write(false, to: store)
            SettingsCatalogue.proofreadDefaultOffApplied.write(true, to: store)
        }
        self.proofreadDictation = SettingsCatalogue.proofreadDictation.load(from: store)
        self.checkBeforePastingRaw = SettingsCatalogue.checkBeforePastingRaw.load(from: store)
        self.samplingPresetRaw = SettingsCatalogue.samplingPresetRaw.load(from: store)
        self.selectedMicrophoneUID = SettingsCatalogue.selectedMicrophoneUID.load(from: store)
        self.captureDumpEnabled = SettingsCatalogue.captureDumpEnabled.load(from: store)
        self.language = SettingsCatalogue.language.load(from: store)
        self.recentDictationLanguages =
            SettingsCatalogue.recentDictationLanguages.load(from: store)
        self.hotkeyKeyCode = SettingsCatalogue.hotkeyKeyCode.load(from: store)
        self.hotkeyModifiers = SettingsCatalogue.hotkeyModifiers.load(from: store)
        self.ttsHotkeyKeyCode = SettingsCatalogue.ttsHotkeyKeyCode.load(from: store)
        self.ttsHotkeyModifiers = SettingsCatalogue.ttsHotkeyModifiers.load(from: store)
        self.agentHotkeyKeyCode = SettingsCatalogue.agentHotkeyKeyCode.load(from: store)
        self.agentHotkeyModifiers = SettingsCatalogue.agentHotkeyModifiers.load(from: store)
        self.appshotHotkeyKeyCode = SettingsCatalogue.appshotHotkeyKeyCode.load(from: store)
        self.appshotHotkeyModifiers = SettingsCatalogue.appshotHotkeyModifiers.load(from: store)
        self.fixHotkeyKeyCode = SettingsCatalogue.fixHotkeyKeyCode.load(from: store)
        self.fixHotkeyModifiers = SettingsCatalogue.fixHotkeyModifiers.load(from: store)
        self.ttsTemperature = SettingsCatalogue.ttsTemperature.load(from: store)
        self.ttsTopP = SettingsCatalogue.ttsTopP.load(from: store)
        self.ttsRepetitionPenalty = SettingsCatalogue.ttsRepetitionPenalty.load(from: store)
        self.ttsDetailTemperature = SettingsCatalogue.ttsDetailTemperature.load(from: store)
        self.ttsMaxTokens = SettingsCatalogue.ttsMaxTokens.load(from: store)
        self.ttsSeed = SettingsCatalogue.ttsSeed.load(from: store)
        self.ttsVoiceDescription = SettingsCatalogue.ttsVoiceDescription.load(from: store)
        self.ttsLanguage = SettingsCatalogue.ttsLanguage.load(from: store)
        self.ttsPlaybackRate = SettingsCatalogue.ttsPlaybackRate.load(from: store)
        self.savedVoices = SettingsCatalogue.savedVoices.load(from: store)
        self.readerTextSize = SettingsCatalogue.readerTextSize.load(from: store)
        self.readerTypefaceRaw = SettingsCatalogue.readerTypefaceRaw.load(from: store)
        self.readerHighlightRaw = SettingsCatalogue.readerHighlightRaw.load(from: store)
        self.speechOverlayStyleRaw = SettingsCatalogue.speechOverlayStyleRaw.load(from: store)
        self.speechOverlaySizeRaw = SettingsCatalogue.speechOverlaySizeRaw.load(from: store)
        self.speechOverlayTintRaw = SettingsCatalogue.speechOverlayTintRaw.load(from: store)
        self.speechOverlayShowsControls = SettingsCatalogue.speechOverlayShowsControls.load(
            from: store)
        self.speechOverlayScopeRaw = SettingsCatalogue.speechOverlayScopeRaw.load(from: store)
        self.agentAutoSpeak = SettingsCatalogue.agentAutoSpeak.load(from: store)
        self.companionHeartbeatEnabled = SettingsCatalogue.companionHeartbeatEnabled.load(
            from: store)
        self.companionLaunchAtLoginAsked = SettingsCatalogue.companionLaunchAtLoginAsked.load(
            from: store)
        self.companionVoiceConceptRaw = SettingsCatalogue.companionVoiceConcept.load(from: store)
        self.companionVoiceAutoSend = SettingsCatalogue.companionVoiceAutoSend.load(from: store)
        self.companionVoiceTrailingSilence = SettingsCatalogue.companionVoiceTrailingSilence
            .load(from: store)
        self.companionVoiceSessionTimeout = SettingsCatalogue.companionVoiceSessionTimeout
            .load(from: store)
        self.companionAreasJSON = SettingsCatalogue.companionAreasJSON.load(from: store)
        self.companionDefaultCalendarID = SettingsCatalogue.companionDefaultCalendarID.load(
            from: store)
        self.companionNudgeLeadMinutes = SettingsCatalogue.companionNudgeLeadMinutes.load(
            from: store)
        self.companionMorningStartHour = SettingsCatalogue.companionMorningStartHour.load(
            from: store)
        self.companionMorningEndHour = SettingsCatalogue.companionMorningEndHour.load(from: store)
        self.companionEveningMinutes = SettingsCatalogue.companionEveningMinutes.load(from: store)
        self.companionBreakpointAwayMinutes = SettingsCatalogue.companionBreakpointAwayMinutes.load(
            from: store)
        self.companionSpeaks = SettingsCatalogue.companionSpeaks.load(from: store)
        self.companionQuietStartMinutes = SettingsCatalogue.companionQuietStartMinutes.load(
            from: store)
        self.companionQuietEndMinutes = SettingsCatalogue.companionQuietEndMinutes.load(from: store)
        self.companionWindDown = SettingsCatalogue.companionWindDown.load(from: store)
        self.companionTriageRulesJSON = SettingsCatalogue.companionTriageRulesJSON.load(from: store)
        self.companionThreadCeilingTokens = SettingsCatalogue.companionThreadCeilingTokens.load(
            from: store)
        self.captureHotkeyKeyCode = SettingsCatalogue.captureHotkeyKeyCode.load(from: store)
        self.captureHotkeyModifiers = SettingsCatalogue.captureHotkeyModifiers.load(from: store)
        self.selectedAgentModelID = SettingsCatalogue.selectedAgentModelID.load(from: store)
        self.selectedSpeechToTextModelID = SettingsCatalogue.selectedSpeechToTextModelID.load(
            from: store)
        self.maxRecordingDuration = SettingsCatalogue.maxRecordingDuration.load(from: store)
        self.playSounds = SettingsCatalogue.playSounds.load(from: store)
        self.webAccessEnabled = SettingsCatalogue.webAccessEnabled.load(from: store)
        self.agentUseMarkdown = SettingsCatalogue.agentUseMarkdown.load(from: store)
        self.useVisionWhenAvailable = SettingsCatalogue.useVisionWhenAvailable.load(from: store)
        // One-time migration from the retired `mtpSpeculationEnabled` bool:
        // a persisted opt-out carries over as `.off`; anything else falls
        // through to the catalogue default (`.automatic`). Writes only when
        // the new key has never been written, so an explicit later choice
        // always wins.
        if store.optionalString(for: SettingsCatalogue.speculationModeRaw.key) == nil,
            SettingsCatalogue.legacyMTPSpeculationEnabled.load(from: store) == false
        {
            SettingsCatalogue.speculationModeRaw.write(SpeculationMode.off.rawValue, to: store)
        }
        self.speculationModeRaw = SettingsCatalogue.speculationModeRaw.load(from: store)
        self.kvCacheCompressionRaw = SettingsCatalogue.kvCacheCompressionRaw.load(from: store)
        self.showSkillPills = SettingsCatalogue.showSkillPills.load(from: store)
        self.agentReasoningEffortRaw = SettingsCatalogue.agentReasoningEffortRaw.load(from: store)
        self.translateTargetLanguage = SettingsCatalogue.translateTargetLanguage.load(from: store)
        self.isServerEnabled = SettingsCatalogue.isServerEnabled.load(from: store)
        self.serverPort = SettingsCatalogue.serverPort.load(from: store)
        self.browserMCPServerEnabled = SettingsCatalogue.browserMCPServerEnabled.load(from: store)
        self.browserMCPTelemetryEnabled = SettingsCatalogue.browserMCPTelemetryEnabled.load(
            from: store)
        self.mcpServers = SettingsCatalogue.mcpServers.load(from: store)
        self.prefixCacheSSDEnabled = SettingsCatalogue.prefixCacheSSDEnabled.load(from: store)
        self.prefixCacheRAMBudgetCapBytes = SettingsCatalogue.prefixCacheRAMBudgetCapBytes.load(
            from: store)
        self.prefixCacheSSDBudgetCapBytes = SettingsCatalogue.prefixCacheSSDBudgetCapBytes.load(
            from: store)
        self.prefixCacheSSDDirectoryOverride = SettingsCatalogue.prefixCacheSSDDirectoryOverride
            .load(from: store)
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
            budgetCapBytes: prefixCacheSSDBudgetCapBytes,
            measuresFreeDisk: true
        )
    }

    /// Build the agent generation parameters implied by the current settings:
    /// model-derived preset + user sampling override. Live-reads both so a
    /// settings change takes effect on the very next call. Factories should
    /// prefer this over assembling the pieces inline to keep the ordering and
    /// sources canonical.
    func makeAgentGenerateParameters() -> AgentGenerateParameters {
        var parameters = AgentGenerateParameters.forModel(selectedAgentModelID)
        parameters = samplingPreset.apply(to: parameters)
        parameters.reasoningEffort = agentReasoningEffort
        parameters.kvScheme = kvCacheCompression.scheme(forModelID: selectedAgentModelID)
        return parameters
    }

    /// The SSD prefix-cache root directory — the size/wipe target for the
    /// status-bar menu's "Clear Disk Cache". Valid even while the SSD tier
    /// is disabled or no model is loaded: the artifacts on disk outlive
    /// both.
    var ssdPrefixCacheRootURL: URL {
        resolvedSSDPrefixCacheRootURL()
    }

    private func resolvedSSDPrefixCacheRootURL() -> URL {
        if let override = prefixCacheSSDDirectoryOverride, !override.isEmpty {
            // Accept either a file URL string or a plain path.
            if let url = URL(string: override), url.isFileURL {
                return url
            }
            return URL(fileURLWithPath: override, isDirectory: true)
        }
        return StorageEnvironment.caches
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
        proofreadDictation = SettingsCatalogue.proofreadDictation.default
        checkBeforePastingRaw = SettingsCatalogue.checkBeforePastingRaw.default
        selectedMicrophoneUID = SettingsCatalogue.selectedMicrophoneUID.default
        captureDumpEnabled = SettingsCatalogue.captureDumpEnabled.default
        language = SettingsCatalogue.language.default
        recentDictationLanguages = SettingsCatalogue.recentDictationLanguages.default
        hotkeyKeyCode = SettingsCatalogue.hotkeyKeyCode.default
        hotkeyModifiers = SettingsCatalogue.hotkeyModifiers.default
        maxRecordingDuration = SettingsCatalogue.maxRecordingDuration.default
        playSounds = SettingsCatalogue.playSounds.default
        ttsHotkeyKeyCode = SettingsCatalogue.ttsHotkeyKeyCode.default
        ttsHotkeyModifiers = SettingsCatalogue.ttsHotkeyModifiers.default
        agentHotkeyKeyCode = SettingsCatalogue.agentHotkeyKeyCode.default
        agentHotkeyModifiers = SettingsCatalogue.agentHotkeyModifiers.default
        appshotHotkeyKeyCode = SettingsCatalogue.appshotHotkeyKeyCode.default
        appshotHotkeyModifiers = SettingsCatalogue.appshotHotkeyModifiers.default
        fixHotkeyKeyCode = SettingsCatalogue.fixHotkeyKeyCode.default
        fixHotkeyModifiers = SettingsCatalogue.fixHotkeyModifiers.default
        ttsTemperature = SettingsCatalogue.ttsTemperature.default
        ttsTopP = SettingsCatalogue.ttsTopP.default
        ttsRepetitionPenalty = SettingsCatalogue.ttsRepetitionPenalty.default
        ttsDetailTemperature = SettingsCatalogue.ttsDetailTemperature.default
        ttsMaxTokens = SettingsCatalogue.ttsMaxTokens.default
        ttsSeed = SettingsCatalogue.ttsSeed.default
        ttsVoiceDescription = SettingsCatalogue.ttsVoiceDescription.default
        ttsLanguage = SettingsCatalogue.ttsLanguage.default
        ttsPlaybackRate = SettingsCatalogue.ttsPlaybackRate.default
        readerTextSize = SettingsCatalogue.readerTextSize.default
        readerTypefaceRaw = SettingsCatalogue.readerTypefaceRaw.default
        readerHighlightRaw = SettingsCatalogue.readerHighlightRaw.default
        speechOverlayStyleRaw = SettingsCatalogue.speechOverlayStyleRaw.default
        speechOverlaySizeRaw = SettingsCatalogue.speechOverlaySizeRaw.default
        speechOverlayTintRaw = SettingsCatalogue.speechOverlayTintRaw.default
        speechOverlayShowsControls = SettingsCatalogue.speechOverlayShowsControls.default
        speechOverlayScopeRaw = SettingsCatalogue.speechOverlayScopeRaw.default
        agentAutoSpeak = SettingsCatalogue.agentAutoSpeak.default
        companionHeartbeatEnabled = SettingsCatalogue.companionHeartbeatEnabled.default
        companionLaunchAtLoginAsked = SettingsCatalogue.companionLaunchAtLoginAsked.default
        companionVoiceConceptRaw = SettingsCatalogue.companionVoiceConcept.default
        companionVoiceAutoSend = SettingsCatalogue.companionVoiceAutoSend.default
        companionVoiceTrailingSilence = SettingsCatalogue.companionVoiceTrailingSilence.default
        companionVoiceSessionTimeout = SettingsCatalogue.companionVoiceSessionTimeout.default
        companionAreasJSON = SettingsCatalogue.companionAreasJSON.default
        companionDefaultCalendarID = SettingsCatalogue.companionDefaultCalendarID.default
        companionNudgeLeadMinutes = SettingsCatalogue.companionNudgeLeadMinutes.default
        companionMorningStartHour = SettingsCatalogue.companionMorningStartHour.default
        companionMorningEndHour = SettingsCatalogue.companionMorningEndHour.default
        companionEveningMinutes = SettingsCatalogue.companionEveningMinutes.default
        companionBreakpointAwayMinutes = SettingsCatalogue.companionBreakpointAwayMinutes.default
        companionSpeaks = SettingsCatalogue.companionSpeaks.default
        companionQuietStartMinutes = SettingsCatalogue.companionQuietStartMinutes.default
        companionQuietEndMinutes = SettingsCatalogue.companionQuietEndMinutes.default
        companionWindDown = SettingsCatalogue.companionWindDown.default
        companionTriageRulesJSON = SettingsCatalogue.companionTriageRulesJSON.default
        companionThreadCeilingTokens = SettingsCatalogue.companionThreadCeilingTokens.default
        captureHotkeyKeyCode = SettingsCatalogue.captureHotkeyKeyCode.default
        captureHotkeyModifiers = SettingsCatalogue.captureHotkeyModifiers.default
        selectedAgentModelID = SettingsCatalogue.selectedAgentModelID.default
        selectedSpeechToTextModelID = SettingsCatalogue.selectedSpeechToTextModelID.default
        webAccessEnabled = SettingsCatalogue.webAccessEnabled.default
        agentUseMarkdown = SettingsCatalogue.agentUseMarkdown.default
        useVisionWhenAvailable = SettingsCatalogue.useVisionWhenAvailable.default
        speculationModeRaw = SettingsCatalogue.speculationModeRaw.default
        kvCacheCompressionRaw = SettingsCatalogue.kvCacheCompressionRaw.default
        showSkillPills = SettingsCatalogue.showSkillPills.default
        translateTargetLanguage = SettingsCatalogue.translateTargetLanguage.default
        samplingPresetRaw = SettingsCatalogue.samplingPresetRaw.default
        agentReasoningEffortRaw = SettingsCatalogue.agentReasoningEffortRaw.default
        isServerEnabled = SettingsCatalogue.isServerEnabled.default
        serverPort = SettingsCatalogue.serverPort.default
        browserMCPServerEnabled = SettingsCatalogue.browserMCPServerEnabled.default
        browserMCPTelemetryEnabled = SettingsCatalogue.browserMCPTelemetryEnabled.default
        mcpServers = SettingsCatalogue.mcpServers.default
        prefixCacheSSDEnabled = SettingsCatalogue.prefixCacheSSDEnabled.default
        prefixCacheRAMBudgetCapBytes = SettingsCatalogue.prefixCacheRAMBudgetCapBytes.default
        prefixCacheSSDBudgetCapBytes = SettingsCatalogue.prefixCacheSSDBudgetCapBytes.default
        prefixCacheSSDDirectoryOverride = SettingsCatalogue.prefixCacheSSDDirectoryOverride.default
        // Dynamic per-model keys are minted on demand and aren't in the static
        // enumeration above; sweep their prefix so a reset truly clears any
        // explicit per-model override and restores the catalogue default
        // (preserve-on where the template declares it, #237).
        store.removeAll(withPrefix: SettingsCatalogue.preserveThinkingRenderKeyPrefix)
        preserveThinkingRenderRevision += 1
        // Per-skill usage counters are minted on demand too — sweep them so a
        // reset restores the curated pill order.
        store.removeAll(withPrefix: SettingsCatalogue.skillUsageCountKeyPrefix)
        skillUsageRevision += 1
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

    /// Stale-value migration (the one deliberate non-hydration step). When a
    /// persisted model selection no longer maps to a known model of its
    /// category, normalise it to the category default. Runs after hydration,
    /// so the re-assignment fires `didSet` and persists through the store for
    /// free.
    private func normalizePersistedSelectionsIfNeeded() {
        let normalizedAgentID = Self.normalizedModelID(
            selectedAgentModelID,
            category: .agent,
            defaultID: ModelDefinition.defaultAgentModelID
        )
        if normalizedAgentID != selectedAgentModelID {
            selectedAgentModelID = normalizedAgentID
        }

        let normalizedSpeechToTextID = Self.normalizedModelID(
            selectedSpeechToTextModelID,
            category: .speechToText,
            defaultID: ModelDefinition.defaultSpeechToTextModelID
        )
        if normalizedSpeechToTextID != selectedSpeechToTextModelID {
            selectedSpeechToTextModelID = normalizedSpeechToTextID
        }
    }

    private static func normalizedModelID(
        _ candidate: String, category: ModelCategory, defaultID: String
    ) -> String {
        let knownIDs = Set(ModelDefinition.ids(in: category))
        if knownIDs.contains(candidate) {
            return candidate
        }
        if knownIDs.contains(defaultID) {
            return defaultID
        }
        return ModelDefinition.models(in: category).first?.id ?? candidate
    }

    func applyDockVisibility() {
        if showInDock {
            NSApp.setActivationPolicy(.regular)
        } else {
            NSApp.setActivationPolicy(.accessory)
        }
    }
}

/// The Mac's speech settings: the read-aloud members the shared speech code
/// reads through `SpeechSettings`.
extension SettingsManager: SpeechSettings {}
