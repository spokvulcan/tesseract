//
//  SettingsCatalogue.swift
//  tesseract
//

import Foundation
import TesseractSpeech

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
/// never swept by Reset to Defaults: `toolPanelPageShowsRaw`, and
/// `server.cache.mode`/`.window`/`.events.open`.
///
/// Split by app: this file holds the settings the iPhone app shares with the
/// Mac (the voice and the Reader), and `SettingsCatalogue+Mac.swift` the Mac's
/// own. Both are this one table.
enum SettingsCatalogue {

    // MARK: - TTS

    // The ADR-0072 sampler applies temperature before top-p and counts the
    // repetition penalty over recent frames only, so values tuned for the old
    // one mean something else. Its keys (`ttsTemperature`, `ttsTopP`,
    // `ttsRepetitionPenalty`) are abandoned, not migrated: reading simply
    // stopped, and these start from the new defaults. The defaults are the
    // engine's own, so the app and `v2-listen` never tune different voices.
    private static let ttsDefaults = TTSParameters()
    /// The talker's temperature: how expressive the reading is.
    static let ttsTemperature = Setting.double(
        "ttsTalkerTemperature", default: Double(ttsDefaults.temperature))
    static let ttsTopP = Setting.double("ttsTalkerTopP", default: Double(ttsDefaults.topP))
    static let ttsRepetitionPenalty = Setting.double(
        "ttsTalkerRepetitionPenalty", default: Double(ttsDefaults.repetitionPenalty))
    /// The code predictor's temperature: the acoustic detail that carries
    /// most of the timbre, kept lower so the voice stays steady.
    static let ttsDetailTemperature = Setting.double(
        "ttsDetailTemperature", default: Double(ttsDefaults.detailTemperature))
    static let ttsMaxTokens = Setting.int("ttsMaxTokens", default: ttsDefaults.maxTokens)
    static let ttsSeed = Setting.int("ttsSeed", default: 0)
    static let ttsVoiceDescription = Setting.string("ttsVoiceDescription", default: "")
    static let ttsLanguage = Setting.string("ttsLanguage", default: "English")
    /// Read-aloud speed, time-stretched so the pitch stays. Voice-session
    /// replies always play at 1×.
    static let ttsPlaybackRate = Setting.double("ttsPlaybackRate", default: 1.0)
    /// Designed voices the owner named and saved. User content, so Reset to
    /// Defaults leaves it alone.
    static let savedVoices = Setting.json("speechSavedVoices", default: [SavedVoice]())

    // MARK: - Speech page (the Reader)

    static let readerTextSize = Setting.double("speechReaderTextSize", default: 18)
    static let readerTypefaceRaw = Setting.string(
        "speechReaderTypeface", default: ReaderTypeface.serif.rawValue)
    static let readerHighlightRaw = Setting.string(
        "speechReaderHighlight", default: ReadAlongHighlight.both.rawValue)
}
