//
//  SpeechSettings.swift
//  tesseract
//

import TesseractSpeech

/// The settings read-aloud reads and writes: the voice, its sampler, the
/// speed and the owner's saved voices. The coordinator, the Reader and the
/// voice library take this instead of a whole Settings Facade, so they run
/// in both apps: the Mac's `SettingsManager` conforms, and so will the
/// phone's facade. Each member persists through the Settings Catalogue's
/// shared entries.
@MainActor
protocol SpeechSettings: AnyObject {
    /// Read-aloud speed, time-stretched so the pitch stays.
    var ttsPlaybackRate: Double { get set }
    /// The designed voice's description; empty means the model's own voice.
    var ttsVoiceDescription: String { get set }
    var ttsLanguage: String { get }
    var ttsSeed: Int { get }
    var ttsParameters: TTSParameters { get }
    /// Designed voices the owner named and saved.
    var savedVoices: [SavedVoice] { get set }
}
