//
//  PhoneSettings.swift
//  tesseract
//
//  The iPhone app's Settings Facade (ADR-0084): what read-aloud needs, over
//  the same Settings Store and Catalogue entries as the Mac's. It follows
//  `SettingsManager`'s rules (ADR-0002): one stored property per setting,
//  declared without a default and hydrated by a direct first assignment, so
//  construction writes nothing; every later change persists in `didSet`.
//

import Foundation
import Observation
import TesseractSpeech

@Observable @MainActor
final class PhoneSettings: SpeechSettings {
    @ObservationIgnored private let store: any SettingsStore

    // MARK: - The voice

    /// The Preset Voice that reads.
    var phoneVoice: String {
        didSet { SettingsCatalogue.phoneVoice.write(phoneVoice, to: store) }
    }

    var presetVoice: String? { phoneVoice }

    /// Whether the voice may download over cellular.
    var allowsCellularDownloads: Bool {
        didSet {
            SettingsCatalogue.phoneCellularDownloads.write(allowsCellularDownloads, to: store)
        }
    }

    var ttsPlaybackRate: Double {
        didSet { SettingsCatalogue.ttsPlaybackRate.write(ttsPlaybackRate, to: store) }
    }

    var ttsVoiceDescription: String {
        didSet { SettingsCatalogue.ttsVoiceDescription.write(ttsVoiceDescription, to: store) }
    }

    var ttsLanguage: String {
        didSet { SettingsCatalogue.ttsLanguage.write(ttsLanguage, to: store) }
    }

    var ttsSeed: Int {
        didSet { SettingsCatalogue.ttsSeed.write(ttsSeed, to: store) }
    }

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

    var savedVoices: [SavedVoice] {
        didSet { SettingsCatalogue.savedVoices.write(savedVoices, to: store) }
    }

    var ttsParameters: TTSParameters {
        TTSParameters(
            temperature: Float(ttsTemperature), topP: Float(ttsTopP),
            repetitionPenalty: Float(ttsRepetitionPenalty),
            detailTemperature: Float(ttsDetailTemperature), maxTokens: ttsMaxTokens)
    }

    // MARK: - The Reader

    var readerTextSize: Double {
        didSet { SettingsCatalogue.readerTextSize.write(readerTextSize, to: store) }
    }

    var readerHighlightRaw: String {
        didSet { SettingsCatalogue.readerHighlightRaw.write(readerHighlightRaw, to: store) }
    }

    var readerHighlight: ReadAlongHighlight {
        get { ReadAlongHighlight(rawValue: readerHighlightRaw) ?? .both }
        set { readerHighlightRaw = newValue.rawValue }
    }

    // MARK: - Init

    init(store: any SettingsStore = UserDefaultsSettingsStore()) {
        self.store = store
        self.phoneVoice = SettingsCatalogue.phoneVoice.load(from: store)
        self.allowsCellularDownloads = SettingsCatalogue.phoneCellularDownloads.load(from: store)
        self.ttsPlaybackRate = SettingsCatalogue.ttsPlaybackRate.load(from: store)
        self.ttsVoiceDescription = SettingsCatalogue.ttsVoiceDescription.load(from: store)
        self.ttsLanguage = SettingsCatalogue.ttsLanguage.load(from: store)
        self.ttsSeed = SettingsCatalogue.ttsSeed.load(from: store)
        self.ttsTemperature = SettingsCatalogue.ttsTemperature.load(from: store)
        self.ttsTopP = SettingsCatalogue.ttsTopP.load(from: store)
        self.ttsRepetitionPenalty = SettingsCatalogue.ttsRepetitionPenalty.load(from: store)
        self.ttsDetailTemperature = SettingsCatalogue.ttsDetailTemperature.load(from: store)
        self.ttsMaxTokens = SettingsCatalogue.ttsMaxTokens.load(from: store)
        self.savedVoices = SettingsCatalogue.savedVoices.load(from: store)
        self.readerTextSize = SettingsCatalogue.readerTextSize.load(from: store)
        self.readerHighlightRaw = SettingsCatalogue.readerHighlightRaw.load(from: store)
    }
}
