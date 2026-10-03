//
//  PresetVoice.swift
//  tesseract
//

/// A speaker of the phone's voice, Qwen3-TTS 0.6B CustomVoice: a **Preset
/// Voice**, chosen from the checkpoint's own list and never designed
/// (ADR-0084). Each speaks every one of the voice's languages, best its own.
nonisolated struct PresetVoice: Identifiable, Hashable, Sendable {
    /// The checkpoint's name for the speaker, what the engine is given.
    let id: String
    let name: String
    /// The language the speaker sounds most at home in.
    let language: TTSLanguage
    /// What it sounds like, after Qwen's own descriptions.
    let detail: String

    /// The checkpoint's nine speakers, English ones first.
    static let all: [PresetVoice] = [
        PresetVoice(
            id: "ryan", name: "Ryan", language: .english,
            detail: "A lively male voice with a strong rhythm"),
        PresetVoice(
            id: "aiden", name: "Aiden", language: .english,
            detail: "A sunny American male voice, clear in the midrange"),
        PresetVoice(
            id: "vivian", name: "Vivian", language: .chinese,
            detail: "A bright young female voice with a slight edge"),
        PresetVoice(
            id: "serena", name: "Serena", language: .chinese,
            detail: "A warm, gentle young female voice"),
        PresetVoice(
            id: "uncle_fu", name: "Uncle Fu", language: .chinese,
            detail: "A seasoned male voice, low and mellow"),
        PresetVoice(
            id: "dylan", name: "Dylan", language: .chinese,
            detail: "A youthful Beijing male voice, clear and natural"),
        PresetVoice(
            id: "eric", name: "Eric", language: .chinese,
            detail: "A lively Chengdu male voice, slightly husky"),
        PresetVoice(
            id: "ono_anna", name: "Ono Anna", language: .japanese,
            detail: "A playful Japanese female voice, light and nimble"),
        PresetVoice(
            id: "sohee", name: "Sohee", language: .korean,
            detail: "A warm Korean female voice, rich in feeling"),
    ]

    /// The speaker called `id`, if the voice has one.
    static func named(_ id: String) -> PresetVoice? {
        all.first { $0.id == id }
    }
}
