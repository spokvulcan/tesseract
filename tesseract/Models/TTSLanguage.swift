//
//  TTSLanguage.swift
//  tesseract
//

import NaturalLanguage

nonisolated enum TTSLanguage: String, CaseIterable, Codable, Sendable, Identifiable {
    case english = "English"
    case chinese = "Chinese"
    case japanese = "Japanese"
    case korean = "Korean"
    case german = "German"
    case french = "French"
    case russian = "Russian"
    case portuguese = "Portuguese"
    case spanish = "Spanish"
    case italian = "Italian"

    var id: String { rawValue }

    var displayName: String { rawValue }

    var flag: String {
        switch self {
        case .english: "🇺🇸"
        case .chinese: "🇨🇳"
        case .japanese: "🇯🇵"
        case .korean: "🇰🇷"
        case .german: "🇩🇪"
        case .french: "🇫🇷"
        case .russian: "🇷🇺"
        case .portuguese: "🇧🇷"
        case .spanish: "🇪🇸"
        case .italian: "🇮🇹"
        }
    }

    var naturalLanguage: NLLanguage {
        switch self {
        case .english: .english
        case .chinese: .simplifiedChinese
        case .japanese: .japanese
        case .korean: .korean
        case .german: .german
        case .french: .french
        case .russian: .russian
        case .portuguese: .portuguese
        case .spanish: .spanish
        case .italian: .italian
        }
    }

    /// The voice's language for `text`: the most likely of the ten, judged
    /// from its first few thousand characters, or nil when there is nothing
    /// to judge by. A text in another language reads in the closest of them.
    static func detected(in text: String) -> TTSLanguage? {
        let recognizer = NLLanguageRecognizer()
        recognizer.languageConstraints =
            allCases.map(\.naturalLanguage)
            + [.traditionalChinese]
        recognizer.processString(String(text.prefix(4_000)))
        guard let language = recognizer.dominantLanguage else { return nil }
        if language == .traditionalChinese { return .chinese }
        return allCases.first { $0.naturalLanguage == language }
    }
}
