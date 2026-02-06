//
//  TTSLanguage.swift
//  tesseract
//

enum TTSLanguage: String, CaseIterable, Codable, Sendable, Identifiable {
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
}
