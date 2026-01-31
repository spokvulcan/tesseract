//
//  WhisperModel.swift
//  whisper-on-device
//

import Foundation

enum WhisperModel: String, CaseIterable, Identifiable, Codable, Sendable {
    case tiny = "openai_whisper-tiny"
    case base = "openai_whisper-base"
    case small = "openai_whisper-small"
    case medium = "openai_whisper-medium"
    case largeV3Turbo = "openai_whisper-large-v3-v20240930_turbo"

    var id: String { rawValue }

    var isRecommended: Bool {
        self == .largeV3Turbo
    }

    enum LanguageSupport: Sendable {
        case english
        case multilingual(count: Int)

        var displayText: String {
            switch self {
            case .english:
                return "English"
            case .multilingual(let count):
                return "\(count) languages"
            }
        }
    }

    var languageSupport: LanguageSupport {
        switch self {
        case .tiny, .base, .small, .medium, .largeV3Turbo:
            return .multilingual(count: 100)
        }
    }

    var displayName: String {
        switch self {
        case .tiny: return "Tiny"
        case .base: return "Base"
        case .small: return "Small"
        case .medium: return "Medium"
        case .largeV3Turbo: return "Large V3 Turbo"
        }
    }

    var sizeGB: Double {
        switch self {
        case .tiny: return 0.075
        case .base: return 0.145
        case .small: return 0.465
        case .medium: return 1.5
        case .largeV3Turbo: return 0.632
        }
    }

    var recommendedRAMGB: Int {
        switch self {
        case .tiny: return 1
        case .base: return 2
        case .small: return 4
        case .medium: return 8
        case .largeV3Turbo: return 8
        }
    }

    var description: String {
        switch self {
        case .tiny:
            return "Fastest, lowest accuracy. Good for quick dictation."
        case .base:
            return "Good balance of speed and accuracy for most users."
        case .small:
            return "Higher accuracy, moderate speed."
        case .medium:
            return "High accuracy, slower transcription. Requires 8GB+ RAM."
        case .largeV3Turbo:
            return "Best balance of accuracy and speed. State-of-the-art multilingual model."
        }
    }
}
