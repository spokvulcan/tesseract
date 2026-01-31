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

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .tiny: return "Tiny"
        case .base: return "Base"
        case .small: return "Small"
        case .medium: return "Medium"
        }
    }

    var sizeGB: Double {
        switch self {
        case .tiny: return 0.075
        case .base: return 0.145
        case .small: return 0.465
        case .medium: return 1.5
        }
    }

    var recommendedRAMGB: Int {
        switch self {
        case .tiny: return 1
        case .base: return 2
        case .small: return 4
        case .medium: return 8
        }
    }

    var description: String {
        switch self {
        case .tiny:
            return "Fastest, lowest accuracy. Good for quick dictation."
        case .base:
            return "Good balance of speed and accuracy. Recommended for most users."
        case .small:
            return "Higher accuracy, moderate speed."
        case .medium:
            return "Best accuracy, slower transcription. Requires 8GB+ RAM."
        }
    }
}
