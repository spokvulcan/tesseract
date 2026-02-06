//
//  SpeechError.swift
//  tesseract
//

import Foundation

enum SpeechError: LocalizedError, Sendable {
    case modelNotLoaded
    case modelLoadFailed(String)
    case noTextSelected
    case generationFailed(String)
    case playbackFailed(String)
    case generationTimeout

    var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            "TTS model is not loaded"
        case .modelLoadFailed(let reason):
            "Failed to load TTS model: \(reason)"
        case .noTextSelected:
            "No text selected"
        case .generationFailed(let reason):
            "Speech generation failed: \(reason)"
        case .playbackFailed(let reason):
            "Audio playback failed: \(reason)"
        case .generationTimeout:
            "Speech generation timed out"
        }
    }
}
