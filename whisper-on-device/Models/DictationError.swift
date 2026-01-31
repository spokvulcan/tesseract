//
//  DictationError.swift
//  whisper-on-device
//

import Foundation

enum DictationError: LocalizedError, Sendable {
    case microphonePermissionDenied
    case accessibilityPermissionDenied
    case modelNotLoaded
    case modelNotDownloaded
    case audioCaptureFailed(String)
    case transcriptionFailed(String)
    case textInjectionFailed(String)
    case noSpeechDetected
    case recordingTooShort
    case recordingTimeout

    var errorDescription: String? {
        switch self {
        case .microphonePermissionDenied:
            return "Microphone access is required for dictation."
        case .accessibilityPermissionDenied:
            return "Accessibility permission is required for text injection."
        case .modelNotLoaded:
            return "No transcription model is loaded."
        case .modelNotDownloaded:
            return "Please download a transcription model first."
        case .audioCaptureFailed(let reason):
            return "Audio capture failed: \(reason)"
        case .transcriptionFailed(let reason):
            return "Transcription failed: \(reason)"
        case .textInjectionFailed(let reason):
            return "Failed to inject text: \(reason)"
        case .noSpeechDetected:
            return "No speech was detected."
        case .recordingTooShort:
            return "Recording was too short."
        case .recordingTimeout:
            return "Recording exceeded maximum duration."
        }
    }

    var recoverySuggestion: String? {
        switch self {
        case .microphonePermissionDenied:
            return "Open System Settings > Privacy & Security > Microphone and enable access for this app."
        case .accessibilityPermissionDenied:
            return "Open System Settings > Privacy & Security > Accessibility and enable access for this app."
        case .modelNotLoaded, .modelNotDownloaded:
            return "Go to Settings > Model and download a transcription model."
        case .audioCaptureFailed:
            return "Check your microphone connection and try again."
        case .transcriptionFailed:
            return "Try again or select a different model."
        case .textInjectionFailed:
            return "Make sure the target application accepts text input."
        case .noSpeechDetected:
            return "Speak clearly and ensure your microphone is working."
        case .recordingTooShort:
            return "Hold the record button longer or speak more."
        case .recordingTimeout:
            return "Break up longer dictations into smaller segments."
        }
    }
}
