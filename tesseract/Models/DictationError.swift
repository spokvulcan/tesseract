//
//  DictationError.swift
//  tesseract
//

import Foundation

/// The typed dictation failure vocabulary the Overlay Feed carries end-to-end
/// (map #283): every case that exists has a construction site, and the type
/// reaches the UI intact — variants choose what of `errorDescription` /
/// `recoverySuggestion` to render, instead of receiving a pre-flattened string.
enum DictationError: LocalizedError, Equatable, Sendable {
    case microphonePermissionDenied
    case modelNotLoaded
    case audioCaptureFailed(String)
    case microphoneBusy
    case transcriptionFailed(String)
    case textInjectionFailed(String)
    case noSpeechDetected
    case recordingTooShort
    case transcriptionInProgress

    var errorDescription: String? {
        switch self {
        case .microphonePermissionDenied:
            return "Microphone access is required for dictation."
        case .modelNotLoaded:
            return "No transcription model is loaded."
        case .audioCaptureFailed(let reason):
            return "Audio capture failed: \(reason)"
        case .microphoneBusy:
            return "The microphone is already in use."
        case .transcriptionFailed(let reason):
            return "Transcription failed: \(reason)"
        case .textInjectionFailed(let reason):
            return "Failed to inject text: \(reason)"
        case .noSpeechDetected:
            return "No speech was detected."
        case .recordingTooShort:
            return "Recording was too short."
        case .transcriptionInProgress:
            return "A transcription is already in progress."
        }
    }

    var recoverySuggestion: String? {
        switch self {
        case .microphonePermissionDenied:
            return
                "Open System Settings > Privacy & Security > Microphone and enable access for this app."
        case .modelNotLoaded:
            return "Go to Settings > Model and download a transcription model."
        case .audioCaptureFailed:
            return "Check your microphone connection and try again."
        case .microphoneBusy:
            return "Wait for the current recording to finish, then try again."
        case .transcriptionFailed:
            return "Try again or select a different model."
        case .textInjectionFailed:
            return "Make sure the target application accepts text input."
        case .noSpeechDetected:
            return "Speak clearly and ensure your microphone is working."
        case .recordingTooShort:
            return "Hold the record button longer or speak more."
        case .transcriptionInProgress:
            return "Wait for the current transcription to finish, or cancel it first."
        }
    }
}
