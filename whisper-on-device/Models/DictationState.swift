//
//  DictationState.swift
//  whisper-on-device
//

import Foundation

enum DictationState: Equatable, Sendable {
    case idle
    case listening
    case recording
    case processing
    case error(String)

    var isActive: Bool {
        switch self {
        case .listening, .recording, .processing:
            return true
        case .idle, .error:
            return false
        }
    }

    var statusText: String {
        switch self {
        case .idle:
            return "Ready"
        case .listening:
            return "Listening..."
        case .recording:
            return "Recording..."
        case .processing:
            return "Processing..."
        case .error(let message):
            return message
        }
    }
}
