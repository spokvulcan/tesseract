//
//  DictationState.swift
//  tesseract
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

    /// Whether a dictation overlay (the pill HUD or the full-screen border) should
    /// be on screen for this state. The single source of truth for the show/hide
    /// rule the `OverlayPanel` and both overlay views used to each switch on.
    ///
    /// Deliberately a *different* set from ``isActive``: an overlay shows for
    /// `.error` (to surface the message) but not `.listening` (pre-recording).
    var showsOverlay: Bool {
        switch self {
        case .recording, .processing, .error:
            return true
        case .idle, .listening:
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
