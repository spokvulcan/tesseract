//
//  OverlayStyle.swift
//  whisper-on-device
//

import Foundation

/// Overlay style options for the dictation visual feedback
enum OverlayStyle: String, CaseIterable, Identifiable {
    case pill              // Existing compact pill overlay
    case fullScreenBorder  // Full-screen Siri-style border

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .pill:
            "Compact Pill"
        case .fullScreenBorder:
            "Full-Screen Border"
        }
    }

    var description: String {
        switch self {
        case .pill:
            "A floating pill at the bottom of the screen"
        case .fullScreenBorder:
            "Animated borders around the entire screen, inspired by iOS 18 Siri"
        }
    }
}
