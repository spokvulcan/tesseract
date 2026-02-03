//
//  OverlayState.swift
//  tesseract
//

import SwiftUI

/// Shared observable state for overlay panels.
/// This allows SwiftUI views to observe state changes without recreating the view hierarchy,
/// which preserves animation state (TimelineView scheduling, @State variables).
@Observable
@MainActor
final class OverlayState {
    var dictationState: DictationState = .idle
    var audioLevel: Float = 0
    var glowTheme: GlowTheme = .appleIntelligence
}
