//
//  PillMetrics.swift
//  tesseract
//

import CoreGraphics

/// The single source of truth for the dictation pill's sizes.
///
/// Per-phase sizes are *content layout* consumed by the pill variant
/// (`GlobalOverlayHUD`); the panel's window is fixed at ``canvasSize`` and
/// never resizes (map #283) — SwiftUI animates the pill between the phase
/// sizes inside that canvas.
///
/// `nonisolated` so it escapes the build's MainActor default isolation — that's
/// what lets ``OverlayPlacement`` and its tests run off the main actor.
nonisolated enum PillMetrics {
    static let recordingSize = CGSize(width: 120, height: 32)
    static let processingSize = CGSize(width: 112, height: 34)
    static let errorSize = CGSize(width: 260, height: 44)

    /// The fixed panel canvas: fits the largest pill (`errorSize`) plus
    /// entrance-scale and antialiasing headroom on every side. The pill is
    /// bottom-anchored inside it, so the visual bottom inset is constant
    /// across phases.
    static let canvasSize = CGSize(width: 300, height: 64)

    /// The pill's content size for a given feed phase; `.idle` resolves to
    /// the recording size (the size the pill scales out from).
    static func size(for phase: DictationFeed.Phase) -> CGSize {
        switch phase {
        case .error:
            return errorSize
        case .processing:
            return processingSize
        case .recording, .idle:
            return recordingSize
        }
    }
}
