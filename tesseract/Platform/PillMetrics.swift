//
//  PillMetrics.swift
//  tesseract
//

import CoreGraphics

/// The single source of truth for the dictation pill's per-state size.
///
/// Both the panel frame math (``OverlayPlacement/pill``, which sizes the NSPanel)
/// and the rendered HUD (`GlobalOverlayHUD`, which sizes the pill it hosts) read
/// from here, so the panel frame and its content can't silently diverge. A plain
/// non-isolated value (no `NSScreen`, no `@MainActor`), which is what lets the
/// frame math stay a pure, off-main-testable function.
///
/// `nonisolated` so it escapes the build's MainActor default isolation — that's
/// what lets ``OverlayPlacement`` and its tests run off the main actor.
nonisolated enum PillMetrics {
    static let recordingSize = CGSize(width: 120, height: 32)
    static let processingSize = CGSize(width: 112, height: 34)
    static let errorSize = CGSize(width: 260, height: 44)

    /// The pill's size for a given dictation state. The non-visible states
    /// (`.idle`, `.listening`) resolve to the recording size — the size the panel
    /// is created at before its first show.
    static func size(for state: DictationState) -> CGSize {
        switch state {
        case .error:
            return errorSize
        case .processing:
            return processingSize
        case .recording, .listening, .idle:
            return recordingSize
        }
    }
}
