//
//  OverlayPlacement.swift
//  tesseract
//

import AppKit

/// The plain-rect screen value an ``OverlayPlacement`` consumes — the full
/// screen `frame` and the `visibleFrame` (inset for the menu bar and Dock).
///
/// Lifted from `OverlayScreenLocator.preferredScreen()` by the ``OverlayPanel``,
/// it lets the placement frame math be a pure function of CoreGraphics rects
/// with no live `NSScreen` dependency — which is what makes it unit-testable.
struct ScreenGeometry: Equatable {
    let frame: NSRect
    let visibleFrame: NSRect
}

/// The whole injected difference between one ``OverlayPanel`` and another.
///
/// A small value carrying a pure `frame(ScreenGeometry, DictationState) -> NSRect`
/// — the only genuinely policy-bearing logic the two overlay controllers held —
/// plus `animatesReposition`, which governs the show / visible-state-apply path
/// only (screen-change relayout is always instant for both overlays).
///
/// Two presets exist: ``pill`` and ``fullScreenBorder``.
struct OverlayPlacement {
    /// Where the panel sits for a given screen geometry and dictation state.
    /// `@MainActor` because the pill reads the MainActor-isolated
    /// `GlobalOverlayHUD.Metrics` sizes (the single source of truth for them).
    let frame: @MainActor (ScreenGeometry, DictationState) -> NSRect

    /// Whether the show / visible-state-apply path animates the reposition/resize.
    /// The pill animates; the border snaps. Screen-change relayout ignores this.
    let animatesReposition: Bool
}

extension OverlayPlacement {
    /// The dictation HUD: a state-sized rect centred horizontally in the visible
    /// frame, sitting at a fixed bottom inset. Animates its resize on show.
    static let pill = OverlayPlacement(
        frame: { geometry, state in
            let size = pillSize(for: state)
            let visible = geometry.visibleFrame
            return NSRect(
                x: visible.midX - size.width / 2,
                y: visible.minY + pillBottomInset,
                width: size.width,
                height: size.height
            )
        },
        animatesReposition: true
    )

    /// The pill floats this far above the bottom of the visible frame.
    private static let pillBottomInset: CGFloat = 60

    private static func pillSize(for state: DictationState) -> CGSize {
        switch state {
        case .error:
            return GlobalOverlayHUD.Metrics.errorSize
        case .processing:
            return GlobalOverlayHUD.Metrics.processingSize
        case .recording, .listening, .idle:
            return GlobalOverlayHUD.Metrics.recordingSize
        }
    }

    /// The full-screen border: fills the whole screen `frame` for any state,
    /// and snaps (no reposition animation).
    static let fullScreenBorder = OverlayPlacement(
        frame: { geometry, _ in geometry.frame },
        animatesReposition: false
    )
}
