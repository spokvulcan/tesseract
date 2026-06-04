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
nonisolated struct ScreenGeometry: Equatable, Sendable {
    let frame: NSRect
    let visibleFrame: NSRect
}

/// The whole injected difference between one ``OverlayPanel`` and another.
///
/// A small value carrying a pure `frame(ScreenGeometry, DictationState) -> NSRect`
/// — the only genuinely policy-bearing logic the two overlay controllers held —
/// plus `animatesResizeOnShow`, which governs the show / visible-state-apply path
/// only (screen-change relayout is always instant for both overlays).
///
/// Two presets exist: ``pill`` and ``fullScreenBorder``.
///
/// `nonisolated` so it escapes the build's MainActor default isolation: a pure
/// value the frame math (and its tests) can use off the main actor.
nonisolated struct OverlayPlacement: Sendable {
    /// Where the panel sits for a given screen geometry and dictation state. A pure
    /// function of CoreGraphics rects (sizes come from the non-isolated
    /// ``PillMetrics``), so it carries no actor isolation and stays off-main-testable.
    let frame: @Sendable (ScreenGeometry, DictationState) -> NSRect

    /// Whether the show / visible-state-apply path animates the resize. Only the
    /// pill's frame changes size between states, so only the pill animates; the
    /// border's frame is state-independent and snaps. Screen-change relayout always
    /// snaps regardless (see `OverlayPanel.refreshPanelLayout`).
    let animatesResizeOnShow: Bool
}

nonisolated extension OverlayPlacement {
    /// The dictation HUD: a state-sized rect centred horizontally in the visible
    /// frame, sitting at a fixed bottom inset. Animates its resize on show.
    static let pill = OverlayPlacement(
        frame: { geometry, state in
            let size = PillMetrics.size(for: state)
            let visible = geometry.visibleFrame
            return NSRect(
                x: visible.midX - size.width / 2,
                y: visible.minY + pillBottomInset,
                width: size.width,
                height: size.height
            )
        },
        animatesResizeOnShow: true
    )

    /// The pill floats this far above the bottom of the visible frame.
    private static let pillBottomInset: CGFloat = 60

    /// The full-screen border: fills the whole screen `frame` for any state,
    /// and snaps (no resize animation — its frame never changes size).
    static let fullScreenBorder = OverlayPlacement(
        frame: { geometry, _ in geometry.frame },
        animatesResizeOnShow: false
    )
}
