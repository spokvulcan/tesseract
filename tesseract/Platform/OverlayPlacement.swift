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

/// Where an ``OverlayPanel``'s fixed canvas sits for a given screen — a pure
/// value an **Overlay Variant** brings along with its hosted view.
///
/// State-free by design (map #283): the panel frame never changes with
/// dictation state, so per-phase size and motion live entirely in the hosted
/// SwiftUI content.
///
/// `nonisolated` so it escapes the build's MainActor default isolation: a pure
/// value the frame math (and its tests) can use off the main actor.
nonisolated struct OverlayPlacement: Sendable {
    /// Where the fixed canvas sits for a given screen geometry. A pure
    /// function of CoreGraphics rects (the canvas size comes from the
    /// non-isolated ``PillMetrics``), so it carries no actor isolation and
    /// stays off-main-testable.
    let frame: @Sendable (ScreenGeometry) -> NSRect
}

nonisolated extension OverlayPlacement {
    /// The dictation pill's canvas: centred horizontally in the visible
    /// frame, its bottom edge at a fixed inset above the visible frame's
    /// bottom. Sized to the largest pill the variant draws (plus entrance
    /// headroom), so per-phase pill sizes are content layout, never window
    /// resizes.
    static let pill = OverlayPlacement(
        frame: { geometry in
            let size = PillMetrics.canvasSize
            let visible = geometry.visibleFrame
            return NSRect(
                x: visible.midX - size.width / 2,
                y: visible.minY + pillBottomInset,
                width: size.width,
                height: size.height
            )
        }
    )

    /// The pill canvas floats this far above the bottom of the visible frame.
    private static let pillBottomInset: CGFloat = 60
}
