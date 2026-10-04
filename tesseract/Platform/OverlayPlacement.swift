//
//  OverlayPlacement.swift
//  tesseract
//

import AppKit

/// The plain-rect screen value an ``OverlayPlacement`` consumes — the full
/// screen `frame` and the `visibleFrame` (inset for the menu bar and Dock).
///
/// Lifted from `OverlayScreenLocator.preferredScreen()` by the panel that
/// places itself, it lets the placement frame math be a pure function of
/// CoreGraphics rects with no live `NSScreen` dependency — which is what makes
/// it unit-testable.
nonisolated struct ScreenGeometry: Equatable, Sendable {
    let frame: NSRect
    let visibleFrame: NSRect
}

/// Where an overlay panel's fixed canvas sits for a given screen — a pure
/// value a Companion voice concept brings along with its hosted view.
///
/// State-free by design (map #283): the panel frame never changes with the
/// overlay's state, so per-state size and motion live entirely in the hosted
/// SwiftUI content.
///
/// `nonisolated` so it escapes the build's MainActor default isolation: a pure
/// value the frame math (and its tests) can use off the main actor.
nonisolated struct OverlayPlacement: Sendable {
    /// Where the fixed canvas sits for a given screen geometry. A pure
    /// function of CoreGraphics rects, so it carries no actor isolation and
    /// stays off-main-testable.
    let frame: @Sendable (ScreenGeometry) -> NSRect
}
