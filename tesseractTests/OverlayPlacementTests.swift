//
//  OverlayPlacementTests.swift
//  tesseractTests
//

import AppKit
import Testing

@testable import Tesseract_Agent

/// Pure-function tests for ``OverlayPlacement`` — the frame math carved out of
/// the overlay controllers during the Overlay Panel carve (#51) and made
/// state-free by the Overlay Feed rework (map #283): a placement is a function
/// of ``ScreenGeometry`` alone, so the fixed canvas never moves or resizes with
/// the overlay's state. The Companion voice concepts are its users now.
///
/// Hand-built geometry, no `NSScreen`, no panel, no running app — and no actor
/// isolation — the suite asserts "this geometry → this rect".
struct OverlayPlacementTests {

    /// A non-origin geometry whose `visibleFrame` is inset from `frame` on *both*
    /// axes (menu bar + a left-edge Dock), as a real secondary display's would be.
    /// The horizontal inset (`x 1000→1075`, `width 1920→1845`) shifts
    /// `visibleFrame.midX` (1997.5) off `frame.midX` (1960), so centring
    /// assertions genuinely distinguish the two rects. The vertical inset
    /// (menu bar on top) does the same for `maxY`.
    private let geometry = ScreenGeometry(
        frame: NSRect(x: 1000, y: -200, width: 1920, height: 1080),
        visibleFrame: NSRect(x: 1075, y: -175, width: 1845, height: 1030)
    )

    @Test
    func emissaryCanvasParksInTheTopRightCornerBelowTheMenuBar() {
        let frame = OverlayPlacement.companionEmissary.frame(geometry)
        // Inset from the *visible* frame's right and top edges, where
        // notification banners land; the menu bar pushes it down, never under.
        #expect(frame.maxX == geometry.visibleFrame.maxX - 12)
        #expect(frame.maxY == geometry.visibleFrame.maxY - 12)
        #expect(frame.size == CGSize(width: 420, height: 600))
        #expect(geometry.visibleFrame.contains(frame))
    }

    @Test
    func prosceniumCanvasSpansTheNotchBandOfTheFullFrame() {
        let frame = OverlayPlacement.companionProscenium.frame(geometry)
        // Anchored to the *full* frame: the stage grows out of the notch and
        // the menu-bar band, which the visible frame excludes.
        #expect(frame.midX == geometry.frame.midX)
        #expect(frame.midX != geometry.visibleFrame.midX)
        #expect(frame.maxY == geometry.frame.maxY)
        #expect(frame.size == CGSize(width: 800, height: 340))
    }
}
