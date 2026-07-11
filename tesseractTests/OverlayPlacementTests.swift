//
//  OverlayPlacementTests.swift
//  tesseractTests
//

import AppKit
import Testing

@testable import Tesseract_Agent

/// Pure-function tests for ``OverlayPlacement`` — the frame math carved out of
/// the dictation-overlay controllers during the Overlay Panel carve (#51) and
/// made state-free by the Overlay Feed rework (map #283): the placement is a
/// function of ``ScreenGeometry`` alone, so the fixed canvas never moves or
/// resizes with dictation phase.
///
/// Hand-built geometry, no `NSScreen`, no panel, no running app — and no actor
/// isolation — the suite asserts "this geometry → this rect".
struct OverlayPlacementTests {

    /// A non-origin geometry whose `visibleFrame` is inset from `frame` on *both*
    /// axes (menu bar + a left-edge Dock), as a real secondary display's would be.
    /// The horizontal inset (`x 1000→1075`, `width 1920→1845`) shifts
    /// `visibleFrame.midX` (1997.5) off `frame.midX` (1960), so the pill's
    /// centring assertions genuinely distinguish the two rects — an impl that
    /// mistakenly centred on `frame.midX` would fail. The vertical inset keeps
    /// centring from passing by accidentally landing on zero.
    private let geometry = ScreenGeometry(
        frame: NSRect(x: 1000, y: -200, width: 1920, height: 1080),
        visibleFrame: NSRect(x: 1075, y: -175, width: 1845, height: 1030)
    )

    // MARK: - Pill

    @Test
    func pillCanvasIsBottomCentredInVisibleFrame() {
        let frame = OverlayPlacement.pill.frame(geometry)
        // Centred horizontally in the *visible* frame, sitting at the 60pt bottom inset.
        #expect(frame.midX == geometry.visibleFrame.midX)
        #expect(frame.minY == geometry.visibleFrame.minY + 60)
        #expect(frame.size == PillMetrics.canvasSize)
    }

    @Test
    func pillCentresOnVisibleFrameNotFullFrame() {
        // Guards the frame-vs-visibleFrame mix-up directly: with a horizontally
        // inset Dock the two midXs differ, so centring on the full `frame` would
        // land elsewhere.
        let frame = OverlayPlacement.pill.frame(geometry)
        #expect(geometry.visibleFrame.midX != geometry.frame.midX)
        #expect(frame.midX == geometry.visibleFrame.midX)
        #expect(frame.midX != geometry.frame.midX)
    }

    @Test
    func pillCanvasCoversEveryPhaseSize() {
        // The canvas is fixed while the hosted content sizes per phase — so it
        // must be at least as large as every phase's pill, or a phase would
        // clip at the panel edge.
        let canvas = PillMetrics.canvasSize
        for size in [PillMetrics.recordingSize, PillMetrics.processingSize, PillMetrics.errorSize] {
            #expect(size.width <= canvas.width)
            #expect(size.height <= canvas.height)
        }
    }
}
