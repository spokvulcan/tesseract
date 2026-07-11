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

    // MARK: - Variant batch (map #283)

    @Test
    func ribbonCanvasHugsTheBottomCentreWithEntranceHeadroom() {
        let frame = OverlayPlacement.ribbon.frame(geometry)
        // The canvas dips `entranceRise` below the resting line so the rise
        // never leaves the fixed window; the resting strip bottom lands at
        // `bottomInset` above the visible frame's floor.
        #expect(frame.midX == geometry.visibleFrame.midX)
        #expect(frame.midX != geometry.frame.midX)
        #expect(
            frame.minY
                == geometry.visibleFrame.minY + RibbonMetrics.bottomInset
                - RibbonMetrics.entranceRise)
        #expect(frame.size == RibbonMetrics.canvasSize)
    }

    @Test
    func orbCanvasParksInTheBottomRightCorner() {
        let frame = OverlayPlacement.orb.frame(geometry)
        // Inset from the *visible* frame's right/bottom edges — a Dock or
        // menu bar must push the orb inward, not underneath.
        #expect(frame.maxX == geometry.visibleFrame.maxX - OrbMetrics.canvasScreenInset)
        #expect(frame.minY == geometry.visibleFrame.minY + OrbMetrics.canvasScreenInset)
        #expect(frame.size == OrbMetrics.canvasSize)
    }

    @Test
    func islandCanvasHangsBelowTheMenuBar() {
        let frame = OverlayPlacement.island.frame(geometry)
        // Top-anchored: the canvas top sits `topInset` below the *visible*
        // frame's top (under the menu bar), never the full frame's.
        #expect(frame.midX == geometry.visibleFrame.midX)
        #expect(frame.maxY == geometry.visibleFrame.maxY - IslandMetrics.topInset)
        #expect(geometry.visibleFrame.maxY != geometry.frame.maxY)
        #expect(frame.maxY != geometry.frame.maxY - IslandMetrics.topInset)
        #expect(frame.size == IslandMetrics.canvasSize)
    }

    @Test
    func whisperCanvasSitsTightAboveTheBottomEdge() {
        let frame = OverlayPlacement.whisper.frame(geometry)
        // 20pt inset — tighter than the pill's 60: the line reads as part of
        // the desktop's floor.
        #expect(frame.midX == geometry.visibleFrame.midX)
        #expect(frame.minY == geometry.visibleFrame.minY + 20)
        #expect(frame.size == WhisperMetrics.canvasSize)
    }

    @Test
    func stageCardCanvasIsBottomCentredBelowThePill() {
        let frame = OverlayPlacement.stageCard.frame(geometry)
        // 52pt inset — lower than the pill's 60: the taller card must not
        // creep toward the screen's centre.
        #expect(frame.midX == geometry.visibleFrame.midX)
        #expect(frame.minY == geometry.visibleFrame.minY + 52)
        #expect(frame.size == StageCardMetrics.canvasSize)
    }

    @Test
    func captionCanvasIsBottomCentredWithRiseHeadroom() {
        let frame = OverlayPlacement.caption.frame(geometry)
        // The canvas dips `riseTravel` below the bar's resting inset so the
        // entrance rise never leaves the fixed window (the ribbon's trick).
        #expect(frame.midX == geometry.visibleFrame.midX)
        #expect(
            frame.minY
                == geometry.visibleFrame.minY + CaptionMetrics.bottomInset
                - CaptionMetrics.riseTravel)
        #expect(frame.size == CaptionMetrics.canvasSize)
    }

    @Test
    func everyRegisteredVariantCanvasFitsTheVisibleFrame() {
        // Whatever the variant, its fixed canvas must land wholly inside the
        // visible frame — an overlay that clips at a screen edge is broken on
        // arrival, before any content draws.
        let placements: [(String, OverlayPlacement)] = [
            ("pill", .pill), ("ribbon", .ribbon), ("orb", .orb),
            ("island", .island), ("whisper", .whisper), ("stageCard", .stageCard),
            ("caption", .caption),
        ]
        for (name, placement) in placements {
            let frame = placement.frame(geometry)
            #expect(
                geometry.visibleFrame.contains(frame),
                "\(name) canvas \(frame) escapes visible frame \(geometry.visibleFrame)")
        }
    }
}
