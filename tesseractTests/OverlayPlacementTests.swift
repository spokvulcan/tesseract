//
//  OverlayPlacementTests.swift
//  tesseractTests
//

import AppKit
import Testing

@testable import Tesseract_Agent

/// Pure-function tests for ``OverlayPlacement`` — the frame math carved out of the
/// two dictation-overlay controllers during the Overlay Panel carve (#51).
///
/// The placement is a pure function of a hand-built ``ScreenGeometry`` and a
/// `DictationState`, so the suite needs no `NSScreen`, no panel, no running app —
/// and no actor isolation — it asserts "this geometry + this state → this rect".
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
    func pillIsBottomCentredInVisibleFrameWhenRecording() {
        let frame = OverlayPlacement.pill.frame(geometry, .recording)
        // Centred horizontally in the *visible* frame, sitting at the 60pt bottom inset.
        #expect(frame.midX == geometry.visibleFrame.midX)
        #expect(frame.minY == geometry.visibleFrame.minY + 60)
        #expect(frame.size == PillMetrics.recordingSize)
    }

    @Test
    func pillTakesProcessingSizeWhenProcessing() {
        let frame = OverlayPlacement.pill.frame(geometry, .processing)
        #expect(frame.size == PillMetrics.processingSize)
        // Still bottom-centred — only the size differs.
        #expect(frame.midX == geometry.visibleFrame.midX)
        #expect(frame.minY == geometry.visibleFrame.minY + 60)
    }

    @Test
    func pillTakesErrorSizeWhenError() {
        let frame = OverlayPlacement.pill.frame(geometry, .error("boom"))
        #expect(frame.size == PillMetrics.errorSize)
        #expect(frame.midX == geometry.visibleFrame.midX)
        #expect(frame.minY == geometry.visibleFrame.minY + 60)
    }

    @Test
    func pillCentresOnVisibleFrameNotFullFrame() {
        // Guards the frame-vs-visibleFrame mix-up directly: with a horizontally
        // inset Dock the two midXs differ, so centring on the full `frame` would
        // land elsewhere.
        let frame = OverlayPlacement.pill.frame(geometry, .recording)
        #expect(geometry.visibleFrame.midX != geometry.frame.midX)
        #expect(frame.midX == geometry.visibleFrame.midX)
        #expect(frame.midX != geometry.frame.midX)
    }

    @Test
    func pillUsesRecordingSizeForIdleAndListening() {
        // Non-visible states, but the panel is created at the idle size, so the
        // placement must still resolve them — to the recording size, as today.
        #expect(OverlayPlacement.pill.frame(geometry, .idle).size == PillMetrics.recordingSize)
        #expect(OverlayPlacement.pill.frame(geometry, .listening).size == PillMetrics.recordingSize)
    }

    // MARK: - Resize-animation semantics

    @Test
    func pillAnimatesResizeOnShow() {
        #expect(OverlayPlacement.pill.animatesResizeOnShow == true)
    }
}
