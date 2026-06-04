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
/// `DictationState`, so the suite needs no `NSScreen`, no panel, and no running
/// app — it asserts "this geometry + this state → this rect".
@MainActor
struct OverlayPlacementTests {

    /// A non-origin geometry whose `visibleFrame` is inset from `frame` (menu bar
    /// + Dock), as a real secondary display's would be — so centring math can't
    /// pass by accidentally landing on zero, and the visible-frame inset is
    /// genuinely exercised.
    private let geometry = ScreenGeometry(
        frame: NSRect(x: 1000, y: -200, width: 1920, height: 1080),
        visibleFrame: NSRect(x: 1000, y: -175, width: 1920, height: 1030)
    )

    // MARK: - Full-screen border

    @Test
    func fullScreenBorderFillsTheWholeScreenFrame() {
        let frame = OverlayPlacement.fullScreenBorder.frame(geometry, .recording)
        #expect(frame == geometry.frame)
    }

    // MARK: - Pill

    @Test
    func pillIsBottomCentredInVisibleFrameWhenRecording() {
        let frame = OverlayPlacement.pill.frame(geometry, .recording)
        // Centred horizontally in the *visible* frame, sitting at the 60pt bottom inset.
        #expect(frame.midX == geometry.visibleFrame.midX)
        #expect(frame.minY == geometry.visibleFrame.minY + 60)
        #expect(frame.size == GlobalOverlayHUD.Metrics.recordingSize)
    }

    @Test
    func pillTakesProcessingSizeWhenProcessing() {
        let frame = OverlayPlacement.pill.frame(geometry, .processing)
        #expect(frame.size == GlobalOverlayHUD.Metrics.processingSize)
        // Still bottom-centred — only the size differs.
        #expect(frame.midX == geometry.visibleFrame.midX)
        #expect(frame.minY == geometry.visibleFrame.minY + 60)
    }

    @Test
    func pillTakesErrorSizeWhenError() {
        let frame = OverlayPlacement.pill.frame(geometry, .error("boom"))
        #expect(frame.size == GlobalOverlayHUD.Metrics.errorSize)
        #expect(frame.midX == geometry.visibleFrame.midX)
        #expect(frame.minY == geometry.visibleFrame.minY + 60)
    }

    @Test
    func pillUsesRecordingSizeForIdleAndListening() {
        // Non-visible states, but the panel is created at the idle size, so the
        // placement must still resolve them — to the recording size, as today.
        #expect(OverlayPlacement.pill.frame(geometry, .idle).size == GlobalOverlayHUD.Metrics.recordingSize)
        #expect(OverlayPlacement.pill.frame(geometry, .listening).size == GlobalOverlayHUD.Metrics.recordingSize)
    }

    // MARK: - Reposition-animation semantics

    @Test
    func pillAnimatesRepositionWhileBorderSnaps() {
        #expect(OverlayPlacement.pill.animatesReposition == true)
        #expect(OverlayPlacement.fullScreenBorder.animatesReposition == false)
    }

    // MARK: - Border is state-independent

    @Test
    func fullScreenBorderFillsFrameForEveryState() {
        let states: [DictationState] = [.idle, .listening, .recording, .processing, .error("x")]
        for state in states {
            #expect(OverlayPlacement.fullScreenBorder.frame(geometry, state) == geometry.frame)
        }
    }
}
