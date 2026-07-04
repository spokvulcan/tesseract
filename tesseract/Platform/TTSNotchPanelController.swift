//
//  TTSNotchPanelController.swift
//  tesseract
//
//  Manages the NSPanel lifecycle for the TTS notch overlay.
//

import AppKit
import SwiftUI
import Combine
import os

@MainActor
final class TTSNotchPanelController: WordHighlightSurface {
    private var panel: NSPanel?
    private let wordTracker = TTSWordTracker()
    /// Re-entrancy guard for `beginDismissal` — every dismissal path converges there,
    /// so the teardown can only be scheduled once per dismissal.
    private var isDismissing = false

    private enum Defaults {
        static let notchWidth: CGFloat = 500
        static let textAreaHeight: CGFloat = 200
        static let maxExtraHeight: CGFloat = 350
        /// Must outlast the overlay view's fade-out (0.15 s content + 0.3 s collapse).
        static let teardownDelay: TimeInterval = 0.4
    }

    // MARK: - Public API

    func show(
        text: String, tokenCharOffsets: [Int], playbackTimeProvider: @escaping () -> TimeInterval
    ) {
        Log.speech.info(
            "[NotchPanel] show() called, text=\(text.prefix(40))…, tokenOffsets=\(tokenCharOffsets.count), isDismissing=\(self.isDismissing)"
        )
        isDismissing = false
        forceClose()

        guard let screen = NSScreen.main else {
            Log.speech.error("[NotchPanel] show() — no NSScreen.main, aborting")
            return
        }

        let screenFrame = screen.frame
        let visibleFrame = screen.visibleFrame
        let menuBarHeight = screenFrame.maxY - visibleFrame.maxY

        Log.speech.info("[NotchPanel] menuBarHeight=\(menuBarHeight)")

        let frameTracker = NotchFrameTracker()
        frameTracker.screenMidX = screenFrame.midX
        frameTracker.screenMaxY = screenFrame.maxY

        let overlayView = TTSNotchOverlayView(
            wordTracker: wordTracker,
            menuBarHeight: menuBarHeight,
            baseTextHeight: Defaults.textAreaHeight,
            maxExtraHeight: Defaults.maxExtraHeight,
            frameTracker: frameTracker,
            requestDismiss: { [weak self] in self?.beginDismissal() }
        )
        let contentView = NSHostingView(rootView: overlayView)

        let targetHeight = menuBarHeight + Defaults.textAreaHeight
        let xPosition = screenFrame.midX - Defaults.notchWidth / 2
        let targetY = screenFrame.maxY - targetHeight

        Log.speech.info(
            "[NotchPanel] panel frame: x=\(xPosition), y=\(targetY), w=\(Defaults.notchWidth), h=\(targetHeight)"
        )

        let panel = NSPanel(
            contentRect: NSRect(
                x: xPosition, y: targetY, width: Defaults.notchWidth, height: targetHeight),
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )

        frameTracker.panel = panel

        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = false
        panel.level = .screenSaver
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]
        panel.ignoresMouseEvents = false
        panel.isReleasedWhenClosed = false
        panel.hidesOnDeactivate = false
        panel.contentView = contentView

        panel.orderFrontRegardless()
        self.panel = panel

        wordTracker.start(
            text: text, tokenCharOffsets: tokenCharOffsets,
            playbackTimeProvider: playbackTimeProvider)
        Log.speech.info("[NotchPanel] show() complete — panel visible, tracker started")
    }

    func updateTotalDuration(_ duration: TimeInterval) {
        wordTracker.updateTotalDuration(duration)
    }

    func switchText(_ text: String, tokenCharOffsets: [Int], segmentBase: TimeInterval) {
        Log.speech.info(
            "[NotchPanel] switchText() — \(text.prefix(40))…, tokenOffsets=\(tokenCharOffsets.count), segmentBase=\(String(format: "%.2f", segmentBase))"
        )
        wordTracker.updateText(text, tokenCharOffsets: tokenCharOffsets, segmentBase: segmentBase)
    }

    func markSegmentComplete() {
        wordTracker.markSegmentComplete()
    }

    func markGenerationComplete() {
        Log.speech.info("[NotchPanel] markGenerationComplete()")
        wordTracker.markGenerationComplete()
    }

    func dismiss() {
        Log.speech.info(
            "[NotchPanel] dismiss() called — panel=\(self.panel != nil), isDismissing=\(self.isDismissing)"
        )
        beginDismissal()
    }

    var isShowing: Bool {
        panel != nil
    }

    // MARK: - Private

    /// The single owned dismissal transition. The port's `dismiss()` and the overlay
    /// view's dismiss requests (close swipe, auto-dismiss after completion) all
    /// converge here: stop the tracker, set its fade state (the view animates on it),
    /// and schedule exactly one deferred teardown, protected by the re-entrancy
    /// guard. The teardown captures the panel it was scheduled for and re-checks
    /// identity, so a `show()` (or a full show-and-dismiss) racing the teardown
    /// window can never have a stale teardown take down the fresh panel.
    private func beginDismissal() {
        guard let dismissingPanel = panel, !isDismissing else {
            Log.speech.info(
                "[NotchPanel] beginDismissal() — skipped (panel=\(self.panel != nil), isDismissing=\(self.isDismissing))"
            )
            return
        }
        isDismissing = true
        wordTracker.stop()
        wordTracker.beginFadeOut()

        DispatchQueue.main.asyncAfter(deadline: .now() + Defaults.teardownDelay) { [weak self] in
            guard let self, self.panel === dismissingPanel else { return }
            Log.speech.info("[NotchPanel] beginDismissal() deferred teardown — removing panel")
            self.panel?.orderOut(nil)
            self.panel = nil
            self.isDismissing = false
        }
    }

    private func forceClose() {
        Log.speech.info("[NotchPanel] forceClose() — panel=\(self.panel != nil)")
        wordTracker.stop()
        panel?.orderOut(nil)
        panel = nil
    }
}
