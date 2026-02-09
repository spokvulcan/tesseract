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
final class TTSNotchPanelController {
    private var panel: NSPanel?
    private let wordTracker = TTSWordTracker()
    private var isDismissing = false
    private var dismissObserver: Timer?

    private enum Defaults {
        static let notchWidth: CGFloat = 500
        static let textAreaHeight: CGFloat = 200
        static let maxExtraHeight: CGFloat = 350
    }

    // MARK: - Public API

    func show(text: String, tokenCharOffsets: [Int] = [], playbackTimeProvider: @escaping () -> TimeInterval) {
        Log.speech.info("[NotchPanel] show() called, text=\(text.prefix(40))…, tokenOffsets=\(tokenCharOffsets.count), isDismissing=\(self.isDismissing)")
        isDismissing = false
        forceClose()

        guard let screen = NSScreen.main else {
            Log.speech.error("[NotchPanel] show() — no NSScreen.main, aborting")
            return
        }

        let screenFrame = screen.frame
        let visibleFrame = screen.visibleFrame
        let menuBarHeight = screenFrame.maxY - visibleFrame.maxY

        let normalized = text
            .replacingOccurrences(of: "\n", with: " ")
            .split(omittingEmptySubsequences: true, whereSeparator: { $0.isWhitespace })
            .map { String($0) }
        let words = normalized
        let totalCharCount = normalized.joined(separator: " ").count

        Log.speech.info("[NotchPanel] words=\(words.count), totalCharCount=\(totalCharCount), menuBarHeight=\(menuBarHeight)")

        let frameTracker = NotchFrameTracker()
        frameTracker.screenMidX = screenFrame.midX
        frameTracker.screenMaxY = screenFrame.maxY

        let overlayView = TTSNotchOverlayView(
            words: words,
            totalCharCount: totalCharCount,
            wordTracker: wordTracker,
            menuBarHeight: menuBarHeight,
            baseTextHeight: Defaults.textAreaHeight,
            maxExtraHeight: Defaults.maxExtraHeight,
            frameTracker: frameTracker
        )
        let contentView = NSHostingView(rootView: overlayView)

        let targetHeight = menuBarHeight + Defaults.textAreaHeight
        let xPosition = screenFrame.midX - Defaults.notchWidth / 2
        let targetY = screenFrame.maxY - targetHeight

        Log.speech.info("[NotchPanel] panel frame: x=\(xPosition), y=\(targetY), w=\(Defaults.notchWidth), h=\(targetHeight)")

        let panel = NSPanel(
            contentRect: NSRect(x: xPosition, y: targetY, width: Defaults.notchWidth, height: targetHeight),
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

        wordTracker.start(text: text, tokenCharOffsets: tokenCharOffsets, playbackTimeProvider: playbackTimeProvider)
        observeDismiss()
        Log.speech.info("[NotchPanel] show() complete — panel visible, tracker started")
    }

    func updateTotalDuration(_ duration: TimeInterval) {
        wordTracker.updateTotalDuration(duration)
    }

    func updateText(_ text: String, tokenCharOffsets: [Int] = []) {
        Log.speech.info("[NotchPanel] updateText() — \(text.prefix(40))…, tokenOffsets=\(tokenCharOffsets.count)")
        wordTracker.updateText(text, tokenCharOffsets: tokenCharOffsets)
    }

    func markGenerationComplete() {
        Log.speech.info("[NotchPanel] markGenerationComplete()")
        wordTracker.markGenerationComplete()
    }

    func dismiss() {
        Log.speech.info("[NotchPanel] dismiss() called — panel=\(self.panel != nil), isDismissing=\(self.isDismissing)")
        guard panel != nil else {
            Log.speech.info("[NotchPanel] dismiss() — no panel, skipping")
            return
        }
        wordTracker.shouldDismiss = true
        wordTracker.stop()

        DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) { [weak self] in
            Log.speech.info("[NotchPanel] dismiss() deferred cleanup — removing panel")
            self?.panel?.orderOut(nil)
            self?.panel = nil
            self?.wordTracker.shouldDismiss = false
        }
    }

    var isShowing: Bool {
        panel != nil
    }

    // MARK: - Private

    private func forceClose() {
        Log.speech.info("[NotchPanel] forceClose() — panel=\(self.panel != nil)")
        dismissObserver?.invalidate()
        dismissObserver = nil
        wordTracker.stop()
        panel?.orderOut(nil)
        panel = nil
        wordTracker.shouldDismiss = false
    }

    private func observeDismiss() {
        dismissObserver?.invalidate()
        dismissObserver = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self else { return }
                let shouldDismiss = self.wordTracker.shouldDismiss
                if shouldDismiss && !self.isDismissing {
                    Log.speech.info("[NotchPanel] observeDismiss timer detected shouldDismiss=true, triggering cleanup")
                    self.isDismissing = true
                    self.wordTracker.stop()
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) { [weak self] in
                        Log.speech.info("[NotchPanel] observeDismiss deferred cleanup — removing panel")
                        self?.dismissObserver?.invalidate()
                        self?.dismissObserver = nil
                        self?.panel?.orderOut(nil)
                        self?.panel = nil
                        self?.wordTracker.shouldDismiss = false
                        self?.isDismissing = false
                    }
                }
            }
        }
    }
}
