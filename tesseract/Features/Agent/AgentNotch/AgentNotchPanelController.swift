//
//  AgentNotchPanelController.swift
//  tesseract
//
//  Manages the NSPanel lifecycle for the agent Dynamic Island overlay.
//  Follows the same pattern as TTSNotchPanelController.
//

import AppKit
import SwiftUI
import os

@MainActor
final class AgentNotchPanelController {
    private var panel: NSPanel?
    let state = AgentNotchState()
    private var isDismissing = false
    private var dismissObserver: Timer?
    private var autoDismissTask: Task<Void, Never>?
    var onTap: (() -> Void)?

    private enum Defaults {
        static let maxWidth: CGFloat = 400
        static let autoDismissDelay: Duration = .milliseconds(2500)
    }

    // MARK: - Public API

    func show() {
        Log.agent.info("[AgentNotch] show()")
        isDismissing = false
        forceClose()

        guard let screen = NSScreen.main else {
            Log.agent.error("[AgentNotch] show() — no NSScreen.main")
            return
        }

        let screenFrame = screen.frame
        let visibleFrame = screen.visibleFrame
        let menuBarHeight = screenFrame.maxY - visibleFrame.maxY

        let frameTracker = NotchFrameTracker()
        frameTracker.screenMidX = screenFrame.midX
        frameTracker.screenMaxY = screenFrame.maxY

        let overlayView = AgentNotchOverlayView(
            state: state,
            menuBarHeight: menuBarHeight,
            frameTracker: frameTracker,
            onTap: { [weak self] in self?.onTap?() }
        )
        let contentView = NSHostingView(rootView: overlayView)

        let initialHeight = menuBarHeight + 40
        let xPosition = screenFrame.midX - Defaults.maxWidth / 2
        let yPosition = screenFrame.maxY - initialHeight

        let panel = NSPanel(
            contentRect: NSRect(x: xPosition, y: yPosition, width: Defaults.maxWidth, height: initialHeight),
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

        state.phase = .listening(audioLevel: 0)
        observeDismiss()
    }

    func updatePhase(_ phase: AgentNotchPhase) {
        guard panel != nil, !isDismissing else { return }
        autoDismissTask?.cancel()
        state.phase = phase

        // Auto-dismiss on complete/error
        if case .complete = phase {
            scheduleAutoDismiss()
        } else if case .error = phase {
            scheduleAutoDismiss()
        }
    }

    func dismiss() {
        guard panel != nil else { return }
        Log.agent.info("[AgentNotch] dismiss()")
        autoDismissTask?.cancel()
        state.shouldDismiss = true

        DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) { [weak self] in
            self?.cleanup()
        }
    }

    func forceClose() {
        autoDismissTask?.cancel()
        dismissObserver?.invalidate()
        dismissObserver = nil
        state.phase = .hidden
        state.shouldDismiss = false
        isDismissing = false
        panel?.orderOut(nil)
        panel = nil
    }

    var isShowing: Bool {
        panel != nil
    }

    // MARK: - Private

    private func scheduleAutoDismiss() {
        autoDismissTask?.cancel()
        autoDismissTask = Task { [weak self] in
            try? await Task.sleep(for: Defaults.autoDismissDelay)
            guard let self, !Task.isCancelled else { return }
            self.dismiss()
        }
    }

    private func cleanup() {
        dismissObserver?.invalidate()
        dismissObserver = nil
        panel?.orderOut(nil)
        panel = nil
        state.phase = .hidden
        state.shouldDismiss = false
        isDismissing = false
    }

    private func observeDismiss() {
        dismissObserver?.invalidate()
        dismissObserver = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self else { return }
                if self.state.shouldDismiss && !self.isDismissing {
                    self.isDismissing = true
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) { [weak self] in
                        self?.cleanup()
                    }
                }
            }
        }
    }
}
