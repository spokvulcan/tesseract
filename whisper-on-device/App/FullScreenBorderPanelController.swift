//
//  FullScreenBorderPanelController.swift
//  whisper-on-device
//

import AppKit
import Combine
import SwiftUI

/// Controls a full-screen NSPanel that displays the Siri-style border overlay.
/// The panel covers the entire screen with animated borders on all four sides.
@MainActor
final class FullScreenBorderPanelController {
    private var panel: NSPanel?
    private var hostingView: NSHostingView<FullScreenBorderOverlayView>?
    private var cancellables = Set<AnyCancellable>()
    private var hideRequestID: UInt = 0
    private var lastState: DictationState = .idle

    /// Shared observable state for the SwiftUI view
    private let overlayState = OverlayState()

    private var isEnabled = true

    private let settings = SettingsManager.shared

    init() {}

    /// Set up the overlay panel with publishers for state and audio level.
    func setup(
        statePublisher: Published<DictationState>.Publisher,
        audioLevelPublisher: Published<Float>.Publisher
    ) {
        createPanel()

        // Subscribe to state changes
        statePublisher
            .receive(on: RunLoop.main)
            .sink { [weak self] state in
                self?.handleStateChange(state)
            }
            .store(in: &cancellables)

        // Subscribe to audio level updates
        audioLevelPublisher
            .receive(on: RunLoop.main)
            .sink { [weak self] level in
                self?.handleAudioLevelChange(level)
            }
            .store(in: &cancellables)
    }

    /// Enable or disable this overlay controller
    func setEnabled(_ enabled: Bool) {
        isEnabled = enabled
        if enabled {
            applyVisibility(for: lastState)
        } else {
            hidePanel()
        }
    }

    private func createPanel() {
        guard let screen = screenForOverlay() else { return }

        // Create full-screen borderless panel
        let panel = NSPanel(
            contentRect: screen.frame,
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )

        // Configure for global overlay behavior
        panel.level = .floating
        panel.collectionBehavior = [
            .canJoinAllSpaces,
            .fullScreenAuxiliary,
            .stationary,
            .ignoresCycle
        ]

        // Non-interactive, transparent (click-through)
        panel.ignoresMouseEvents = true
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = false
        panel.hidesOnDeactivate = false

        // Initialize overlay state with current theme
        overlayState.glowTheme = settings.glowTheme

        // Create SwiftUI hosting view with observable state (created once, not replaced)
        let overlayView = FullScreenBorderOverlayView(overlayState: overlayState)
        let hostingView = NSHostingView(rootView: overlayView)
        hostingView.frame = panel.contentView?.bounds ?? screen.frame
        hostingView.autoresizingMask = [.width, .height]

        panel.contentView?.addSubview(hostingView)

        self.panel = panel
        self.hostingView = hostingView

        // Initially hidden
        panel.alphaValue = 0
    }

    private func screenForOverlay() -> NSScreen? {
        let mouseLocation = NSEvent.mouseLocation
        if let screen = NSScreen.screens.first(where: { $0.frame.contains(mouseLocation) }) {
            return screen
        }
        return NSScreen.main ?? NSScreen.screens.first
    }

    private func handleStateChange(_ state: DictationState) {
        // Update observable state (SwiftUI view will react automatically)
        lastState = state
        overlayState.dictationState = state
        // Also update theme in case it changed
        overlayState.glowTheme = settings.glowTheme
        guard isEnabled else { return }
        applyVisibility(for: state)
    }

    private func applyVisibility(for state: DictationState) {
        switch state {
        case .recording, .processing, .error:
            showPanel()
        default:
            hidePanel()
        }
    }

    private func handleAudioLevelChange(_ level: Float) {
        guard isEnabled else { return }

        // Update observable state (SwiftUI view will react automatically)
        overlayState.audioLevel = level
    }

    private func showPanel() {
        guard let panel = panel else { return }
        hideRequestID &+= 1

        // Reposition to fill main screen in case it changed
        if let screen = screenForOverlay() {
            panel.setFrame(screen.frame, display: false)
        }

        panel.orderFrontRegardless()

        NSAnimationContext.runAnimationGroup { context in
            context.duration = 0.25
            context.timingFunction = CAMediaTimingFunction(name: .easeOut)
            panel.animator().alphaValue = 1
        }
    }

    private func hidePanel() {
        guard let panel = panel else { return }
        hideRequestID &+= 1
        let requestID = hideRequestID

        NSAnimationContext.runAnimationGroup({ context in
            context.duration = 0.2
            context.timingFunction = CAMediaTimingFunction(name: .easeIn)
            panel.animator().alphaValue = 0
        }, completionHandler: {
            // Ensure orderOut is called on MainActor
            Task { @MainActor [weak panel] in
                guard requestID == self.hideRequestID else { return }
                panel?.orderOut(nil)
            }
        })
    }
}
