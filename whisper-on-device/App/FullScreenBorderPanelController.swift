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
        if !enabled {
            hidePanel()
        }
    }

    private func createPanel() {
        guard let screen = NSScreen.main else { return }

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

    private func handleStateChange(_ state: DictationState) {
        guard isEnabled else { return }

        // Update observable state (SwiftUI view will react automatically)
        overlayState.dictationState = state
        // Also update theme in case it changed
        overlayState.glowTheme = settings.glowTheme

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

        // Reposition to fill main screen in case it changed
        if let screen = NSScreen.main {
            panel.setFrame(screen.frame, display: false)
        }

        panel.orderFront(nil)

        NSAnimationContext.runAnimationGroup { context in
            context.duration = 0.25
            context.timingFunction = CAMediaTimingFunction(name: .easeOut)
            panel.animator().alphaValue = 1
        }
    }

    private func hidePanel() {
        guard let panel = panel else { return }

        NSAnimationContext.runAnimationGroup({ context in
            context.duration = 0.2
            context.timingFunction = CAMediaTimingFunction(name: .easeIn)
            panel.animator().alphaValue = 0
        }, completionHandler: {
            // Ensure orderOut is called on MainActor
            Task { @MainActor [weak panel] in
                panel?.orderOut(nil)
            }
        })
    }
}
