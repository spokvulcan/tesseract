//
//  OverlayPanelController.swift
//  whisper-on-device
//

import AppKit
import Combine
import SwiftUI

/// Controls a global floating NSPanel that displays the recording/processing overlay.
/// The panel appears on top of all applications, including full-screen apps.
@MainActor
final class OverlayPanelController {
    private var panel: NSPanel?
    private var hostingView: NSHostingView<GlobalOverlayHUD>?
    private var cancellables = Set<AnyCancellable>()

    /// Shared observable state for the SwiftUI view
    private let overlayState = OverlayState()

    private var isEnabled = true

    private let settings = SettingsManager.shared

    init() {}

    /// Enable or disable this overlay controller
    func setEnabled(_ enabled: Bool) {
        isEnabled = enabled
        if !enabled {
            hidePanel()
        }
    }

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

    private func createPanel() {
        // Create borderless panel with the new smaller size
        let panel = NSPanel(
            contentRect: NSRect(x: 0, y: 0, width: 120, height: 32),
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

        // Non-interactive, transparent
        panel.ignoresMouseEvents = true
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = false  // SwiftUI handles shadow
        panel.hidesOnDeactivate = false

        // Create SwiftUI hosting view with observable state (created once, not replaced)
        let hudView = GlobalOverlayHUD(overlayState: overlayState)
        let hostingView = NSHostingView(rootView: hudView)
        hostingView.frame = panel.contentView?.bounds ?? .zero
        hostingView.autoresizingMask = [.width, .height]

        panel.contentView?.addSubview(hostingView)

        self.panel = panel
        self.hostingView = hostingView

        // Position at bottom center of main screen
        positionPanel()

        // Initially hidden
        panel.alphaValue = 0
    }

    private func positionPanel() {
        guard let panel = panel,
              let screen = NSScreen.main else { return }

        let screenFrame = screen.visibleFrame
        let panelSize = panel.frame.size

        // Bottom center, with some padding from bottom
        let x = screenFrame.midX - panelSize.width / 2
        let y = screenFrame.minY + 60  // 60pt from bottom

        panel.setFrameOrigin(NSPoint(x: x, y: y))
    }

    private func handleStateChange(_ state: DictationState) {
        guard isEnabled else { return }

        // Update observable state (SwiftUI view will react automatically)
        overlayState.dictationState = state

        switch state {
        case .recording, .processing:
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

        // Reposition in case screen changed
        positionPanel()

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
