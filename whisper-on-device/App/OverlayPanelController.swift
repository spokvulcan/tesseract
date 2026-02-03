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
    private var hideRequestID: UInt = 0
    private var lastState: DictationState = .idle
    private let bottomInset: CGFloat = 60

    /// Shared observable state for the SwiftUI view
    private let overlayState = OverlayState()

    private var isEnabled = true

    private let settings = SettingsManager.shared

    init() {}

    /// Enable or disable this overlay controller
    func setEnabled(_ enabled: Bool) {
        isEnabled = enabled
        if enabled {
            applyVisibility(for: lastState)
        } else {
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
        let initialSize = panelSize(for: .idle)

        // Create borderless panel with the new smaller size
        let panel = NSPanel(
            contentRect: NSRect(origin: .zero, size: initialSize),
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
        updatePanelFrame(for: .idle, animated: false)

        // Initially hidden
        panel.alphaValue = 0
    }

    private func updatePanelFrame(for state: DictationState, animated: Bool) {
        guard let panel = panel,
              let screen = screenForOverlay() else { return }

        let screenFrame = screen.visibleFrame
        let size = panelSize(for: state)
        let x = screenFrame.midX - size.width / 2
        let y = screenFrame.minY + bottomInset
        let frame = NSRect(x: x, y: y, width: size.width, height: size.height)

        if animated {
            panel.animator().setFrame(frame, display: false)
        } else {
            panel.setFrame(frame, display: false)
        }
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

        // Reposition in case screen changed
        updatePanelFrame(for: lastState, animated: true)

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

    private func panelSize(for state: DictationState) -> CGSize {
        switch state {
        case .error:
            return GlobalOverlayHUD.Metrics.errorSize
        case .recording, .processing, .listening, .idle:
            return GlobalOverlayHUD.Metrics.pillSize
        }
    }
}
