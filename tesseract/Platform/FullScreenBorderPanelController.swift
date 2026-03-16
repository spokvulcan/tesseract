//
//  FullScreenBorderPanelController.swift
//  tesseract
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
    private var visibilityCheckTask: Task<Void, Never>?

    /// Shared observable state for the SwiftUI view
    private let overlayState = OverlayState()

    private var isEnabled = true

    private let settings: SettingsManager

    init(settings: SettingsManager) {
        self.settings = settings
    }

    /// Creates the overlay panel and starts screen observation.
    /// Call `handleStateChange` and `handleAudioLevelChange` to push values.
    func setup() {
        createPanel()
        startScreenObservation()
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
        guard let screen = OverlayScreenLocator.preferredScreen() else { return }

        // Create full-screen borderless panel
        let panel = NSPanel(
            contentRect: screen.frame,
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )

        // Configure for global overlay behavior
        panel.level = .statusBar
        panel.collectionBehavior = [
            .canJoinAllSpaces,
            .fullScreenAuxiliary,
            .ignoresCycle
        ]
        panel.isReleasedWhenClosed = false

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

    func handleStateChange(_ state: DictationState) {
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

    private func shouldBeVisible(for state: DictationState) -> Bool {
        switch state {
        case .recording, .processing, .error:
            return true
        default:
            return false
        }
    }

    func handleAudioLevelChange(_ level: Float) {
        guard isEnabled else { return }

        // Update observable state (SwiftUI view will react automatically)
        overlayState.audioLevel = level
    }

    func handleGlowThemeChange(_ glowTheme: GlowTheme) {
        overlayState.glowTheme = glowTheme
    }

    private func showPanel() {
        guard let panel = panel else { return }
        hideRequestID &+= 1

        // Reposition to fill main screen in case it changed
        if let screen = OverlayScreenLocator.preferredScreen() {
            panel.setFrame(screen.frame, display: false)
        }

        panel.orderFrontRegardless()

        NSAnimationContext.runAnimationGroup { context in
            context.duration = 0.25
            context.timingFunction = CAMediaTimingFunction(name: .easeOut)
            panel.animator().alphaValue = 1
        }

        scheduleVisibilityCheck()
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

    private func startScreenObservation() {
        NotificationCenter.default.publisher(for: NSApplication.didChangeScreenParametersNotification)
            .receive(on: RunLoop.main)
            .sink { [weak self] _ in
                self?.refreshPanelLayout()
            }
            .store(in: &cancellables)

        NSWorkspace.shared.notificationCenter.publisher(for: NSWorkspace.activeSpaceDidChangeNotification)
            .receive(on: RunLoop.main)
            .sink { [weak self] _ in
                self?.refreshPanelLayout()
            }
            .store(in: &cancellables)

        NSWorkspace.shared.notificationCenter.publisher(for: NSWorkspace.didWakeNotification)
            .receive(on: RunLoop.main)
            .sink { [weak self] _ in
                self?.refreshPanelLayout()
            }
            .store(in: &cancellables)

        NSWorkspace.shared.notificationCenter.publisher(for: NSWorkspace.screensDidWakeNotification)
            .receive(on: RunLoop.main)
            .sink { [weak self] _ in
                self?.refreshPanelLayout()
            }
            .store(in: &cancellables)
    }

    private func refreshPanelLayout() {
        guard isEnabled else { return }
        guard let panel = panel else { return }
        if let screen = OverlayScreenLocator.preferredScreen() {
            panel.setFrame(screen.frame, display: false)
        }
        if shouldBeVisible(for: lastState) {
            panel.orderFrontRegardless()
        }
        ensureVisibleIfNeeded()
    }

    private func scheduleVisibilityCheck() {
        visibilityCheckTask?.cancel()
        visibilityCheckTask = Task { @MainActor [weak self] in
            try? await Task.sleep(for: .milliseconds(200))
            self?.ensureVisibleIfNeeded()
        }
    }

    private func ensureVisibleIfNeeded() {
        guard isEnabled else { return }
        guard shouldBeVisible(for: lastState) else { return }
        guard let panel = panel else { return }
        if !panel.isVisible || !panel.occlusionState.contains(.visible) {
            panel.orderFrontRegardless()
        }
        if panel.alphaValue < 0.95 {
            panel.alphaValue = 1
        }
    }
}
