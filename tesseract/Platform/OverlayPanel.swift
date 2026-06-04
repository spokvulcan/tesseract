//
//  OverlayPanel.swift
//  tesseract
//

import AppKit
import Combine
import SwiftUI

/// Controls a transparent, click-through global `NSPanel` that floats above all
/// apps (including full-screen) and reacts to `DictationState`.
///
/// This is the one home for everything the dictation pill HUD and the full-screen
/// border overlay used to share line-for-line: panel creation, the alpha fade with
/// stale-fade cancellation, the four-notification screen observation, the post-show
/// occlusion re-assertion, and the `DictationState`→visible rule. The only injected
/// difference between overlays is the ``OverlayPlacement`` (the frame math) and the
/// hosted SwiftUI `Content`. The two former controllers collapse into two configured
/// instances of this type, wired in `DependencyContainer`.
///
/// Visibility flows through one side-effecting entry, ``handleStateChange(_:)``.
/// Pure view data (`audioLevel`, `glowTheme`) carries no panel-side behaviour, so
/// the caller sets it directly on the exposed ``state`` and the content view reacts.
@MainActor
final class OverlayPanel<Content: View> {
    /// The observable state the caller owns and the hosted content reads. The
    /// caller mutates `audioLevel` / `glowTheme` on it directly; only
    /// `dictationState` (which drives show/hide) flows through a method.
    let state: OverlayState

    private let placement: OverlayPlacement
    private let content: @MainActor (OverlayState) -> Content

    private var panel: NSPanel?
    private var hostingView: NSHostingView<Content>?
    private var cancellables = Set<AnyCancellable>()
    private var hideRequestID: UInt = 0
    private var lastState: DictationState = .idle
    private var visibilityCheckTask: Task<Void, Never>?
    private var isEnabled = true

    /// - Parameters:
    ///   - state: the `OverlayState` the caller owns; seed any initial pure view
    ///     data (e.g. the border's `glowTheme`) on it *before* calling ``setup()``.
    ///   - placement: where the panel sits and whether it animates its reposition.
    ///   - content: builds the hosted SwiftUI view from the state.
    init(
        state: OverlayState,
        placement: OverlayPlacement,
        content: @escaping @MainActor (OverlayState) -> Content
    ) {
        self.state = state
        self.placement = placement
        self.content = content
    }

    /// Creates the overlay panel and starts screen observation. Call
    /// ``handleStateChange(_:)`` to drive show/hide afterward.
    func setup() {
        createPanel()
        startScreenObservation()
    }

    /// Enable or disable this overlay (the pill-vs-border style switch).
    func setEnabled(_ enabled: Bool) {
        isEnabled = enabled
        if enabled {
            applyVisibility(for: lastState)
        } else {
            hidePanel()
        }
    }

    /// The single side-effecting entry: updates dictation state and drives show/hide.
    func handleStateChange(_ dictationState: DictationState) {
        lastState = dictationState
        state.dictationState = dictationState
        guard isEnabled else { return }
        applyVisibility(for: dictationState)
    }

    // MARK: - Panel creation

    private func createPanel() {
        // The initial content rect comes from the placement at `.idle`. If no
        // screen is resolvable yet (no displays at all — unreachable in a running
        // UI app), fall back to a zero geometry; the first show / screen-change
        // relayout repositions the panel onto the real screen.
        let geometry = currentGeometry() ?? ScreenGeometry(frame: .zero, visibleFrame: .zero)
        let initialFrame = placement.frame(geometry, .idle)

        let panel = NSPanel(
            contentRect: initialFrame,
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )

        // Global overlay behaviour: floats above everything, including full-screen.
        panel.level = .statusBar
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .ignoresCycle]
        panel.isReleasedWhenClosed = false

        // Non-interactive (click-through), transparent.
        panel.ignoresMouseEvents = true
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = false
        panel.hidesOnDeactivate = false

        // Hosting view reads `state`, which the caller has already seeded. Created
        // once and never replaced, so SwiftUI animation/`@State` survives updates.
        let hostingView = NSHostingView(rootView: content(state))
        hostingView.frame = panel.contentView?.bounds ?? .zero
        hostingView.autoresizingMask = [.width, .height]
        panel.contentView?.addSubview(hostingView)

        self.panel = panel
        self.hostingView = hostingView

        // Initially hidden.
        panel.alphaValue = 0
    }

    // MARK: - Visibility

    private func applyVisibility(for dictationState: DictationState) {
        if shouldBeVisible(for: dictationState) {
            showPanel()
        } else {
            hidePanel()
        }
    }

    private func shouldBeVisible(for dictationState: DictationState) -> Bool {
        switch dictationState {
        case .recording, .processing, .error:
            return true
        case .idle, .listening:
            return false
        }
    }

    private func showPanel() {
        guard let panel = panel else { return }
        hideRequestID &+= 1

        // Reposition in case the screen changed. The placement decides whether the
        // reposition/resize animates (the pill) or snaps (the border).
        applyFrame(for: lastState, animated: placement.animatesReposition)

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
        // Bump the token so an in-flight fade-out from a previous hide can't order
        // out a panel that has since been re-shown.
        hideRequestID &+= 1
        let requestID = hideRequestID

        NSAnimationContext.runAnimationGroup({ context in
            context.duration = 0.2
            context.timingFunction = CAMediaTimingFunction(name: .easeIn)
            panel.animator().alphaValue = 0
        }, completionHandler: {
            Task { @MainActor [weak self, weak panel] in
                guard let self, requestID == self.hideRequestID else { return }
                panel?.orderOut(nil)
            }
        })
    }

    // MARK: - Frame

    private func applyFrame(for dictationState: DictationState, animated: Bool) {
        guard let panel = panel, let geometry = currentGeometry() else { return }
        let frame = placement.frame(geometry, dictationState)
        if animated {
            panel.animator().setFrame(frame, display: false)
        } else {
            panel.setFrame(frame, display: false)
        }
    }

    /// The single screen seam. Lifts the preferred screen's rects into a
    /// ``ScreenGeometry`` the placement consumes; `nil` only when no display
    /// exists, in which case the reposition paths no-op (keep the current frame).
    private func currentGeometry() -> ScreenGeometry? {
        guard let screen = OverlayScreenLocator.preferredScreen() else { return nil }
        return ScreenGeometry(frame: screen.frame, visibleFrame: screen.visibleFrame)
    }

    // MARK: - Screen observation

    private func startScreenObservation() {
        // didChangeScreenParameters fires on the default center; the space/wake
        // notifications fire on the workspace center.
        let sources: [(NotificationCenter, Notification.Name)] = [
            (.default, NSApplication.didChangeScreenParametersNotification),
            (NSWorkspace.shared.notificationCenter, NSWorkspace.activeSpaceDidChangeNotification),
            (NSWorkspace.shared.notificationCenter, NSWorkspace.didWakeNotification),
            (NSWorkspace.shared.notificationCenter, NSWorkspace.screensDidWakeNotification)
        ]
        for (center, name) in sources {
            center.publisher(for: name)
                .receive(on: DispatchQueue.main)
                .sink { [weak self] _ in self?.refreshPanelLayout() }
                .store(in: &cancellables)
        }
    }

    private func refreshPanelLayout() {
        guard isEnabled else { return }
        guard let panel = panel else { return }
        // Screen-change relayout is always instant — `animatesReposition` governs
        // only the show / visible-state path, not following the active screen.
        applyFrame(for: lastState, animated: false)
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
