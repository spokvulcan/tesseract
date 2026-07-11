//
//  OverlayPanel.swift
//  tesseract
//

import AppKit
import Combine
import SwiftUI

/// The transparent, click-through global `NSPanel` that hosts the dictation
/// overlay — a dumb host (map #283): it owns panel creation, a fixed
/// per-placement frame, screen-following, and z-order hygiene, and knows
/// nothing about dictation phases. Show/hide is entirely the hosted
/// **Overlay Variant**'s: the variant fades its own content in and out while
/// the panel stays ordered front at a constant frame, so exactly one
/// animation system (SwiftUI's, inside the content) ever runs.
///
/// The frame never animates and never changes with dictation state — it is
/// the placement's fixed canvas, sized to the largest content the variant
/// draws (audit #285 item 3: the old animated `NSPanel` resize raced the
/// un-animated SwiftUI size snap on every stage change).
@MainActor
final class OverlayPanel {

    private var placement: OverlayPlacement
    private let contentAppearance: NSAppearance?

    private var panel: NSPanel?
    private var hostingView: NSHostingView<AnyView>?
    private var cancellables = Set<AnyCancellable>()

    /// - Parameters:
    ///   - placement: the fixed canvas frame for a given screen.
    ///   - contentAppearance: forced `NSAppearance` for the hosted content, or
    ///     `nil` to follow the system. Glass materials read the AppKit
    ///     appearance (not the SwiftUI color scheme), so this is the seam that
    ///     controls how the pill's Liquid Glass renders.
    init(placement: OverlayPlacement, contentAppearance: NSAppearance? = nil) {
        self.placement = placement
        self.contentAppearance = contentAppearance
    }

    /// Creates the panel and starts screen observation. The panel is ordered
    /// front immediately and stays there — with no content (or a variant whose
    /// content is at opacity 0) it is invisible, and materializing the backing
    /// store now keeps first-render cost off the first press.
    func setup() {
        createPanel()
        panel?.orderFrontRegardless()
        startScreenObservation()
    }

    /// Swaps the hosted variant view. Called at launch (the variant rule's
    /// initial emission) and whenever the overlay-variant setting changes.
    func setContent(_ content: AnyView) {
        guard let panel else { return }
        if let hostingView {
            hostingView.rootView = content
            return
        }
        let hosting = NSHostingView(rootView: content)
        if let contentAppearance {
            hosting.appearance = contentAppearance
        }
        hosting.frame = panel.contentView?.bounds ?? .zero
        hosting.autoresizingMask = [.width, .height]
        panel.contentView?.addSubview(hosting)
        hostingView = hosting
    }

    /// Swaps the placement (a variant may bring its own canvas) and relayouts.
    func setPlacement(_ newPlacement: OverlayPlacement) {
        placement = newPlacement
        refreshPanelLayout()
    }

    /// Flips the panel between click-through (the resting state — an
    /// invisible always-front panel must never swallow clicks) and
    /// interactive. Driven by one App Bindings rule for the lingering-beat
    /// affordance window (ticket #289); the panel stays nonactivating, so
    /// clicks never steal focus from the frontmost app.
    func setInteractive(_ interactive: Bool) {
        panel?.ignoresMouseEvents = !interactive
    }

    /// Z-order hygiene on dictation activity: something may have ordered
    /// above the panel since launch, so a press re-asserts front. Pure
    /// command — the caller (one App Bindings rule) decides when.
    func reassertFront() {
        guard let panel else { return }
        if !panel.isVisible || !panel.occlusionState.contains(.visible) {
            panel.orderFrontRegardless()
        }
        DictationPerf.markPanelShown()
    }

    // MARK: - Panel creation

    private func createPanel() {
        // The initial content rect comes from the placement. If no screen is
        // resolvable yet (no displays at all — unreachable in a running UI
        // app), fall back to a zero geometry; the first screen-change relayout
        // repositions the panel onto the real screen.
        let geometry = currentGeometry() ?? ScreenGeometry(frame: .zero, visibleFrame: .zero)
        let initialFrame = placement.frame(geometry)

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

        // Non-interactive (click-through), transparent. No window shadow: the
        // system derives it from the content's alpha silhouette and is known
        // to leave it stale across `setFrame` resizes on transparent windows —
        // behind a translucent glass capsule that reads as a second, offset
        // pill outline. Depth, if wanted, belongs inside the hosted content.
        panel.ignoresMouseEvents = true
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = false
        panel.hidesOnDeactivate = false

        self.panel = panel
    }

    // MARK: - Frame

    /// The screen the panel last resolved to. Refreshed on the screen-change
    /// notifications; resolving is window-server IPC (a full window-list
    /// copy), so it is never done per state change.
    private var cachedGeometry: ScreenGeometry?

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
            (NSWorkspace.shared.notificationCenter, NSWorkspace.screensDidWakeNotification),
        ]
        for (center, name) in sources {
            center.publisher(for: name)
                .receive(on: DispatchQueue.main)
                .sink { [weak self] _ in self?.refreshPanelLayout() }
                .store(in: &cancellables)
        }
    }

    private func refreshPanelLayout() {
        guard let panel else { return }
        cachedGeometry = currentGeometry() ?? cachedGeometry
        guard let geometry = cachedGeometry else { return }
        panel.setFrame(placement.frame(geometry), display: false)
        panel.orderFrontRegardless()
    }
}
