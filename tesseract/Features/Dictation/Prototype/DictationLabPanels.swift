//
//  DictationLabPanels.swift
//  tesseract
//
//  PROTOTYPE — THROWAWAY (Dictation page redesign, 2026-09-28). Never merge.
//
//  Two floating surfaces the prototypes draw their fix UI in, separate from
//  the Overlay Panel (which stays the click-through recording pill):
//
//  - a key panel that takes the keyboard without activating Tesseract, so the
//    app you dictated into stays in front (Spotlight-style), rebuilt fresh on
//    every show so its text field exists at first layout (the overlay
//    focus-hang note in OverlayAffordance.swift);
//  - a card panel for mouse-only cards and toasts: clickable, never key.
//

import AppKit
import SwiftUI

private final class LabKeyPanel: NSPanel {
    var onResignKey: (() -> Void)?
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { false }
    override func resignKey() {
        super.resignKey()
        onResignKey?()
    }
}

private final class LabCardPanel: NSPanel {
    override var canBecomeKey: Bool { false }
    override var canBecomeMain: Bool { false }
}

@MainActor
final class LabPanelHost {
    private var keyPanel: LabKeyPanel?
    private var cardPanel: LabCardPanel?
    private var cardHide: Task<Void, Never>?

    var isKeyPanelShown: Bool { keyPanel?.isVisible == true }

    // MARK: Key panel

    /// Shows `content` in a fresh key panel. `anchor` is a screen rect to sit
    /// under (a selection); nil centres it in the upper third of the screen
    /// with the mouse.
    func showKey(
        _ content: some View, size: CGSize, anchor: CGRect? = nil, onDismiss: @escaping () -> Void
    ) {
        closeKey()
        let panel = LabKeyPanel(
            contentRect: NSRect(origin: .zero, size: size),
            styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: false)
        panel.isFloatingPanel = true
        panel.level = .statusBar
        panel.hidesOnDeactivate = false
        panel.becomesKeyOnlyIfNeeded = false
        panel.backgroundColor = .clear
        panel.isOpaque = false
        panel.hasShadow = false
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .transient]
        let hosting = NSHostingView(rootView: AnyView(content))
        hosting.frame = NSRect(origin: .zero, size: size)
        panel.contentView = hosting
        panel.setFrameOrigin(Self.origin(for: size, anchor: anchor))
        panel.onResignKey = { [weak self, weak panel] in
            guard let self, let panel, self.keyPanel === panel else { return }
            self.closeKey()
            onDismiss()
        }
        keyPanel = panel
        panel.makeKeyAndOrderFront(nil)
    }

    func closeKey() {
        guard let panel = keyPanel else { return }
        keyPanel = nil
        panel.onResignKey = nil
        panel.orderOut(nil)
    }

    // MARK: Card panel

    /// Shows a click-only card at the bottom centre (above the pill), for
    /// `duration`, or until `closeCard`.
    func showCard(
        _ content: some View, size: CGSize, bottomInset: CGFloat = 120, duration: Duration?
    ) {
        cardHide?.cancel()
        let panel = cardPanel ?? makeCardPanel()
        cardPanel = panel
        let hosting = NSHostingView(rootView: AnyView(content))
        hosting.frame = NSRect(origin: .zero, size: size)
        panel.contentView = hosting
        let visible = Self.activeScreen().visibleFrame
        panel.setFrame(
            NSRect(
                x: visible.midX - size.width / 2, y: visible.minY + bottomInset,
                width: size.width, height: size.height), display: true)
        panel.orderFrontRegardless()
        if let duration {
            cardHide = Task { [weak self] in
                try? await Task.sleep(for: duration)
                guard !Task.isCancelled else { return }
                self?.closeCard()
            }
        }
    }

    func extendCard(by duration: Duration) {
        cardHide?.cancel()
        cardHide = Task { [weak self] in
            try? await Task.sleep(for: duration)
            guard !Task.isCancelled else { return }
            self?.closeCard()
        }
    }

    func holdCard() { cardHide?.cancel() }

    func closeCard() {
        cardHide?.cancel()
        cardPanel?.orderOut(nil)
        cardPanel?.contentView = nil
    }

    private func makeCardPanel() -> LabCardPanel {
        let panel = LabCardPanel(
            contentRect: .zero, styleMask: [.borderless, .nonactivatingPanel], backing: .buffered,
            defer: false)
        panel.isFloatingPanel = true
        panel.level = .statusBar
        panel.hidesOnDeactivate = false
        panel.backgroundColor = .clear
        panel.isOpaque = false
        panel.hasShadow = false
        panel.ignoresMouseEvents = false
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]
        return panel
    }

    // MARK: Geometry

    static func activeScreen() -> NSScreen {
        let mouse = NSEvent.mouseLocation
        return NSScreen.screens.first { $0.frame.contains(mouse) } ?? NSScreen.main
            ?? NSScreen.screens[0]
    }

    private static func origin(for size: CGSize, anchor: CGRect?) -> CGPoint {
        let screen =
            anchor.flatMap { rect in NSScreen.screens.first { $0.frame.intersects(rect) } }
            ?? activeScreen()
        let visible = screen.visibleFrame
        if let anchor {
            var x = anchor.minX - 12
            var y = anchor.minY - size.height - 6
            if y < visible.minY { y = anchor.maxY + 6 }
            x = min(max(x, visible.minX + 8), visible.maxX - size.width - 8)
            y = min(max(y, visible.minY + 8), visible.maxY - size.height - 8)
            return CGPoint(x: x, y: y)
        }
        return CGPoint(
            x: visible.midX - size.width / 2,
            y: visible.minY + visible.height * 0.62 - size.height / 2)
    }
}
