//
//  GlassPanel.swift
//  tesseract
//
//  A floating Liquid Glass panel over every app, in the style of the
//  macOS 27 Siri panel: borderless, non-activating, with an
//  `NSGlassEffectView` as its content view and the SwiftUI content inside.
//  Public API only; the 27 SDK adds no new glass styles, so `.regular` and
//  `.clear` are the choices. The Jarvis panel and the capture panel share it.
//
//  The macOS 27.0 focus freeze (tools/overlay-focus-hang-lab): a focusable
//  control that appears after the first layout freezes the main thread. So
//  hosted content keeps every button `.focusable(false)` and creates its text
//  field in the first layout, never later.
//

import AppKit
import SwiftUI

@MainActor
final class GlassPanel: NSPanel {

    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { false }

    /// Called when the panel should go away (Escape).
    var onCancel: (() -> Void)?

    private let glass: NSGlassEffectView

    /// - Parameters:
    ///   - becomesKeyOnlyIfNeeded: true for a panel that must never steal
    ///     typing from the app in front (the Jarvis panel turns key only when
    ///     its field is clicked); false for one the owner summoned to type in.
    init(
        size: NSSize, cornerRadius: CGFloat = 28, clearGlass: Bool = false,
        becomesKeyOnlyIfNeeded: Bool
    ) {
        glass = NSGlassEffectView(frame: NSRect(origin: .zero, size: size))
        super.init(
            contentRect: NSRect(origin: .zero, size: size),
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered, defer: false)
        isOpaque = false
        backgroundColor = .clear
        hasShadow = true
        level = .floating
        collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        isMovableByWindowBackground = true
        hidesOnDeactivate = false
        isReleasedWhenClosed = false
        self.becomesKeyOnlyIfNeeded = becomesKeyOnlyIfNeeded
        glass.style = clearGlass ? .clear : .regular
        glass.cornerRadius = cornerRadius
        contentView = glass
    }

    /// Install (or replace) the SwiftUI content.
    func host<Content: View>(_ content: Content) {
        let hosting = NSHostingView(rootView: content)
        hosting.frame = glass.bounds
        hosting.autoresizingMask = [.width, .height]
        glass.contentView = hosting
    }

    /// Resize, keeping the top edge where it is.
    func setHeight(_ height: CGFloat) {
        var frame = self.frame
        frame.origin.y += frame.height - height
        frame.size.height = height
        setFrame(frame, display: true, animate: false)
    }

    /// Top-right of the visible frame, the Siri panel's corner.
    func placeTopRight(margin: CGFloat = 16) {
        guard let screen = NSScreen.main else { return }
        let visible = screen.visibleFrame
        setFrameOrigin(
            NSPoint(x: visible.maxX - frame.width - margin, y: visible.maxY - frame.height - margin)
        )
    }

    /// Centred horizontally, a little below the top: where Spotlight sits.
    func placeTopCenter(fromTop: CGFloat = 180) {
        guard let screen = NSScreen.main else { return }
        let visible = screen.visibleFrame
        setFrameOrigin(
            NSPoint(x: visible.midX - frame.width / 2, y: visible.maxY - frame.height - fromTop))
    }

    override func cancelOperation(_ sender: Any?) {
        onCancel?()
    }
}
