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
//  field in the first layout, never later. The Lens (PRD #612) shares it too,
//  anchored at the bottom of the screen and seeing its key presses first.
//

import AppKit
import SwiftUI

@MainActor
final class GlassPanel: NSPanel {

    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { false }

    /// Called when the panel should go away (Escape).
    var onCancel: (() -> Void)?

    /// Sees each key press before the panel's views do; returning true
    /// swallows it. The Lens takes ⇥, ↩, Esc and the arrows here, so its
    /// text field only ever receives the word being typed.
    var interceptKeyDown: ((NSEvent) -> Bool)?

    /// Told of every key press (not its repeats) and click the panel
    /// receives, before anything handles it.
    var onInput: (() -> Void)?

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

    /// Resize, keeping the bottom edge where it is.
    func setHeightKeepingBottom(_ height: CGFloat) {
        var frame = self.frame
        frame.size.height = height
        setFrame(frame, display: true, animate: false)
    }

    /// Centred horizontally near the bottom of `screen`'s visible frame,
    /// where the dictation overlay sits.
    func placeBottomCenter(on screen: NSScreen? = NSScreen.main, fromBottom: CGFloat) {
        guard let screen else { return }
        let visible = screen.visibleFrame
        setFrameOrigin(
            NSPoint(x: visible.midX - frame.width / 2, y: visible.minY + fromBottom))
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

    override func sendEvent(_ event: NSEvent) {
        switch event.type {
        case .keyDown where !event.isARepeat, .leftMouseDown, .rightMouseDown, .otherMouseDown:
            onInput?()
        default:
            break
        }
        if event.type == .keyDown, let interceptKeyDown, interceptKeyDown(event) {
            return
        }
        super.sendEvent(event)
    }
}
