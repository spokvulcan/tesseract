//
//  WhipRenderView.swift
//  tesseract
//
//  Draws the whip over whatever happens to be on screen.
//
//  The visibility problem is real: this view composites over black terminals,
//  white documents and photographs alike, and any single flat colour disappears
//  against something. So every part is drawn three times — a soft shadow beneath,
//  a dark body, and a thin light rim — which is the same trick the system cursor
//  uses. Whatever the backdrop, at least one of the three contrasts with it.
//
//  Drawing is plain CoreGraphics in a plain `NSView` rather than SwiftUI: at
//  120 Hz over a canvas the size of every display combined, a diffed view tree
//  is overhead for no benefit, and `setNeedsDisplay(_:)` on a tight dirty rect
//  is exactly the control this needs.
//

import AppKit
import Foundation

/// Everything the view needs for one frame. A value, handed over whole, so the
/// view never reaches back into the simulation.
nonisolated struct WhipFrame: Sendable, Equatable {
    var thongPoints: [CGPoint] = []
    var thongRadii: [CGFloat] = []
    var grip: CGPoint = .zero
    var handleTip: CGPoint = .zero
    var handleRadius: CGFloat = 7
    /// The grab affordance: ⌥ is down and the cursor is close enough that a
    /// click would take hold. Drawn as a ring rather than a cursor change, which
    /// a click-through panel cannot reliably own.
    var isArmed = false
    var isGrabbed = false

    static let empty = WhipFrame()
}

final class WhipRenderView: NSView {

    /// Called on mouse-down when the panel is interactive, which only happens
    /// while the grab affordance is showing.
    var onGrab: (() -> Void)?
    var onRelease: (() -> Void)?

    private var frameData: WhipFrame = .empty
    private var lastDirtyRect: CGRect = .zero

    override var isFlipped: Bool { false }
    override var isOpaque: Bool { false }

    /// A non-activating panel still has to accept the very first click, or
    /// grabbing would always take two — one to notice the window, one to act.
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }

    /// Swaps in a new frame and invalidates only what moved. The canvas spans
    /// every display, so redrawing all of it per frame would hand the window
    /// server a screen-sized blend 120 times a second for a few hundred points
    /// of moving line.
    func apply(_ newFrame: WhipFrame) {
        guard newFrame != frameData else { return }
        let previousRect = lastDirtyRect
        frameData = newFrame

        let padding: CGFloat = 26  // room for the rim, the shadow and the ring
        var rect = WhipGeometry.bounds(
            points: newFrame.thongPoints, radii: newFrame.thongRadii, padding: padding)
        let handleRect = CGRect(
            x: min(newFrame.grip.x, newFrame.handleTip.x),
            y: min(newFrame.grip.y, newFrame.handleTip.y),
            width: abs(newFrame.handleTip.x - newFrame.grip.x),
            height: abs(newFrame.handleTip.y - newFrame.grip.y)
        ).insetBy(dx: -(newFrame.handleRadius + padding), dy: -(newFrame.handleRadius + padding))
        rect = rect.isNull ? handleRect : rect.union(handleRect)

        lastDirtyRect = rect
        setNeedsDisplay(previousRect.isNull ? rect : rect.union(previousRect))
    }

    override func draw(_ dirtyRect: NSRect) {
        guard let context = NSGraphicsContext.current?.cgContext else { return }
        guard frameData.thongPoints.count >= 2 else { return }

        let thong = WhipGeometry.outline(
            points: frameData.thongPoints, radii: frameData.thongRadii)
        let handle = WhipGeometry.handlePath(
            from: frameData.grip, to: frameData.handleTip, radius: frameData.handleRadius)

        let body = CGMutablePath()
        body.addPath(thong)
        body.addPath(handle)

        // 1 — the shadow. Cast down and to the right as though lit from the
        // upper left, which is where macOS puts its light. This is what makes
        // the whip read as lying *above* the screen rather than painted onto it.
        context.saveGState()
        context.setShadow(
            offset: CGSize(width: 1.5, height: -3.5),
            blur: 7,
            color: NSColor.black.withAlphaComponent(0.42).cgColor)
        context.addPath(body)
        context.setFillColor(NSColor.black.withAlphaComponent(0.9).cgColor)
        context.fillPath()
        context.restoreGState()

        // 2 — the body. Not pure black: a near-black with a hint of warmth reads
        // as leather, and leaves room for the rim to be seen against it.
        context.saveGState()
        context.addPath(thong)
        context.setFillColor(
            NSColor(calibratedRed: 0.09, green: 0.08, blue: 0.09, alpha: 0.97).cgColor)
        context.fillPath()

        context.addPath(handle)
        context.setFillColor(
            NSColor(calibratedRed: 0.16, green: 0.13, blue: 0.12, alpha: 0.99).cgColor)
        context.fillPath()
        context.restoreGState()

        // 3 — the rim. Thin and bright, and the only thing that survives when
        // the whip lies over a black terminal.
        context.saveGState()
        context.addPath(thong)
        context.setStrokeColor(NSColor.white.withAlphaComponent(0.34).cgColor)
        context.setLineWidth(0.75)
        context.strokePath()

        context.addPath(handle)
        context.setStrokeColor(NSColor.white.withAlphaComponent(0.30).cgColor)
        context.setLineWidth(1)
        context.strokePath()
        context.restoreGState()

        // A single highlight running along the handle, so the grip reads as a
        // round object rather than a flat lozenge.
        drawHandleHighlight(in: context)

        if frameData.isArmed || frameData.isGrabbed {
            drawGrabRing(in: context)
        }
    }

    private func drawHandleHighlight(in context: CGContext) {
        let grip = frameData.grip
        let tip = frameData.handleTip
        let direction = CGVector(dx: tip.x - grip.x, dy: tip.y - grip.y)
        let length = hypot(direction.dx, direction.dy)
        guard length > 0.5 else { return }

        let unit = CGVector(dx: direction.dx / length, dy: direction.dy / length)
        let normal = CGVector(dx: -unit.dy, dy: unit.dx)
        let offset = frameData.handleRadius * 0.42

        context.saveGState()
        context.setStrokeColor(NSColor.white.withAlphaComponent(0.20).cgColor)
        context.setLineWidth(1.6)
        context.setLineCap(.round)
        context.move(
            to: CGPoint(
                x: grip.x + normal.dx * offset + unit.dx * 5,
                y: grip.y + normal.dy * offset + unit.dy * 5))
        context.addLine(
            to: CGPoint(
                x: tip.x + normal.dx * offset - unit.dx * 5,
                y: tip.y + normal.dy * offset - unit.dy * 5))
        context.strokePath()
        context.restoreGState()
    }

    /// The affordance ring: shown while ⌥ arms the grab, filled once you have
    /// hold of it. Drawn by us rather than set as an `NSCursor`, because a
    /// click-through non-activating panel has no reliable claim on the cursor.
    private func drawGrabRing(in context: CGContext) {
        let radius: CGFloat = frameData.isGrabbed ? 15 : 19
        let rect = CGRect(
            x: frameData.grip.x - radius, y: frameData.grip.y - radius,
            width: radius * 2, height: radius * 2)

        context.saveGState()
        context.setLineWidth(frameData.isGrabbed ? 2.4 : 1.6)
        context.setStrokeColor(
            NSColor.white.withAlphaComponent(frameData.isGrabbed ? 0.85 : 0.55).cgColor)
        context.strokeEllipse(in: rect)

        context.setStrokeColor(NSColor.black.withAlphaComponent(0.35).cgColor)
        context.setLineWidth(1)
        context.strokeEllipse(in: rect.insetBy(dx: -1.4, dy: -1.4))
        context.restoreGState()
    }

    // MARK: - Input

    override func mouseDown(with event: NSEvent) {
        onGrab?()
    }

    override func mouseUp(with event: NSEvent) {
        onRelease?()
    }
}
