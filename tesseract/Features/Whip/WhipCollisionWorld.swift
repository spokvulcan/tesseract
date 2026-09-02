//
//  WhipCollisionWorld.swift
//  tesseract
//
//  Where the whip's obstacles come from: the real windows on your screen.
//
//  `CGWindowListCopyWindowInfo` is already used ungated in this app to place
//  the dictation overlay (`OverlayScreenLocator`), so reading live window
//  geometry needs no permission and no new entitlement. What it *is* is
//  expensive — `OverlayPanel` warns that resolving it is "window-server IPC (a
//  full window-list copy), so it is never done per state change". This type
//  therefore caches, refreshes at a few hertz while the whip is awake, and does
//  nothing at all while it sleeps.
//

import AppKit
import CoreGraphics
import Foundation

/// Snapshots the on-screen window rects into simulation space and hands them to
/// the physics as plain rects, so the solver never learns what a window is.
@MainActor
final class WhipCollisionWorld {

    /// Minimum gap between window-list refreshes. Four hertz is far below the
    /// rate at which windows actually move and far above the rate at which you
    /// would notice a stale edge.
    private static let refreshInterval: TimeInterval = 0.25

    /// Windows smaller than this in either axis are ignored — tooltips, shadows
    /// and helper windows should not be surfaces the whip lands on.
    private static let minimumWindowSide: CGFloat = 48

    private var cached: [CGRect] = []
    private var lastRefresh: TimeInterval = 0

    /// Windows the whip must not collide with — in practice just the panel it is
    /// drawn on. Excluding by window number rather than by process means the
    /// whip still drapes over Tesseract's *own* chat window, which it should:
    /// nothing about the toy should treat this app as special.
    var excludedWindowNumbers: Set<Int> = []

    /// The panel's frame in Cocoa global coordinates — the offset that converts
    /// screen space into the simulation's space.
    var canvasOrigin: CGPoint = .zero

    /// Union of the displays, in simulation space. The walls and the floor.
    var canvasBounds: CGRect = .zero

    /// Returns the current world, refreshing the window list at most every
    /// `refreshInterval`. `force` bypasses the throttle for the wake path, where
    /// a stale list would let the whip fall through a window that moved while it
    /// was asleep.
    func world(now: TimeInterval, force: Bool = false) -> WhipWorld {
        if force || now - lastRefresh >= Self.refreshInterval {
            lastRefresh = now
            cached = fetchWindowRects()
        }
        return WhipWorld(obstacles: cached, bounds: canvasBounds)
    }

    /// Drops the cache so the next wake starts from a fresh look at the screen.
    func invalidate() {
        lastRefresh = 0
        cached = []
    }

    // MARK: - Window list

    private func fetchWindowRects() -> [CGRect] {
        let options: CGWindowListOption = [.optionOnScreenOnly, .excludeDesktopElements]
        guard
            let infoList = CGWindowListCopyWindowInfo(options, kCGNullWindowID) as? [[String: Any]]
        else { return [] }

        // The Cocoa/CoreGraphics origin mismatch: CG measures from the top-left
        // of the *primary* display downward, Cocoa from its bottom-left upward.
        // Everything below converts through this one height.
        let primaryHeight = Self.primaryDisplayHeight()

        var rects: [CGRect] = []
        rects.reserveCapacity(infoList.count)

        for info in infoList {
            // Our own panel must never be an obstacle — the whip would collide
            // with the surface it is drawn on.
            if let number = info[kCGWindowNumber as String] as? Int,
                excludedWindowNumbers.contains(number)
            {
                continue
            }

            // Layer 0 is the ordinary application-window layer. Everything above
            // it is system furniture (the Dock at 20, the menu bar at 24) which
            // the whip passes in front of rather than landing on.
            guard let layer = info[kCGWindowLayer as String] as? Int, layer == 0 else { continue }

            if let alpha = info[kCGWindowAlpha as String] as? Double, alpha <= 0.01 { continue }

            guard let boundsDict = info[kCGWindowBounds as String] as? [String: Any],
                let cgRect = CGRect(dictionaryRepresentation: boundsDict as CFDictionary),
                cgRect.width >= Self.minimumWindowSide,
                cgRect.height >= Self.minimumWindowSide
            else { continue }

            let cocoa = CGRect(
                x: cgRect.origin.x,
                y: primaryHeight - cgRect.origin.y - cgRect.height,
                width: cgRect.width,
                height: cgRect.height)

            let local = cocoa.offsetBy(dx: -canvasOrigin.x, dy: -canvasOrigin.y)
            guard canvasBounds.isEmpty || local.intersects(canvasBounds) else { continue }
            rects.append(local)
        }

        return rects
    }

    /// Height of the display CoreGraphics measures from. That is the screen
    /// whose Cocoa frame sits at the origin — not necessarily `NSScreen.main`,
    /// which follows the key window and moves between displays.
    static func primaryDisplayHeight() -> CGFloat {
        if let primary = NSScreen.screens.first(where: { $0.frame.origin == .zero }) {
            return primary.frame.height
        }
        return NSScreen.screens.first?.frame.height ?? 0
    }

    /// The union of every display, in Cocoa global coordinates — the panel's
    /// frame, and after offsetting, the simulation's walls.
    static func displayUnion() -> CGRect {
        NSScreen.screens.reduce(CGRect.null) { $0.union($1.frame) }
    }
}
