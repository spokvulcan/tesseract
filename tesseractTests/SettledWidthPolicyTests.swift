//
//  SettledWidthPolicyTests.swift
//  tesseractTests
//
//  Pins the **Settled Width** policy (`Core/SettledWidth.swift`): when the
//  page's layout width is pinned during a sidebar slide, and what releases
//  it. Geometry samples mirror the ones the design prototype recorded from
//  AppKit's real `toggleSidebar:` animation.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct SettledWidthPolicyTests {
    typealias Span = SettledWidthPolicy.ColumnSpan

    @Test
    func firstSampleNeverPins() {
        var policy = SettledWidthPolicy()
        #expect(policy.apply(.columnSpanChanged(Span(minX: 0, width: 900))) == .none)
        #expect(policy.pinnedWidth == nil)
    }

    @Test
    func windowResizeStaysLive() {
        // A window resize moves the width alone; the origin stays put.
        var policy = SettledWidthPolicy()
        _ = policy.apply(.columnSpanChanged(Span(minX: 0, width: 900)))
        #expect(policy.apply(.columnSpanChanged(Span(minX: 0, width: 906))) == .none)
        #expect(policy.apply(.columnSpanChanged(Span(minX: 0, width: 915))) == .none)
        #expect(policy.pinnedWidth == nil)
    }

    @Test
    func originOnlyMoveStaysLive() {
        var policy = SettledWidthPolicy()
        _ = policy.apply(.columnSpanChanged(Span(minX: 0, width: 900)))
        #expect(policy.apply(.columnSpanChanged(Span(minX: 4, width: 900))) == .none)
        #expect(policy.pinnedWidth == nil)
    }

    @Test
    func sidebarSlidePinsThePreviousWidthOnItsFirstFrame() {
        // Opening: the column's origin and width move together from frame 1.
        var policy = SettledWidthPolicy()
        _ = policy.apply(.columnSpanChanged(Span(minX: 0, width: 900)))
        #expect(policy.apply(.columnSpanChanged(Span(minX: 1, width: 899))) == .armSettleTimer)
        #expect(policy.pinnedWidth == 900)
    }

    @Test
    func framesWhilePinnedKeepThePinAndRearmTheTimer() {
        var policy = SettledWidthPolicy()
        _ = policy.apply(.columnSpanChanged(Span(minX: 0, width: 900)))
        _ = policy.apply(.columnSpanChanged(Span(minX: 1, width: 899)))
        #expect(policy.apply(.columnSpanChanged(Span(minX: 2, width: 898))) == .armSettleTimer)
        #expect(policy.apply(.columnSpanChanged(Span(minX: 4, width: 896))) == .armSettleTimer)
        // Width-only jitter mid-slide also counts as motion while pinned.
        #expect(policy.apply(.columnSpanChanged(Span(minX: 4, width: 897))) == .armSettleTimer)
        // A repeated identical sample is not motion.
        #expect(policy.apply(.columnSpanChanged(Span(minX: 4, width: 897))) == .none)
        #expect(policy.pinnedWidth == 900)
    }

    @Test
    func visibilityFlipReleasesAndDisarmsTheTimer() {
        var policy = SettledWidthPolicy()
        _ = policy.apply(.columnSpanChanged(Span(minX: 0, width: 900)))
        _ = policy.apply(.columnSpanChanged(Span(minX: 1, width: 899)))
        _ = policy.apply(.columnSpanChanged(Span(minX: 147, width: 753)))
        #expect(policy.apply(.columnVisibilityChanged) == .disarmSettleTimer)
        #expect(policy.pinnedWidth == nil)
    }

    @Test
    func settleTimerReleasesWithoutAVisibilityFlip() {
        // The split divider drag: motion, then stillness, never a flip.
        var policy = SettledWidthPolicy()
        _ = policy.apply(.columnSpanChanged(Span(minX: 147, width: 753)))
        _ = policy.apply(.columnSpanChanged(Span(minX: 160, width: 740)))
        #expect(policy.pinnedWidth == 753)
        #expect(policy.apply(.settleTimerFired) == .none)
        #expect(policy.pinnedWidth == nil)
    }

    @Test
    func visibilityFlipWhileLiveIsANoop() {
        var policy = SettledWidthPolicy()
        _ = policy.apply(.columnSpanChanged(Span(minX: 0, width: 900)))
        #expect(policy.apply(.columnVisibilityChanged) == .none)
        #expect(policy.apply(.settleTimerFired) == .none)
        #expect(policy.pinnedWidth == nil)
    }

    @Test
    func aSecondSlideAfterReleasePinsAgainAtTheSettledWidth() {
        // Open, settle, then close: the close pins the *open* width.
        var policy = SettledWidthPolicy()
        _ = policy.apply(.columnSpanChanged(Span(minX: 0, width: 900)))
        _ = policy.apply(.columnSpanChanged(Span(minX: 1, width: 899)))
        _ = policy.apply(.columnSpanChanged(Span(minX: 148, width: 752)))
        _ = policy.apply(.columnVisibilityChanged)
        #expect(policy.pinnedWidth == nil)
        #expect(policy.apply(.columnSpanChanged(Span(minX: 147, width: 753))) == .armSettleTimer)
        #expect(policy.pinnedWidth == 752)
    }

    @Test
    func aZeroWidthProbeNeverBecomesThePin() {
        // The scene probes the detail minimum at near-zero width; a pin at 0
        // would collapse the page.
        var policy = SettledWidthPolicy()
        _ = policy.apply(.columnSpanChanged(Span(minX: 0, width: 0)))
        #expect(policy.apply(.columnSpanChanged(Span(minX: 1, width: 899))) == .none)
        #expect(policy.pinnedWidth == nil)
    }
}
