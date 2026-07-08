//
//  SkillClusterControllerTests.swift
//  tesseractTests
//
//  Tests the **Skill Cluster** interaction state machine at its own seam
//  (ADR-0030) — pointer and click events in, `phase` out. Durations are
//  injected as zero so transitions settle deterministically via `settle()`.
//  No SwiftUI, no Agent.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct SkillClusterControllerTests {

    private func makeController() -> SkillClusterController {
        SkillClusterController(openDelay: .zero, exitGrace: .zero)
    }

    // MARK: - Hover open / close

    @Test func hoverOpensAfterDelay() async {
        let controller = makeController()
        #expect(controller.phase == .collapsed)

        controller.pointerEntered()
        // Not yet — the open rides the delay, never synchronous.
        #expect(controller.phase == .collapsed)

        await controller.settle()
        #expect(controller.phase == .expanded)
    }

    @Test func exitBeforeDelayCancelsTheOpen() async {
        let controller = makeController()
        controller.pointerEntered()
        controller.pointerExited()

        await controller.settle()
        #expect(controller.phase == .collapsed)
    }

    @Test func exitCollapsesAfterGrace() async {
        let controller = makeController()
        controller.pointerEntered()
        await controller.settle()

        controller.pointerExited()
        // Still open during the grace window.
        #expect(controller.phase == .expanded)

        await controller.settle()
        #expect(controller.phase == .collapsed)
    }

    @Test func reenterDuringGraceKeepsTheClusterOpen() async {
        let controller = makeController()
        controller.pointerEntered()
        await controller.settle()

        controller.pointerExited()
        controller.pointerEntered()

        await controller.settle()
        #expect(controller.phase == .expanded)
    }

    // MARK: - Pinning

    @Test func clickPinsImmediately() {
        let controller = makeController()
        controller.buttonClicked()
        // Pinning is synchronous — no hover delay on a deliberate click.
        #expect(controller.phase == .pinned)
    }

    @Test func clickWhileHoverOpenUpgradesToPinned() async {
        let controller = makeController()
        controller.pointerEntered()
        await controller.settle()

        controller.buttonClicked()
        #expect(controller.phase == .pinned)
    }

    @Test func pointerExitDoesNotCloseAPinnedCluster() async {
        let controller = makeController()
        controller.buttonClicked()

        controller.pointerExited()
        await controller.settle()
        #expect(controller.phase == .pinned)
    }

    @Test func clickWhilePinnedCollapses() {
        let controller = makeController()
        controller.buttonClicked()
        controller.buttonClicked()
        #expect(controller.phase == .collapsed)
    }

    @Test func clickAwayCollapsesAPinnedCluster() {
        let controller = makeController()
        controller.buttonClicked()
        controller.clickedAway()
        #expect(controller.phase == .collapsed)
    }

    // MARK: - Escape

    @Test func escapeCollapsesAndReportsHandled() async {
        let controller = makeController()
        controller.buttonClicked()
        #expect(controller.escapePressed() == true)
        #expect(controller.phase == .collapsed)
    }

    @Test func escapeWhileCollapsedReportsUnhandled() {
        let controller = makeController()
        #expect(controller.escapePressed() == false)
    }

    // MARK: - Firing

    @Test func firingAPillCollapses() async {
        let controller = makeController()
        controller.pointerEntered()
        await controller.settle()

        controller.pillFired()
        #expect(controller.phase == .collapsed)
    }

    // MARK: - Suppression (generating / slash popup open)

    @Test func hoverIsInertWhileSuppressed() async {
        let controller = makeController()
        controller.isSuppressed = true

        controller.pointerEntered()
        await controller.settle()
        #expect(controller.phase == .collapsed)
    }

    @Test func clickIsInertWhileSuppressed() {
        let controller = makeController()
        controller.isSuppressed = true

        controller.buttonClicked()
        #expect(controller.phase == .collapsed)
    }

    @Test func becomingSuppressedCollapsesAnOpenCluster() {
        let controller = makeController()
        controller.buttonClicked()
        #expect(controller.phase == .pinned)

        controller.isSuppressed = true
        #expect(controller.phase == .collapsed)
    }

    @Test func becomingSuppressedCancelsAPendingOpen() async {
        let controller = makeController()
        controller.pointerEntered()
        controller.isSuppressed = true

        await controller.settle()
        #expect(controller.phase == .collapsed)
    }

    @Test func unsuppressingRestoresHover() async {
        let controller = makeController()
        controller.isSuppressed = true
        controller.isSuppressed = false

        controller.pointerEntered()
        await controller.settle()
        #expect(controller.phase == .expanded)
    }
}
