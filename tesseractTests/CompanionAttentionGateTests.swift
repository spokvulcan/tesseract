//
//  CompanionAttentionGateTests.swift
//  tesseractTests
//
//  The attention gate: background turns hold while the owner is engaged and
//  for the quiet window after; a machine-busy GPU yields without arming the
//  window; an explicit lift (test wake, notification reply) opens the gate
//  regardless. Signals and clocks are injected — no HID, no AppKit.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
@Suite struct CompanionAttentionGateTests {

    @Test func opensWhenNothingIsHappening() {
        let gate = CompanionAttentionGate(
            isOwnerEngaged: { false }, isMachineBusy: { false })
        #expect(gate.mayRunTurn(now: Date()))
    }

    @Test func closesWhileOwnerIsEngagedAndForTheQuietWindowAfter() {
        var engaged = true
        let gate = CompanionAttentionGate(
            isOwnerEngaged: { engaged }, isMachineBusy: { false })
        let t0 = Date()
        #expect(!gate.mayRunTurn(now: t0))

        engaged = false
        // One second after he stops: still inside the quiet window.
        #expect(!gate.mayRunTurn(now: t0.addingTimeInterval(1)))
        // Just short of the window: still closed.
        #expect(
            !gate.mayRunTurn(
                now: t0.addingTimeInterval(CompanionAttentionGate.quietWindow - 1)))
        // Window elapsed: open.
        #expect(
            gate.mayRunTurn(
                now: t0.addingTimeInterval(CompanionAttentionGate.quietWindow + 1)))
    }

    @Test func reEngagementReArmsTheWindow() {
        var engaged = true
        let gate = CompanionAttentionGate(
            isOwnerEngaged: { engaged }, isMachineBusy: { false })
        let t0 = Date()
        _ = gate.mayRunTurn(now: t0)
        engaged = false
        _ = gate.mayRunTurn(now: t0.addingTimeInterval(60))

        // He comes back at t0+90 — the window restarts from there.
        engaged = true
        _ = gate.mayRunTurn(now: t0.addingTimeInterval(90))
        engaged = false
        #expect(
            !gate.mayRunTurn(
                now: t0.addingTimeInterval(90 + CompanionAttentionGate.quietWindow - 1)))
        #expect(
            gate.mayRunTurn(
                now: t0.addingTimeInterval(90 + CompanionAttentionGate.quietWindow + 1)))
    }

    @Test func machineBusyYieldsWithoutArmingTheWindow() {
        var busy = true
        let gate = CompanionAttentionGate(
            isOwnerEngaged: { false }, isMachineBusy: { busy })
        let t0 = Date()
        #expect(!gate.mayRunTurn(now: t0))

        // The GPU frees: the very next tick may run — no quiet-window penalty.
        busy = false
        #expect(gate.mayRunTurn(now: t0.addingTimeInterval(1)))
    }

    @Test func liftOpensTheGateThroughEngagementThenExpires() {
        let gate = CompanionAttentionGate(
            isOwnerEngaged: { true }, isMachineBusy: { false })
        let t0 = Date()
        #expect(!gate.mayRunTurn(now: t0))

        gate.lift(now: t0)
        #expect(gate.mayRunTurn(now: t0.addingTimeInterval(1)))

        // Lift expired and he is still engaged: closed again.
        #expect(
            !gate.mayRunTurn(
                now: t0.addingTimeInterval(CompanionAttentionGate.liftWindow + 1)))
    }

    @Test func engagementIsStampedAsEvidence() {
        var engaged = true
        let gate = CompanionAttentionGate(
            isOwnerEngaged: { engaged }, isMachineBusy: { false })
        #expect(gate.lastOwnerEngagedAt == nil)

        let t0 = Date()
        _ = gate.mayRunTurn(now: t0)
        #expect(gate.lastOwnerEngagedAt == t0)

        // Disengaged polls don't move the stamp — it is "last seen", not "now".
        engaged = false
        _ = gate.mayRunTurn(now: t0.addingTimeInterval(30))
        #expect(gate.lastOwnerEngagedAt == t0)
    }
}
