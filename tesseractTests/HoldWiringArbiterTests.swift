//
//  HoldWiringArbiterTests.swift
//  tesseractTests
//
//  The **Hold Wiring Arbiter** at its own seam (ADR-0050): decision tables
//  over the voice hold's async arbitration — schedule folding, generation
//  staleness, and the landing verdicts (commit / discard-and-start-next /
//  discard-and-idle). Before the cut this was a 12-field inline state
//  machine on `AudioCaptureEngine` with zero tests, even though it is
//  precisely the discipline that prevents the 2026-07-17 tap-rewire crash
//  class; the engine drivers now perform verdicts decided here.
//

import Testing

@testable import Tesseract_Agent

struct HoldWiringArbiterTests {

    /// Begin activates the hold and stales anything older; a second begin
    /// is a no-op and leaves the machine untouched.
    @Test func beginActivatesOnceAndBumpsTheGeneration() {
        var arbiter = HoldWiringArbiter()
        let began = arbiter.beginHold()
        #expect(began)
        #expect(arbiter.isHoldActive)
        let generation = arbiter.currentGeneration

        let before = arbiter
        let beganAgain = arbiter.beginHold()
        #expect(!beganAgain)
        #expect(arbiter == before)
        #expect(arbiter.currentGeneration == generation)
    }

    /// An idle machine starts a wiring immediately, stamps it with the
    /// bumped generation, and takes ownership of the engine.
    @Test func scheduleStartsWhenIdle() {
        var arbiter = HoldWiringArbiter()
        _ = arbiter.beginHold()
        let verdict = arbiter.schedule(rebuildFirst: false)
        #expect(
            verdict == .start(rebuildFirst: false, generation: arbiter.currentGeneration))
        #expect(arbiter.isWiringInFlight)
    }

    /// Requests arriving while a wiring is in flight fold into one queued
    /// request — OR on rebuild-first, so a rebuild demand is never
    /// downgraded by a later plain re-wire.
    @Test func scheduleFoldsIntoTheQueueWhileInFlight() {
        var arbiter = HoldWiringArbiter()
        _ = arbiter.beginHold()
        _ = arbiter.schedule(rebuildFirst: false)

        let rebuild = arbiter.schedule(rebuildFirst: true)
        #expect(rebuild == .folded)
        #expect(arbiter.queuedRebuildFirst == true)
        let plain = arbiter.schedule(rebuildFirst: false)
        #expect(plain == .folded)
        #expect(arbiter.queuedRebuildFirst == true)
    }

    /// The unraced path: the wiring lands carrying the current generation
    /// and commits, releasing the engine.
    @Test func currentGenerationLandingCommits() {
        var arbiter = HoldWiringArbiter()
        _ = arbiter.beginHold()
        guard case .start(_, let generation) = arbiter.schedule(rebuildFirst: true) else {
            Issue.record("expected a start verdict")
            return
        }
        let landing = arbiter.wiringLanded(generation: generation)
        #expect(landing == .commit)
        #expect(!arbiter.isWiringInFlight)
    }

    /// A re-schedule stales the in-flight wiring: its landing discards and
    /// starts the queued request in the same breath — the engine never has
    /// an ownerless moment, and the queued wiring's own landing commits.
    @Test func rescheduleStalesTheInFlightWiringAndHandsOff() {
        var arbiter = HoldWiringArbiter()
        _ = arbiter.beginHold()
        guard case .start(_, let first) = arbiter.schedule(rebuildFirst: false) else {
            Issue.record("expected a start verdict")
            return
        }
        _ = arbiter.schedule(rebuildFirst: true)

        let landing = arbiter.wiringLanded(generation: first)
        #expect(
            landing
                == .discardAndStartNext(
                    rebuildFirst: true, generation: arbiter.currentGeneration))
        #expect(arbiter.isWiringInFlight)
        #expect(arbiter.queuedRebuildFirst == nil)

        guard case .discardAndStartNext(_, let second) = landing else { return }
        let handedOff = arbiter.wiringLanded(generation: second)
        #expect(handedOff == .commit)
        #expect(!arbiter.isWiringInFlight)
    }

    /// Ending the hold mid-wiring leaves the discard to the landing —
    /// touching the engine at end would race the detached work — and kills
    /// whatever request had queued behind it.
    @Test func endMidWiringLeavesDiscardToTheLanding() {
        var arbiter = HoldWiringArbiter()
        _ = arbiter.beginHold()
        guard case .start(_, let generation) = arbiter.schedule(rebuildFirst: false) else {
            Issue.record("expected a start verdict")
            return
        }
        _ = arbiter.schedule(rebuildFirst: true)  // queued, then killed by end

        let end = arbiter.endHold()
        #expect(end == .leaveDiscardToCommit)
        #expect(arbiter.queuedRebuildFirst == nil)
        let landing = arbiter.wiringLanded(generation: generation)
        #expect(landing == .discardAndIdle)
        #expect(!arbiter.isWiringInFlight)
    }

    /// Ending an unwired hold unwires synchronously; ending an inactive
    /// machine is a no-op.
    @Test func endWithoutInFlightWiringUnwiresNow() {
        var arbiter = HoldWiringArbiter()
        let idleEnd = arbiter.endHold()
        #expect(idleEnd == .alreadyIdle)
        _ = arbiter.beginHold()
        let activeEnd = arbiter.endHold()
        #expect(activeEnd == .unwireNow)
        #expect(!arbiter.isHoldActive)
    }

    /// The rapid end→begin cycle around an in-flight wiring — the crash
    /// class scenario. Without a fresh schedule the stale landing idles
    /// out (never adopts work built for a dead hold), and only a new
    /// schedule wires the reborn hold.
    @Test func endThenBeginStalesTheOldWiringForTheRebornHold() {
        var arbiter = HoldWiringArbiter()
        _ = arbiter.beginHold()
        guard case .start(_, let stale) = arbiter.schedule(rebuildFirst: false) else {
            Issue.record("expected a start verdict")
            return
        }
        _ = arbiter.endHold()
        _ = arbiter.beginHold()

        let landing = arbiter.wiringLanded(generation: stale)
        #expect(landing == .discardAndIdle)
        #expect(!arbiter.isWiringInFlight)
        guard case .start(_, let fresh) = arbiter.schedule(rebuildFirst: true) else {
            Issue.record("expected a start verdict")
            return
        }
        let freshLanding = arbiter.wiringLanded(generation: fresh)
        #expect(freshLanding == .commit)
    }

    /// The reborn hold that re-schedules while the dead hold's wiring is
    /// still in flight: the request folds, and the stale landing hands
    /// straight off to it.
    @Test func rebornHoldQueuesBehindTheDyingWiring() {
        var arbiter = HoldWiringArbiter()
        _ = arbiter.beginHold()
        guard case .start(_, let stale) = arbiter.schedule(rebuildFirst: false) else {
            Issue.record("expected a start verdict")
            return
        }
        _ = arbiter.endHold()
        _ = arbiter.beginHold()
        let folded = arbiter.schedule(rebuildFirst: true)
        #expect(folded == .folded)

        let landing = arbiter.wiringLanded(generation: stale)
        #expect(
            landing
                == .discardAndStartNext(
                    rebuildFirst: true, generation: arbiter.currentGeneration))
        #expect(arbiter.isWiringInFlight)
    }
}
