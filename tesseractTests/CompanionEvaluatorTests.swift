//
//  CompanionEvaluatorTests.swift
//  tesseractTests
//
//  The Wake Evaluator's decision tables (ADR-0040 §2, ADR-0043): one gathered
//  `Signals` snapshot in, one `Decision` out — pure, so every gate, the origin
//  pick, the batching rule, and the deferral dedup are pinned here with no
//  store, no clock, and no loop. The braid these rules used to live in had
//  zero tests; each table below is one rule the field depends on.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@Suite struct CompanionEvaluatorTests {

    /// A fixed instant — decision tables never read the real clock.
    private static let now = Date(timeIntervalSinceReferenceDate: 800_000_000)

    private func wake(
        _ content: String = "check on him",
        dueAgo: TimeInterval = 60,
        wakeClass: CompanionWakeClass = .promise
    ) -> CompanionWake {
        CompanionWake(
            content: content, due: Self.now.addingTimeInterval(-dueAgo),
            wakeClass: wakeClass)
    }

    private func signals(
        hour: Int = 10,
        due: [CompanionWake] = [],
        day: CompanionLoopDayState = CompanionLoopDayState(),
        gateOpen: Bool = true,
        present: Bool = true,
        ac: Bool = true,
        gpuBusy: Bool = false
    ) -> CompanionEvaluator.Signals {
        CompanionEvaluator.Signals(
            now: Self.now, localHour: hour, dueWakes: due, dayState: day,
            gateOpen: gateOpen, ownerPresent: present, onACPower: ac,
            gpuBusy: gpuBusy)
    }

    /// A day state that keeps the transition and ambient gates quiet, so a
    /// table about something else reads `.wait` as its baseline.
    private func settledDay() -> CompanionLoopDayState {
        CompanionLoopDayState(
            dayStartedAt: Self.now.addingTimeInterval(-4 * 3600),
            lastAmbientAt: Self.now.addingTimeInterval(-60))
    }

    // MARK: - Due wakes and the origin pick

    @Test func quietTickDecidesWait() {
        var evaluator = CompanionEvaluator()
        #expect(evaluator.decide(signals(day: settledDay())) == .wait)
    }

    @Test func dueWakesGrantOneTurnCarryingTheWholeBatch() {
        var evaluator = CompanionEvaluator()
        let first = wake("morning check", dueAgo: 120)
        let second = wake("the dentist", dueAgo: 60, wakeClass: .followup)
        let decision = evaluator.decide(signals(due: [first, second]))
        #expect(
            decision
                == .wakeTurn(batch: [first, second], origin: .wake, carriesBeat: false))
    }

    @Test func aBatchCarryingARhythmWakeIsABeat() {
        var evaluator = CompanionEvaluator()
        let beat = wake("evening journal", dueAgo: 60, wakeClass: .rhythm)
        let promise = wake(dueAgo: 30)
        let decision = evaluator.decide(signals(due: [beat, promise]))
        #expect(
            decision == .wakeTurn(batch: [beat, promise], origin: .beat, carriesBeat: true))
    }

    @Test func overduePastTheGracePreemptsTheMerelyDue() {
        var evaluator = CompanionEvaluator()
        let stale = wake("missed while asleep", dueAgo: CompanionEvaluator.catchUpGrace + 60)
        let fresh = wake("on time", dueAgo: 60)
        let decision = evaluator.decide(signals(due: [stale, fresh]))
        // The overdue wake goes to triage alone; the on-time wake stays booked
        // and re-presents next tick.
        #expect(decision == .wakeTurn(batch: [stale], origin: .catchup, carriesBeat: false))
    }

    @Test func overdueByExactlyTheGraceIsNotOverdue() {
        var evaluator = CompanionEvaluator()
        let edge = wake("on the line", dueAgo: CompanionEvaluator.catchUpGrace)
        let decision = evaluator.decide(signals(due: [edge]))
        #expect(decision == .wakeTurn(batch: [edge], origin: .wake, carriesBeat: false))
    }

    @Test func anOverdueRhythmWakeIsCatchupButStillCarriesTheBeat() {
        var evaluator = CompanionEvaluator()
        let staleBeat = wake(
            "evening journal", dueAgo: CompanionEvaluator.catchUpGrace + 600,
            wakeClass: .rhythm)
        let decision = evaluator.decide(signals(due: [staleBeat]))
        // Origin says triage; carriesBeat still runs the resurfacing ladder —
        // the two facts are independent and the decision carries both.
        #expect(
            decision == .wakeTurn(batch: [staleBeat], origin: .catchup, carriesBeat: true))
    }

    // MARK: - The attention gate and the deferral dedup

    @Test func aClosedGateOutranksDueness() {
        var evaluator = CompanionEvaluator()
        let due = wake()
        let decision = evaluator.decide(signals(due: [due], gateOpen: false))
        #expect(decision == .recordDeferral(dueCount: 1, firstWakeID: due.id))
    }

    @Test func deferralRecordsOncePerClosedGateEpisode() {
        var evaluator = CompanionEvaluator()
        let due = wake()
        _ = evaluator.decide(signals(due: [due], gateOpen: false))
        // The gate stays closed, the wake stays due: no second record.
        #expect(evaluator.decide(signals(due: [due], gateOpen: false)) == .wait)
        #expect(evaluator.decide(signals(due: [due], gateOpen: false)) == .wait)
    }

    @Test func theDedupResetsWhenTheGateReopens() {
        var evaluator = CompanionEvaluator()
        let due = wake()
        _ = evaluator.decide(signals(due: [due], gateOpen: false))
        // Gate opens — the batch fires; a later closed-gate episode records
        // its own deferral.
        _ = evaluator.decide(signals(due: [due], gateOpen: true))
        let decision = evaluator.decide(signals(due: [due], gateOpen: false))
        #expect(decision == .recordDeferral(dueCount: 1, firstWakeID: due.id))
    }

    @Test func aClosedGateOverNothingDueRecordsNothingAndBurnsNothing() {
        var evaluator = CompanionEvaluator()
        #expect(evaluator.decide(signals(gateOpen: false)) == .wait)
        // A wake becomes due while the gate is still closed: the episode's one
        // record fires now — the empty ticks did not consume it.
        let due = wake()
        let decision = evaluator.decide(signals(due: [due], gateOpen: false))
        #expect(decision == .recordDeferral(dueCount: 1, firstWakeID: due.id))
    }

    // MARK: - Day start

    @Test func firstPresenceOfTheDayStartsIt() {
        var evaluator = CompanionEvaluator()
        let decision = evaluator.decide(signals(hour: 9))
        guard case .dayStart(let updated) = decision else {
            Issue.record("expected dayStart, got \(decision)")
            return
        }
        #expect(updated.dayStartedAt == Self.now)
    }

    @Test func aOneAMTailCountsAsYesterday() {
        var evaluator = CompanionEvaluator()
        #expect(evaluator.decide(signals(hour: 3)) == .wait)
        #expect(
            evaluator.decide(signals(hour: CompanionEvaluator.dayStartEarliestHour))
                != .wait)
    }

    @Test func dayStartWaitsForHimToActuallyBeThere() {
        var evaluator = CompanionEvaluator()
        #expect(evaluator.decide(signals(hour: 9, present: false)) == .wait)
    }

    @Test func dayStartFiresOnlyOncePerDay() {
        var evaluator = CompanionEvaluator()
        let started = CompanionLoopDayState(
            dayStartedAt: Self.now.addingTimeInterval(-3600),
            lastAmbientAt: Self.now.addingTimeInterval(-60))
        #expect(evaluator.decide(signals(hour: 9, day: started)) == .wait)
    }

    @Test func dueWakesOutrankTheDayStart() {
        var evaluator = CompanionEvaluator()
        let due = wake()
        let decision = evaluator.decide(signals(hour: 9, due: [due]))
        #expect(decision == .wakeTurn(batch: [due], origin: .wake, carriesBeat: false))
    }

    // MARK: - Ambient eligibility

    @Test func ambientNeedsTheWholeEligibilityGate() {
        let started = CompanionLoopDayState(
            dayStartedAt: Self.now.addingTimeInterval(-3600))

        var evaluator = CompanionEvaluator()
        let eligible = evaluator.decide(signals(day: started))
        guard case .ambient(let updated) = eligible else {
            Issue.record("expected ambient, got \(eligible)")
            return
        }
        #expect(updated.lastAmbientAt == Self.now)
        #expect(updated.dayStartedAt == started.dayStartedAt)

        // Each leg of the gate alone keeps it closed: battery, busy GPU, a
        // day that never started.
        #expect(evaluator.decide(signals(day: started, ac: false)) == .wait)
        #expect(evaluator.decide(signals(day: started, gpuBusy: true)) == .wait)
        #expect(evaluator.decide(signals(hour: 3)) == .wait)
    }

    @Test func ambientSpacingHoldsUntilTheIntervalPasses() {
        var evaluator = CompanionEvaluator()
        let recent = CompanionLoopDayState(
            dayStartedAt: Self.now.addingTimeInterval(-4 * 3600),
            lastAmbientAt: Self.now.addingTimeInterval(-CompanionEvaluator.ambientSpacing + 60))
        #expect(evaluator.decide(signals(day: recent)) == .wait)

        let spaced = CompanionLoopDayState(
            dayStartedAt: Self.now.addingTimeInterval(-4 * 3600),
            lastAmbientAt: Self.now.addingTimeInterval(-CompanionEvaluator.ambientSpacing - 60))
        guard case .ambient = evaluator.decide(signals(day: spaced)) else {
            Issue.record("expected ambient after the spacing interval")
            return
        }
    }
}
