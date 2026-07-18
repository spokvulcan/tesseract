//
//  CompanionEvaluatorTests.swift
//  tesseractTests
//
//  The fold clock's decision tables (ADR-0043's pure shape, ADR-0046 #371):
//  one gathered `Signals` snapshot in, one `Decision` out — no store, no real
//  clock, no loop. The purist rule, the coalescing window, the deferral
//  dedup, the origin pick, and day-start-as-perception are each pinned here;
//  the retired grants (ambient cadence, attention gate) are pinned absent.
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

    /// A pending Event as the store would hand it back — `admittedAt` drives
    /// the coalescing table, so the full-row init is the right fixture.
    private func event(
        _ content: String = "he plugged in",
        kind: CompanionEventKind = .powerChange,
        admittedAgo: TimeInterval = 60
    ) -> CompanionEvent {
        CompanionEvent(
            id: UUID(), kind: kind, content: content, payload: nil,
            occurredAt: Self.now.addingTimeInterval(-admittedAgo), state: .pending,
            seq: nil, admittedAt: Self.now.addingTimeInterval(-admittedAgo),
            presentedAt: nil, consumedAt: nil, turnID: nil)
    }

    private func signals(
        hour: Int = 10,
        pending: [CompanionEvent] = [],
        due: [CompanionWake] = [],
        day: CompanionLoopDayState = CompanionLoopDayState(),
        present: Bool = true,
        ac: Bool = true,
        gpuBusy: Bool = false
    ) -> CompanionEvaluator.Signals {
        CompanionEvaluator.Signals(
            now: Self.now, localHour: hour, pendingEvents: pending, dueWakes: due,
            dayState: day, ownerPresent: present, onACPower: ac, gpuBusy: gpuBusy)
    }

    /// A day already started, so a table about something else reads its own
    /// rule instead of tripping day-start perception.
    private func settledDay() -> CompanionLoopDayState {
        CompanionLoopDayState(dayStartedAt: Self.now.addingTimeInterval(-4 * 3600))
    }

    // MARK: - The purist rule

    @Test func aQuietQueueGrantsNothingForever() {
        var evaluator = CompanionEvaluator()
        // Present, on AC, GPU free, mid-morning — every old ambient
        // eligibility leg green, and still: no pending, no due, no turn.
        for _ in 0..<50 {
            #expect(evaluator.decide(signals(day: settledDay())) == .wait)
        }
    }

    @Test func pendingEventsAloneGrantTheFoldTurn() {
        var evaluator = CompanionEvaluator()
        let decision = evaluator.decide(
            signals(pending: [event()], day: settledDay()))
        #expect(decision == .foldTurn(dueWakes: [], origin: .event, carriesBeat: false))
    }

    @Test func aReportBackDepositGrantsTheFoldTurn() {
        var evaluator = CompanionEvaluator()
        // #372: a summoned dialogue's deposit rides the drain like any other
        // perception — the next turn is how the one mind learns what its
        // conversation concluded.
        let deposit = event(
            "He decided to move the dentist to Thursday.",
            kind: .reportBack, admittedAgo: 60)
        let decision = evaluator.decide(signals(pending: [deposit], day: settledDay()))
        #expect(decision == .foldTurn(dueWakes: [], origin: .event, carriesBeat: false))
    }

    @Test func theEventGrantIgnoresPowerAndPresence() {
        var evaluator = CompanionEvaluator()
        // Battery, owner away: the fold still runs — eligibility is the model
        // slot only; the old ambient gate legs died with #371.
        let decision = evaluator.decide(
            signals(pending: [event()], day: settledDay(), present: false, ac: false))
        #expect(decision == .foldTurn(dueWakes: [], origin: .event, carriesBeat: false))
    }

    // MARK: - Due wakes and the origin pick

    @Test func dueWakesGrantOneTurnCarryingTheWholeBatch() {
        var evaluator = CompanionEvaluator()
        let first = wake("morning check", dueAgo: 120)
        let second = wake("the dentist", dueAgo: 60, wakeClass: .followup)
        let decision = evaluator.decide(signals(due: [first, second], day: settledDay()))
        #expect(
            decision
                == .foldTurn(dueWakes: [first, second], origin: .wake, carriesBeat: false))
    }

    @Test func aBatchCarryingARhythmWakeIsABeat() {
        var evaluator = CompanionEvaluator()
        let beat = wake("evening journal", dueAgo: 60, wakeClass: .rhythm)
        let promise = wake(dueAgo: 30)
        let decision = evaluator.decide(signals(due: [beat, promise], day: settledDay()))
        #expect(
            decision
                == .foldTurn(dueWakes: [beat, promise], origin: .beat, carriesBeat: true))
    }

    @Test func anOverdueWakeMakesTheWholeTurnATriage() {
        var evaluator = CompanionEvaluator()
        let stale = wake("missed while asleep", dueAgo: CompanionEvaluator.catchUpGrace + 60)
        let fresh = wake("on time", dueAgo: 60)
        let decision = evaluator.decide(signals(due: [stale, fresh], day: settledDay()))
        // The fold reasons over the whole backlog at once — no preemption
        // split (that was the pre-fold shape): one triage turn, both wakes.
        #expect(
            decision
                == .foldTurn(dueWakes: [stale, fresh], origin: .catchup, carriesBeat: false))
    }

    @Test func overdueByExactlyTheGraceIsNotOverdue() {
        var evaluator = CompanionEvaluator()
        let edge = wake("on the line", dueAgo: CompanionEvaluator.catchUpGrace)
        let decision = evaluator.decide(signals(due: [edge], day: settledDay()))
        #expect(decision == .foldTurn(dueWakes: [edge], origin: .wake, carriesBeat: false))
    }

    @Test func anOverdueRhythmWakeIsCatchupButStillCarriesTheBeat() {
        var evaluator = CompanionEvaluator()
        let staleBeat = wake(
            "evening journal", dueAgo: CompanionEvaluator.catchUpGrace + 600,
            wakeClass: .rhythm)
        let decision = evaluator.decide(signals(due: [staleBeat], day: settledDay()))
        // Origin says triage; carriesBeat still runs the resurfacing ladder —
        // the two facts are independent and the decision carries both.
        #expect(
            decision == .foldTurn(dueWakes: [staleBeat], origin: .catchup, carriesBeat: true))
    }

    @Test func aDueWakeOutranksTheEventOriginEvenWithEventsPending() {
        var evaluator = CompanionEvaluator()
        let due = wake()
        let decision = evaluator.decide(
            signals(pending: [event()], due: [due], day: settledDay()))
        // One fold turn drains both; the wake names the occasion.
        #expect(decision == .foldTurn(dueWakes: [due], origin: .wake, carriesBeat: false))
    }

    // MARK: - Coalescing

    @Test func aBurstStillLandingWaitsOneBeat() {
        var evaluator = CompanionEvaluator()
        let landing = event("app switch", admittedAgo: CompanionEvaluator.coalesceWindow - 5)
        #expect(
            evaluator.decide(signals(pending: [landing], day: settledDay())) == .wait)
    }

    @Test func aSettledBurstFolds() {
        var evaluator = CompanionEvaluator()
        let settled = event("app switch", admittedAgo: CompanionEvaluator.coalesceWindow + 5)
        let decision = evaluator.decide(signals(pending: [settled], day: settledDay()))
        #expect(decision == .foldTurn(dueWakes: [], origin: .event, carriesBeat: false))
    }

    @Test func theNewestArrivalRestartsTheCoalesceClock() {
        var evaluator = CompanionEvaluator()
        let old = event("first", admittedAgo: 300)
        let fresh = event("still landing", admittedAgo: 2)
        #expect(
            evaluator.decide(signals(pending: [old, fresh], day: settledDay())) == .wait)
    }

    @Test func aDueWakeOutranksTheCoalesceWait() {
        var evaluator = CompanionEvaluator()
        let landing = event("still landing", admittedAgo: 2)
        let due = wake()
        let decision = evaluator.decide(
            signals(pending: [landing], due: [due], day: settledDay()))
        // The wake fires now; the fresh event rides along in the same drain.
        #expect(decision == .foldTurn(dueWakes: [due], origin: .wake, carriesBeat: false))
    }

    // MARK: - The model slot and the deferral dedup

    @Test func aBusySlotOutranksDueness() {
        var evaluator = CompanionEvaluator()
        let due = wake()
        let decision = evaluator.decide(
            signals(pending: [event()], due: [due], day: settledDay(), gpuBusy: true))
        #expect(decision == .recordDeferral(pendingCount: 2, firstWakeID: due.id))
    }

    @Test func deferralRecordsOncePerBusyEpisode() {
        var evaluator = CompanionEvaluator()
        let due = wake()
        _ = evaluator.decide(signals(due: [due], day: settledDay(), gpuBusy: true))
        // The slot stays taken, the wake stays due: no second record.
        #expect(evaluator.decide(signals(due: [due], day: settledDay(), gpuBusy: true)) == .wait)
        #expect(evaluator.decide(signals(due: [due], day: settledDay(), gpuBusy: true)) == .wait)
    }

    @Test func theDedupResetsWhenTheSlotFrees() {
        var evaluator = CompanionEvaluator()
        let due = wake()
        _ = evaluator.decide(signals(due: [due], day: settledDay(), gpuBusy: true))
        // Slot frees — the batch folds; a later busy episode records its own
        // deferral.
        _ = evaluator.decide(signals(due: [due], day: settledDay()))
        let decision = evaluator.decide(signals(due: [due], day: settledDay(), gpuBusy: true))
        #expect(decision == .recordDeferral(pendingCount: 1, firstWakeID: due.id))
    }

    @Test func theDedupAlsoResetsWhenTheQueueDrainsQuiet() {
        var evaluator = CompanionEvaluator()
        _ = evaluator.decide(signals(due: [wake()], day: settledDay(), gpuBusy: true))
        // Whatever was due got handled elsewhere; a quiet tick closes the
        // episode, so the next busy-over-due tick records again.
        _ = evaluator.decide(signals(day: settledDay(), gpuBusy: true))
        let due = wake("new work")
        let decision = evaluator.decide(signals(due: [due], day: settledDay(), gpuBusy: true))
        #expect(decision == .recordDeferral(pendingCount: 1, firstWakeID: due.id))
    }

    @Test func eventOnlyDeferralCarriesNoWakeID() {
        var evaluator = CompanionEvaluator()
        let decision = evaluator.decide(
            signals(pending: [event()], day: settledDay(), gpuBusy: true))
        #expect(decision == .recordDeferral(pendingCount: 1, firstWakeID: nil))
    }

    // MARK: - Day start as perception

    @Test func firstPresenceOfTheDayIsAPerceptionNotAGrant() {
        var evaluator = CompanionEvaluator()
        let decision = evaluator.decide(signals(hour: 9))
        guard case .perceiveDayStart(let updated) = decision else {
            Issue.record("expected perceiveDayStart, got \(decision)")
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
        #expect(evaluator.decide(signals(hour: 9, day: settledDay())) == .wait)
    }

    @Test func dayStartOutranksPendingWork() {
        var evaluator = CompanionEvaluator()
        // The perception lands first; the next tick folds the whole queue —
        // day-start event included — as one turn.
        let decision = evaluator.decide(signals(hour: 9, pending: [event()], due: [wake()]))
        guard case .perceiveDayStart = decision else {
            Issue.record("expected perceiveDayStart, got \(decision)")
            return
        }
    }

    @Test func dayStartFiresEvenOnBatteryAndABusySlot() {
        var evaluator = CompanionEvaluator()
        // Perception is a store write, not a generation — no eligibility leg
        // applies (the battery-morning liveness the old ambient gate broke).
        let decision = evaluator.decide(signals(hour: 9, ac: false, gpuBusy: true))
        guard case .perceiveDayStart = decision else {
            Issue.record("expected perceiveDayStart, got \(decision)")
            return
        }
    }
}
