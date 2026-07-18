//
//  CompanionEvaluatorTests.swift
//  tesseractTests
//
//  The fold clock's decision tables (ADR-0043's pure shape, ADR-0046 #371):
//  one gathered `Signals` snapshot in, one `Decision` out — no store, no real
//  clock, no loop. The purist rule, the coalescing window, the deferral
//  dedup, the origin pick, and the ceiling are each pinned here; the retired
//  grants (ambient cadence, attention gate) are pinned absent by the
//  `Signals` shape itself — presence, power, and the hour are no longer even
//  gatherable. Day-start detection lives with the producers now
//  (`CompanionPerceptionTests`).
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
        pending: [CompanionEvent] = [],
        due: [CompanionWake] = [],
        gpuBusy: Bool = false,
        foldTokens: Int = 0
    ) -> CompanionEvaluator.Signals {
        CompanionEvaluator.Signals(
            now: Self.now, pendingEvents: pending, dueWakes: due, gpuBusy: gpuBusy,
            foldTokens: foldTokens)
    }

    // MARK: - The purist rule

    @Test func aQuietQueueGrantsNothingForever() {
        var evaluator = CompanionEvaluator()
        // GPU free, and still: no pending, no due, no turn. The old ambient
        // eligibility legs (presence, power, the hour) are not even signals
        // any more — nothing here can ever make a quiet queue fold.
        for _ in 0..<50 {
            #expect(evaluator.decide(signals()) == .wait)
        }
    }

    @Test func pendingEventsAloneGrantTheFoldTurn() {
        var evaluator = CompanionEvaluator()
        let decision = evaluator.decide(signals(pending: [event()]))
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
        let decision = evaluator.decide(signals(pending: [deposit]))
        #expect(decision == .foldTurn(dueWakes: [], origin: .event, carriesBeat: false))
    }

    // MARK: - Due wakes and the origin pick

    @Test func dueWakesGrantOneTurnCarryingTheWholeBatch() {
        var evaluator = CompanionEvaluator()
        let first = wake("morning check", dueAgo: 120)
        let second = wake("the dentist", dueAgo: 60, wakeClass: .followup)
        let decision = evaluator.decide(signals(due: [first, second]))
        #expect(
            decision
                == .foldTurn(dueWakes: [first, second], origin: .wake, carriesBeat: false))
    }

    @Test func aBatchCarryingARhythmWakeIsABeat() {
        var evaluator = CompanionEvaluator()
        let beat = wake("evening journal", dueAgo: 60, wakeClass: .rhythm)
        let promise = wake(dueAgo: 30)
        let decision = evaluator.decide(signals(due: [beat, promise]))
        #expect(
            decision
                == .foldTurn(dueWakes: [beat, promise], origin: .beat, carriesBeat: true))
    }

    @Test func anOverdueWakeMakesTheWholeTurnATriage() {
        var evaluator = CompanionEvaluator()
        let stale = wake("missed while asleep", dueAgo: CompanionEvaluator.catchUpGrace + 60)
        let fresh = wake("on time", dueAgo: 60)
        let decision = evaluator.decide(signals(due: [stale, fresh]))
        // The fold reasons over the whole backlog at once — no preemption
        // split (that was the pre-fold shape): one triage turn, both wakes.
        #expect(
            decision
                == .foldTurn(dueWakes: [stale, fresh], origin: .catchup, carriesBeat: false))
    }

    @Test func overdueByExactlyTheGraceIsNotOverdue() {
        var evaluator = CompanionEvaluator()
        let edge = wake("on the line", dueAgo: CompanionEvaluator.catchUpGrace)
        let decision = evaluator.decide(signals(due: [edge]))
        #expect(decision == .foldTurn(dueWakes: [edge], origin: .wake, carriesBeat: false))
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
            decision == .foldTurn(dueWakes: [staleBeat], origin: .catchup, carriesBeat: true))
    }

    @Test func aDueWakeOutranksTheEventOriginEvenWithEventsPending() {
        var evaluator = CompanionEvaluator()
        let due = wake()
        let decision = evaluator.decide(signals(pending: [event()], due: [due]))
        // One fold turn drains both; the wake names the occasion.
        #expect(decision == .foldTurn(dueWakes: [due], origin: .wake, carriesBeat: false))
    }

    // MARK: - Coalescing

    @Test func aBurstStillLandingWaitsOneBeat() {
        var evaluator = CompanionEvaluator()
        let landing = event("app switch", admittedAgo: CompanionEvaluator.coalesceWindow - 5)
        #expect(evaluator.decide(signals(pending: [landing])) == .wait)
    }

    @Test func aSettledBurstFolds() {
        var evaluator = CompanionEvaluator()
        let settled = event("app switch", admittedAgo: CompanionEvaluator.coalesceWindow + 5)
        let decision = evaluator.decide(signals(pending: [settled]))
        #expect(decision == .foldTurn(dueWakes: [], origin: .event, carriesBeat: false))
    }

    @Test func theNewestArrivalRestartsTheCoalesceClock() {
        var evaluator = CompanionEvaluator()
        let old = event("first", admittedAgo: 300)
        let fresh = event("still landing", admittedAgo: 2)
        #expect(evaluator.decide(signals(pending: [old, fresh])) == .wait)
    }

    @Test func aDueWakeOutranksTheCoalesceWait() {
        var evaluator = CompanionEvaluator()
        let landing = event("still landing", admittedAgo: 2)
        let due = wake()
        let decision = evaluator.decide(signals(pending: [landing], due: [due]))
        // The wake fires now; the fresh event rides along in the same drain.
        #expect(decision == .foldTurn(dueWakes: [due], origin: .wake, carriesBeat: false))
    }

    // MARK: - The model slot and the deferral dedup

    @Test func aBusySlotOutranksDueness() {
        var evaluator = CompanionEvaluator()
        let due = wake()
        let decision = evaluator.decide(
            signals(pending: [event()], due: [due], gpuBusy: true))
        #expect(decision == .recordDeferral(pendingCount: 2, firstWakeID: due.id))
    }

    @Test func deferralRecordsOncePerBusyEpisode() {
        var evaluator = CompanionEvaluator()
        let due = wake()
        _ = evaluator.decide(signals(due: [due], gpuBusy: true))
        // The slot stays taken, the wake stays due: no second record.
        #expect(evaluator.decide(signals(due: [due], gpuBusy: true)) == .wait)
        #expect(evaluator.decide(signals(due: [due], gpuBusy: true)) == .wait)
    }

    @Test func theDedupResetsWhenTheSlotFrees() {
        var evaluator = CompanionEvaluator()
        let due = wake()
        _ = evaluator.decide(signals(due: [due], gpuBusy: true))
        // Slot frees — the batch folds; a later busy episode records its own
        // deferral.
        _ = evaluator.decide(signals(due: [due]))
        let decision = evaluator.decide(signals(due: [due], gpuBusy: true))
        #expect(decision == .recordDeferral(pendingCount: 1, firstWakeID: due.id))
    }

    @Test func theDedupAlsoResetsWhenTheQueueDrainsQuiet() {
        var evaluator = CompanionEvaluator()
        _ = evaluator.decide(signals(due: [wake()], gpuBusy: true))
        // Whatever was due got handled elsewhere; a quiet tick closes the
        // episode, so the next busy-over-due tick records again.
        _ = evaluator.decide(signals(gpuBusy: true))
        let due = wake("new work")
        let decision = evaluator.decide(signals(due: [due], gpuBusy: true))
        #expect(decision == .recordDeferral(pendingCount: 1, firstWakeID: due.id))
    }

    @Test func eventOnlyDeferralCarriesNoWakeID() {
        var evaluator = CompanionEvaluator()
        let decision = evaluator.decide(signals(pending: [event()], gpuBusy: true))
        #expect(decision == .recordDeferral(pendingCount: 1, firstWakeID: nil))
    }

    // MARK: - The ceiling (#373)

    @Test func theCeilingGrantsTheFoldDownEvenOnAQuietQueue() {
        var evaluator = CompanionEvaluator()
        let over = CompanionEvaluator.contextCeilingTokens
        let decision = evaluator.decide(signals(foldTokens: over))
        #expect(decision == .compactFold(estimatedTokens: over))
    }

    @Test func theCeilingOutranksPendingWork() {
        var evaluator = CompanionEvaluator()
        let over = CompanionEvaluator.contextCeilingTokens + 5_000
        // A turn must never append past the budget: the fold-down runs first;
        // the queue and the wake drain on the next tick, over the folded head.
        let decision = evaluator.decide(
            signals(pending: [event()], due: [wake()], foldTokens: over))
        #expect(decision == .compactFold(estimatedTokens: over))
    }

    @Test func aBusySlotHoldsTheFoldDownQuietly() {
        var evaluator = CompanionEvaluator()
        let over = CompanionEvaluator.contextCeilingTokens
        // Over the ceiling with the slot taken: no grant, and the normal
        // deferral bookkeeping still speaks for the due work behind it.
        #expect(evaluator.decide(signals(gpuBusy: true, foldTokens: over)) == .wait)
        let due = wake()
        let decision = evaluator.decide(signals(due: [due], gpuBusy: true, foldTokens: over))
        #expect(decision == .recordDeferral(pendingCount: 1, firstWakeID: due.id))
    }

    @Test func theFoldDownFiresAtTheHeadroomLineNotTheCeiling() {
        var evaluator = CompanionEvaluator()
        // Pre-emptive: the fold runs one turn's growth BEFORE the ceiling, so
        // a granted turn can never append past the ceiling itself.
        let line =
            CompanionEvaluator.contextCeilingTokens
            - CompanionEvaluator.ceilingHeadroomTokens
        #expect(
            evaluator.decide(signals(foldTokens: line))
                == .compactFold(estimatedTokens: line))
    }

    @Test func underTheHeadroomLineNothingChanges() {
        var evaluator = CompanionEvaluator()
        let under =
            CompanionEvaluator.contextCeilingTokens
            - CompanionEvaluator.ceilingHeadroomTokens - 1
        #expect(evaluator.decide(signals(foldTokens: under)) == .wait)
    }
}
