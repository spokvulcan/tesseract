//
//  CompanionEvaluator.swift
//  tesseract
//
//  The Wake Evaluator, holding the fold's whole clock (ADR-0043's pure shape,
//  extended by ADR-0046 #371): every tick the loop gathers one `Signals`
//  snapshot and this decider returns at most one `Decision`. The purist rule:
//  a turn is granted iff (pending Events or a due Wake) and mechanically
//  eligible — there is no harness cadence and no safety tick; a quiet queue
//  grants no turns, indefinitely, by design. Due-ness and eligibility only —
//  judgment is the entity's, at the turn this grants. The loop performs:
//  store writes, recorder events, and the turn itself never happen here.
//
//  The ambient cadence, the Ambient Turn, and the attention gate's granting
//  role died here (#371): the one mechanical eligibility is the model slot,
//  and the owner's attention is protected by the arbiter's FIFO, not a gate.
//

import Foundation

nonisolated struct CompanionEvaluator {

    /// Overdue past this: the fold turn is a catch-up triage — the opening
    /// says so — instead of firing as if the moment were now.
    static let catchUpGrace: TimeInterval = 30 * 60
    /// The coalescing window (ADR-0046): a burst still landing waits one
    /// beat so one turn drains it whole — unless a wake is due, which fires
    /// now regardless.
    static let coalesceWindow: TimeInterval = 10
    /// Day start needs the calendar day to have begun in earnest — a 1 a.m.
    /// tail counts as yesterday.
    static let dayStartEarliestHour = 4
    /// Mission Control's context ceiling (ADR-0046 #373): the fold-down
    /// outranks any new grant once the conversation is within one turn's
    /// growth of it — pre-emptive, so a granted turn never appends past the
    /// ceiling itself.
    static let contextCeilingTokens = 80_000
    /// One generous turn's growth (opening + reply + tools). The fold fires
    /// at `ceiling - headroom`, keeping the ceiling a hard line, not a
    /// high-water mark.
    static let ceilingHeadroomTokens = 8_000

    /// One tick's gathered facts. Gathering is the loop's job and effectful;
    /// deciding over the gathered value is this module's, and is not.
    struct Signals: Equatable, Sendable {
        var now: Date
        /// The local calendar hour of `now` — the day-start gate's clock.
        var localHour: Int
        /// Everything pending in the Event queue, in total order.
        var pendingEvents: [CompanionEvent]
        /// Booked wakes whose due time has arrived, ordered by due.
        var dueWakes: [CompanionWake]
        var dayState: CompanionLoopDayState
        /// The one home of "he is with the machine": recent input, unlocked.
        var ownerPresent: Bool
        var onACPower: Bool
        var gpuBusy: Bool
        /// Mission Control's estimated size (#373) — the ceiling's signal.
        var foldTokens: Int = 0
    }

    enum Decision: Equatable, Sendable {
        /// Nothing is pending and nothing is due — the usual tick, forever if
        /// need be: silence is the entity's, not a timer's.
        case wait
        /// The model slot is taken over due work: record one deferral (once
        /// per busy episode; the dedup lives here, not in the loop).
        case recordDeferral(pendingCount: Int, firstWakeID: UUID?)
        /// Grant the fold turn: drain everything pending, fire these wakes.
        /// `carriesBeat` is the resurfacing trigger — a batch carrying a
        /// rhythm wake is a beat even when it fires overdue as `.catchup`.
        case foldTurn(dueWakes: [CompanionWake], origin: TurnOrigin, carriesBeat: Bool)
        /// First presence of the calendar day: persist `updated` and admit
        /// the day-start Event — the turn then follows over the queue, like
        /// every other perception (the de facto morning liveness).
        case perceiveDayStart(updated: CompanionLoopDayState)
        /// The ceiling is hit (#373): run the fold-down now — the entity
        /// authors its Digest and the splice lands, on the record — before
        /// any turn appends past the budget.
        case compactFold(estimatedTokens: Int)
    }

    /// One deferral record per busy episode — a tick every 30 s behind a long
    /// generation must not spam the flight log.
    private var deferralLogged = false

    mutating func decide(_ signals: Signals) -> Decision {
        // 0. Day start is perception, not a grant: the first presence of the
        // calendar day becomes an Event in the queue.
        if signals.dayState.dayStartedAt == nil,
            signals.localHour >= Self.dayStartEarliestHour,
            signals.ownerPresent
        {
            var updated = signals.dayState
            updated.dayStartedAt = signals.now
            return .perceiveDayStart(updated: updated)
        }

        // 1. The ceiling outranks every grant (#373): a turn must never
        // append past the budget, so within one turn's headroom of the
        // ceiling the fold-down runs first. Same one eligibility — the model
        // slot; while it is taken, hold quietly (the turn behind it defers
        // through the normal path when the slot frees).
        if signals.foldTokens >= Self.contextCeilingTokens - Self.ceilingHeadroomTokens,
            !signals.gpuBusy
        {
            return .compactFold(estimatedTokens: signals.foldTokens)
        }

        // 2. The purist clock: pending Events or a due Wake, or nothing.
        guard !signals.pendingEvents.isEmpty || !signals.dueWakes.isEmpty else {
            deferralLogged = false
            return .wait
        }

        // 3. Mechanical eligibility — the model slot. Ineligibility over due
        // work is recorded once per episode, then held quietly.
        if signals.gpuBusy {
            if !deferralLogged {
                deferralLogged = true
                return .recordDeferral(
                    pendingCount: signals.pendingEvents.count + signals.dueWakes.count,
                    firstWakeID: signals.dueWakes.first?.id)
            }
            return .wait
        }
        deferralLogged = false

        // 4. Coalescing: a burst still landing waits one beat so a single
        // turn drains it whole. A due wake outranks the wait — it fires now.
        if signals.dueWakes.isEmpty,
            let newest = signals.pendingEvents.compactMap(\.admittedAt).max(),
            signals.now.timeIntervalSince(newest) < Self.coalesceWindow
        {
            return .wait
        }

        // 5. The fold turn. A batch carrying a rhythm wake is a beat; any
        // wake past the catch-up grace makes the whole turn a triage.
        let carriesBeat = signals.dueWakes.contains { $0.wakeClass == .rhythm }
        let overdue = signals.dueWakes.contains {
            signals.now.timeIntervalSince($0.due) > Self.catchUpGrace
        }
        let origin: TurnOrigin =
            signals.dueWakes.isEmpty
            ? .event : (overdue ? .catchup : (carriesBeat ? .beat : .wake))
        return .foldTurn(dueWakes: signals.dueWakes, origin: origin, carriesBeat: carriesBeat)
    }
}
