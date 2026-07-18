//
//  CompanionEvaluator.swift
//  tesseract
//
//  The Wake Evaluator ADR-0040 §2 promised: `(now, persisted loop state,
//  signals) → due wakes / eligibility` — a pure function, replayable over a
//  recorder snapshot. Every tick the loop gathers one `Signals` snapshot and
//  this decider returns at most one `Decision`: which turn to grant, a
//  deferral to record, or nothing. Due-ness and eligibility only — judgment
//  is the entity's, at the turn this grants. The loop performs: store writes,
//  recorder events, and the turn itself never happen here.
//

import Foundation

nonisolated struct CompanionEvaluator {

    /// Overdue past this: the wake goes to a catch-up triage turn instead of
    /// firing as if the moment were now.
    static let catchUpGrace: TimeInterval = 30 * 60
    /// Ambient turns: at most one per this interval (ADR-0040 §7, revisable).
    static let ambientSpacing: TimeInterval = 30 * 60
    /// Day start needs the calendar day to have begun in earnest — a 1 a.m.
    /// tail counts as yesterday.
    static let dayStartEarliestHour = 4

    /// One tick's gathered facts. Gathering is the loop's job and effectful;
    /// deciding over the gathered value is this module's, and is not.
    struct Signals: Equatable, Sendable {
        var now: Date
        /// The local calendar hour of `now` — the day-start gate's clock.
        var localHour: Int
        /// Booked wakes whose due time has arrived, ordered by due.
        var dueWakes: [CompanionWake]
        var dayState: CompanionLoopDayState
        /// The attention gate's verdict for this tick (it outranks due-ness).
        var gateOpen: Bool
        /// The one home of "he is with the machine": recent input, unlocked.
        var ownerPresent: Bool
        var onACPower: Bool
        var gpuBusy: Bool
    }

    enum Decision: Equatable, Sendable {
        /// Nothing is due and nothing is eligible — the usual tick.
        case wait
        /// The gate is closed over due work: record one deferral (this fires
        /// once per closed-gate episode; the dedup lives here, not the loop).
        case recordDeferral(dueCount: Int, firstWakeID: UUID?)
        /// Grant a turn for this batch. `carriesBeat` is the resurfacing
        /// trigger — a batch carrying a rhythm wake is a beat even when it
        /// fires overdue as `.catchup`, so origin alone cannot encode it.
        case wakeTurn(batch: [CompanionWake], origin: TurnOrigin, carriesBeat: Bool)
        /// First presence of the calendar day: persist `updated`, then grant
        /// the day-start transition turn.
        case dayStart(updated: CompanionLoopDayState)
        /// Ambient eligibility passed: persist `updated` (the spacing stamp),
        /// then grant the ambient turn.
        case ambient(updated: CompanionLoopDayState)
    }

    /// One deferral record per closed-gate episode — a tick every 30 s while
    /// he works must not spam the flight log.
    private var deferralLogged = false

    mutating func decide(_ signals: Signals) -> Decision {
        // The attention gate outranks due-ness: while the owner is using the
        // app (and for the quiet window after), nothing fires. Deferred wakes
        // stay booked and batch into one turn when the gate opens — the
        // evening journal must never run beside his live voice session again.
        guard signals.gateOpen else {
            if !deferralLogged, !signals.dueWakes.isEmpty {
                deferralLogged = true
                return .recordDeferral(
                    dueCount: signals.dueWakes.count,
                    firstWakeID: signals.dueWakes.first?.id)
            }
            return .wait
        }
        deferralLogged = false

        // 1. Due wakes — the entity's booked present. Anything past the
        // catch-up grace preempts the batch: the overdue triage first, the
        // merely-due re-present next tick. A batch carrying a rhythm wake is
        // a beat (#327 §2's origin vocabulary).
        if !signals.dueWakes.isEmpty {
            let overdue = signals.dueWakes.filter {
                signals.now.timeIntervalSince($0.due) > Self.catchUpGrace
            }
            let batch = overdue.isEmpty ? signals.dueWakes : overdue
            let carriesBeat = batch.contains { $0.wakeClass == .rhythm }
            let origin: TurnOrigin =
                !overdue.isEmpty ? .catchup : (carriesBeat ? .beat : .wake)
            return .wakeTurn(batch: batch, origin: origin, carriesBeat: carriesBeat)
        }

        // 2. Day start — first presence of the calendar day.
        if signals.dayState.dayStartedAt == nil,
            signals.localHour >= Self.dayStartEarliestHour,
            signals.ownerPresent
        {
            var updated = signals.dayState
            updated.dayStartedAt = signals.now
            return .dayStart(updated: updated)
        }

        // 3. Ambient cognition — eligibility, not judgment (ADR-0040 §7).
        if signals.dayState.dayStartedAt != nil,
            signals.onACPower,
            !signals.gpuBusy,
            signals.now.timeIntervalSince(signals.dayState.lastAmbientAt ?? .distantPast)
                > Self.ambientSpacing
        {
            var updated = signals.dayState
            updated.lastAmbientAt = signals.now
            return .ambient(updated: updated)
        }

        return .wait
    }
}
