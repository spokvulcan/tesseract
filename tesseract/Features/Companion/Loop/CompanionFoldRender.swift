//
//  CompanionFoldRender.swift
//  tesseract
//
//  The line primitives both briefings compose (#403). The turn-facing
//  Situation Briefing (CompanionBriefing) and the chat-facing Fold Briefing
//  (CompanionFoldBriefing) project the same domain concepts — today's
//  contract, due wakes, upcoming wakes, recently-fired wakes — and had each
//  hand-rendered them, drifting apart by hand ("DUE NOW —" vs "Due now:",
//  "Your booked future:" vs "Booked ahead:"). One rendering per concept lives
//  here; headings and section order stay per-surface (a turn opening and a
//  chat preamble differ legitimately). ADR-0052 reuses the Situation
//  Briefing's discipline for the Fold Briefing; a third surface should
//  inherit these lines, not fork them a third time.
//

import Foundation

nonisolated enum CompanionFoldRender {

    /// Today's contract, or its absence — rendered identically on both
    /// surfaces already, so it collapses to one call with no phrasing choice.
    static func contractLine(today: DayRecord?) -> String {
        guard let today, !today.chain.isEmpty else { return "No contract for today yet." }
        return today.chainSummary
    }

    /// The due-wakes heading. Phrasing pick: the situation said "DUE NOW — the
    /// wakes that triggered this turn:", the fold said "Due now:". The "DUE
    /// NOW" emphasis wins on both (a due wake is the loudest thing on either
    /// surface); the "…triggered this turn" clause is real information only
    /// the turn carries — a chat's due wakes did not occasion the chat — so it
    /// rides only when `triggeredThisTurn`.
    static func dueHeading(triggeredThisTurn: Bool) -> String {
        triggeredThisTurn
            ? "DUE NOW — the wakes that triggered this turn:"
            : "DUE NOW:"
    }

    /// One due wake as a line. `now` supplied (the situation, where the wake
    /// literally triggered this turn) appends its overdue lateness past a
    /// 5-minute grace; the fold lists due wakes without a turn-relative
    /// lateness it has no turn to relate to.
    static func dueWakeLine(_ wake: CompanionWake, now: Date? = nil) -> String {
        let base = "- \(wake.briefingLine)"
        guard let now else { return base }
        let overdue = Int(now.timeIntervalSince(wake.due) / 60)
        return overdue > 5 ? base + " (overdue by \(overdue) min)" : base
    }

    /// The upcoming-wakes heading. Phrasing pick: the situation said "Your
    /// booked future:", the fold said "Booked ahead:". "Booked ahead:" wins —
    /// leaner, in the situation's own pilot-instrument register, and it shares
    /// vocabulary with the situation's empty-state line ("You have NOTHING
    /// booked ahead").
    static let upcomingHeading = "Booked ahead:"

    /// One upcoming wake as a line — its due date and time, then the shared
    /// briefing line. Identical on both surfaces already.
    static func upcomingWakeLine(_ wake: CompanionWake) -> String {
        "- \(wake.due.formatted(date: .abbreviated, time: .shortened)) "
            + wake.briefingLine
    }

    /// One recently-fired ("delivery") wake as a line — the fold's account of
    /// what already went out (ADR-0052): the fire time, the shared briefing
    /// line, then the wake's terminal state and whether it was ever heard.
    /// Fold-only today; owned here because it builds on `wake.briefingLine`
    /// and a future surface reporting deliveries should inherit it.
    static func firedWakeLine(_ wake: CompanionWake) -> String {
        let when = wake.firedAt?.formatted(date: .omitted, time: .shortened) ?? "?"
        return "- \(when) \(wake.briefingLine) "
            + "(\(wake.state.rawValue)\(wake.heardAt == nil ? ", unheard" : ""))"
    }
}
