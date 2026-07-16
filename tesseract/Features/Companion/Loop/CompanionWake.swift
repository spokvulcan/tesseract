//
//  CompanionWake.swift
//  tesseract
//
//  A wake is one persisted row granting the entity a future turn (ADR-0040):
//  a promise, a rhythm beat, a follow-up, or a re-summons — one table, one
//  machinery, one invariant: a wake is consumed only by a completed turn.
//  The entity books them through a typed tool; state transitions are written
//  only by app code.
//

import Foundation

nonisolated enum CompanionWakeClass: String, Codable, Sendable {
    /// A self-booked touchpoint about something specific (#309) — quiet
    /// delivery, must-fire, ~2/day discretionary budget.
    case promise
    /// The daily rhythm's beats — morning planning, midday pulse, evening
    /// journal. Booked by the entity itself (it books its own day).
    case rhythm
    /// "Wake me in 40 minutes to see if he started" — cheap self-checks.
    case followup
    /// An escalation repeat: "he's here and hasn't answered — wake me in 12".
    case resummons
}

nonisolated enum CompanionWakeState: String, Codable, Sendable {
    /// Waiting for its due time. The only state the evaluator fires from.
    case booked
    /// Presented to the entity — a turn is running (or died mid-run; an
    /// unconsumed fired wake re-presents, that is the correctness invariant).
    case fired
    /// Terminal: the turn delivered it (or the harness's generic fallback did).
    case delivered
    /// Terminal: the owner engaged with it.
    case engaged
    /// Resurfaced once in a later beat's agenda after being ignored (#309).
    case resurfaced
    /// Terminal: resurfaced and still unheard — dead, no third attempt.
    case deliveredUnheard = "delivered_unheard"
    /// Terminal defect: silently lost. The aggregator counts these as defects;
    /// nothing in the app writes it on purpose.
    case dropped
}

nonisolated struct CompanionWake: Sendable, Identifiable {
    let id: UUID
    var content: String
    var due: Date
    let wakeClass: CompanionWakeClass
    var state: CompanionWakeState
    /// "Wake me for this" (#309): grants the spoken-summons ladder; without it
    /// a promise delivers quietly.
    var summonsGrant: Bool
    /// The conversation whose turn booked it — provenance for "why did you
    /// ping me?".
    var conversationID: UUID?
    let createdAt: Date
    var updatedAt: Date
    var firedAt: Date?
    var consumedAt: Date?

    init(
        id: UUID = UUID(),
        content: String,
        due: Date,
        wakeClass: CompanionWakeClass = .promise,
        state: CompanionWakeState = .booked,
        summonsGrant: Bool = false,
        conversationID: UUID? = nil,
        createdAt: Date = Date(),
        updatedAt: Date = Date(),
        firedAt: Date? = nil,
        consumedAt: Date? = nil
    ) {
        self.id = id
        self.content = content
        self.due = due
        self.wakeClass = wakeClass
        self.state = state
        self.summonsGrant = summonsGrant
        self.conversationID = conversationID
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.firedAt = firedAt
        self.consumedAt = consumedAt
    }
}

/// The loop's small per-day persisted state — what must survive a restart but
/// is not derivable from the wakes table.
nonisolated struct CompanionLoopDayState: Codable, Sendable {
    /// The day-start transition fired (first sustained presence after the
    /// overnight gap). Nil until then.
    var dayStartedAt: Date?
    /// The last ambient turn's completion — the spacing gate reads it.
    var lastAmbientAt: Date?

    init(dayStartedAt: Date? = nil, lastAmbientAt: Date? = nil) {
        self.dayStartedAt = dayStartedAt
        self.lastAmbientAt = lastAmbientAt
    }
}
