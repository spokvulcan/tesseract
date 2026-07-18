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
    /// Terminal: the entity deliberately withdrew it through `cancel_wake`
    /// (#369) — a recorded decision with a why, never the silent-loss defect.
    case cancelled
    /// Terminal defect: silently lost. The aggregator counts these as defects;
    /// nothing in the app writes it on purpose.
    case dropped
}

nonisolated struct CompanionWake: Sendable, Identifiable, Equatable {
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
    /// The owner's first reaction of any kind (engage, reply, dismiss) — the
    /// resurfacing ladder's heard-vs-ignored evidence (#309). Nil means no
    /// reaction ever reached this wake.
    var heardAt: Date?

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
        consumedAt: Date? = nil,
        heardAt: Date? = nil
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
        self.heardAt = heardAt
    }
}

/// #309's ignored-promise ladder, app-written (state transitions are never the
/// entity's): a delivered promise he never reacted to rides the next beat's
/// agenda once (`resurfaced`); if it is still unheard when the following beat
/// fires, it dies `delivered_unheard` — no third attempt, and the death is a
/// recorded fact the weekly review counts (#313's promise-integrity measure).
nonisolated enum CompanionResurfacing {

    /// Runs when a rhythm beat fires. Kill pass first (what the previous beat
    /// resurfaced and he still never heard is dead); then the resurface pass
    /// (newly ignored promises join this beat's agenda). Returns the promises
    /// the current beat should carry as agenda lines.
    static func pass(
        store: MemoryStore, recorder: CompanionFlightRecorder, now: Date = Date()
    ) async -> [CompanionWake] {
        if let stale = try? await store.resurfacedWakes() {
            for var wake in stale {
                if wake.heardAt == nil {
                    wake.state = .deliveredUnheard
                    recorder.record(
                        "wake.delivered-unheard", wakeID: wake.id, note: wake.content)
                } else {
                    // Resurfaced and heard — it did its job; terminal delivered.
                    wake.state = .delivered
                }
                try? await store.upsertWake(wake)
            }
        }

        guard let candidates = try? await store.unheardDeliveredPromises(), !candidates.isEmpty
        else { return [] }
        var resurfaced: [CompanionWake] = []
        for var wake in candidates {
            wake.state = .resurfaced
            try? await store.upsertWake(wake)
            recorder.record("wake.resurfaced", wakeID: wake.id, note: wake.content)
            resurfaced.append(wake)
        }
        return resurfaced
    }
}

/// The loop's small per-day persisted state — what must survive a restart but
/// is not derivable from the wakes table.
nonisolated struct CompanionLoopDayState: Codable, Sendable, Equatable {
    /// The day-start perception fired (first sustained presence after the
    /// overnight gap). Nil until then. (`lastAmbientAt` lived here until #371
    /// retired the ambient cadence; a persisted key decodes ignored.)
    var dayStartedAt: Date?
    /// The day's standing-instructions review ran (#370) — sleep passes fire
    /// on every idle, the review at most once per day.
    var instructionsReviewedAt: Date?

    init(dayStartedAt: Date? = nil, instructionsReviewedAt: Date? = nil) {
        self.dayStartedAt = dayStartedAt
        self.instructionsReviewedAt = instructionsReviewedAt
    }
}
