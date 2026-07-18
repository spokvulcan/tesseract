//
//  CompanionEvent.swift
//  tesseract
//
//  One perception queued for the entity (ADR-0046, #368): every digital input
//  becomes exactly one Event, events queue in total order, and a granted turn
//  drains everything pending. This file is the vocabulary; the queue's math
//  lives in `MemoryStore+Events.swift`, the producers in
//  `CompanionPerception`.
//

import CryptoKit
import Foundation

/// The v1 Event kinds. `wakeDue` and `reportBack` are defined here but
/// produced by their own tickets (#371's clock, #372's deposit door); the
/// rest are the perception substrate's. Raw values are the persisted tag.
nonisolated enum CompanionEventKind: String, Codable, Sendable {
    /// A booked wake came due (#371 — the clock admits these).
    case wakeDue = "wake-due"
    /// A summoned dialogue's deposit landed (#372).
    case reportBack = "report-back"
    /// First presence of the calendar day.
    case dayStart = "day-start"
    /// The calendar day rolled over.
    case dayEnd = "day-end"
    /// The Mac woke from sleep.
    case macWake = "mac-wake"
    /// The app launched — the gap behind is unwatched time.
    case launchCatchUp = "launch-catch-up"
    /// External power appeared or vanished.
    case powerChange = "power-change"
    /// A sustained app switch (brief flips never become Events).
    case appSwitch = "app-switch"
}

/// The queue's state machine — the wake table's proven shape (fired-but-
/// unconsumed re-presents): `pending` → `presented` (handed to a turn) →
/// `consumed` (that turn completed). A crash between the last two leaves the
/// recovery set.
nonisolated enum CompanionEventState: String, Codable, Sendable {
    case pending
    case presented
    case consumed
}

nonisolated struct CompanionEvent: Identifiable, Equatable, Sendable {
    let id: UUID
    let kind: CompanionEventKind
    /// One announceable line — what the turn's opening renders (#371).
    let content: String
    /// Kind-shaped JSON detail (an app-session span, a power verdict); nil
    /// when the content line is the whole fact.
    let payload: String?
    let occurredAt: Date
    var state: CompanionEventState
    /// Total order, assigned by the store at admission; nil before it.
    var seq: Int64?
    var admittedAt: Date?
    var presentedAt: Date?
    var consumedAt: Date?
    /// The turn that consumed it (#371 wires this).
    var turnID: UUID?

    /// A producer's perception: the five facts a producer owns. Everything
    /// else — state, seq, the admission stamp — is the store's to assign, so
    /// this init doesn't offer them.
    init(
        id: UUID = UUID(),
        kind: CompanionEventKind,
        content: String,
        payload: String? = nil,
        occurredAt: Date = Date()
    ) {
        self.id = id
        self.kind = kind
        self.content = content
        self.payload = payload
        self.occurredAt = occurredAt
        self.state = .pending
        self.seq = nil
        self.admittedAt = nil
        self.presentedAt = nil
        self.consumedAt = nil
        self.turnID = nil
    }

    /// The full row — only the store's decode constructs this shape.
    init(
        id: UUID, kind: CompanionEventKind, content: String, payload: String?,
        occurredAt: Date, state: CompanionEventState, seq: Int64?, admittedAt: Date?,
        presentedAt: Date?, consumedAt: Date?, turnID: UUID?
    ) {
        self.id = id
        self.kind = kind
        self.content = content
        self.payload = payload
        self.occurredAt = occurredAt
        self.state = state
        self.seq = seq
        self.admittedAt = admittedAt
        self.presentedAt = presentedAt
        self.consumedAt = consumedAt
        self.turnID = turnID
    }

    /// Exactly-once for once-per-occasion perceptions: the same occasion
    /// (e.g. `day-end:2026-07-18`) always mints the same id, so a producer
    /// firing twice — a repeated notification, a re-arm — collapses to one
    /// Event at admission instead of needing its own dedupe state.
    static func deterministicID(_ occasion: String) -> UUID {
        SHA256.hash(data: Data(occasion.utf8))
            .withUnsafeBytes { UUID(uuid: $0.loadUnaligned(as: uuid_t.self)) }
    }

    /// Kind-shaped payload JSON — the one encoder every producer shares, so
    /// no door hand-rolls (and mis-escapes) its own literal.
    static func payloadJSON(_ value: some Encodable) -> String? {
        (try? JSONEncoder().encode(value)).flatMap { String(data: $0, encoding: .utf8) }
    }
}

/// The drained batch as the turn's opening sees it (#371): everything that
/// reached the entity since its last turn, in total order — the fold's
/// `events` argument, rendered.
nonisolated enum CompanionEventBatch {

    static func render(_ events: [CompanionEvent], now: Date = Date()) -> String {
        guard !events.isEmpty else { return "" }
        let lines = events.enumerated().map { index, event -> String in
            let when = event.occurredAt.formatted(date: .omitted, time: .shortened)
            return "\(index + 1). [\(event.kind.rawValue)] \(when) — \(event.content)"
        }
        return """
            <events>
            Everything that reached you since your last turn, in order:
            \(lines.joined(separator: "\n"))
            </events>
            """
    }
}
