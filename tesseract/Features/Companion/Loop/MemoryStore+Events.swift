//
//  MemoryStore+Events.swift
//  tesseract
//
//  The Event queue's methods (ADR-0046, #368) — the fold's math, and nothing
//  else. Exactly-once: `id` is UNIQUE and admission is INSERT OR IGNORE, so a
//  producer firing twice collapses to one row. Total order: `seq` is assigned
//  by the database at admission and every read orders by it. Batch drain:
//  one call returns everything pending and marks it presented in the same
//  transaction. The consume/re-present pair carries the wake table's proven
//  correctness invariant: an Event is consumed only by a completed turn;
//  a crash mid-turn leaves the presented-but-unconsumed recovery set.
//
//  Same database as memory, tracking, and wakes — one backup, one inspection
//  story. Nothing here decides anything: the clock that grants turns over
//  pending Events is #371's.
//

import Foundation

extension MemoryStore {

    /// Admit one Event, exactly once. Returns false when the id was already
    /// admitted — the caller can tell a fresh perception from a collapsed
    /// duplicate (deterministic ids make repeat occasions duplicates by
    /// construction).
    @discardableResult
    func admitEvent(_ event: CompanionEvent) throws -> Bool {
        let stmt = try db.prepare(
            """
            INSERT OR IGNORE INTO events
                (id, kind, content, payload, occurredAt, admittedAt, state)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'pending')
            """)
        stmt.bind(1, event.id.uuidString)
            .bind(2, event.kind.rawValue)
            .bind(3, event.content)
            .bind(4, event.payload)
            .bind(5, event.occurredAt.timeIntervalSince1970)
            .bind(6, Date().timeIntervalSince1970)
        try stmt.run()
        return db.changes > 0
    }

    /// Everything pending, in total order — the evaluator's future read
    /// (#371) and the record's view of what has accumulated.
    func pendingEvents() throws -> [CompanionEvent] {
        let stmt = try db.prepare(
            "\(Self.eventSelect) WHERE state = 'pending' ORDER BY seq")
        return try Self.decodeEvents(stmt)
    }

    /// Drain: return everything pending, ordered, and mark it presented — one
    /// transaction, so a burst admitted before this call coalesces into one
    /// batch and a second drain returns nothing. Presented is not consumed:
    /// the batch stays in the recovery set until a completed turn calls
    /// `consumeEvents`.
    func drainPendingEvents(at now: Date = Date()) throws -> [CompanionEvent] {
        try db.transaction {
            let pending = try pendingEvents()
            guard !pending.isEmpty else { return [] }
            let stmt = try db.prepare(
                "UPDATE events SET state = 'presented', presentedAt = ?1 "
                    + "WHERE state = 'pending'")
            stmt.bind(1, now.timeIntervalSince1970)
            try stmt.run()
            return pending.map { event in
                var presented = event
                presented.state = .presented
                presented.presentedAt = now
                return presented
            }
        }
    }

    /// A completed turn consumed its batch — the only way an Event leaves the
    /// recovery set (the wake invariant, ADR-0040 §13). One transaction, one
    /// prepared statement (the `markSeen` batch idiom): all-or-nothing, so a
    /// crash mid-batch cannot leave it half-consumed.
    func consumeEvents(ids: [UUID], turnID: UUID, at now: Date = Date()) throws {
        guard !ids.isEmpty else { return }
        try db.transaction {
            let stmt = try db.prepare(
                "UPDATE events SET state = 'consumed', consumedAt = ?2, turnID = ?3 "
                    + "WHERE id = ?1 AND state = 'presented'")
            for id in ids {
                stmt.bind(1, id.uuidString)
                    .bind(2, now.timeIntervalSince1970)
                    .bind(3, turnID.uuidString)
                try stmt.run()
                stmt.reset()
            }
        }
    }

    /// Presented-but-unconsumed — a crash between drain and turn completion
    /// leaves these; launch recovery re-presents them. (`state = 'presented'`
    /// alone is the predicate: `consumeEvents` flips state and stamps
    /// `consumedAt` in the same UPDATE, so a presented row is unconsumed by
    /// construction.)
    func unconsumedPresentedEvents() throws -> [CompanionEvent] {
        let stmt = try db.prepare(
            "\(Self.eventSelect) WHERE state = 'presented' ORDER BY seq")
        return try Self.decodeEvents(stmt)
    }

    /// Put a crashed batch back in the queue: pending again, order untouched
    /// (`seq` never changes), the failed presentation erased.
    func representEvents(ids: [UUID]) throws {
        guard !ids.isEmpty else { return }
        try db.transaction {
            let stmt = try db.prepare(
                "UPDATE events SET state = 'pending', presentedAt = NULL "
                    + "WHERE id = ?1 AND state = 'presented'")
            for id in ids {
                stmt.bind(1, id.uuidString)
                try stmt.run()
                stmt.reset()
            }
        }
    }

    // MARK: - Decoding

    private static let eventSelect = """
        SELECT seq, id, kind, content, payload, occurredAt, admittedAt,
               state, presentedAt, consumedAt, turnID
        FROM events
        """

    private nonisolated static func decodeEvents(_ stmt: SQLiteDatabase.Statement) throws
        -> [CompanionEvent]
    {
        var out: [CompanionEvent] = []
        while try stmt.step() { out.append(decodeEvent(stmt)) }
        return out
    }

    private nonisolated static func decodeEvent(_ stmt: SQLiteDatabase.Statement)
        -> CompanionEvent
    {
        CompanionEvent(
            id: stmt.string(1).flatMap(UUID.init(uuidString:)) ?? UUID(),
            kind: CompanionEventKind(rawValue: stmt.string(2) ?? "") ?? .launchCatchUp,
            content: stmt.string(3) ?? "",
            payload: stmt.string(4),
            occurredAt: Date(timeIntervalSince1970: stmt.double(5)),
            state: CompanionEventState(rawValue: stmt.string(7) ?? "") ?? .pending,
            seq: Int64(stmt.int(0)),
            admittedAt: Date(timeIntervalSince1970: stmt.double(6)),
            presentedAt: stmt.isNull(8) ? nil : Date(timeIntervalSince1970: stmt.double(8)),
            consumedAt: stmt.isNull(9) ? nil : Date(timeIntervalSince1970: stmt.double(9)),
            turnID: stmt.string(10).flatMap(UUID.init(uuidString:))
        )
    }
}
