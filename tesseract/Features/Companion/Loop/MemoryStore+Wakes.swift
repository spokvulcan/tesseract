//
//  MemoryStore+Wakes.swift
//  tesseract
//
//  The wake table's methods (ADR-0040). Same database as memory and tracking
//  — a promise's provenance can FK to the conversation that booked it, and
//  one backup carries the whole record of commitments.
//

import Foundation

extension MemoryStore {

    func upsertWake(_ wake: CompanionWake) throws {
        let stmt = try db.prepare(
            """
            INSERT INTO wakes
                (id, content, due, class, state, summonsGrant, conversationID,
                 createdAt, updatedAt, firedAt, consumedAt, heardAt)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
            ON CONFLICT(id) DO UPDATE SET
                content=excluded.content, due=excluded.due, state=excluded.state,
                summonsGrant=excluded.summonsGrant, updatedAt=excluded.updatedAt,
                firedAt=excluded.firedAt, consumedAt=excluded.consumedAt,
                heardAt=excluded.heardAt
            """)
        stmt.bind(1, wake.id.uuidString)
            .bind(2, wake.content)
            .bind(3, wake.due.timeIntervalSince1970)
            .bind(4, wake.wakeClass.rawValue)
            .bind(5, wake.state.rawValue)
            .bind(6, wake.summonsGrant ? 1 : 0)
            .bind(7, wake.conversationID?.uuidString)
            .bind(8, wake.createdAt.timeIntervalSince1970)
            .bind(9, Date().timeIntervalSince1970)
            .bind(10, wake.firedAt?.timeIntervalSince1970)
            .bind(11, wake.consumedAt?.timeIntervalSince1970)
            .bind(12, wake.heardAt?.timeIntervalSince1970)
        try stmt.run()
    }

    func wake(id: UUID) throws -> CompanionWake? {
        let stmt = try db.prepare(
            "\(Self.wakeSelect) WHERE id = ?1")
        stmt.bind(1, id.uuidString)
        guard try stmt.step() else { return nil }
        return Self.decodeWake(stmt)
    }

    /// Booked wakes whose due time has arrived — the evaluator's read.
    func dueWakes(asOf now: Date) throws -> [CompanionWake] {
        let stmt = try db.prepare(
            "\(Self.wakeSelect) WHERE state = 'booked' AND due <= ?1 ORDER BY due")
        stmt.bind(1, now.timeIntervalSince1970)
        return try Self.decodeWakes(stmt)
    }

    /// Fired-but-never-consumed wakes — a crash mid-turn leaves these; the
    /// correctness invariant says they re-present.
    func unconsumedFiredWakes() throws -> [CompanionWake] {
        let stmt = try db.prepare(
            "\(Self.wakeSelect) WHERE state = 'fired' AND consumedAt IS NULL ORDER BY due")
        return try Self.decodeWakes(stmt)
    }

    /// What's ahead — the situation briefing shows the entity its own booked
    /// future.
    func upcomingWakes(after now: Date, limit: Int = 10) throws -> [CompanionWake] {
        let stmt = try db.prepare(
            "\(Self.wakeSelect) WHERE state = 'booked' AND due > ?1 ORDER BY due LIMIT ?2")
        stmt.bind(1, now.timeIntervalSince1970).bind(2, limit)
        return try Self.decodeWakes(stmt)
    }

    /// Promise-class wakes landing on a given local day — the visible-budget
    /// check reads this count. Keyed on `due` (the day the touchpoint reaches
    /// the owner), not the booking day: a promise booked today for Thursday
    /// draws on Thursday's budget. Delivered and engaged promises still count
    /// — the budget is what the day carries, not what remains booked. Only
    /// `dropped` (the defect state) is excluded.
    func promisesBooked(onDay dayKey: String, calendar: Calendar = .current) throws -> Int {
        guard let dayStart = TrackingDay.startOfDay(forKey: dayKey, calendar: calendar),
            let dayEnd = calendar.date(byAdding: .day, value: 1, to: dayStart)
        else { return 0 }
        let stmt = try db.prepare(
            "SELECT COUNT(*) FROM wakes WHERE class = 'promise' AND state != 'dropped' "
                + "AND due >= ?1 AND due < ?2")
        stmt.bind(1, dayStart.timeIntervalSince1970).bind(2, dayEnd.timeIntervalSince1970)
        guard try stmt.step() else { return 0 }
        return stmt.int(0)
    }

    // MARK: - Resurfacing (#309)

    /// Delivered promise-class wakes he never reacted to — the resurfacing
    /// candidates a firing beat reads.
    func unheardDeliveredPromises() throws -> [CompanionWake] {
        let stmt = try db.prepare(
            "\(Self.wakeSelect) WHERE state = 'delivered' AND class = 'promise' "
                + "AND heardAt IS NULL ORDER BY due")
        return try Self.decodeWakes(stmt)
    }

    /// What the previous beat resurfaced — the next beat's kill-pass set.
    func resurfacedWakes() throws -> [CompanionWake] {
        let stmt = try db.prepare(
            "\(Self.wakeSelect) WHERE state = 'resurfaced' ORDER BY due")
        return try Self.decodeWakes(stmt)
    }

    /// First reaction wins; later reactions never move the stamp.
    func stampWakeHeard(id: UUID, at: Date) throws {
        let stmt = try db.prepare(
            "UPDATE wakes SET heardAt = ?2, updatedAt = ?2 WHERE id = ?1 AND heardAt IS NULL")
        stmt.bind(1, id.uuidString).bind(2, at.timeIntervalSince1970)
        try stmt.run()
    }

    /// The owner engaging any companion banner is proof the beat that carried
    /// the resurfacing reached him — spare everything it resurfaced.
    func stampResurfacedHeard(at: Date) throws {
        let stmt = try db.prepare(
            "UPDATE wakes SET heardAt = ?1, updatedAt = ?1 "
                + "WHERE state = 'resurfaced' AND heardAt IS NULL")
        stmt.bind(1, at.timeIntervalSince1970)
        try stmt.run()
    }

    // MARK: - Loop day state

    func loopDayState(_ date: String) throws -> CompanionLoopDayState {
        let stmt = try db.prepare("SELECT state FROM loop_days WHERE date = ?1")
        stmt.bind(1, date)
        guard try stmt.step(), let json = stmt.string(0), let data = json.data(using: .utf8),
            let state = try? JSONDecoder().decode(CompanionLoopDayState.self, from: data)
        else { return CompanionLoopDayState() }
        return state
    }

    func setLoopDayState(_ date: String, _ state: CompanionLoopDayState) throws {
        let json =
            (try? JSONEncoder().encode(state)).flatMap { String(data: $0, encoding: .utf8) }
            ?? "{}"
        let stmt = try db.prepare(
            """
            INSERT INTO loop_days (date, state, updatedAt) VALUES (?1, ?2, ?3)
            ON CONFLICT(date) DO UPDATE SET state=excluded.state, updatedAt=excluded.updatedAt
            """)
        stmt.bind(1, date).bind(2, json).bind(3, Date().timeIntervalSince1970)
        try stmt.run()
    }

    // MARK: - Decoding

    private static let wakeSelect = """
        SELECT id, content, due, class, state, summonsGrant, conversationID,
               createdAt, updatedAt, firedAt, consumedAt, heardAt
        FROM wakes
        """

    private nonisolated static func decodeWakes(_ stmt: SQLiteDatabase.Statement) throws
        -> [CompanionWake]
    {
        var out: [CompanionWake] = []
        while try stmt.step() { out.append(decodeWake(stmt)) }
        return out
    }

    private nonisolated static func decodeWake(_ stmt: SQLiteDatabase.Statement) -> CompanionWake {
        CompanionWake(
            id: stmt.string(0).flatMap(UUID.init(uuidString:)) ?? UUID(),
            content: stmt.string(1) ?? "",
            due: Date(timeIntervalSince1970: stmt.double(2)),
            wakeClass: CompanionWakeClass(rawValue: stmt.string(3) ?? "") ?? .promise,
            state: CompanionWakeState(rawValue: stmt.string(4) ?? "") ?? .booked,
            summonsGrant: stmt.int(5) != 0,
            conversationID: stmt.string(6).flatMap(UUID.init(uuidString:)),
            createdAt: Date(timeIntervalSince1970: stmt.double(7)),
            updatedAt: Date(timeIntervalSince1970: stmt.double(8)),
            firedAt: stmt.isNull(9) ? nil : Date(timeIntervalSince1970: stmt.double(9)),
            consumedAt: stmt.isNull(10) ? nil : Date(timeIntervalSince1970: stmt.double(10)),
            heardAt: stmt.isNull(11) ? nil : Date(timeIntervalSince1970: stmt.double(11))
        )
    }
}
