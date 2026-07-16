//
//  MemoryStore+Tracking.swift
//  tesseract
//
//  The tracking tables' methods (#308) — Companion domain code over memory's
//  connection, in memory's actor. One database on purpose: a sleep pass can
//  read observations and mint a belief in one transaction; one backup story.
//
//  Observations are append-only: there is deliberately no update or delete —
//  a correction is a newer row, and recency wins at read.
//

import Foundation

extension MemoryStore {

    // MARK: - Days

    /// Insert-or-replace by day key. The chain and support ride as JSON — the
    /// morning read is one PK lookup, and the schema stays terminal-readable.
    func upsertDay(_ day: DayRecord) throws {
        let chainJSON =
            (try? JSONEncoder().encode(day.chain))
            .flatMap { String(data: $0, encoding: .utf8) } ?? "[]"
        let supportJSON =
            (try? JSONEncoder().encode(day.support))
            .flatMap { String(data: $0, encoding: .utf8) } ?? "[]"
        let stmt = try db.prepare(
            """
            INSERT INTO days (date, seed, chain, support, closedAt, createdAt, updatedAt)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
            ON CONFLICT(date) DO UPDATE SET
                seed=excluded.seed, chain=excluded.chain, support=excluded.support,
                closedAt=excluded.closedAt, updatedAt=excluded.updatedAt
            """)
        stmt.bind(1, day.date)
            .bind(2, day.seed)
            .bind(3, chainJSON)
            .bind(4, supportJSON)
            .bind(5, day.closedAt?.timeIntervalSince1970)
            .bind(6, day.createdAt.timeIntervalSince1970)
            .bind(7, Date().timeIntervalSince1970)
        try stmt.run()
    }

    func day(_ date: String) throws -> DayRecord? {
        let stmt = try db.prepare(
            "SELECT date, seed, chain, support, closedAt, createdAt, updatedAt FROM days WHERE date = ?1"
        )
        stmt.bind(1, date)
        guard try stmt.step() else { return nil }
        return Self.decodeDay(stmt)
    }

    /// Most-recent-first. The morning turn's "yesterday" read and the weekly
    /// review's streak walk both come through here.
    func recentDays(limit: Int) throws -> [DayRecord] {
        let stmt = try db.prepare(
            """
            SELECT date, seed, chain, support, closedAt, createdAt, updatedAt
            FROM days ORDER BY date DESC LIMIT ?1
            """)
        stmt.bind(1, limit)
        var out: [DayRecord] = []
        while try stmt.step() { out.append(Self.decodeDay(stmt)) }
        return out
    }

    private nonisolated static func decodeDay(_ stmt: SQLiteDatabase.Statement) -> DayRecord {
        let chain =
            stmt.string(2).flatMap { $0.data(using: .utf8) }
            .flatMap { try? JSONDecoder().decode([ContractStep].self, from: $0) } ?? []
        let support =
            stmt.string(3).flatMap { $0.data(using: .utf8) }
            .flatMap { try? JSONDecoder().decode([String].self, from: $0) } ?? []
        return DayRecord(
            date: stmt.string(0) ?? "",
            seed: stmt.string(1),
            chain: chain,
            support: support,
            closedAt: stmt.isNull(4) ? nil : Date(timeIntervalSince1970: stmt.double(4)),
            createdAt: Date(timeIntervalSince1970: stmt.double(5)),
            updatedAt: Date(timeIntervalSince1970: stmt.double(6))
        )
    }

    // MARK: - Observations

    func appendObservation(_ observation: TrackingObservation) throws {
        let stmt = try db.prepare(
            """
            INSERT INTO observations (id, ts, domain, kind, value, source, stream, episodeRef)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
            """)
        stmt.bind(1, observation.id.uuidString)
            .bind(2, observation.ts.timeIntervalSince1970)
            .bind(3, observation.domain.rawValue)
            .bind(4, observation.kind)
            .bind(5, observation.value)
            .bind(6, observation.source.rawValue)
            .bind(7, observation.stream)
            .bind(8, observation.episodeRef?.uuidString)
        try stmt.run()
    }

    /// Newest-first. `kind`/`domain` nil = all; the rhythm's reads are
    /// WHERE-clauses, never ranked retrieval — that is the point of the grain.
    func observations(
        kind: String? = nil,
        domain: TrackingDomain? = nil,
        since: Date? = nil,
        limit: Int = 50
    ) throws -> [TrackingObservation] {
        let stmt = try db.prepare(
            """
            SELECT id, ts, domain, kind, value, source, stream, episodeRef FROM observations
            WHERE (?1 IS NULL OR kind = ?1)
              AND (?2 IS NULL OR domain = ?2)
              AND (?3 IS NULL OR ts >= ?3)
            ORDER BY ts DESC LIMIT ?4
            """)
        stmt.bind(1, kind)
            .bind(2, domain?.rawValue)
            .bind(3, since?.timeIntervalSince1970)
            .bind(4, limit)
        var out: [TrackingObservation] = []
        while try stmt.step() {
            out.append(
                TrackingObservation(
                    id: stmt.string(0).flatMap(UUID.init(uuidString:)) ?? UUID(),
                    ts: Date(timeIntervalSince1970: stmt.double(1)),
                    domain: TrackingDomain(rawValue: stmt.string(2) ?? "") ?? .work,
                    kind: stmt.string(3) ?? "",
                    value: stmt.string(4) ?? "",
                    source: ObservationSource(rawValue: stmt.string(5) ?? "") ?? .sensed,
                    stream: stmt.string(6),
                    episodeRef: stmt.string(7).flatMap(UUID.init(uuidString:))
                ))
        }
        return out
    }

    // MARK: - Work items

    func upsertWorkItem(_ item: WorkItemRecord) throws {
        let stmt = try db.prepare(
            """
            INSERT INTO work_items
                (id, title, stream, domain, cadence, status, due, episodeRef, createdAt, updatedAt)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
            ON CONFLICT(id) DO UPDATE SET
                title=excluded.title, stream=excluded.stream, domain=excluded.domain,
                cadence=excluded.cadence, status=excluded.status, due=excluded.due,
                updatedAt=excluded.updatedAt
            """)
        stmt.bind(1, item.id.uuidString)
            .bind(2, item.title)
            .bind(3, item.stream)
            .bind(4, item.domain.rawValue)
            .bind(5, item.cadence.rawValue)
            .bind(6, item.status.rawValue)
            .bind(7, item.due?.timeIntervalSince1970)
            .bind(8, item.episodeRef?.uuidString)
            .bind(9, item.createdAt.timeIntervalSince1970)
            .bind(10, Date().timeIntervalSince1970)
        try stmt.run()
    }

    func workItems(status: WorkItemStatus? = nil) throws -> [WorkItemRecord] {
        let stmt = try db.prepare(
            """
            SELECT id, title, stream, domain, cadence, status, due, episodeRef, createdAt, updatedAt
            FROM work_items WHERE (?1 IS NULL OR status = ?1) ORDER BY createdAt
            """)
        stmt.bind(1, status?.rawValue)
        var out: [WorkItemRecord] = []
        while try stmt.step() { out.append(Self.decodeWorkItem(stmt)) }
        return out
    }

    /// Fuzzy resolution for the tools: exact id first, then unique
    /// case-insensitive title substring among open items.
    func findWorkItem(idOrTitle: String) throws -> WorkItemRecord? {
        if let id = UUID(uuidString: idOrTitle) {
            let stmt = try db.prepare(
                """
                SELECT id, title, stream, domain, cadence, status, due, episodeRef, createdAt, updatedAt
                FROM work_items WHERE id = ?1
                """)
            stmt.bind(1, id.uuidString)
            guard try stmt.step() else { return nil }
            return Self.decodeWorkItem(stmt)
        }
        let open = try workItems(status: .open)
        let needle = idOrTitle.lowercased()
        let matches = open.filter { $0.title.lowercased().contains(needle) }
        return matches.count == 1 ? matches[0] : nil
    }

    private nonisolated static func decodeWorkItem(_ stmt: SQLiteDatabase.Statement)
        -> WorkItemRecord
    {
        WorkItemRecord(
            id: stmt.string(0).flatMap(UUID.init(uuidString:)) ?? UUID(),
            title: stmt.string(1) ?? "",
            stream: stmt.string(2),
            domain: TrackingDomain(rawValue: stmt.string(3) ?? "") ?? .work,
            cadence: WorkItemCadence(rawValue: stmt.string(4) ?? "") ?? .once,
            status: WorkItemStatus(rawValue: stmt.string(5) ?? "") ?? .open,
            due: stmt.isNull(6) ? nil : Date(timeIntervalSince1970: stmt.double(6)),
            episodeRef: stmt.string(7).flatMap(UUID.init(uuidString:)),
            createdAt: Date(timeIntervalSince1970: stmt.double(8)),
            updatedAt: Date(timeIntervalSince1970: stmt.double(9))
        )
    }
}
