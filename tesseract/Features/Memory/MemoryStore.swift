//
//  MemoryStore.swift
//  tesseract
//
//  The memory store (ADR-0035 §8, #319). An actor over one SQLite connection.
//
//  Layout:
//
//    episodes      — APPEND-ONLY. The immutable layer. Nothing in this file
//                    updates or deletes an episode except `deleteEverything`
//                    (the owner's reset); consolidation only ever stamps
//                    `consolidatedAt`. No path rewrites what was said, and
//                    that is the point.
//    memories      — the derived, first-person, mutable layer.
//    embeddings    — 1024-dim float32 BLOBs, for episodes and memories alike.
//    retrievals    — the lifecycle's sensor log: what was surfaced, for which
//                    turn, and (once sleep grades it) whether it helped.
//    journal       — every mutation sleep makes, so a bad consolidation is
//                    inspectable and revertable.
//    cue_affinity  — per-(cue, memory) suppression. Retrieved-and-ignored
//                    decrements this and nothing else.
//    memories_fts  — FTS5 mirror for BM25 keyword search.
//
//  Retrieval is hybrid: cosine over the embedding BLOBs (brute force, via
//  Accelerate — at personal scale this is sub-millisecond and needs no ANN
//  index, which also sidesteps the hubness pathology) fused with FTS5/BM25.
//

import Accelerate
import Foundation

/// The vector width of `Qwen3-Embedding-0.6B` (measured: 1024).
let memoryEmbeddingDimension = 1024

/// The two owners a side-table row can belong to. One switch, owned here —
/// instead of "memory"/"memories"/"memories_fts" strings threaded through
/// every signature with a re-derivation branch at each site.
nonisolated enum MemoryOwner: String, Sendable {
    case memory
    case episode

    var table: String { self == .memory ? "memories" : "episodes" }
    var fts: String { table + "_fts" }
}

actor MemoryStore {

    /// v2: the `meta` table — the embedding-scheme stamp lives there (#332).
    /// v3: the Companion's tracking + loop tables beside memory's (#308,
    /// ADR-0040) — same database on purpose: FK-able provenance, one backup
    /// and inspection story. Their methods live in `MemoryStore+Tracking.swift`.
    /// v4: the entity's standing-instructions versions (ADR-0040 §12).
    /// v5: `heardAt` on wakes — the resurfacing ladder (#309) must tell an
    /// ignored promise from a heard one; any owner reaction stamps it.
    /// v6: the Event queue (ADR-0046, #368) — the fold's perception substrate.
    static let schemaVersion = 6

    /// Internal, not private: the tracking extension (a separate file by
    /// design — Companion domain, memory's connection) prepares against it.
    let db: SQLiteDatabase
    let directory: URL

    /// `directory` is injectable — it is simultaneously the test seam and the
    /// only defence against the scheme's parallel twin test runners writing
    /// into each other's store.
    init(directory: URL) throws {
        self.directory = directory
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        self.db = try SQLiteDatabase(path: directory.appendingPathComponent("memory.sqlite"))
        try Self.migrate(db)
    }

    // MARK: - Schema

    /// `nonisolated` so the actor's `init` may run it: at that point nothing
    /// else can reach the connection, so there is nothing to protect it from.
    private nonisolated static func migrate(_ db: SQLiteDatabase) throws {
        guard db.userVersion < Self.schemaVersion else { return }

        // v5: stores that already carry the wakes table (v3/v4) gain `heardAt`
        // in place; fresh stores get it from the CREATE below. Upper-bounded:
        // a v5+ store already has the column, and re-running the ALTER on the
        // way to a later version would throw duplicate-column and take the
        // whole memory system down with it.
        if db.userVersion >= 3 && db.userVersion < 5 {
            try db.execute("ALTER TABLE wakes ADD COLUMN heardAt REAL")
        }

        try db.execute(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                id             TEXT PRIMARY KEY NOT NULL,
                source         TEXT NOT NULL,
                conversationID TEXT,
                occurredAt     REAL NOT NULL,
                text           TEXT NOT NULL,
                meta           TEXT NOT NULL DEFAULT '{}',
                consolidatedAt REAL
            );
            CREATE INDEX IF NOT EXISTS episodes_occurredAt ON episodes(occurredAt);
            CREATE INDEX IF NOT EXISTS episodes_unconsolidated
                ON episodes(consolidatedAt) WHERE consolidatedAt IS NULL;

            CREATE TABLE IF NOT EXISTS memories (
                id               TEXT PRIMARY KEY NOT NULL,
                text             TEXT NOT NULL,
                kind             TEXT NOT NULL,
                provenance       TEXT NOT NULL,
                specificity      TEXT NOT NULL,
                status           TEXT NOT NULL,
                tier             TEXT NOT NULL,
                bornAt           REAL NOT NULL,
                stability        REAL NOT NULL,
                storageStrength  REAL NOT NULL,
                difficulty       REAL NOT NULL,
                lastUsefulUseAt  REAL NOT NULL,
                usefulUseCount   INTEGER NOT NULL DEFAULT 0,
                lastSeenAt       REAL NOT NULL,
                seenCount        INTEGER NOT NULL DEFAULT 0,
                confirmations    INTEGER NOT NULL DEFAULT 0,
                supersededBy     TEXT,
                cueClusterID     TEXT
            );
            CREATE INDEX IF NOT EXISTS memories_tier ON memories(tier);
            CREATE INDEX IF NOT EXISTS memories_status ON memories(status);

            CREATE TABLE IF NOT EXISTS memory_sources (
                memoryID  TEXT NOT NULL,
                episodeID TEXT NOT NULL,
                PRIMARY KEY (memoryID, episodeID),
                FOREIGN KEY (memoryID) REFERENCES memories(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS embeddings (
                ownerID   TEXT PRIMARY KEY NOT NULL,
                ownerKind TEXT NOT NULL,   -- 'memory' | 'episode'
                vector    BLOB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS embeddings_kind ON embeddings(ownerKind);

            CREATE TABLE IF NOT EXISTS retrievals (
                id            TEXT PRIMARY KEY NOT NULL,
                memoryID      TEXT NOT NULL,
                episodeID     TEXT NOT NULL,
                retrievedAt   REAL NOT NULL,
                cue           TEXT NOT NULL,
                isExploration INTEGER NOT NULL DEFAULT 0,
                grade         TEXT
            );
            CREATE INDEX IF NOT EXISTS retrievals_ungraded
                ON retrievals(grade) WHERE grade IS NULL;
            CREATE INDEX IF NOT EXISTS retrievals_memory ON retrievals(memoryID);

            CREATE TABLE IF NOT EXISTS journal (
                id       TEXT PRIMARY KEY NOT NULL,
                at       REAL NOT NULL,
                mutation TEXT NOT NULL,
                memoryID TEXT NOT NULL,
                detail   TEXT NOT NULL,
                before   TEXT,
                after    TEXT
            );
            CREATE INDEX IF NOT EXISTS journal_at ON journal(at);

            CREATE TABLE IF NOT EXISTS cue_affinity (
                cue      TEXT NOT NULL,
                memoryID TEXT NOT NULL,
                affinity REAL NOT NULL DEFAULT 1.0,
                PRIMARY KEY (cue, memoryID)
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                text,
                content='memories',
                content_rowid='rowid'
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
                text,
                content='episodes',
                content_rowid='rowid'
            );

            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY NOT NULL,
                value TEXT NOT NULL
            );

            -- Companion tracking (#308): testimony/measurement/conclusion —
            -- these tables are the measurement grain. Dates: REAL Unix seconds
            -- (the store-wide convention); day keys: local 'yyyy-MM-dd' TEXT.

            CREATE TABLE IF NOT EXISTS days (
                date      TEXT PRIMARY KEY NOT NULL,  -- local 'yyyy-MM-dd'
                seed      TEXT,                       -- tomorrow's seed, written at close
                chain     TEXT NOT NULL DEFAULT '[]', -- JSON [ContractStep]
                support   TEXT NOT NULL DEFAULT '[]', -- JSON [String], <= 2 by convention
                closedAt  REAL,                       -- NULL = "we didn't close" flag
                createdAt REAL NOT NULL,
                updatedAt REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS observations (
                id         TEXT PRIMARY KEY NOT NULL,
                ts         REAL NOT NULL,
                domain     TEXT NOT NULL,             -- work | body | mind
                kind       TEXT NOT NULL,
                value      TEXT NOT NULL,
                source     TEXT NOT NULL,             -- sensed | elicited | imported
                stream     TEXT,
                episodeRef TEXT                       -- elicited: the utterance behind the fact
            );
            CREATE INDEX IF NOT EXISTS observations_ts ON observations(ts);
            CREATE INDEX IF NOT EXISTS observations_kind_ts ON observations(kind, ts);

            CREATE TABLE IF NOT EXISTS work_items (
                id         TEXT PRIMARY KEY NOT NULL,
                title      TEXT NOT NULL,
                stream     TEXT,
                domain     TEXT NOT NULL DEFAULT 'work',
                cadence    TEXT NOT NULL,             -- once | daily
                status     TEXT NOT NULL,             -- open | done | dropped
                due        REAL,
                episodeRef TEXT,
                createdAt  REAL NOT NULL,
                updatedAt  REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS work_items_status ON work_items(status);

            -- The Companion loop (ADR-0040): wakes are rows; a wake is consumed
            -- only by a completed turn.

            CREATE TABLE IF NOT EXISTS wakes (
                id             TEXT PRIMARY KEY NOT NULL,
                content        TEXT NOT NULL,          -- announceable in one line
                due            REAL NOT NULL,
                class          TEXT NOT NULL,          -- promise | rhythm | followup | resummons
                state          TEXT NOT NULL,          -- booked | fired | engaged | delivered | resurfaced | delivered_unheard | dropped
                summonsGrant   INTEGER NOT NULL DEFAULT 0,
                conversationID TEXT,                   -- the conversation that booked it
                createdAt      REAL NOT NULL,
                updatedAt      REAL NOT NULL,
                firedAt        REAL,
                consumedAt     REAL,                   -- set only by a completed turn
                heardAt        REAL                    -- his first reaction, any kind (#309)
            );
            CREATE INDEX IF NOT EXISTS wakes_state_due ON wakes(state, due);

            CREATE TABLE IF NOT EXISTS loop_days (
                date      TEXT PRIMARY KEY NOT NULL,   -- local 'yyyy-MM-dd'
                state     TEXT NOT NULL DEFAULT '{}',  -- JSON CompanionLoopDayState
                updatedAt REAL NOT NULL
            );

            -- The entity's standing instructions (ADR-0040 §12): append-only
            -- versions; the highest version is what every turn injects. The
            -- entity revises through its tool, the owner through the editor —
            -- both append, nothing is ever silently rewritten.

            CREATE TABLE IF NOT EXISTS companion_instructions (
                version   INTEGER PRIMARY KEY AUTOINCREMENT,
                text      TEXT NOT NULL,
                author    TEXT NOT NULL,               -- seed | entity | owner
                note      TEXT,                        -- why this revision
                createdAt REAL NOT NULL
            );

            -- The Event queue (ADR-0046, #368): every digital input becomes
            -- exactly one Event; `seq` is the total order; `id` is UNIQUE so
            -- admission is exactly-once (INSERT OR IGNORE). The state machine
            -- mirrors wakes: pending → presented → consumed; presented-but-
            -- unconsumed is the crash-recovery set.

            CREATE TABLE IF NOT EXISTS events (
                seq         INTEGER PRIMARY KEY AUTOINCREMENT,
                id          TEXT UNIQUE NOT NULL,
                kind        TEXT NOT NULL,
                content     TEXT NOT NULL,             -- announceable in one line
                payload     TEXT,                      -- kind-shaped JSON detail
                occurredAt  REAL NOT NULL,
                admittedAt  REAL NOT NULL,
                state       TEXT NOT NULL DEFAULT 'pending',
                presentedAt REAL,
                consumedAt  REAL,                      -- set only by a completed turn
                turnID      TEXT
            );
            CREATE INDEX IF NOT EXISTS events_state_seq ON events(state, seq);
            """
        )
        try db.setUserVersion(Self.schemaVersion)
        Log.memory.info("Memory store schema at version \(Self.schemaVersion)")
    }

    // MARK: - Episodes (append-only)

    /// Append an episode. There is deliberately no `updateEpisode`.
    /// **Dates are stored as Unix seconds, and the round-trip is lossy below the
    /// microsecond.** `Date` holds seconds since the 2001 reference date; writing
    /// `timeIntervalSince1970` adds ~1.78e9 to that and reading it back subtracts
    /// the same, which costs the low bits of the mantissa. So a `Date` that goes
    /// into this store and comes out is *equal to within ~100 ns, and not
    /// bit-identical* — never compare one with `==`.
    ///
    /// Kept anyway, deliberately: Unix seconds are what `sqlite3
    /// "SELECT datetime(occurredAt,'unixepoch')"` understands, and being able to
    /// read your own memory store from a terminal is worth more in a local-first
    /// app than a nanosecond nobody will ever observe. Nothing in the lifecycle
    /// works at a finer grain than a day.
    func append(_ episode: Episode, embedding: [Float]? = nil) throws {
        try db.transaction {
            let metaJSON =
                (try? JSONEncoder().encode(episode.meta)).flatMap {
                    String(data: $0, encoding: .utf8)
                }
                ?? "{}"
            let stmt = try db.prepare(
                """
                INSERT OR IGNORE INTO episodes (id, source, conversationID, occurredAt, text, meta)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                """)
            stmt.bind(1, episode.id.uuidString)
                .bind(2, episode.source.rawValue)
                .bind(3, episode.conversationID)
                .bind(4, episode.occurredAt.timeIntervalSince1970)
                .bind(5, episode.text)
                .bind(6, metaJSON)
            try stmt.run()

            // `INSERT OR IGNORE` skipped an existing row: skip its side tables
            // too. Re-inserting the FTS row for the same rowid would index it
            // twice, and `keywordScores` would then return the same id twice —
            // reachable once episode ids are the messages' own and a turn can
            // be re-captured.
            //
            // One field may still move: the attached reply. A turn that opens
            // with a tool call has no answer text at its first `turnEnd`, and
            // the real answer only exists at the last one — so a re-capture of
            // the same turn carrying a fuller reply updates `meta.reply` in
            // place. The testimony (text, source, time) never changes; the
            // reply is context riding alongside it, and the newest non-empty
            // one wins.
            guard db.changes > 0 else {
                if let reply = episode.meta["reply"], !reply.isEmpty,
                    let current = try self.episode(id: episode.id),
                    current.meta["reply"] != reply
                {
                    var meta = current.meta
                    meta["reply"] = reply
                    let updatedJSON =
                        (try? JSONEncoder().encode(meta)).flatMap {
                            String(data: $0, encoding: .utf8)
                        } ?? metaJSON
                    let update = try db.prepare("UPDATE episodes SET meta = ?2 WHERE id = ?1")
                    update.bind(1, episode.id.uuidString).bind(2, updatedJSON)
                    try update.run()
                }
                return
            }

            let fts = try db.prepare(
                """
                INSERT INTO episodes_fts (rowid, text)
                SELECT rowid, text FROM episodes WHERE id = ?1
                """)
            fts.bind(1, episode.id.uuidString)
            try fts.run()

            if let embedding {
                try writeEmbedding(ownerID: episode.id, owner: .episode, vector: embedding)
            }
        }
    }

    func episode(id: UUID) throws -> Episode? {
        let stmt = try db.prepare("SELECT \(episodeColumns) FROM episodes WHERE id = ?1")
        stmt.bind(1, id.uuidString)
        guard try stmt.step() else { return nil }
        return decodeEpisode(stmt)
    }

    /// Batch fetch — the retrieval scan collects ~60 candidate ids per turn,
    /// and one `IN` query beats sixty actor round-trips.
    func episodes(ids: [UUID]) throws -> [Episode] {
        guard !ids.isEmpty else { return [] }
        let placeholders = (1...ids.count).map { "?\($0)" }.joined(separator: ",")
        let stmt = try db.prepare(
            "SELECT \(episodeColumns) FROM episodes WHERE id IN (\(placeholders))")
        for (index, id) in ids.enumerated() {
            stmt.bind(Int32(index + 1), id.uuidString)
        }
        var out: [Episode] = []
        while try stmt.step() { out.append(decodeEpisode(stmt)) }
        return out
    }

    /// Episodes that consolidation has not yet processed. This is the sleep
    /// session's work queue.
    func unconsolidatedEpisodes(limit: Int = 500) throws -> [Episode] {
        let stmt = try db.prepare(
            """
            SELECT \(episodeColumns) FROM episodes
            WHERE consolidatedAt IS NULL
            ORDER BY occurredAt ASC LIMIT ?1
            """)
        stmt.bind(1, limit)
        var out: [Episode] = []
        while try stmt.step() { out.append(decodeEpisode(stmt)) }
        return out
    }

    func markConsolidated(_ episodeIDs: [UUID], at date: Date) throws {
        guard !episodeIDs.isEmpty else { return }
        try db.transaction {
            let stmt = try db.prepare("UPDATE episodes SET consolidatedAt = ?2 WHERE id = ?1")
            for id in episodeIDs {
                stmt.bind(1, id.uuidString).bind(2, date.timeIntervalSince1970)
                try stmt.run()
                stmt.reset()
            }
        }
    }

    func episodeCount() throws -> Int {
        let stmt = try db.prepare("SELECT COUNT(*) FROM episodes")
        guard try stmt.step() else { return 0 }
        return stmt.int(0)
    }

    // MARK: - Memories

    /// Write a memory — and, when given, its journal line — in one transaction,
    /// so a crash can never leave a mutation the journal knows nothing about.
    func upsert(_ memory: MemoryRecord, embedding: [Float]? = nil, journal: JournalEntry? = nil)
        throws
    {
        try db.transaction {
            try upsertInTransaction(memory, embedding: embedding)
            if let journal { try appendJournal(journal) }
        }
    }

    /// The row write itself — callers must already hold a transaction.
    private func upsertInTransaction(_ memory: MemoryRecord, embedding: [Float]?) throws {
        let stmt = try db.prepare(
            """
            INSERT INTO memories (
                id, text, kind, provenance, specificity, status, tier, bornAt,
                stability, storageStrength, difficulty, lastUsefulUseAt,
                usefulUseCount, lastSeenAt, seenCount, confirmations,
                supersededBy, cueClusterID
            ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18)
            ON CONFLICT(id) DO UPDATE SET
                text=excluded.text, kind=excluded.kind, provenance=excluded.provenance,
                specificity=excluded.specificity, status=excluded.status, tier=excluded.tier,
                stability=excluded.stability, storageStrength=excluded.storageStrength,
                difficulty=excluded.difficulty, lastUsefulUseAt=excluded.lastUsefulUseAt,
                usefulUseCount=excluded.usefulUseCount, lastSeenAt=excluded.lastSeenAt,
                seenCount=excluded.seenCount, confirmations=excluded.confirmations,
                supersededBy=excluded.supersededBy, cueClusterID=excluded.cueClusterID
            """)
        stmt.bind(1, memory.id.uuidString)
            .bind(2, memory.text)
            .bind(3, memory.kind.rawValue)
            .bind(4, memory.provenance.rawValue)
            .bind(5, memory.specificity.rawValue)
            .bind(6, memory.status.rawValue)
            .bind(7, memory.tier.rawValue)
            .bind(8, memory.bornAt.timeIntervalSince1970)
            .bind(9, memory.stability)
            .bind(10, memory.storageStrength)
            .bind(11, memory.difficulty)
            .bind(12, memory.lastUsefulUseAt.timeIntervalSince1970)
            .bind(13, memory.usefulUseCount)
            .bind(14, memory.lastSeenAt.timeIntervalSince1970)
            .bind(15, memory.seenCount)
            .bind(16, memory.confirmations)
            .bind(17, memory.supersededBy?.uuidString)
            .bind(18, memory.cueClusterID?.uuidString)
        try stmt.run()

        // Keep the FTS mirror in step (external-content table).
        let delFTS = try db.prepare(
            "INSERT INTO memories_fts(memories_fts, rowid, text) "
                + "SELECT 'delete', rowid, text FROM memories WHERE id = ?1")
        delFTS.bind(1, memory.id.uuidString)
        try? delFTS.run()
        let insFTS = try db.prepare(
            "INSERT INTO memories_fts(rowid, text) SELECT rowid, ?2 FROM memories WHERE id = ?1"
        )
        insFTS.bind(1, memory.id.uuidString).bind(2, memory.text)
        try insFTS.run()

        let src = try db.prepare(
            "INSERT OR IGNORE INTO memory_sources (memoryID, episodeID) VALUES (?1, ?2)")
        for episodeID in memory.sourceEpisodeIDs {
            src.bind(1, memory.id.uuidString).bind(2, episodeID.uuidString)
            try src.run()
            src.reset()
        }

        if let embedding {
            try writeEmbedding(ownerID: memory.id, owner: .memory, vector: embedding)
        }
    }

    func memory(id: UUID) throws -> MemoryRecord? {
        let stmt = try db.prepare("SELECT \(memoryColumns) FROM memories WHERE id = ?1")
        stmt.bind(1, id.uuidString)
        guard try stmt.step() else { return nil }
        return try decodeMemory(stmt)
    }

    func memories(tier: MemoryTier? = nil, status: MemoryStatus? = .live, limit: Int = 10_000)
        throws
        -> [MemoryRecord]
    {
        var sql = "SELECT \(memoryColumns) FROM memories WHERE 1=1"
        if tier != nil { sql += " AND tier = ?1" }
        if status != nil { sql += " AND status = ?2" }
        sql += " ORDER BY bornAt DESC LIMIT ?3"
        let stmt = try db.prepare(sql)
        stmt.bind(1, tier?.rawValue).bind(2, status?.rawValue).bind(3, limit)
        return try decodeMemories(stmt)
    }

    /// Every live memory. The retrieval scan reads this — at personal scale
    /// (thousands, not millions) a full scan plus brute-force cosine is
    /// sub-millisecond, and it buys us exactness and no index to corrupt.
    func allLiveMemories() throws -> [MemoryRecord] {
        let stmt = try db.prepare(
            "SELECT \(memoryColumns) FROM memories WHERE status != 'superseded'")
        return try decodeMemories(stmt)
    }

    /// Decode a scan's rows with **one** grouped sources query instead of one
    /// sub-query per row — the scan paths (retrieve per turn, sweep and
    /// reconcile per sleep item) were paying N statements for arrays most of
    /// them never read.
    private func decodeMemories(_ stmt: SQLiteDatabase.Statement) throws -> [MemoryRecord] {
        let sources = try sourceEpisodeIDsByMemory()
        var out: [MemoryRecord] = []
        while try stmt.step() { out.append(try decodeMemory(stmt, sources: sources)) }
        return out
    }

    private func sourceEpisodeIDsByMemory() throws -> [UUID: [UUID]] {
        let stmt = try db.prepare("SELECT memoryID, episodeID FROM memory_sources")
        var out: [UUID: [UUID]] = [:]
        while try stmt.step() {
            guard let m = stmt.string(0).flatMap(UUID.init(uuidString:)),
                let e = stmt.string(1).flatMap(UUID.init(uuidString:))
            else { continue }
            out[m, default: []].append(e)
        }
        return out
    }

    func memoryCount() throws -> Int {
        let stmt = try db.prepare("SELECT COUNT(*) FROM memories")
        guard try stmt.step() else { return 0 }
        return stmt.int(0)
    }

    func countsByTier() throws -> [MemoryTier: Int] {
        let stmt = try db.prepare(
            "SELECT tier, COUNT(*) FROM memories WHERE status != 'superseded' GROUP BY tier")
        var out: [MemoryTier: Int] = [:]
        while try stmt.step() {
            if let raw = stmt.string(0), let tier = MemoryTier(rawValue: raw) {
                out[tier] = stmt.int(1)
            }
        }
        return out
    }

    /// The owner's hand — the one true-deletion path in the system.
    func deleteMemory(id: UUID) throws {
        try db.transaction {
            let text = try memory(id: id)?.text
            let delFTS = try db.prepare(
                "INSERT INTO memories_fts(memories_fts, rowid, text) "
                    + "SELECT 'delete', rowid, text FROM memories WHERE id = ?1")
            delFTS.bind(1, id.uuidString)
            try? delFTS.run()

            for sql in [
                "DELETE FROM memories WHERE id = ?1",
                "DELETE FROM embeddings WHERE ownerID = ?1",
                "DELETE FROM memory_sources WHERE memoryID = ?1",
                "DELETE FROM cue_affinity WHERE memoryID = ?1",
                // Its retrieval events go too: the judge can never grade them,
                // and left behind they would sit in the ungraded queue forever.
                "DELETE FROM retrievals WHERE memoryID = ?1",
            ] {
                let stmt = try db.prepare(sql)
                stmt.bind(1, id.uuidString)
                try stmt.run()
            }
            try appendJournal(
                JournalEntry(
                    at: Date(), mutation: .deletedByOwner, memoryID: id,
                    detail: "Deleted by the owner in the Memory window.", before: text, after: nil))
        }
    }

    // MARK: - Targeted mutations
    //
    // Every read-modify-write below happens inside one actor call, against the
    // freshly-read row — never against a snapshot a caller has been holding
    // across `await`s. The full-row `upsert` is for *inserting* records; using
    // it to bump a counter lets a stale copy silently roll back whatever landed
    // in between (a seen-mark clobbering a grade's strength bump, a contest
    // resurrecting a superseded belief). These methods exist so no caller has
    // to make that mistake.

    /// Retrieved into context — diagnostic only, never the lifecycle.
    func markSeen(_ ids: [UUID], at now: Date) throws {
        guard !ids.isEmpty else { return }
        try db.transaction {
            let stmt = try db.prepare(
                "UPDATE memories SET seenCount = seenCount + 1, lastSeenAt = ?2 WHERE id = ?1")
            for id in ids {
                stmt.bind(1, id.uuidString).bind(2, now.timeIntervalSince1970)
                try stmt.run()
                stmt.reset()
            }
        }
    }

    /// A re-encounter with no surprise: `confirmations += 1`, journaled, in one
    /// transaction. The rewriter is never invoked — that absence is the design,
    /// and the journal line is how the owner gets to watch it happen.
    func confirm(id: UUID, at now: Date) throws -> MemoryRecord? {
        try db.transaction {
            let stmt = try db.prepare(
                "UPDATE memories SET confirmations = confirmations + 1 WHERE id = ?1")
            stmt.bind(1, id.uuidString)
            try stmt.run()
            guard let fresh = try memory(id: id) else { return nil }
            try appendJournal(
                JournalEntry(
                    at: now, mutation: .confirmed, memoryID: id,
                    detail: "Said again, in different words — confirmed, not rewritten "
                        + "(\(fresh.confirmations)×).",
                    after: fresh.text))
            return fresh
        }
    }

    /// Move a memory's tier and nothing else, with its journal line in the same
    /// transaction.
    func setTier(id: UUID, to tier: MemoryTier, journal: JournalEntry? = nil) throws {
        try db.transaction {
            let stmt = try db.prepare("UPDATE memories SET tier = ?2 WHERE id = ?1")
            stmt.bind(1, id.uuidString).bind(2, tier.rawValue)
            try stmt.run()
            if let journal { try appendJournal(journal) }
        }
    }

    /// The owner's veto. Guarded on `status = 'live'` so a contest clicked over
    /// a stale window snapshot cannot resurrect a belief sleep has since
    /// superseded. Returns the contested record, or nil if it was no longer
    /// live and nothing happened.
    ///
    /// `reason` is what he said when he rejected it. It rides in the journal
    /// line — the same line `latestContestNote` hands to sleep's re-read, so
    /// the correction is minted against his words, not just against the old
    /// evidence that produced the wrong belief in the first place.
    func contest(id: UUID, at now: Date, reason: String? = nil) throws -> MemoryRecord? {
        try db.transaction {
            guard let before = try memory(id: id), before.status == .live else { return nil }
            let stmt = try db.prepare(
                "UPDATE memories SET status = ?2 WHERE id = ?1 AND status = ?3")
            stmt.bind(1, id.uuidString)
                .bind(2, MemoryStatus.contested.rawValue)
                .bind(3, MemoryStatus.live.rawValue)
            try stmt.run()
            try appendJournal(
                JournalEntry(
                    at: now, mutation: .contested, memoryID: id,
                    detail: reason.map { "He rejected this: \($0)" }
                        ?? "The owner contested this. Queued for reconciliation in sleep.",
                    before: before.text))
            return try memory(id: id)
        }
    }

    /// The note left when a memory was contested — what he said when he
    /// rejected it. Sleep's re-read wants this next to the evidence: the
    /// source episodes alone are exactly what produced the wrong belief.
    func latestContestNote(memoryID: UUID) throws -> String? {
        let stmt = try db.prepare(
            """
            SELECT detail FROM journal
            WHERE memoryID = ?1 AND mutation = ?2
            ORDER BY at DESC LIMIT 1
            """)
        stmt.bind(1, memoryID.uuidString).bind(2, MemoryMutation.contested.rawValue)
        guard try stmt.step() else { return nil }
        return stmt.string(0)
    }

    /// Retire `id` in favour of a successor that already exists — the second
    /// and further targets when one observation contradicts several beliefs at
    /// once. Guarded like `supersede`: acts on the fresh row, does nothing if
    /// it was already superseded or is the successor itself.
    func markSuperseded(id: UUID, by successorID: UUID, at now: Date) throws -> MemoryRecord? {
        try db.transaction {
            guard id != successorID,
                let old = try memory(id: id), old.status != .superseded,
                let successor = try memory(id: successorID)
            else { return nil }
            let stmt = try db.prepare(
                "UPDATE memories SET status = ?2, supersededBy = ?3 WHERE id = ?1")
            stmt.bind(1, id.uuidString)
                .bind(2, MemoryStatus.superseded.rawValue)
                .bind(3, successorID.uuidString)
            try stmt.run()
            try appendJournal(
                JournalEntry(
                    at: now, mutation: .superseded, memoryID: id,
                    detail: "Contradicted by the same observation that replaced a sibling belief.",
                    before: old.text, after: successor.text))
            return try memory(id: id)
        }
    }

    /// Supersede `oldID` with `new` — both rows, the inheritance, and both
    /// journal lines in **one transaction**, so a crash can never leave the old
    /// belief retired while its successor was never born (ADR-0035 §8:
    /// "together or not at all"). Inheritance reads the *fresh* old row, not
    /// whatever snapshot the caller reconciled against.
    ///
    /// Returns the successor as written, or nil (nothing changed) when the old
    /// belief has vanished or was already superseded in the meantime.
    func supersede(
        oldID: UUID, with new: MemoryRecord, embedding: [Float]?,
        inheritStrength: Bool, at now: Date
    ) throws -> MemoryRecord? {
        try db.transaction {
            guard let old = try memory(id: oldID), old.status != .superseded else { return nil }

            let stmt = try db.prepare(
                "UPDATE memories SET status = ?2, supersededBy = ?3 WHERE id = ?1")
            stmt.bind(1, oldID.uuidString)
                .bind(2, MemoryStatus.superseded.rawValue)
                .bind(3, new.id.uuidString)
            try stmt.run()

            var successor = new
            if inheritStrength {
                // The world moved on; the successor is the belief's continuation
                // and keeps what it earned. `max` keeps storage strength monotone
                // across the succession.
                successor.storageStrength = max(successor.storageStrength, old.storageStrength)
                successor.confirmations = old.confirmations
            }
            try upsertInTransaction(successor, embedding: embedding)

            try appendJournal(
                JournalEntry(
                    at: now, mutation: .superseded, memoryID: old.id,
                    detail: old.status == .contested
                        ? "He said this was wrong. Corrected against what he actually said."
                        : "Replaced by a later observation.",
                    before: old.text, after: successor.text))
            try appendJournal(
                JournalEntry(
                    at: now, mutation: .added, memoryID: successor.id,
                    detail: "Learned this, replacing an older belief.", after: successor.text))
            return successor
        }
    }

    /// Grade one retrieval event and move the memory's lifecycle, atomically:
    /// the grade, the strength update (computed against the freshly-read row),
    /// and — for `.ignored` — the per-cue affinity decay commit together.
    /// Returns the updated memory, or nil when it no longer exists (the event
    /// is still marked graded so it leaves the queue).
    func grade(_ grade: UseGrade, event: RetrievalEvent, now: Date) throws -> MemoryRecord? {
        try db.transaction {
            let mark = try db.prepare("UPDATE retrievals SET grade = ?2 WHERE id = ?1")
            mark.bind(1, event.id.uuidString).bind(2, grade.rawValue)
            try mark.run()

            guard let fresh = try memory(id: event.memoryID) else { return nil }
            let updated = MemoryLifecycle.applyGrade(grade, to: fresh, now: now)
            let stmt = try db.prepare(
                """
                UPDATE memories SET
                    stability = ?2, storageStrength = ?3, difficulty = ?4,
                    lastUsefulUseAt = ?5, usefulUseCount = ?6
                WHERE id = ?1
                """)
            stmt.bind(1, updated.id.uuidString)
                .bind(2, updated.stability)
                .bind(3, updated.storageStrength)
                .bind(4, updated.difficulty)
                .bind(5, updated.lastUsefulUseAt.timeIntervalSince1970)
                .bind(6, updated.usefulUseCount)
            try stmt.run()

            if grade == .ignored {
                try decayCueAffinity(cue: event.cue, memoryID: event.memoryID)
            }
            return updated
        }
    }

    // MARK: - Embeddings

    /// The embedding scheme whose vectors this store holds. A store from
    /// before the stamp existed reports 1 — the pre-#332 era.
    func embeddingScheme() throws -> Int {
        let stmt = try db.prepare("SELECT value FROM meta WHERE key = 'embedding_scheme'")
        guard try stmt.step(), let value = stmt.string(0) else { return 1 }
        return Int(value) ?? 1
    }

    /// Wipe every vector — and the cue affinities, which were learned from
    /// ranking those vectors and are exactly as stale as they are. The scheme
    /// stamp is deliberately NOT touched here: `stampEmbeddingScheme` runs
    /// after the re-embed completes, so a crash mid-refill re-runs the whole
    /// job instead of leaving a store that claims vectors it does not have.
    func resetEmbeddings() throws {
        try db.transaction {
            try db.execute("DELETE FROM embeddings; DELETE FROM cue_affinity;")
        }
    }

    func stampEmbeddingScheme(_ scheme: Int) throws {
        let stmt = try db.prepare(
            """
            INSERT INTO meta (key, value) VALUES ('embedding_scheme', ?1)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """)
        stmt.bind(1, String(scheme))
        try stmt.run()
    }

    /// (id, text) for every row of `owner` — the re-embed worklist.
    func allTexts(of owner: MemoryOwner, limit: Int = 100_000) throws -> [(id: UUID, text: String)]
    {
        let stmt = try db.prepare("SELECT id, text FROM \(owner.table) LIMIT ?1")
        stmt.bind(1, limit)
        var out: [(id: UUID, text: String)] = []
        while try stmt.step() {
            guard let raw = stmt.string(0), let id = UUID(uuidString: raw),
                let text = stmt.string(1)
            else { continue }
            out.append((id: id, text: text))
        }
        return out
    }

    /// One transaction per batch: a re-embed that dies mid-batch leaves whole
    /// rows or no rows, never a torn vector.
    func setEmbeddings(_ rows: [(UUID, [Float])], of owner: MemoryOwner) throws {
        try db.transaction {
            for (id, vector) in rows {
                try writeEmbedding(ownerID: id, owner: owner, vector: vector)
            }
        }
    }

    private func writeEmbedding(ownerID: UUID, owner: MemoryOwner, vector: [Float]) throws {
        let stmt = try db.prepare(
            """
            INSERT INTO embeddings (ownerID, ownerKind, vector) VALUES (?1, ?2, ?3)
            ON CONFLICT(ownerID) DO UPDATE SET vector = excluded.vector, ownerKind = excluded.ownerKind
            """)
        stmt.bind(1, ownerID.uuidString).bind(2, owner.rawValue).bind(3, Self.data(from: vector))
        try stmt.run()
    }

    func embedding(for ownerID: UUID) throws -> [Float]? {
        let stmt = try db.prepare("SELECT vector FROM embeddings WHERE ownerID = ?1")
        stmt.bind(1, ownerID.uuidString)
        guard try stmt.step(), let blob = stmt.data(0) else { return nil }
        return Self.vector(from: blob)
    }

    /// All embeddings of one owner, as (id, vector). The retrieval scan's input.
    func embeddings(of owner: MemoryOwner) throws -> [(UUID, [Float])] {
        let stmt = try db.prepare(
            "SELECT ownerID, vector FROM embeddings WHERE ownerKind = ?1")
        stmt.bind(1, owner.rawValue)
        var out: [(UUID, [Float])] = []
        while try stmt.step() {
            guard let idString = stmt.string(0), let id = UUID(uuidString: idString),
                let blob = stmt.data(1)
            else { continue }
            out.append((id, Self.vector(from: blob)))
        }
        return out
    }

    // MARK: - Vector math

    /// float32 little-endian, packed. 1024 dims → 4 KiB per record.
    static func data(from vector: [Float]) -> Data {
        vector.withUnsafeBufferPointer { Data(buffer: $0) }
    }

    static func vector(from data: Data) -> [Float] {
        let count = data.count / MemoryLayout<Float>.size
        guard count > 0 else { return [] }
        return data.withUnsafeBytes { raw in
            Array(
                UnsafeBufferPointer(
                    start: raw.baseAddress!.assumingMemoryBound(to: Float.self),
                    count: count))
        }
    }

    /// Cosine similarity. Both sides are L2-normalized by the embedder, so
    /// this is a dot product — `vDSP` makes it a single vectorized pass.
    static func cosine(_ a: [Float], _ b: [Float]) -> Double {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        var dot: Float = 0
        vDSP_dotpr(a, 1, b, 1, &dot, vDSP_Length(a.count))
        return Double(dot)
    }

    // MARK: - Hybrid relevance

    /// Keyword relevance via FTS5/BM25 — a model-free signal that costs
    /// nothing and catches the proper nouns embeddings smear.
    /// Returns id → normalized score in [0, 1].
    func keywordScores(query: String, in owner: MemoryOwner, limit: Int = 50) throws
        -> [UUID: Double]
    {
        let sanitized = Self.ftsQuery(query)
        guard !sanitized.isEmpty else { return [:] }
        let stmt = try db.prepare(
            """
            SELECT c.id, bm25(\(owner.fts)) AS rank
            FROM \(owner.fts) f JOIN \(owner.table) c ON c.rowid = f.rowid
            WHERE \(owner.fts) MATCH ?1
            ORDER BY rank LIMIT ?2
            """)
        stmt.bind(1, sanitized).bind(2, limit)
        var raw: [(UUID, Double)] = []
        while try stmt.step() {
            guard let s = stmt.string(0), let id = UUID(uuidString: s) else { continue }
            // bm25() is negative, more-negative = better. Flip it.
            raw.append((id, -stmt.double(1)))
        }
        guard let best = raw.map(\.1).max(), best > 0 else { return [:] }
        // `uniquingKeysWith`, defensively: an FTS mirror that ever double-indexed
        // a row must degrade to a duplicate score, not a crash.
        return Dictionary(raw.map { ($0.0, $0.1 / best) }, uniquingKeysWith: max)
    }

    /// FTS5 treats a lot of punctuation as syntax. Quote every bare term so a
    /// user's apostrophe or hyphen cannot become a query error.
    static func ftsQuery(_ query: String) -> String {
        let terms =
            query
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { $0.count > 2 }
            .prefix(12)
        guard !terms.isEmpty else { return "" }
        return terms.map { "\"\($0)\"" }.joined(separator: " OR ")
    }

    // MARK: - Retrieval log (the lifecycle's sensor)

    func log(_ events: [RetrievalEvent]) throws {
        guard !events.isEmpty else { return }
        try db.transaction {
            let stmt = try db.prepare(
                """
                INSERT OR REPLACE INTO retrievals
                    (id, memoryID, episodeID, retrievedAt, cue, isExploration, grade)
                VALUES (?1,?2,?3,?4,?5,?6,?7)
                """)
            for e in events {
                stmt.bind(1, e.id.uuidString)
                    .bind(2, e.memoryID.uuidString)
                    .bind(3, e.episodeID.uuidString)
                    .bind(4, e.retrievedAt.timeIntervalSince1970)
                    .bind(5, e.cue)
                    .bind(6, e.isExploration ? 1 : 0)
                    .bind(7, e.grade?.rawValue)
                try stmt.run()
                stmt.reset()
            }
        }
    }

    /// The sleep judge's work queue.
    func ungradedRetrievals(limit: Int = 500) throws -> [RetrievalEvent] {
        let stmt = try db.prepare(
            """
            SELECT id, memoryID, episodeID, retrievedAt, cue, isExploration, grade
            FROM retrievals WHERE grade IS NULL ORDER BY retrievedAt ASC LIMIT ?1
            """)
        stmt.bind(1, limit)
        var out: [RetrievalEvent] = []
        while try stmt.step() {
            guard let id = UUID(uuidString: stmt.string(0) ?? ""),
                let memoryID = UUID(uuidString: stmt.string(1) ?? ""),
                let episodeID = UUID(uuidString: stmt.string(2) ?? "")
            else { continue }
            out.append(
                RetrievalEvent(
                    id: id, memoryID: memoryID, episodeID: episodeID,
                    retrievedAt: Date(timeIntervalSince1970: stmt.double(3)),
                    cue: stmt.string(4) ?? "", isExploration: stmt.int(5) == 1,
                    grade: stmt.string(6).flatMap(UseGrade.init(rawValue:))))
        }
        return out
    }

    /// How many events await the judge — the stats line's number, without
    /// decoding ten thousand rows to count them.
    func ungradedRetrievalCount() throws -> Int {
        let stmt = try db.prepare("SELECT COUNT(*) FROM retrievals WHERE grade IS NULL")
        guard try stmt.step() else { return 0 }
        return stmt.int(0)
    }

    func setGrade(_ grade: UseGrade, for retrievalID: UUID) throws {
        let stmt = try db.prepare("UPDATE retrievals SET grade = ?2 WHERE id = ?1")
        stmt.bind(1, retrievalID.uuidString).bind(2, grade.rawValue)
        try stmt.run()
    }

    /// How many *distinct days* each memory was usefully used on, in one
    /// grouped query — the sweep asks this for every live memory, and one
    /// round-trip per memory per night was the store's biggest query loop.
    /// The spacing effect: three uses inside one conversation are one massed
    /// episode, and promotion must not be fooled by them.
    func distinctUsefulDaysByMemory() throws -> [UUID: Int] {
        let stmt = try db.prepare(
            """
            SELECT memoryID, COUNT(DISTINCT CAST(retrievedAt / 86400 AS INTEGER))
            FROM retrievals WHERE grade IN ('decisive','used') GROUP BY memoryID
            """)
        var out: [UUID: Int] = [:]
        while try stmt.step() {
            guard let id = stmt.string(0).flatMap(UUID.init(uuidString:)) else { continue }
            out[id] = stmt.int(1)
        }
        return out
    }

    // MARK: - Cue affinity

    /// `.ignored` decrements this and nothing else. The memory is not at
    /// fault — the retriever surfaced it for the wrong cue.
    func decayCueAffinity(cue: String, memoryID: UUID, by factor: Double = 0.9) throws {
        let stmt = try db.prepare(
            """
            INSERT INTO cue_affinity (cue, memoryID, affinity) VALUES (?1, ?2, ?3)
            ON CONFLICT(cue, memoryID) DO UPDATE SET affinity = affinity * ?3
            """)
        stmt.bind(1, Self.cueKey(cue)).bind(2, memoryID.uuidString).bind(3, factor)
        try stmt.run()
    }

    func cueAffinities(cue: String) throws -> [UUID: Double] {
        let stmt = try db.prepare(
            "SELECT memoryID, affinity FROM cue_affinity WHERE cue = ?1")
        stmt.bind(1, Self.cueKey(cue))
        var out: [UUID: Double] = [:]
        while try stmt.step() {
            if let s = stmt.string(0), let id = UUID(uuidString: s) { out[id] = stmt.double(1) }
        }
        return out
    }

    /// Cues are bucketed, not verbatim — a per-exact-string affinity would
    /// never see the same key twice.
    static func cueKey(_ cue: String) -> String {
        cue.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { $0.count > 2 }
            .sorted()
            .prefix(6)
            .joined(separator: " ")
    }

    // MARK: - Journal

    func appendJournal(_ entry: JournalEntry) throws {
        let stmt = try db.prepare(
            """
            INSERT INTO journal (id, at, mutation, memoryID, detail, before, after)
            VALUES (?1,?2,?3,?4,?5,?6,?7)
            """)
        stmt.bind(1, entry.id.uuidString)
            .bind(2, entry.at.timeIntervalSince1970)
            .bind(3, entry.mutation.rawValue)
            .bind(4, entry.memoryID.uuidString)
            .bind(5, entry.detail)
            .bind(6, entry.before)
            .bind(7, entry.after)
        try stmt.run()
    }

    func journal(since: Date? = nil, limit: Int = 200) throws -> [JournalEntry] {
        let stmt = try db.prepare(
            """
            SELECT id, at, mutation, memoryID, detail, before, after FROM journal
            WHERE at >= ?1 ORDER BY at DESC LIMIT ?2
            """)
        stmt.bind(1, (since ?? .distantPast).timeIntervalSince1970).bind(2, limit)
        var out: [JournalEntry] = []
        while try stmt.step() {
            guard let id = UUID(uuidString: stmt.string(0) ?? ""),
                let memoryID = UUID(uuidString: stmt.string(3) ?? ""),
                let mutation = stmt.string(2).flatMap(MemoryMutation.init(rawValue:))
            else { continue }
            out.append(
                JournalEntry(
                    id: id, at: Date(timeIntervalSince1970: stmt.double(1)), mutation: mutation,
                    memoryID: memoryID, detail: stmt.string(4) ?? "",
                    before: stmt.string(5), after: stmt.string(6)))
        }
        return out
    }

    // MARK: - Maintenance

    func deleteEverything() throws {
        try db.execute(
            """
            DELETE FROM memories; DELETE FROM episodes; DELETE FROM embeddings;
            DELETE FROM retrievals; DELETE FROM journal; DELETE FROM cue_affinity;
            DELETE FROM memories_fts; DELETE FROM episodes_fts;
            """)
    }

    // MARK: - Decoding

    private let episodeColumns = "id, source, conversationID, occurredAt, text, meta"

    private func decodeEpisode(_ s: SQLiteDatabase.Statement) -> Episode {
        let meta: [String: String] =
            s.string(5).flatMap { $0.data(using: .utf8) }
            .flatMap { try? JSONDecoder().decode([String: String].self, from: $0) } ?? [:]
        return Episode(
            id: UUID(uuidString: s.string(0) ?? "") ?? UUID(),
            source: MemorySource(rawValue: s.string(1) ?? "") ?? .chat,
            conversationID: s.string(2),
            occurredAt: Date(timeIntervalSince1970: s.double(3)),
            text: s.string(4) ?? "",
            meta: meta)
    }

    private let memoryColumns = """
        id, text, kind, provenance, specificity, status, tier, bornAt, stability,
        storageStrength, difficulty, lastUsefulUseAt, usefulUseCount, lastSeenAt,
        seenCount, confirmations, supersededBy, cueClusterID
        """

    /// `sources` is the pre-grouped map on scan paths; nil falls back to the
    /// per-row sub-query, which is right for single-record fetches.
    private func decodeMemory(_ s: SQLiteDatabase.Statement, sources: [UUID: [UUID]]? = nil)
        throws -> MemoryRecord
    {
        let id = UUID(uuidString: s.string(0) ?? "") ?? UUID()
        return MemoryRecord(
            id: id,
            text: s.string(1) ?? "",
            kind: MemoryKind(rawValue: s.string(2) ?? "") ?? .belief,
            provenance: Provenance(rawValue: s.string(3) ?? "") ?? .inferred,
            specificity: Specificity(rawValue: s.string(4) ?? "") ?? .general,
            status: MemoryStatus(rawValue: s.string(5) ?? "") ?? .live,
            tier: MemoryTier(rawValue: s.string(6) ?? "") ?? .hot,
            sourceEpisodeIDs: try sources.map { $0[id] ?? [] } ?? sourceEpisodeIDs(memoryID: id),
            bornAt: Date(timeIntervalSince1970: s.double(7)),
            stability: s.double(8),
            storageStrength: s.double(9),
            difficulty: s.double(10),
            lastUsefulUseAt: Date(timeIntervalSince1970: s.double(11)),
            usefulUseCount: s.int(12),
            lastSeenAt: Date(timeIntervalSince1970: s.double(13)),
            seenCount: s.int(14),
            confirmations: s.int(15),
            supersededBy: s.string(16).flatMap(UUID.init(uuidString:)),
            cueClusterID: s.string(17).flatMap(UUID.init(uuidString:)))
    }

    private func sourceEpisodeIDs(memoryID: UUID) throws -> [UUID] {
        let stmt = try db.prepare(
            "SELECT episodeID FROM memory_sources WHERE memoryID = ?1")
        stmt.bind(1, memoryID.uuidString)
        var out: [UUID] = []
        while try stmt.step() {
            if let s = stmt.string(0), let id = UUID(uuidString: s) { out.append(id) }
        }
        return out
    }
}
