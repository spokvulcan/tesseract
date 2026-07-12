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
//                    (the owner's reset) and `deleteEpisode` (the owner's hand
//                    in the Memory window). No consolidation path can reach
//                    them, and that is the point.
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

actor MemoryStore {

    static let schemaVersion = 1

    private let db: SQLiteDatabase
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

            let fts = try db.prepare(
                """
                INSERT INTO episodes_fts (rowid, text)
                SELECT rowid, text FROM episodes WHERE id = ?1
                """)
            fts.bind(1, episode.id.uuidString)
            try fts.run()

            if let embedding {
                try writeEmbedding(ownerID: episode.id, kind: "episode", vector: embedding)
            }
        }
    }

    func episode(id: UUID) throws -> Episode? {
        let stmt = try db.prepare("SELECT \(episodeColumns) FROM episodes WHERE id = ?1")
        stmt.bind(1, id.uuidString)
        guard try stmt.step() else { return nil }
        return decodeEpisode(stmt)
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

    func recentEpisodes(since: Date, limit: Int = 100) throws -> [Episode] {
        let stmt = try db.prepare(
            """
            SELECT \(episodeColumns) FROM episodes
            WHERE occurredAt >= ?1 ORDER BY occurredAt DESC LIMIT ?2
            """)
        stmt.bind(1, since.timeIntervalSince1970).bind(2, limit)
        var out: [Episode] = []
        while try stmt.step() { out.append(decodeEpisode(stmt)) }
        return out
    }

    func episodeCount() throws -> Int {
        let stmt = try db.prepare("SELECT COUNT(*) FROM episodes")
        guard try stmt.step() else { return 0 }
        return stmt.int(0)
    }

    // MARK: - Memories

    func upsert(_ memory: MemoryRecord, embedding: [Float]? = nil) throws {
        try db.transaction {
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
                try writeEmbedding(ownerID: memory.id, kind: "memory", vector: embedding)
            }
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
        var out: [MemoryRecord] = []
        while try stmt.step() { out.append(try decodeMemory(stmt)) }
        return out
    }

    /// Every live memory. The retrieval scan reads this — at personal scale
    /// (thousands, not millions) a full scan plus brute-force cosine is
    /// sub-millisecond, and it buys us exactness and no index to corrupt.
    func allLiveMemories() throws -> [MemoryRecord] {
        let stmt = try db.prepare(
            "SELECT \(memoryColumns) FROM memories WHERE status != 'superseded'")
        var out: [MemoryRecord] = []
        while try stmt.step() { out.append(try decodeMemory(stmt)) }
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

    // MARK: - Embeddings

    private func writeEmbedding(ownerID: UUID, kind: String, vector: [Float]) throws {
        let stmt = try db.prepare(
            """
            INSERT INTO embeddings (ownerID, ownerKind, vector) VALUES (?1, ?2, ?3)
            ON CONFLICT(ownerID) DO UPDATE SET vector = excluded.vector, ownerKind = excluded.ownerKind
            """)
        stmt.bind(1, ownerID.uuidString).bind(2, kind).bind(3, Self.data(from: vector))
        try stmt.run()
    }

    func setEmbedding(ownerID: UUID, kind: String, vector: [Float]) throws {
        try db.transaction { try writeEmbedding(ownerID: ownerID, kind: kind, vector: vector) }
    }

    func embedding(for ownerID: UUID) throws -> [Float]? {
        let stmt = try db.prepare("SELECT vector FROM embeddings WHERE ownerID = ?1")
        stmt.bind(1, ownerID.uuidString)
        guard try stmt.step(), let blob = stmt.data(0) else { return nil }
        return Self.vector(from: blob)
    }

    /// All embeddings of one kind, as (id, vector). The retrieval scan's input.
    func embeddings(kind: String) throws -> [(UUID, [Float])] {
        let stmt = try db.prepare(
            "SELECT ownerID, vector FROM embeddings WHERE ownerKind = ?1")
        stmt.bind(1, kind)
        var out: [(UUID, [Float])] = []
        while try stmt.step() {
            guard let idString = stmt.string(0), let id = UUID(uuidString: idString),
                let blob = stmt.data(1)
            else { continue }
            out.append((id, Self.vector(from: blob)))
        }
        return out
    }

    func idsMissingEmbeddings(kind: String) throws -> [UUID] {
        let table = kind == "memory" ? "memories" : "episodes"
        let stmt = try db.prepare(
            """
            SELECT t.id FROM \(table) t
            LEFT JOIN embeddings e ON e.ownerID = t.id
            WHERE e.ownerID IS NULL
            """)
        var out: [UUID] = []
        while try stmt.step() {
            if let s = stmt.string(0), let id = UUID(uuidString: s) { out.append(id) }
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
    func keywordScores(query: String, table: String, limit: Int = 50) throws -> [UUID: Double] {
        let sanitized = Self.ftsQuery(query)
        guard !sanitized.isEmpty else { return [:] }
        let content = table == "memories" ? "memories" : "episodes"
        let fts = table == "memories" ? "memories_fts" : "episodes_fts"
        let stmt = try db.prepare(
            """
            SELECT c.id, bm25(\(fts)) AS rank
            FROM \(fts) f JOIN \(content) c ON c.rowid = f.rowid
            WHERE \(fts) MATCH ?1
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
        return Dictionary(uniqueKeysWithValues: raw.map { ($0.0, $0.1 / best) })
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

    func setGrade(_ grade: UseGrade, for retrievalID: UUID) throws {
        let stmt = try db.prepare("UPDATE retrievals SET grade = ?2 WHERE id = ?1")
        stmt.bind(1, retrievalID.uuidString).bind(2, grade.rawValue)
        try stmt.run()
    }

    /// How many *distinct days* this memory was usefully used on. The spacing
    /// effect: three uses inside one conversation are one massed episode, and
    /// promotion must not be fooled by them.
    func distinctUsefulDays(memoryID: UUID) throws -> Int {
        let stmt = try db.prepare(
            """
            SELECT COUNT(DISTINCT CAST(retrievedAt / 86400 AS INTEGER))
            FROM retrievals WHERE memoryID = ?1 AND grade IN ('decisive','used')
            """)
        stmt.bind(1, memoryID.uuidString)
        guard try stmt.step() else { return 0 }
        return stmt.int(0)
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

    private func decodeMemory(_ s: SQLiteDatabase.Statement) throws -> MemoryRecord {
        let id = UUID(uuidString: s.string(0) ?? "") ?? UUID()
        return MemoryRecord(
            id: id,
            text: s.string(1) ?? "",
            kind: MemoryKind(rawValue: s.string(2) ?? "") ?? .belief,
            provenance: Provenance(rawValue: s.string(3) ?? "") ?? .inferred,
            specificity: Specificity(rawValue: s.string(4) ?? "") ?? .general,
            status: MemoryStatus(rawValue: s.string(5) ?? "") ?? .live,
            tier: MemoryTier(rawValue: s.string(6) ?? "") ?? .hot,
            sourceEpisodeIDs: try sourceEpisodeIDs(memoryID: id),
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
