//
//  RecallIndex.swift
//  tesseract
//
//  Finding things in past conversations: "what did I say about X last week?"
//  A full-text index (SQLite FTS5) over every saved conversation — the
//  owner's words and Jarvis's replies, never moment cards — kept current by
//  file modification time, so a search re-reads only what changed. Derived
//  data under Application Support, rebuilt from the conversations whenever
//  it goes missing. Searched only when asked (`recall`); nothing from it is
//  ever put into a chat on its own.
//

import Foundation

nonisolated struct RecallHit: Sendable, Equatable {
    var conversationID: String
    var title: String
    var role: String
    var at: Date
    var snippet: String
    var score: Double
}

nonisolated enum RecallQuery {
    private static let stopwords: Set<String> = [
        "the", "a", "an", "and", "or", "to", "of", "in", "on", "at", "for", "with", "about", "what",
        "did", "i", "me", "my", "you", "we", "it", "is", "was", "were", "be", "say", "said", "tell",
        "last", "week", "day", "that", "this", "do", "does", "when", "where", "how", "who", "why",
    ]

    /// The words worth searching for, lowercased.
    static func terms(_ query: String) -> [String] {
        query.lowercased()
            .split { !$0.isLetter && !$0.isNumber }
            .map(String.init)
            .filter { $0.count > 1 && !stopwords.contains($0) }
    }

    /// An FTS5 match expression: any term, prefix-matched.
    static func match(_ query: String) -> String? {
        let terms = terms(query)
        guard !terms.isEmpty else { return nil }
        return terms.map { "\"\($0.replacingOccurrences(of: "\"", with: ""))\"*" }.joined(
            separator: " OR ")
    }
}

actor RecallIndex {

    private let conversationsDirectory: URL
    private let databaseURL: URL
    private var database: SQLiteDatabase?

    init(conversationsDirectory: URL, databaseURL: URL) {
        self.conversationsDirectory = conversationsDirectory
        self.databaseURL = databaseURL
    }

    /// The best matches for `query`, most relevant first, optionally only
    /// from the last `days` days.
    func search(_ query: String, days: Int? = nil, limit: Int = 30, now: Date = Date())
        -> [RecallHit]
    {
        guard let match = RecallQuery.match(query), let db = open() else { return [] }
        refresh(db)
        let since =
            days.map { now.addingTimeInterval(-TimeInterval($0) * 86_400).timeIntervalSince1970 }
            ?? 0
        do {
            let statement = try db.prepare(
                """
                SELECT conversation, title, role, at, snippet(chunks, 0, '', '', '…', 24), bm25(chunks)
                FROM chunks WHERE chunks MATCH ? AND CAST(at AS REAL) >= ?
                ORDER BY bm25(chunks) LIMIT ?
                """
            )
            .bind(1, match).bind(2, since).bind(3, limit)
            var hits: [RecallHit] = []
            while try statement.step() {
                hits.append(
                    RecallHit(
                        conversationID: statement.string(0) ?? "", title: statement.string(1) ?? "",
                        role: statement.string(2) ?? "",
                        at: Date(timeIntervalSince1970: statement.double(3)),
                        snippet: statement.string(4) ?? "", score: -statement.double(5)))
            }
            return hits
        } catch {
            Log.companion.error("Recall search failed: \(error)")
            return []
        }
    }

    // MARK: Index

    private func open() -> SQLiteDatabase? {
        if let database { return database }
        do {
            try FileManager.default.createDirectory(
                at: databaseURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            let db = try SQLiteDatabase(path: databaseURL)
            try db.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
                    text, conversation UNINDEXED, title UNINDEXED, role UNINDEXED, at UNINDEXED,
                    tokenize = 'unicode61 remove_diacritics 2');
                CREATE TABLE IF NOT EXISTS files (id TEXT PRIMARY KEY, modified REAL NOT NULL);
                """)
            database = db
            return db
        } catch {
            Log.companion.error("Recall index unavailable: \(error)")
            return nil
        }
    }

    /// Re-read conversations whose files changed; drop deleted ones.
    private func refresh(_ db: SQLiteDatabase) {
        let fm = FileManager.default
        guard
            let files = try? fm.contentsOfDirectory(
                at: conversationsDirectory,
                includingPropertiesForKeys: [.contentModificationDateKey])
        else { return }
        var known: [String: Double] = [:]
        if let statement = try? db.prepare("SELECT id, modified FROM files") {
            while (try? statement.step()) == true {
                known[statement.string(0) ?? ""] = statement.double(1)
            }
        }
        var present = Set<String>()
        for url in files where url.pathExtension == "json" && url.lastPathComponent != "index.json"
        {
            let id = url.deletingPathExtension().lastPathComponent
            present.insert(id)
            let modified =
                ((try? url.resourceValues(forKeys: [.contentModificationDateKey]))?
                .contentModificationDate ?? .distantPast).timeIntervalSince1970
            guard known[id] != modified else { continue }
            index(url, id: id, modified: modified, db: db)
        }
        for id in known.keys where !present.contains(id) {
            try? db.transaction {
                try db.prepare("DELETE FROM chunks WHERE conversation = ?").bind(1, id).run()
                try db.prepare("DELETE FROM files WHERE id = ?").bind(1, id).run()
            }
        }
    }

    private func index(_ url: URL, id: String, modified: Double, db: SQLiteDatabase) {
        let messages = Self.messages(in: url)
        try? db.transaction {
            try db.prepare("DELETE FROM chunks WHERE conversation = ?").bind(1, id).run()
            for message in messages {
                try db.prepare(
                    "INSERT INTO chunks (text, conversation, title, role, at) VALUES (?, ?, ?, ?, ?)"
                )
                .bind(1, message.text).bind(2, id).bind(3, message.title).bind(4, message.role)
                .bind(5, message.at.timeIntervalSince1970).run()
            }
            try db.prepare("INSERT OR REPLACE INTO files (id, modified) VALUES (?, ?)")
                .bind(1, id).bind(2, modified).run()
        }
    }

    struct Message: Equatable {
        var role: String
        var at: Date
        var text: String
        var title: String
    }

    /// The searchable messages of one conversation file: the owner's own
    /// words (wrappers stripped) and Jarvis's text replies. A moment's request
    /// and its card are skipped.
    static func messages(in url: URL) -> [Message] {
        guard let data = try? Data(contentsOf: url),
            let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let raw = root["messages"] as? [[String: Any]]
        else { return [] }
        let title = root["title"] as? String ?? ""
        var out: [Message] = []
        var skipReply = false
        for message in raw {
            guard let type = message["type"] as? String,
                let payload = message["payload"] as? [String: Any]
            else { continue }
            let at = Date(timeIntervalSinceReferenceDate: payload["timestamp"] as? Double ?? 0)
            switch type {
            case "user":
                if payload["turnOrigin"] as? String == "moment" {
                    skipReply = true
                    continue
                }
                skipReply = false
                if let content = payload["content"] as? String,
                    let spoken = MemorySpeech.spoken(content)
                {
                    out.append(Message(role: "owner", at: at, text: spoken, title: title))
                }
            case "assistant":
                if skipReply {
                    skipReply = false
                    continue
                }
                let parts = payload["content"] as? [[String: Any]] ?? []
                let text = parts.compactMap {
                    $0["type"] as? String == "text" ? $0["text"] as? String : nil
                }
                .joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
                if !text.isEmpty {
                    out.append(Message(role: "jarvis", at: at, text: text, title: title))
                }
            default:
                continue
            }
        }
        return out
    }
}
