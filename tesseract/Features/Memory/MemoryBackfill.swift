//
//  MemoryBackfill.swift
//  tesseract
//
//  The cold start (ADR-0035 §6, map #314 owner call 1).
//
//  A memory system that begins empty is, on its first morning, indistinguishable
//  from no memory system at all — and the owner has been talking to this app for
//  months. Sixty-six conversations of him are already on disk. Not reading them
//  would be a strange kind of amnesia: the app *has* the past, it just never
//  looked at it.
//
//  So on first run the store is seeded from what already exists:
//
//    1. `memories.md` — the six hand-written facts of the old regime. These were
//       written by the owner, so they enter as **STATED**, and they are the only
//       memories in the system that never had an episode: their provenance is
//       the file itself. The file is then *renamed*, not deleted — it is the one
//       artifact here that has no other copy.
//
//    2. The conversation corpus → **episodes**. Nothing is distilled here. The
//       backfill is a write path and only a write path; turning those episodes
//       into beliefs is sleep's job, and it will run over them exactly as it
//       runs over a normal day's. The first sleep is just a very long one.
//
//  Idempotent by construction: an episode takes the *id of the message it came
//  from*, so re-running inserts nothing new (`INSERT OR IGNORE`), and the
//  markdown file is gone from the import path after the first pass.
//

import Foundation

// MARK: - The corpus reader

/// Reads the agent's persisted conversations into episodes.
///
/// Deliberately decoded through its own minimal `Decodable` shapes rather than
/// through `AgentConversationStore`'s codecs: this must survive files written by
/// older builds, and a backfill that throws on one unreadable conversation from
/// March is worse than a backfill that skips it.
nonisolated enum ConversationCorpus {

    static func episodes(in directory: URL) -> [Episode] {
        let fm = FileManager.default
        guard let names = try? fm.contentsOfDirectory(atPath: directory.path) else {
            Log.memory.info("No conversation corpus at \(directory.path)")
            return []
        }

        var out: [Episode] = []
        var skipped = 0
        for name in names.sorted() where name.hasSuffix(".json") && name != "index.json" {
            let url = directory.appendingPathComponent(name)
            guard let data = try? Data(contentsOf: url) else { skipped += 1; continue }
            do {
                let file = try decoder.decode(StoredConversation.self, from: data)
                out.append(contentsOf: episodes(of: file))
            } catch {
                skipped += 1
                Log.memory.info("Backfill skipped \(name): \(error.localizedDescription)")
            }
        }
        Log.memory.info(
            "Corpus: \(out.count) episodes from \(names.count - skipped) conversations "
                + "(\(skipped) unreadable)")
        return out
    }

    /// One episode per *user* message — because an episode is something the
    /// owner said. The assistant's reply rides along in `meta` as the context
    /// sleep needs to resolve "it", "that one", "the second approach". This is
    /// the same grain the live capture in `ChatSession` writes, so the backfill
    /// and the present are the same shape of memory.
    private static func episodes(of file: StoredConversation) -> [Episode] {
        var out: [Episode] = []
        for (index, message) in file.messages.enumerated() where message.type == "user" {
            let payload = message.payload
            guard let text = payload.content?.plainText, !text.isEmpty else { continue }
            let reply = nextAssistantText(in: file.messages, after: index)
            out.append(
                Episode(
                    // The message's own id: re-running the backfill is a no-op.
                    id: payload.id ?? UUID(),
                    source: .backfill,
                    conversationID: file.id.uuidString,
                    occurredAt: payload.timestamp ?? file.createdAt,
                    text: text,
                    meta: reply.isEmpty ? [:] : ["reply": String(reply.prefix(2_000))]
                ))
        }
        return out
    }

    private static func nextAssistantText(in messages: [StoredMessage], after index: Int) -> String
    {
        for message in messages[(index + 1)...] {
            if message.type == "user" { return "" }
            if message.type == "assistant", let text = message.payload.content?.plainText,
                !text.isEmpty
            {
                return text
            }
        }
        return ""
    }

    private static let decoder = JSONDecoder()

    // MARK: On-disk shapes

    private struct StoredConversation: Decodable {
        let id: UUID
        let createdAt: Date
        let messages: [StoredMessage]
    }

    private struct StoredMessage: Decodable {
        let type: String
        let payload: StoredPayload
    }

    private struct StoredPayload: Decodable {
        let id: UUID?
        let timestamp: Date?
        let content: StoredContent?
    }

    /// User content is a string; assistant content is an array of parts. One
    /// field, two shapes — so it decodes as either and flattens to text.
    private enum StoredContent: Decodable {
        case text(String)
        case parts([StoredPart])

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let text = try? container.decode(String.self) {
                self = .text(text)
            } else if let parts = try? container.decode([StoredPart].self) {
                self = .parts(parts)
            } else {
                self = .parts([])
            }
        }

        /// Thinking and tool calls are dropped: what the assistant *said* is
        /// context on the owner's turn; what it thought is not.
        var plainText: String {
            switch self {
            case .text(let text):
                return text.trimmingCharacters(in: .whitespacesAndNewlines)
            case .parts(let parts):
                return
                    parts
                    .filter { $0.type == "text" }
                    .compactMap(\.text)
                    .joined()
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }
    }

    private struct StoredPart: Decodable {
        let type: String?
        let text: String?
    }
}

// MARK: - The markdown migration

/// The `memories.md` era, ended.
nonisolated enum LegacyMemoriesFile {

    /// Every top-level bullet is one claim — which is exactly the atomicity the
    /// new store demands, so the translation is honest and one-to-one.
    static func claims(at url: URL) -> [String] {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return [] }
        return
            text
            .split(separator: "\n")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { $0.hasPrefix("- ") || $0.hasPrefix("* ") }
            .map { String($0.dropFirst(2)).trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
    }
}

// MARK: - The run

@MainActor
enum MemoryBackfill {

    struct Result: Sendable, Equatable {
        var claims = 0
        var episodes = 0
        var alreadyDone = false
    }

    /// Seed the store from what is already on disk. Safe to call on every
    /// launch: it does nothing once the store has been seeded.
    @discardableResult
    static func run(
        engine: MemoryEngine,
        sandboxRoot: URL = PathSandbox.defaultRoot,
        now: Date = Date()
    ) async -> Result {
        var result = Result()

        // The gate. An episode count above zero means this machine has already
        // been through here (or has simply been used), and the corpus import
        // would be re-inserting ids the store already holds.
        let existing = (try? await engine.store.episodeCount()) ?? 0
        guard existing == 0 else {
            result.alreadyDone = true
            return result
        }

        let markdown = sandboxRoot.appendingPathComponent("memories.md")
        let claims = LegacyMemoriesFile.claims(at: markdown)
        for claim in claims {
            // STATED: the owner wrote this file with his own hand. There is no
            // episode behind it, and that is the truth of its provenance — it
            // came from the old regime, not from a conversation.
            if await engine.remember(claim, kind: .belief, now: now) != nil {
                result.claims += 1
            }
        }
        if !claims.isEmpty {
            // Renamed, never deleted: this file is the only copy of these six
            // facts, and the owner gets to keep it.
            let archived = sandboxRoot.appendingPathComponent("memories.md.migrated")
            try? FileManager.default.removeItem(at: archived)
            try? FileManager.default.moveItem(at: markdown, to: archived)
            Log.memory.info("Migrated \(claims.count) claims from memories.md, archived the file")
        }

        let corpus = ConversationCorpus.episodes(
            in: sandboxRoot.appendingPathComponent("conversations", isDirectory: true))
        result.episodes = await engine.append(corpus)

        await engine.refreshStats()
        Log.memory.info(
            "Backfill complete: \(result.claims) claims, \(result.episodes) episodes. "
                + "They are unconsolidated — the first sleep will distil them.")
        return result
    }
}
