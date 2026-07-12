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
            // The same door the live capture goes through: a skill fire is the
            // app's words in his message, and 28 of the owner's first 207
            // episodes were nothing but that (see `MemorySpeech`).
            guard let raw = payload.content?.plainText, let text = MemorySpeech.spoken(raw)
            else { continue }
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

    /// **A conversation file encodes dates two different ways**, and this cost an
    /// evening to find because the failure is total and silent: the envelope's
    /// `createdAt` is an ISO-8601 *string* (`"2026-07-06T21:45:36Z"`), while each
    /// message's own `timestamp` is a bare reference-epoch `Double` — Swift's
    /// default `Date` coding, applied to the message structs but not to the
    /// envelope the store writes around them.
    ///
    /// No single `JSONDecoder.dateDecodingStrategy` reads both. Pick `.iso8601`
    /// and every message timestamp fails; leave it `.deferredToDate` — as this
    /// did — and every *file* fails on its first key, so the backfill quietly
    /// imports nothing at all and looks for all the world like it ran. So the
    /// dates here are decoded by hand, leniently, accepting either shape.
    private struct StoredConversation: Decodable {
        let id: UUID
        let createdAt: Date
        let messages: [StoredMessage]

        private enum CodingKeys: String, CodingKey { case id, createdAt, messages }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            id = try container.decode(UUID.self, forKey: .id)
            messages = try container.decode([StoredMessage].self, forKey: .messages)
            createdAt =
                LenientDate.decode(from: container, forKey: .createdAt) ?? Date.distantPast
        }
    }

    private struct StoredMessage: Decodable {
        let type: String
        let payload: StoredPayload
    }

    private struct StoredPayload: Decodable {
        let id: UUID?
        let timestamp: Date?
        let content: StoredContent?

        private enum CodingKeys: String, CodingKey { case id, timestamp, content }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            id = try? container.decodeIfPresent(UUID.self, forKey: .id)
            content = try? container.decodeIfPresent(StoredContent.self, forKey: .content)
            timestamp = LenientDate.decode(from: container, forKey: .timestamp)
        }
    }

    /// Either encoding, whichever this key happens to carry.
    private enum LenientDate {
        static func decode<Key: CodingKey>(
            from container: KeyedDecodingContainer<Key>, forKey key: Key
        ) -> Date? {
            if let seconds = try? container.decodeIfPresent(Double.self, forKey: key) {
                // Swift's default: seconds since the 2001 reference date, not 1970.
                // Reading these as Unix time shifts every episode by 31 years.
                return Date(timeIntervalSinceReferenceDate: seconds)
            }
            if let text = try? container.decodeIfPresent(String.self, forKey: key) {
                return iso.date(from: text) ?? isoWithFraction.date(from: text)
            }
            return nil
        }

        // `ISO8601DateFormatter` is not `Sendable`. These are configured once and
        // then only ever read from, so the unchecked annotation is the truth here
        // rather than a papering-over — and the backfill reads the corpus from
        // one task anyway.
        nonisolated(unsafe) static let iso = ISO8601DateFormatter()
        nonisolated(unsafe) static let isoWithFraction: ISO8601DateFormatter = {
            let formatter = ISO8601DateFormatter()
            formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
            return formatter
        }()
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

        // **Two imports, two gates.** They were one, and that was a bug: a run in
        // which the corpus imported but the markdown did not (or the reverse)
        // left the survivor permanently locked out, because the single gate had
        // already flipped. The two seeds are independent, so their gates are too:
        //
        //   - the corpus is gated on the store being empty of episodes (ids are
        //     the messages' own, so a re-run would insert nothing anyway — this
        //     is a shortcut, not a correctness guard);
        //   - the markdown is gated on the *file still being there*, which is the
        //     only honest record of whether it has been consumed.
        let existing = await engine.episodeCount()

        let markdown = sandboxRoot.appendingPathComponent("memories.md")
        let claims = LegacyMemoriesFile.claims(at: markdown)
        if !claims.isEmpty {
            // A partial import leaves the file in place for the next launch, so
            // the import must be re-runnable without duplicating what already
            // landed: skip any claim whose exact text is already a memory.
            let alreadyStored = Set(await engine.allMemories().map(\.text))
            var stored = 0
            for claim in claims {
                if alreadyStored.contains(claim) {
                    stored += 1
                    continue
                }
                // STATED: the owner wrote this file with his own hand. There is
                // no episode behind it, and that is the truth of its provenance
                // — it came from the old regime, not from a conversation.
                if await engine.remember(claim, kind: .belief, now: now) != nil {
                    stored += 1
                    result.claims += 1
                }
            }

            // **Archive only when every claim is durably stored.** This said
            // `!claims.isEmpty` — it archived on having *parsed* the claims, not
            // on having *kept* them — and so, on a run where the writes silently
            // failed, it moved the only copy of six hand-written facts out of
            // the import path and reported success. A migration that can destroy
            // its own source before the destination is durable is not a
            // migration.
            if stored == claims.count {
                let archived = sandboxRoot.appendingPathComponent("memories.md.migrated")
                try? FileManager.default.removeItem(at: archived)
                try? FileManager.default.moveItem(at: markdown, to: archived)
                Log.memory.info(
                    "Migrated \(result.claims) claims from memories.md; archived the file")
            } else {
                Log.memory.error(
                    "memories.md has \(claims.count) claims and only \(stored) could be stored — "
                        + "leaving the file alone so nothing is lost")
            }
        }

        if existing == 0 {
            // The corpus read is plain file IO over every conversation on disk —
            // off the main actor, or it stalls the launch it runs on.
            let directory = sandboxRoot.appendingPathComponent("conversations", isDirectory: true)
            let corpus = await Task.detached(priority: .utility) {
                ConversationCorpus.episodes(in: directory)
            }.value
            result.episodes = await engine.append(corpus)
        } else {
            result.alreadyDone = true
        }

        await engine.refreshStats()
        Log.memory.info(
            "Backfill: \(result.claims) claims, \(result.episodes) episodes"
                + "\(result.alreadyDone ? " (corpus already imported)" : "")"
                + ". Episodes are unconsolidated — the first sleep will distil them.")
        return result
    }
}
