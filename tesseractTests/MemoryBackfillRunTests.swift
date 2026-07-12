//
//  MemoryBackfillRunTests.swift
//  tesseractTests
//
//  The backfill **end to end** — store, engine, markdown, corpus, all of it.
//
//  The unit tests around `ConversationCorpus` and `LegacyMemoriesFile` were all
//  green while the real backfill imported nothing at all: the parsers were right
//  and the *run* was wrong. Parsers are easy to test and the run is the thing
//  that ships, so the run gets its own suite.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@Suite("Memory backfill — the run")
struct MemoryBackfillRunTests {

    private func sandbox() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("backfill-run-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: url.appendingPathComponent("conversations"), withIntermediateDirectories: true)
        return url
    }

    /// The store comes back alongside the engine: these tests assert against
    /// what actually landed on disk, and the engine no longer exposes its store.
    @MainActor
    private func engine(at root: URL) throws -> (store: MemoryStore, engine: MemoryEngine) {
        let store = try MemoryStore(directory: root.appendingPathComponent("memory"))
        let engine = MemoryEngine(
            store: store,
            embedder: MemoryEmbedder(),
            isEnabled: { true },
            isDictationCaptureEnabled: { true },
            // No embedder in tests: retrieval degrades to keyword-only, and the
            // backfill must still import everything. Memory may never depend on a
            // model being present to record what happened.
            embedderDirectory: { nil }
        )
        return (store, engine)
    }

    private func writeMarkdown(_ root: URL) throws {
        try """
        - Bohdan is my user.
        - He loves cats.
        - His favorite Star Trek character is Spock.
        """.write(
            to: root.appendingPathComponent("memories.md"), atomically: true, encoding: .utf8)
    }

    private func writeConversation(_ root: URL, said: String) throws {
        let id = UUID()
        try """
        {"id": "\(id.uuidString)", "title": "t",
         "createdAt": "2026-07-06T21:45:36Z", "updatedAt": "2026-07-06T21:45:36Z",
         "messages": [
           {"type": "user", "payload": {"id": "\(UUID().uuidString)",
            "timestamp": 805076608.8, "content": "\(said)", "images": []}}]}
        """.write(
            to: root.appendingPathComponent("conversations/\(id.uuidString).json"),
            atomically: true, encoding: .utf8)
    }

    @MainActor
    @Test("The run imports the markdown claims AND the corpus, and archives the file")
    func theRunImportsEverything() async throws {
        let root = try sandbox()
        try writeMarkdown(root)
        try writeConversation(root, said: "I am allergic to shellfish")
        let (store, engine) = try engine(at: root)

        let result = await MemoryBackfill.run(engine: engine, sandboxRoot: root)

        // The bug this suite exists for: the archive ran, the log said "0 claims",
        // and six of the owner's hand-written facts went nowhere. Counting the
        // *result* is not enough — count what is actually in the store.
        #expect(result.claims == 3)
        #expect(result.episodes == 1)

        let memories = try await store.memories(status: nil)
        #expect(memories.count == 3)
        #expect(memories.contains { $0.text == "He loves cats." })
        // The owner wrote them, so they are testimony, not inference.
        #expect(memories.allSatisfy { $0.provenance == .stated })

        let episodes = try await store.episodeCount()
        #expect(episodes == 1)

        // Every claim is journalled — the Memory window's record of where a
        // belief came from.
        let journal = try await store.journal()
        #expect(journal.count == 3)

        // Archived, not deleted: it is the only copy of those facts.
        #expect(
            !FileManager.default.fileExists(atPath: root.appendingPathComponent("memories.md").path)
        )
        #expect(
            FileManager.default.fileExists(
                atPath: root.appendingPathComponent("memories.md.migrated").path))
    }

    @MainActor
    @Test("Running twice does not double-import")
    func theRunIsIdempotent() async throws {
        let root = try sandbox()
        try writeMarkdown(root)
        try writeConversation(root, said: "hello")
        let (store, engine) = try engine(at: root)

        _ = await MemoryBackfill.run(engine: engine, sandboxRoot: root)
        let second = await MemoryBackfill.run(engine: engine, sandboxRoot: root)

        #expect(second.alreadyDone)
        #expect(try await store.memoryCount() == 3)
        #expect(try await store.episodeCount() == 1)
    }

    /// The one that actually bit.
    ///
    /// The archive was gated on having *parsed* the claims, not on having *kept*
    /// them — so on a run where the writes silently went nowhere, it moved the
    /// only copy of six hand-written facts out of the import path and reported
    /// success. A migration that destroys its source before the destination is
    /// durable is not a migration.
    @MainActor
    @Test("A failed import leaves memories.md exactly where it is")
    func aFailedImportNeverDestroysItsSource() async throws {
        let root = try sandbox()
        try writeMarkdown(root)

        let store = try MemoryStore(directory: root.appendingPathComponent("memory"))
        let engine = MemoryEngine(
            store: store, embedder: MemoryEmbedder(),
            // Memory switched off: `remember` returns nil for every claim, which
            // is exactly the shape the live failure took.
            isEnabled: { false },
            isDictationCaptureEnabled: { false },
            embedderDirectory: { nil })

        let result = await MemoryBackfill.run(engine: engine, sandboxRoot: root)

        #expect(result.claims == 0)
        #expect(try await store.memoryCount() == 0)
        // The file is still there. It is the only copy.
        #expect(
            FileManager.default.fileExists(atPath: root.appendingPathComponent("memories.md").path))
        #expect(
            !FileManager.default.fileExists(
                atPath: root.appendingPathComponent("memories.md.migrated").path))
    }

    /// The corpus and the markdown are seeded independently, so one succeeding
    /// must not lock the other out — which a single shared gate did.
    @MainActor
    @Test("A store that already has episodes still imports a markdown file it has never seen")
    func theTwoSeedsAreGatedIndependently() async throws {
        let root = try sandbox()
        try writeConversation(root, said: "already here")
        let (store, engine) = try engine(at: root)

        // First run: corpus only, no markdown yet.
        _ = await MemoryBackfill.run(engine: engine, sandboxRoot: root)
        #expect(try await store.episodeCount() == 1)
        #expect(try await store.memoryCount() == 0)

        // The markdown turns up afterwards. The episode gate has already flipped;
        // the claims must import anyway.
        try writeMarkdown(root)
        let second = await MemoryBackfill.run(engine: engine, sandboxRoot: root)

        #expect(second.claims == 3)
        #expect(try await store.memoryCount() == 3)
        // And the corpus is not re-imported.
        #expect(try await store.episodeCount() == 1)
    }

    @MainActor
    @Test("An empty markdown file is not an error, and does not archive anything")
    func emptyMarkdownIsFine() async throws {
        let root = try sandbox()
        try "".write(
            to: root.appendingPathComponent("memories.md"), atomically: true, encoding: .utf8)
        try writeConversation(root, said: "still recorded")
        let (_, engine) = try engine(at: root)

        let result = await MemoryBackfill.run(engine: engine, sandboxRoot: root)

        #expect(result.claims == 0)
        #expect(result.episodes == 1)
        // Nothing was migrated, so nothing is archived — an empty file is left
        // exactly where it was.
        #expect(
            !FileManager.default.fileExists(
                atPath: root.appendingPathComponent("memories.md.migrated").path))
    }

    @MainActor
    @Test("Backfilled episodes land unconsolidated — the first sleep has work to do")
    func episodesAwaitTheFirstSleep() async throws {
        let root = try sandbox()
        try writeConversation(root, said: "my sister is flying in on Thursday")
        let (store, engine) = try engine(at: root)

        _ = await MemoryBackfill.run(engine: engine, sandboxRoot: root)

        let pending = try await store.unconsolidatedEpisodes()
        #expect(pending.count == 1)
        // Nothing is distilled by the backfill itself. That is sleep's job, and
        // the first sleep is simply a long one.
        #expect(try await store.memoryCount() == 0)
    }
}
