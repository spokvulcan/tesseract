//
//  MemoryStoreTests.swift
//  tesseractTests
//
//  The store (ADR-0035 §8, #319).
//
//  Every suite here injects a UUID-suffixed temp directory: it is the test
//  seam *and* the only defence against the scheme's parallel twin test runners
//  writing into one another's database.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@Suite("Memory store")
struct MemoryStoreTests {

    private func makeStore() throws -> (MemoryStore, URL) {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("memory-store-tests-\(UUID().uuidString)", isDirectory: true)
        return (try MemoryStore(directory: directory), directory)
    }

    private func vector(_ seed: Float, dimension: Int = 8) -> [Float] {
        // A unit vector, so cosine is a dot product — as the real embedder
        // guarantees.
        var v = [Float](repeating: 0, count: dimension)
        v[Int(seed) % dimension] = 1
        return v
    }

    // MARK: - Episodes: the immutable layer

    @Test("An episode round-trips, verbatim")
    func episodeRoundTrips() async throws {
        let (store, directory) = try makeStore()
        defer { try? FileManager.default.removeItem(at: directory) }

        let episode = Episode(
            source: .chat, conversationID: "conv-1",
            occurredAt: Date(timeIntervalSince1970: 1_700_000_000),
            text: "I want to ship the memory system tonight.",
            meta: ["model": "qwen3.5-4b"])
        try await store.append(episode, embedding: vector(1))

        let loaded = try await store.episode(id: episode.id)
        #expect(loaded?.text == episode.text)
        #expect(loaded?.source == .chat)
        #expect(loaded?.conversationID == "conv-1")
        #expect(loaded?.meta["model"] == "qwen3.5-4b")
        #expect(try await store.episodeCount() == 1)
    }

    @Test("Re-appending the same episode is a no-op — the layer is append-only")
    func episodeAppendIsIdempotent() async throws {
        let (store, directory) = try makeStore()
        defer { try? FileManager.default.removeItem(at: directory) }

        let episode = Episode(source: .chat, occurredAt: Date(), text: "Original text.")
        try await store.append(episode)

        // A second append with the SAME id but different text must not
        // overwrite. Nothing in the system may rewrite an episode — it is the
        // only thing that can correct a drifted belief.
        let impostor = Episode(
            id: episode.id, source: .chat, occurredAt: Date(), text: "Rewritten text.")
        try await store.append(impostor)

        let loaded = try await store.episode(id: episode.id)
        #expect(loaded?.text == "Original text.")
        #expect(try await store.episodeCount() == 1)
    }

    @Test("Unconsolidated episodes are the sleep queue, and get drained")
    func unconsolidatedQueue() async throws {
        let (store, directory) = try makeStore()
        defer { try? FileManager.default.removeItem(at: directory) }

        let now = Date()
        let episodes = (0..<3).map {
            Episode(
                source: .chat, occurredAt: now.addingTimeInterval(Double($0)),
                text: "Turn \($0)")
        }
        for e in episodes { try await store.append(e) }

        #expect(try await store.unconsolidatedEpisodes().count == 3)

        try await store.markConsolidated([episodes[0].id, episodes[1].id], at: now)
        let remaining = try await store.unconsolidatedEpisodes()
        #expect(remaining.count == 1)
        #expect(remaining.first?.id == episodes[2].id)
    }

    // MARK: - Memories

    @Test("A memory round-trips with its lifecycle state and sources")
    func memoryRoundTrips() async throws {
        let (store, directory) = try makeStore()
        defer { try? FileManager.default.removeItem(at: directory) }

        let episode = Episode(source: .chat, occurredAt: Date(), text: "source turn")
        try await store.append(episode)

        let memory = MemoryRecord(
            text: "I've noticed Bohdan prefers to start with the hardest task.",
            kind: .belief, provenance: .inferred, specificity: .general,
            tier: .hot, sourceEpisodeIDs: [episode.id], bornAt: Date(),
            stability: 12.5, storageStrength: 2.25, usefulUseCount: 3, confirmations: 4)
        try await store.upsert(memory, embedding: vector(2))

        let loaded = try #require(try await store.memory(id: memory.id))
        #expect(loaded.text == memory.text)
        #expect(loaded.provenance == .inferred)
        #expect(loaded.stability == 12.5)
        #expect(loaded.storageStrength == 2.25)
        #expect(loaded.usefulUseCount == 3)
        #expect(loaded.confirmations == 4)
        #expect(loaded.sourceEpisodeIDs == [episode.id])
    }

    @Test("Upsert updates in place without duplicating")
    func upsertUpdates() async throws {
        let (store, directory) = try makeStore()
        defer { try? FileManager.default.removeItem(at: directory) }

        var memory = MemoryRecord(
            text: "First version.", kind: .belief, provenance: .inferred, bornAt: Date())
        try await store.upsert(memory, embedding: vector(1))

        memory.text = "Second version."
        memory.stability = 40
        try await store.upsert(memory, embedding: vector(2))

        #expect(try await store.memoryCount() == 1)
        let loaded = try await store.memory(id: memory.id)
        #expect(loaded?.text == "Second version.")
        #expect(loaded?.stability == 40)
    }

    @Test("Deleting is the owner's hand alone — and it journals itself")
    func deleteIsJournaled() async throws {
        let (store, directory) = try makeStore()
        defer { try? FileManager.default.removeItem(at: directory) }

        let memory = MemoryRecord(
            text: "Something wrong about me.", kind: .belief, provenance: .inferred,
            bornAt: Date())
        try await store.upsert(memory, embedding: vector(1))
        try await store.deleteMemory(id: memory.id)

        #expect(try await store.memoryCount() == 0)
        #expect(try await store.embedding(for: memory.id) == nil)

        let journal = try await store.journal()
        #expect(journal.count == 1)
        #expect(journal.first?.mutation == .deletedByOwner)
        #expect(journal.first?.before == "Something wrong about me.")
    }

    // MARK: - Retrieval signals

    @Test("Cosine similarity finds the near neighbour")
    func cosineFindsNeighbour() {
        let a: [Float] = [1, 0, 0, 0]
        let b: [Float] = [0.9, 0.1, 0, 0]
        let c: [Float] = [0, 0, 0, 1]
        #expect(MemoryStore.cosine(a, b) > MemoryStore.cosine(a, c))
        #expect(abs(MemoryStore.cosine(a, a) - 1.0) < 0.0001)
    }

    @Test("Embeddings round-trip through the BLOB column bit-for-bit")
    func embeddingRoundTrips() async throws {
        let (store, directory) = try makeStore()
        defer { try? FileManager.default.removeItem(at: directory) }

        let memory = MemoryRecord(
            text: "vector test", kind: .belief, provenance: .stated, bornAt: Date())
        let v: [Float] = [0.1, -0.25, 0.5, 0.75]
        try await store.upsert(memory, embedding: v)

        let loaded = try await store.embedding(for: memory.id)
        #expect(loaded == v)
    }

    @Test("FTS5 keyword search finds a memory by a rare token")
    func keywordSearchWorks() async throws {
        let (store, directory) = try makeStore()
        defer { try? FileManager.default.removeItem(at: directory) }

        let hit = MemoryRecord(
            text: "I've noticed Bohdan's favourite Star Trek character is Spock.",
            kind: .belief, provenance: .stated, bornAt: Date())
        let miss = MemoryRecord(
            text: "I've noticed he prefers summer rain.",
            kind: .belief, provenance: .stated, bornAt: Date())
        try await store.upsert(hit)
        try await store.upsert(miss)

        // The proper-noun case: exactly what dense embeddings smear and
        // keyword search nails.
        let scores = try await store.keywordScores(query: "Spock", table: "memories")
        #expect(scores[hit.id] != nil)
        #expect(scores[miss.id] == nil)
    }

    @Test("An apostrophe in the query does not become a syntax error")
    func ftsQueryIsSanitized() async throws {
        let (store, directory) = try makeStore()
        defer { try? FileManager.default.removeItem(at: directory) }

        let memory = MemoryRecord(
            text: "I've noticed he likes rain.", kind: .belief, provenance: .stated,
            bornAt: Date())
        try await store.upsert(memory)

        // FTS5 treats a lot of punctuation as syntax; a raw user string would
        // blow up the query.
        let scores = try await store.keywordScores(
            query: "what's he like? (rain — yes!)", table: "memories")
        #expect(scores[memory.id] != nil)
    }

    // MARK: - The retrieval log

    @Test("Retrievals log ungraded, and grading drains the queue")
    func retrievalLogAndGrading() async throws {
        let (store, directory) = try makeStore()
        defer { try? FileManager.default.removeItem(at: directory) }

        let episode = Episode(source: .chat, occurredAt: Date(), text: "the turn")
        try await store.append(episode)
        let memory = MemoryRecord(
            text: "a belief", kind: .belief, provenance: .inferred, bornAt: Date())
        try await store.upsert(memory)

        let event = RetrievalEvent(
            memoryID: memory.id, episodeID: episode.id, retrievedAt: Date(), cue: "hardest task")
        try await store.log([event])

        #expect(try await store.ungradedRetrievals().count == 1)
        try await store.setGrade(.decisive, for: event.id)
        #expect(try await store.ungradedRetrievals().isEmpty)
    }

    @Test("Distinct useful days counts days, not uses — the spacing effect needs this")
    func distinctUsefulDaysCountsDays() async throws {
        let (store, directory) = try makeStore()
        defer { try? FileManager.default.removeItem(at: directory) }

        let episode = Episode(source: .chat, occurredAt: Date(), text: "turn")
        try await store.append(episode)
        let memory = MemoryRecord(
            text: "a belief", kind: .belief, provenance: .inferred, bornAt: Date())
        try await store.upsert(memory)

        let day: TimeInterval = 86_400
        let base = Date(timeIntervalSince1970: 1_700_000_000)
        // Three uses, but two of them inside the same day — one massed episode.
        let times = [base, base.addingTimeInterval(600), base.addingTimeInterval(3 * day)]
        for t in times {
            let e = RetrievalEvent(
                memoryID: memory.id, episodeID: episode.id, retrievedAt: t, cue: "cue")
            try await store.log([e])
            try await store.setGrade(.used, for: e.id)
        }

        #expect(try await store.distinctUsefulDays(memoryID: memory.id) == 2)
    }

    @Test("Ignored retrievals decay per-cue affinity, and nothing else")
    func cueAffinityDecays() async throws {
        let (store, directory) = try makeStore()
        defer { try? FileManager.default.removeItem(at: directory) }

        let memory = MemoryRecord(
            text: "a belief", kind: .belief, provenance: .inferred, bornAt: Date())
        try await store.upsert(memory)

        try await store.decayCueAffinity(cue: "morning routine", memoryID: memory.id)
        try await store.decayCueAffinity(cue: "morning routine", memoryID: memory.id)

        let affinities = try await store.cueAffinities(cue: "morning routine")
        let value = try #require(affinities[memory.id])
        #expect(abs(value - 0.81) < 0.0001)

        // The memory itself is untouched.
        let loaded = try await store.memory(id: memory.id)
        #expect(loaded?.stability == memory.stability)
        #expect(loaded?.storageStrength == memory.storageStrength)
    }

    @Test("Cue keys bucket, so the same question twice hits the same key")
    func cueKeysBucket() {
        let a = MemoryStore.cueKey("What does Bohdan do in the morning?")
        let b = MemoryStore.cueKey("in the MORNING, what does Bohdan do")
        #expect(a == b)
    }

    // MARK: - Schema

    @Test("Reopening an existing store keeps its contents")
    func storeIsDurable() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("memory-durable-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let memoryID: UUID
        do {
            let store = try MemoryStore(directory: directory)
            let memory = MemoryRecord(
                text: "survives a restart", kind: .belief, provenance: .stated, bornAt: Date())
            memoryID = memory.id
            try await store.upsert(memory, embedding: vector(3))
        }

        let reopened = try MemoryStore(directory: directory)
        let loaded = try await reopened.memory(id: memoryID)
        #expect(loaded?.text == "survives a restart")
    }
}
