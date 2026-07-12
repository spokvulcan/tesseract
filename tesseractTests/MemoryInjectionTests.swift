//
//  MemoryInjectionTests.swift
//  tesseractTests
//
//  What reaches the model, and what reaches the user (ADR-0035 §5, §6).
//
//  These pin the two claims that the injection design rests on and that a
//  refactor could quietly break without failing anything else:
//
//    - the user sees what the user wrote, and only that;
//    - the model sees provenance at the point of use, not in a legend.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@Suite("Memory injection")
struct MemoryInjectionTests {

    private func memory(
        _ text: String, provenance: Provenance = .stated, status: MemoryStatus = .live,
        tier: MemoryTier = .hot
    ) -> MemoryRecord {
        var record = MemoryRecord(
            text: text, kind: .belief, provenance: provenance, tier: tier, bornAt: Date())
        record.status = status
        return record
    }

    // MARK: - The user's bubble

    @Test("The injected block never touches the message the user sees")
    func injectionIsInvisibleToTheUser() {
        let plain = UserMessage(content: "what do you think?")
        let carrying = UserMessage(
            content: "what do you think?", injectedContext: "<memory>\n- He loves cats.\n</memory>")

        // The displayed content — the only thing `UserMessageRow` renders — is
        // identical. If this ever fails, the user is reading the app's prompt
        // engineering in their own words.
        #expect(carrying.content == plain.content)
        #expect(carrying.content == "what do you think?")
    }

    @Test("The model sees the block, then the question")
    func theModelSeesBoth() throws {
        let message = UserMessage(
            content: "what should I cook?",
            injectedContext: "<memory>\n- He is vegetarian.\n</memory>"
        )
        let llm = try #require(message.toLLMMessage())
        guard case .user(let content, _) = llm else {
            Issue.record("expected a user message")
            return
        }
        #expect(content.hasPrefix("<memory>"))
        #expect(content.hasSuffix("what should I cook?"))
        // Order matters: what I know, then what I was asked.
        let memoryAt = try #require(content.range(of: "vegetarian")).lowerBound
        let questionAt = try #require(content.range(of: "cook")).lowerBound
        #expect(memoryAt < questionAt)
    }

    @Test("A message with no injection is byte-identical to one that never had the field")
    func noInjectionIsNoChange() throws {
        let message = UserMessage(content: "hello")
        let llm = try #require(message.toLLMMessage())
        guard case .user(let content, _) = llm else {
            Issue.record("expected a user message")
            return
        }
        #expect(content == "hello")
    }

    @Test("The injected block survives a round-trip through the conversation file")
    func injectionPersists() throws {
        // Load-time recomputation would rebuild an old conversation with
        // *different* context on every reopen — a prefix-cache miss on every
        // turn of the thread, and an unauditable history. So it is stored.
        let original = UserMessage(content: "hi", injectedContext: "<memory>\n- x\n</memory>")
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(UserMessage.self, from: data)
        #expect(decoded.injectedContext == original.injectedContext)
    }

    @Test("A message written before memory existed still decodes")
    func oldMessagesStillDecode() throws {
        // The owner has 66 conversations on disk, all written without this
        // field. Decoding must not have become stricter.
        let json = """
            {"id":"\(UUID().uuidString)","content":"old","images":[],"timestamp":123.4}
            """
        let decoded = try JSONDecoder().decode(UserMessage.self, from: Data(json.utf8))
        #expect(decoded.content == "old")
        #expect(decoded.injectedContext == nil)
    }

    // MARK: - The block

    @Test("Inferred memories are marked at the point of use, stated ones are not")
    func provenanceIsMarkedInline() throws {
        let block = try #require(
            MemoryPrompt.block(
                memories: [
                    memory("He loves cats.", provenance: .stated),
                    memory("He works late most evenings.", provenance: .inferred),
                ],
                episodes: [], now: Date()))

        #expect(block.contains("- He loves cats."))
        #expect(block.contains("- ~ He works late most evenings."))
        // The `~` is the whole safety story: a model handed a flat list of
        // "facts" defends a guess as hard as it defends testimony.
        #expect(!block.contains("~ He loves cats."))
    }

    @Test("A contested memory is carried with its dispute, never silently")
    func contestedIsCarriedWithItsDispute() throws {
        let block = try #require(
            MemoryPrompt.block(
                memories: [memory("He hates rain.", status: .contested)],
                episodes: [], now: Date()))
        #expect(block.contains("disputed"))
    }

    @Test("Nothing to say means no block at all")
    func emptyMeansNoBlock() {
        // The common case mid-conversation: everything relevant was already
        // injected on an earlier turn and is still in the window. Memory should
        // cost exactly zero tokens then.
        #expect(MemoryPrompt.block(memories: [], episodes: [], now: Date()) == nil)
    }

    @Test("Episodes are quoted verbatim, with the day they were said")
    func episodesAreQuoted() throws {
        let when = Date(timeIntervalSince1970: 1_760_000_000)
        let episode = Episode(
            source: .chat, occurredAt: when, text: "my sister is flying in on Thursday")
        let block = try #require(
            MemoryPrompt.block(memories: [], episodes: [episode], now: Date()))
        #expect(block.contains("\"my sister is flying in on Thursday\""))
        #expect(block.contains("2025-10-09"))
    }
}

// MARK: - The cold start

@Suite("Memory backfill")
struct MemoryBackfillTests {

    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("memory-backfill-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    @Test("memories.md becomes one stated claim per bullet")
    func markdownBecomesClaims() throws {
        let dir = try temporaryDirectory()
        let file = dir.appendingPathComponent("memories.md")
        try """
        - Bohdan is my user.
        - He loves cats.
        * His favorite Star Trek character is Spock.

        Some prose that is not a claim.
        """.write(to: file, atomically: true, encoding: .utf8)

        let claims = LegacyMemoriesFile.claims(at: file)
        #expect(claims.count == 3)
        #expect(claims.contains("He loves cats."))
        // A bullet is one claim, which is exactly the atomicity the new store
        // demands — so the translation is one-to-one and honest.
        #expect(!claims.contains { $0.contains("prose") })
    }

    @Test("The corpus yields one episode per user message, carrying the reply")
    func corpusYieldsUserEpisodes() throws {
        let dir = try temporaryDirectory()
        let conversationID = UUID()
        let userID = UUID()
        try """
        {
          "id": "\(conversationID.uuidString)",
          "title": "t",
          "createdAt": 700000000,
          "updatedAt": 700000000,
          "messages": [
            {"type": "user", "payload": {"id": "\(userID.uuidString)",
              "timestamp": 700000001, "content": "I am allergic to shellfish", "images": []}},
            {"type": "assistant", "payload": {"id": "\(UUID().uuidString)",
              "timestamp": 700000002,
              "content": [{"type": "thinking", "thinking": "hidden"},
                          {"type": "text", "text": "Noted."}]}}
          ]
        }
        """.write(
            to: dir.appendingPathComponent("\(conversationID.uuidString).json"),
            atomically: true, encoding: .utf8)

        let episodes = ConversationCorpus.episodes(in: dir)
        #expect(episodes.count == 1)
        let episode = try #require(episodes.first)

        // An episode is something the *owner* said.
        #expect(episode.text == "I am allergic to shellfish")
        #expect(episode.source == .backfill)
        #expect(episode.conversationID == conversationID.uuidString)
        // The id is the message's own — which is the whole reason re-running the
        // backfill inserts nothing.
        #expect(episode.id == userID)
        // The reply rides in meta as the context sleep needs to resolve "it".
        #expect(episode.meta["reply"] == "Noted.")
        // Thinking is not what the assistant said.
        #expect(episode.meta["reply"]?.contains("hidden") != true)
    }

    @Test("Re-reading the corpus produces the same episode ids")
    func backfillIsIdempotent() throws {
        let dir = try temporaryDirectory()
        let conversationID = UUID()
        let userID = UUID()
        try """
        {"id": "\(conversationID.uuidString)", "title": "t", "createdAt": 700000000,
         "updatedAt": 700000000, "messages": [
           {"type": "user", "payload": {"id": "\(userID.uuidString)",
            "timestamp": 700000001, "content": "hello", "images": []}}]}
        """.write(
            to: dir.appendingPathComponent("\(conversationID.uuidString).json"),
            atomically: true, encoding: .utf8)

        let first = ConversationCorpus.episodes(in: dir).map(\.id)
        let second = ConversationCorpus.episodes(in: dir).map(\.id)
        #expect(first == second)
    }

    @Test("One unreadable conversation does not take the backfill down")
    func corruptFilesAreSkipped() throws {
        let dir = try temporaryDirectory()
        let good = UUID()
        try """
        {"id": "\(good.uuidString)", "title": "t", "createdAt": 700000000,
         "updatedAt": 700000000, "messages": [
           {"type": "user", "payload": {"id": "\(UUID().uuidString)",
            "timestamp": 700000001, "content": "still here", "images": []}}]}
        """.write(
            to: dir.appendingPathComponent("\(good.uuidString).json"),
            atomically: true, encoding: .utf8)
        try "{ this is not json".write(
            to: dir.appendingPathComponent("\(UUID().uuidString).json"),
            atomically: true, encoding: .utf8)
        // The real corpus directory has one of these, and it is not a conversation.
        try "[]".write(
            to: dir.appendingPathComponent("index.json"), atomically: true, encoding: .utf8)

        let episodes = ConversationCorpus.episodes(in: dir)
        #expect(episodes.count == 1)
        #expect(episodes.first?.text == "still here")
    }
}
