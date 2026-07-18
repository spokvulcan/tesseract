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
                episodes: []))

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
                episodes: []))
        #expect(block.contains("disputed"))
    }

    @Test("Nothing to say means no block at all")
    func emptyMeansNoBlock() {
        // The common case mid-conversation: everything relevant was already
        // injected on an earlier turn and is still in the window. Memory should
        // cost exactly zero tokens then.
        #expect(MemoryPrompt.block(memories: [], episodes: []) == nil)
    }

    @Test("Episodes are quoted verbatim, with the day they were said")
    func episodesAreQuoted() throws {
        let when = Date(timeIntervalSince1970: 1_760_000_000)
        let episode = Episode(
            source: .chat, occurredAt: when, text: "my sister is flying in on Thursday")
        let block = try #require(
            MemoryPrompt.block(memories: [], episodes: [episode]))
        #expect(block.contains("\"my sister is flying in on Thursday\""))
        #expect(block.contains("2025-10-09"))
    }

    // MARK: - What counts as something he said

    @Test("A skill fire is the app talking — only his words survive it")
    func skillWrappersAreNotTestimony() throws {
        let fired = """
            <skill name="proofread" location="/Users/owl/…/proofread/SKILL.md">
            References are relative to /Users/owl/…/proofread.

            You are a proofreader. Fix grammar. Do not change meaning.
            </skill>

            proofread this: their going to the office
            """
        #expect(MemorySpeech.spoken(fired) == "proofread this: their going to the office")

        // A bare fire with no arguments is the app talking to itself. It is not
        // an episode, and recording it would put the app's own instructions into
        // the record of his life — which is exactly what happened to 28 of the
        // first 207.
        let bare = """
            <skill name="translate" location="/x/SKILL.md">
            Translate to Ukrainian.
            </skill>
            """
        #expect(MemorySpeech.spoken(bare) == nil)

        // Ordinary messages are untouched, angle brackets and all.
        #expect(
            MemorySpeech.spoken("is a<b true when b>a?") == "is a<b true when b>a?")
        // A word that merely starts with the tag name is not the tag.
        #expect(MemorySpeech.spoken("<skills> what can you do?") == "<skills> what can you do?")
        // An unterminated wrapper takes the rest with it — a truncated wrapper
        // is not testimony either.
        #expect(MemorySpeech.spoken("<skill name=\"x\"> body with no end") == nil)
    }

    @Test("The same sentence is quoted once, however many times he said it")
    func identicalQuotesAreCollapsed() throws {
        let when = Date(timeIntervalSince1970: 1_760_000_000)
        let episodes = (0..<3).map { index in
            Episode(
                source: .chat,
                occurredAt: when.addingTimeInterval(Double(index) * 60),
                text: "Continue, please")
        }
        let block = try #require(
            MemoryPrompt.block(memories: [], episodes: episodes))
        let occurrences = block.components(separatedBy: "\"Continue, please\"").count - 1
        #expect(occurrences == 1)
    }

    // MARK: - The lifecycle's sensor

    // The wrapper-shape seam — the unwrap that once silently dropped every
    // retrieved memory — is pinned in `ConversationMemoryTests`, through the
    // interface that owns it now (ADR-0045).

    /// The lifecycle's sensor. Without an episode id, `retrieve` logs nothing —
    /// and that is exactly what shipped: `attachMemory` called
    /// `injection(cue:excluding:)`, the `retrievals` table stayed empty forever,
    /// grading had no input, and every surfacing still counted as "seen". Seen
    /// climbing with useful-use pinned at zero is the second retirement path's
    /// trigger: the lifecycle inverts into a retire-everything machine. The id
    /// is the user message's own — the same id `captureEpisode` later writes
    /// the turn under, which is what lets the log point at an episode that does
    /// not exist yet.
    @Test("Injection logs its retrievals against the turn's episode id")
    @MainActor
    func injectionLogsRetrievals() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("memory-injection-log-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = try MemoryStore(directory: directory)
        let engine = MemoryEngine(
            store: store, embedder: MemoryEmbedder(),
            isEnabled: { true }, isDictationCaptureEnabled: { true },
            embedderDirectory: { nil })

        try await store.upsert(
            MemoryRecord(
                text: "He is allergic to shellfish.", kind: .belief, provenance: .stated,
                bornAt: Date()))

        // The user message's id, minted before its episode exists.
        let turnID = UUID()
        let injection = await engine.injection(
            cue: "can I order the shellfish platter?", forEpisode: turnID)
        #expect(injection.text != nil)

        let events = try await store.ungradedRetrievals()
        #expect(!events.isEmpty, "nothing was logged — grading has no input")
        #expect(events.allSatisfy { $0.episodeID == turnID })
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
          "createdAt": "2026-07-06T21:45:36Z",
          "updatedAt": "2026-07-06T21:45:36Z",
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
        {"id": "\(conversationID.uuidString)", "title": "t", "createdAt": "2026-07-06T21:45:36Z",
         "updatedAt": "2026-07-06T21:45:36Z", "messages": [
           {"type": "user", "payload": {"id": "\(userID.uuidString)",
            "timestamp": 700000001, "content": "hello", "images": []}}]}
        """.write(
            to: dir.appendingPathComponent("\(conversationID.uuidString).json"),
            atomically: true, encoding: .utf8)

        let first = ConversationCorpus.episodes(in: dir).map(\.id)
        let second = ConversationCorpus.episodes(in: dir).map(\.id)
        #expect(first == second)
    }

    /// The test that should have existed first.
    ///
    /// Every hand-written fixture above passed while the production reader
    /// decoded **exactly zero** of the owner's 65 real conversations: the
    /// envelope's `createdAt` is an ISO-8601 string, the messages' timestamps are
    /// reference-epoch Doubles, and a fixture written from memory had neither
    /// quirk. A green suite over invented data is not evidence about real data.
    ///
    /// Skips where there is no corpus (CI), so it is a guard, not a dependency.
    /// "No corpus" means no conversation files, not a missing directory — the
    /// test host's app boot creates the directory empty, so an existence check
    /// fails open on CI.
    @Test(
        "The real corpus decodes — not a fixture of it",
        .enabled(if: MemoryEvalCorpus.hasEvalCorpus(at: MemoryBackfillTests.realCorpus)))
    func theRealCorpusDecodes() throws {
        let episodes = ConversationCorpus.episodes(in: Self.realCorpus)

        // If this reads zero, the backfill silently imports nothing and the memory
        // system starts amnesiac — exactly the bug that shipped and had to be
        // caught by hand in the log.
        #expect(episodes.count > 50, "the real corpus should yield real episodes")

        for episode in episodes {
            #expect(!episode.text.isEmpty)
            // A date that failed to decode lands on `distantPast`; a Unix-vs-2001
            // epoch mix-up lands 31 years off. Both are caught here.
            #expect(episode.occurredAt > Date(timeIntervalSince1970: 1_600_000_000))
            #expect(episode.occurredAt < Date().addingTimeInterval(86_400))
        }

        // Ids come from the messages, so a second read is the same set — which is
        // what makes the backfill safe to re-run on every launch.
        #expect(Set(episodes.map(\.id)).count == episodes.count, "episode ids are unique")
    }

    /// The production path, resolved the production way.
    ///
    /// Not `homeDirectoryForCurrentUser` + a literal container path: the test host
    /// runs *inside* the sandbox, where the home directory already **is** the
    /// container — so that spelling silently pointed at a directory that does not
    /// exist, `.enabled(if:)` went false, and the test skipped instead of failing.
    /// A guard that skips itself is not a guard.
    static let realCorpus = PathSandbox.defaultRoot
        .appendingPathComponent("conversations", isDirectory: true)

    @Test("One unreadable conversation does not take the backfill down")
    func corruptFilesAreSkipped() throws {
        let dir = try temporaryDirectory()
        let good = UUID()
        try """
        {"id": "\(good.uuidString)", "title": "t", "createdAt": "2026-07-06T21:45:36Z",
         "updatedAt": "2026-07-06T21:45:36Z", "messages": [
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

    @Test("The block leads with core — identity before the tail")
    func coreLeadsTheBlock() throws {
        let now = Date()
        var cold = MemoryRecord(
            text: "He once asked about a Dota tournament.", kind: .event,
            provenance: .inferred, bornAt: now)
        cold.tier = .cold
        var core = MemoryRecord(
            text: "He is the developer of Tesseract.", kind: .belief,
            provenance: .stated, bornAt: now)
        core.tier = .core

        // Handed over worst-first on purpose: the sort is the thing under test.
        let block = try #require(
            MemoryPrompt.block(memories: [cold, core], episodes: []))
        let coreAt = try #require(block.range(of: "developer of Tesseract"))
        let coldAt = try #require(block.range(of: "Dota tournament"))
        #expect(
            coreAt.lowerBound < coldAt.lowerBound,
            "a core belief has stopped being a retrieval — the model meets it first")
    }
}
