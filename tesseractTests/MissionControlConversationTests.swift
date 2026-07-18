//
//  MissionControlConversationTests.swift
//  tesseractTests
//
//  Mission Control (#367, ADR-0046): the one standing conversation the loop
//  folds into. The persistence seam — identity under the well-known id, turns
//  appending across loads and relaunches, per-turn origin tags riding the
//  opening messages — plus the chat-side read-only guards that keep the fold's
//  only writer the loop. Each test opens its own unique temp directory so the
//  scheme's parallel twin runners can't collide.
//

import Foundation
import Testing

@testable import Tesseract_Agent

// MARK: - Fixtures

@MainActor
private func makeTempDir() -> URL {
    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("mission-control-tests-\(UUID().uuidString)", isDirectory: true)
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
}

/// One simulated loop turn: the origin-tagged opening plus the entity's reply
/// — the exact message shapes the runner appends.
@MainActor
private func turnMessages(
    opening: String, reply: String, origin: TurnOrigin
) -> [any AgentMessageProtocol & Sendable] {
    [
        CoreMessage.user(UserMessage(content: opening, turnOrigin: origin)),
        AssistantMessage(
            content: [.text(TextPart(text: reply))],
            model: "test-model",
            stopReason: .stop),
    ]
}

// MARK: - The persistence seam

@MainActor
struct MissionControlStoreTests {

    @Test func seedsEmptyUnderTheWellKnownID() {
        let dir = makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let store = AgentConversationStore(directory: dir)
        let missionControl = store.missionControl()

        #expect(missionControl.id == AgentConversation.missionControlID)
        #expect(missionControl.origin == .missionControl)
        #expect(missionControl.messages.isEmpty)
        #expect(missionControl.title == "Mission Control")
    }

    @Test func turnsAppendAndSurviveRelaunch() {
        let dir = makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        // Turn one — the runner's path: load the fold, append, save wholesale.
        let store = AgentConversationStore(directory: dir)
        var missionControl = store.missionControl()
        missionControl.messages += turnMessages(
            opening: "morning briefing", reply: "booking the day", origin: .wake)
        store.save(missionControl)

        // Turn two builds on turn one within the same launch.
        var second = store.missionControl()
        #expect(second.messages.count == 2)
        second.messages += turnMessages(
            opening: "nothing due", reply: "(silent turn)", origin: .ambient)
        store.save(second)

        // Relaunch: a fresh store over the same directory finds and continues
        // the same conversation — the whole fold, in order, tags intact.
        let relaunched = AgentConversationStore(directory: dir)
        let continued = relaunched.missionControl()
        #expect(continued.id == AgentConversation.missionControlID)
        #expect(continued.messages.count == 4)
        let tags = continued.messages.compactMap { $0.asUser?.turnOrigin }
        #expect(tags == [.wake, .ambient])

        // The one list shows one Mission Control row, named and tagged.
        let summary = relaunched.conversations.first {
            $0.id == AgentConversation.missionControlID
        }
        #expect(summary?.title == "Mission Control")
        #expect(summary?.turnOrigin == .missionControl)
    }

    @Test func loadMostRecentNeverLandsInsideTheFold() {
        let dir = makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let store = AgentConversationStore(directory: dir)

        // The owner's own chat, saved first — strictly older than the fold.
        store.createNew()
        store.updateCurrentMessages([CoreMessage.user(UserMessage(content: "hello"))])
        store.saveCurrent()
        let ownChatID = store.currentConversation?.id

        var missionControl = store.missionControl()
        missionControl.messages = turnMessages(
            opening: "opening", reply: "thinking", origin: .wake)
        store.save(missionControl)

        // Mission Control is the most recently updated — launch still lands
        // on the owner's own last chat.
        let relaunched = AgentConversationStore(directory: dir)
        relaunched.loadMostRecent()
        #expect(relaunched.currentConversation?.id == ownChatID)

        // With nothing but the fold on disk, launch opens a fresh chat.
        let foldOnly = makeTempDir()
        defer { try? FileManager.default.removeItem(at: foldOnly) }
        let foldStore = AgentConversationStore(directory: foldOnly)
        var onlyFold = foldStore.missionControl()
        onlyFold.messages = turnMessages(opening: "o", reply: "r", origin: .beat)
        foldStore.save(onlyFold)
        let foldRelaunch = AgentConversationStore(directory: foldOnly)
        foldRelaunch.loadMostRecent()
        #expect(foldRelaunch.currentConversation?.isMissionControl == false)
        #expect(foldRelaunch.currentConversation?.messages.isEmpty == true)
    }

    @Test func ownerDeletionReseedsEmptyUnderTheSameID() {
        let dir = makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let store = AgentConversationStore(directory: dir)
        var missionControl = store.missionControl()
        missionControl.messages = turnMessages(
            opening: "opening", reply: "reply", origin: .wake)
        store.save(missionControl)

        store.delete(id: AgentConversation.missionControlID)

        let reseeded = store.missionControl()
        #expect(reseeded.id == AgentConversation.missionControlID)
        #expect(reseeded.messages.isEmpty)
    }
}

// MARK: - The per-turn origin tag

@MainActor
struct UserMessageTurnOriginTests {

    private func decoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }

    @Test func tagRoundTripsThroughCodable() throws {
        // Whole-second timestamp: .iso8601 drops fractional seconds, and this
        // test is about the tag, not date precision.
        let message = UserMessage(
            content: "opening", timestamp: Date(timeIntervalSince1970: 1_752_000_000),
            turnOrigin: .catchup)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(message)
        let decoded = try decoder().decode(UserMessage.self, from: data)
        #expect(decoded.turnOrigin == .catchup)
        #expect(decoded == message)
    }

    @Test func preTagFilesDecodeUntagged() throws {
        // The exact shape every pre-ADR-0046 conversation file holds.
        let json = """
            {"id":"\(UUID().uuidString)","content":"hello",
            "timestamp":"2026-07-18T09:00:00Z"}
            """
        let decoded = try decoder().decode(UserMessage.self, from: Data(json.utf8))
        #expect(decoded.turnOrigin == nil)
    }

    @Test func unknownFutureTagReadsUntaggedInsteadOfFailingTheFile() throws {
        let json = """
            {"id":"\(UUID().uuidString)","content":"hello",
            "timestamp":"2026-07-18T09:00:00Z","turnOrigin":"from-the-future"}
            """
        let decoded = try decoder().decode(UserMessage.self, from: Data(json.utf8))
        #expect(decoded.turnOrigin == nil)
    }

    @Test func memoryEnrichmentPreservesTheTag() async {
        let memory = ConversationMemory(
            recall: { _, _, _ in
                MemoryInjection(text: "<memory>he likes tea</memory>", memoryIDs: [])
            },
            record: { _ in })
        let opening = UserMessage(content: "opening", turnOrigin: .wake)
        let enriched = (await memory.enrich(opening)).asUser
        #expect(enriched?.injectedContext != nil)
        #expect(enriched?.turnOrigin == .wake)
    }
}

// MARK: - The chat-side read-only guards

@MainActor
struct MissionControlChatGuardTests {

    private func makeSession(store: any AgentConversationStoring) -> ChatSession {
        ChatSession(
            agent: makeNoOpAgent(modelID: "test-model"),
            conversationStore: store,
            arbiter: InMemoryInferenceArbiter(),
            restoreComposerDraft: { _, _ in },
            liveMarkdownThrottle: .zero
        )
    }

    @Test func sendMessageIntoMissionControlIsRefused() {
        let dir = makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let store = AgentConversationStore(directory: dir)
        var missionControl = store.missionControl()
        missionControl.messages = turnMessages(
            opening: "opening", reply: "reply", origin: .wake)
        store.save(missionControl)

        let session = makeSession(store: store)
        session.loadConversation(AgentConversation.missionControlID)
        #expect(session.isMissionControlOpen)
        let itemsBefore = session.items.count

        session.sendMessage("typed into the fold")

        #expect(session.items.count == itemsBefore)
        #expect(store.missionControl().messages.count == 2)
    }

    @Test func switchingAwayNeverClobbersALoopTurnThatLandedMeanwhile() {
        let dir = makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let store = AgentConversationStore(directory: dir)
        var missionControl = store.missionControl()
        missionControl.messages = turnMessages(
            opening: "turn one", reply: "reply one", origin: .wake)
        store.save(missionControl)

        // The owner opens the fold to read — the session now holds a snapshot.
        let session = makeSession(store: store)
        session.loadConversation(AgentConversation.missionControlID)
        #expect(session.isMissionControlOpen)

        // A loop turn lands on disk while the fold is open in the UI — the
        // runner's exact path: load, append, save wholesale.
        var appended = store.missionControl()
        appended.messages += turnMessages(
            opening: "turn two", reply: "reply two", origin: .ambient)
        store.save(appended)

        // Switching away persists the outgoing conversation — for Mission
        // Control that write is skipped, so the meanwhile-turn survives.
        session.newConversation()
        #expect(store.missionControl().messages.count == 4)
    }

    @Test func editAndResendIsRefusedInsideTheFold() {
        let dir = makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let store = AgentConversationStore(directory: dir)
        let opening = UserMessage(content: "opening", turnOrigin: .wake)
        var missionControl = store.missionControl()
        missionControl.messages = [
            CoreMessage.user(opening),
            AssistantMessage(
                content: [.text(TextPart(text: "reply"))],
                model: "test-model",
                stopReason: .stop),
        ]
        store.save(missionControl)

        let session = makeSession(store: store)
        session.loadConversation(AgentConversation.missionControlID)

        #expect(session.beginEditingMessage(opening.id) == nil)
    }

    @Test func launchNeverOpensOnTheFold() {
        var missionControl = AgentConversation(
            id: AgentConversation.missionControlID, origin: .missionControl)
        missionControl.messages = turnMessages(
            opening: "opening", reply: "reply", origin: .wake)
        let store = InMemoryAgentConversationStore(seed: [missionControl])

        let session = makeSession(store: store)

        #expect(!session.isMissionControlOpen)
    }
}
