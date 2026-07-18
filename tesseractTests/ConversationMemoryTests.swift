//
//  ConversationMemoryTests.swift
//  tesseractTests
//
//  The Conversation Memory's decision tables (ADR-0045): the chat's two verbs
//  against the living memory — enrich on send, capture at turn end — pinned
//  through the interface, with canned recall/record closures and no store.
//
//  The bug class that shipped is representable here: the pipeline hands
//  `prepare` a `CoreMessage.user`, not a bare `UserMessage`, and the first
//  unwrap written against it matched nothing — memory was retrieved and then
//  silently dropped on the floor on every single turn, and not one unit test
//  caught it because none of them crossed this seam. These do.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct ConversationMemoryTests {

    private static let block = "<memory>\n- He loves cats.\n</memory>"

    /// A collaborator whose recall always has something to say.
    private func alwaysRecalling(
        ids: Set<UUID> = [], onRecall: @MainActor @escaping (Set<UUID>) -> Void = { _ in }
    ) -> ConversationMemory {
        ConversationMemory(
            recall: { _, _, excluding in
                onRecall(excluding)
                return MemoryInjection(text: Self.block, memoryIDs: ids)
            },
            record: { _ in })
    }

    // MARK: - Enrich: the wrapper the pipeline actually sends

    @Test("The user message is found inside the pipeline's wrapper, and put back in it")
    func enrichUnwrapsAndRestoresTheWrapper() async throws {
        var askedCue: String?
        var askedEpisode: UUID?
        let memory = ConversationMemory(
            recall: { cue, episodeID, _ in
                askedCue = cue
                askedEpisode = episodeID
                return MemoryInjection(text: Self.block)
            },
            record: { _ in })

        let user = UserMessage(content: "where did I say I'm based?")
        let out = await memory.enrich(CoreMessage.user(user))

        // Wrapped in, wrapped out — hand the agent a bare `UserMessage` where
        // it expects a `CoreMessage` and the injection is lost a second way.
        let core = try #require(out as? CoreMessage)
        guard case .user(let enriched) = core else {
            Issue.record("rewrapped into the wrong case")
            return
        }
        #expect(enriched.injectedContext == Self.block)
        #expect(enriched.id == user.id)
        // The displayed content is untouched: what he wrote is what he sees.
        #expect(enriched.content == user.content)
        // The cue is his words; the episode id is the message's own — minted
        // before its episode exists, which is what makes the lifecycle live.
        #expect(askedCue == user.content)
        #expect(askedEpisode == user.id)
    }

    @Test("A bare user message comes back bare")
    func enrichKeepsABareMessageBare() async throws {
        let memory = alwaysRecalling()
        let user = UserMessage(content: "hello")
        let out = await memory.enrich(user)
        let enriched = try #require(out as? UserMessage)
        #expect(enriched.injectedContext == Self.block)
        #expect(enriched.id == user.id)
    }

    @Test("Anything that isn't a user message passes through without a recall")
    func enrichLeavesNonUserMessagesAlone() async throws {
        var recalls = 0
        let memory = ConversationMemory(
            recall: { _, _, _ in
                recalls += 1
                return MemoryInjection(text: Self.block)
            },
            record: { _ in })
        let out = await memory.enrich(CoreMessage.assistant(AssistantMessage(content: "hi")))
        #expect(out.asAssistant != nil)
        #expect(recalls == 0)
    }

    @Test("Nothing new to say means the message is untouched")
    func nothingNewMeansIdentity() async throws {
        let memory = ConversationMemory(
            recall: { _, _, _ in .none }, record: { _ in })
        let user = UserMessage(content: "hello again")
        let out = await memory.enrich(CoreMessage.user(user))
        let core = try #require(out as? CoreMessage)
        guard case .user(let unchanged) = core else {
            Issue.record("expected the user case back")
            return
        }
        #expect(unchanged.injectedContext == nil)
        #expect(unchanged == user)
    }

    // MARK: - The dedupe set's lifecycle

    @Test("What was injected on one turn is excluded on the next, until reset")
    func injectedIDsAreExcludedUntilReset() async {
        let carried: Set<UUID> = [UUID(), UUID()]
        var exclusions: [Set<UUID>] = []
        let memory = alwaysRecalling(ids: carried) { exclusions.append($0) }

        await _ = memory.enrich(CoreMessage.user(UserMessage(content: "turn one")))
        await _ = memory.enrich(CoreMessage.user(UserMessage(content: "turn two")))
        // A conversation switch: the new window carries none of the old
        // injections.
        memory.reset()
        await _ = memory.enrich(CoreMessage.user(UserMessage(content: "fresh window")))

        #expect(exclusions.count == 3)
        #expect(exclusions[0] == [])
        #expect(exclusions[1] == carried)
        #expect(exclusions[2] == [])
    }

    // MARK: - Capture: one turn, one id

    private func capturing() -> (ConversationMemory, () -> [ConversationMemory.EpisodeDraft]) {
        var drafts: [ConversationMemory.EpisodeDraft] = []
        let memory = ConversationMemory(
            recall: { _, _, _ in .none },
            record: { drafts.append($0) })
        return (memory, { drafts })
    }

    @Test("The episode takes the user message's own id, in either wrapper shape")
    func captureWritesUnderTheUserMessagesOwnID() async throws {
        let (memory, drafts) = capturing()
        let user = UserMessage(content: "I am allergic to shellfish")

        // The wrapped shape — the one the run loop's context actually carries,
        // and the one a bare cast silently stops capturing.
        let task = try #require(
            memory.capture(
                reply: AssistantMessage(content: "  Noted.  "),
                context: [
                    CoreMessage.user(user),
                    CoreMessage.assistant(AssistantMessage(content: "Noted.")),
                ],
                conversationID: "conv-1"))
        await task.value

        let draft = try #require(drafts().first)
        #expect(draft.id == user.id)
        #expect(draft.text == user.content)
        #expect(draft.conversationID == "conv-1")
        #expect(draft.occurredAt == user.timestamp)
        // The reply rides in meta, trimmed — sleep needs it to resolve "it".
        #expect(draft.meta == ["reply": "Noted."])
    }

    @Test("The last user message in the context is the turn's testimony")
    func captureFindsTheLastUserMessage() async throws {
        let (memory, drafts) = capturing()
        let earlier = UserMessage(content: "first question")
        let latest = UserMessage(content: "follow-up")

        let task = try #require(
            memory.capture(
                reply: AssistantMessage(content: "sure"),
                context: [CoreMessage.user(earlier), latest],
                conversationID: nil))
        await task.value
        #expect(drafts().first?.id == latest.id)
        #expect(drafts().first?.text == "follow-up")
    }

    @Test("A turn with no user message writes nothing")
    func noUserMessageMeansNoEpisode() {
        let (memory, _) = capturing()
        let task = memory.capture(
            reply: AssistantMessage(content: "hello"),
            context: [CoreMessage.assistant(AssistantMessage(content: "hello"))],
            conversationID: nil)
        #expect(task == nil)
    }

    @Test("An empty user message is not testimony")
    func emptyUserMessageIsNotAnEpisode() {
        let (memory, _) = capturing()
        // An images-only send has empty content — there is nothing he said.
        let task = memory.capture(
            reply: AssistantMessage(content: "a picture"),
            context: [CoreMessage.user(UserMessage(content: ""))],
            conversationID: nil)
        #expect(task == nil)
    }

    @Test("A whitespace-only reply leaves no meta; a long one is capped")
    func replyMetaIsTrimmedAndCapped() async throws {
        let (memory, drafts) = capturing()
        let user = UserMessage(content: "do the thing")

        // A turn that opens with a tool call has emitted only whitespace by its
        // first `turnEnd` — an episode whose stored reply is "\n\n" has lost
        // the context the field exists to carry.
        let silent = try #require(
            memory.capture(
                reply: AssistantMessage(content: "\n \n"),
                context: [CoreMessage.user(user)], conversationID: nil))
        await silent.value
        #expect(drafts()[0].meta == [:])

        let long = try #require(
            memory.capture(
                reply: AssistantMessage(content: String(repeating: "x", count: 2_500)),
                context: [CoreMessage.user(user)], conversationID: nil))
        await long.value
        #expect(drafts()[1].meta["reply"]?.count == 2_000)
    }
}
