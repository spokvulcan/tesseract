//
//  AgentEditAndResendTests.swift
//  tesseractTests
//
//  Edit & resend is the recovery path for a bricked session: a sent user message
//  the model can't process (e.g. an over-large image set that trips the
//  vision-tower guard) is re-fed on every follow-up and re-rejected forever.
//  Editing it truncates the conversation to the turns before it and restores its
//  text + images to the composer, so the user can trim the images and re-send.
//
//  These drive the PUBLIC coordinator interface against a seeded in-memory store
//  (the same fixture the dispatch-ordering test uses), asserting observable
//  output — the committed rows and the composer state — never internals.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct AgentEditAndResendTests {

    // MARK: - Fixtures

    private func image(_ name: String) -> ImageAttachment {
        ImageAttachment(
            data: Data([0x89, 0x50, 0x4E, 0x47]), mimeType: "image/png", filename: name)
    }

    /// An idle `Agent` — its generate is never driven here; these tests exercise
    /// the synchronous edit operation, not a run.
    private func makeIdleAgent() -> Agent {
        let config = AgentLoopConfig(
            model: AgentModelRef(id: "edit-resend-test-model"),
            convertToLlm: { _ in [] },
            contextTransform: nil,
            getSteeringMessages: nil,
            getFollowUpMessages: nil
        )
        return Agent(
            config: config,
            systemPrompt: "test",
            tools: [],
            generate: { _, _, _, _ in AsyncThrowingStream { $0.finish() } }
        )
    }

    private func makeCoordinator(
        seeded messages: [any AgentMessageProtocol & Sendable],
        visionAvailable: Bool = true
    ) -> AgentCoordinator {
        let conversation = AgentConversation(messages: messages)
        let coordinator = AgentCoordinator(
            agent: makeIdleAgent(),
            conversationStore: InMemoryAgentConversationStore(seed: [conversation]),
            settings: SettingsManager(store: InMemorySettingsStore()),
            arbiter: InMemoryInferenceArbiter()
        )
        // The composer normally syncs this from the selected model; image-bearing
        // tests represent a vision-capable model unless they opt out.
        coordinator.imageInputAvailable = visionAvailable
        return coordinator
    }

    private func userContents(_ coordinator: AgentCoordinator) -> [String] {
        coordinator.rows.compactMap {
            if case .user(let row) = $0.kind { return row.content } else { return nil }
        }
    }

    private func assistantContents(_ coordinator: AgentCoordinator) -> [String] {
        coordinator.rows.compactMap {
            if case .assistantText(let row) = $0.kind { return row.content } else { return nil }
        }
    }

    // MARK: - Tests

    @Test func editingTheLastMessageLiftsItOutAndRestoresItToTheComposer() {
        let imgA = image("a")
        let imgB = image("b")
        let target = UserMessage(content: "describe these", images: [imgA, imgB])
        let coordinator = makeCoordinator(seeded: [
            CoreMessage.user(UserMessage(content: "first question")),
            CoreMessage.assistant(AssistantMessage(content: "an earlier answer")),
            CoreMessage.user(target),
        ])

        coordinator.beginEditingMessage(target.id)

        // The edited message is gone from the transcript; the earlier turn stays.
        #expect(userContents(coordinator).contains("describe these") == false)
        #expect(userContents(coordinator).contains("first question") == true)
        #expect(assistantContents(coordinator).contains("an earlier answer") == true)
        // Its text and images are back in the composer for editing + re-send.
        #expect(coordinator.editDraftRestore == "describe these")
        #expect(coordinator.pendingImages.map(\.id) == [imgA.id, imgB.id])
    }

    @Test func editingAMiddleMessageDropsItAndEverythingAfter() {
        let target = UserMessage(content: "second question")
        let coordinator = makeCoordinator(seeded: [
            CoreMessage.user(UserMessage(content: "first question")),
            CoreMessage.assistant(AssistantMessage(content: "first answer")),
            CoreMessage.user(target),
            CoreMessage.assistant(AssistantMessage(content: "second answer")),
        ])

        coordinator.beginEditingMessage(target.id)

        // Everything from the edited message onward is dropped; the first turn
        // survives intact.
        #expect(userContents(coordinator) == ["first question"])
        #expect(assistantContents(coordinator) == ["first answer"])
        #expect(coordinator.editDraftRestore == "second question")
        #expect(coordinator.pendingImages.isEmpty)
    }

    @Test func editingAnUnknownIdIsANoOp() {
        let coordinator = makeCoordinator(seeded: [
            CoreMessage.user(UserMessage(content: "only message"))
        ])

        coordinator.beginEditingMessage(UUID())

        #expect(userContents(coordinator) == ["only message"])
        #expect(coordinator.editDraftRestore == nil)
    }

    @Test func editingAnAssistantMessageIsANoOp() {
        let assistant = AssistantMessage(content: "the assistant reply")
        let coordinator = makeCoordinator(seeded: [
            CoreMessage.user(UserMessage(content: "a question")),
            CoreMessage.assistant(assistant),
        ])

        // Edit is only offered on user bubbles; targeting an assistant id changes
        // nothing.
        coordinator.beginEditingMessage(assistant.id)

        #expect(userContents(coordinator) == ["a question"])
        #expect(assistantContents(coordinator) == ["the assistant reply"])
        #expect(coordinator.editDraftRestore == nil)
    }

    @Test func editingTheFirstAndOnlyMessageDurablyDeletesTheBrickedConversation() {
        // The headline recovery case: a single over-large-image turn. Editing it
        // must DELETE the stored conversation, not write an empty one — the real
        // store's `saveSync` skips empty saves, so an empty write would leave the
        // bricked turn on disk to reload on relaunch.
        let target = UserMessage(content: "describe these", images: [image("a")])
        let conversation = AgentConversation(messages: [CoreMessage.user(target)])
        let store = InMemoryAgentConversationStore(seed: [conversation])
        let coordinator = AgentCoordinator(
            agent: makeIdleAgent(),
            conversationStore: store,
            settings: SettingsManager(store: InMemorySettingsStore()),
            arbiter: InMemoryInferenceArbiter()
        )
        coordinator.imageInputAvailable = true

        coordinator.beginEditingMessage(target.id)

        // Transcript cleared; content restored to the composer.
        #expect(userContents(coordinator).isEmpty)
        #expect(coordinator.editDraftRestore == "describe these")
        #expect(coordinator.pendingImages.map(\.id) == [target.images[0].id])
        // The bricked conversation is GONE, not merely emptied in place: `delete`
        // resets `currentConversation` to a fresh one (new id), whereas the buggy
        // empty-write path would keep the same id with cleared messages.
        #expect(store.currentConversation?.id != conversation.id)
        #expect(store.currentConversation?.messages.isEmpty == true)
    }

    @Test func editingTheFirstUserMessageAfterCompactionDeletesTheUnpersistableHead() {
        // A leading compaction summary makes `head` non-empty yet carrying no user
        // message when the first post-summary user turn is edited. The store skips
        // saving a head with no user message (its `hasUserMessages` guard), so the
        // truncation must DELETE the stored conversation — not write an
        // unpersistable head that leaves the (bricked) turn on disk to reload. A
        // `head.isEmpty` check would miss this; the store's own persist predicate
        // is the right gate.
        let target = UserMessage(content: "describe these", images: [image("a")])
        let conversation = AgentConversation(messages: [
            CompactionSummaryMessage(summary: "earlier context", tokensBefore: 4_000),
            CoreMessage.user(target),
            CoreMessage.assistant(AssistantMessage(content: "an answer")),
        ])
        let store = InMemoryAgentConversationStore(seed: [conversation])
        let coordinator = AgentCoordinator(
            agent: makeIdleAgent(),
            conversationStore: store,
            settings: SettingsManager(store: InMemorySettingsStore()),
            arbiter: InMemoryInferenceArbiter()
        )
        coordinator.imageInputAvailable = true

        coordinator.beginEditingMessage(target.id)

        // The summary-only head is not persistable, so the conversation is deleted
        // (fresh id), not rewritten to a head the store would silently drop.
        #expect(store.currentConversation?.id != conversation.id)
        #expect(store.currentConversation?.messages.isEmpty == true)
        #expect(coordinator.editDraftRestore == "describe these")
        #expect(coordinator.pendingImages.map(\.id) == [target.images[0].id])
    }

    @Test func editingAnImageMessageUnderATextOnlyModelDropsImagesAndHints() {
        // On a model that can't see images, restoring the attachments would render
        // chips that get silently dropped on send. Restore only the text and raise
        // the switch hint instead — matching the paste/drop gate.
        let target = UserMessage(content: "describe these", images: [image("a"), image("b")])
        let coordinator = makeCoordinator(
            seeded: [CoreMessage.user(target)],
            visionAvailable: false
        )

        coordinator.beginEditingMessage(target.id)

        #expect(coordinator.editDraftRestore == "describe these")
        #expect(coordinator.pendingImages.isEmpty)
        #expect(coordinator.showImageSwitchHint == true)
    }
}
