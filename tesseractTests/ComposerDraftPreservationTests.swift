//
//  ComposerDraftPreservationTests.swift
//  tesseractTests
//
//  The **Composer Draft** — the unsent composer text plus its pending images —
//  lives ABOVE any one conversation. Starting a new chat, switching threads, or
//  deleting the current conversation resets the transcript but carries the draft
//  across intact; only the explicit `/clear` hard reset discards it. These drive
//  the public coordinator interface against a seeded in-memory store, asserting
//  the observable composer state — never internals.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct ComposerDraftPreservationTests {

    // MARK: - Fixtures

    private func image(_ name: String) -> ImageAttachment {
        ImageAttachment(
            data: Data([0x89, 0x50, 0x4E, 0x47]), mimeType: "image/png", filename: name)
    }

    /// An idle `Agent` — its generate is never driven here; these exercise the
    /// synchronous conversation-lifecycle operations, not a run.
    private func makeIdleAgent() -> Agent {
        let config = AgentLoopConfig(
            model: AgentModelRef(id: "composer-draft-test-model"),
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

    private func makeCoordinator(store: InMemoryAgentConversationStore) -> AgentCoordinator {
        let coordinator = AgentCoordinator(
            agent: makeIdleAgent(),
            conversationStore: store,
            settings: SettingsManager(store: InMemorySettingsStore()),
            arbiter: InMemoryInferenceArbiter()
        )
        // The composer normally syncs this from the selected model; the pending
        // image represents a vision-capable model so it isn't gated away.
        coordinator.composerDraft.imageInputAvailable = true
        return coordinator
    }

    private func seededStore(_ contents: String...) -> InMemoryAgentConversationStore {
        InMemoryAgentConversationStore(
            seed: contents.map {
                AgentConversation(messages: [CoreMessage.user(UserMessage(content: $0))])
            })
    }

    /// Stage a representative draft — typed text plus one pending image (the
    /// freshly-captured Appshot case). Returns the staged image for id assertions.
    @discardableResult
    private func stageDraft(_ coordinator: AgentCoordinator) -> ImageAttachment {
        let shot = image("appshot")
        coordinator.composerDraft.text = "half-written thought"
        coordinator.composerDraft.pendingImages = [shot]
        return shot
    }

    // MARK: - Preserved across every conversation switch

    @Test func newConversationPreservesTheComposerDraft() {
        let coordinator = makeCoordinator(store: seededStore("prior turn"))
        let shot = stageDraft(coordinator)

        coordinator.newConversation()

        // The transcript is fresh...
        #expect(coordinator.rows.isEmpty)
        // ...but the unsent draft rode across intact — text AND image.
        #expect(coordinator.composerDraft.text == "half-written thought")
        #expect(coordinator.composerDraft.pendingImages.map(\.id) == [shot.id])
    }

    @Test func loadingAnotherConversationPreservesTheComposerDraft() {
        let current = AgentConversation(
            messages: [CoreMessage.user(UserMessage(content: "current thread"))])
        let other = AgentConversation(
            messages: [CoreMessage.user(UserMessage(content: "other thread"))])
        let store = InMemoryAgentConversationStore(seed: [current, other])
        let coordinator = makeCoordinator(store: store)
        let shot = stageDraft(coordinator)

        coordinator.loadConversation(other.id)

        #expect(coordinator.composerDraft.text == "half-written thought")
        #expect(coordinator.composerDraft.pendingImages.map(\.id) == [shot.id])
    }

    @Test func deletingTheCurrentConversationPreservesTheComposerDraft() {
        let store = seededStore("doomed thread")
        let coordinator = makeCoordinator(store: store)
        let currentID = store.currentConversation?.id
        let shot = stageDraft(coordinator)

        #expect(currentID != nil)
        coordinator.deleteConversation(currentID ?? UUID())

        #expect(coordinator.composerDraft.text == "half-written thought")
        #expect(coordinator.composerDraft.pendingImages.map(\.id) == [shot.id])
    }

    // MARK: - Discarded only by the explicit /clear hard reset

    @Test func clearHardResetDiscardsTheComposerDraft() {
        let coordinator = makeCoordinator(store: seededStore("prior turn"))
        stageDraft(coordinator)

        coordinator.clearConversation()  // the `/clear` hard reset

        #expect(coordinator.rows.isEmpty)
        #expect(coordinator.composerDraft.text == "")
        #expect(coordinator.composerDraft.pendingImages.isEmpty)
    }
}
