//
//  InMemoryAgentConversationStore.swift
//  tesseractTests
//
//  A hermetic, in-memory Agent Conversation Store — a dictionary, not a mock,
//  and a *peer implementation* of `AgentConversationStore`. Sharing no global
//  state and never touching disk, it runs hermetically and in parallel — the
//  same seam + fixture pattern as `InMemorySettingsStore`. Seed it with known
//  conversations to drive characterization of the coordinator's row output
//  through the public `loadConversation` path, with no Application Support I/O.
//

import Foundation

@testable import Tesseract_Agent

@MainActor
final class InMemoryAgentConversationStore: AgentConversationStoring {
    private(set) var currentConversation: AgentConversation?
    private var stored: [UUID: AgentConversation] = [:]

    /// Seed with known conversations. `currentConversation` starts nil until
    /// `loadMostRecent()` or `load(id:)` selects one — mirroring the real store,
    /// whose init calls `loadMostRecent()`.
    init(seed: [AgentConversation] = []) {
        for conversation in seed { stored[conversation.id] = conversation }
    }

    func loadMostRecent() {
        // Same rule as the real store, from the rule's one home (ADR-0046).
        currentConversation =
            stored.values
            .filter { $0.origin.opensAtLaunch }
            .min { $0.updatedAt > $1.updatedAt } ?? AgentConversation()
    }

    @discardableResult
    func createNew() -> AgentConversation {
        saveOutgoingCurrent()
        let conversation = AgentConversation()
        currentConversation = conversation
        return conversation
    }

    func load(id: UUID) {
        saveOutgoingCurrent()
        // A miss leaves the current conversation unchanged, like the real store.
        guard let conversation = stored[id] else { return }
        currentConversation = conversation
    }

    /// Same rule as the real store, from the seam's contract: the chat funnel
    /// never writes the fold (ADR-0046).
    private func saveOutgoingCurrent() {
        guard let current = currentConversation, !current.messages.isEmpty,
            !current.isMissionControl
        else { return }
        stored[current.id] = current
    }

    func delete(id: UUID) {
        stored.removeValue(forKey: id)
        if currentConversation?.id == id {
            currentConversation = AgentConversation()
        }
    }

    func updateCurrentMessages(_ messages: [any AgentMessageProtocol & Sendable]) {
        guard currentConversation?.isMissionControl != true else { return }
        currentConversation?.messages = messages
    }

    /// Fidelity boundary — where this fixture intentionally diverges from the
    /// real `AgentConversationStore.saveSync`, kept simple because no current
    /// characterization asserts against any of it:
    ///   - persists unconditionally (the real store guards on `hasUserMessages`),
    ///   - does not stamp `updatedAt` (the real store sets `updatedAt = Date()`),
    ///   - does not round-trip messages through `SyncMessageCodec`, and
    ///   - `loadMostRecent()` tie-breaks off unordered `stored.values` rather than
    ///     a maintained, recency-ordered conversation index.
    /// A future multi-conversation, recency-ordered, or codec-sensitive test must
    /// not lean on these — extend the fixture to match the real store first.
    func saveCurrent() {
        guard let current = currentConversation, !current.isMissionControl else { return }
        stored[current.id] = current
    }
}
