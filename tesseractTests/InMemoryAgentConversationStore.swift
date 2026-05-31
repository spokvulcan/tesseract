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
        currentConversation = stored.values
            .sorted { $0.updatedAt > $1.updatedAt }
            .first ?? AgentConversation()
    }

    @discardableResult
    func createNew() -> AgentConversation {
        if let current = currentConversation, !current.messages.isEmpty {
            stored[current.id] = current
        }
        let conversation = AgentConversation()
        currentConversation = conversation
        return conversation
    }

    func load(id: UUID) {
        if let current = currentConversation, !current.messages.isEmpty {
            stored[current.id] = current
        }
        // A miss leaves the current conversation unchanged, like the real store.
        guard let conversation = stored[id] else { return }
        currentConversation = conversation
    }

    func delete(id: UUID) {
        stored.removeValue(forKey: id)
        if currentConversation?.id == id {
            currentConversation = AgentConversation()
        }
    }

    func updateCurrentMessages(_ messages: [any AgentMessageProtocol & Sendable]) {
        currentConversation?.messages = messages
    }

    func saveCurrent() {
        guard let current = currentConversation else { return }
        stored[current.id] = current
    }
}
