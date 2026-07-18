//
//  CompanionIdentity.swift
//  tesseract
//
//  One Jarvis everywhere (ADR-0046, #370): the IDENTITY section of the
//  standing instructions rides every conversation, not just loop turns. This
//  is the chat side's decorator — the voice session sends through the same
//  chat path, so one seam covers both. It hangs the `<jarvis-identity>`
//  block on the conversation's first outgoing message, beside memory's
//  `<memory>` block on the same `injectedContext` field, so the persisted
//  transcript records exactly what the turn saw (the ADR-0045 rule).
//
//  Injected once per conversation: the block sits in the context window for
//  every later turn, same argument as memory's dedupe. The transcript scan
//  keeps a reopened conversation from collecting a second copy.
//

import Foundation

@MainActor
final class CompanionIdentity {

    private let store: MemoryStore
    private let isEnabled: () -> Bool
    private var injectedThisConversation = false

    init(store: MemoryStore, isEnabled: @escaping () -> Bool) {
        self.store = store
        self.isEnabled = isEnabled
    }

    /// Decorate the outgoing message with the identity block — once per
    /// conversation, only while the Companion exists (the toggle), and never
    /// into a transcript that already carries one.
    func decorate(
        _ user: UserMessage, transcript: [any AgentMessageProtocol & Sendable]
    ) async -> UserMessage {
        guard isEnabled(), !injectedThisConversation else { return user }
        guard
            !transcript.contains(where: {
                $0.asUser?.injectedContext?
                    .contains(CompanionInstructions.identityBlockMarker) == true
            })
        else {
            injectedThisConversation = true
            return user
        }
        guard let version = try? await store.currentInstructions() else { return user }
        injectedThisConversation = true
        let block = CompanionInstructions.wrapIdentity(version)
        let combined = [block, user.injectedContext].compactMap { $0 }
            .joined(separator: "\n\n")
        return user.with(injectedContext: combined)
    }

    /// The pipeline door — the same wrapper-restore contract as
    /// `ConversationMemory.enrich(_:)`: whatever shape the send handed over
    /// goes back in that shape.
    func decorate(
        _ outgoing: any AgentMessageProtocol & Sendable,
        transcript: [any AgentMessageProtocol & Sendable]
    ) async -> any AgentMessageProtocol & Sendable {
        guard let user = outgoing.asUser else { return outgoing }
        let decorated = await decorate(user, transcript: transcript)
        return outgoing is CoreMessage ? CoreMessage.user(decorated) : decorated
    }

    /// A conversation switch — the next conversation carries its own block.
    func reset() {
        injectedThisConversation = false
    }
}
