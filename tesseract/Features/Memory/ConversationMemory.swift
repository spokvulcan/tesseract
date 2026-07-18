//
//  ConversationMemory.swift
//  tesseract
//
//  The chat's side of the living memory (ADR-0035 §5–§6): what this
//  conversation has already been told, and what the owner said this turn.
//
//  The Chat Session used to carry this fold inline, and the seam it sat off —
//  `handle(event)` — is exactly why a bare unwrap once shipped: the pipeline
//  hands `prepare` a `CoreMessage.user`, not a bare `UserMessage`, so
//  `outgoing as? UserMessage` matched nothing and every retrieved memory was
//  dropped between the store and the model. Nothing failed, no error was
//  logged, the store kept filling up — the model was simply never told. Not
//  one unit test caught it, because none of them crossed this seam; only
//  asking the running app what it remembered did. This module *is* that seam:
//  both verbs unwrap through the one canonical `asUser` accessor, and the
//  tests drive the wrapped shape the pipeline actually sends.
//

import Foundation

/// One conversation's memory decoration, behind two verbs: `enrich` hangs the
/// `<memory>` block on an outgoing user message (§5), `capture` appends the
/// turn to the episodic layer at turn end (§6). Owns the injection dedupe set
/// and the episode-identity rule — one turn, one id, in both directions.
///
/// A policy object over injected closures (the ADR-0034 shape): production
/// wires `MemoryEngine.injection` / `MemoryEngine.record` via
/// `init(memory:)`; tests can hand canned closures and read decision tables
/// off the interface without a store or an embedder.
@MainActor
final class ConversationMemory {

    /// What one turn hands the episodic layer. The engine still holds its own
    /// doors (enablement, `MemorySpeech` testimony trimming) — this value is
    /// the *turn's* facts: which id, whose words, what reply rode along.
    nonisolated struct EpisodeDraft: Equatable, Sendable {
        let id: UUID
        let text: String
        let conversationID: String?
        let occurredAt: Date
        let meta: [String: String]
    }

    /// One turn's recall — production: `MemoryEngine.injection` (ADR-0035 §5).
    private let recall:
        @MainActor (_ cue: String, _ episodeID: UUID, _ excluding: Set<UUID>) async ->
            MemoryInjection
    /// One turn's episodic append — production: `MemoryEngine.record` (§6).
    private let record: @MainActor (_ draft: EpisodeDraft) async -> Void

    /// What this conversation has already been told, so the next turn doesn't
    /// tell it again — a memory injected on turn 1 is still sitting in the
    /// context window on turn 7 (ADR-0035 §5).
    ///
    /// Session-scoped, and deliberately not persisted: reopening a conversation
    /// may re-inject a handful of lines it already carries. That costs a few
    /// dozen tokens once; threading the ids through the conversation file to
    /// avoid it would not be worth the schema.
    private var injectedMemoryIDs: Set<UUID> = []

    init(
        recall:
            @escaping @MainActor (
                _ cue: String, _ episodeID: UUID, _ excluding: Set<UUID>
            ) async -> MemoryInjection,
        record: @escaping @MainActor (_ draft: EpisodeDraft) async -> Void
    ) {
        self.recall = recall
        self.record = record
    }

    /// The production wiring: recall renders the `<memory>` block via
    /// `MemoryEngine.injection`; capture appends through `MemoryEngine.record`
    /// as `.chat` — the chat door of One Door Per Testimony.
    convenience init(memory: MemoryEngine) {
        self.init(
            recall: { cue, episodeID, excluding in
                await memory.injection(cue: cue, forEpisode: episodeID, excluding: excluding)
            },
            record: { draft in
                await memory.record(
                    id: draft.id, source: .chat, text: draft.text,
                    conversationID: draft.conversationID, occurredAt: draft.occurredAt,
                    meta: draft.meta)
            })
    }

    // MARK: - Enrich (ADR-0035 §5)

    /// Recall for this message and hang the `<memory>` block on it, returning
    /// the result in the same wrapper it arrived in — the agent pipeline is
    /// entitled to whatever shape it handed us. Identity for anything that
    /// isn't a user message, and for turns with nothing new to say.
    ///
    /// The message's *displayed* content is untouched: what the user wrote is
    /// what the user sees (`injectedContext` rides beside it).
    func enrich(
        _ outgoing: any AgentMessageProtocol & Sendable
    ) async -> any AgentMessageProtocol & Sendable {
        guard let user = outgoing.asUser else { return outgoing }
        let enriched = await enrich(user)
        return outgoing is CoreMessage ? CoreMessage.user(enriched) : enriched
    }

    /// The typed door — callers that hold a bare `UserMessage` (the loop's
    /// turn opening) get one back, no unwrap and no dead fallback at the
    /// call site.
    func enrich(_ user: UserMessage) async -> UserMessage {
        // `forEpisode: user.id` is what makes the lifecycle live. The episode
        // for this turn does not exist yet — `capture` writes it at turn end
        // *under the same id* — but the retrieval log can already point at it,
        // and that log is the only input the sleep judge ever gets. Without
        // this id nothing is ever logged, nothing is ever graded, and the
        // whole usefulness signal the lifecycle runs on is silently never
        // produced.
        let injection = await recall(user.content, user.id, injectedMemoryIDs)
        guard let text = injection.text else { return user }
        injectedMemoryIDs.formUnion(injection.memoryIDs)
        return user.with(injectedContext: text)
    }

    // MARK: - Capture (ADR-0035 §6)

    /// The turn ended — append it to the episodic layer.
    ///
    /// **An episode is something the owner said**, so the body is his message;
    /// the reply rides in `meta` as the context sleep needs to resolve "it",
    /// "that", "the second one". Nothing is judged here and nothing is
    /// extracted: the whole hot-path cost is one insert and one embedding
    /// (~3 ms), because at this moment the information that decides what
    /// *mattered* about this turn has not arrived yet.
    ///
    /// Detached and never awaited by the session — a turn that cannot be
    /// remembered is still a turn that was answered. Returns nil when the turn
    /// writes nothing (no user message in the context); the task handle is for
    /// tests, production discards it.
    @discardableResult
    func capture(
        reply: AssistantMessage,
        context: [any AgentMessageProtocol & Sendable],
        conversationID: String?
    ) -> Task<Void, Never>? {
        // The pipeline carries both wrapper shapes, so the walk unwraps through
        // the same canonical accessor as `enrich` — a bare cast here would
        // silently stop capturing the moment a caller wrapped its message. That
        // failure looks exactly like "he said nothing today".
        guard
            let user = context.reversed().lazy.compactMap(\.asUser).first,
            !user.content.isEmpty
        else { return nil }

        // Trimmed, because a turn that opens with a tool call has emitted only
        // whitespace by its first `turnEnd` — and an episode whose stored reply
        // is "\n\n" has lost the context this field exists to carry. The store
        // lets a later re-capture of the same turn fill the reply in.
        let replyText = reply.text.trimmingCharacters(in: .whitespacesAndNewlines)
        // The episode takes the user message's own id — the same id `enrich`
        // logged this turn's retrievals against, and the same grain the
        // backfill writes. One turn, one id, in both directions.
        let draft = EpisodeDraft(
            id: user.id,
            text: user.content,
            conversationID: conversationID,
            occurredAt: user.timestamp,
            meta: replyText.isEmpty ? [:] : ["reply": String(replyText.prefix(2_000))])
        let record = self.record
        return Task { await record(draft) }
    }

    // MARK: - Conversation boundary

    /// A conversation switch: the new window carries none of the old
    /// injections, so the dedupe set starts empty.
    func reset() {
        injectedMemoryIDs.removeAll()
    }
}
