import Foundation

/// The seam the ``ChatSession`` uses to load, persist, and switch agent
/// conversations — exactly the members the coordinator calls, nothing more.
///
/// The production adapter is ``AgentConversationStore`` (JSON files under
/// Application Support); ``InMemoryAgentConversationStore`` is the hermetic test
/// fixture — a peer implementation sharing no global state and never touching
/// disk. Same protocol-seam + in-memory-fixture pattern as
/// `SettingsStore` / `InMemorySettingsStore`.
///
/// Class-bound: both adapters are reference types, and the coordinator holds one
/// by reference and mutates its current conversation through it.
///
/// Mission Control is read-only through this seam (ADR-0046): every write that
/// funnels through `currentConversation` — the save-outgoing inside
/// `createNew`/`load`, `updateCurrentMessages`, `saveCurrent` — refuses the
/// fold, in both adapters. The loop writes it through the concrete store's
/// `save(_:)`, deliberately not a member here.
protocol AgentConversationStoring: AnyObject {
    /// The conversation currently displayed and edited.
    var currentConversation: AgentConversation? { get }

    /// Loads the most recent conversation on startup (or starts a fresh one).
    func loadMostRecent()

    /// Creates a fresh conversation, persisting any non-empty current one first,
    /// and makes it current.
    @discardableResult
    func createNew() -> AgentConversation

    /// Loads a conversation by id and makes it current. A miss leaves the current
    /// conversation unchanged.
    func load(id: UUID)

    /// Deletes a conversation by id; if it was current, resets to a fresh one.
    func delete(id: UUID)

    /// Replaces the current conversation's messages in memory (the caller saves).
    func updateCurrentMessages(_ messages: [any AgentMessageProtocol & Sendable])

    /// Persists the current conversation.
    func saveCurrent()
}
