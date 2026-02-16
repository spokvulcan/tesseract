import Combine
import Foundation
import os

/// Persists agent conversations as JSON files following the ``TranscriptionHistory`` pattern.
///
/// Storage layout:
/// ```
/// ~/Library/Application Support/tesse-ract/agent/conversations/
/// ├── index.json              ← lightweight summaries for fast startup
/// ├── {uuid}.json             ← full conversation data
/// └── ...
/// ```
@MainActor
final class AgentConversationStore: ObservableObject {

    @Published private(set) var conversations: [AgentConversationSummary] = []
    @Published private(set) var currentConversation: AgentConversation?

    private let conversationsDir: URL

    init() {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        conversationsDir = appSupport
            .appendingPathComponent("tesse-ract/agent/conversations", isDirectory: true)

        ensureDirectory()
        loadIndex()
    }

    // MARK: - Public API

    /// Creates a fresh conversation, saves any existing current, and sets it as current.
    @discardableResult
    func createNew() -> AgentConversation {
        if let current = currentConversation, !current.messages.isEmpty {
            save(current)
        }
        let conversation = AgentConversation()
        currentConversation = conversation
        return conversation
    }

    /// Loads a conversation from disk by ID and sets it as current.
    func load(id: UUID) {
        if let current = currentConversation, !current.messages.isEmpty {
            save(current)
        }
        guard let conversation = loadFromDisk(id: id) else {
            Log.agent.error("Failed to load conversation \(id)")
            return
        }
        currentConversation = conversation
    }

    /// Saves a conversation to disk and updates the index.
    func save(_ conversation: AgentConversation) {
        // Only persist conversations that have user messages
        let hasUserMessages = conversation.messages.contains { $0.role == .user }
        guard hasUserMessages else { return }

        var updated = conversation
        updated.updatedAt = Date()

        saveToDisk(updated)

        // Update index
        let summary = AgentConversationSummary(from: updated)
        if let idx = conversations.firstIndex(where: { $0.id == updated.id }) {
            conversations[idx] = summary
        } else {
            conversations.insert(summary, at: 0)
        }
        saveIndex()

        // Keep currentConversation in sync
        if currentConversation?.id == updated.id {
            currentConversation = updated
        }
    }

    /// Deletes a conversation from disk and removes from index.
    func delete(id: UUID) {
        let fileURL = conversationFileURL(for: id)
        try? FileManager.default.removeItem(at: fileURL)

        conversations.removeAll { $0.id == id }
        saveIndex()

        if currentConversation?.id == id {
            currentConversation = AgentConversation()
        }
    }

    /// Saves the current conversation (convenience for coordinator).
    func saveCurrent() {
        guard let current = currentConversation else { return }
        save(current)
    }

    /// Updates the current conversation's messages in memory (caller is responsible for saving).
    func updateCurrentMessages(_ messages: [AgentChatMessage]) {
        currentConversation?.messages = messages
    }

    // MARK: - Private

    private func ensureDirectory() {
        try? FileManager.default.createDirectory(at: conversationsDir, withIntermediateDirectories: true)
    }

    private func conversationFileURL(for id: UUID) -> URL {
        conversationsDir.appendingPathComponent("\(id.uuidString).json")
    }

    private var indexURL: URL {
        conversationsDir.appendingPathComponent("index.json")
    }

    // MARK: Index

    private func loadIndex() {
        guard FileManager.default.fileExists(atPath: indexURL.path) else {
            // First launch or no conversations yet
            return
        }
        do {
            let data = try Data(contentsOf: indexURL)
            conversations = try JSONDecoder().decode([AgentConversationSummary].self, from: data)
            conversations.sort { $0.updatedAt > $1.updatedAt }
        } catch {
            Log.agent.error("Failed to load conversation index: \(error)")
        }
    }

    private func saveIndex() {
        do {
            let data = try JSONEncoder().encode(conversations)
            try data.write(to: indexURL, options: .atomic)
        } catch {
            Log.agent.error("Failed to save conversation index: \(error)")
        }
    }

    // MARK: Conversation Files

    private func saveToDisk(_ conversation: AgentConversation) {
        do {
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            let data = try encoder.encode(conversation)
            try data.write(to: conversationFileURL(for: conversation.id), options: .atomic)
        } catch {
            Log.agent.error("Failed to save conversation \(conversation.id): \(error)")
        }
    }

    private func loadFromDisk(id: UUID) -> AgentConversation? {
        let fileURL = conversationFileURL(for: id)
        guard FileManager.default.fileExists(atPath: fileURL.path) else { return nil }
        do {
            let data = try Data(contentsOf: fileURL)
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            return try decoder.decode(AgentConversation.self, from: data)
        } catch {
            Log.agent.error("Failed to load conversation \(id): \(error)")
            return nil
        }
    }

    /// Loads the most recent conversation on startup (or creates a fresh one).
    func loadMostRecent() {
        guard let mostRecent = conversations.first else {
            currentConversation = AgentConversation()
            return
        }
        if let conversation = loadFromDisk(id: mostRecent.id) {
            currentConversation = conversation
        } else {
            currentConversation = AgentConversation()
        }
    }
}
