import Combine
import Foundation
import os

// MARK: - ConversationFile

/// On-disk representation of a conversation using tagged messages.
private struct ConversationFile: Codable {
    let id: UUID
    let title: String
    let createdAt: Date
    let updatedAt: Date
    let messages: [TaggedMessage]
}

/// Persists agent conversations as JSON files using ``MessageCodecRegistry`` for
/// protocol-backed messages.
///
/// Storage layout:
/// ```
/// ~/Library/Application Support/Tesseract Agent/agent/conversations/
/// ├── index.json              ← lightweight summaries for fast startup
/// ├── {uuid}.json             ← full conversation data (tagged messages)
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
            .appendingPathComponent("Tesseract Agent/agent/conversations", isDirectory: true)

        ensureDirectory()
        loadOrResetIndex()
    }

    // MARK: - Public API

    /// Creates a fresh conversation, saves any existing current, and sets it as current.
    @discardableResult
    func createNew() -> AgentConversation {
        if let current = currentConversation, !current.messages.isEmpty {
            saveSync(current)
        }
        let conversation = AgentConversation()
        currentConversation = conversation
        return conversation
    }

    /// Loads a conversation from disk by ID and sets it as current.
    func load(id: UUID) {
        if let current = currentConversation, !current.messages.isEmpty {
            saveSync(current)
        }
        guard let conversation = loadFromDiskSync(id: id) else {
            Log.agent.error("Failed to load conversation \(id)")
            return
        }
        currentConversation = conversation
    }

    /// Saves a conversation to disk and updates the index.
    func save(_ conversation: AgentConversation) {
        saveSync(conversation)
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
    func updateCurrentMessages(_ messages: [any AgentMessageProtocol & Sendable]) {
        currentConversation?.messages = messages
    }

    /// Loads the most recent conversation on startup (or creates a fresh one).
    func loadMostRecent() {
        guard let mostRecent = conversations.first else {
            currentConversation = AgentConversation()
            return
        }
        if let conversation = loadFromDiskSync(id: mostRecent.id) {
            currentConversation = conversation
        } else {
            currentConversation = AgentConversation()
        }
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

    /// Loads the index. If the index decodes as new-format (iso8601 dates), filters out
    /// entries whose backing files are missing/corrupt and keeps the rest. Only wipes the
    /// directory when the index itself is undecodable (pre-redesign data).
    private func loadOrResetIndex() {
        guard FileManager.default.fileExists(atPath: indexURL.path) else {
            return
        }
        do {
            let data = try Data(contentsOf: indexURL)
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            let summaries = try decoder.decode([AgentConversationSummary].self, from: data)

            // Filter to only entries whose backing file exists and decodes.
            let valid = summaries.filter { canLoadNewFormat(id: $0.id) }
            conversations = valid.sorted { $0.updatedAt > $1.updatedAt }

            // Rewrite index if we pruned any corrupt/missing entries.
            if valid.count < summaries.count {
                Log.agent.info("Pruned \(summaries.count - valid.count) unreadable conversation(s) from index")
                saveIndex()
            }
            return
        } catch {
            // Index doesn't decode with .iso8601 — likely old format
        }

        // Old-format index — clear the entire conversations directory
        Log.agent.info("Clearing pre-redesign conversation data")
        try? FileManager.default.removeItem(at: conversationsDir)
        ensureDirectory()
        conversations = []
    }

    /// Checks whether a conversation file is in the new tagged format.
    private func canLoadNewFormat(id: UUID) -> Bool {
        let fileURL = conversationFileURL(for: id)
        guard let data = try? Data(contentsOf: fileURL) else { return false }
        do {
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            _ = try decoder.decode(ConversationFile.self, from: data)
            return true
        } catch {
            return false
        }
    }

    private func saveIndex() {
        do {
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            let data = try encoder.encode(conversations)
            try data.write(to: indexURL, options: .atomic)
        } catch {
            Log.agent.error("Failed to save conversation index: \(error)")
        }
    }

    // MARK: Save/Load — Synchronous (MessageCodecRegistry is sync-safe for encode/decode)

    /// Synchronous save using direct codec calls.
    private func saveSync(_ conversation: AgentConversation) {
        // Only persist conversations that have user messages
        let hasUserMessages = conversation.messages.contains { msg in
            if let core = msg as? CoreMessage, case .user = core { return true }
            if msg is UserMessage { return true }
            if let chat = msg as? AgentChatMessage, chat.role == .user { return true }
            return false
        }
        guard hasUserMessages else { return }

        var updated = conversation
        updated.updatedAt = Date()

        saveToDiskSync(updated)

        let summary = AgentConversationSummary(from: updated)
        if let idx = conversations.firstIndex(where: { $0.id == updated.id }) {
            conversations[idx] = summary
        } else {
            conversations.insert(summary, at: 0)
        }
        saveIndex()

        if currentConversation?.id == updated.id {
            currentConversation = updated
        }
    }

    private func saveToDiskSync(_ conversation: AgentConversation) {
        do {
            let taggedMessages = try SyncMessageCodec.encodeAll(conversation.messages)
            let file = ConversationFile(
                id: conversation.id,
                title: conversation.title,
                createdAt: conversation.createdAt,
                updatedAt: conversation.updatedAt,
                messages: taggedMessages
            )
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            let data = try encoder.encode(file)
            try data.write(to: conversationFileURL(for: conversation.id), options: .atomic)
        } catch {
            Log.agent.error("Failed to save conversation \(conversation.id): \(error)")
        }
    }

    private func loadFromDiskSync(id: UUID) -> AgentConversation? {
        let fileURL = conversationFileURL(for: id)
        guard FileManager.default.fileExists(atPath: fileURL.path) else { return nil }
        do {
            let data = try Data(contentsOf: fileURL)
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            let file = try decoder.decode(ConversationFile.self, from: data)
            let messages = try SyncMessageCodec.decodeAll(file.messages)
            return AgentConversation(
                id: file.id,
                messages: messages,
                createdAt: file.createdAt,
                updatedAt: file.updatedAt
            )
        } catch {
            Log.agent.error("Failed to load conversation \(id): \(error)")
            return nil
        }
    }
}
