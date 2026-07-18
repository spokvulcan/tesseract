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
    /// Turn-class tag (#327). Optional so pre-tag files decode without a
    /// storage-version bump — a bump wipes the owner's history.
    let origin: String?
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
final class AgentConversationStore: ObservableObject, AgentConversationStoring {

    @Published private(set) var conversations: [AgentConversationSummary] = []
    @Published private(set) var currentConversation: AgentConversation?

    private let conversationsDir: URL

    /// Bump this when on-disk message schemas change in incompatible ways.
    /// Mismatched versions wipe the conversations directory on launch.
    /// Version 3: the ADR-0024 parts-based `AssistantMessage` format — v2
    /// flat-string conversations are wiped on first launch (explicit decision,
    /// no migrator).
    private static let storageVersion = 3

    /// `directory` overrides the Application Support location — the seam the
    /// wipe-on-version-mismatch tests use to run against a temp directory.
    init(directory: URL? = nil) {
        if let directory {
            conversationsDir = directory
        } else {
            let appSupport =
                FileManager.default.urls(
                    for: .applicationSupportDirectory, in: .userDomainMask
                ).first
                ?? FileManager.default.temporaryDirectory
            conversationsDir =
                appSupport
                .appendingPathComponent("Tesseract Agent/agent/conversations", isDirectory: true)
        }

        ensureDirectory()
        migrateStorageVersionIfNeeded()
        loadOrResetIndex()
    }

    // MARK: - Public API

    /// Creates a fresh conversation, saves any existing current, and sets it as current.
    @discardableResult
    func createNew() -> AgentConversation {
        saveOutgoingCurrent()
        let conversation = AgentConversation()
        currentConversation = conversation
        return conversation
    }

    /// Installs a caller-built conversation as current — the summoned-dialogue
    /// mint (ADR-0046 #372). Same switch discipline as `createNew`.
    func adopt(_ conversation: AgentConversation) {
        saveOutgoingCurrent()
        currentConversation = conversation
    }

    /// Loads a conversation from disk by ID and sets it as current. Mission
    /// Control is served from `missionControl()` — the warm cache the loop
    /// refreshes on every fold save — instead of re-parsing the all-day file.
    func load(id: UUID) {
        saveOutgoingCurrent()
        if id == AgentConversation.missionControlID {
            currentConversation = missionControl()
            return
        }
        guard let conversation = loadFromDiskSync(id: id) else {
            Log.agent.error("Failed to load conversation \(id)")
            return
        }
        currentConversation = conversation
    }

    /// The switch-away half of `createNew`/`load`. Mission Control never
    /// persists through here: the chat side holds a read snapshot of the fold,
    /// and writing it back would clobber any loop turn that appended to disk
    /// since it was opened (ADR-0046) — `save(_:)` is the fold's one write
    /// door, and it belongs to the loop.
    private func saveOutgoingCurrent() {
        guard let current = currentConversation, !current.messages.isEmpty,
            !current.isMissionControl
        else { return }
        saveSync(current)
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

        if id == AgentConversation.missionControlID {
            missionControlCache = nil
        }

        if currentConversation?.id == id {
            currentConversation = AgentConversation()
        }
    }

    /// Saves the current conversation (convenience for coordinator). Refuses
    /// the fold — the chat funnel never writes Mission Control (ADR-0046).
    func saveCurrent() {
        guard let current = currentConversation, !current.isMissionControl else { return }
        save(current)
    }

    /// Updates the current conversation's messages in memory (caller is
    /// responsible for saving). Refuses the fold, same rule as `saveCurrent`.
    func updateCurrentMessages(_ messages: [any AgentMessageProtocol & Sendable]) {
        guard currentConversation?.isMissionControl != true else { return }
        currentConversation?.messages = messages
    }

    /// Loads the most recent conversation on startup (or creates a fresh one).
    /// Filtered on `opensAtLaunch`: launch never lands inside the fold.
    func loadMostRecent() {
        guard let mostRecent = conversations.first(where: { $0.turnOrigin.opensAtLaunch })
        else {
            currentConversation = AgentConversation()
            return
        }
        if let conversation = loadFromDiskSync(id: mostRecent.id) {
            currentConversation = conversation
        } else {
            currentConversation = AgentConversation()
        }
    }

    /// Mission Control (ADR-0046): the fold's one standing conversation —
    /// never through `currentConversation`, which belongs to the chat UI. A
    /// miss (first run, owner deletion, storage wipe) re-seeds it empty under
    /// the same well-known id.
    ///
    /// Cached: the loop reloads the fold at every turn and is its only writer
    /// (the chat side is guarded), so only the first call pays the disk
    /// round-trip of a file that grows all day. `saveSync` refreshes the
    /// cache on every fold save; `delete` invalidates it.
    func missionControl() -> AgentConversation {
        if let missionControlCache { return missionControlCache }
        let loaded =
            loadFromDiskSync(id: AgentConversation.missionControlID)
            ?? AgentConversation(
                id: AgentConversation.missionControlID, origin: .missionControl)
        missionControlCache = loaded
        return loaded
    }

    private var missionControlCache: AgentConversation?

    // MARK: - Private

    private func ensureDirectory() {
        try? FileManager.default.createDirectory(
            at: conversationsDir, withIntermediateDirectories: true)
    }

    private var versionFileURL: URL {
        conversationsDir.appendingPathComponent(".storage_version")
    }

    /// Wipes conversations directory when the on-disk storage version doesn't match.
    private func migrateStorageVersionIfNeeded() {
        let currentOnDisk =
            (try? String(contentsOf: versionFileURL, encoding: .utf8))
            .flatMap(Int.init) ?? 0

        if currentOnDisk == Self.storageVersion { return }

        Log.agent.info(
            "Storage version mismatch (disk=\(currentOnDisk), app=\(Self.storageVersion)) — clearing conversations"
        )
        try? FileManager.default.removeItem(at: conversationsDir)
        ensureDirectory()
        try? String(Self.storageVersion).write(
            to: versionFileURL, atomically: true, encoding: .utf8)
        conversations = []
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

            // Filter to only entries whose backing file exists and decodes
            // (kinds exempt from the parse skip it — `validatesAtLaunch`).
            let valid = summaries.filter {
                !$0.turnOrigin.validatesAtLaunch || canLoadNewFormat(id: $0.id)
            }
            conversations = valid.sorted { $0.updatedAt > $1.updatedAt }

            // Rewrite index if we pruned any corrupt/missing entries.
            if valid.count < summaries.count {
                Log.agent.info(
                    "Pruned \(summaries.count - valid.count) unreadable conversation(s) from index")
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

    /// Whether `saveSync`/`saveCurrent` would actually persist these messages: it
    /// skips a conversation with no user message. A caller that truncates a
    /// conversation (Edit & resend) must consult this to choose delete-vs-save —
    /// writing an unpersistable head silently leaves the prior file on disk. The
    /// single source of truth for the predicate, so callers can't drift from it.
    static func persists(_ messages: [any AgentMessageProtocol & Sendable]) -> Bool {
        messages.contains { msg in
            if msg.asUser != nil { return true }
            // Legacy display-shape conversations (`asUser` doesn't reach these).
            if let chat = msg as? AgentChatMessage, chat.role == .user { return true }
            return false
        }
    }

    /// Synchronous save using direct codec calls.
    private func saveSync(_ conversation: AgentConversation) {
        // Only persist conversations that have user messages.
        guard Self.persists(conversation.messages) else { return }

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
        if updated.id == AgentConversation.missionControlID {
            missionControlCache = updated
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
                messages: taggedMessages,
                origin: conversation.origin.rawValue
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
                updatedAt: file.updatedAt,
                origin: TurnOrigin(persisted: file.origin) ?? .interactive
            )
        } catch {
            Log.agent.error("Failed to load conversation \(id): \(error)")
            return nil
        }
    }
}
