//
//  BackgroundSessionStore.swift
//  tesseract
//

import Foundation

/// Persists background agent sessions as JSON files. Accessed from `SchedulingActor`
/// which runs off-MainActor, so this is an `actor` (not `@MainActor`).
///
/// Storage layout:
/// ```
/// ~/Library/Application Support/Tesseract Agent/agent/background-sessions/
/// ├── .storage_version
/// ├── index.json
/// ├── {session-uuid}.json
/// ```
actor BackgroundSessionStore {

    static let storageVersion = 1

    private let baseDir: URL
    private var sessions: [BackgroundSessionSummary] = []

    // MARK: - Init

    init() {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first ?? FileManager.default.temporaryDirectory
        let dir = appSupport
            .appendingPathComponent("Tesseract Agent/agent/background-sessions", isDirectory: true)
        baseDir = dir
        sessions = Self.bootstrap(baseDir: dir)
    }

    /// Test-friendly initializer that uses a custom base directory.
    init(baseDirectory: URL) {
        baseDir = baseDirectory
        sessions = Self.bootstrap(baseDir: baseDirectory)
    }

    /// Runs the filesystem bootstrap sequence (create dirs, version check, index load).
    /// Static + nonisolated so it can be called from the synchronous actor `init`.
    private static func bootstrap(baseDir: URL) -> [BackgroundSessionSummary] {
        // Ensure directory
        try? FileManager.default.createDirectory(at: baseDir, withIntermediateDirectories: true)

        // Version check
        let versionURL = baseDir.appendingPathComponent(".storage_version")
        let currentOnDisk = (try? String(contentsOf: versionURL, encoding: .utf8))
            .flatMap(Int.init) ?? 0

        if currentOnDisk != storageVersion {
            Log.agent.info(
                "Background session storage version mismatch (disk=\(currentOnDisk), app=\(storageVersion)) — clearing data"
            )
            try? FileManager.default.removeItem(at: baseDir)
            try? FileManager.default.createDirectory(at: baseDir, withIntermediateDirectories: true)
            try? String(storageVersion).write(to: versionURL, atomically: true, encoding: .utf8)
            return []
        }

        // Load index
        let indexURL = baseDir.appendingPathComponent("index.json")
        guard FileManager.default.fileExists(atPath: indexURL.path) else { return [] }
        do {
            let data = try Data(contentsOf: indexURL)
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            let index = try decoder.decode(BackgroundSessionIndex.self, from: data)

            let valid = index.sessions.filter { summary in
                let fileURL = baseDir.appendingPathComponent("\(summary.id.uuidString).json")
                guard let data = try? Data(contentsOf: fileURL) else { return false }
                return (try? decoder.decode(BackgroundSession.self, from: data)) != nil
            }

            if valid.count < index.sessions.count {
                Log.agent.info(
                    "Pruned \(index.sessions.count - valid.count) orphaned/corrupt background session(s) from index"
                )
                // Re-save pruned index
                let enc = JSONEncoder()
                enc.dateEncodingStrategy = .iso8601
                if let encoded = try? enc.encode(BackgroundSessionIndex(version: storageVersion, sessions: valid)) {
                    try? encoded.write(to: indexURL, options: .atomic)
                }
            }
            return valid
        } catch {
            Log.agent.error("Failed to decode background session index: \(error)")
            return []
        }
    }

    // MARK: - Public API

    /// Load an existing session or create a new one with empty messages.
    func loadOrCreate(
        sessionId: UUID,
        taskId: UUID,
        taskName: String,
        sessionType: SessionType
    ) -> BackgroundSession {
        if let existing = loadFromDisk(sessionId: sessionId) {
            return existing
        }
        return BackgroundSession(
            id: sessionId,
            sessionType: sessionType,
            displayName: taskName,
            taskId: taskId,
            messages: [],
            lastRunAt: nil,
            createdAt: Date()
        )
    }

    /// Atomically save session to disk and update the index.
    func save(_ session: BackgroundSession) {
        do {
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            let data = try encoder.encode(session)
            try data.write(to: sessionFileURL(for: session.id), options: .atomic)
        } catch {
            Log.agent.error("Failed to save background session \(session.id): \(error)")
            return
        }

        let summary = BackgroundSessionSummary(from: session)
        if let idx = sessions.firstIndex(where: { $0.id == session.id }) {
            sessions[idx] = summary
        } else {
            sessions.append(summary)
        }
        saveIndex()
    }

    /// Remove a session file and its index entry.
    func delete(sessionId: UUID) {
        try? FileManager.default.removeItem(at: sessionFileURL(for: sessionId))
        sessions.removeAll { $0.id == sessionId }
        saveIndex()
    }

    /// Return all session summaries from the in-memory index.
    func listAll() -> [BackgroundSessionSummary] {
        sessions
    }

    // MARK: - Private — Storage

    private var indexURL: URL {
        baseDir.appendingPathComponent("index.json")
    }

    private func sessionFileURL(for id: UUID) -> URL {
        baseDir.appendingPathComponent("\(id.uuidString).json")
    }

    private func saveIndex() {
        do {
            let index = BackgroundSessionIndex(version: Self.storageVersion, sessions: sessions)
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            let data = try encoder.encode(index)
            try data.write(to: indexURL, options: .atomic)
        } catch {
            Log.agent.error("Failed to save background session index: \(error)")
        }
    }

    private func loadFromDisk(sessionId: UUID) -> BackgroundSession? {
        let fileURL = sessionFileURL(for: sessionId)
        guard FileManager.default.fileExists(atPath: fileURL.path) else { return nil }
        do {
            let data = try Data(contentsOf: fileURL)
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            return try decoder.decode(BackgroundSession.self, from: data)
        } catch {
            Log.agent.error("Failed to load background session \(sessionId): \(error)")
            return nil
        }
    }
}
