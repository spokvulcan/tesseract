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

        // Try loading the index; fall back to scanning session files if missing or corrupt.
        // Session files are the durable source of truth — index.json is a summary cache.
        let indexURL = baseDir.appendingPathComponent("index.json")
        if let data = try? Data(contentsOf: indexURL),
           let index = try? {
               let decoder = JSONDecoder()
               decoder.dateDecodingStrategy = .iso8601
               return try decoder.decode(BackgroundSessionIndex.self, from: data)
           }() {
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            let valid = index.sessions.filter { summary in
                let fileURL = baseDir.appendingPathComponent("\(summary.id.uuidString).json")
                guard let data = try? Data(contentsOf: fileURL) else { return false }
                return (try? decoder.decode(BackgroundSession.self, from: data)) != nil
            }

            if valid.count < index.sessions.count {
                Log.agent.info(
                    "Pruned \(index.sessions.count - valid.count) orphaned/corrupt background session(s) from index"
                )
                writeIndex(summaries: valid, to: indexURL)
            }
            return valid
        }

        // Index missing or corrupt — rebuild from session files on disk
        let rebuilt = rebuildSummaries(in: baseDir)
        if !rebuilt.isEmpty {
            Log.agent.info("Rebuilt background session index from \(rebuilt.count) session file(s)")
            writeIndex(summaries: rebuilt, to: indexURL)
        }
        return rebuilt
    }

    /// Scan the directory for `{uuid}.json` session files and build summaries.
    private static func rebuildSummaries(in baseDir: URL) -> [BackgroundSessionSummary] {
        let fm = FileManager.default
        guard let contents = try? fm.contentsOfDirectory(
            at: baseDir, includingPropertiesForKeys: nil
        ) else { return [] }

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        return contents.compactMap { url -> BackgroundSessionSummary? in
            guard url.pathExtension == "json",
                  url.lastPathComponent != "index.json",
                  let data = try? Data(contentsOf: url),
                  let session = try? decoder.decode(BackgroundSession.self, from: data)
            else { return nil }
            return BackgroundSessionSummary(from: session)
        }
    }

    /// Write an index file atomically.
    private static func writeIndex(summaries: [BackgroundSessionSummary], to indexURL: URL) {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        if let data = try? encoder.encode(
            BackgroundSessionIndex(version: storageVersion, sessions: summaries)
        ) {
            try? data.write(to: indexURL, options: .atomic)
        }
    }

    // MARK: - Public API

    /// Load an existing session by UUID. Returns nil if no session file exists on disk.
    func load(sessionId: UUID) -> BackgroundSession? {
        loadFromDisk(sessionId: sessionId)
    }

    /// Load an existing session or create a new empty one for the given UUID.
    /// Callers set metadata (taskId, displayName, sessionType) on the returned value before saving.
    func loadOrCreate(sessionId: UUID) -> BackgroundSession {
        if let existing = loadFromDisk(sessionId: sessionId) {
            return existing
        }
        return BackgroundSession(
            id: sessionId,
            sessionType: .cron,
            displayName: "",
            taskId: UUID(),
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
        Self.writeIndex(summaries: sessions, to: indexURL)
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
