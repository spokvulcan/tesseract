//
//  CompanionHeartbeatLog.swift
//  tesseract
//

import Foundation

// MARK: - CompanionHeartbeatLog

/// PROTOTYPE — the flight recorder v0 (map #301, tickets #303/#326).
///
/// Append-only JSONL, one line per heartbeat event, at
/// `companion/heartbeat.jsonl` inside the agent root — deliberately within the
/// agent's file-tool sandbox, so the owner can ask the assistant about their
/// own ping history. Two consumers by design: the assistant (read via file
/// tools) and dev mining for the architecture grillings. The full recorder
/// schema is #326's question; this stays a dumb event log until then.
///
/// Events: `fired`, `spoken`, `engaged`, `replied` (with `note`), `dismissed`,
/// `expired`, `missed`, `authDenied`. Synchronous tiny appends on the calling
/// (main) actor — fine for a handful of lines a day.
struct CompanionHeartbeatLog {

    private let fileURL: URL

    init(
        fileURL: URL = PathSandbox.defaultRoot
            .appendingPathComponent("companion", isDirectory: true)
            .appendingPathComponent("heartbeat.jsonl")
    ) {
        self.fileURL = fileURL
    }

    private struct Entry: Encodable {
        let ts: String
        let event: String
        var beat: String?
        var ping: String?
        var scheduledFor: String?
        var lateSeconds: Int?
        var trigger: String?
        var note: String?
    }

    /// Local-timezone ISO 8601 (`2026-07-12T09:00:03+02:00`) — the owner reads
    /// this file; wall-clock times are the point.
    private static let timestampFormatter: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        formatter.timeZone = .current
        return formatter
    }()

    func append(
        event: String, beat: String? = nil, ping: UUID? = nil, scheduledFor: Date? = nil,
        lateSeconds: Int? = nil, trigger: String? = nil, note: String? = nil
    ) {
        let entry = Entry(
            ts: Self.timestampFormatter.string(from: Date()),
            event: event,
            beat: beat,
            ping: ping?.uuidString,
            scheduledFor: scheduledFor.map { Self.timestampFormatter.string(from: $0) },
            lateSeconds: lateSeconds,
            trigger: trigger,
            note: note
        )
        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
            try appendLine(encoder.encode(entry))
        } catch {
            Log.companion.error("Heartbeat log append failed: \(error)")
        }
    }

    private func appendLine(_ data: Data) throws {
        let fileManager = FileManager.default
        try fileManager.createDirectory(
            at: fileURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        if !fileManager.fileExists(atPath: fileURL.path) {
            fileManager.createFile(atPath: fileURL.path, contents: nil)
        }
        let handle = try FileHandle(forWritingTo: fileURL)
        defer { try? handle.close() }
        try handle.seekToEnd()
        try handle.write(contentsOf: data + Data("\n".utf8))
    }
}
