//
//  CompanionFlightRecorder.swift
//  tesseract
//
//  The Companion's interaction-fact log (#326): app-owned, append-only,
//  retention forever. Written only by app code at the loop's decision seams;
//  the model reads through a typed tool and may add testimony via
//  `log_feedback` — it can never edit history. This record is the trust
//  contract's enforcement backbone (#309) and the acceptance measurement
//  substrate (#313), which is exactly why it is not model-writable and not
//  inside the agent's file sandbox.
//
//  Storage: `Application Support/CompanionFlightRecorder/` (tests divert via
//  `TelemetryEnvironment`), daily files, crash-safe appends, pruning
//  disabled — a few MB/year is the cheapest evidence there is.
//

import Foundation

nonisolated final class CompanionFlightRecorder: Sendable {

    let directory: URL
    private let writer: RotatingJSONLWriter

    init(directory: URL? = nil) {
        let home =
            directory
            ?? TelemetryEnvironment.durableDirectory(component: "CompanionFlightRecorder")
        self.directory = home
        self.writer = RotatingJSONLWriter(
            directory: home,
            queueLabel: "companion.flight-recorder",
            filenamePrefix: "flight-",
            maxFileBytes: 32 * 1024 * 1024,
            retainedDayFiles: nil,  // retention forever (#326)
            freshFilePreamble: {
                try? JSONEncoder().encode(
                    CompanionTraceHeader(
                        schemaVersion: CompanionTraceRecord.currentSchemaVersion,
                        createdAt: Date().timeIntervalSince1970
                    ))
            }
        )
    }

    // MARK: - Writing (app code only)

    func record(
        _ event: String,
        source: CompanionTraceSource = .appObserved,
        at timestamp: Date = Date(),
        wakeID: UUID? = nil,
        turnID: UUID? = nil,
        conversationID: UUID? = nil,
        policyVersion: String? = nil,
        modelID: String? = nil,
        snapshot: [String: String]? = nil,
        note: String? = nil
    ) {
        let line = CompanionTraceRecord(
            ts: timestamp.timeIntervalSince1970,
            event: event,
            source: source,
            wakeID: wakeID?.uuidString,
            turnID: turnID?.uuidString,
            conversationID: conversationID?.uuidString,
            policyVersion: policyVersion,
            modelID: modelID,
            snapshot: snapshot,
            note: note
        )
        writer.append(timestamp: timestamp) { try? JSONEncoder().encode(line) }
    }

    /// Test barrier — every queued line is on disk when this returns.
    func flushForTesting() { writer.flushForTesting() }

    // MARK: - Reading (the typed tool and the aggregator come through here)

    /// All records within the window, oldest first. File-per-day makes the
    /// scan cheap; the schema guard skips files a future version can't read.
    func records(since: Date, until: Date = Date()) -> [CompanionTraceRecord] {
        flushForTesting()
        guard
            let files = try? FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: nil)
        else { return [] }
        var out: [CompanionTraceRecord] = []
        for url in files.sorted(by: { $0.lastPathComponent < $1.lastPathComponent })
        where url.pathExtension == "jsonl" || url.lastPathComponent.hasSuffix(".jsonl.old") {
            guard let data = try? Data(contentsOf: url) else { continue }
            for chunk in data.split(separator: 0x0A) {
                guard let line = CompanionTraceLine.decode(Data(chunk)) else { continue }
                if case .record(let record) = line,
                    record.ts >= since.timeIntervalSince1970,
                    record.ts <= until.timeIntervalSince1970
                {
                    out.append(record)
                }
            }
        }
        return out.sorted { $0.ts < $1.ts }
    }

    // MARK: - v0 import (#326 cutover)

    /// One-time cutover: the walking skeleton's `heartbeat.jsonl` lines become
    /// historical `beat.*` records — the lived evidence belongs in the one
    /// corpus — and the source file is renamed `.imported` so this never runs
    /// twice. Safe to call every launch.
    func importV0IfNeeded(from url: URL) {
        guard FileManager.default.fileExists(atPath: url.path) else { return }
        guard let data = try? Data(contentsOf: url), !data.isEmpty else {
            try? FileManager.default.moveItem(
                at: url, to: url.appendingPathExtension("imported"))
            return
        }
        let decoder = JSONDecoder()
        var imported = 0
        for chunk in data.split(separator: 0x0A) {
            guard let entry = try? decoder.decode(V0Entry.self, from: Data(chunk)) else {
                continue
            }
            let ts = V0Entry.parseTimestamp(entry.ts) ?? Date()
            var snapshot: [String: String] = [:]
            if let beat = entry.beat { snapshot["beat"] = beat }
            if let trigger = entry.trigger { snapshot["trigger"] = trigger }
            if let scheduledFor = entry.scheduledFor { snapshot["scheduledFor"] = scheduledFor }
            if let late = entry.lateSeconds { snapshot["lateSeconds"] = String(late) }
            record(
                "beat.\(entry.event)",
                at: ts,
                wakeID: entry.ping.flatMap(UUID.init(uuidString:)),
                snapshot: snapshot.isEmpty ? nil : snapshot,
                note: entry.note
            )
            imported += 1
        }
        flushForTesting()
        try? FileManager.default.moveItem(at: url, to: url.appendingPathExtension("imported"))
        Log.companion.info("Flight recorder imported \(imported) v0 heartbeat lines")
    }

    /// The skeleton's v0 line shape (`CompanionHeartbeatLog.Entry`).
    private struct V0Entry: Codable {
        let ts: String
        let event: String
        let beat: String?
        let ping: String?
        let scheduledFor: String?
        let lateSeconds: Int?
        let trigger: String?
        let note: String?

        static func parseTimestamp(_ raw: String) -> Date? {
            let formatter = ISO8601DateFormatter()
            formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
            if let date = formatter.date(from: raw) { return date }
            formatter.formatOptions = [.withInternetDateTime]
            return formatter.date(from: raw)
        }
    }
}
