//
//  BrowserMCPTelemetryLog.swift
//  tesseract
//
//  Durable JSONL appender for Browser MCP telemetry events (ADR-0031).
//  One file per day under
//  `Application Support/BrowserMCPTelemetry/mcp-<yyyy-MM-dd>.jsonl`;
//  every file opens with a `header` line carrying the schema version so
//  readers gate before analyzing.
//
//  The file machinery (serial-queue confinement, day roll, size-cap
//  rotation, retention pruning, failing-disk handle drop) is
//  `RotatingJSONLWriter`; this type owns the telemetry contract: the
//  header, the event encoding, and the readers.
//

import Foundation

nonisolated final class BrowserMCPTelemetryLog: @unchecked Sendable {
    /// Rotation cap per day file. Tool-call events are ~0.5–4 KB (text
    /// previews and arguments are capped), so this holds weeks of heavy
    /// browsing before rotating once to `.old`.
    static let maxFileBytes = 32 * 1024 * 1024

    /// Day files kept on disk; older ones are pruned at day-roll so the
    /// corpus stays bounded. A month of tool-usage data is comfortably
    /// more than a tuning pass needs.
    static let retainedDayFiles = 30

    private let writer: RotatingJSONLWriter

    private static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return encoder
    }()

    /// Default durable home. Application Support (not the tmp debug
    /// root) — usage analysis spans sessions, so the OS must not
    /// garbage-collect it. Diverts to an isolated per-process directory
    /// under a test runner (issue #159).
    static var defaultDirectory: URL {
        TelemetryEnvironment.durableDirectory(component: "BrowserMCPTelemetry")
    }

    let directory: URL

    init(directory: URL = BrowserMCPTelemetryLog.defaultDirectory) {
        self.directory = directory
        writer = RotatingJSONLWriter(
            directory: directory,
            queueLabel: "app.tesseract.agent.browser-mcp-telemetry",
            filenamePrefix: "mcp-",
            maxFileBytes: Self.maxFileBytes,
            retainedDayFiles: Self.retainedDayFiles,
            freshFilePreamble: {
                try? Self.encoder.encode(
                    BrowserMCPTelemetryLine.header(
                        BrowserMCPTelemetryHeader(
                            schemaVersion: BrowserMCPTelemetryEvent.currentSchemaVersion,
                            serverName: MCPProtocol.serverName,
                            serverVersion: MCPProtocol.serverVersion,
                            createdAt: Date().timeIntervalSince1970
                        )
                    ))
            }
        )
    }

    /// Append one event. Callable from any thread; ordering follows
    /// enqueue order on the writer's serial queue.
    func append(_ event: BrowserMCPTelemetryEvent) {
        writer.append {
            try? Self.encoder.encode(BrowserMCPTelemetryLine.event(event))
        }
    }

    /// Barrier for tests: returns after every previously appended event
    /// is on disk.
    func flushForTesting() {
        writer.flushForTesting()
    }

    // MARK: - Reading (analysis + tests)

    /// Telemetry files in `directory`, in deterministic day order — a
    /// day's `.old` rotation ordered before its current file.
    static func telemetryFiles(in directory: URL) -> [URL] {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: directory.path)) ?? []
        func sortKey(_ name: String) -> (day: String, rank: Int) {
            name.hasSuffix(".old") ? (String(name.dropLast(4)), 0) : (name, 1)
        }
        return
            names
            .filter {
                $0.hasPrefix("mcp-") && ($0.hasSuffix(".jsonl") || $0.hasSuffix(".jsonl.old"))
            }
            .sorted { sortKey($0) < sortKey($1) }
            .map { directory.appendingPathComponent($0) }
    }

    /// Decode every line of one telemetry file. Undecodable lines are
    /// skipped (a torn final line after a crash must not condemn the
    /// corpus).
    static func readLines(at url: URL) -> [BrowserMCPTelemetryLine] {
        guard let data = try? Data(contentsOf: url),
            let text = String(data: data, encoding: .utf8)
        else { return [] }
        let decoder = JSONDecoder()
        return text.split(separator: "\n").compactMap { line in
            guard let lineData = line.data(using: .utf8) else { return nil }
            return try? decoder.decode(BrowserMCPTelemetryLine.self, from: lineData)
        }
    }

    /// All events across one or more telemetry files, in file order.
    /// The header gate drops files whose schema disagrees with the
    /// current reader.
    static func readEvents(at urls: [URL]) -> [BrowserMCPTelemetryEvent] {
        var events: [BrowserMCPTelemetryEvent] = []
        for url in urls {
            let lines = readLines(at: url)
            guard case .header(let header)? = lines.first else { continue }
            guard header.schemaVersion == BrowserMCPTelemetryEvent.currentSchemaVersion
            else { continue }
            for case .event(let event) in lines.dropFirst() {
                events.append(event)
            }
        }
        return events
    }
}
