//
//  CompanionTrace.swift
//  tesseract
//
//  The Companion Trace: an append-only JSONL record of every Jarvis decision,
//  card, reaction and agenda change, for analysing later how well he works.
//  App code writes it; the model never does. One file per day, kept
//  indefinitely, under Application Support (tests divert through
//  `TelemetryEnvironment`). Every record carries the Day Thread it belongs
//  to, so it can be read against that day's transcript.
//
//  Reading it:
//
//      jq -c 'select(.event == "moment.finished") | .fields' \
//        ~/Library/Application\ Support/CompanionTrace/trace-*.jsonl
//

import Foundation

// MARK: - Values

/// One field value. Encoded as a plain JSON scalar, so `jq` and DuckDB read
/// numbers as numbers.
nonisolated enum CompanionTraceValue: Codable, Sendable, Equatable {
    case string(String)
    case int(Int)
    case double(Double)
    case bool(Bool)

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Int.self) {
            self = .int(value)
        } else if let value = try? container.decode(Double.self) {
            self = .double(value)
        } else {
            self = .string(try container.decode(String.self))
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value): try container.encode(value)
        case .int(let value): try container.encode(value)
        case .double(let value): try container.encode(value)
        case .bool(let value): try container.encode(value)
        }
    }

    var stringValue: String? {
        if case .string(let value) = self { return value }
        return nil
    }

    var intValue: Int? {
        if case .int(let value) = self { return value }
        return nil
    }
}

extension CompanionTraceValue: ExpressibleByStringLiteral, ExpressibleByIntegerLiteral,
    ExpressibleByFloatLiteral, ExpressibleByBooleanLiteral
{
    nonisolated init(stringLiteral value: String) { self = .string(value) }
    nonisolated init(integerLiteral value: Int) { self = .int(value) }
    nonisolated init(floatLiteral value: Double) { self = .double(value) }
    nonisolated init(booleanLiteral value: Bool) { self = .bool(value) }
}

// MARK: - Lines

nonisolated struct CompanionTraceHeader: Codable, Sendable {
    let schemaVersion: Int
    let createdAt: TimeInterval
}

nonisolated struct CompanionTraceRecord: Codable, Sendable, Equatable {
    static let currentSchemaVersion = 2

    /// Unix seconds.
    let ts: TimeInterval
    /// A `CompanionTraceEvent` raw value.
    let event: String
    /// The Day Thread the event belongs to: its `DayKey`.
    let thread: String
    /// The conversation the event concerns, when there is one.
    let conversationID: String?
    let fields: [String: CompanionTraceValue]?

    var traceEvent: CompanionTraceEvent? { CompanionTraceEvent(rawValue: event) }
}

nonisolated enum CompanionTraceLine: Sendable {
    case header(CompanionTraceHeader)
    case record(CompanionTraceRecord)

    static func decode(_ data: Data, decoder: JSONDecoder = JSONDecoder()) -> CompanionTraceLine? {
        if let record = try? decoder.decode(CompanionTraceRecord.self, from: data) {
            return .record(record)
        }
        if let header = try? decoder.decode(CompanionTraceHeader.self, from: data) {
            return .header(header)
        }
        return nil
    }
}

// MARK: - Writer

nonisolated final class CompanionTrace: Sendable {

    let directory: URL
    private let writer: RotatingJSONLWriter
    private let calendar: Calendar

    init(directory: URL? = nil, calendar: Calendar = .current) {
        let home = directory ?? TelemetryEnvironment.durableDirectory(component: "CompanionTrace")
        self.directory = home
        self.calendar = calendar
        self.writer = RotatingJSONLWriter(
            directory: home,
            queueLabel: "companion.trace",
            filenamePrefix: "trace-",
            maxFileBytes: 64 * 1024 * 1024,
            retainedDayFiles: nil,
            freshFilePreamble: {
                try? JSONEncoder().encode(
                    CompanionTraceHeader(
                        schemaVersion: CompanionTraceRecord.currentSchemaVersion,
                        createdAt: Date().timeIntervalSince1970))
            }
        )
    }

    /// Append one event. The Day Thread comes from the timestamp.
    func record(
        _ event: CompanionTraceEvent,
        at timestamp: Date = Date(),
        conversationID: UUID? = nil,
        fields: [String: CompanionTraceValue] = [:]
    ) {
        let record = CompanionTraceRecord(
            ts: timestamp.timeIntervalSince1970,
            event: event.rawValue,
            thread: DayKey(for: timestamp, calendar: calendar).rawValue,
            conversationID: conversationID?.uuidString,
            fields: fields.isEmpty ? nil : fields)
        writer.append(timestamp: timestamp) { try? JSONEncoder().encode(record) }
    }

    /// String-valued fields, the shape the voice session machine emits.
    func record(
        _ event: CompanionTraceEvent,
        conversationID: UUID?,
        snapshot: [String: String]
    ) {
        record(event, conversationID: conversationID, fields: snapshot.mapValues { .string($0) })
    }

    /// Test barrier: every queued line is on disk when this returns.
    func flushForTesting() { writer.flushForTesting() }

    /// All records within the window, oldest first.
    func records(since: Date, until: Date = Date()) -> [CompanionTraceRecord] {
        writer.flushForTesting()
        guard
            let files = try? FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: nil)
        else { return [] }
        let decoder = JSONDecoder()
        var out: [CompanionTraceRecord] = []
        for url in files.sorted(by: { $0.lastPathComponent < $1.lastPathComponent })
        where url.pathExtension == "jsonl" || url.lastPathComponent.hasSuffix(".jsonl.old") {
            guard let data = try? Data(contentsOf: url) else { continue }
            for chunk in data.split(separator: 0x0A) {
                guard
                    case .record(let record) = CompanionTraceLine.decode(
                        Data(chunk), decoder: decoder),
                    record.ts >= since.timeIntervalSince1970,
                    record.ts <= until.timeIntervalSince1970
                else { continue }
                out.append(record)
            }
        }
        return out.sorted { $0.ts < $1.ts }
    }
}
