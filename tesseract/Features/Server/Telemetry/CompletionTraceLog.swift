//
//  CompletionTraceLog.swift
//  tesseract
//
//  Durable JSONL appender for `CompletionTraceRecord`s — the replay
//  corpus (PRD #82, slice #83). One file per day under
//  `Application Support/PrefixCacheTraces/trace-<yyyy-MM-dd>.jsonl`
//  (durable across sessions, unlike the tmp-rooted diagnostics sink);
//  every file opens with a `header` line carrying the schema version
//  and the prefix-digest block size, followed by one `record` line per
//  cache-aware completion.
//
//  The file machinery (serial-queue confinement, day roll, size-cap
//  rotation, retention pruning, failing-disk handle drop) is
//  `RotatingJSONLWriter`; this type owns the corpus contract: the
//  header, the record encoding, and the readers.
//

import Foundation

nonisolated final class CompletionTraceLog: @unchecked Sendable {
    /// Rotation cap per day file. Records are ~1–4 KB, so this holds
    /// weeks of heavy agent traffic before rotating once to `.old`.
    static let maxFileBytes = 32 * 1024 * 1024

    /// Day files kept on disk; older ones are pruned at day-roll so the
    /// corpus stays bounded (worst case ~2 GB at the per-day cap,
    /// realistically tens of MB). A month of traces is comfortably more
    /// than the replay ablation needs.
    static let retainedDayFiles = 30

    private let writer: RotatingJSONLWriter

    private static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return encoder
    }()

    /// Default durable home for the corpus. Application Support (not
    /// the tmp debug root) — the harness replays traces across
    /// sessions, so the OS must not garbage-collect them.
    static var defaultDirectory: URL {
        let base =
            FileManager.default.urls(
                for: .applicationSupportDirectory, in: .userDomainMask
            ).first ?? FileManager.default.temporaryDirectory
        return base.appendingPathComponent("PrefixCacheTraces", isDirectory: true)
    }

    init(directory: URL = CompletionTraceLog.defaultDirectory) {
        writer = RotatingJSONLWriter(
            directory: directory,
            queueLabel: "app.tesseract.agent.completion-trace-log",
            filenamePrefix: "trace-",
            maxFileBytes: Self.maxFileBytes,
            retainedDayFiles: Self.retainedDayFiles,
            freshFilePreamble: {
                // A fresh (or previously empty) file opens with the
                // corpus contract so readers can gate on schema before
                // replaying.
                try? Self.encoder.encode(
                    CompletionTraceLine.header(
                        CompletionTraceHeader(
                            schemaVersion: CompletionTraceRecord.currentSchemaVersion,
                            blockSize: TraceBlockDigest.blockSize,
                            createdAt: Date().timeIntervalSinceReferenceDate
                        )
                    ))
            }
        )
    }

    /// Append one record. Callable from any thread; ordering follows
    /// enqueue order on the writer's serial queue.
    func append(_ record: CompletionTraceRecord) {
        writer.append {
            try? Self.encoder.encode(CompletionTraceLine.record(record))
        }
    }

    /// Barrier for tests: returns after every previously appended
    /// record is on disk.
    func flushForTesting() {
        writer.flushForTesting()
    }

    // MARK: - Reading (harness + tests)

    /// Trace files in `directory`, in deterministic day order. A
    /// size-cap rotation moves a day's earlier records to `.jsonl.old`
    /// (each starts with its own header line), so both shapes are part
    /// of the corpus — the `.old` file ordered before that day's
    /// current file. Pairs with the writer's `trace-` filename prefix
    /// in `init` — the replay CLI resolves the corpus through this,
    /// never by hand-matching filenames.
    static func traceFiles(in directory: URL) -> [URL] {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: directory.path)) ?? []
        func sortKey(_ name: String) -> (day: String, rank: Int) {
            name.hasSuffix(".old") ? (String(name.dropLast(4)), 0) : (name, 1)
        }
        return
            names
            .filter {
                $0.hasPrefix("trace-") && ($0.hasSuffix(".jsonl") || $0.hasSuffix(".jsonl.old"))
            }
            .sorted { sortKey($0) < sortKey($1) }
            .map { directory.appendingPathComponent($0) }
    }

    /// Decode every line of one trace file. Undecodable lines are
    /// skipped (a torn final line after a crash must not condemn the
    /// corpus).
    static func readLines(at url: URL) -> [CompletionTraceLine] {
        guard let data = try? Data(contentsOf: url),
            let text = String(data: data, encoding: .utf8)
        else { return [] }
        let decoder = JSONDecoder()
        return text.split(separator: "\n").compactMap { line in
            guard let lineData = line.data(using: .utf8) else { return nil }
            return try? decoder.decode(CompletionTraceLine.self, from: lineData)
        }
    }

    /// All records across one or more trace files, in file order. The
    /// header gate drops files whose schema or block size disagrees
    /// with the current reader.
    static func readRecords(at urls: [URL]) -> [CompletionTraceRecord] {
        var records: [CompletionTraceRecord] = []
        for url in urls {
            let lines = readLines(at: url)
            guard case .header(let header)? = lines.first else { continue }
            guard header.schemaVersion == CompletionTraceRecord.currentSchemaVersion,
                header.blockSize == TraceBlockDigest.blockSize
            else { continue }
            for case .record(let record) in lines.dropFirst() {
                records.append(record)
            }
        }
        return records
    }
}
