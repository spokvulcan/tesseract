//
//  TraceReplayRunner.swift
//  tesseract
//
//  CLI entry for the offline trace-replay harness (PRD #82, slice
//  #85): `--trace-replay [directory]`. Unlike the other verification
//  runners, it needs **no loaded model** — it reads the **Completion
//  Trace Log** corpus (default: the app container's
//  `Application Support/PrefixCacheTraces`), replays it through the
//  pure `TraceReplayHarness`, and writes the report to
//  `benchmark/trace-replay/latest.log` so `dev.sh trace-replay` can
//  tail it. An explicit directory argument must be readable inside
//  the sandbox.
//

import Foundation

@MainActor
struct TraceReplayRunner {
    enum Failure: Error, CustomStringConvertible {
        case emptyCorpus(directory: URL, filesScanned: Int)

        var description: String {
            switch self {
            case .emptyCorpus(let directory, let filesScanned):
                return "no replayable trace records in \(directory.path) "
                    + "(\(filesScanned) trace file(s) scanned — schema or "
                    + "block-size mismatches are skipped)"
            }
        }
    }

    private let arguments: [String]

    init(arguments: [String] = CommandLine.arguments) {
        self.arguments = arguments
    }

    func run() async throws {
        let directory = resolveDirectory()
        let files = CompletionTraceLog.traceFiles(in: directory)
        let records = CompletionTraceLog.readRecords(at: files)

        let reportDir = DebugPaths.benchmark.appendingPathComponent("trace-replay")
        try FileManager.default.createDirectory(
            at: reportDir, withIntermediateDirectories: true
        )
        let logURL = reportDir.appendingPathComponent("latest.log")

        guard !records.isEmpty else {
            let failure = Failure.emptyCorpus(
                directory: directory, filesScanned: files.count
            )
            try? (failure.description + "\n").data(using: .utf8)?.write(to: logURL)
            throw failure
        }

        let report = TraceReplayHarness.replay(records: records)
        let text =
            "corpus: \(directory.path) — \(files.count) file(s)\n"
            + TraceReplayHarness.renderText(report)
        try (text + "\n").data(using: .utf8)!.write(to: logURL)
        Log.server.info(text)
    }

    /// The directory after `--trace-replay`, when one is given;
    /// otherwise the live corpus home.
    private func resolveDirectory() -> URL {
        if let index = arguments.firstIndex(of: "--trace-replay"),
            index + 1 < arguments.count,
            !arguments[index + 1].hasPrefix("--")
        {
            return URL(fileURLWithPath: arguments[index + 1], isDirectory: true)
        }
        return CompletionTraceLog.defaultDirectory
    }
}
