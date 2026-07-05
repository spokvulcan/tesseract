import Foundation
import Testing

@testable import Tesseract_Agent

/// Issue #159: the shared JSONL file machinery must survive a second writer
/// on the same file. Two `RotatingJSONLWriter` instances have independent
/// file handles and byte accounting — mechanically the same situation as two
/// processes (app + parallel test host) appending to one trace file, which
/// is how the 2026-07-05 production trace got a torn mid-record line.
struct RotatingJSONLWriterTests {

    private func makeDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("rotating-jsonl-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    /// Both writers pin the same day via an explicit timestamp so they
    /// contend on one file.
    private static let day = Date(timeIntervalSinceReferenceDate: 0)

    @Test func concurrentWritersOnOneFileNeverTearLines() throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let first = RotatingJSONLWriter(
            directory: directory,
            queueLabel: "test.rotating-jsonl.first",
            maxFileBytes: 64 * 1024 * 1024
        )
        let second = RotatingJSONLWriter(
            directory: directory,
            queueLabel: "test.rotating-jsonl.second",
            maxFileBytes: 64 * 1024 * 1024
        )

        let linesPerWriter = 500
        // Long enough payloads that a stale-offset overwrite would land
        // mid-line rather than exactly on a boundary.
        for index in 0..<linesPerWriter {
            first.append(timestamp: Self.day) {
                Data(
                    "{\"writer\":\"first\",\"index\":\(index),\"pad\":\"\(String(repeating: "a", count: 200))\"}"
                        .utf8)
            }
            second.append(timestamp: Self.day) {
                Data(
                    "{\"writer\":\"second\",\"index\":\(index),\"pad\":\"\(String(repeating: "b", count: 200))\"}"
                        .utf8)
            }
        }
        first.flushForTesting()
        second.flushForTesting()

        // The day-file name derives from the writer's local-timezone
        // formatter — find it rather than hardcoding a timezone assumption.
        let names = try FileManager.default.contentsOfDirectory(atPath: directory.path)
            .filter { $0.hasSuffix(".jsonl") }
        #expect(names.count == 1)
        let url = directory.appendingPathComponent(try #require(names.first))
        let contents = try String(contentsOf: url, encoding: .utf8)
        let lines = contents.split(separator: "\n", omittingEmptySubsequences: true)

        #expect(lines.count == linesPerWriter * 2)
        for line in lines {
            let object = try? JSONSerialization.jsonObject(with: Data(line.utf8))
            #expect(object != nil, "torn or corrupt line: \(line.prefix(80))")
        }

        let firstCount = lines.filter { $0.contains("\"writer\":\"first\"") }.count
        #expect(firstCount == linesPerWriter)
    }
}

/// Issue #159: durable telemetry homes divert away from the production
/// Application Support directories when running under a test host — this
/// suite *is* a test host, so the divert must be observable directly.
struct TelemetryEnvironmentTests {

    @Test func testProcessIsDetected() {
        #expect(TelemetryEnvironment.isRunningTests)
    }

    @Test func durableDirectoriesDivertAwayFromApplicationSupport() {
        let diagnostics = PromptCacheDiagnosticsFileSink.defaultDirectory
        let traces = CompletionTraceLog.defaultDirectory

        #expect(diagnostics.path.contains("TesseractTestTelemetry"))
        #expect(traces.path.contains("TesseractTestTelemetry"))
        #expect(!diagnostics.path.contains("Application Support"))
        #expect(!traces.path.contains("Application Support"))
    }
}
