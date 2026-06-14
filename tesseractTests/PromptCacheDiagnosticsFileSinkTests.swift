import Foundation
import Testing

@testable import Tesseract_Agent

struct PromptCacheDiagnosticsFileSinkTests {

    private func makeTempDirectory() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("file-sink-tests-\(UUID().uuidString)", isDirectory: true)
    }

    private func event(
        timestamp: Date,
        name: String = "lookup",
        fields: [(String, String)] = [("reason", "hit"), ("promptTokens", "1024")]
    ) -> PromptCacheTelemetryEvent {
        PromptCacheTelemetryEvent(
            timestamp: timestamp,
            scope: .request,
            eventName: name,
            requestID: UUID(),
            modelID: "qwen3.5",
            kvBits: 8,
            kvGroupSize: 64,
            fields: fields
        )
    }

    @Test func appendsOneDecodableJSONLinePerEvent() throws {
        let directory = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let sink = PromptCacheDiagnosticsFileSink(directory: directory, registerSink: false)

        let day = Date(timeIntervalSince1970: 1_750_000_000)
        sink.record(event(timestamp: day, name: "lookup"))
        sink.record(event(timestamp: day, name: "ttft", fields: [("ttftMs", "63")]))
        sink.flushForTesting()

        let files = try FileManager.default.contentsOfDirectory(atPath: directory.path)
        #expect(files.count == 1)
        #expect(files[0].hasSuffix(".jsonl"))

        let contents = try String(
            contentsOf: directory.appendingPathComponent(files[0]), encoding: .utf8)
        let lines = contents.split(separator: "\n")
        #expect(lines.count == 2)

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let decoded = try lines.map {
            try decoder.decode(PromptCacheTelemetryEvent.self, from: Data($0.utf8))
        }
        #expect(decoded[0].eventName == "lookup")
        #expect(decoded[1].eventName == "ttft")
        #expect(decoded[1].field("ttftMs") == "63")
    }

    @Test func rollsToANewFileWhenTheDayChanges() throws {
        let directory = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let sink = PromptCacheDiagnosticsFileSink(directory: directory, registerSink: false)

        let dayOne = Date(timeIntervalSince1970: 1_750_000_000)
        let dayTwo = dayOne.addingTimeInterval(60 * 60 * 24)
        sink.record(event(timestamp: dayOne))
        sink.record(event(timestamp: dayTwo))
        sink.flushForTesting()

        let files = try FileManager.default.contentsOfDirectory(atPath: directory.path).sorted()
        #expect(files.count == 2)
        #expect(files.allSatisfy { $0.hasSuffix(".jsonl") })
    }

    @Test func appendsAcrossSinkInstancesWithoutTruncating() throws {
        let directory = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let timestamp = Date(timeIntervalSince1970: 1_750_000_000)

        let first = PromptCacheDiagnosticsFileSink(directory: directory, registerSink: false)
        first.record(event(timestamp: timestamp))
        first.flushForTesting()

        let second = PromptCacheDiagnosticsFileSink(directory: directory, registerSink: false)
        second.record(event(timestamp: timestamp))
        second.flushForTesting()

        let files = try FileManager.default.contentsOfDirectory(atPath: directory.path)
        #expect(files.count == 1)
        let contents = try String(
            contentsOf: directory.appendingPathComponent(files[0]), encoding: .utf8)
        #expect(contents.split(separator: "\n").count == 2)
    }
}
