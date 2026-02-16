import Foundation
import os

/// Thread-safe JSON file I/O for all agent tool domains.
///
/// Storage root: `~/Library/Application Support/tesse-ract/agent/`
actor AgentDataStore {
    private let baseDir: URL
    private nonisolated let logger = Logger(subsystem: "com.tesseract.app", category: "agent")

    init() {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first ?? FileManager.default.temporaryDirectory
        baseDir = appSupport.appendingPathComponent("tesse-ract/agent", isDirectory: true)
        try? FileManager.default.createDirectory(at: baseDir, withIntermediateDirectories: true)
    }

    // MARK: - Generic Load / Save

    func load<T: Decodable>(_ type: T.Type, from filename: String) -> T? {
        let url = baseDir.appendingPathComponent(filename)
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        do {
            let data = try Data(contentsOf: url)
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            return try decoder.decode(T.self, from: data)
        } catch {
            logger.error("Failed to load \(filename): \(error)")
            return nil
        }
    }

    func save<T: Encodable>(_ value: T, to filename: String) {
        let url = baseDir.appendingPathComponent(filename)
        do {
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(value)
            try data.write(to: url, options: .atomic)
        } catch {
            logger.error("Failed to save \(filename): \(error)")
        }
    }

    // MARK: - Convenience for Array-Based Domains

    func loadArray<T: Decodable>(_ type: T.Type, from filename: String) -> [T] {
        load([T].self, from: filename) ?? []
    }

    func append<T: Codable>(_ item: T, to filename: String) {
        var items = loadArray(T.self, from: filename)
        items.append(item)
        save(items, to: filename)
    }
}
