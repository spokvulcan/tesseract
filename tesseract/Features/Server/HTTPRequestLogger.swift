import Foundation
import os

/// File-based debug logger for `/v1/chat/completions` request bodies.
///
/// Writes each incoming request body to a numbered JSON file under:
/// `tmp/tesseract-debug/http-completions/{HH-mm-ss}-{seq:04d}-request.json`
///
/// Useful for offline investigation of prefix cache misses, message
/// normalization drift, and client format differences. Logging is best-effort —
/// failures are silently ignored.
nonisolated struct HTTPRequestLogger: Sendable {

    static let shared = HTTPRequestLogger()

    private let rootURL: URL
    private let sequence = OSAllocatedUnfairLock(initialState: 0)
    private let logger = Logger(subsystem: "app.tesseract.agent", category: "server")

    init() {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("tesseract-debug", isDirectory: true)
            .appendingPathComponent("http-completions", isDirectory: true)
        self.rootURL = tmp

        try? FileManager.default.createDirectory(
            at: tmp, withIntermediateDirectories: true
        )
    }

    /// Returns a filename prefix `{HH-mm-ss}-{seq:04d}`.
    private func filenamePrefix() -> String {
        let seq = sequence.withLock { seq in
            seq += 1
            return seq
        }
        let formatter = DateFormatter()
        formatter.dateFormat = "HH-mm-ss"
        return String(format: "%@-%04d", formatter.string(from: Date()), seq)
    }

    /// Write a request body (raw JSON bytes) to disk under a new numbered file.
    /// Returns the prefix used, so the caller can correlate logs.
    @discardableResult
    func logRequest(body: Data, sessionAffinity: String?) -> String {
        let prefix = filenamePrefix()
        let url = rootURL.appendingPathComponent("\(prefix)-request.json")
        var header = "// session=\(sessionAffinity ?? "nil")\n".data(using: .utf8) ?? Data()
        if let parsed = try? JSONSerialization.jsonObject(with: body),
           let pretty = try? JSONSerialization.data(
               withJSONObject: parsed,
               options: [.prettyPrinted, .sortedKeys])
        {
            header.append(pretty)
        } else {
            header.append(body)
        }
        do {
            try header.write(to: url)
        } catch {
            logger.warning("HTTPRequestLogger write failed: \(error.localizedDescription, privacy: .public)")
        }
        return prefix
    }

    var directoryURL: URL { rootURL }
}
