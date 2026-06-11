import Foundation

/// Always-on durable sink for prefix-cache diagnostics: every
/// `PromptCacheTelemetryEvent` is appended as one JSON line to
/// `tmp/tesseract-debug/cache-diagnostics/<yyyy-MM-dd>.jsonl`.
///
/// Exists because the unified-logging copy of these events (`Log.agent.info`)
/// lives in a memory-only buffer — after the process moves on, `log show`
/// returns nothing and a cache investigation has no per-request evidence.
/// Events carry offsets, byte counts, and reasons — never message content —
/// so an always-on file is safe.
///
/// Size cap: when the current day file would exceed `maxFileBytes`, it is
/// rotated once to `<yyyy-MM-dd>.jsonl.old` (replacing any previous rotation)
/// and writing continues on a fresh file.
///
/// All mutable state is confined to the private serial queue; the registered
/// telemetry-sink closure only hops onto it. `@unchecked Sendable` mirrors
/// the registry pattern in `PrefixCacheDiagnostics`.
nonisolated final class PromptCacheDiagnosticsFileSink: @unchecked Sendable {
    static let maxFileBytes = 64 * 1024 * 1024

    private let queue = DispatchQueue(
        label: "app.tesseract.agent.cache-diagnostics-file-sink",
        qos: .utility
    )
    private let directory: URL
    private var handle: FileHandle?
    private var currentDay: String?
    private var currentFileBytes: Int = 0
    private var sinkHandle: PrefixCacheDiagnostics.TelemetrySinkHandle?

    private static let dayFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        return formatter
    }()

    private static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }()

    init(
        directory: URL = DebugPaths.root.appendingPathComponent(
            "cache-diagnostics", isDirectory: true
        ),
        registerSink: Bool = true
    ) {
        self.directory = directory
        if registerSink {
            sinkHandle = PrefixCacheDiagnostics.addTelemetrySink { [weak self] event in
                self?.record(event)
            }
        }
    }

    deinit {
        if let sinkHandle {
            PrefixCacheDiagnostics.removeTelemetrySink(sinkHandle)
        }
        try? handle?.close()
    }

    /// Enqueue one event for appending. Called from arbitrary threads by the
    /// telemetry sink; also the direct entry point for tests.
    func record(_ event: PromptCacheTelemetryEvent) {
        queue.async { [weak self] in
            self?.append(event)
        }
    }

    /// Barrier for tests: returns after every previously recorded event has
    /// been written.
    func flushForTesting() {
        queue.sync {}
    }

    // MARK: - Queue-confined

    private func append(_ event: PromptCacheTelemetryEvent) {
        guard let data = try? Self.encoder.encode(event) else { return }
        var line = data
        line.append(0x0A)

        let day = Self.dayFormatter.string(from: event.timestamp)
        if day != currentDay {
            rollToDay(day)
        }
        if currentFileBytes + line.count > Self.maxFileBytes {
            rotateCurrentFile(day: day)
        }
        guard let handle else { return }

        do {
            try handle.write(contentsOf: line)
            currentFileBytes += line.count
        } catch {
            // A failing disk must never take diagnostics emission down with
            // it; drop the handle so the next day-roll retries cleanly.
            try? handle.close()
            self.handle = nil
        }
    }

    private func fileURL(day: String) -> URL {
        directory.appendingPathComponent("\(day).jsonl", isDirectory: false)
    }

    private func rollToDay(_ day: String) {
        try? handle?.close()
        handle = nil
        currentDay = day
        openCurrentFile(day: day)
    }

    private func rotateCurrentFile(day: String) {
        try? handle?.close()
        handle = nil
        let url = fileURL(day: day)
        let rotated = url.appendingPathExtension("old")
        try? FileManager.default.removeItem(at: rotated)
        try? FileManager.default.moveItem(at: url, to: rotated)
        openCurrentFile(day: day)
    }

    private func openCurrentFile(day: String) {
        let manager = FileManager.default
        do {
            try manager.createDirectory(at: directory, withIntermediateDirectories: true)
        } catch {
            return
        }
        let url = fileURL(day: day)
        if !manager.fileExists(atPath: url.path) {
            manager.createFile(atPath: url.path, contents: nil)
        }
        guard let opened = try? FileHandle(forWritingTo: url) else { return }
        let end = (try? opened.seekToEnd()) ?? 0
        handle = opened
        currentFileBytes = Int(end)
    }
}
