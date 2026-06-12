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
/// The file machinery (serial-queue confinement, day roll, size-cap rotation
/// to `.old`, failing-disk handle drop) is `RotatingJSONLWriter`; this type
/// owns the event encoding and the telemetry-sink registration.
nonisolated final class PromptCacheDiagnosticsFileSink: @unchecked Sendable {
    static let maxFileBytes = 64 * 1024 * 1024

    private let writer: RotatingJSONLWriter
    private var sinkHandle: PrefixCacheDiagnostics.TelemetrySinkHandle?

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
        writer = RotatingJSONLWriter(
            directory: directory,
            queueLabel: "app.tesseract.agent.cache-diagnostics-file-sink",
            maxFileBytes: Self.maxFileBytes
        )
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
    }

    /// Enqueue one event for appending. Called from arbitrary threads by the
    /// telemetry sink; also the direct entry point for tests.
    func record(_ event: PromptCacheTelemetryEvent) {
        writer.append(timestamp: event.timestamp) {
            try? Self.encoder.encode(event)
        }
    }

    /// Barrier for tests: returns after every previously recorded event has
    /// been written.
    func flushForTesting() {
        writer.flushForTesting()
    }
}
