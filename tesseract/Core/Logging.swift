//
//  Logging.swift
//  tesseract
//

import Foundation
import os

/// A wrapper around `os.Logger` that logs all dynamic values as public.
///
/// Call sites use normal string interpolation: `Log.speech.info("path: \(path)")`.
/// Swift resolves `\(path)` as standard `String` interpolation first, then the
/// wrapper passes the entire string to `os.Logger` with `privacy: .public`.
/// This means `log stream` shows real values instead of `<private>`.
nonisolated struct PublicLogger: Sendable {
    private let logger: Logger

    init(subsystem: String, category: String) {
        self.logger = Logger(subsystem: subsystem, category: category)
    }

    func debug(_ message: String) {
        logger.debug("\(message, privacy: .public)")
    }

    func info(_ message: String) {
        logger.info("\(message, privacy: .public)")
    }

    func notice(_ message: String) {
        logger.notice("\(message, privacy: .public)")
    }

    func warning(_ message: String) {
        logger.warning("\(message, privacy: .public)")
    }

    func error(_ message: String) {
        logger.error("\(message, privacy: .public)")
    }

    func fault(_ message: String) {
        logger.fault("\(message, privacy: .public)")
    }
}

/// Shared debug output paths under the sandbox-safe temp directory.
nonisolated enum DebugPaths: Sendable {
    static let root: URL = FileManager.default.temporaryDirectory
        .appendingPathComponent("tesseract-debug")
    static let agent: URL = root.appendingPathComponent("agent")
    static let benchmark: URL = root.appendingPathComponent("benchmark")

    /// Cached formatter for timestamped directory names (`2026-03-04_001629`).
    static let timestampFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd_HHmmss"
        return f
    }()

    static func timestamp() -> String {
        timestampFormatter.string(from: Date())
    }
}

nonisolated enum Log: Sendable {
    static let audio = PublicLogger(subsystem: "app.tesseract.agent", category: "audio")
    static let transcription = PublicLogger(subsystem: "app.tesseract.agent", category: "transcription")
    static let general = PublicLogger(subsystem: "app.tesseract.agent", category: "general")
    static let speech = PublicLogger(subsystem: "app.tesseract.agent", category: "speech")
    static let image = PublicLogger(subsystem: "app.tesseract.agent", category: "image")
    static let agent = PublicLogger(subsystem: "app.tesseract.agent", category: "agent")
    static let server = PublicLogger(subsystem: "app.tesseract.agent", category: "server")
}

/// Chat view performance profiling — signposts visible in Instruments, logs to console.
/// Category: "ChatViewPerf" — filter in Instruments → os_signpost.
nonisolated enum ChatViewPerf: Sendable {
    static let log = OSLog(subsystem: "app.tesseract.agent", category: "ChatViewPerf")
    static let signposter = OSSignposter(logHandle: log)
}
