//
//  Logging.swift
//  tesseract
//

import os

/// A wrapper around `os.Logger` that logs all dynamic values as public.
///
/// Call sites use normal string interpolation: `Log.speech.info("path: \(path)")`.
/// Swift resolves `\(path)` as standard `String` interpolation first, then the
/// wrapper passes the entire string to `os.Logger` with `privacy: .public`.
/// This means `log stream` shows real values instead of `<private>`.
struct PublicLogger {
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

enum Log {
    static let audio = PublicLogger(subsystem: "com.tesseract.app", category: "audio")
    static let transcription = PublicLogger(subsystem: "com.tesseract.app", category: "transcription")
    static let general = PublicLogger(subsystem: "com.tesseract.app", category: "general")
    static let speech = PublicLogger(subsystem: "com.tesseract.app", category: "speech")
    static let image = PublicLogger(subsystem: "com.tesseract.app", category: "image")
}
