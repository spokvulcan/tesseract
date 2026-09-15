//
//  ObjCExceptions.swift
//  tesseract
//

import Foundation

/// Runs Objective-C code that can raise an `NSException` and converts the
/// exception into a thrown Swift error at the call site.
///
/// Why: an `NSException` that escapes a Swift concurrency job unwinds through
/// the Swift runtime without its cleanups, AppKit swallows it at the event
/// loop, and the main thread's executor-tracking slot is left pointing at a
/// dead stack frame. The next compiler-inserted `@MainActor` check on that
/// thread segfaults somewhere unrelated (2026-09-15: `AVAudioEngine` raised
/// inside `AudioCaptureEngine.prewarm()`; the app died 21 s later entering the
/// hotkey event-tap callback). The discipline is one `catching` per AVFAudio
/// graph-mutating step, so no such call runs outside the seam.
nonisolated enum ObjCExceptions {
    /// An `NSException` raised inside `catching`, carried as a Swift error.
    struct Raised: LocalizedError {
        let name: String
        let reason: String?

        var errorDescription: String? { "\(name): \(reason ?? "no reason")" }
    }

    /// Runs `body`, rethrowing any Swift error it throws and converting any
    /// `NSException` it raises into `Raised`.
    static func catching<T>(_ body: () throws -> T) throws -> T {
        var outcome: Result<T, any Error>!
        if let exception = ObjCExceptionCatcherRun({ outcome = Result { try body() } }) {
            throw Raised(name: exception.name.rawValue, reason: exception.reason)
        }
        return try outcome.get()
    }
}
