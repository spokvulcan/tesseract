//
//  ControllableTranscribing.swift
//  tesseractTests
//
//  A `@MainActor` `Transcribing` double whose `transcribe` suspends until the
//  test resolves it — so a test can deliver a *success* (or a cancellation) at a
//  moment it chooses, e.g. *after* `cancelTranscription()`. This models an engine
//  whose recognizer ignores or races cancellation and returns a result anyway —
//  the exact condition that must NOT commit history/injection/callbacks once the
//  caller has cancelled. (The real `TranscriptionEngine` can't model this: its
//  timeout race turns a cancel into `CancellationError`, discarding any late
//  success — so the coordinator-level stale-guard is tested at the engine-facing
//  `Transcribing` seam instead.)
//

import Foundation

@testable import Tesseract_Agent

@MainActor
final class ControllableTranscribing: Transcribing {
    private var pending: CheckedContinuation<TranscriptionResult, Error>?
    private(set) var transcribeCount = 0
    private(set) var cancelCount = 0
    var result: TranscriptionResult

    init(
        result: TranscriptionResult = TranscriptionResult(
            text: "result", segments: [], language: "en", processingTime: 0)
    ) {
        self.result = result
    }

    /// `true` once a `transcribe` call is suspended awaiting resolution.
    var isAwaiting: Bool { pending != nil }

    func transcribe(_ audioData: AudioData, language: String) async throws -> TranscriptionResult {
        transcribeCount += 1
        return try await withCheckedThrowingContinuation { pending = $0 }
    }

    func cancelTranscription() { cancelCount += 1 }

    /// Resolve the in-flight `transcribe` with a success result.
    func completeWithSuccess() {
        pending?.resume(returning: result)
        pending = nil
    }

    /// Resolve the in-flight `transcribe` by throwing `CancellationError`.
    func completeWithCancellation() {
        pending?.resume(throwing: CancellationError())
        pending = nil
    }
}
