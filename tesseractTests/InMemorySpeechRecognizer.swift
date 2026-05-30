//
//  InMemorySpeechRecognizer.swift
//  tesseractTests
//
//  A hermetic, in-memory Speech Recognizer adapter for tests — a *peer
//  implementation* of `WhisperKitSpeechRecognizer`, not a mock. It returns a
//  canned `TranscriptionResult`, can be programmed with a load/transcribe error
//  or a cancellation-sensitive latency, and records what it saw (loads,
//  transcribe calls, the languages passed, cancellation arrivals, and whether an
//  over-running `transcribe` was interrupted). An actor — like the production
//  adapter — so `Sendable` is free and its recorded state is actor-isolated;
//  tests `await` it. No model files, no `@unchecked Sendable`.
//

import Foundation

@testable import Tesseract_Agent

/// A non-`DictationError` model failure, so facade error-mapping is observable.
struct FakeModelError: Error, Sendable, Equatable {
    let message: String
}

actor InMemorySpeechRecognizer: SpeechRecognizer {
    // MARK: Programmed behavior
    private let cannedResult: TranscriptionResult
    private var latency: Duration?
    private let loadError: (any Error & Sendable)?
    private let transcribeError: (any Error & Sendable)?

    /// Adjust the programmed latency mid-test — e.g. drop it to `nil` after
    /// cancelling an over-running call so a follow-up `transcribe` returns fast.
    func setLatency(_ latency: Duration?) { self.latency = latency }

    // MARK: Recorded state
    private(set) var loadCount = 0
    private(set) var loadedPaths: [URL] = []
    private(set) var transcribeCount = 0
    private(set) var recordedLanguages: [String?] = []
    private(set) var cancelCount = 0
    private(set) var transcribeWasInterrupted = false

    init(
        result: TranscriptionResult = TranscriptionResult(
            text: "canned transcription",
            segments: [],
            language: "en",
            processingTime: 0
        ),
        latency: Duration? = nil,
        loadError: (any Error & Sendable)? = nil,
        transcribeError: (any Error & Sendable)? = nil
    ) {
        self.cannedResult = result
        self.latency = latency
        self.loadError = loadError
        self.transcribeError = transcribeError
    }

    func load(modelPath: URL) async throws {
        loadCount += 1
        loadedPaths.append(modelPath)
        if let loadError { throw loadError }
    }

    func transcribe(_ audioData: AudioData, language: String?) async throws -> TranscriptionResult {
        transcribeCount += 1
        recordedLanguages.append(language)

        if let latency {
            do {
                try await Task.sleep(for: latency)
            } catch {
                transcribeWasInterrupted = true
                throw error
            }
        }

        if let transcribeError { throw transcribeError }
        return cannedResult
    }

    func cancel() {
        cancelCount += 1
    }
}
