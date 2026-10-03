//
//  ScriptedSpeechRecognizer.swift
//  tesseractTests
//
//  A Speech Recognizer peer for Live Preview tests (PRD #612): where
//  `InMemorySpeechRecognizer` returns one canned result, this one plays a
//  script, one `TranscriptionResult` per `transcribe` call in order, and keeps
//  returning the last entry once the script runs out. The preview's decodes and
//  the final pass share one recognizer, so a test scripts the previews first
//  and the full pass last. Every call takes the next entry when it starts, so
//  a call cancelled during its latency still uses up its entry.
//
//  It records how many calls it served and the duration of the audio each one
//  was given (the preview decodes from the end of its confirmed text, so the
//  durations show which slice of the take each decode saw), plus the languages
//  passed. An optional latency, applied to every call and changeable mid-test,
//  sleeps cancellably: a call cancelled while it sleeps throws
//  `CancellationError` and is counted in `interruptedCount`. An actor, like the
//  production adapter, so `Sendable` is free; tests `await` its state.
//

import Foundation

@testable import Tesseract_Agent

actor ScriptedSpeechRecognizer {
    // MARK: Programmed behavior
    private let script: [TranscriptionResult]
    private var latency: Duration?

    /// Adjust the latency mid-test, e.g. hold a preview decode in flight
    /// across a release, then drop it to `nil` so the final pass returns fast.
    func setLatency(_ latency: Duration?) { self.latency = latency }

    // MARK: Recorded state
    private(set) var loadCount = 0
    private(set) var transcribeCount = 0
    /// `audioData.duration` of every call, in call order.
    private(set) var audioDurations: [TimeInterval] = []
    private(set) var recordedLanguages: [String?] = []
    /// Calls cancelled while sleeping out their latency.
    private(set) var interruptedCount = 0
    /// Calls that have started and not yet returned or thrown.
    private(set) var inFlightCount = 0

    /// - Parameters:
    ///   - script: the results to return, one per call; the last repeats once
    ///     the script is exhausted. An empty script returns an empty result
    ///     for every call.
    ///   - latency: how long every call sleeps before returning, or `nil`.
    init(_ script: [TranscriptionResult], latency: Duration? = nil) {
        self.script = script
        self.latency = latency
    }

    func load(modelPath: URL) async throws {
        loadCount += 1
    }

    func transcribe(_ audioData: AudioData, language: String?) async throws -> TranscriptionResult {
        // `AudioData` is main-actor isolated in the app module (its default
        // isolation), so its properties are read there. Everything below is
        // recorded in one synchronous step after the hop, so a call's index,
        // count and duration always line up.
        let duration = await MainActor.run { audioData.duration }
        let index = transcribeCount
        transcribeCount += 1
        audioDurations.append(duration)
        recordedLanguages.append(language)
        inFlightCount += 1
        defer { inFlightCount -= 1 }

        if let latency {
            do {
                try await Task.sleep(for: latency)
            } catch {
                interruptedCount += 1
                throw error
            }
        }

        guard let last = script.last else {
            // Built here: `TranscriptionResult.empty` is main-actor isolated.
            return TranscriptionResult(text: "", segments: [], language: "en", processingTime: 0)
        }
        return index < script.count ? script[index] : last
    }

    // MARK: Building scripted results

    /// A result built from `(text, start, end)` segments, times in seconds from
    /// the start of the audio passed (as WhisperKit reports them). The result's
    /// text is the segments joined and trimmed, as the production adapter
    /// reports it; give segment texts their leading space as Whisper does.
    static func result(
        _ segments: [(text: String, start: TimeInterval, end: TimeInterval)],
        language: String = "en"
    ) -> TranscriptionResult {
        TranscriptionResult(
            text: segments.map(\.text).joined()
                .trimmingCharacters(in: .whitespacesAndNewlines),
            segments: segments.map {
                TranscriptionSegment(text: $0.text, startTime: $0.start, endTime: $0.end)
            },
            language: language,
            processingTime: 0
        )
    }
}

// Conformance in an extension, as on `InMemorySpeechRecognizer`: declared on
// the actor itself, the test target's nonisolated default would infer
// `nonisolated` onto the synchronous initializer, which is invalid.
extension ScriptedSpeechRecognizer: SpeechRecognizer {}
