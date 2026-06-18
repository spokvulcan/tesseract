//
//  DictationCoordinatorTests.swift
//  tesseractTests
//
//  Exercises `DictationCoordinator` as a thin composer over the **Voice Capture
//  Session**: the `StopResult`/`Outcome` → `DictationState` mapping, the commit
//  closure's effects (history write + auto-insert text injection with
//  restore-clipboard), and `DictationError` mapping. The deep staleness/supersede
//  races now live once in `VoiceCaptureSessionTests`, driven through the session's
//  own interface — they are no longer duplicated here.
//
//  Composes the engine-facing `Transcribing` seam over the *real*
//  `TranscriptionEngine` with an `InMemorySpeechRecognizer` below it (or the
//  `ControllableTranscribing` double where a failure must be delivered on demand),
//  plus hermetic fakes for audio capture, text injection, and history. No model
//  files, no microphone, no `UserDefaults`.
//

import Foundation
import Testing

@testable import Tesseract_Agent

// MARK: - Hermetic peer doubles for the coordinator's collaborators

@MainActor
final class FakeAudioCapture: AudioCapturing {
    var isCapturing = false
    var startError: (any Error)?
    var cannedAudio: AudioData?
    private(set) var startCount = 0
    private(set) var stopCount = 0

    init(cannedAudio: AudioData?) { self.cannedAudio = cannedAudio }

    func startCapture() throws {
        startCount += 1
        if let startError { throw startError }
        isCapturing = true
    }

    func stopCapture() -> AudioData? {
        stopCount += 1
        isCapturing = false
        return cannedAudio
    }
}

@MainActor
final class FakeTextInjector: TextInjecting {
    var restoreClipboard = false
    private(set) var injected: [String] = []

    func inject(_ text: String) async throws {
        // Mirror the real `TextInjector`, whose paste is gated behind a
        // cancellation-aware `Task.sleep`: if the processing task is cancelled,
        // the side effect (recording the injection) must NOT happen.
        try Task.checkCancellation()
        injected.append(text)
    }
}

@MainActor
final class FakeTranscriptionStore: TranscriptionStoring {
    struct Entry: Equatable {
        let text: String
        let duration: TimeInterval
        let model: String
    }
    private(set) var entries: [Entry] = []
    private(set) var copyCount = 0

    func add(text: String, duration: TimeInterval, model: String) {
        entries.append(Entry(text: text, duration: duration, model: model))
    }

    func copyLatestToPasteboard() { copyCount += 1 }
}

@MainActor
struct DictationCoordinatorTests {

    // MARK: - Helpers

    private func makeFakeModelBundle() throws -> URL {
        let fm = FileManager.default
        let dir = fm.temporaryDirectory
            .appendingPathComponent(
                "DictationCoordinatorTests-\(UUID().uuidString)", isDirectory: true)
        try fm.createDirectory(
            at: dir.appendingPathComponent("AudioEncoder.mlmodelc"),
            withIntermediateDirectories: true)
        try fm.createDirectory(
            at: dir.appendingPathComponent("TextDecoder.mlmodelc"),
            withIntermediateDirectories: true)
        return dir
    }

    private struct WaitTimedOut: Error {}

    /// Awaits an `@Observable`-driven condition by yielding (no wall-clock sleep).
    private func waitUntil(
        _ condition: () -> Bool,
        attempts: Int = 100_000,
        sourceLocation: SourceLocation = #_sourceLocation
    ) async throws {
        var n = 0
        while !condition() {
            n += 1
            if n > attempts {
                Issue.record(
                    "condition not met within \(attempts) yields", sourceLocation: sourceLocation)
                throw WaitTimedOut()
            }
            await Task.yield()
        }
    }

    private func makeEngine(recognizer: InMemorySpeechRecognizer, bundle: URL) async throws
        -> TranscriptionEngine
    {
        let engine = TranscriptionEngine(makeRecognizer: { recognizer })
        try await engine.loadModel(from: bundle)
        return engine
    }

    // MARK: - Happy path (Outcome → state mapping + commit effects)

    @Test
    func runsIdleToRecordingToProcessingToIdleWithHistoryAndInjection() async throws {
        let bundle = try makeFakeModelBundle()
        defer { try? FileManager.default.removeItem(at: bundle) }

        let recognizer = InMemorySpeechRecognizer(
            result: TranscriptionResult(
                text: "hello world", segments: [], language: "en", processingTime: 0)
        )
        let engine = try await makeEngine(recognizer: recognizer, bundle: bundle)

        let audio = AudioData(samples: [0.1, 0.2], sampleRate: 16_000, duration: 2.0)
        let capture = FakeAudioCapture(cannedAudio: audio)
        let injector = FakeTextInjector()
        let store = FakeTranscriptionStore()
        let settings = SettingsManager(store: InMemorySettingsStore())

        let coordinator = DictationCoordinator(
            audioCapture: capture,
            transcriptionEngine: engine,
            textInjector: injector,
            history: store,
            settings: settings
        )

        #expect(coordinator.state == .idle)

        coordinator.onHotkeyDown()
        #expect(coordinator.state == .recording)
        #expect(capture.isCapturing)
        #expect(capture.startCount == 1)

        coordinator.onHotkeyUp()
        #expect(coordinator.state == .processing)

        try await waitUntil { coordinator.state == .idle }

        let expected = TranscriptionPostProcessor().process("hello world")
        #expect(!expected.isEmpty)
        #expect(coordinator.lastTranscription == expected)
        #expect(
            store.entries == [
                FakeTranscriptionStore.Entry(text: expected, duration: 2.0, model: "Whisper Turbo")
            ])
        #expect(injector.injected == [expected + " "])
        #expect(injector.restoreClipboard == settings.restoreClipboard)
        #expect(capture.stopCount == 1)
    }

    // MARK: - Recording too short (StopResult.tooShort → error)

    @Test
    func recordingShorterThanMinimumGoesToErrorWithoutTranscribing() async throws {
        let bundle = try makeFakeModelBundle()
        defer { try? FileManager.default.removeItem(at: bundle) }

        let recognizer = InMemorySpeechRecognizer()
        let engine = try await makeEngine(recognizer: recognizer, bundle: bundle)

        // Below the 0.5s minimum.
        let audio = AudioData(samples: [0.1], sampleRate: 16_000, duration: 0.1)
        let coordinator = DictationCoordinator(
            audioCapture: FakeAudioCapture(cannedAudio: audio),
            transcriptionEngine: engine,
            textInjector: FakeTextInjector(),
            history: FakeTranscriptionStore(),
            settings: SettingsManager(store: InMemorySettingsStore())
        )

        coordinator.onHotkeyDown()
        coordinator.onHotkeyUp()

        // The too-short guard maps to an error synchronously.
        #expect(coordinator.state == .error(DictationError.recordingTooShort.localizedDescription))
        #expect(
            coordinator.lastError?.localizedDescription
                == DictationError.recordingTooShort.localizedDescription)
        #expect(await recognizer.transcribeCount == 0)
    }

    // MARK: - Microphone busy (StartResult.micBusy → error)

    /// Dictation now refuses to start while the shared capture engine is already
    /// capturing, surfacing a clear "microphone in use" error instead of silently
    /// recording nothing. (The guard lives once in the session; dictation gains it.)
    @Test
    func startWhileMicrophoneBusyGoesToErrorWithoutRecording() async throws {
        let capture = FakeAudioCapture(
            cannedAudio: AudioData(samples: [0.1], sampleRate: 16_000, duration: 2.0))
        capture.isCapturing = true  // the shared engine is already in use

        let coordinator = DictationCoordinator(
            audioCapture: capture,
            transcriptionEngine: ControllableTranscribing(),
            textInjector: FakeTextInjector(),
            history: FakeTranscriptionStore(),
            settings: SettingsManager(store: InMemorySettingsStore())
        )

        coordinator.onHotkeyDown()

        #expect(coordinator.state == .error(DictationError.microphoneBusy.localizedDescription))
        #expect(
            coordinator.lastError?.localizedDescription
                == DictationError.microphoneBusy.localizedDescription)
        #expect(capture.startCount == 0)
    }

    // MARK: - No speech detected (Outcome.empty → noSpeech error)

    @Test
    func emptyTranscriptionResultGoesToNoSpeechDetectedError() async throws {
        let bundle = try makeFakeModelBundle()
        defer { try? FileManager.default.removeItem(at: bundle) }

        let recognizer = InMemorySpeechRecognizer(
            result: TranscriptionResult(
                text: "   ", segments: [], language: "en", processingTime: 0)
        )
        let engine = try await makeEngine(recognizer: recognizer, bundle: bundle)

        let store = FakeTranscriptionStore()
        let coordinator = DictationCoordinator(
            audioCapture: FakeAudioCapture(
                cannedAudio: AudioData(samples: [0.1], sampleRate: 16_000, duration: 2.0)),
            transcriptionEngine: engine,
            textInjector: FakeTextInjector(),
            history: store,
            settings: SettingsManager(store: InMemorySettingsStore())
        )

        coordinator.onHotkeyDown()
        coordinator.onHotkeyUp()

        try await waitUntil {
            if case .error = coordinator.state { return true } else { return false }
        }
        #expect(
            coordinator.lastError?.localizedDescription
                == DictationError.noSpeechDetected.localizedDescription)
        #expect(store.entries.isEmpty)
    }

    // MARK: - Transcription failure (Outcome.failed → DictationError mapping)

    /// A non-`DictationError` failure from the engine maps onto
    /// `.transcriptionFailed`, surfacing the underlying description.
    @Test
    func transcriptionFailureMapsToTranscriptionFailedError() async throws {
        let engine = ControllableTranscribing()
        let coordinator = DictationCoordinator(
            audioCapture: FakeAudioCapture(
                cannedAudio: AudioData(samples: [0.1], sampleRate: 16_000, duration: 2.0)),
            transcriptionEngine: engine,
            textInjector: FakeTextInjector(),
            history: FakeTranscriptionStore(),
            settings: SettingsManager(store: InMemorySettingsStore())
        )

        coordinator.onHotkeyDown()
        coordinator.onHotkeyUp()
        #expect(coordinator.state == .processing)
        while !engine.isAwaiting { await Task.yield() }

        engine.completeWithFailure(FakeModelError(message: "boom"))

        try await waitUntil {
            if case .error = coordinator.state { return true } else { return false }
        }
        if case .transcriptionFailed = coordinator.lastError {
            // expected mapping
        } else {
            Issue.record(
                "expected .transcriptionFailed, got \(String(describing: coordinator.lastError))")
        }
    }

    // MARK: - Cancel (returns to idle, stops capture)

    @Test
    func cancelFromRecordingReturnsToIdleAndStopsCapture() async throws {
        let bundle = try makeFakeModelBundle()
        defer { try? FileManager.default.removeItem(at: bundle) }

        let recognizer = InMemorySpeechRecognizer()
        let engine = try await makeEngine(recognizer: recognizer, bundle: bundle)

        let capture = FakeAudioCapture(
            cannedAudio: AudioData(samples: [0.1], sampleRate: 16_000, duration: 2.0))
        let coordinator = DictationCoordinator(
            audioCapture: capture,
            transcriptionEngine: engine,
            textInjector: FakeTextInjector(),
            history: FakeTranscriptionStore(),
            settings: SettingsManager(store: InMemorySettingsStore())
        )

        coordinator.onHotkeyDown()
        #expect(coordinator.state == .recording)

        coordinator.cancel()
        #expect(coordinator.state == .idle)
        #expect(!capture.isCapturing)
        #expect(capture.stopCount == 1)
    }
}
