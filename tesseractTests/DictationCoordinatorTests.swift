//
//  DictationCoordinatorTests.swift
//  tesseractTests
//
//  Exercises `DictationCoordinator` as a thin composer over the **Voice Capture
//  Session**: the `StopResult`/`Outcome` → **Overlay Feed** phase/beat mapping,
//  the commit closure's effects (history write + auto-insert text injection with
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

    // MARK: - Happy path (Outcome → phase/beat mapping + commit effects)

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
        let feed = DictationFeed()

        let coordinator = DictationCoordinator(
            audioCapture: capture,
            transcriptionEngine: engine,
            textInjector: injector,
            history: store,
            settings: settings,
            feed: feed
        )

        #expect(coordinator.state == .idle)
        #expect(feed.beat == nil)

        coordinator.onHotkeyDown()
        #expect(coordinator.state == .recording)
        #expect(feed.phase == .recording)
        #expect(feed.recordingStarted != nil)
        #expect(capture.isCapturing)
        #expect(capture.startCount == 1)

        coordinator.onHotkeyUp()
        #expect(coordinator.state == .processing)
        #expect(feed.recordingStarted == nil)

        try await waitUntil { coordinator.state == .idle }

        let expected = TranscriptionPostProcessor().process("hello world")
        #expect(!expected.isEmpty)
        #expect(coordinator.lastTranscription == expected)
        // The terminal beat carries the committed text so a variant can end the
        // happy path (and a future correction affordance can hook it).
        #expect(feed.beat?.outcome == .committed(text: expected, duration: 2.0))
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
            settings: SettingsManager(store: InMemorySettingsStore()),
            feed: DictationFeed()
        )

        coordinator.onHotkeyDown()
        coordinator.onHotkeyUp()

        // The too-short guard maps to a *typed* error synchronously — variants
        // receive the case, not a pre-flattened string.
        #expect(coordinator.state == .error(.recordingTooShort))
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
            settings: SettingsManager(store: InMemorySettingsStore()),
            feed: DictationFeed()
        )

        coordinator.onHotkeyDown()

        #expect(coordinator.state == .error(.microphoneBusy))
        #expect(capture.startCount == 0)
    }

    // MARK: - No speech detected (Outcome.empty → noSpeech error + empty beat)

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
        let feed = DictationFeed()
        let coordinator = DictationCoordinator(
            audioCapture: FakeAudioCapture(
                cannedAudio: AudioData(samples: [0.1], sampleRate: 16_000, duration: 2.0)),
            transcriptionEngine: engine,
            textInjector: FakeTextInjector(),
            history: store,
            settings: SettingsManager(store: InMemorySettingsStore()),
            feed: feed
        )

        coordinator.onHotkeyDown()
        coordinator.onHotkeyUp()

        try await waitUntil { coordinator.state == .error(.noSpeechDetected) }
        #expect(feed.beat?.outcome == .empty)
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
            settings: SettingsManager(store: InMemorySettingsStore()),
            feed: DictationFeed()
        )

        coordinator.onHotkeyDown()
        coordinator.onHotkeyUp()
        #expect(coordinator.state == .processing)
        while !engine.isAwaiting { await Task.yield() }

        engine.completeWithFailure(FakeModelError(message: "boom"))

        try await waitUntil {
            if case .error = coordinator.state { return true } else { return false }
        }
        if case .error(.transcriptionFailed) = coordinator.state {
            // expected mapping
        } else {
            Issue.record(
                "expected .error(.transcriptionFailed), got \(String(describing: coordinator.state))"
            )
        }
    }

    // MARK: - Cancel (returns to idle, stops capture, emits the cancelled beat)

    @Test
    func cancelFromRecordingReturnsToIdleAndStopsCapture() async throws {
        let bundle = try makeFakeModelBundle()
        defer { try? FileManager.default.removeItem(at: bundle) }

        let recognizer = InMemorySpeechRecognizer()
        let engine = try await makeEngine(recognizer: recognizer, bundle: bundle)

        let capture = FakeAudioCapture(
            cannedAudio: AudioData(samples: [0.1], sampleRate: 16_000, duration: 2.0))
        let feed = DictationFeed()
        let coordinator = DictationCoordinator(
            audioCapture: capture,
            transcriptionEngine: engine,
            textInjector: FakeTextInjector(),
            history: FakeTranscriptionStore(),
            settings: SettingsManager(store: InMemorySettingsStore()),
            feed: feed
        )

        coordinator.onHotkeyDown()
        #expect(coordinator.state == .recording)

        coordinator.cancel()
        #expect(coordinator.state == .idle)
        #expect(feed.beat?.outcome == .cancelled)
        #expect(!capture.isCapturing)
        #expect(capture.stopCount == 1)
    }

    // MARK: - Hotkey re-engagement (an error pill is feedback, never a gate)

    /// A too-short tap used to park the hotkey behind the 3 s error auto-reset;
    /// the next press must start recording immediately instead.
    @Test
    func hotkeyDownOnAnErrorPillRetriesImmediately() async throws {
        // Below the 0.5s minimum.
        let audio = AudioData(samples: [0.1], sampleRate: 16_000, duration: 0.1)
        let capture = FakeAudioCapture(cannedAudio: audio)
        let coordinator = DictationCoordinator(
            audioCapture: capture,
            transcriptionEngine: ControllableTranscribing(),
            textInjector: FakeTextInjector(),
            history: FakeTranscriptionStore(),
            settings: SettingsManager(store: InMemorySettingsStore()),
            feed: DictationFeed()
        )

        coordinator.onHotkeyDown()
        coordinator.onHotkeyUp()
        #expect(coordinator.state == .error(.recordingTooShort))

        coordinator.onHotkeyDown()
        #expect(coordinator.state == .recording)
        #expect(capture.startCount == 2)
    }

    /// A press that lands while a previous capture is still transcribing is not
    /// swallowed: if the key is still held when processing resolves, recording
    /// starts right then — and the finished dictation still commits.
    @Test
    func hotkeyHeldThroughProcessingStartsRecordingWhenItResolves() async throws {
        let engine = ControllableTranscribing()
        let capture = FakeAudioCapture(
            cannedAudio: AudioData(samples: [0.1], sampleRate: 16_000, duration: 2.0))
        let injector = FakeTextInjector()
        let coordinator = DictationCoordinator(
            audioCapture: capture,
            transcriptionEngine: engine,
            textInjector: injector,
            history: FakeTranscriptionStore(),
            settings: SettingsManager(store: InMemorySettingsStore()),
            feed: DictationFeed()
        )

        coordinator.onHotkeyDown()
        coordinator.onHotkeyUp()
        #expect(coordinator.state == .processing)
        while !engine.isAwaiting { await Task.yield() }

        coordinator.onHotkeyDown()  // lands mid-processing, key stays held
        #expect(coordinator.state == .processing)

        engine.completeWithSuccess()
        try await waitUntil { coordinator.state == .recording }
        #expect(capture.startCount == 2)
        #expect(injector.injected.count == 1)  // the finished dictation still committed
    }

    /// Releasing the key while still `.processing` abandons the pending start —
    /// a tap wholly inside the processing window has no audio to offer.
    @Test
    func hotkeyReleasedDuringProcessingAbandonsThePendingStart() async throws {
        let engine = ControllableTranscribing()
        let capture = FakeAudioCapture(
            cannedAudio: AudioData(samples: [0.1], sampleRate: 16_000, duration: 2.0))
        let coordinator = DictationCoordinator(
            audioCapture: capture,
            transcriptionEngine: engine,
            textInjector: FakeTextInjector(),
            history: FakeTranscriptionStore(),
            settings: SettingsManager(store: InMemorySettingsStore()),
            feed: DictationFeed()
        )

        coordinator.onHotkeyDown()
        coordinator.onHotkeyUp()
        while !engine.isAwaiting { await Task.yield() }

        coordinator.onHotkeyDown()  // press mid-processing…
        coordinator.onHotkeyUp()  // …released before it resolves

        engine.completeWithSuccess()
        try await waitUntil { coordinator.state == .idle }
        #expect(capture.startCount == 1)
    }

    /// "No speech detected" resolving under a held key: the error does not gate
    /// — recording starts immediately, because the press *is* the retry.
    @Test
    func noSpeechResolutionWithKeyHeldStartsRecordingImmediately() async throws {
        let engine = ControllableTranscribing(
            result: TranscriptionResult(
                text: "   ", segments: [], language: "en", processingTime: 0)
        )
        let capture = FakeAudioCapture(
            cannedAudio: AudioData(samples: [0.1], sampleRate: 16_000, duration: 2.0))
        let coordinator = DictationCoordinator(
            audioCapture: capture,
            transcriptionEngine: engine,
            textInjector: FakeTextInjector(),
            history: FakeTranscriptionStore(),
            settings: SettingsManager(store: InMemorySettingsStore()),
            feed: DictationFeed()
        )

        coordinator.onHotkeyDown()
        coordinator.onHotkeyUp()
        while !engine.isAwaiting { await Task.yield() }

        coordinator.onHotkeyDown()  // held while "no speech" resolves

        engine.completeWithSuccess()
        try await waitUntil { coordinator.state == .recording }
        #expect(capture.startCount == 2)
    }
}
