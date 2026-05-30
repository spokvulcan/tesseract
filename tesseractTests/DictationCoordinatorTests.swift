//
//  DictationCoordinatorTests.swift
//  tesseractTests
//
//  Exercises `DictationCoordinator`'s full state machine
//  (idle→recording→processing→idle), history capture, and post-processing — by
//  composing the existing engine-facing `Transcribing` seam over the *real*
//  `TranscriptionEngine` with an `InMemorySpeechRecognizer` below it, plus
//  hermetic fakes for audio capture, text injection, and history. No model files,
//  no microphone, no `UserDefaults`.
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

    /// When `gated`, `inject` suspends until `releaseGate()` — lets a test hold
    /// the success path *inside* the injection await and mutate coordinator state
    /// meanwhile (the injection-suspension window).
    var gated = false
    private var gate: CheckedContinuation<Void, Never>?
    var isAwaitingGate: Bool { gate != nil }

    func inject(_ text: String) async throws {
        if gated {
            await withCheckedContinuation { gate = $0 }
        }
        // Mirror the real `TextInjector`, whose paste is gated behind a
        // cancellation-aware `Task.sleep`: if the processing task is cancelled,
        // the side effect (recording the injection) must NOT happen.
        try Task.checkCancellation()
        injected.append(text)
    }

    func releaseGate() {
        gate?.resume()
        gate = nil
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
            .appendingPathComponent("DictationCoordinatorTests-\(UUID().uuidString)", isDirectory: true)
        try fm.createDirectory(at: dir.appendingPathComponent("AudioEncoder.mlmodelc"), withIntermediateDirectories: true)
        try fm.createDirectory(at: dir.appendingPathComponent("TextDecoder.mlmodelc"), withIntermediateDirectories: true)
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
                Issue.record("condition not met within \(attempts) yields", sourceLocation: sourceLocation)
                throw WaitTimedOut()
            }
            await Task.yield()
        }
    }

    private func makeEngine(recognizer: InMemorySpeechRecognizer, bundle: URL) async throws -> TranscriptionEngine {
        let engine = TranscriptionEngine(makeRecognizer: { recognizer })
        try await engine.loadModel(from: bundle)
        return engine
    }

    // MARK: - Happy path

    @Test
    func runsIdleToRecordingToProcessingToIdleWithHistoryAndInjection() async throws {
        let bundle = try makeFakeModelBundle()
        defer { try? FileManager.default.removeItem(at: bundle) }

        let recognizer = InMemorySpeechRecognizer(
            result: TranscriptionResult(text: "hello world", segments: [], language: "en", processingTime: 0)
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
        #expect(store.entries == [FakeTranscriptionStore.Entry(text: expected, duration: 2.0, model: "Large V3 Turbo")])
        #expect(injector.injected == [expected + " "])
        #expect(capture.stopCount == 1)
    }

    // MARK: - Recording too short

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

        // handleError runs synchronously for the too-short guard.
        #expect(coordinator.state == .error(DictationError.recordingTooShort.localizedDescription))
        #expect(coordinator.lastError?.localizedDescription == DictationError.recordingTooShort.localizedDescription)
        #expect(await recognizer.transcribeCount == 0)
    }

    // MARK: - No speech detected (post-processing yields empty)

    @Test
    func emptyTranscriptionResultGoesToNoSpeechDetectedError() async throws {
        let bundle = try makeFakeModelBundle()
        defer { try? FileManager.default.removeItem(at: bundle) }

        let recognizer = InMemorySpeechRecognizer(
            result: TranscriptionResult(text: "   ", segments: [], language: "en", processingTime: 0)
        )
        let engine = try await makeEngine(recognizer: recognizer, bundle: bundle)

        let store = FakeTranscriptionStore()
        let coordinator = DictationCoordinator(
            audioCapture: FakeAudioCapture(cannedAudio: AudioData(samples: [0.1], sampleRate: 16_000, duration: 2.0)),
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
        #expect(coordinator.lastError?.localizedDescription == DictationError.noSpeechDetected.localizedDescription)
        #expect(store.entries.isEmpty)
    }

    // MARK: - Cancel

    @Test
    func cancelFromRecordingReturnsToIdleAndStopsCapture() async throws {
        let bundle = try makeFakeModelBundle()
        defer { try? FileManager.default.removeItem(at: bundle) }

        let recognizer = InMemorySpeechRecognizer()
        let engine = try await makeEngine(recognizer: recognizer, bundle: bundle)

        let capture = FakeAudioCapture(cannedAudio: AudioData(samples: [0.1], sampleRate: 16_000, duration: 2.0))
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

    @Test
    func cancelWhileProcessingStaysIdleAndDoesNotSurfaceAnError() async throws {
        let bundle = try makeFakeModelBundle()
        defer { try? FileManager.default.removeItem(at: bundle) }

        // Over-running recognizer so the transcription is still in flight when we
        // cancel; a long timeout so the timeout race doesn't fire first.
        let recognizer = InMemorySpeechRecognizer(latency: .seconds(60))
        let engine = TranscriptionEngine(makeRecognizer: { recognizer }, timeout: { _ in .seconds(120) })
        try await engine.loadModel(from: bundle)

        let coordinator = DictationCoordinator(
            audioCapture: FakeAudioCapture(cannedAudio: AudioData(samples: [0.1], sampleRate: 16_000, duration: 2.0)),
            transcriptionEngine: engine,
            textInjector: FakeTextInjector(),
            history: FakeTranscriptionStore(),
            settings: SettingsManager(store: InMemorySettingsStore())
        )

        coordinator.onHotkeyDown()
        coordinator.onHotkeyUp()
        #expect(coordinator.state == .processing)

        // Cancel only once the transcription is genuinely in flight below the seam.
        while await recognizer.transcribeCount == 0 { await Task.yield() }
        coordinator.cancel()
        #expect(coordinator.state == .idle)

        // Let the cancelled background transcription task fully unwind: the
        // CancellationError must NOT be reclassified as a transcription failure.
        while await recognizer.transcribeWasInterrupted == false { await Task.yield() }
        for _ in 0..<500 { await Task.yield() }

        #expect(coordinator.state == .idle)
        #expect(coordinator.lastError == nil)
    }

    @Test
    func cancelThenImmediatelyRestartIsNotClobberedByStaleTranscription() async throws {
        let bundle = try makeFakeModelBundle()
        defer { try? FileManager.default.removeItem(at: bundle) }

        let recognizer = InMemorySpeechRecognizer(latency: .seconds(60))
        let engine = TranscriptionEngine(makeRecognizer: { recognizer }, timeout: { _ in .seconds(120) })
        try await engine.loadModel(from: bundle)

        let capture = FakeAudioCapture(cannedAudio: AudioData(samples: [0.1], sampleRate: 16_000, duration: 2.0))
        let coordinator = DictationCoordinator(
            audioCapture: capture,
            transcriptionEngine: engine,
            textInjector: FakeTextInjector(),
            history: FakeTranscriptionStore(),
            settings: SettingsManager(store: InMemorySettingsStore())
        )

        // First attempt reaches processing, then is cancelled.
        coordinator.onHotkeyDown()
        coordinator.onHotkeyUp()
        #expect(coordinator.state == .processing)
        while await recognizer.transcribeCount == 0 { await Task.yield() }
        coordinator.cancel()
        #expect(coordinator.state == .idle)

        // Start a NEW recording synchronously (before the cancelled task unwinds).
        coordinator.onHotkeyDown()
        #expect(coordinator.state == .recording)

        // Let the stale task's cancellation fully unwind: it must NOT overwrite the
        // new operation's .recording state with .idle.
        while await recognizer.transcribeWasInterrupted == false { await Task.yield() }
        for _ in 0..<500 { await Task.yield() }

        #expect(coordinator.state == .recording)
    }

    @Test
    func successfulTranscriptionArrivingAfterCancelCommitsNothing() async throws {
        // Engine-facing double that delivers SUCCESS on demand, even after cancel.
        let engine = ControllableTranscribing(
            result: TranscriptionResult(text: "late success", segments: [], language: "en", processingTime: 0)
        )
        let injector = FakeTextInjector()
        let store = FakeTranscriptionStore()
        let coordinator = DictationCoordinator(
            audioCapture: FakeAudioCapture(cannedAudio: AudioData(samples: [0.1], sampleRate: 16_000, duration: 2.0)),
            transcriptionEngine: engine,
            textInjector: injector,
            history: store,
            settings: SettingsManager(store: InMemorySettingsStore())
        )

        coordinator.onHotkeyDown()
        coordinator.onHotkeyUp()
        #expect(coordinator.state == .processing)
        while !engine.isAwaiting { await Task.yield() }

        coordinator.cancel()
        #expect(coordinator.state == .idle)
        #expect(engine.cancelCount == 1)

        // The engine returns a successful transcription *after* the cancel.
        engine.completeWithSuccess()
        for _ in 0..<500 { await Task.yield() }

        // The stale success must commit nothing.
        #expect(store.entries.isEmpty)
        #expect(injector.injected.isEmpty)
        #expect(coordinator.lastTranscription == "")
        #expect(coordinator.state == .idle)
    }

    @Test
    func cancelAndRestartDuringInjectionDoesNotClobberNewRecording() async throws {
        let bundle = try makeFakeModelBundle()
        defer { try? FileManager.default.removeItem(at: bundle) }

        let recognizer = InMemorySpeechRecognizer(
            result: TranscriptionResult(text: "done", segments: [], language: "en", processingTime: 0)
        )
        let engine = TranscriptionEngine(makeRecognizer: { recognizer })
        try await engine.loadModel(from: bundle)

        let injector = FakeTextInjector()
        injector.gated = true
        let coordinator = DictationCoordinator(
            audioCapture: FakeAudioCapture(cannedAudio: AudioData(samples: [0.1], sampleRate: 16_000, duration: 2.0)),
            transcriptionEngine: engine,
            textInjector: injector,
            history: FakeTranscriptionStore(),
            settings: SettingsManager(store: InMemorySettingsStore())
        )

        coordinator.onHotkeyDown()
        coordinator.onHotkeyUp()

        // Wait until the success path is suspended inside injection.
        while !injector.isAwaitingGate { await Task.yield() }

        // Cancel and immediately start a new recording while injection is suspended.
        coordinator.cancel()
        coordinator.onHotkeyDown()
        #expect(coordinator.state == .recording)

        // Resume injection; the stale task must not play success or reset state.
        injector.releaseGate()
        for _ in 0..<500 { await Task.yield() }

        #expect(coordinator.state == .recording)
        #expect(injector.injected.isEmpty)
    }

    @Test
    func cancelDuringInjectionAbortsTheInjectionSideEffect() async throws {
        let bundle = try makeFakeModelBundle()
        defer { try? FileManager.default.removeItem(at: bundle) }

        let recognizer = InMemorySpeechRecognizer(
            result: TranscriptionResult(text: "done", segments: [], language: "en", processingTime: 0)
        )
        let engine = TranscriptionEngine(makeRecognizer: { recognizer })
        try await engine.loadModel(from: bundle)

        let injector = FakeTextInjector()
        injector.gated = true
        let coordinator = DictationCoordinator(
            audioCapture: FakeAudioCapture(cannedAudio: AudioData(samples: [0.1], sampleRate: 16_000, duration: 2.0)),
            transcriptionEngine: engine,
            textInjector: injector,
            history: FakeTranscriptionStore(),
            settings: SettingsManager(store: InMemorySettingsStore())
        )

        coordinator.onHotkeyDown()
        coordinator.onHotkeyUp()

        // Suspend the success path inside injection, then cancel.
        while !injector.isAwaitingGate { await Task.yield() }
        coordinator.cancel()
        #expect(coordinator.state == .idle)

        // Releasing the gate must NOT result in stale text being injected: the
        // processing task is cancelled, so the cancellation-aware injection aborts
        // before its side effect (the real injector aborts before the paste).
        injector.releaseGate()
        for _ in 0..<500 { await Task.yield() }

        #expect(injector.injected.isEmpty)
        #expect(coordinator.state == .idle)
    }
}
