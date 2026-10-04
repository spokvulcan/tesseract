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

    /// What the **Live Partial** pump reads mid-capture (ticket #291).
    var cannedSnapshot: AudioData?

    func captureSnapshot() -> AudioData? {
        isCapturing ? cannedSnapshot : nil
    }
}

@MainActor
final class FakeTextInjector: TextInjecting {
    var restoreClipboard = false
    var injectError: DictationError?
    private(set) var injected: [String] = []

    func inject(_ text: String) async throws {
        // Mirror the real `TextInjector`, whose paste is gated behind a
        // cancellation-aware `Task.sleep`: if the processing task is cancelled,
        // the side effect (recording the injection) must NOT happen.
        try Task.checkCancellation()
        if let injectError { throw injectError }
        injected.append(text)
    }
}

@MainActor
final class FakeTranscriptionStore: TranscriptionStoring {
    struct Entry: Equatable {
        var text: String
        let duration: TimeInterval
        let model: String
        var pairID: UUID?
        var catches: [LearnedWordCatch]
        var app: TranscriptionEntry.App?

        init(
            text: String, duration: TimeInterval, model: String, pairID: UUID? = nil,
            catches: [LearnedWordCatch] = [], app: TranscriptionEntry.App? = nil
        ) {
            self.text = text
            self.duration = duration
            self.model = model
            self.pairID = pairID
            self.catches = catches
            self.app = app
        }
    }
    private(set) var entries: [Entry] = []
    private(set) var copyCount = 0

    func add(
        text: String, duration: TimeInterval, model: String, pairID: UUID?,
        catches: [LearnedWordCatch], app: TranscriptionEntry.App?
    ) {
        entries.append(
            Entry(
                text: text, duration: duration, model: model, pairID: pairID,
                catches: catches, app: app))
    }

    func copyLatestToPasteboard() { copyCount += 1 }

    func replaceText(forPairID pairID: UUID, with text: String, catches: [LearnedWordCatch]) {
        guard let index = entries.firstIndex(where: { $0.pairID == pairID }) else { return }
        entries[index].text = text
        entries[index].catches = catches
    }
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

    // MARK: - Live Preview (PRD #612)

    private func makeLearned() -> LearnedWordStore {
        LearnedWordStore(
            directory: FileManager.default.temporaryDirectory
                .appendingPathComponent("coordinator-preview-\(UUID().uuidString)"))
    }

    /// The preview streams into the feed with the regex cleanup and the
    /// Learned Words applied, and clears the moment the key comes up; what
    /// pastes is the full pass.
    @Test
    func thePreviewFlowsIntoTheFeedWithLearnedWordsAndClearsAtRelease() async throws {
        let bundle = try makeFakeModelBundle()
        defer { try? FileManager.default.removeItem(at: bundle) }
        let recognizer = InMemorySpeechRecognizer(
            result: TranscriptionResult(
                text: "ask cloud",
                segments: [TranscriptionSegment(text: " ask cloud", startTime: 0, endTime: 0.9)],
                language: "en", processingTime: 0))
        let engine = try await makeEngine(recognizer: recognizer, bundle: bundle)
        let words = makeLearned()
        words.learn(heard: "cloud", meant: "Claude")

        let audio = AudioData(samples: [0.1, 0.2], sampleRate: 16_000, duration: 2.0)
        let capture = FakeAudioCapture(cannedAudio: audio)
        // A snapshot past the pump's minimum-audio gate.
        capture.cannedSnapshot = AudioData(
            samples: [Float](repeating: 0.1, count: 16_000), sampleRate: 16_000, duration: 1.0)
        let injector = FakeTextInjector()
        let feed = DictationFeed()
        let coordinator = DictationCoordinator(
            audioCapture: capture, transcriptionEngine: engine, textInjector: injector,
            history: FakeTranscriptionStore(),
            settings: SettingsManager(store: InMemorySettingsStore()), feed: feed,
            learnedWords: words, frontmostApp: { nil })

        coordinator.onHotkeyDown()
        try await waitUntil { feed.preview != nil }
        #expect(feed.preview?.text == "Ask Claude")
        #expect(feed.preview?.catches.map(\.meant) == ["Claude"])
        // Shown, not counted: only a committed take counts its catches.
        #expect(words.word(heard: "cloud")?.totalCatches == 0)

        coordinator.onHotkeyUp()
        // The pump's stop clears the preview synchronously, before the final
        // commit resolves.
        #expect(feed.preview == nil)
        try await waitUntil { feed.phase == .idle && !injector.injected.isEmpty }
        #expect(injector.injected == ["Ask Claude "])
        #expect(words.word(heard: "cloud")?.totalCatches == 1)
    }

    /// Each preview decode reads only the audio after the last confirmed
    /// segment (WhisperKit's rule: all but the last two segments of a decode
    /// are confirmed), so the preview shows the whole take while a decode
    /// stays short.
    @Test
    func thePreviewDecodesFromTheLastConfirmedSegment() async throws {
        let bundle = try makeFakeModelBundle()
        defer { try? FileManager.default.removeItem(at: bundle) }
        let first = ScriptedSpeechRecognizer.result([
            (" Ask Claude", 0, 1.2), (" why the server", 1.2, 2.5), (" drops", 2.5, 3.0),
        ])
        let second = ScriptedSpeechRecognizer.result([
            (" why the server", 0, 1.3), (" drops the", 1.3, 2.2), (" first", 2.2, 2.8),
        ])
        let recognizer = ScriptedSpeechRecognizer([first, second])
        let engine = TranscriptionEngine(makeRecognizer: { recognizer })
        try await engine.loadModel(from: bundle)

        let capture = FakeAudioCapture(
            cannedAudio: AudioData(samples: [0.1, 0.2], sampleRate: 16_000, duration: 2.0))
        capture.cannedSnapshot = AudioData(
            samples: [Float](repeating: 0.1, count: 160_000), sampleRate: 16_000, duration: 10)
        let feed = DictationFeed()
        let coordinator = DictationCoordinator(
            audioCapture: capture, transcriptionEngine: engine, textInjector: FakeTextInjector(),
            history: FakeTranscriptionStore(),
            settings: SettingsManager(store: InMemorySettingsStore()), feed: feed,
            frontmostApp: { nil })

        coordinator.onHotkeyDown()
        try await waitUntil { feed.preview?.text == "Ask Claude why the server drops the first" }
        #expect(feed.preview?.confirmedTokens == 5)
        let durations = await recognizer.audioDurations
        #expect(durations.count >= 2)
        #expect(abs(durations[0] - 10) < 0.001)
        #expect(abs(durations[1] - 8.8) < 0.001)
        coordinator.cancel()
    }

    /// No snapshot past the minimum: the recognizer never hears mid-capture
    /// audio.
    @Test
    func noPreviewDecodeBeforeThereIsEnoughAudio() async throws {
        let bundle = try makeFakeModelBundle()
        defer { try? FileManager.default.removeItem(at: bundle) }
        let recognizer = InMemorySpeechRecognizer()
        let engine = try await makeEngine(recognizer: recognizer, bundle: bundle)
        let capture = FakeAudioCapture(
            cannedAudio: AudioData(samples: [0.1], sampleRate: 16_000, duration: 2.0))
        capture.cannedSnapshot = AudioData(
            samples: [Float](repeating: 0.1, count: 8_000), sampleRate: 16_000, duration: 0.5)
        let feed = DictationFeed()
        let coordinator = DictationCoordinator(
            audioCapture: capture, transcriptionEngine: engine, textInjector: FakeTextInjector(),
            history: FakeTranscriptionStore(),
            settings: SettingsManager(store: InMemorySettingsStore()), feed: feed)

        coordinator.onHotkeyDown()
        for _ in 0..<2000 { await Task.yield() }
        #expect(feed.preview == nil)
        #expect(await recognizer.transcribeCount == 0)
        coordinator.cancel()
    }

    // MARK: - Holding a take (PRD #612)

    private func runHeld(
        setting: CheckBeforePasting, tapShift: Bool
    ) async throws -> (injected: [String], take: DictatedTake?, held: Bool) {
        let bundle = try makeFakeModelBundle()
        defer { try? FileManager.default.removeItem(at: bundle) }
        let recognizer = InMemorySpeechRecognizer(
            result: TranscriptionResult(
                text: "ship it", segments: [], language: "en", processingTime: 0))
        let engine = try await makeEngine(recognizer: recognizer, bundle: bundle)
        let injector = FakeTextInjector()
        let feed = DictationFeed()
        let settings = SettingsManager(store: InMemorySettingsStore())
        settings.checkBeforePasting = setting
        let coordinator = DictationCoordinator(
            audioCapture: FakeAudioCapture(
                cannedAudio: AudioData(samples: [0.1, 0.2], sampleRate: 16_000, duration: 2.0)),
            transcriptionEngine: engine, textInjector: injector,
            history: FakeTranscriptionStore(), settings: settings, feed: feed,
            frontmostApp: { nil })
        var delivered: DictatedTake?
        coordinator.onTakeCommitted = { delivered = $0 }

        coordinator.onHotkeyDown()
        if tapShift { coordinator.shiftTapped() }
        let held = feed.isHeld
        coordinator.onHotkeyUp()
        try await waitUntil { feed.phase == .idle && delivered != nil }
        return (injector.injected, delivered, held)
    }

    /// ⇧ while talking keeps the take in the Lens: committed, not pasted.
    @Test
    func tappingShiftHoldsTheTakeInsteadOfPasting() async throws {
        let run = try await runHeld(setting: .whenShiftTapped, tapShift: true)
        #expect(run.held)
        #expect(run.injected.isEmpty)
        #expect(run.take?.held == true)
        #expect(run.take?.pasted == false)
    }

    @Test
    func withoutShiftTheTakePastesAsBefore() async throws {
        let run = try await runHeld(setting: .whenShiftTapped, tapShift: false)
        #expect(run.injected == ["Ship it "])
        #expect(run.take?.held == false)
    }

    @Test
    func theAlwaysSettingHoldsEveryTakeAndNeverIgnoresShift() async throws {
        let always = try await runHeld(setting: .always, tapShift: false)
        #expect(always.injected.isEmpty)
        #expect(always.take?.held == true)

        let never = try await runHeld(setting: .never, tapShift: true)
        #expect(!never.held)
        #expect(never.injected == ["Ship it "])
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
            feed: feed,
            frontmostApp: { TargetApp(bundleID: "com.apple.Terminal", name: "Terminal", pid: 7) }
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
        #expect(feed.beat?.outcome == .committed(text: expected, duration: 2.0, edits: []))
        // The entry keeps the app in front when the take started.
        #expect(
            store.entries == [
                FakeTranscriptionStore.Entry(
                    text: expected, duration: 2.0, model: "Whisper Turbo",
                    app: TranscriptionEntry.App(bundleID: "com.apple.Terminal", name: "Terminal"))
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

    // MARK: - Proofread Pass (rejected beat + insert-raw-anyway, edits on the beat)

    /// Records what the coordinator's proofread wrapper narrates while the
    /// pass runs — the model call must see the `.proofreading` phase.
    @MainActor
    final class PhaseProbe {
        var phase: DictationFeed.Phase?
    }

    private func makeProofreadPass(
        replying reply: @escaping @Sendable (String) -> String
    ) -> ProofreadPass {
        ProofreadPass(
            isEnabled: { true },
            isLLMBusy: { false },
            modelDirectory: { URL(fileURLWithPath: "/tmp/proofread-model") },
            loadModel: { _ in },
            runModel: { _, text in reply(text) },
            unloadModel: {}
        )
    }

    /// A rejected take is passive feedback, not an error gate: the phase
    /// returns to `.idle`, the beat carries the raw text and reason, nothing
    /// is committed — and "insert raw anyway" still delivers the words.
    @Test
    func rejectedTakeEmitsTheBeatAndInsertRawAnywayDelivers() async throws {
        let bundle = try makeFakeModelBundle()
        defer { try? FileManager.default.removeItem(at: bundle) }

        let recognizer = InMemorySpeechRecognizer(
            result: TranscriptionResult(
                text: "hello world", segments: [], language: "en", processingTime: 0)
        )
        let engine = try await makeEngine(recognizer: recognizer, bundle: bundle)

        let injector = FakeTextInjector()
        let store = FakeTranscriptionStore()
        let feed = DictationFeed()
        let coordinator = DictationCoordinator(
            audioCapture: FakeAudioCapture(
                cannedAudio: AudioData(samples: [0.1, 0.2], sampleRate: 16_000, duration: 2.0)),
            transcriptionEngine: engine,
            textInjector: injector,
            history: store,
            settings: SettingsManager(store: InMemorySettingsStore()),
            feed: feed,
            proofreadPass: makeProofreadPass(replying: { _ in "REJECT: garbled noise" })
        )

        coordinator.onHotkeyDown()
        coordinator.onHotkeyUp()
        try await waitUntil { feed.beat != nil }

        let expected = TranscriptionPostProcessor().process("hello world")
        #expect(feed.beat?.outcome == .rejected(raw: expected, reason: "garbled noise"))
        #expect(coordinator.state == .idle)  // passive: no error gate
        #expect(coordinator.lastRejectedRaw == expected)
        #expect(store.entries.isEmpty)
        #expect(injector.injected.isEmpty)

        coordinator.insertRawAnyway()
        try await waitUntil { injector.injected.count == 1 }
        #expect(injector.injected == [expected + " "])
        #expect(store.entries.count == 1)
        #expect(store.entries.first?.text == expected)
        #expect(coordinator.lastRejectedRaw == nil)
    }

    /// "Insert raw anyway" runs detached from any commit flow, so a failed
    /// injection there has no outcome switch to surface it — the coordinator
    /// must report it itself, never leave the press a silent no-op.
    @Test
    func insertRawAnywaySurfacesInjectionFailure() async throws {
        let bundle = try makeFakeModelBundle()
        defer { try? FileManager.default.removeItem(at: bundle) }

        let recognizer = InMemorySpeechRecognizer(
            result: TranscriptionResult(
                text: "hello world", segments: [], language: "en", processingTime: 0)
        )
        let engine = try await makeEngine(recognizer: recognizer, bundle: bundle)

        let injector = FakeTextInjector()
        let feed = DictationFeed()
        let coordinator = DictationCoordinator(
            audioCapture: FakeAudioCapture(
                cannedAudio: AudioData(samples: [0.1, 0.2], sampleRate: 16_000, duration: 2.0)),
            transcriptionEngine: engine,
            textInjector: injector,
            history: FakeTranscriptionStore(),
            settings: SettingsManager(store: InMemorySettingsStore()),
            feed: feed,
            proofreadPass: makeProofreadPass(replying: { _ in "REJECT: garbled noise" })
        )

        coordinator.onHotkeyDown()
        coordinator.onHotkeyUp()
        try await waitUntil { feed.beat != nil }

        injector.injectError = .textInjectionFailed("Clipboard contents could not be read safely")
        coordinator.insertRawAnyway()
        try await waitUntil {
            coordinator.state
                == .error(.textInjectionFailed("Clipboard contents could not be read safely"))
        }
        #expect(injector.injected.isEmpty)
    }

    /// A corrected take commits the corrected text; the terminal beat carries
    /// the word edits for variant narration, and the model call runs under
    /// the `.proofreading` phase.
    @Test
    func correctedTakeCommitsCorrectedTextAndBeatCarriesEdits() async throws {
        let bundle = try makeFakeModelBundle()
        defer { try? FileManager.default.removeItem(at: bundle) }

        let recognizer = InMemorySpeechRecognizer(
            result: TranscriptionResult(
                text: "hello world", segments: [], language: "en", processingTime: 0)
        )
        let engine = try await makeEngine(recognizer: recognizer, bundle: bundle)

        let raw = TranscriptionPostProcessor().process("hello world")
        let corrected = raw + " indeed"
        let probe = PhaseProbe()
        let feed = DictationFeed()
        let pass = ProofreadPass(
            isEnabled: { true },
            isLLMBusy: { false },
            modelDirectory: { URL(fileURLWithPath: "/tmp/proofread-model") },
            loadModel: { _ in },
            runModel: { _, _ in
                await MainActor.run { probe.phase = feed.phase }
                return corrected
            },
            unloadModel: {}
        )

        let injector = FakeTextInjector()
        let store = FakeTranscriptionStore()
        let coordinator = DictationCoordinator(
            audioCapture: FakeAudioCapture(
                cannedAudio: AudioData(samples: [0.1, 0.2], sampleRate: 16_000, duration: 2.0)),
            transcriptionEngine: engine,
            textInjector: injector,
            history: store,
            settings: SettingsManager(store: InMemorySettingsStore()),
            feed: feed,
            proofreadPass: pass
        )

        coordinator.onHotkeyDown()
        coordinator.onHotkeyUp()
        try await waitUntil { coordinator.state == .idle && feed.beat != nil }

        #expect(
            feed.beat?.outcome
                == .committed(
                    text: corrected, duration: 2.0,
                    edits: [WordEdit(original: "", replacement: "indeed")]))
        #expect(coordinator.lastTranscription == corrected)
        #expect(injector.injected == [corrected + " "])
        #expect(store.entries.first?.text == corrected)
        #expect(probe.phase == .proofreading)
    }

    // MARK: - Correction Pair flywheel (ticket #289)

    private func makePairStore() -> (store: CorrectionPairStore, cleanup: () -> Void) {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("coordinator-pairs-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return (
            CorrectionPairStore(directory: directory),
            { try? FileManager.default.removeItem(at: directory) }
        )
    }

    /// A committed take records a pair carrying the full lineage, and the
    /// history entry links to it.
    @Test
    func committedTakeRecordsALinkedPair() async throws {
        let bundle = try makeFakeModelBundle()
        defer { try? FileManager.default.removeItem(at: bundle) }
        let (pairs, cleanup) = makePairStore()
        defer { cleanup() }

        let recognizer = InMemorySpeechRecognizer(
            result: TranscriptionResult(
                text: "hello world", segments: [], language: "en", processingTime: 0)
        )
        let engine = try await makeEngine(recognizer: recognizer, bundle: bundle)

        let raw = TranscriptionPostProcessor().process("hello world")
        let corrected = raw + " indeed"
        let store = FakeTranscriptionStore()
        let feed = DictationFeed()
        let coordinator = DictationCoordinator(
            audioCapture: FakeAudioCapture(
                cannedAudio: AudioData(samples: [0.1, 0.2], sampleRate: 16_000, duration: 2.0)),
            transcriptionEngine: engine,
            textInjector: FakeTextInjector(),
            history: store,
            settings: SettingsManager(store: InMemorySettingsStore()),
            feed: feed,
            proofreadPass: makeProofreadPass(replying: { _ in corrected }),
            pairs: pairs
        )
        coordinator.onHotkeyDown()
        coordinator.onHotkeyUp()
        try await waitUntil { coordinator.state == .idle && feed.beat != nil }

        let pair = try #require(pairs.pairs.first)
        #expect(pair.rawASR == "hello world")
        #expect(pair.cleaned == raw)
        #expect(pair.proofread == corrected)
        #expect(pair.verdict == .corrected)
        #expect(pair.committed == corrected)
        #expect(coordinator.lastTakePairID == pair.id)
        #expect(store.entries.first?.pairID == pair.id)
    }

    /// A rejected take records its pair; "insert raw anyway" flags it (using
    /// it *is* "the pass was wrong") and links the history entry it creates.
    @Test
    func rejectedTakeRecordsAPairAndInsertRawAnywayFlagsIt() async throws {
        let bundle = try makeFakeModelBundle()
        defer { try? FileManager.default.removeItem(at: bundle) }
        let (pairs, cleanup) = makePairStore()
        defer { cleanup() }

        let recognizer = InMemorySpeechRecognizer(
            result: TranscriptionResult(
                text: "hello world", segments: [], language: "en", processingTime: 0)
        )
        let engine = try await makeEngine(recognizer: recognizer, bundle: bundle)

        let store = FakeTranscriptionStore()
        let feed = DictationFeed()
        let coordinator = DictationCoordinator(
            audioCapture: FakeAudioCapture(
                cannedAudio: AudioData(samples: [0.1, 0.2], sampleRate: 16_000, duration: 2.0)),
            transcriptionEngine: engine,
            textInjector: FakeTextInjector(),
            history: store,
            settings: SettingsManager(store: InMemorySettingsStore()),
            feed: feed,
            proofreadPass: makeProofreadPass(replying: { _ in "REJECT: garbled" }),
            pairs: pairs
        )

        coordinator.onHotkeyDown()
        coordinator.onHotkeyUp()
        try await waitUntil { feed.beat != nil }

        let pair = try #require(pairs.pairs.first)
        #expect(pair.verdict == .rejected)
        #expect(pair.rejectReason == "garbled")
        #expect(pair.committed == nil)
        #expect(pair.flaggedWrong == false)

        coordinator.insertRawAnyway()

        #expect(pairs.pair(withID: pair.id)?.flaggedWrong == true)
        #expect(store.entries.first?.pairID == pair.id)
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

    // MARK: - Learned Words and the Lens (PRD #612)

    /// A committed take pastes the owner's spelling and reaches the Lens with
    /// its pair, its catches, its app and whether it was pasted.
    @Test
    func aCommittedTakeReachesTheLensWithItsCatchesAndApp() async throws {
        let bundle = try makeFakeModelBundle()
        defer { try? FileManager.default.removeItem(at: bundle) }
        let (pairs, cleanup) = makePairStore()
        defer { cleanup() }
        let words = LearnedWordStore(
            directory: FileManager.default.temporaryDirectory
                .appendingPathComponent("coordinator-learned-\(UUID().uuidString)"))
        words.learn(heard: "sract", meant: "Tesseract")

        let recognizer = InMemorySpeechRecognizer(
            result: TranscriptionResult(
                text: "open the SRACT repo", segments: [], language: "en", processingTime: 0))
        let engine = try await makeEngine(recognizer: recognizer, bundle: bundle)
        let injector = FakeTextInjector()
        let feed = DictationFeed()
        let terminal = TargetApp(bundleID: "com.apple.Terminal", name: "Terminal", pid: 7)
        let store = FakeTranscriptionStore()
        let coordinator = DictationCoordinator(
            audioCapture: FakeAudioCapture(
                cannedAudio: AudioData(samples: [0.1, 0.2], sampleRate: 16_000, duration: 2.0)),
            transcriptionEngine: engine,
            textInjector: injector,
            history: store,
            settings: SettingsManager(store: InMemorySettingsStore()),
            feed: feed,
            pairs: pairs,
            learnedWords: words,
            frontmostApp: { terminal }
        )
        var delivered: [DictatedTake] = []
        coordinator.onTakeCommitted = { take in
            // Handed over right after the paste, before the commit resolves.
            #expect(injector.injected.count == 1)
            delivered.append(take)
        }

        coordinator.onHotkeyDown()
        coordinator.onHotkeyUp()
        try await waitUntil { coordinator.state == .idle && feed.beat != nil }

        #expect(injector.injected == ["Open the Tesseract repo "])
        let take = try #require(delivered.first)
        #expect(take.text == "Open the Tesseract repo")
        #expect(take.pastedText == "Open the Tesseract repo ")
        #expect(take.catches.map(\.meant) == ["Tesseract"])
        #expect(take.catches.first?.tokenStart == 2)
        #expect(take.app == terminal)
        #expect(take.pastedInto == terminal)
        #expect(take.pasted)
        #expect(take.pairID == pairs.pairs.first?.id)
        #expect(pairs.pairs.first?.learned == "Open the Tesseract repo")
        #expect(pairs.pairs.first?.cleaned == "Open the SRACT repo")
        // The history entry keeps the same catches and the app, for the
        // Dictation page.
        #expect(store.entries.count == 1)
        let entry = try #require(store.entries.first)
        #expect(entry.catches == take.catches)
        #expect(entry.pairID == take.pairID)
        #expect(
            entry.app == TranscriptionEntry.App(bundleID: "com.apple.Terminal", name: "Terminal"))
    }

    /// A silent capture is never transcribed: no "Thank you." out of silence.
    @Test
    func aSilentCaptureIsNotTranscribed() async throws {
        let bundle = try makeFakeModelBundle()
        defer { try? FileManager.default.removeItem(at: bundle) }
        let recognizer = InMemorySpeechRecognizer(
            result: TranscriptionResult(
                text: "Thank you.", segments: [], language: "en", processingTime: 0))
        let engine = try await makeEngine(recognizer: recognizer, bundle: bundle)
        let injector = FakeTextInjector()
        let feed = DictationFeed()
        let coordinator = DictationCoordinator(
            audioCapture: FakeAudioCapture(
                cannedAudio: AudioData(
                    samples: [Float](repeating: 0, count: 32_000), sampleRate: 16_000,
                    duration: 2.0)),
            transcriptionEngine: engine,
            textInjector: injector,
            history: FakeTranscriptionStore(),
            settings: SettingsManager(store: InMemorySettingsStore()),
            feed: feed
        )

        coordinator.onHotkeyDown()
        coordinator.onHotkeyUp()

        #expect(coordinator.state == .error(.noSpeechDetected))
        #expect(injector.injected.isEmpty)
        #expect(await recognizer.transcribeCount == 0)
    }

    /// "Insert raw anyway" pastes the Learned Words' spelling, so it counts
    /// its catches and hands the Lens a take that knows them.
    @Test
    func insertingARejectedTakeAnywayCountsAndCarriesItsCatches() async throws {
        let bundle = try makeFakeModelBundle()
        defer { try? FileManager.default.removeItem(at: bundle) }
        let (pairs, cleanup) = makePairStore()
        defer { cleanup() }
        let words = LearnedWordStore(
            directory: FileManager.default.temporaryDirectory
                .appendingPathComponent("coordinator-raw-\(UUID().uuidString)"))
        let id = try #require(words.learn(heard: "sract", meant: "Tesseract")?.wordID)
        let recognizer = InMemorySpeechRecognizer(
            result: TranscriptionResult(
                text: "open the SRACT repo", segments: [], language: "en", processingTime: 0))
        let engine = try await makeEngine(recognizer: recognizer, bundle: bundle)
        let injector = FakeTextInjector()
        let feed = DictationFeed()
        let store = FakeTranscriptionStore()
        let coordinator = DictationCoordinator(
            audioCapture: FakeAudioCapture(
                cannedAudio: AudioData(samples: [0.1, 0.2], sampleRate: 16_000, duration: 2.0)),
            transcriptionEngine: engine, textInjector: injector,
            history: store,
            settings: SettingsManager(store: InMemorySettingsStore()), feed: feed,
            proofreadPass: makeProofreadPass(replying: { _ in "REJECT: mumbling" }),
            pairs: pairs, learnedWords: words, frontmostApp: { nil })
        var delivered: [DictatedTake] = []
        coordinator.onTakeCommitted = { delivered.append($0) }

        coordinator.onHotkeyDown()
        coordinator.onHotkeyUp()
        try await waitUntil { coordinator.lastRejectedRaw != nil }
        #expect(words.word(withID: id)?.totalCatches == 0)
        // A rejected take is not in the history until it is inserted anyway.
        #expect(store.entries.isEmpty)

        coordinator.insertRawAnyway()
        try await waitUntil { !delivered.isEmpty }
        #expect(injector.injected == ["Open the Tesseract repo "])
        #expect(delivered.first?.catches.map(\.meant) == ["Tesseract"])
        #expect(words.word(withID: id)?.totalCatches == 1)
        // The history entry carries the same catches.
        #expect(store.entries.count == 1)
        #expect(store.entries.first?.text == "Open the Tesseract repo")
        #expect(store.entries.first?.catches == delivered.first?.catches)
        #expect(store.entries.first?.catches.first?.tokenStart == 2)
    }

}
