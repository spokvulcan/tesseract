//
//  TranscriptionEngine.swift
//  tesseract
//

import Foundation
import Observation
import os

@MainActor
protocol Transcribing: AnyObject {
    func transcribe(_ audioData: AudioData, language: String) async throws -> TranscriptionResult
    func cancelTranscription()
}

@Observable @MainActor
final class TranscriptionEngine: Transcribing {
    /// Production timeout budget: `duration * 3 + 10s` overhead, clamped to
    /// `[30, 240]`s. Injectable so tests can shrink the budget and assert the
    /// race fires without a 30-second floor; the race itself stays on the facade.
    /// Literals are inlined rather than named constants because this `@Sendable`
    /// closure runs outside the engine's MainActor isolation.
    static let defaultTimeout: @Sendable (_ audioDuration: TimeInterval) -> Duration = {
        audioDuration in
        let estimated = (audioDuration * 3.0) + 10
        let clamped = min(240, max(30, estimated))
        return .seconds(clamped)
    }

    private(set) var isModelLoaded = false
    private(set) var isTranscribing = false

    private let makeRecognizer: @Sendable () -> any SpeechRecognizer
    private let timeout: @Sendable (TimeInterval) -> Duration
    private var recognizer: (any SpeechRecognizer)?
    private var configuredModelPath: URL?

    /// The single in-flight transcription. Occupied synchronously at the start of
    /// `transcribe` (MainActor-atomic), retained so `cancelTranscription` can
    /// cancel it, and cleared by the task's own identity-guarded `defer`.
    private var currentTranscription: Task<TranscriptionResult, Error>?

    /// The in-flight model load. Retained so a transcription that races it
    /// (`ensureModelLoaded` — e.g. a dictation issued while the launch load is
    /// still running) awaits this load instead of starting a second full
    /// WhisperKit load: double ANE prepare, double transient memory.
    private var inFlightLoad: Task<Void, Error>?

    init(
        makeRecognizer: @escaping @Sendable () -> any SpeechRecognizer = {
            WhisperKitSpeechRecognizer()
        },
        timeout: @escaping @Sendable (TimeInterval) -> Duration = TranscriptionEngine.defaultTimeout
    ) {
        self.makeRecognizer = makeRecognizer
        self.timeout = timeout
    }

    /// Loads the bundled Whisper model from the specified path.
    func loadModel(from modelPath: URL) async throws {
        configuredModelPath = modelPath

        // Unload existing model first
        if recognizer != nil {
            unloadModel()
        }

        // Verify model files exist before loading — stays on the facade so the
        // model port never needs real files.
        let missing = WhisperModelContract.missingFiles(at: modelPath)
        guard missing.isEmpty else {
            Log.transcription.error(
                "Model files missing at \(modelPath.path): \(missing.joined(separator: ", "))")
            throw DictationError.modelNotLoaded
        }

        // Build a fresh adapter via the factory and load it. The load runs in a
        // retained task that also publishes the loaded recognizer, so a waiter
        // on `inFlightLoad` observes the recognizer set the moment the task
        // completes — no gap for a duplicate load to slip through.
        let recognizer = makeRecognizer()
        let load = Task { [weak self] in
            try await recognizer.load(modelPath: modelPath)
            guard let self else { return }
            self.recognizer = recognizer
            self.isModelLoaded = true
        }
        inFlightLoad = load
        defer {
            // Identity-guarded: don't clear a successor load started meanwhile.
            if inFlightLoad == load { inFlightLoad = nil }
        }
        try await load.value
    }

    func unloadModel() {
        recognizer = nil
        isModelLoaded = false
    }

    /// Release Whisper model memory when prioritizing TTS performance.
    func releaseModelForTTSIfIdle() {
        guard !isTranscribing else { return }
        guard recognizer != nil else { return }
        unloadModel()
        Log.transcription.info("Unloaded Whisper model to prioritize TTS performance")
    }

    func transcribe(_ audioData: AudioData, language: String = "auto") async throws
        -> TranscriptionResult
    {
        // Single-in-flight: synchronously occupy the slot before the first
        // `await` (MainActor-atomic). A `transcribe` issued while one is in
        // flight is rejected — it does not overlap, queue, or supersede.
        // `isTranscribing` is *not* the lock (it is observable UI state set only
        // after `ensureModelLoaded`'s suspension); the retained `Task` is.
        guard currentTranscription == nil else {
            throw DictationError.transcriptionInProgress
        }

        let task = Task { [weak self] () throws -> TranscriptionResult in
            guard let self else { throw DictationError.modelNotLoaded }
            return try await self.performTranscription(audioData, language: language)
        }
        currentTranscription = task
        defer {
            // Identity-guarded: clear only if the slot still holds *this* task,
            // so a synchronous cancel can't free occupancy early and a stale exit
            // can't clobber a successor.
            if currentTranscription == task { currentTranscription = nil }
        }
        return try await task.value
    }

    private func performTranscription(
        _ audioData: AudioData, language: String
    ) async throws -> TranscriptionResult {
        try await ensureModelLoaded()

        guard let recognizer else {
            throw DictationError.modelNotLoaded
        }

        guard !audioData.isEmpty else {
            throw DictationError.noSpeechDetected
        }

        isTranscribing = true
        defer { isTranscribing = false }

        // Pass nil for auto-detect, otherwise pass the language code
        let languageCode = language == "auto" ? nil : language
        let timeoutDuration = timeout(audioData.duration)

        // The recognizer runs as an *unstructured* task so the budget can
        // abandon it (audit #285 item 10). The previous structured race could
        // not exit until every child yielded — a recognizer hung between
        // cancellation checks (deep in CoreML/ANE code) defeated the timeout
        // entirely and pinned `.processing`. Now the first of {completion,
        // budget expiry, caller cancellation} releases the caller; the orphan
        // is cancelled best-effort, and a result it produces later is dropped
        // both here (the in-flight slot has already cleared) and by the Voice
        // Capture Session's Operation Guard staleness.
        let recognizerTask = Task { () throws -> TranscriptionResult in
            do {
                return try await recognizer.transcribe(audioData, language: languageCode)
            } catch let error as DictationError {
                throw error
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                // Map model-layer failures onto the facade's error vocabulary.
                throw DictationError.transcriptionFailed(error.localizedDescription)
            }
        }

        let race = TranscriptionRace()
        var timeoutTask: Task<Void, Never>?
        defer { timeoutTask?.cancel() }
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                // This closure runs synchronously in the current main-actor
                // job, so `begin` always lands before any resumer below (each
                // is a fresh main-actor job) can finish the race.
                race.begin(continuation)

                Task {
                    let result: Result<TranscriptionResult, any Error>
                    do {
                        result = .success(try await recognizerTask.value)
                    } catch {
                        result = .failure(error)
                    }
                    race.finish(result)
                }

                timeoutTask = Task {
                    guard (try? await Task.sleep(for: timeoutDuration)) != nil else { return }
                    recognizerTask.cancel()
                    race.finish(
                        .failure(DictationError.transcriptionFailed("Transcription timed out")))
                }
            }
        } onCancel: {
            recognizerTask.cancel()
            Task { @MainActor in
                race.finish(.failure(CancellationError()))
            }
        }
    }

    private func ensureModelLoaded() async throws {
        guard recognizer == nil else { return }

        // A load already in flight (the launch load, or a model switch) is
        // awaited, never duplicated.
        if let inFlightLoad {
            try await inFlightLoad.value
            guard recognizer == nil else { return }
        }

        guard let modelPath = configuredModelPath else {
            throw DictationError.modelNotLoaded
        }

        Log.transcription.info("Whisper model not loaded, loading on demand...")
        try await loadModel(from: modelPath)
    }

    /// Synchronous (its call sites in `DictationCoordinator`/`AgentCoordinator`
    /// invoke it fire-and-forget). Cancels the retained in-flight task, which
    /// releases the caller through the race's cancellation arm immediately and
    /// propagates `CancellationError` into the recognizer task cooperatively.
    /// It does **not** null the in-flight slot; the task's identity-guarded
    /// `defer` does, so occupancy stays accurate until the task actually exits.
    /// (The former bridge to the recognizer's `async cancel()` is gone with
    /// the port method — WhisperKit's was an empty body; task cancellation is
    /// the one cancellation channel.)
    func cancelTranscription() {
        currentTranscription?.cancel()
        isTranscribing = false
    }
}

/// One-shot resumption cell for the transcription race: whichever of
/// {recognizer completion, budget expiry, caller cancellation} lands first
/// resumes the caller; later arrivals are dropped. MainActor-confined, so
/// arrivals are serialized without a lock.
@MainActor
private final class TranscriptionRace {
    private var continuation: CheckedContinuation<TranscriptionResult, any Error>?

    func begin(_ continuation: CheckedContinuation<TranscriptionResult, any Error>) {
        self.continuation = continuation
    }

    func finish(_ result: Result<TranscriptionResult, any Error>) {
        continuation?.resume(with: result)
        continuation = nil
    }
}
