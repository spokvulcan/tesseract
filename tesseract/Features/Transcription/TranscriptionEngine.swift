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

    /// The tracked task bridging the synchronous `cancelTranscription()` to the
    /// recognizer's `async cancel()`.
    private var recognizerCancelTask: Task<Void, Never>?

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
        let fileManager = FileManager.default
        let encoderPath = modelPath.appendingPathComponent("AudioEncoder.mlmodelc")
        let decoderPath = modelPath.appendingPathComponent("TextDecoder.mlmodelc")

        guard fileManager.fileExists(atPath: encoderPath.path),
            fileManager.fileExists(atPath: decoderPath.path)
        else {
            Log.transcription.error("Model files not found at: \(modelPath.path)")
            Log.transcription.error(
                "AudioEncoder at: \(encoderPath.path) - exists: \(fileManager.fileExists(atPath: encoderPath.path))"
            )
            Log.transcription.error(
                "TextDecoder at: \(decoderPath.path) - exists: \(fileManager.fileExists(atPath: decoderPath.path))"
            )
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

        // Race transcription against a timeout to prevent stuck processing state
        return try await withThrowingTaskGroup(of: TranscriptionResult.self) { group in
            defer { group.cancelAll() }

            group.addTask {
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
            group.addTask {
                try await Task.sleep(for: timeoutDuration)
                throw DictationError.transcriptionFailed("Transcription timed out")
            }
            let result = try await group.next()!
            try Task.checkCancellation()
            return result
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
    /// invoke it fire-and-forget). Cancels the retained in-flight task — which
    /// propagates `CancellationError` into the task group and the adapter's
    /// `transcribe` — and fires a tracked task to call the recognizer's `async
    /// cancel()` for cooperative model teardown. It does **not** null the
    /// in-flight slot; the task's identity-guarded `defer` does, so occupancy
    /// stays accurate until the task actually exits.
    func cancelTranscription() {
        currentTranscription?.cancel()
        if let recognizer {
            recognizerCancelTask = Task { await recognizer.cancel() }
        }
        isTranscribing = false
    }
}
