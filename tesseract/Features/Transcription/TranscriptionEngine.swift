//
//  TranscriptionEngine.swift
//  tesseract
//

import Foundation
import Combine
import CoreML
import os
@preconcurrency import WhisperKit

@MainActor
protocol Transcribing: AnyObject {
    func transcribe(_ audioData: AudioData, language: String) async throws -> TranscriptionResult
    func cancelTranscription()
}

@MainActor
final class TranscriptionEngine: ObservableObject, Transcribing {
    private enum Defaults {
        static let minimumTranscriptionTimeout: TimeInterval = 30
        static let maximumTranscriptionTimeout: TimeInterval = 240
        static let transcriptionTimeoutMultiplier: Double = 3.0
        static let transcriptionTimeoutOverhead: TimeInterval = 10
    }

    @Published private(set) var isModelLoaded = false
    @Published private(set) var isTranscribing = false

    private var whisperActor: WhisperActor?
    private var transcriptionTask: Task<TranscriptionResult, Error>?
    private var configuredModelPath: URL?

    init() {}

    /// Loads the bundled Whisper model from the specified path.
    func loadModel(from modelPath: URL) async throws {
        configuredModelPath = modelPath

        // Unload existing model first
        if whisperActor != nil {
            unloadModel()
        }

        // Verify model files exist before loading
        let fileManager = FileManager.default
        let encoderPath = modelPath.appendingPathComponent("AudioEncoder.mlmodelc")
        let decoderPath = modelPath.appendingPathComponent("TextDecoder.mlmodelc")

        guard fileManager.fileExists(atPath: encoderPath.path),
              fileManager.fileExists(atPath: decoderPath.path) else {
            Log.transcription.error("Model files not found at: \(modelPath.path)")
            Log.transcription.error("AudioEncoder at: \(encoderPath.path) - exists: \(fileManager.fileExists(atPath: encoderPath.path))")
            Log.transcription.error("TextDecoder at: \(decoderPath.path) - exists: \(fileManager.fileExists(atPath: decoderPath.path))")
            throw DictationError.modelNotLoaded
        }

        // Create the actor and load model
        let actor = WhisperActor()
        try await actor.loadModel(from: modelPath)

        whisperActor = actor
        isModelLoaded = true
    }

    func unloadModel() {
        whisperActor = nil
        isModelLoaded = false
    }

    /// Release Whisper model memory when prioritizing TTS performance.
    func releaseModelForTTSIfIdle() {
        guard !isTranscribing else { return }
        guard whisperActor != nil else { return }
        unloadModel()
        Log.transcription.info("Unloaded Whisper model to prioritize TTS performance")
    }

    func transcribe(_ audioData: AudioData, language: String = "auto") async throws -> TranscriptionResult {
        try await ensureModelLoaded()

        guard let whisperActor else {
            throw DictationError.modelNotLoaded
        }

        guard !audioData.isEmpty else {
            throw DictationError.noSpeechDetected
        }

        isTranscribing = true
        defer { isTranscribing = false }

        // Pass nil for auto-detect, otherwise pass the language code
        let languageCode = language == "auto" ? nil : language
        let timeout = transcriptionTimeout(for: audioData.duration)

        // Race transcription against a timeout to prevent stuck processing state
        return try await withThrowingTaskGroup(of: TranscriptionResult.self) { group in
            group.addTask {
                try await whisperActor.transcribe(audioData, language: languageCode)
            }
            group.addTask {
                try await Task.sleep(for: timeout)
                throw DictationError.transcriptionFailed("Transcription timed out")
            }
            let result = try await group.next()!
            group.cancelAll()
            return result
        }
    }

    private func ensureModelLoaded() async throws {
        guard whisperActor == nil else { return }
        guard let modelPath = configuredModelPath else {
            throw DictationError.modelNotLoaded
        }

        Log.transcription.info("Whisper model not loaded, loading on demand...")
        try await loadModel(from: modelPath)
    }

    func cancelTranscription() {
        transcriptionTask?.cancel()
        transcriptionTask = nil
        isTranscribing = false
    }

    private func transcriptionTimeout(for audioDuration: TimeInterval) -> Duration {
        let estimatedTimeout = (audioDuration * Defaults.transcriptionTimeoutMultiplier)
            + Defaults.transcriptionTimeoutOverhead
        let clampedTimeout = min(
            Defaults.maximumTranscriptionTimeout,
            max(Defaults.minimumTranscriptionTimeout, estimatedTimeout)
        )
        return .seconds(clampedTimeout)
    }
}

// Actor to isolate WhisperKit usage
actor WhisperActor {
    enum Defaults {
        static let noSpeechThreshold: Float = 0.6
    }
    private var whisperKit: WhisperKit?

    func loadModel(from modelPath: URL) async throws {
        let logger = Logger(subsystem: "app.tesseract.agent", category: "transcription")
        logger.info("Loading model from path: \(modelPath.path)")

        // List contents of model folder for debugging
        if let contents = try? FileManager.default.contentsOfDirectory(atPath: modelPath.path) {
            logger.debug("Model folder contents: \(contents)")
        }

        // Configure compute units for each model component
        // GPU is faster for encoding, Neural Engine for decoding
        let computeOptions = ModelComputeOptions(
            melCompute: .cpuAndGPU,
            audioEncoderCompute: .cpuAndGPU,
            textDecoderCompute: .cpuAndNeuralEngine,
            prefillCompute: .cpuAndGPU
        )

        // Load from bundled model path - use the exact folder containing model files
        let config = WhisperKitConfig(
            modelFolder: modelPath.path,
            computeOptions: computeOptions,
            verbose: true,
            prewarm: true,
            load: true,
            download: false
        )

        whisperKit = try await WhisperKit(config)
    }

    func transcribe(_ audioData: AudioData, language: String? = nil) async throws -> TranscriptionResult {
        guard let whisperKit else {
            throw DictationError.modelNotLoaded
        }

        let startTime = Date()

        let options = DecodingOptions(
            task: .transcribe,
            language: language,
            temperature: 0.0,                    // Greedy decoding for deterministic output
            usePrefillPrompt: language != nil,   // Use prefill prompt when language is specified
            skipSpecialTokens: true,
            withoutTimestamps: false,
            clipTimestamps: [],
            noSpeechThreshold: Defaults.noSpeechThreshold
        )

        // Capture whisperKit in a local constant to satisfy concurrency checking
        let kit = whisperKit
        let results = try await kit.transcribe(
            audioArray: audioData.samples,
            decodeOptions: options
        )

        let processingTime = Date().timeIntervalSince(startTime)

        guard let firstResult = results.first else {
            throw DictationError.noSpeechDetected
        }

        let segments = firstResult.segments.map { segment in
            TranscriptionSegment(
                text: segment.text,
                startTime: TimeInterval(segment.start),
                endTime: TimeInterval(segment.end)
            )
        }

        return TranscriptionResult(
            text: firstResult.text.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines),
            segments: segments,
            language: firstResult.language,
            processingTime: processingTime
        )
    }
}
