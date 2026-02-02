//
//  TranscriptionEngine.swift
//  whisper-on-device
//

import Foundation
import Combine
import CoreML
@preconcurrency import WhisperKit

@MainActor
final class TranscriptionEngine: ObservableObject {
    @Published private(set) var isModelLoaded = false
    @Published private(set) var isTranscribing = false

    private var whisperActor: WhisperActor?
    private var transcriptionTask: Task<TranscriptionResult, Error>?

    init() {}

    /// Loads the bundled Whisper model from the specified path.
    func loadModel(from modelPath: URL) async throws {
        // Unload existing model first
        if whisperActor != nil {
            unloadModel()
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

    func transcribe(_ audioData: AudioData, language: String = "auto") async throws -> TranscriptionResult {
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
        return try await whisperActor.transcribe(audioData, language: languageCode)
    }

    func cancelTranscription() {
        transcriptionTask?.cancel()
        transcriptionTask = nil
        isTranscribing = false
    }
}

// Actor to isolate WhisperKit usage
actor WhisperActor {
    private var whisperKit: WhisperKit?

    func loadModel(from modelPath: URL) async throws {
        // Configure compute units for each model component
        // GPU is faster for encoding, Neural Engine for decoding
        let computeOptions = ModelComputeOptions(
            melCompute: .cpuAndGPU,
            audioEncoderCompute: .cpuAndGPU,
            textDecoderCompute: .cpuAndNeuralEngine,
            prefillCompute: .cpuAndGPU
        )

        // Load from bundled model path
        let config = WhisperKitConfig(
            modelFolder: modelPath.path,
            computeOptions: computeOptions,
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

        var options = DecodingOptions(
            task: .transcribe,
            usePrefillPrompt: false,
            skipSpecialTokens: true,
            withoutTimestamps: false,
            clipTimestamps: []
        )
        options.language = language

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
