//
//  TranscriptionEngine.swift
//  tesseract
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

        // Verify model files exist before loading
        let fileManager = FileManager.default
        let encoderPath = modelPath.appendingPathComponent("AudioEncoder.mlmodelc")
        let decoderPath = modelPath.appendingPathComponent("TextDecoder.mlmodelc")

        guard fileManager.fileExists(atPath: encoderPath.path),
              fileManager.fileExists(atPath: decoderPath.path) else {
            print("Model files not found at: \(modelPath.path)")
            print("Looking for AudioEncoder at: \(encoderPath.path) - exists: \(fileManager.fileExists(atPath: encoderPath.path))")
            print("Looking for TextDecoder at: \(decoderPath.path) - exists: \(fileManager.fileExists(atPath: decoderPath.path))")
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
        print("Loading model from path: \(modelPath.path)")

        // List contents of model folder for debugging
        if let contents = try? FileManager.default.contentsOfDirectory(atPath: modelPath.path) {
            print("Model folder contents: \(contents)")
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
            noSpeechThreshold: 0.6               // Standard silence detection threshold
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
