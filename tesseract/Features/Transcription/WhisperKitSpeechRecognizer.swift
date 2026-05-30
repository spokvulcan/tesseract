//
//  WhisperKitSpeechRecognizer.swift
//  tesseract
//
//  The framework-backed Speech Recognizer adapter — the only production code that
//  touches WhisperKit for ASR. Formerly `WhisperActor`. An actor, so `Sendable`
//  is free and it satisfies the `@Sendable` capture in the engine's timeout race.
//

import Foundation
import CoreML
import os
@preconcurrency import WhisperKit

actor WhisperKitSpeechRecognizer: SpeechRecognizer {
    enum Defaults {
        static let noSpeechThreshold: Float = 0.6
    }
    private var whisperKit: WhisperKit?

    func load(modelPath: URL) async throws {
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

    func transcribe(_ audioData: AudioData, language: String?) async throws -> TranscriptionResult {
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

    /// WhisperKit exposes no in-flight cancellation hook; cooperative
    /// cancellation arrives via `Task` cancellation propagating into the
    /// suspended `transcribe`. This satisfies the port contract and is where
    /// model-side teardown would live if WhisperKit gained one.
    func cancel() {}
}
