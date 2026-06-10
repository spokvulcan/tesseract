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
import WhisperKit

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

        // Compute units stay on WhisperKit's defaults: mel on GPU, encoder and
        // decoder on the Neural Engine. The ANE encoder is what WhisperKit is
        // architected for, and it keeps ASR off the GPU that the MLX LLM owns —
        // dictation doesn't contend with agent/server generation.
        // Load from bundled model path - use the exact folder containing model files
        let config = WhisperKitConfig(
            modelFolder: modelPath.path,
            verbose: false,
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
            noSpeechThreshold: Defaults.noSpeechThreshold,
            chunkingStrategy: .vad               // Concurrent windows for >30s recordings
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

    /// Cooperative cancellation arrives via `Task` cancellation propagating
    /// into the suspended `transcribe` — WhisperKit checks it between decode
    /// windows, so an in-flight transcription stops at the next window
    /// boundary. This satisfies the port contract; nothing else to tear down.
    func cancel() {}
}
