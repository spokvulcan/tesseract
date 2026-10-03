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

        // The audio encoder runs on the GPU; mel stays on the GPU and the text
        // decoder on the Neural Engine (WhisperKit's defaults). Measured on an
        // M3 Max over 100 real takes, in sections 1.6 and 1.7 of
        // docs/research/2026-10-03-dictation-streaming.md:
        // - On the Neural Engine every decode spends about 1.1 s in the encoder
        //   whatever the audio length (it always encodes a 30 s window); on the
        //   GPU 0.4 s, after a one-time shader compile of about 4 s during the
        //   load (with `prewarm`), not during a take.
        // - The full pass lands 0.65 s after release instead of 1.29 s, with the
        //   same text within run-to-run noise, and the Live Preview runs about
        //   1 s behind the voice instead of 2 s.
        // - The cost lands on a generation running at the same time: the 27B
        //   model slows from 21.8 to 14.8 tokens/s while Whisper decodes (about
        //   15% over a take), and dictation is still faster on the GPU than on
        //   the Neural Engine with the LLM busy.
        // Models share memory, not turns (ADR-0081); the Lens streams and the
        // paste stays a full pass (ADR-0085). If the slowdown matters, the
        // fallback is a second encoder on the Neural Engine while the LLM Gate
        // is held, for about 1.2 GB more memory.
        // Load from bundled model path - use the exact folder containing model files
        let config = WhisperKitConfig(
            modelFolder: modelPath.path,
            computeOptions: ModelComputeOptions(audioEncoderCompute: .cpuAndGPU),
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

        // The capture arrives at its native rate; convert to Whisper's 16 kHz
        // here, on this actor — the key-release path (which sits under the
        // app's system-wide event tap) must not pay for it on the main thread.
        let samples =
            audioData.sampleRate == AudioConverter.whisperSampleRate
            ? audioData.samples
            : AudioConverter.resample(
                audioData.samples,
                from: audioData.sampleRate,
                to: AudioConverter.whisperSampleRate)

        let options = DecodingOptions(
            task: .transcribe,
            language: language,
            temperature: 0.0,  // Greedy decoding for deterministic output
            usePrefillPrompt: language != nil,  // Use prefill prompt when language is specified
            skipSpecialTokens: true,
            withoutTimestamps: false,
            clipTimestamps: [],
            noSpeechThreshold: Defaults.noSpeechThreshold,
            chunkingStrategy: .vad  // Concurrent windows for >30s recordings
        )

        // Capture whisperKit in a local constant to satisfy concurrency checking
        let kit = whisperKit
        let results = try await kit.transcribe(
            audioArray: samples,
            decodeOptions: options
        )

        let processingTime = Date().timeIntervalSince(startTime)

        // With `.vad` chunking, recordings longer than one 30s window come back
        // as one result per chunk (in order, segment timings already rebased to
        // the full recording) — merge them all, or everything after the first
        // chunk is silently dropped.
        guard !results.isEmpty else {
            throw DictationError.noSpeechDetected
        }

        let segments = results.flatMap(\.segments).map { segment in
            TranscriptionSegment(
                text: segment.text,
                startTime: TimeInterval(segment.start),
                endTime: TimeInterval(segment.end)
            )
        }

        let text =
            results
            .map { $0.text.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: " ")

        return TranscriptionResult(
            text: text,
            segments: segments,
            language: results[0].language,
            processingTime: processingTime
        )
    }

    // Cooperative cancellation arrives via `Task` cancellation propagating
    // into the suspended `transcribe`. WhisperKit checks it before the mel,
    // before the encoder and before every decoder step, so an in-flight
    // transcription stops within the encoder pass already running. That is
    // the port's one cancellation channel.
}
