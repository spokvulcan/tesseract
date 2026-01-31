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
    @Published private(set) var loadedModel: WhisperModel?

    private var whisperActor: WhisperActor?
    private var transcriptionTask: Task<TranscriptionResult, Error>?

    init() {}

    func loadModel(_ model: WhisperModel) async throws {
        // Unload existing model first
        if whisperActor != nil {
            unloadModel()
        }

        // Create the actor and load model
        let actor = WhisperActor()
        try await actor.loadModel(model)

        whisperActor = actor
        isModelLoaded = true
        loadedModel = model
    }

    func unloadModel() {
        whisperActor = nil
        isModelLoaded = false
        loadedModel = nil
    }

    func transcribe(_ audioData: AudioData) async throws -> TranscriptionResult {
        guard let whisperActor else {
            throw DictationError.modelNotLoaded
        }

        guard !audioData.isEmpty else {
            throw DictationError.noSpeechDetected
        }

        isTranscribing = true
        defer { isTranscribing = false }

        return try await whisperActor.transcribe(audioData)
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

    func loadModel(_ model: WhisperModel) async throws {
        whisperKit = try await WhisperKit(
            model: model.rawValue,
            computeOptions: .init(
                audioEncoderCompute: .cpuAndNeuralEngine,
                textDecoderCompute: .cpuAndNeuralEngine
            )
        )
    }

    func transcribe(_ audioData: AudioData) async throws -> TranscriptionResult {
        guard let whisperKit else {
            throw DictationError.modelNotLoaded
        }

        let startTime = Date()

        let options = DecodingOptions(
            task: .transcribe,
            usePrefillPrompt: false,
            skipSpecialTokens: true,
            withoutTimestamps: false,
            clipTimestamps: []
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
            text: firstResult.text.trimmingCharacters(in: .whitespacesAndNewlines),
            segments: segments,
            language: firstResult.language,
            processingTime: processingTime
        )
    }
}
