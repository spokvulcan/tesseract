//
//  SpeechEngine.swift
//  tesseract
//

import Foundation
import Combine
import os
import MLX
import MLXLMCommon
import MLXAudioTTS
import MLXAudioCore

@MainActor
final class SpeechEngine: ObservableObject {
    private enum Defaults {
        static let modelRepo = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
    }

    @Published private(set) var isModelLoaded = false
    @Published private(set) var isLoading = false
    @Published private(set) var loadingStatus: String = ""

    private var ttsActor: TTSActor?

    func loadModel() async throws {
        guard !isModelLoaded, !isLoading else { return }

        isLoading = true
        loadingStatus = "Downloading model..."

        do {
            let actor = TTSActor()
            try await actor.loadModel(repo: Defaults.modelRepo)
            ttsActor = actor
            isModelLoaded = true
            loadingStatus = ""
            Log.speech.info("TTS model loaded successfully")
        } catch {
            loadingStatus = ""
            isLoading = false
            Log.speech.error("Failed to load TTS model: \(error)")
            throw SpeechError.modelLoadFailed(error.localizedDescription)
        }

        isLoading = false
    }

    func unloadModel() {
        ttsActor = nil
        isModelLoaded = false
        loadingStatus = ""
        Log.speech.info("TTS model unloaded")
    }

    func generate(
        text: String,
        voice: String?,
        language: String?,
        parameters: TTSParameters
    ) async throws -> (samples: [Float], sampleRate: Int) {
        guard let actor = ttsActor else {
            throw SpeechError.modelNotLoaded
        }

        return try await actor.generate(
            text: text,
            voice: voice,
            language: language,
            parameters: parameters
        )
    }

    func generateStreaming(
        text: String,
        voice: String?,
        language: String?,
        parameters: TTSParameters
    ) async throws -> (stream: AsyncThrowingStream<[Float], Error>, sampleRate: Int) {
        guard let actor = ttsActor else {
            throw SpeechError.modelNotLoaded
        }

        return try await actor.generateStreaming(
            text: text,
            voice: voice,
            language: language,
            parameters: parameters
        )
    }
}

actor TTSActor {
    private nonisolated(unsafe) var model: (any SpeechGenerationModel)?
    private static let ttsCacheLimitMB: Int = {
        let raw = ProcessInfo.processInfo.environment["QWEN3TTS_CACHE_LIMIT_MB"] ?? ""
        return max(16, Int(raw) ?? 100)
    }()

    func loadModel(repo: String) async throws {
        let loadedModel = try await TTSModelUtils.loadModel(modelRepo: repo)
        model = loadedModel
    }

    private func configureTTSMemoryCacheLimit() {
        Memory.cacheLimit = Self.ttsCacheLimitMB * 1024 * 1024
    }

    func generate(
        text: String,
        voice: String?,
        language: String?,
        parameters: TTSParameters
    ) async throws -> ([Float], Int) {
        guard let model else {
            throw SpeechError.modelNotLoaded
        }

        let genParams = GenerateParameters(
            maxTokens: parameters.maxTokens,
            temperature: parameters.temperature,
            topP: parameters.topP,
            repetitionPenalty: parameters.repetitionPenalty,
            repetitionContextSize: parameters.repetitionContextSize
        )

        configureTTSMemoryCacheLimit()
        let audioArray = try await model.generate(
            text: text,
            voice: voice,
            refAudio: nil,
            refText: nil,
            language: language,
            generationParameters: genParams
        )

        let samples = audioArray.asArray(Float.self)
        Memory.clearCache()
        return (samples, model.sampleRate)
    }

    func generateStreaming(
        text: String,
        voice: String?,
        language: String?,
        parameters: TTSParameters
    ) async throws -> (stream: AsyncThrowingStream<[Float], Error>, sampleRate: Int) {
        guard let model else {
            throw SpeechError.modelNotLoaded
        }

        let genParams = GenerateParameters(
            maxTokens: parameters.maxTokens,
            temperature: parameters.temperature,
            topP: parameters.topP,
            repetitionPenalty: parameters.repetitionPenalty,
            repetitionContextSize: parameters.repetitionContextSize
        )

        configureTTSMemoryCacheLimit()
        let sampleRate = model.sampleRate
        let modelStream = model.generateStream(
            text: text,
            voice: voice,
            refAudio: nil,
            refText: nil,
            language: language,
            generationParameters: genParams
        )

        let outputStream = convertAudioStream(modelStream)
        return (outputStream, sampleRate)
    }

    private func convertAudioStream(
        _ modelStream: AsyncThrowingStream<AudioGeneration, Error>
    ) -> AsyncThrowingStream<[Float], Error> {
        let (stream, continuation) = AsyncThrowingStream<[Float], Error>.makeStream()

        Task {
            do {
                for try await event in modelStream {
                    extractSamples(from: event, continuation: continuation)
                }
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }

        return stream
    }

    private nonisolated func extractSamples(
        from event: AudioGeneration,
        continuation: AsyncThrowingStream<[Float], Error>.Continuation
    ) {
        switch event {
        case .audioChunk(let chunk):
            let samples = chunk.asArray(Float.self)
            if !samples.isEmpty {
                continuation.yield(samples)
            }
        case .audio(let audio):
            let samples = audio.asArray(Float.self)
            if !samples.isEmpty {
                continuation.yield(samples)
            }
        case .token, .info:
            break
        }
    }
}
