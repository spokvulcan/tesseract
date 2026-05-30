//
//  InMemorySpeechSynthesizer.swift
//  tesseractTests
//
//  A hermetic, in-memory Speech Synthesizer adapter for tests — a *peer
//  implementation* of `Qwen3SpeechSynthesizer`, not a mock. It returns canned
//  samples (and canned streaming chunks), can be programmed with a load/generate
//  error, and records what it saw (loads + the repos passed, generate calls + the
//  text/voice/language/parameters passed, streaming/voice-anchor/cancel/offsets
//  calls). An actor — like the production adapter — so `Sendable` is free and its
//  recorded state is actor-isolated; tests `await` it. No MLX, no model files.
//

import Foundation

@testable import Tesseract_Agent

actor InMemorySpeechSynthesizer: SpeechSynthesizer {
    // MARK: Programmed behavior
    private let cannedSamples: [Float]
    private let cannedSampleRate: Int
    private let loadError: (any Error & Sendable)?
    private let generateError: (any Error & Sendable)?
    private let streamingError: (any Error & Sendable)?
    private let cannedTokenOffsets: [Int]

    // MARK: Recorded state
    private(set) var loadCount = 0
    private(set) var loadedRepos: [String] = []
    private(set) var generateCount = 0
    private(set) var recordedTexts: [String] = []
    private(set) var recordedVoices: [String?] = []
    private(set) var recordedLanguages: [String?] = []
    private(set) var recordedParameters: [TTSParameters] = []
    private(set) var streamingCount = 0
    private(set) var recordedStreamingTexts: [String] = []
    private(set) var recordedStreamingVoices: [String?] = []
    private(set) var recordedStreamingLanguages: [String?] = []
    private(set) var recordedStreamingParameters: [TTSParameters] = []
    private(set) var recordedUseVoiceAnchor: [Bool] = []
    private(set) var buildVoiceAnchorCount = 0
    private(set) var recordedReferenceCounts: [Int] = []
    private(set) var clearVoiceAnchorCount = 0
    private(set) var cancelCount = 0
    private(set) var recordedOffsetTexts: [String] = []

    init(
        samples: [Float] = [0.0],
        sampleRate: Int = 24_000,
        loadError: (any Error & Sendable)? = nil,
        generateError: (any Error & Sendable)? = nil,
        streamingError: (any Error & Sendable)? = nil,
        tokenOffsets: [Int] = []
    ) {
        self.cannedSamples = samples
        self.cannedSampleRate = sampleRate
        self.loadError = loadError
        self.generateError = generateError
        self.streamingError = streamingError
        self.cannedTokenOffsets = tokenOffsets
    }

    func load(modelRepo: String) async throws {
        loadCount += 1
        loadedRepos.append(modelRepo)
        if let loadError { throw loadError }
    }

    func generate(
        text: String,
        voice: String?,
        language: String?,
        parameters: TTSParameters
    ) async throws -> (samples: [Float], sampleRate: Int) {
        generateCount += 1
        recordedTexts.append(text)
        recordedVoices.append(voice)
        recordedLanguages.append(language)
        recordedParameters.append(parameters)
        if let generateError { throw generateError }
        return (cannedSamples, cannedSampleRate)
    }

    func generateStreaming(
        text: String,
        voice: String?,
        language: String?,
        parameters: TTSParameters,
        useVoiceAnchor: Bool
    ) async throws -> (stream: AsyncThrowingStream<[Float], Error>, sampleRate: Int) {
        streamingCount += 1
        recordedStreamingTexts.append(text)
        recordedStreamingVoices.append(voice)
        recordedStreamingLanguages.append(language)
        recordedStreamingParameters.append(parameters)
        recordedUseVoiceAnchor.append(useVoiceAnchor)
        let chunks = [cannedSamples]
        let streamingError = self.streamingError
        let stream = AsyncThrowingStream<[Float], Error> { continuation in
            // A programmed `streamingError` terminates the stream the way the real
            // adapter does (`continuation.finish(throwing:)`) — the failure lands
            // as the stream's terminating error, not at the `generateStreaming`
            // call site.
            if let streamingError {
                continuation.finish(throwing: streamingError)
            } else {
                for chunk in chunks { continuation.yield(chunk) }
                continuation.finish()
            }
        }
        return (stream, cannedSampleRate)
    }

    func buildVoiceAnchor(referenceCount: Int, voice: String?, language: String?) async {
        buildVoiceAnchorCount += 1
        recordedReferenceCounts.append(referenceCount)
    }

    func clearVoiceAnchor() async {
        clearVoiceAnchorCount += 1
    }

    func cancelGeneration() async {
        cancelCount += 1
    }

    func computeTokenCharOffsets(text: String) async -> [Int] {
        recordedOffsetTexts.append(text)
        return cannedTokenOffsets
    }
}
