// TesseractSpeech — the production Speech Synthesizer adapter over the
// Qwen3-TTS model (ADR-0038, ADR-0071).
//
// Conditioning and the seed are values in every request, and a captured
// Reference Take comes back in the segment's stream (ADR-0072). The model
// keeps state between generations (the instruct-prefix cache keyed by
// description, the rewound KV caches) and runs one generation at a time: the
// engine's GPU lease orders them, and a cancelled one still finishing its
// frame holds off the next (ADR-0074).

import Foundation
import MLX
import Qwen3TTS

public actor Qwen3Synthesizer: SpeechSynthesizing {
    private let checkpointDirectory: @Sendable (TTSModelSpec) -> URL
    private let neuralEngineCache: URL?
    private var model: Qwen3TTSModel?
    private var loadedSpec: TTSModelSpec?
    private var warmed = false
    private var neuralEngineTask: Task<Void, Never>?

    /// What moving the codec's conv stack to the Neural Engine came to:
    /// nil until it finishes; then its placement and check, or why the
    /// synthesizer stayed on MLX (`neuralEngineReady()`).
    private var neuralEngineReport: String?

    /// `checkpointDirectory` says where a spec's checkpoint lives on disk. In
    /// the app that's the Model Catalog's directory for the Voice Engine. The
    /// synthesizer only loads from there and never downloads.
    ///
    /// With `neuralEngineCache`, the codec's conv stack moves to the Neural
    /// Engine after warm-up, in the background: a Core ML model built from
    /// the checkpoint's weights once and kept there (about 140 MB). Until it
    /// is ready, and wherever it can't run, the MLX conv stack decodes.
    /// Without one it stays on MLX, so nothing is written anywhere: the app
    /// passes a directory under its storage root (ADR-0073, ADR-0075).
    public init(
        checkpointDirectory: @escaping @Sendable (TTSModelSpec) -> URL,
        neuralEngineCache: URL? = nil
    ) {
        self.checkpointDirectory = checkpointDirectory
        self.neuralEngineCache = neuralEngineCache
    }

    /// Where the Neural Engine codec is kept under a caches directory.
    public static func neuralEngineCache(in caches: URL) -> URL {
        caches.appendingPathComponent("tesseract-speech/neural-codec", isDirectory: true)
    }

    /// The user's Caches, for tools run outside the app.
    public static var defaultNeuralEngineCache: URL? {
        FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            .map(neuralEngineCache(in:))
    }

    /// Waits for the Neural Engine preparation, if one is running.
    public func neuralEngineReady() async -> String? {
        await neuralEngineTask?.value
        return neuralEngineReport
    }

    // MARK: - Lifecycle

    public func checkAvailable(_ spec: TTSModelSpec) async throws {
        let directory = checkpointDirectory(spec)
        let missing = Qwen3Checkpoint.missingFiles(in: directory)
        guard missing.isEmpty else {
            throw SpeechEngineError.modelUnavailable(
                "\(spec.repo) is not downloaded (\(directory.path) is missing "
                    + "\(missing.joined(separator: ", ")))")
        }
    }

    public func load(_ spec: TTSModelSpec, onPhase: (@Sendable (EnginePhase) -> Void)?) async throws {
        if model != nil, loadedSpec == spec { return }
        // Again here for direct callers: without it a missing speech
        // tokenizer loads "fine" and then can't make sound.
        try await checkAvailable(spec)
        onPhase?(.loadingWeights)
        let qwen = try await Qwen3TTSModel.fromModelDirectory(checkpointDirectory(spec))
        model = qwen
        loadedSpec = spec
        warmed = false
        onPhase?(.ready)
    }

    public func warmUp() async throws {
        guard let model, !warmed else { return }
        // A tiny end-to-end generation compiles the talker's, the code
        // predictor's and the decoder's kernels, so the first real request
        // pays generation only (autopsy F2).
        try model.warmUp(neuralEngine: neuralEngineCache != nil)
        warmed = true
        startNeuralEngine(for: model)
    }

    private func startNeuralEngine(for model: Qwen3TTSModel) {
        guard let cache = neuralEngineCache, neuralEngineTask == nil else { return }
        neuralEngineTask = Task.detached(priority: .utility) { [weak self] in
            let report: String
            do {
                report = try await model.prepareNeuralEngine(cacheDirectory: cache)
            } catch is CancellationError {
                return
            } catch {
                report = "MLX (\(error.localizedDescription))"
            }
            await self?.finishNeuralEngine(report)
        }
    }

    private func finishNeuralEngine(_ report: String) {
        neuralEngineReport = report
    }

    public func primeVoice(description: String?, language: String?) async throws {
        // The description's instruct-turn KV, cached by the model and reused
        // by every generation in that voice: off the hot path (autopsy F4).
        try model?.primeVoice(description)
    }

    public func unload() async {
        neuralEngineTask?.cancel()
        await neuralEngineTask?.value
        neuralEngineTask = nil
        neuralEngineReport = nil
        model = nil
        loadedSpec = nil
        warmed = false
        Memory.clearCache()
        Stream.gpu.synchronize()
    }

    public func audioFormat() async -> AudioFormat? {
        // One codec frame per alignment token (the K1 invariant): 1,920
        // samples at 24 kHz for the 12Hz family.
        guard let model else { return nil }
        return AudioFormat(sampleRate: model.sampleRate, samplesPerFrame: model.samplesPerFrame)
    }

    public func alignmentOffsets(for text: String) async throws -> [Int] {
        guard let model else { throw SpeechEngineError.engineUnloaded }
        return model.tokenizeForAlignment(text: text)
    }

    public func trimCaches() async {
        // The KV caches kept across the utterance's segments, then the pool.
        model?.releaseWorkingMemory()
        Memory.clearCache()
    }

    // MARK: - Synthesis

    public func synthesizeSegment(_ request: SegmentRequest) async
        -> AsyncThrowingStream<SynthesisEvent, Error>
    {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    guard let model = self.model, let format = await self.audioFormat() else {
                        throw SpeechEngineError.engineUnloaded
                    }

                    let modelStream = model.generateStream(
                        text: request.text,
                        voice: request.voiceDescription,
                        language: request.language,
                        reference: request.reference.map {
                            Qwen3TTSReference(codeFrames: $0.codeFrames, text: $0.text)
                        },
                        sampling: Self.sampling(request.parameters),
                        seed: request.seed,
                        // 0.4 s chunks on MLX: how often audio is handed over.
                        // The samples don't depend on it (ADR-0074), and the
                        // Neural Engine decodes its own fixed chunk.
                        streamingInterval: 0.4)

                    // A stall or a long trailing silence plays as a gap in the
                    // reading; the cap keeps pauses and drops the excess.
                    var silence = SilenceCap(format: format)
                    var captured: ReferenceTake?
                    for try await event in modelStream {
                        switch event {
                        case .audio(let audio):
                            let samples = silence.apply(audio)
                            if !samples.isEmpty {
                                continuation.yield(.chunk(samples))
                            }
                        case .codeFrames(let frames):
                            if request.capturesReference, !frames.isEmpty {
                                captured = ReferenceTake(codeFrames: frames, text: request.text)
                            }
                        }
                        try Task.checkCancellation()
                    }
                    continuation.yield(.done(capturedReference: captured))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// The engine's parameters as the model's sampling: the talker's
    /// temperature and top-p, and the code predictor's own temperature for
    /// the acoustic detail (ADR-0072).
    static func sampling(_ parameters: TTSParameters) -> Qwen3TTSSampling {
        Qwen3TTSSampling(
            temperature: parameters.temperature,
            topP: parameters.topP,
            repetitionPenalty: parameters.repetitionPenalty,
            detailTemperature: parameters.detailTemperature,
            maxTokens: parameters.maxTokens)
    }
}
