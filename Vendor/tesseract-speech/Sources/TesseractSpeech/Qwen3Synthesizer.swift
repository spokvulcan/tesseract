// TesseractSpeech — the production Speech Synthesizer adapter over the
// Qwen3-TTS model (ADR-0038, ADR-0071).
//
// Conditioning is a value in every request: the Reference Take goes to the
// model with the segment (ADR-0072). The only model-side state is the
// instruct-prefix cache, keyed by description, which this actor's
// serialization keeps consistent.

import Foundation
import MLX
import MLXLMCommon
import Qwen3TTS

public actor Qwen3Synthesizer: SpeechSynthesizing {
    private let checkpointDirectory: @Sendable (TTSModelSpec) -> URL
    private var model: Qwen3TTSModel?
    private var loadedSpec: TTSModelSpec?
    private var warmed = false

    /// `checkpointDirectory` says where a spec's checkpoint lives on disk. In
    /// the app that's the Model Catalog's directory for the Voice Engine. The
    /// synthesizer only loads from there and never downloads.
    public init(checkpointDirectory: @escaping @Sendable (TTSModelSpec) -> URL) {
        self.checkpointDirectory = checkpointDirectory
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
        // Straight to the local loader. `TTS.loadModel(modelRepo:)` would
        // treat a repo id (or a directory it can't read a config.json from)
        // as something to fetch from the hub.
        let qwen = try await Qwen3TTSModel.fromModelDirectory(checkpointDirectory(spec))
        model = qwen
        loadedSpec = spec
        warmed = false
        onPhase?(.ready)
    }

    public func warmUp() async throws {
        guard let model, !warmed else { return }
        // A tiny end-to-end generation exercises tokenizer materialization,
        // fused-weight eval, and Metal kernel JIT for talker, code predictor,
        // and streaming decoder — so the first real request pays generation
        // only (autopsy F2).
        _ = try? await model.generate(
            text: ".", voice: nil, language: "English",
            sampling: Qwen3TTSSampling(maxTokens: 3))
        warmed = true
    }

    public func primeVoice(description: String?, language: String?) async throws {
        guard let model, let description, !description.isEmpty else { return }
        // The vendor populates its instruct-prefix KV cache (keyed on the
        // description) during generation; a minimal generation primes it off
        // the hot path (autopsy F4).
        _ = try? await model.generate(
            text: ".", voice: description, language: language ?? "English",
            sampling: Qwen3TTSSampling(maxTokens: 2))
    }

    public func unload() async {
        model = nil
        loadedSpec = nil
        warmed = false
        Memory.clearCache()
        Stream.gpu.synchronize()
    }

    public func audioFormat() async -> AudioFormat? {
        guard let model else { return nil }
        // Qwen3-TTS 12Hz family: 12.5 codec frames/s (the K1 one-token-per-
        // frame invariant); 24 kHz → 1,920 samples per frame.
        let sampleRate = model.sampleRate
        return AudioFormat(sampleRate: sampleRate, samplesPerFrame: Int(Double(sampleRate) / 12.5))
    }

    public func alignmentOffsets(for text: String) async throws -> [Int] {
        guard let model else { throw SpeechEngineError.engineUnloaded }
        return model.tokenizeForAlignment(text: text)
    }

    public func trimCaches() async {
        Memory.clearCache()
    }

    // MARK: - Synthesis

    public func synthesizeSegment(_ request: SegmentRequest) async
        -> AsyncThrowingStream<SynthesisEvent, Error>
    {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    guard let model = self.model else { throw SpeechEngineError.engineUnloaded }

                    model.seed = request.seed  // the model seeds MLXRandom per generation
                    let modelStream = model.generateStream(
                        text: request.text,
                        voice: request.voiceDescription,
                        language: request.language,
                        reference: request.reference.map {
                            Qwen3TTSReference(codeFrames: $0.codeFrames, text: $0.text)
                        },
                        sampling: Self.sampling(request.parameters),
                        // 0.4s chunks: pacing/cancel granularity. Not a perf
                        // lever — the 2026-07-13 perf pass measured RTF flat at
                        // interval 2.0, and the streaming decoder is NOT
                        // chunk-size invariant (samples diverge), so changing
                        // this alters output audio at a fixed seed.
                        streamingInterval: 0.4)

                    // A stall or a long trailing silence plays as a gap in the
                    // reading; the cap keeps pauses and drops the excess.
                    var silence = SilenceCap(
                        samplesPerFrame: Int(Double(model.sampleRate) / 12.5))
                    for try await event in modelStream {
                        if case .audio(let audio) = event {
                            let samples = silence.apply(audio.asArray(Float.self))
                            if !samples.isEmpty {
                                continuation.yield(.chunk(samples))
                            }
                        }
                        try Task.checkCancellation()
                    }

                    var captured: ReferenceTake?
                    if request.capturesReference {
                        let frames = model.lastGeneratedCodeFrames
                        if !frames.isEmpty {
                            captured = ReferenceTake(
                                codeFrames: frames,
                                text: request.text,
                                voiceDescription: request.voiceDescription,
                                language: request.language)
                        }
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
    /// temperature, top-k and top-p, and the code predictor's own
    /// temperature for the acoustic detail (ADR-0072).
    static func sampling(_ parameters: TTSParameters) -> Qwen3TTSSampling {
        Qwen3TTSSampling(
            temperature: parameters.temperature,
            topK: parameters.topK,
            topP: parameters.topP,
            repetitionPenalty: parameters.repetitionPenalty,
            detailTemperature: parameters.detailTemperature,
            maxTokens: parameters.maxTokens)
    }
}
