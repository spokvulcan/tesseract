// TesseractSpeech — the production Speech Synthesizer adapter over the
// re-vendored mlx-audio-swift Qwen3-TTS model (ADR-0036/0038).
//
// Value semantics at the port, vendor state inside: the vendor keeps one
// anchor slot and one prefix cache as mutable model fields; this actor
// serializes access and installs/rebuilds them so that, observed through the
// port, conditioning behaves as request-scoped values.

import Foundation
import MLX
import MLXAudioCore
import MLXAudioTTS
import MLXLMCommon

public actor Qwen3Synthesizer: SpeechSynthesizing {
    private var model: Qwen3TTSModel?
    private var loadedSpec: TTSModelSpec?
    private var installedAnchor: AnchorHandle?
    private var warmed = false

    public init() {}

    // MARK: - Lifecycle

    public func load(_ spec: TTSModelSpec, onPhase: (@Sendable (EnginePhase) -> Void)?) async throws {
        if model != nil, loadedSpec == spec { return }
        onPhase?(.loadingWeights)
        let loaded = try await TTSModelUtils.loadModel(modelRepo: spec.repo)
        guard let qwen = loaded as? Qwen3TTSModel else {
            throw SpeechEngineError.modelUnavailable(
                "loaded model for \(spec.repo) is not a Qwen3-TTS model")
        }
        model = qwen
        loadedSpec = spec
        installedAnchor = nil
        warmed = false
        onPhase?(.ready)
    }

    public func warmUp() async throws {
        guard let model, !warmed else { return }
        // A tiny end-to-end generation exercises tokenizer materialization,
        // fused-weight eval, and Metal kernel JIT for talker, code predictor,
        // and streaming decoder — so the first real request pays generation
        // only (autopsy F2).
        let params = GenerateParameters(
            maxTokens: 3, temperature: 0.9, topP: 1.0, repetitionPenalty: 1.05)
        _ = try? await model.generate(
            text: ".", voice: nil, refAudio: nil, refText: nil,
            language: "English", generationParameters: params)
        warmed = true
    }

    public func primeVoice(description: String?, language: String?) async throws {
        guard let model, let description, !description.isEmpty else { return }
        // The vendor populates its instruct-prefix KV cache (keyed on the
        // description) during generation; a minimal generation primes it off
        // the hot path (autopsy F4).
        let params = GenerateParameters(
            maxTokens: 2, temperature: 0.9, topP: 1.0, repetitionPenalty: 1.05)
        _ = try? await model.generate(
            text: ".", voice: description, refAudio: nil, refText: nil,
            language: language ?? "English", generationParameters: params)
    }

    public func unload() async {
        model = nil
        loadedSpec = nil
        installedAnchor = nil
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
                    try await self.installConditioning(request.anchor, model: model)

                    model.seed = request.seed  // vendor seeds MLXRandom per generation
                    let params = GenerateParameters(
                        maxTokens: request.parameters.maxTokens,
                        temperature: request.parameters.temperature,
                        topP: request.parameters.topP,
                        repetitionPenalty: request.parameters.repetitionPenalty)

                    let vendorStream = model.generateStream(
                        text: request.text,
                        voice: request.voiceDescription,
                        refAudio: nil,
                        refText: nil,
                        language: request.language,
                        generationParameters: params,
                        // 0.4s chunks: pacing/cancel granularity. Not a perf
                        // lever — the 2026-07-13 perf pass measured RTF flat at
                        // interval 2.0, and the streaming decoder is NOT
                        // chunk-size invariant (samples diverge), so changing
                        // this alters output audio at a fixed seed.
                        streamingInterval: 0.4,
                        useVoiceAnchor: request.anchor != nil)

                    for try await event in vendorStream {
                        if case .audio(let audio) = event {
                            let samples = audio.asArray(Float.self)
                            if !samples.isEmpty {
                                continuation.yield(.chunk(samples))
                            }
                        }
                        try Task.checkCancellation()
                    }

                    var captured: AnchorHandle?
                    if let steps = request.captureAnchorSteps {
                        captured = await self.captureAnchor(
                            steps: steps, request: request, model: model)
                    }
                    continuation.yield(.done(capturedAnchor: captured))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    // MARK: - Anchor plumbing (value semantics over the vendor's single slot)

    private func installConditioning(_ anchor: AnchorHandle?, model: Qwen3TTSModel) async throws {
        guard anchor != installedAnchor else { return }
        if let anchor {
            model.buildVoiceAnchor(
                fromCodeFrames: anchor.codeFrames,
                referenceCount: anchor.codeFrames.count,
                instruct: anchor.voiceDescription,
                language: anchor.language)
        } else {
            model.clearVoiceAnchor()
        }
        installedAnchor = anchor
    }

    private func captureAnchor(
        steps: Int, request: SegmentRequest, model: Qwen3TTSModel
    ) async -> AnchorHandle? {
        let frames = Array(model.lastGeneratedCodeFrames.prefix(steps))
        guard !frames.isEmpty else { return nil }
        model.buildVoiceAnchor(
            referenceCount: frames.count,
            instruct: request.voiceDescription,
            language: request.language)
        let handle = AnchorHandle(
            codeFrames: frames,
            voiceDescription: request.voiceDescription,
            language: request.language)
        installedAnchor = handle
        return handle
    }
}
