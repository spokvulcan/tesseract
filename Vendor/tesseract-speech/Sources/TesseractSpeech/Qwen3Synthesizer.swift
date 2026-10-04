// TesseractSpeech — the production Speech Synthesizer adapter over the
// Qwen3-TTS model (ADR-0038, ADR-0071).
//
// Conditioning and the seed are values in every request, and a captured
// Reference Take comes back in the segment's stream (ADR-0072). The model
// keeps state between generations (the instruct-prefix cache keyed by
// description, the rewound KV caches) and runs one generation at a time: the
// engine actor orders them, and a cancelled one still finishing its frame
// holds off the next (ADR-0074).

import Foundation
import MLX
import Qwen3TTS

public actor Qwen3Synthesizer: SpeechSynthesizing {
    private let checkpointDirectory: @Sendable (TTSModelSpec) -> URL
    private let neuralEngineCache: URL?
    /// The phone's voice (ADR-0084): the whole voice on the Neural Engine
    /// once `prepareNeuralVoice` has prepared it, and MLX only ever on the
    /// CPU, since iOS refuses GPU work from a backgrounded app.
    private let neuralVoice: Bool
    private var neuralVoiceReady = false
    private var model: Qwen3TTSModel?
    private var loadedSpec: TTSModelSpec?
    /// The loaded checkpoint's alignment head, when its words can be timed.
    private var alignmentHead: Qwen3TTSAlignmentHead?
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
    /// With `neuralVoice` (the phone), nothing runs until
    /// `prepareNeuralVoice` has moved the voice to the Neural Engine, and MLX
    /// runs on the CPU: loading, the prompt, the codec's front end.
    public init(
        checkpointDirectory: @escaping @Sendable (TTSModelSpec) -> URL,
        neuralEngineCache: URL? = nil, neuralVoice: Bool = false
    ) {
        self.checkpointDirectory = checkpointDirectory
        self.neuralEngineCache = neuralEngineCache
        self.neuralVoice = neuralVoice
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
        let directory = checkpointDirectory(spec)
        let qwen =
            neuralVoice
            ? try await Device.withDefaultDevice(.cpu) {
                try await Qwen3TTSModel.fromModelDirectory(directory)
            }
            : try await Qwen3TTSModel.fromModelDirectory(directory)
        model = qwen
        neuralVoiceReady = false
        loadedSpec = spec
        alignmentHead = spec.alignmentHead.map {
            Qwen3TTSAlignmentHead(layer: $0.layer, head: $0.head)
        }
        warmed = false
        onPhase?(.ready)
    }

    public func warmUp() async throws {
        // The phone's voice warms up as it is prepared, never on the GPU.
        guard let model, !warmed, !neuralVoice else { return }
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

    public func presetSpeakers() async -> [String] {
        model?.presetSpeakers ?? []
    }

    public func unload() async {
        neuralEngineTask?.cancel()
        await neuralEngineTask?.value
        neuralEngineTask = nil
        neuralEngineReport = nil
        let loaded = model != nil
        model = nil
        loadedSpec = nil
        alignmentHead = nil
        warmed = false
        neuralVoiceReady = false
        // The phone's voice touches MLX only once it has loaded: where there
        // is no Neural Engine to load it for (the simulator), MLX can't run.
        guard !neuralVoice || loaded else { return }
        Memory.clearCache()
        if !neuralVoice { Stream.gpu.synchronize() }
    }

    // MARK: - The phone's voice

    /// Whether `prepareNeuralVoice` has finished.
    public var isNeuralVoiceReady: Bool { neuralVoiceReady }

    /// Prepares the voice for the Neural Engine (ADR-0084): loads `spec`'s
    /// checkpoint if it isn't, then builds the codec's conv stack, the talker
    /// and the code predictor from it into the Neural Engine cache (or loads
    /// them, built before) and checks each against MLX the first time. Every
    /// op must land on the Neural Engine, or it throws and the voice isn't
    /// used. Returns what it found.
    public func prepareNeuralVoice(
        _ spec: TTSModelSpec, onPhase: (@Sendable (VoicePreparationPhase) -> Void)? = nil
    ) async throws -> String {
        guard neuralVoice, let cache = neuralEngineCache else {
            throw SpeechEngineError.modelUnavailable("This synthesizer has no Neural Engine voice.")
        }
        onPhase?(.loading)
        try await load(spec, onPhase: nil)
        guard let model else { throw SpeechEngineError.engineUnloaded }
        onPhase?(.building("codec"))
        let codec = try await Device.withDefaultDevice(.cpu) {
            try await model.prepareNeuralEngine(cacheDirectory: cache)
        }
        let voice = try await model.prepareNeuralVoice(
            cacheDirectory: cache,
            alignment: spec.alignmentHead.map { Qwen3TTSAlignmentHead(layer: $0.layer, head: $0.head) }
        ) { phase in
            switch phase {
            case .measuring: onPhase?(.measuring)
            case .building(let part): onPhase?(.building(part))
            case .checking: onPhase?(.checking)
            }
        }
        // The Neural Engine has its own copy now: the MLX one is dead weight
        // on a phone.
        model.releaseMLXVoice()
        neuralVoiceReady = true
        warmed = true
        return "codec: \(codec); \(voice)"
    }

    /// Times one render of `text` in `speaker`'s voice on the prepared
    /// voice: compute seconds per second of audio, and seconds to the first.
    /// The **Speed Check**'s measurement.
    public func timeRender(_ text: String, speaker: String, language: String?) async throws
        -> (realTimeFactor: Double, firstAudio: Double)
    {
        guard neuralVoiceReady, let model else { throw SpeechEngineError.engineUnloaded }
        let clock = ContinuousClock()
        let start = clock.now
        var first: Duration?
        var samples = 0
        for try await event in model.generateStream(
            text: text, voice: speaker, language: language, sampling: Qwen3TTSSampling(),
            seed: 1, streamingInterval: 0.4)
        {
            if case .audio(let audio) = event {
                if first == nil { first = clock.now - start }
                samples += audio.count
            }
        }
        let seconds = (clock.now - start) / .seconds(1)
        let audio = Double(samples) / Double(model.sampleRate)
        guard audio > 0 else { throw SpeechEngineError.generationFailed("The voice made no sound.") }
        return (seconds / audio, (first ?? .zero) / .seconds(1))
    }

    public func audioFormat() async -> AudioFormat? {
        // One codec frame: 1,920 samples at 24 kHz for the 12Hz family.
        guard let model else { return nil }
        return AudioFormat(sampleRate: model.sampleRate, samplesPerFrame: model.samplesPerFrame)
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
                    let alignment = self.alignmentHead

                    // A CustomVoice checkpoint reads a speaker where
                    // VoiceDesign reads a description.
                    let modelStream = model.generateStream(
                        text: request.text,
                        voice: request.speaker ?? request.voiceDescription,
                        language: request.language,
                        reference: request.reference.map {
                            Qwen3TTSReference(codeFrames: $0.codeFrames, text: $0.text)
                        },
                        sampling: Self.sampling(request.parameters),
                        seed: request.seed,
                        // 0.4 s chunks on MLX: how often audio is handed over.
                        // The samples don't depend on it (ADR-0074), and the
                        // Neural Engine decodes its own fixed chunk.
                        streamingInterval: 0.4,
                        alignment: alignment)

                    // A stall or a long trailing silence plays as a gap in the
                    // reading; the cap keeps pauses and drops the excess.
                    var silence = SilenceCap(format: format)
                    var captured: ReferenceTake?
                    // Word timing (ADR-0077): the alignment head's rows come
                    // ahead of their frames' audio; a word's start is sent once
                    // the audio settles it.
                    var timer: WordTimer?
                    func sendStarts() {
                        guard let starts = timer?.takeStarts(), !starts.isEmpty else { return }
                        continuation.yield(.words(starts))
                    }
                    for try await event in modelStream {
                        switch event {
                        case .audio(let audio):
                            var levels: [SilenceCap.Frame] = []
                            let samples = silence.apply(audio, frames: &levels)
                            if !samples.isEmpty {
                                continuation.yield(.chunk(samples))
                            }
                            timer?.appendAudio(levels)
                            sendStarts()
                        case .textTrack(let track):
                            timer = WordTimer(
                                text: request.text, tokenOffsets: track.characterOffsets,
                                referenceTokens: track.referenceTokenCount)
                        case .alignment(let row):
                            timer?.appendAttention(row)
                            sendStarts()
                        case .codeFrames(let frames):
                            if request.capturesReference, !frames.isEmpty {
                                captured = ReferenceTake(codeFrames: frames, text: request.text)
                            }
                        }
                        try Task.checkCancellation()
                    }
                    timer?.finish()
                    sendStarts()
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
