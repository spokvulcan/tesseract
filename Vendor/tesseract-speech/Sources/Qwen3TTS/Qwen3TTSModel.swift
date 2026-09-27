import Foundation
@preconcurrency import MLX
@preconcurrency import MLXLMCommon
import Tokenizers

/// Qwen3-TTS (VoiceDesign and CustomVoice checkpoints): text and a voice in,
/// 24 kHz audio out.
///
/// Three models run per 80 ms frame: the talker picks the frame's first
/// codebook, the code predictor fills in the other fifteen, and the codec
/// decoder turns frames into audio as they accumulate. Generation is
/// pipelined: while the GPU runs one frame, the next frame's graph is built,
/// and the end-of-speech check reads the frame before.
public final class Qwen3TTSModel: @unchecked Sendable {
    let config: Qwen3TTSModelConfig
    let talkerConfig: Qwen3TTSTalkerConfig
    let talker: Qwen3TTSTalker
    let textEmbedding: Qwen3TTSTextEmbedding
    let codecDecoder: Qwen3TTSCodecDecoder
    let tokenizer: Tokenizers.Tokenizer
    let prompts: Qwen3TTSPromptBuilder
    /// The speech tokenizer directory, whose files key the Neural Engine
    /// codec's cache.
    var speechTokenizerDirectory: URL?

    /// The conv stack on the Neural Engine, once prepared; nil runs it in MLX.
    private var neuralCodec: Qwen3TTSNeuralCodec?
    var neuralCodecForBench: Qwen3TTSNeuralCodec? { lock.withLock { neuralCodec } }
    private let neuralQueue = DispatchQueue(label: "qwen3tts.neural-codec", qos: .userInitiated)

    /// The instruct turn's KV for the last description used: every prompt
    /// opens with it, and under causal attention it reads the same whatever
    /// follows.
    private var voicePrefix: (description: String, state: [[MLXArray]])?
    private let lock = NSLock()
    /// Held for as long as a generation or a priming uses the kept KV
    /// caches. A cancelled stream's generation runs on to the end of its
    /// frame after its consumer has gone and the engine's GPU lease has
    /// passed on; the next one waits for it here instead of rewinding
    /// caches it is still writing.
    private let generationLock = NSLock()

    /// Working memory kept across the segments of an utterance: the talker's
    /// and the code predictor's KV caches, rewound for each generation
    /// instead of allocated anew. MLX reuses a freed buffer only for a
    /// request of nearly its size, so a fresh cache per segment (each a
    /// different length) would pile up in the buffer pool.
    private var talkerCache: (caches: [KVCache], capacity: Int)?
    private var codeCache: [KVCache]?

    /// Talker KV positions are reserved in steps of this many.
    static let cacheStep = 256
    /// Prompts longer than this prefill in pieces, which bounds the
    /// prefill's working memory; shorter ones (nearly all) in one pass.
    static let prefillChunk = 512

    public var sampleRate: Int { config.sampleRate }

    /// Audio samples per codec frame, the decoder's upsampling: 1,920 at
    /// 24 kHz, 12.5 frames a second.
    public var samplesPerFrame: Int { codecDecoder.samplesPerFrame }

    init(
        config: Qwen3TTSModelConfig, talker: Qwen3TTSTalker, textEmbedding: Qwen3TTSTextEmbedding,
        codecDecoder: Qwen3TTSCodecDecoder, tokenizer: Tokenizers.Tokenizer
    ) throws {
        self.config = config
        self.talkerConfig = config.talkerConfig ?? .defaults
        self.talker = talker
        self.textEmbedding = textEmbedding
        self.codecDecoder = codecDecoder
        self.tokenizer = tokenizer
        self.prompts = try Qwen3TTSPromptBuilder(
            config: config, tokenizer: tokenizer, talker: talker, textEmbedding: textEmbedding)
    }

    // MARK: - Loading

    /// Loads a VoiceDesign or CustomVoice checkpoint directory: its config,
    /// the talker's safetensors, the text tokenizer and `speech_tokenizer/`.
    /// `decoderPrecision` is the codec decoder's dtype (fp16 by default).
    public static func fromModelDirectory(
        _ directory: URL, decoderPrecision: DType = .float16
    ) async throws -> Qwen3TTSModel {
        let config = try JSONDecoder().decode(
            Qwen3TTSModelConfig.self,
            from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        guard ["voice_design", "custom_voice"].contains(config.ttsModelType) else {
            throw AudioGenerationError.invalidInput(
                "Qwen3-TTS \(config.ttsModelType) checkpoints are not supported: "
                    + "the engine runs VoiceDesign and CustomVoice only.")
        }

        let speechTokenizer = directory.appendingPathComponent("speech_tokenizer")
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: speechTokenizer.path, isDirectory: &isDirectory),
            isDirectory.boolValue
        else {
            // A file where the directory should be is a broken download; the
            // engine never deletes checkpoints, the model catalog repairs them.
            throw AudioGenerationError.modelNotInitialized(
                "\(speechTokenizer.path) is not the speech tokenizer directory.")
        }

        Qwen3TTSWeights.generateTokenizerJSONIfMissing(in: directory)
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        let (talker, textEmbedding) = try Qwen3TTSWeights.loadTalker(
            config: config, directory: directory)
        let decoder = try Qwen3TTSCodecDecoder(
            directory: speechTokenizer, dtype: decoderPrecision)
        let model = try Qwen3TTSModel(
            config: config, talker: talker, textEmbedding: textEmbedding, codecDecoder: decoder,
            tokenizer: tokenizer)
        model.speechTokenizerDirectory = speechTokenizer
        // What loading freed (the files' fp32 and unstacked originals) sits
        // in MLX's buffer pool; hand it back.
        Memory.clearCache()
        return model
    }

    // MARK: - Neural Engine

    /// Whether the codec's conv stack runs on the Neural Engine.
    public var usesNeuralEngine: Bool {
        lock.withLock { neuralCodec != nil }
    }

    /// Moves the codec's conv stack to the Neural Engine: builds the Core ML
    /// model from this checkpoint's weights (or loads the one built before
    /// from `cacheDirectory`), checks that every op lands on the Neural
    /// Engine, and checks its audio against the MLX conv stack. Until it
    /// returns, and whenever it throws, generation uses MLX. Returns what
    /// the compute plan and the check found. `frames` is the chunk the
    /// Neural Engine decodes per call (at most 8: its width limit on M1–M3).
    @discardableResult
    public func prepareNeuralEngine(cacheDirectory: URL, frames: Int = 3) async throws -> String {
        precondition(frames >= 1 && frames <= 8)
        let probe = try lock.withLock { neuralProbe } ?? makeNeuralProbe()
        let (codec, placement) = try await Qwen3TTSNeuralCodec.load(
            decoder: codecDecoder, frames: frames, cacheDirectory: cacheDirectory,
            sourceKey: sourceKey())
        let snr = try agreement(codec, probe)
        guard snr >= 35 else {
            throw AudioGenerationError.modelNotInitialized(
                "The Neural Engine codec's audio differs from MLX's (\(String(format: "%.1f", snr)) dB).")
        }
        lock.withLock { neuralCodec = codec }
        return "\(placement); \(String(format: "%.1f", snr)) dB against MLX"
    }

    /// Back to the MLX conv stack.
    public func disableNeuralEngine() {
        lock.withLock { neuralCodec = nil }
    }

    /// The speech tokenizer weights' identity: names, sizes and dates.
    private func sourceKey() -> String {
        guard let directory = speechTokenizerDirectory else { return "memory" }
        let files =
            ((try? FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: [.fileSizeKey, .contentModificationDateKey]))
            ?? []).filter { $0.pathExtension == "safetensors" }.sorted { $0.path < $1.path }
        return files.map { file in
            let values = try? file.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey])
            return "\(file.lastPathComponent):\(values?.fileSize ?? 0):"
                + "\(values?.contentModificationDate?.timeIntervalSince1970 ?? 0)"
        }.joined(separator: ",")
    }

    /// A latent and MLX's audio for it, to check a Neural Engine codec
    /// against: six frames of random codes, several calls' worth, so the
    /// carried state is checked too. Runs on the GPU; `warmUp` makes it
    /// under the engine's GPU lease.
    struct NeuralProbe {
        let latent: [Float16]
        let frames: Int
        let expected: [Float]
    }

    private var neuralProbe: NeuralProbe?

    private func makeNeuralProbe() throws -> NeuralProbe {
        let frames = 6
        let codes = MLXRandom.randInt(
            Int32(0) ..< Int32(codecDecoder.config.codebookSize),
            [1, frames, codecDecoder.config.numQuantizers], key: MLXRandom.RandomState(seed: 1))
        var stream = codecDecoder.makeStream()
        let latent = codecDecoder.latent(codes, stream: &stream)
        let expected = try codecDecoder.synthesize(latent, stream: &stream).asArray(Float.self)
        return NeuralProbe(
            latent: latent.asType(.float16).asArray(Float16.self), frames: frames,
            expected: expected)
    }

    /// SNR of the Neural Engine conv stack against MLX's on the probe.
    private func agreement(_ codec: Qwen3TTSNeuralCodec, _ probe: NeuralProbe) throws -> Double {
        let dim = codecDecoder.config.latentDim
        codec.reset()
        defer { codec.reset() }
        var actual: [Float] = []
        var start = 0
        while start < probe.frames {
            let valid = min(codec.frames, probe.frames - start)
            var values = Array(probe.latent[(start * dim) ..< ((start + valid) * dim)])
            values += [Float16](repeating: 0, count: (codec.frames - valid) * dim)
            actual += try codec.decode(latent: values, valid: valid)
            start += valid
        }
        var signal = 0.0
        var noise = 0.0
        for (a, e) in zip(actual, probe.expected) {
            signal += Double(e * e)
            noise += Double((a - e) * (a - e))
        }
        return 10 * log10(signal / max(noise, 1e-20))
    }

    // MARK: - Token alignment

    /// The character offset where each of `text`'s tokens starts.
    public func tokenizeForAlignment(text: String) -> [Int] {
        let tokens = tokenizer.encode(text: text)
        guard !tokens.isEmpty else { return [] }
        // Decoded prefixes, not single tokens: a byte-level token can end
        // mid-character.
        var offsets: [Int] = [0]
        offsets.reserveCapacity(tokens.count)
        for i in 1 ..< tokens.count {
            let prefix = tokenizer.decode(tokens: Array(tokens[0 ..< i]))
            offsets.append(min(prefix.count, text.count))
        }
        return offsets
    }

    // MARK: - Generation

    /// Renders the whole utterance, then returns its audio. `voice` is the
    /// VoiceDesign description, or CustomVoice's "speaker, instruction".
    /// `seed` makes a render reproducible.
    public func generate(
        text: String,
        voice: String?,
        language: String?,
        sampling: Qwen3TTSSampling,
        seed: UInt64 = 0,
        layout: Qwen3TTSTextLayout = .interleaved
    ) async throws -> MLXArray {
        let frames = try run(
            prompt: prompt(text: text, voice: voice, language: language, reference: nil, layout: layout),
            voice: voice, sampling: sampling, seed: seed, chunkFrames: nil, onAudio: nil)
        guard !frames.isEmpty else { return MLXArray.zeros([1]) }
        // In pieces: the stream's state makes that exact, and it bounds the
        // decoder's working memory.
        let codes = MLXArray(frames.flatMap { $0 }).reshaped(1, frames.count, frames[0].count)
        var stream = codecDecoder.makeStream()
        var pieces: [MLXArray] = []
        var start = 0
        while start < frames.count {
            let end = min(start + 50, frames.count)
            let audio = try codecDecoder.decode(codes[0..., start ..< end, 0...], stream: &stream)
            eval([audio] + stream.arrays)
            pieces.append(audio)
            start = end
        }
        return concatenated(pieces, axis: 0)
    }

    /// Streams decoded audio as it renders, then the code frames it rendered.
    /// With a `reference`, the voice continues that take: same person, new
    /// words. Audio comes every `streamingInterval` seconds of frames (the
    /// first chunk sooner); the samples never depend on the chunking.
    public func generateStream(
        text: String,
        voice: String?,
        language: String?,
        reference: Qwen3TTSReference? = nil,
        sampling: Qwen3TTSSampling,
        seed: UInt64 = 0,
        streamingInterval: Double = 2.0,
        layout: Qwen3TTSTextLayout = .interleaved
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        let (stream, continuation) = AsyncThrowingStream<AudioGeneration, Error>.makeStream()
        let chunkFrames = max(
            1, Int((streamingInterval * Double(sampleRate) / Double(samplesPerFrame)).rounded()))
        let task = Task { @Sendable [weak self] in
            guard let self else { return }
            do {
                let frames = try run(
                    prompt: prompt(
                        text: text, voice: voice, language: language, reference: reference,
                        layout: layout),
                    voice: voice, sampling: sampling, seed: seed, chunkFrames: chunkFrames,
                    onAudio: { continuation.yield(.audio($0)) })
                continuation.yield(.codeFrames(frames))
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { @Sendable _ in task.cancel() }
        return stream
    }

    /// Builds and caches `description`'s instruct-turn KV, so the next
    /// generation in that voice starts after it.
    public func primeVoice(_ description: String?) throws {
        let instruct = isCustomVoice ? Self.parseCustomVoicePrompt(description)?.instruction : description
        guard let embed = try prompts.instruct(instruct), let description else { return }
        generationLock.lock()
        defer { generationLock.unlock() }
        let cache = acquireTalkerCache(capacity: embed.dim(1))
        restoreOrBuildPrefix(description: description, embed: embed, cache: cache)
    }

    /// Releases the working memory kept between generations (the KV
    /// caches), for the end of an utterance. The weights and the voice
    /// prefix stay.
    public func releaseWorkingMemory() {
        lock.lock()
        talkerCache = nil
        codeCache = nil
        lock.unlock()
    }

    /// The talker's KV caches, rewound, with room for `capacity` positions.
    private func acquireTalkerCache(capacity: Int) -> [KVCache] {
        lock.lock()
        defer { lock.unlock() }
        if let kept = talkerCache, kept.capacity >= capacity {
            for cache in kept.caches { cache.trim(cache.offset) }
            return kept.caches
        }
        let rounded = (capacity + Self.cacheStep - 1) / Self.cacheStep * Self.cacheStep
        let caches = talker.makeCache(capacity: rounded)
        talkerCache = (caches, rounded)
        return caches
    }

    private func acquireCodeCache() -> [KVCache] {
        lock.lock()
        defer { lock.unlock() }
        if let kept = codeCache { return kept }
        let caches = talker.codePredictor.makeCache()
        codeCache = caches
        return caches
    }

    /// Runs `embeds` through the talker in pieces of `prefillChunk`, each a
    /// layer at a time (`Qwen3TTSTalker.prefill`).
    private func prefill(_ embeds: MLXArray, cache: [KVCache]) -> (logits: MLXArray, hidden: MLXArray) {
        let length = embeds.dim(1)
        var start = 0
        var result: (logits: MLXArray, hidden: MLXArray)?
        while start < length {
            let end = min(start + Self.prefillChunk, length)
            result = talker.prefill(embeds[0..., start ..< end, 0...], cache: cache)
            start = end
            if start < length, let result { eval(result.hidden) }
        }
        return result!
    }

    /// A tiny end-to-end generation: compiles the kernels of the talker, the
    /// code predictor and the decoder, so the first real request pays
    /// generation only.
    public func warmUp() throws {
        let prompt = try prompts.plain(
            text: ".", instruct: nil, language: "English", speaker: nil, layout: .interleaved)
        _ = try run(
            prompt: prompt, voice: nil, sampling: Qwen3TTSSampling(maxTokens: 3), seed: 0,
            chunkFrames: 1, onAudio: { _ in })
        // What a Neural Engine codec is checked against, made here on the GPU.
        let probe = try makeNeuralProbe()
        lock.withLock { neuralProbe = probe }
    }

    private var isCustomVoice: Bool { config.ttsModelType == "custom_voice" }

    func prompt(
        text: String, voice: String?, language: String?, reference: Qwen3TTSReference?,
        layout: Qwen3TTSTextLayout
    ) throws -> Qwen3TTSPrompt {
        // CustomVoice reads `voice` as "speaker, instruction"; VoiceDesign's
        // is all description.
        let customVoice = isCustomVoice ? Self.parseCustomVoicePrompt(voice) : nil
        let instruct = isCustomVoice ? customVoice?.instruction : voice
        if let reference {
            return try prompts.reference(
                text: text, take: reference, instruct: instruct, language: language)
        }
        return try prompts.plain(
            text: text, instruct: instruct, language: language, speaker: customVoice?.speaker,
            layout: layout)
    }

    static func parseCustomVoicePrompt(_ voice: String?) -> (speaker: String, instruction: String?)? {
        guard let voice = voice?.trimmingCharacters(in: .whitespacesAndNewlines), !voice.isEmpty
        else { return nil }
        guard let comma = voice.firstIndex(of: ",") else { return (voice, nil) }
        let speaker = voice[..<comma].trimmingCharacters(in: .whitespacesAndNewlines)
        let instruction = voice[voice.index(after: comma)...]
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !speaker.isEmpty else { return (voice, nil) }
        return (speaker, instruction.isEmpty ? nil : instruction)
    }

    // MARK: - Voice prefix cache

    /// Leaves `cache` holding the instruct turn: restored when `description`
    /// is the cached one, else computed and cached.
    private func restoreOrBuildPrefix(description: String, embed: MLXArray, cache: [KVCache]) {
        lock.lock()
        let cached = voicePrefix?.description == description ? voicePrefix?.state : nil
        lock.unlock()
        if let cached {
            // Written into the cache's own buffer, which stays the one the
            // rest of the generation appends to.
            for (layerCache, state) in zip(cache, cached) {
                _ = layerCache.update(keys: state[0], values: state[1])
            }
            return
        }
        _ = prefill(embed, cache: cache)
        // Compact copies: the cache's state is a view into its preallocated
        // buffer, which would otherwise stay alive with the prefix.
        let state = cache.map { $0.state.map { contiguous($0) } }
        eval(state.flatMap { $0 })
        lock.lock()
        voicePrefix = (description, state)
        lock.unlock()
    }

    // MARK: - The frame loop

    /// Runs the talker and the code predictor from `prompt` until EOS or the
    /// frame cap, and returns the frames (EOS excluded), one row of
    /// `numCodeGroups` codes each. With `chunkFrames`, the frames are also
    /// decoded as they arrive and handed to `onAudio`, in order; with the
    /// Neural Engine, from its queue.
    private func run(
        prompt: Qwen3TTSPrompt, voice: String?, sampling: Qwen3TTSSampling, seed: UInt64,
        chunkFrames: Int?, onAudio: (@Sendable ([Float]) -> Void)?
    ) throws -> [[Int32]] {
        generationLock.lock()
        defer { generationLock.unlock() }
        let maxFrames = min(sampling.maxTokens, max(75, prompt.textTokenCount * 6))
        let promptLength = (prompt.instruct?.dim(1) ?? 0) + prompt.body.dim(1)
        // Room for the expected frames (about four per text token); a longer
        // render grows the cache as it goes.
        let cache = acquireTalkerCache(
            capacity: promptLength + min(maxFrames, prompt.textTokenCount * 4 + 32) + 1)
        if let instruct = prompt.instruct, let voice {
            restoreOrBuildPrefix(description: voice, embed: instruct, cache: cache)
        } else if let instruct = prompt.instruct {
            _ = prefill(instruct, cache: cache)
        }

        let random = MLXRandom.RandomState(seed: seed)
        var talkerSampler = Qwen3TTSTalkerSampler(
            sampling: sampling, vocabSize: talkerConfig.vocabSize,
            eosTokenID: talkerConfig.codecEosTokenId, dtype: prompt.body.dtype)
        let detailSampler = Qwen3TTSSampler(
            temperature: sampling.detailTemperature, topK: Qwen3TTSSampling.topK,
            topP: Qwen3TTSSampling.detailTopP)
        let codeCache = acquireCodeCache()
        let eos = MLXArray(Int32(talkerConfig.codecEosTokenId))

        // The talker's step from the prompt; each frame then runs in two
        // dispatches. A: sample the first code from the talker's last step
        // and run the code predictor. B: the talker's next step, which
        // appends to its KV cache. B is encoded only once the previous B has
        // finished, so the cache is updated in place: MLX copies a buffer an
        // unfinished command buffer still reads. A keeps the GPU busy while
        // that wait and B's encoding happen.
        var (logits, hidden) = prefill(prompt.body, cache: cache)
        asyncEval(logits, hidden)

        var accepted: [MLXArray] = []
        accepted.reserveCapacity(maxFrames)
        var pending: (codes: MLXArray, isEOS: MLXArray)?
        var ended = false

        var decoder = codecDecoder.makeStream()
        var decoded = 0
        // A decoded chunk not yet handed over: MLX samples, or a latent for
        // the Neural Engine.
        enum InFlight {
            case audio(MLXArray)
            case latent(MLXArray, valid: Int)
        }
        var inFlight: InFlight?
        let neural = chunkFrames != nil ? lock.withLock { neuralCodec } : nil
        if let neural {
            neural.reset()
            // The Neural Engine has its own copy of the conv stack; MLX's
            // comes back from the checkpoint if it is ever needed again.
            // Released here, where no MLX synthesis can be running.
            codecDecoder.releaseConvStack()
        }
        let neuralFailure = NeuralFailure()
        // A cancelled stream still lets its queued chunks finish before the
        // next stream resets the codec's state.
        defer { if neural != nil { neuralQueue.sync {} } }
        let latentDim = codecDecoder.config.latentDim
        // The Neural Engine decodes a fixed chunk; MLX's first chunk is short,
        // for time to first audio.
        let firstChunk = neural.map { $0.frames } ?? chunkFrames.map { min(3, $0) }
        let steadyChunk = neural.map { $0.frames } ?? chunkFrames

        func deliverAudio() {
            switch inFlight {
            case .audio(let audio):
                onAudio?(audio.asArray(Float.self))
            case .latent(let latent, let valid):
                guard let neural else { break }
                var values = latent.asType(.float16).asArray(Float16.self)
                // The stream's last chunk, padded to the codec's size.
                values += [Float16](repeating: 0, count: (neural.frames - valid) * latentDim)
                neuralQueue.async { [onAudio, values] in
                    guard !neuralFailure.happened else { return }
                    do {
                        onAudio?(try neural.decode(latent: values, valid: valid))
                    } catch {
                        neuralFailure.record(error)
                    }
                }
            case nil:
                break
            }
            inFlight = nil
        }

        func decodeReady(final: Bool) throws {
            guard let steadyChunk, let firstChunk else { return }
            // At most one chunk in flight: hand over the last before the next.
            deliverAudio()
            let ready = accepted.count - decoded
            let due = decoded == 0 ? firstChunk : steadyChunk
            guard ready > 0, final || ready >= due else { return }
            let take = neural.map { min(ready, $0.frames) } ?? ready
            let codes = concatenated(Array(accepted[decoded ..< (decoded + take)]), axis: 0)
                .reshaped(1, take, -1)
            let latent = codecDecoder.latent(codes, stream: &decoder)
            if neural != nil {
                asyncEval([latent] + decoder.arrays)
                inFlight = .latent(latent, valid: take)
            } else {
                let audio = try codecDecoder.synthesize(latent, stream: &decoder)
                asyncEval([audio] + decoder.arrays)
                inFlight = .audio(audio)
            }
            let isFirst = decoded == 0
            decoded += take
            // The first chunk is handed over at once: time to first audio.
            // Later ones wait for the next frame's settle, where the loop
            // waits on the GPU anyway.
            if isFirst { deliverAudio() }
            // A flush can leave more than one Neural Engine chunk.
            if final, decoded < accepted.count { try decodeReady(final: true) }
        }

        /// Frame `previous` is done (its B followed its A); keeps it unless
        /// it is EOS.
        func settle(_ previous: (codes: MLXArray, isEOS: MLXArray)) throws -> Bool {
            if previous.isEOS.item(Bool.self) { return false }
            accepted.append(previous.codes)
            try decodeReady(final: false)
            return true
        }

        for frame in 0 ..< maxFrames {
            try Task.checkCancellation()

            // A: the frame's sixteen codes and the sum of their embeddings.
            let first = talkerSampler(logits, frame: frame, random: random)
            let (rest, embeddingSum) = talker.codePredictor.predict(
                hidden: hidden, firstEmbedding: talker.embedCodec(first), cache: codeCache
            ) { _, logits in detailSampler(logits, random: random) }
            let codes = concatenated([first] + rest, axis: 1)
            let isEOS = first .== eos
            asyncEval(codes, isEOS, embeddingSum, talkerSampler.recent)

            // The previous B is done, and with it the previous frame and
            // any chunk decoded before it.
            eval(logits)
            deliverAudio()
            if let previous = pending, try !settle(previous) {
                ended = true
                break
            }
            pending = (codes, isEOS)

            // B: the talker's step for this frame.
            let text =
                frame < prompt.trailingCount
                ? prompt.trailingText![0..., frame ..< (frame + 1), 0...] : prompt.pad
            (logits, hidden) = talker(embeddingSum + text, cache: cache)
            asyncEval(logits, hidden)
        }
        if !ended, let last = pending {
            eval(last.codes, last.isEOS)
            _ = try settle(last)
        }
        // The last B ran ahead for a frame that won't come; let it finish so
        // the next generation rewinds an idle cache (in place, no copy).
        eval(logits)
        try Task.checkCancellation()

        deliverAudio()
        try decodeReady(final: true)
        deliverAudio()
        if neural != nil {
            neuralQueue.sync {}
            if let error = neuralFailure.error {
                // Back to MLX from the next generation on.
                disableNeuralEngine()
                throw error
            }
        }

        guard !accepted.isEmpty else { return [] }
        let all = concatenated(accepted, axis: 0)
        eval(all)
        let flat = all.asArray(Int32.self)
        let groups = talkerConfig.numCodeGroups
        return (0 ..< accepted.count).map { Array(flat[($0 * groups) ..< (($0 + 1) * groups)]) }
    }
}

/// The first error of a stream's Neural Engine jobs.
private final class NeuralFailure: @unchecked Sendable {
    private let lock = NSLock()
    private var first: Error?

    var happened: Bool { error != nil }

    var error: Error? {
        lock.lock()
        defer { lock.unlock() }
        return first
    }

    func record(_ error: Error) {
        lock.lock()
        if first == nil { first = error }
        lock.unlock()
    }
}
