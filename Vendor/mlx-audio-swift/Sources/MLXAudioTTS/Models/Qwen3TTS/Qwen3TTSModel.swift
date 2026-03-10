// Port of mlx_audio/tts/models/qwen3_tts/qwen3_tts.py
// Main Qwen3-TTS conditional generation model (VoiceDesign path)

@preconcurrency import MLX
import MLXNN
@preconcurrency import MLXLMCommon
import MLXAudioCore
import HuggingFace
import Tokenizers
import Foundation
import os

private let logger = Logger(subsystem: "app.tesseract.agent", category: "speech")

// MARK: - Step Timings (profiling accumulator)

struct StepTimings {
    var talkerForward: Double = 0
    var mainSampling: Double = 0
    var eosCheck: Double = 0
    var codePredictorTotal: Double = 0
    var codePredictorPasses: Double = 0  // sum of individual pass times
    var embeddingPrep: Double = 0
    var stepTotal: Double = 0
    var stepCount: Int = 0
    var codePredictorPassCount: Int = 0
}

// MARK: - Qwen3TTS Full Model

public final class Qwen3TTSFullModel: Module, SpeechGenerationModel, @unchecked Sendable {
    let config: Qwen3TTSModelConfig
    let talker: Qwen3TTSTalkerForConditionalGeneration
    var speechTokenizer: Qwen3TTSSpeechTokenizer?
    var tokenizer: Tokenizer?

    // Voice prefix KV cache: instruct tokens are text-independent under causal attention,
    // so we cache them once per voice description and restore for each new text.
    private var voicePrefixKVState: [[MLXArray]]?
    private var cachedVoiceDescription: String?

    // Voice anchor KV cache: extends instruct cache with codec-level voice examples
    // from a previously generated segment, anchoring subsequent segments to the same voice.
    private var voiceAnchorKVState: [[MLXArray]]?
    private var voiceAnchorCodecCount: Int = 0

    /// Codec codes from the most recent generation, used to build voice anchor.
    public private(set) var lastGeneratedCodes: [MLXArray]?

    /// Active generation task, stored so it can be cancelled externally.
    private var activeGenerationTask: Task<Void, Never>?

    /// Random seed for deterministic generation. Set by caller before generate/generateStream.
    public var seed: UInt64 = 0

    /// Enable per-component profiling. Use --qwen3tts-profile launch arg or QWEN3TTS_PROFILE=1 env var.
    /// Profiling forces sync eval() at every sub-step, preventing lazy graph fusion
    /// and adding ~18 GPU sync points per step (vs 2 normally). Expect ~2-3x slowdown.
    static let profilingEnabled: Bool = {
        ProcessInfo.processInfo.arguments.contains("--qwen3tts-profile")
            || ProcessInfo.processInfo.environment["QWEN3TTS_PROFILE"] == "1"
    }()

    /// Experimental: cache prefill + single-token first code predictor pass.
    /// Disabled by default (direct 2-token first pass benchmarks faster overall).
    /// Set QWEN3TTS_CP_PREFILL=1 to enable.
    static let codePredictorPrefillEnabled: Bool = {
        ProcessInfo.processInfo.environment["QWEN3TTS_CP_PREFILL"] == "1"
    }()

    /// Use lightweight sampler for code predictor passes.
    /// Set QWEN3TTS_CODE_SAMPLER=0 to force legacy sampleToken() path.
    static let optimizedCodeSamplerEnabled: Bool = {
        ProcessInfo.processInfo.environment["QWEN3TTS_CODE_SAMPLER"] != "0"
    }()

    /// Clear temporary GPU cache every N generation steps.
    /// Set `QWEN3TTS_CLEAR_CACHE_EVERY` to tune; default is 4.
    static let clearCacheEverySteps: Int = {
        let raw = ProcessInfo.processInfo.environment["QWEN3TTS_CLEAR_CACHE_EVERY"] ?? ""
        return max(1, Int(raw) ?? 4)
    }()

    /// Mirror perf logs to stdout (useful for headless CLI benchmarking).
    static let stdoutPerfLogsEnabled: Bool = {
        ProcessInfo.processInfo.environment["QWEN3TTS_STDOUT_LOG"] == "1"
    }()

    /// Final decode strategy:
    /// - `chunked` (default): tokenizer `decode()` with larger internal chunks
    /// - `stream`: tokenizer `streamingDecode()` with chunkTokens=100
    /// - `single`: one decoder pass via `decodeSingleChunk()` (highest memory)
    static let finalDecodeMode: String = {
        let mode = (ProcessInfo.processInfo.environment["QWEN3TTS_DECODE_MODE"] ?? "chunked").lowercased()
        if mode == "stream" || mode == "single" {
            return mode
        }
        return "chunked"
    }()

    public var sampleRate: Int { config.sampleRate }

    // MARK: - Voice prefix cache helpers

    private func saveVoicePrefixCache(_ cache: [KVCacheSimple]) {
        voicePrefixKVState = cache.map { $0.state }
        eval(voicePrefixKVState!.flatMap { $0 })  // Materialize, detach from lazy graph
    }

    private func restoreVoicePrefixCache(into cache: [KVCacheSimple]) {
        guard let saved = voicePrefixKVState else { return }
        for (i, layerState) in saved.enumerated() {
            cache[i].state = layerState  // .state setter auto-sets offset from keys.dim(2)
        }
    }

    // MARK: - Voice anchor cache helpers

    private func saveVoiceAnchorCache(_ cache: [KVCacheSimple]) {
        voiceAnchorKVState = cache.map { $0.state }
        eval(voiceAnchorKVState!.flatMap { $0 })
    }

    private func restoreVoiceAnchorCache(into cache: [KVCacheSimple]) {
        guard let saved = voiceAnchorKVState else { return }
        for (i, layerState) in saved.enumerated() {
            cache[i].state = layerState
        }
    }

    public func buildVoiceAnchor(
        referenceCount: Int,
        instruct: String?,
        language: String?
    ) {
        guard let codes = lastGeneratedCodes, !codes.isEmpty else {
            logger.info("No generated codes available for voice anchor")
            return
        }

        let refCount = min(referenceCount, codes.count)
        guard refCount > 0 else { return }

        let talkerConfig = config.talkerConfig!
        let inputEmbedding = talker.getInputEmbeddings()
        let codeEmbeddings = talker.codePredictor.codecEmbedding

        // Start from instruct-only KV cache
        let cache = talker.makeCache()

        // Ensure instruct prefix is cached
        if cachedVoiceDescription != instruct || voicePrefixKVState == nil {
            let prepared = prepareGenerationInputs(
                text: ".", language: language ?? "auto", instruct: instruct
            )
            if let instructEmbed = prepared.instructEmbed {
                let _ = talker(instructEmbed, cache: cache)
                saveVoicePrefixCache(cache)
                cachedVoiceDescription = instruct
            }
        } else {
            restoreVoicePrefixCache(into: cache)
        }

        // Build codec prompt embeddings from reference codes
        // Each code step has shape [1, numCodeGroups], we embed all codebooks and sum
        let refCodes = Array(codes.prefix(refCount))
        let ttsPadEmbed: MLXArray = {
            let padTokens = MLXArray([Int32(config.ttsPadTokenId)]).reshaped(1, 1)
            return talker.textProjection(talker.getTextEmbeddings()(padTokens))
        }()

        var codecPromptEmbeds = [MLXArray]()
        codecPromptEmbeds.reserveCapacity(refCount)

        for stepCodes in refCodes {
            // stepCodes: [1, numCodeGroups]
            // Embed code0 with talker input embedding, remaining with code predictor embeddings
            var embed = inputEmbedding(stepCodes[0..., 0..<1])  // code0
            for codeIdx in 0..<(talkerConfig.numCodeGroups - 1) {
                let codeToken = stepCodes[0..., (codeIdx + 1)..<(codeIdx + 2)]
                embed = embed + codeEmbeddings[codeIdx](codeToken)
            }
            // Add ttsPadEmbed (same as during generation when text is exhausted)
            embed = embed + ttsPadEmbed
            codecPromptEmbeds.append(embed)
        }

        // Concatenate all codec prompt steps and forward through talker
        let codecPrompt = concatenated(codecPromptEmbeds, axis: 1)  // [1, refCount, hidden]
        let _ = talker(codecPrompt, cache: cache)

        // Save the extended KV state (instruct + codec prompt)
        saveVoiceAnchorCache(cache)
        voiceAnchorCodecCount = refCount

        let audioSeconds = String(format: "%.1f", Double(refCount) / 12.0)
        logger.info("Voice anchor built: \(refCount, privacy: .public) codec steps (\(audioSeconds, privacy: .public)s) from first segment")
    }

    public func clearVoiceAnchor() {
        voiceAnchorKVState = nil
        voiceAnchorCodecCount = 0
        lastGeneratedCodes = nil
        logger.info("Voice anchor cleared")
    }

    init(config: Qwen3TTSModelConfig) {
        let talkerConfig = config.talkerConfig ?? {
            let json = "{}".data(using: .utf8)!
            return try! JSONDecoder().decode(Qwen3TTSTalkerConfig.self, from: json)
        }()
        self.config = config
        self.talker = Qwen3TTSTalkerForConditionalGeneration(config: talkerConfig)
    }

    // MARK: - SpeechGenerationModel protocol

    public func generate(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        guard speechTokenizer != nil else {
            throw AudioGenerationError.modelNotInitialized("Speech tokenizer not loaded")
        }
        guard tokenizer != nil else {
            throw AudioGenerationError.modelNotInitialized("Text tokenizer not loaded")
        }

        // VoiceDesign: voice parameter is the instruct (voice description)
        let instruct = voice
        let lang = language ?? "auto"
        let temp = generationParameters.temperature
        let topP = generationParameters.topP
        let repPenalty = generationParameters.repetitionPenalty ?? 1.05
        let maxTokens = generationParameters.maxTokens ?? 4096

        let audio = generateVoiceDesign(
            text: text,
            instruct: instruct,
            language: lang,
            temperature: temp,
            topP: topP,
            repetitionPenalty: repPenalty,
            maxTokens: maxTokens
        )
        return audio
    }

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        generateStream(
            text: text, voice: voice, refAudio: refAudio,
            refText: refText, language: language,
            generationParameters: generationParameters,
            useVoiceAnchor: false
        )
    }

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters,
        useVoiceAnchor: Bool
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        let (stream, continuation) = AsyncThrowingStream<AudioGeneration, Error>.makeStream()
        let capturedSeed = self.seed
        let task = Task { @Sendable [weak self] in
            guard let self else { return }
            do {
                guard self.speechTokenizer != nil else {
                    throw AudioGenerationError.modelNotInitialized("Speech tokenizer not loaded")
                }
                guard self.tokenizer != nil else {
                    throw AudioGenerationError.modelNotInitialized("Text tokenizer not loaded")
                }

                // Set seed inside Task to guarantee deterministic sampling
                // (unstructured Tasks don't inherit caller's random state)
                MLXRandom.seed(capturedSeed)

                let instruct = voice
                let lang = language ?? "auto"
                let temp = generationParameters.temperature
                let topP = generationParameters.topP
                let repPenalty = generationParameters.repetitionPenalty ?? 1.05
                let maxTokens = generationParameters.maxTokens ?? 4096

                self.generateStreamingVoiceDesign(
                    text: text,
                    instruct: instruct,
                    language: lang,
                    temperature: temp,
                    topP: topP,
                    repetitionPenalty: repPenalty,
                    maxTokens: maxTokens,
                    useVoiceAnchor: useVoiceAnchor,
                    onAudioChunk: { chunkSamples in
                        let arr = MLXArray(chunkSamples)
                        continuation.yield(.audioChunk(arr))
                    }
                )
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        activeGenerationTask = task
        return stream
    }

    public func cancelGeneration() {
        activeGenerationTask?.cancel()
        activeGenerationTask = nil
    }

    // MARK: - VoiceDesign generation

    func generateVoiceDesign(
        text: String,
        instruct: String?,
        language: String,
        temperature: Float,
        topP: Float,
        repetitionPenalty: Float,
        maxTokens: Int
    ) -> MLXArray {
        guard let speechTokenizer, let tokenizer else {
            return MLXArray.zeros([1])
        }

        let talkerConfig = config.talkerConfig!
        let inputEmbedding = talker.getInputEmbeddings()
        let codePredictor = talker.codePredictor
        let codeEmbeddings = codePredictor.codecEmbedding
        let codePassCount = talkerConfig.numCodeGroups - 1
        let lastCodeEmbeddingIndex = talkerConfig.numCodeGroups - 2
        let codePredictorPrefillEnabled = Qwen3TTSFullModel.codePredictorPrefillEnabled
        let optimizedCodeSamplerEnabled = Qwen3TTSFullModel.optimizedCodeSamplerEnabled
        let clearCacheEverySteps = Qwen3TTSFullModel.clearCacheEverySteps
        let codeSamplerGreedy = temperature <= 0
        let codeSamplerInvTemperature: Float = codeSamplerGreedy ? 0 : (1 / temperature)
        let genStartTime = CFAbsoluteTimeGetCurrent()
        let emitPerf: (String) -> Void = { msg in
            logger.info("\(msg, privacy: .public)")
            if Qwen3TTSFullModel.stdoutPerfLogsEnabled {
                NSLog("[Qwen3TTS] %@", msg)
            }
        }

        let metalFastSynch = ProcessInfo.processInfo.environment["MLX_METAL_FAST_SYNCH"] ?? "0"
        let settingsMsg = "Runtime knobs: cpPrefill=\(codePredictorPrefillEnabled), codeSampler=\(optimizedCodeSamplerEnabled), clearCacheEvery=\(clearCacheEverySteps), profiling=\(Qwen3TTSFullModel.profilingEnabled), metalFastSynch=\(metalFastSynch)"
        emitPerf(settingsMsg)

        // Prepare inputs (instruct embed split from text part for KV caching)
        let prepared = prepareGenerationInputs(
            text: text, language: language, instruct: instruct
        )
        let trailingTextHidden = prepared.trailingTextHidden
        let ttsPadEmbed = prepared.ttsPadEmbed

        // Cap max tokens based on text length
        let targetTokenCount = tokenizer.encode(text: text).count
        let effectiveMaxTokens = min(maxTokens, max(75, targetTokenCount * 6))

        // Initialize cache and run instruct prefill (cached across calls with same voice)
        let cache = talker.makeCache()

        if let instructEmbed = prepared.instructEmbed {
            if cachedVoiceDescription == instruct, voicePrefixKVState != nil {
                // Cache HIT — restore saved instruct KV state
                restoreVoicePrefixCache(into: cache)
                logger.info("Voice prefix cache HIT (\(instruct?.prefix(40) ?? "", privacy: .public))")
            } else {
                // Cache MISS — run instruct forward pass and save KV state
                let _ = talker(instructEmbed, cache: cache)
                saveVoicePrefixCache(cache)
                cachedVoiceDescription = instruct
                logger.info("Voice prefix cache MISS — computed and saved (\(instruct?.prefix(40) ?? "", privacy: .public))")
            }
        }

        var generatedCodes = [MLXArray]()
        var generatedTokenIds = [Int]()  // Host-side token IDs for repetition penalty (avoids per-step .item() calls)
        generatedCodes.reserveCapacity(effectiveMaxTokens)
        generatedTokenIds.reserveCapacity(effectiveMaxTokens)
        let eosTokenId = talkerConfig.codecEosTokenId

        // Suppress special tokens
        let suppressTokens = (talkerConfig.vocabSize - 1024 ..< talkerConfig.vocabSize)
            .filter { $0 != eosTokenId }

        var trailingIdx = 0
        var inputEmbeds = prepared.textPartEmbed

        let profiling = Qwen3TTSFullModel.profilingEnabled
        var timings = StepTimings()

        // Pre-allocate code predictor KV caches once; reuse via trim() each step.
        // step=16 matches the exact number of positions needed (vs default 256).
        let codePredictorCache = codePredictor.makeCache()
        for c in codePredictorCache {
            c.step = talkerConfig.numCodeGroups
        }

        // Seed random state right before generation loop to ensure deterministic sampling.
        // This is set here (after prefill, before first categorical()) rather than relying
        // on the caller's MLXRandom.seed() surviving across Task/thread boundaries.
        MLXRandom.seed(self.seed)

        for step in 0 ..< effectiveMaxTokens {
            if Task.isCancelled { break }
            let stepStart = CFAbsoluteTimeGetCurrent()

            // Forward pass through talker
            var t0 = CFAbsoluteTimeGetCurrent()
            let (logits, hidden) = talker(inputEmbeds, cache: cache)
            if profiling { eval(logits, hidden) }
            let talkerTime = CFAbsoluteTimeGetCurrent() - t0

            // Sample first codebook token
            t0 = CFAbsoluteTimeGetCurrent()
            let nextToken = sampleToken(
                logits,
                temperature: temperature,
                topP: topP,
                repetitionPenalty: repetitionPenalty,
                generatedTokens: generatedTokenIds,
                suppressTokens: suppressTokens,
                eosTokenId: eosTokenId
            )
            if profiling { eval(nextToken) }
            let samplingTime = CFAbsoluteTimeGetCurrent() - t0

            // Check EOS (this .item() is unavoidable — one per step)
            t0 = CFAbsoluteTimeGetCurrent()
            let tokenId = Int(nextToken[0, 0].item(Int32.self))
            let eosTime = CFAbsoluteTimeGetCurrent() - t0
            if tokenId == eosTokenId {
                if profiling {
                    timings.talkerForward += talkerTime
                    timings.mainSampling += samplingTime
                    timings.eosCheck += eosTime
                    timings.stepCount += 1
                }
                break
            }

            // Track token ID on host side for repetition penalty
            generatedTokenIds.append(tokenId)

            // Generate remaining codebook tokens with code predictor
            // Reset pre-allocated caches to reuse buffers (avoids allocation per step)
            for c in codePredictorCache { c.trim(c.offset) }
            var codeTokens = [nextToken]
            codeTokens.reserveCapacity(talkerConfig.numCodeGroups)
            let codeHidden = hidden.dim(1) == 1 ? hidden : hidden[0..., (-1)..., 0...]
            let code0Embed = inputEmbedding(nextToken)
            var codecEmbed = code0Embed

            let cpStart = CFAbsoluteTimeGetCurrent()
            if codePredictorPrefillEnabled {
                // Prime KV cache with talker hidden state so all sampled passes run at seqLen=1.
                codePredictor.prefill(codeHidden, cache: codePredictorCache)

                for codeIdx in 0 ..< codePassCount {
                    let passStart = CFAbsoluteTimeGetCurrent()

                    let codeInput: MLXArray
                    if codeIdx == 0 {
                        codeInput = code0Embed
                    } else {
                        let nextEmbed = codeEmbeddings[codeIdx - 1](codeTokens.last!)
                        codecEmbed = codecEmbed + nextEmbed
                        codeInput = nextEmbed
                    }

                    let codeLogits = codePredictor.predictStepSingleToken(
                        codeInput, cache: codePredictorCache, generationStep: codeIdx
                    )

                    let nextCode: MLXArray
                    if optimizedCodeSamplerEnabled {
                        nextCode = sampleCodeToken(
                            codeLogits,
                            greedy: codeSamplerGreedy,
                            invTemperature: codeSamplerInvTemperature
                        )
                    } else {
                        nextCode = sampleToken(codeLogits, temperature: temperature, topP: 1.0)
                    }
                    if profiling { eval(nextCode) }
                    codeTokens.append(nextCode)

                    if profiling {
                        timings.codePredictorPasses += CFAbsoluteTimeGetCurrent() - passStart
                        timings.codePredictorPassCount += 1
                    }
                }
            } else {
                for codeIdx in 0 ..< codePassCount {
                    let passStart = CFAbsoluteTimeGetCurrent()

                    let codeInput: MLXArray
                    if codeIdx == 0 {
                        codeInput = concatenated([codeHidden, code0Embed], axis: 1)
                    } else {
                        let nextEmbed = codeEmbeddings[codeIdx - 1](codeTokens.last!)
                        codecEmbed = codecEmbed + nextEmbed
                        codeInput = nextEmbed
                    }

                    let codeLogits = if codeIdx == 0 {
                        codePredictor.predictStep(
                            codeInput, cache: codePredictorCache, generationStep: codeIdx
                        )
                    } else {
                        codePredictor.predictStepSingleToken(
                            codeInput, cache: codePredictorCache, generationStep: codeIdx
                        )
                    }

                    let nextCode: MLXArray
                    if optimizedCodeSamplerEnabled {
                        nextCode = sampleCodeToken(
                            codeLogits,
                            greedy: codeSamplerGreedy,
                            invTemperature: codeSamplerInvTemperature
                        )
                    } else {
                        nextCode = sampleToken(codeLogits, temperature: temperature, topP: 1.0)
                    }
                    if profiling { eval(nextCode) }
                    codeTokens.append(nextCode)

                    if profiling {
                        timings.codePredictorPasses += CFAbsoluteTimeGetCurrent() - passStart
                        timings.codePredictorPassCount += 1
                    }
                }
            }
            let cpTime = CFAbsoluteTimeGetCurrent() - cpStart

            // Final codebook token embedding (index numCodeGroups-2) is not reused as a next-pass input.
            codecEmbed = codecEmbed + codeEmbeddings[lastCodeEmbeddingIndex](codeTokens.last!)

            let allCodes = concatenated(codeTokens, axis: 1)  // [1, num_code_groups]
            generatedCodes.append(allCodes)

            // Prepare next input
            t0 = CFAbsoluteTimeGetCurrent()
            let textEmbed: MLXArray
            if trailingIdx < trailingTextHidden.dim(1) {
                textEmbed = trailingTextHidden[0..., trailingIdx ..< (trailingIdx + 1), 0...]
                trailingIdx += 1
            } else {
                textEmbed = ttsPadEmbed
            }

            // Sync eval the full lazy graph (code predictor 15 passes + embedding prep),
            // then free temporary GPU memory so the next step's talker gets clean GPU state.
            inputEmbeds = textEmbed + codecEmbed
            // Co-evaluate `allCodes` so stored per-step codes are materialized tensors,
            // not references to the full lazy step graph.
            eval(inputEmbeds, allCodes)
            if (step + 1) % clearCacheEverySteps == 0 {
                Memory.clearCache()
            }
            let embedTime = CFAbsoluteTimeGetCurrent() - t0

            if profiling {
                timings.talkerForward += talkerTime
                timings.mainSampling += samplingTime
                timings.eosCheck += eosTime
                timings.codePredictorTotal += cpTime
                timings.embeddingPrep += embedTime
                timings.stepTotal += CFAbsoluteTimeGetCurrent() - stepStart
                timings.stepCount += 1
            }
        }

        let tokenGenTime = CFAbsoluteTimeGetCurrent() - genStartTime
        let tokenCount = generatedCodes.count
        let tokensPerSec = tokenCount > 0 ? Double(tokenCount) / tokenGenTime : 0
        let audioSeconds = Double(tokenCount) / 12.0  // 12Hz token rate
        let rtf = tokenGenTime / max(audioSeconds, 0.001)
        let tokenMsg = "Token generation: \(tokenCount) tokens in \(String(format: "%.2f", tokenGenTime))s (\(String(format: "%.1f", tokensPerSec)) tok/s, RTF=\(String(format: "%.2f", rtf))x)"
        emitPerf(tokenMsg)

        // Log per-component profiling summary
        if profiling && timings.stepCount > 0 {
            let n = Double(timings.stepCount)
            let total = timings.stepTotal
            let pct = { (v: Double) -> String in
                String(format: "%5.1f%%", total > 0 ? v / total * 100 : 0)
            }
            let avg = { (v: Double) -> String in
                String(format: "%7.2fms", v / n * 1000)
            }
            let passAvg = timings.codePredictorPassCount > 0
                ? String(format: "%.2fms", timings.codePredictorPasses / Double(timings.codePredictorPassCount) * 1000)
                : "N/A"

            let summary = """
                Profile (\(timings.stepCount) steps): \
                talker=\(avg(timings.talkerForward)) (\(pct(timings.talkerForward))), \
                sampling=\(avg(timings.mainSampling)) (\(pct(timings.mainSampling))), \
                eos=\(avg(timings.eosCheck)) (\(pct(timings.eosCheck))), \
                codePredictor=\(avg(timings.codePredictorTotal)) (\(pct(timings.codePredictorTotal))), \
                perPass=\(passAvg), \
                embedPrep=\(avg(timings.embeddingPrep)) (\(pct(timings.embeddingPrep))), \
                stepAvg=\(avg(timings.stepTotal))
                """
            emitPerf(summary)
        }

        // Save generated codes for voice anchor building
        self.lastGeneratedCodes = generatedCodes.isEmpty ? nil : generatedCodes

        guard !generatedCodes.isEmpty else {
            Memory.clearCache()
            return MLXArray.zeros([1])
        }

        // Stack and decode
        let decodeStartTime = CFAbsoluteTimeGetCurrent()
        let codes = stacked(generatedCodes, axis: 1)  // [1, seq_len, num_code_groups]

        let decodeMode = Qwen3TTSFullModel.finalDecodeMode
        var audio: MLXArray
        let validLen: Int
        switch decodeMode {
        case "stream":
            // Streaming decode for lower peak memory.
            var audioChunks = [MLXArray]()
            for chunk in speechTokenizer.streamingDecode(codes, chunkTokens: 100) {
                audioChunks.append(chunk)
            }
            audio = concatenated(audioChunks, axis: -1)[0]  // Remove batch dim
            validLen = Int((codes[0..., 0..., 0] .> 0).sum().item(Int32.self)) * speechTokenizer.decodeUpsampleRate
        case "single":
            // One decoder call on the full code sequence (fastest if memory allows).
            let wav = speechTokenizer.decodeSingleChunk(codes, leftContextTokens: 0)
            audio = wav[0]
            validLen = Int((codes[0..., 0..., 0] .> 0).sum().item(Int32.self)) * speechTokenizer.decodeUpsampleRate
        default:
            // Lower-overhead path with larger internal chunks.
            let (wav, audioLengths) = speechTokenizer.decode(codes)
            audio = wav[0]
            validLen = Int(audioLengths[0].item(Int32.self))
        }

        // Trim to valid length.
        if validLen > 0 && validLen < audio.dim(0) {
            audio = audio[..<validLen]
        }

        eval(audio)
        Memory.clearCache()

        let decodeTime = CFAbsoluteTimeGetCurrent() - decodeStartTime
        let totalTime = CFAbsoluteTimeGetCurrent() - genStartTime
        let decodeMsg = "Decode: \(String(format: "%.2f", decodeTime))s | Total: \(String(format: "%.2f", totalTime))s for \(String(format: "%.1f", audioSeconds))s audio (overall RTF=\(String(format: "%.2f", totalTime / max(audioSeconds, 0.001)))x)"
        emitPerf(decodeMsg)

        return audio
    }

    // MARK: - Streaming VoiceDesign generation

    /// Generates speech tokens and progressively decodes/yields audio chunks.
    /// Each chunk is a `[Float]` array of audio samples at `sampleRate` Hz.
    ///
    /// Uses a two-phase strategy for low first-chunk latency:
    /// - Phase 1: emit after `firstChunkEmitEvery` tokens with a shorter decode window
    /// - Phase 2: switch to `emitEvery` tokens with full `decodeWindow`
    ///
    /// Adjacent chunks are Hann-crossfaded over `blendSamples` to eliminate boundary artifacts.
    func generateStreamingVoiceDesign(
        text: String,
        instruct: String?,
        language: String,
        temperature: Float,
        topP: Float,
        repetitionPenalty: Float,
        maxTokens: Int,
        firstChunkEmitEvery: Int = 5,
        firstChunkDecodeWindow: Int = 48,
        firstChunkFrames: Int = 48,
        emitEvery: Int = 8,
        decodeWindow: Int = 80,
        blendSamples: Int = 512,
        useVoiceAnchor: Bool = false,
        onAudioChunk: ([Float]) -> Void
    ) {
        guard let speechTokenizer, let tokenizer else { return }

        let talkerConfig = config.talkerConfig!
        let inputEmbedding = talker.getInputEmbeddings()
        let codePredictor = talker.codePredictor
        let codeEmbeddings = codePredictor.codecEmbedding
        let codePassCount = talkerConfig.numCodeGroups - 1
        let lastCodeEmbeddingIndex = talkerConfig.numCodeGroups - 2
        let codePredictorPrefillEnabled = Qwen3TTSFullModel.codePredictorPrefillEnabled
        let optimizedCodeSamplerEnabled = Qwen3TTSFullModel.optimizedCodeSamplerEnabled
        let clearCacheEverySteps = Qwen3TTSFullModel.clearCacheEverySteps
        let codeSamplerGreedy = temperature <= 0
        let codeSamplerInvTemperature: Float = codeSamplerGreedy ? 0 : (1 / temperature)
        let genStartTime = CFAbsoluteTimeGetCurrent()
        let emitPerf: (String) -> Void = { msg in
            logger.info("\(msg, privacy: .public)")
            if Qwen3TTSFullModel.stdoutPerfLogsEnabled {
                NSLog("[Qwen3TTS] %@", msg)
            }
        }

        let prepared = prepareGenerationInputs(
            text: text, language: language, instruct: instruct
        )
        let trailingTextHidden = prepared.trailingTextHidden
        let ttsPadEmbed = prepared.ttsPadEmbed

        let targetTokenCount = tokenizer.encode(text: text).count
        let effectiveMaxTokens = min(maxTokens, max(75, targetTokenCount * 6))

        // Initialize cache and restore voice state
        let cache = talker.makeCache()

        if useVoiceAnchor, voiceAnchorKVState != nil {
            // Restore voice anchor: instruct + codec voice examples
            restoreVoiceAnchorCache(into: cache)
            let anchorCount = self.voiceAnchorCodecCount
            logger.info("Voice anchor cache restored (\(anchorCount, privacy: .public) codec steps)")
        } else if let instructEmbed = prepared.instructEmbed {
            if cachedVoiceDescription == instruct, voicePrefixKVState != nil {
                restoreVoicePrefixCache(into: cache)
                logger.info("Voice prefix cache HIT (\(instruct?.prefix(40) ?? "", privacy: .public))")
            } else {
                let _ = talker(instructEmbed, cache: cache)
                saveVoicePrefixCache(cache)
                cachedVoiceDescription = instruct
                logger.info("Voice prefix cache MISS — computed and saved (\(instruct?.prefix(40) ?? "", privacy: .public))")
            }
        }

        var generatedCodes = [MLXArray]()
        var generatedTokenIds = [Int]()
        generatedCodes.reserveCapacity(effectiveMaxTokens)
        generatedTokenIds.reserveCapacity(effectiveMaxTokens)
        let eosTokenId = talkerConfig.codecEosTokenId

        let suppressTokens = (talkerConfig.vocabSize - 1024 ..< talkerConfig.vocabSize)
            .filter { $0 != eosTokenId }

        var trailingIdx = 0
        var inputEmbeds = prepared.textPartEmbed
        var totalEmitted = 0  // Number of tokens whose audio has been emitted
        var prevChunkTail: [Float]? = nil  // Last `blendSamples` from previous chunk for crossfade
        var isFirstChunk = true
        var phase = 1  // 1 = aggressive first chunk, 2 = steady state
        let sampleRate = config.sampleRate

        // Pre-compute Hann window tables
        let hannFadeIn = (0..<blendSamples).map { i -> Float in
            0.5 * (1 - cos(Float.pi * Float(i) / Float(blendSamples)))
        }
        let hannFadeOut = (0..<blendSamples).map { i -> Float in
            0.5 * (1 + cos(Float.pi * Float(i) / Float(blendSamples)))
        }

        // Pre-allocate code predictor KV caches once; reuse via trim() each step.
        let codePredictorCache = codePredictor.makeCache()
        for c in codePredictorCache {
            c.step = talkerConfig.numCodeGroups
        }

        // Seed random state right before generation loop (see generateVoiceDesign for rationale)
        MLXRandom.seed(self.seed)

        for step in 0 ..< effectiveMaxTokens {
            if Task.isCancelled { break }

            let (logits, hidden) = talker(inputEmbeds, cache: cache)

            let nextToken = sampleToken(
                logits,
                temperature: temperature,
                topP: topP,
                repetitionPenalty: repetitionPenalty,
                generatedTokens: generatedTokenIds,
                suppressTokens: suppressTokens,
                eosTokenId: eosTokenId
            )

            let tokenId = Int(nextToken[0, 0].item(Int32.self))
            if tokenId == eosTokenId { break }

            generatedTokenIds.append(tokenId)

            for c in codePredictorCache { c.trim(c.offset) }
            var codeTokens = [nextToken]
            codeTokens.reserveCapacity(talkerConfig.numCodeGroups)
            let codeHidden = hidden.dim(1) == 1 ? hidden : hidden[0..., (-1)..., 0...]
            let code0Embed = inputEmbedding(nextToken)
            var codecEmbed = code0Embed

            if codePredictorPrefillEnabled {
                codePredictor.prefill(codeHidden, cache: codePredictorCache)

                for codeIdx in 0 ..< codePassCount {
                    let codeInput: MLXArray
                    if codeIdx == 0 {
                        codeInput = code0Embed
                    } else {
                        let nextEmbed = codeEmbeddings[codeIdx - 1](codeTokens.last!)
                        codecEmbed = codecEmbed + nextEmbed
                        codeInput = nextEmbed
                    }

                    let codeLogits = codePredictor.predictStepSingleToken(
                        codeInput, cache: codePredictorCache, generationStep: codeIdx
                    )

                    let nextCode: MLXArray
                    if optimizedCodeSamplerEnabled {
                        nextCode = sampleCodeToken(
                            codeLogits,
                            greedy: codeSamplerGreedy,
                            invTemperature: codeSamplerInvTemperature
                        )
                    } else {
                        nextCode = sampleToken(codeLogits, temperature: temperature, topP: 1.0)
                    }
                    codeTokens.append(nextCode)
                }
            } else {
                for codeIdx in 0 ..< codePassCount {
                    let codeInput: MLXArray
                    if codeIdx == 0 {
                        codeInput = concatenated([codeHidden, code0Embed], axis: 1)
                    } else {
                        let nextEmbed = codeEmbeddings[codeIdx - 1](codeTokens.last!)
                        codecEmbed = codecEmbed + nextEmbed
                        codeInput = nextEmbed
                    }

                    let codeLogits = if codeIdx == 0 {
                        codePredictor.predictStep(
                            codeInput, cache: codePredictorCache, generationStep: codeIdx
                        )
                    } else {
                        codePredictor.predictStepSingleToken(
                            codeInput, cache: codePredictorCache, generationStep: codeIdx
                        )
                    }

                    let nextCode: MLXArray
                    if optimizedCodeSamplerEnabled {
                        nextCode = sampleCodeToken(
                            codeLogits,
                            greedy: codeSamplerGreedy,
                            invTemperature: codeSamplerInvTemperature
                        )
                    } else {
                        nextCode = sampleToken(codeLogits, temperature: temperature, topP: 1.0)
                    }
                    codeTokens.append(nextCode)
                }
            }

            codecEmbed = codecEmbed + codeEmbeddings[lastCodeEmbeddingIndex](codeTokens.last!)

            let allCodes = concatenated(codeTokens, axis: 1)
            generatedCodes.append(allCodes)

            // Prepare next input
            let textEmbed: MLXArray
            if trailingIdx < trailingTextHidden.dim(1) {
                textEmbed = trailingTextHidden[0..., trailingIdx ..< (trailingIdx + 1), 0...]
                trailingIdx += 1
            } else {
                textEmbed = ttsPadEmbed
            }

            inputEmbeds = textEmbed + codecEmbed
            eval(inputEmbeds, allCodes)
            if (step + 1) % clearCacheEverySteps == 0 {
                Memory.clearCache()
            }

            // Phase transition
            if phase == 1 && generatedCodes.count >= firstChunkFrames {
                phase = 2
            }

            // Sliding window decode: emit audio using phase-appropriate intervals
            let currentEmitEvery = phase == 1 ? firstChunkEmitEvery : emitEvery
            let currentDecodeWindow = phase == 1 ? firstChunkDecodeWindow : decodeWindow
            let pendingCount = generatedCodes.count - totalEmitted
            if pendingCount >= currentEmitEvery {
                let rawSamples = decodeWindowTail(
                    generatedCodes: generatedCodes,
                    speechTokenizer: speechTokenizer,
                    emitTokens: currentEmitEvery,
                    decodeWindow: currentDecodeWindow,
                    totalEmitted: totalEmitted
                )

                let emitted = crossfadeAndEmit(
                    rawSamples: rawSamples,
                    prevChunkTail: &prevChunkTail,
                    isFirstChunk: isFirstChunk,
                    blendSamples: blendSamples,
                    hannFadeIn: hannFadeIn,
                    hannFadeOut: hannFadeOut,
                    sampleRate: sampleRate
                )
                if !emitted.isEmpty {
                    onAudioChunk(emitted)
                    isFirstChunk = false
                }
                totalEmitted += currentEmitEvery
            }
        }

        // Decode and yield any remaining tokens
        let remaining = generatedCodes.count - totalEmitted
        if remaining > 0 {
            let samplesPerToken = speechTokenizer.decodeUpsampleRate
            let windowSize = min(decodeWindow, generatedCodes.count)
            let windowStart = generatedCodes.count - windowSize
            let windowCodes = stacked(Array(generatedCodes[windowStart...]), axis: 1)

            let wav = speechTokenizer.decodeSingleChunk(windowCodes, leftContextTokens: 0)
            let allSamples = wav[0]

            let remainingSamples = remaining * samplesPerToken
            let startSample = max(0, allSamples.dim(0) - remainingSamples)
            var tailSlice = allSamples[startSample...]

            // Trim padding tokens
            let newTokenCodes = stacked(Array(generatedCodes[totalEmitted...]), axis: 1)
            let validTokens = Int((newTokenCodes[0..., 0..., 0] .> 0).sum().item(Int32.self))
            let validSamples = validTokens * samplesPerToken
            if validSamples > 0 && validSamples < tailSlice.dim(0) {
                tailSlice = tailSlice[..<validSamples]
            }

            let rawSamples = tailSlice.asArray(Float.self)
            if !rawSamples.isEmpty {
                let emitted = crossfadeAndEmit(
                    rawSamples: rawSamples,
                    prevChunkTail: &prevChunkTail,
                    isFirstChunk: isFirstChunk,
                    blendSamples: blendSamples,
                    hannFadeIn: hannFadeIn,
                    hannFadeOut: hannFadeOut,
                    sampleRate: sampleRate
                )
                if !emitted.isEmpty {
                    onAudioChunk(emitted)
                    isFirstChunk = false
                }
            }
        }

        // Flush held-back overlap tail with fade-out
        if var tail = prevChunkTail, !tail.isEmpty {
            let fadeLen = min(Int(Float(sampleRate) * 0.005), tail.count)
            for i in 0..<fadeLen {
                tail[tail.count - fadeLen + i] *= Float(fadeLen - 1 - i) / Float(fadeLen)
            }
            onAudioChunk(tail)
        }

        let tokenGenTime = CFAbsoluteTimeGetCurrent() - genStartTime
        let tokenCount = generatedCodes.count
        let tokensPerSec = tokenCount > 0 ? Double(tokenCount) / tokenGenTime : 0
        let audioSeconds = Double(tokenCount) / 12.0
        let rtf = tokenGenTime / max(audioSeconds, 0.001)
        emitPerf("Streaming generation: \(tokenCount) tokens in \(String(format: "%.2f", tokenGenTime))s (\(String(format: "%.1f", tokensPerSec)) tok/s, RTF=\(String(format: "%.2f", rtf))x)")

        // Save generated codes for voice anchor building
        self.lastGeneratedCodes = generatedCodes.isEmpty ? nil : generatedCodes

        Memory.clearCache()
    }

    // MARK: - Streaming helpers

    /// Decode a window of tokens and extract the tail samples for the newly emitted tokens.
    private func decodeWindowTail(
        generatedCodes: [MLXArray],
        speechTokenizer: Qwen3TTSSpeechTokenizer,
        emitTokens: Int,
        decodeWindow: Int,
        totalEmitted: Int
    ) -> [Float] {
        let samplesPerToken = speechTokenizer.decodeUpsampleRate
        let stepSamples = emitTokens * samplesPerToken

        let windowSize = min(decodeWindow, generatedCodes.count)
        let windowStart = generatedCodes.count - windowSize
        let windowCodes = stacked(Array(generatedCodes[windowStart..<generatedCodes.count]), axis: 1)

        let wav = speechTokenizer.decodeSingleChunk(windowCodes, leftContextTokens: 0)
        let allSamples = wav[0]

        let startSample = max(0, allSamples.dim(0) - stepSamples)
        let chunk = allSamples[startSample...]
        return chunk.asArray(Float.self)
    }

    /// Apply Hann crossfade between previous chunk's tail and new chunk's head, emit the result.
    /// Holds back the last `blendSamples` of each chunk for the next crossfade.
    private func crossfadeAndEmit(
        rawSamples: [Float],
        prevChunkTail: inout [Float]?,
        isFirstChunk: Bool,
        blendSamples: Int,
        hannFadeIn: [Float],
        hannFadeOut: [Float],
        sampleRate: Int
    ) -> [Float] {
        guard !rawSamples.isEmpty else { return [] }

        var output = [Float]()
        output.reserveCapacity(rawSamples.count)

        if let prevTail = prevChunkTail, !prevTail.isEmpty {
            // Crossfade: blend prevTail with first blendSamples of rawSamples
            let overlapLen = min(blendSamples, prevTail.count, rawSamples.count)
            for i in 0..<overlapLen {
                output.append(prevTail[i] * hannFadeOut[i] + rawSamples[i] * hannFadeIn[i])
            }
            // Append remainder after overlap
            if rawSamples.count > overlapLen {
                output.append(contentsOf: rawSamples[overlapLen...])
            }
        } else {
            // First chunk — apply 5ms linear fade-in
            output.append(contentsOf: rawSamples)
            if isFirstChunk {
                let fadeLen = min(Int(Float(sampleRate) * 0.005), output.count)
                for i in 0..<fadeLen {
                    output[i] *= Float(i) / Float(fadeLen)
                }
            }
        }

        // Hold back last blendSamples for next crossfade
        if output.count > blendSamples {
            let holdStart = output.count - blendSamples
            prevChunkTail = Array(output[holdStart...])
            output.removeLast(blendSamples)
        } else {
            // Chunk too small to split — hold entire thing
            prevChunkTail = output
            output = []
        }

        return output
    }

    // MARK: - Prepare generation inputs

    func prepareGenerationInputs(
        text: String,
        language: String,
        instruct: String?
    ) -> (instructEmbed: MLXArray?, textPartEmbed: MLXArray, trailingTextHidden: MLXArray, ttsPadEmbed: MLXArray) {
        guard let tokenizer, let talkerConfig = config.talkerConfig else {
            fatalError("Tokenizer/config not loaded")
        }

        // Tokenize text with ChatML template
        let chatText = "<|im_start|>assistant\n\(text)<|im_end|>\n<|im_start|>assistant\n"
        let inputIds = MLXArray(tokenizer.encode(text: chatText).map { Int32($0) }).reshaped(1, -1)

        // Get text embeddings
        let textEmbed = talker.textProjection(talker.getTextEmbeddings()(inputIds))

        // TTS special tokens
        let ttsTokens = MLXArray(
            [Int32(config.ttsBosTokenId), Int32(config.ttsEosTokenId), Int32(config.ttsPadTokenId)]
        ).reshaped(1, 3)
        let ttsEmbeds = talker.textProjection(talker.getTextEmbeddings()(ttsTokens))
        let ttsBosEmbed = ttsEmbeds[0..., 0 ..< 1, 0...]
        let ttsEosEmbed = ttsEmbeds[0..., 1 ..< 2, 0...]
        let ttsPadEmbed = ttsEmbeds[0..., 2 ..< 3, 0...]

        // Language ID
        var languageId: Int? = nil
        if language.lowercased() != "auto", let langMap = talkerConfig.codecLanguageId {
            languageId = langMap[language.lowercased()]
        }

        // Build codec prefix
        var codecPrefill: [Int32]
        if let langId = languageId {
            codecPrefill = [
                Int32(talkerConfig.codecThinkId),
                Int32(talkerConfig.codecThinkBosId),
                Int32(langId),
                Int32(talkerConfig.codecThinkEosId),
            ]
        } else {
            codecPrefill = [
                Int32(talkerConfig.codecNothinkId),
                Int32(talkerConfig.codecThinkBosId),
                Int32(talkerConfig.codecThinkEosId),
            ]
        }

        var codecEmbed = talker.getInputEmbeddings()(MLXArray(codecPrefill).reshaped(1, -1))
        let codecEmbedSuffix = talker.getInputEmbeddings()(
            MLXArray([Int32(talkerConfig.codecPadId), Int32(talkerConfig.codecBosId)]).reshaped(1, 2)
        )
        codecEmbed = concatenated([codecEmbed, codecEmbedSuffix], axis: 1)

        // Instruct embedding
        var instructEmbed: MLXArray? = nil
        if let instruct, !instruct.isEmpty {
            let instructText = "<|im_start|>user\n\(instruct)<|im_end|>\n"
            let instructIds = MLXArray(tokenizer.encode(text: instructText).map { Int32($0) }).reshaped(1, -1)
            instructEmbed = talker.textProjection(talker.getTextEmbeddings()(instructIds))
        }

        // Role embedding (first 3 tokens: <|im_start|>assistant\n)
        let roleEmbed = textEmbed[0..., ..<3, 0...]

        // Build pad/bos prefix
        let padCount = codecEmbed.dim(1) - 2
        let padEmbeds = broadcast(ttsPadEmbed, to: [1, padCount, ttsPadEmbed.dim(-1)])
        var combinedEmbed = concatenated([padEmbeds, ttsBosEmbed], axis: 1)
        combinedEmbed = combinedEmbed + codecEmbed[0..., ..<(-1), 0...]

        // Build text-part embedding (everything after instruct)
        let firstTextEmbed = textEmbed[0..., 3 ..< 4, 0...] + codecEmbed[0..., (-1)..., 0...]
        let textPartEmbed = concatenated([roleEmbed, combinedEmbed, firstTextEmbed], axis: 1)

        // Trailing text (tokens 4 to -5, plus EOS)
        let trailingTextHidden = concatenated(
            [textEmbed[0..., 4 ..< (textEmbed.dim(1) - 5), 0...], ttsEosEmbed],
            axis: 1
        )

        return (instructEmbed, textPartEmbed, trailingTextHidden, ttsPadEmbed)
    }

    // MARK: - Token sampling

    func sampleToken(
        _ logits: MLXArray,
        temperature: Float = 0.9,
        topP: Float = 1.0,
        repetitionPenalty: Float = 1.0,
        generatedTokens: [Int]? = nil,
        suppressTokens: [Int]? = nil,
        eosTokenId: Int? = nil
    ) -> MLXArray {
        var logitsSlice = logits[0..., (-1)..., 0...].squeezed(axis: 1)  // [batch, vocab_size]

        // Suppress tokens by setting to -inf
        if let suppress = suppressTokens, !suppress.isEmpty {
            let suppressArr = MLXArray(suppress.map { Int32($0) }).reshaped(1, -1)
            let negInf = MLXArray.full([1, suppress.count], values: MLXArray(-Float.infinity), dtype: logitsSlice.dtype)
            logitsSlice = putAlong(logitsSlice, suppressArr, values: negInf, axis: -1)
        }

        // Repetition penalty
        if let tokens = generatedTokens, !tokens.isEmpty, repetitionPenalty != 1.0 {
            let unique = Array(Set(tokens)).filter { $0 < logitsSlice.dim(-1) }
            if !unique.isEmpty {
                let tokenIds = MLXArray(unique.map { Int32($0) }).reshaped(1, -1)
                let selected = takeAlong(logitsSlice, tokenIds, axis: -1)
                let penalized = which(
                    selected .< 0,
                    selected * repetitionPenalty,
                    selected / repetitionPenalty
                )
                logitsSlice = putAlong(logitsSlice, tokenIds, values: penalized, axis: -1)
            }
        }

        // Greedy if temperature 0
        if temperature <= 0 {
            return argMax(logitsSlice, axis: -1, keepDims: true)
        }

        // Apply top-p sampling
        if topP > 0 && topP < 1.0 {
            let scaledLogits = logitsSlice / temperature
            let probs = softmax(scaledLogits, axis: -1)
            let sortedIndices = argSort(probs, axis: -1)
            // argSort returns ascending order, reverse for descending
            let descIndices = sortedIndices[0..., .stride(by: -1)]
            let descProbs = takeAlong(probs, descIndices, axis: -1)
            let cumProbs = cumsum(descProbs, axis: -1)
            let mask = cumProbs .> topP
            let filteredProbs = which(mask, MLXArray(Float(0)), descProbs)

            // Sample from filtered distribution
            let token = categorical(log(filteredProbs + 1e-10))
            return takeAlong(descIndices, token.reshaped(1, 1), axis: -1)
        }

        // Simple temperature sampling
        let token = categorical(logitsSlice / temperature)
        return token.reshaped(1, 1)
    }

    /// Hot path for code predictor sampling: equivalent to sampleToken(topP=1, no penalties/suppression).
    @inline(__always)
    func sampleCodeToken(_ logits: MLXArray, greedy: Bool, invTemperature: Float) -> MLXArray {
        let logitsSlice = logits[0..., (-1)..., 0...].squeezed(axis: 1)
        if greedy {
            return argMax(logitsSlice, axis: -1, keepDims: true)
        }
        return categorical(logitsSlice * invTemperature).reshaped(1, 1)
    }

    // MARK: - Token alignment

    public func tokenizeForAlignment(text: String) -> [Int] {
        guard let tokenizer else { return [] }
        let tokens = tokenizer.encode(text: text)
        guard !tokens.isEmpty else { return [] }

        // Decode increasing prefixes to find the character boundary of each token
        var offsets: [Int] = [0]
        for i in 1..<tokens.count {
            let prefix = tokenizer.decode(tokens: Array(tokens[0..<i]))
            offsets.append(min(prefix.count, text.count))
        }
        return offsets
    }

    // MARK: - fromPretrained

    public static func fromPretrained(_ modelRepo: String) async throws -> Qwen3TTSFullModel {
        let repoID = Repo.ID(rawValue: modelRepo)!
        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID, requiredExtension: "safetensors"
        )

        // Load main config
        let configData = try Data(contentsOf: modelDir.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(Qwen3TTSModelConfig.self, from: configData)

        let model = Qwen3TTSFullModel(config: config)

        // Load talker weights
        var allWeights = [String: MLXArray]()
        let fm = FileManager.default
        let modelFiles = try fm.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
        for file in modelFiles where file.pathExtension == "safetensors" {
            let weights = try MLX.loadArrays(url: file)
            allWeights.merge(weights) { _, new in new }
        }

        // Sanitize and load talker weights
        let talkerWeights = Qwen3TTSTalkerForConditionalGeneration.sanitize(weights: allWeights)
        let talkerPairs = talkerWeights.map { ($0.key, $0.value) }
        try model.talker.update(parameters: ModuleParameters.unflattened(talkerPairs), verify: .noUnusedKeys)
        eval(model.talker.parameters())

        // Generate tokenizer.json if missing (Qwen3-TTS ships without it)
        let tokenizerJsonPath = modelDir.appendingPathComponent("tokenizer.json")
        if !fm.fileExists(atPath: tokenizerJsonPath.path) {
            let vocabPath = modelDir.appendingPathComponent("vocab.json")
            let mergesPath = modelDir.appendingPathComponent("merges.txt")
            let hasVocab = fm.fileExists(atPath: vocabPath.path)
            let hasMerges = fm.fileExists(atPath: mergesPath.path)
            if hasVocab && hasMerges {
                do {
                    try generateTokenizerJson(
                        vocabPath: vocabPath,
                        mergesPath: mergesPath,
                        tokenizerConfigPath: modelDir.appendingPathComponent("tokenizer_config.json"),
                        outputPath: tokenizerJsonPath
                    )
                    print("Generated tokenizer.json from vocab.json + merges.txt")
                } catch {
                    print("Warning: Failed to generate tokenizer.json: \(error)")
                }
            } else {
                print("Warning: Cannot generate tokenizer.json — vocab.json: \(hasVocab), merges.txt: \(hasMerges)")
            }
        }

        // Load tokenizer
        do {
            model.tokenizer = try await AutoTokenizer.from(modelFolder: modelDir)
        } catch {
            print("Warning: Could not load tokenizer: \(error)")
        }

        // Load speech tokenizer — check that it's a directory, not a stale file
        let speechTokenizerPath = modelDir.appendingPathComponent("speech_tokenizer")
        var isDir: ObjCBool = false
        if fm.fileExists(atPath: speechTokenizerPath.path, isDirectory: &isDir), isDir.boolValue {
            try loadSpeechTokenizer(model: model, path: speechTokenizerPath)
        } else if fm.fileExists(atPath: speechTokenizerPath.path) {
            // speech_tokenizer exists but is not a directory — stale cache
            // Remove the entire model cache so it re-downloads cleanly next time
            print("speech_tokenizer is not a directory (stale cache), clearing model cache...")
            try? fm.removeItem(at: modelDir)
            throw AudioGenerationError.modelNotInitialized(
                "Model cache was corrupted (speech_tokenizer). It has been cleared. Please try loading again."
            )
        } else {
            print("Warning: speech_tokenizer directory not found, speech decoding unavailable")
        }

        print("Loaded Qwen3-TTS model (\(config.ttsModelType))")
        return model
    }

    private static func loadSpeechTokenizer(model: Qwen3TTSFullModel, path: URL) throws {
        // Load config — fall back to defaults if config.json is missing
        let tokenizerConfig: Qwen3TTSTokenizerConfig
        let configPath = path.appendingPathComponent("config.json")
        if let configData = try? Data(contentsOf: configPath) {
            tokenizerConfig = try JSONDecoder().decode(Qwen3TTSTokenizerConfig.self, from: configData)
        } else {
            print("Warning: speech_tokenizer/config.json not found, using defaults")
            let defaultJson = "{}".data(using: .utf8)!
            tokenizerConfig = try JSONDecoder().decode(Qwen3TTSTokenizerConfig.self, from: defaultJson)
        }

        let speechTokenizer = Qwen3TTSSpeechTokenizer(config: tokenizerConfig)

        // Load weights
        var tokenizerWeights = [String: MLXArray]()
        let files = try FileManager.default.contentsOfDirectory(at: path, includingPropertiesForKeys: nil)
        for file in files where file.pathExtension == "safetensors" {
            let weights = try MLX.loadArrays(url: file)
            tokenizerWeights.merge(weights) { _, new in new }
        }

        if !tokenizerWeights.isEmpty {
            let sanitized = Qwen3TTSSpeechTokenizer.sanitize(weights: tokenizerWeights)
            let pairs = sanitized.map { ($0.key, $0.value) }
            try speechTokenizer.update(parameters: ModuleParameters.unflattened(pairs), verify: .noUnusedKeys)
            eval(speechTokenizer.parameters())
        }

        model.speechTokenizer = speechTokenizer
        print("Loaded speech tokenizer decoder")
    }

    // MARK: - Generate tokenizer.json from vocab.json + merges.txt

    /// Qwen3-TTS repos ship with a slow tokenizer (vocab.json + merges.txt) but
    /// swift-transformers requires tokenizer.json (fast tokenizer format). This
    /// generates the fast tokenizer JSON from the available files.
    private static func generateTokenizerJson(
        vocabPath: URL,
        mergesPath: URL,
        tokenizerConfigPath: URL,
        outputPath: URL
    ) throws {
        // Read vocab
        let vocabData = try Data(contentsOf: vocabPath)
        let vocabDict = try JSONSerialization.jsonObject(with: vocabData) as? [String: Int] ?? [:]

        // Read merges (skip header line "#version: ...")
        let mergesText = try String(contentsOf: mergesPath, encoding: .utf8)
        let mergeLines = mergesText.components(separatedBy: .newlines)
            .filter { !$0.isEmpty && !$0.hasPrefix("#") }

        // Read added_tokens from tokenizer_config.json
        var addedTokens = [[String: Any]]()
        if let configData = try? Data(contentsOf: tokenizerConfigPath),
           let configDict = try? JSONSerialization.jsonObject(with: configData) as? [String: Any],
           let addedTokensDecoder = configDict["added_tokens_decoder"] as? [String: [String: Any]] {
            for (idStr, tokenInfo) in addedTokensDecoder {
                guard let tokenId = Int(idStr),
                      let content = tokenInfo["content"] as? String else { continue }
                var entry: [String: Any] = [
                    "id": tokenId,
                    "content": content,
                    "single_word": tokenInfo["single_word"] as? Bool ?? false,
                    "lstrip": tokenInfo["lstrip"] as? Bool ?? false,
                    "rstrip": tokenInfo["rstrip"] as? Bool ?? false,
                    "normalized": tokenInfo["normalized"] as? Bool ?? false,
                    "special": tokenInfo["special"] as? Bool ?? true
                ]
                _ = entry  // suppress unused warning
                addedTokens.append(entry)
            }
            addedTokens.sort { ($0["id"] as? Int ?? 0) < ($1["id"] as? Int ?? 0) }
        }

        // Build tokenizer.json
        // Qwen2 uses ByteLevel BPE with a GPT-2-style regex pre-tokenizer
        let tokenizerJson: [String: Any] = [
            "version": "1.0",
            "truncation": NSNull(),
            "padding": NSNull(),
            "added_tokens": addedTokens,
            "normalizer": NSNull(),
            "pre_tokenizer": [
                "type": "Sequence",
                "pretokenizers": [
                    [
                        "type": "Split",
                        "pattern": [
                            "Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                        ],
                        "behavior": "Isolated",
                        "invert": false
                    ] as [String: Any],
                    [
                        "type": "ByteLevel",
                        "add_prefix_space": false,
                        "trim_offsets": true,
                        "use_regex": false
                    ] as [String: Any]
                ] as [[String: Any]]
            ] as [String: Any],
            "post_processor": NSNull(),
            "decoder": [
                "type": "ByteLevel",
                "add_prefix_space": true,
                "trim_offsets": true,
                "use_regex": true
            ] as [String: Any],
            "model": [
                "type": "BPE",
                "dropout": NSNull(),
                "unk_token": NSNull(),
                "continuing_subword_prefix": "",
                "end_of_word_suffix": "",
                "fuse_unk": false,
                "byte_fallback": false,
                "ignore_merges": false,
                "vocab": vocabDict,
                "merges": mergeLines
            ] as [String: Any]
        ]

        let jsonData = try JSONSerialization.data(withJSONObject: tokenizerJson, options: [.sortedKeys])
        try jsonData.write(to: outputPath)
    }
}
