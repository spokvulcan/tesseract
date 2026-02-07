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

private let logger = Logger(subsystem: "com.tesseract.app", category: "speech")

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

    /// Enable per-component profiling. Set env QWEN3TTS_PROFILE=1 to enable.
    /// Profiling forces sync eval() at every sub-step, preventing lazy graph fusion
    /// and adding ~18 GPU sync points per step (vs 2 normally). Expect ~2-3x slowdown.
    static let profilingEnabled: Bool = {
        ProcessInfo.processInfo.environment["QWEN3TTS_PROFILE"] == "1"
    }()

    public var sampleRate: Int { config.sampleRate }

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
        let (stream, continuation) = AsyncThrowingStream<AudioGeneration, Error>.makeStream()
        Task { @Sendable [weak self] in
            guard let self else { return }
            do {
                guard self.speechTokenizer != nil else {
                    throw AudioGenerationError.modelNotInitialized("Speech tokenizer not loaded")
                }
                guard self.tokenizer != nil else {
                    throw AudioGenerationError.modelNotInitialized("Text tokenizer not loaded")
                }

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
                    onAudioChunk: { chunk in
                        continuation.yield(.audioChunk(chunk))
                    }
                )
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        return stream
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
        let genStartTime = CFAbsoluteTimeGetCurrent()

        // Prepare inputs
        let (inputEmbedsInit, trailingTextHidden, ttsPadEmbed) = prepareGenerationInputs(
            text: text, language: language, instruct: instruct
        )

        // Cap max tokens based on text length
        let targetTokenCount = tokenizer.encode(text: text).count
        let effectiveMaxTokens = min(maxTokens, max(75, targetTokenCount * 6))

        // Initialize cache
        let cache = talker.makeCache()
        var generatedCodes = [MLXArray]()
        var generatedTokenIds = [Int]()  // Host-side token IDs for repetition penalty (avoids per-step .item() calls)
        let eosTokenId = talkerConfig.codecEosTokenId

        // Suppress special tokens
        let suppressTokens = (talkerConfig.vocabSize - 1024 ..< talkerConfig.vocabSize)
            .filter { $0 != eosTokenId }

        var trailingIdx = 0
        var inputEmbeds = inputEmbedsInit

        let profiling = Qwen3TTSFullModel.profilingEnabled
        var timings = StepTimings()

        // Pre-allocate code predictor KV caches once; reuse via trim() each step.
        // step=16 matches the exact number of positions needed (vs default 256).
        let codePredictorCache = talker.codePredictor.makeCache()
        for c in codePredictorCache {
            (c as? KVCacheSimple)?.step = talkerConfig.numCodeGroups
        }

        for step in 0 ..< effectiveMaxTokens {
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
            let codeHidden = hidden[0..., (-1)..., 0...]

            let cpStart = CFAbsoluteTimeGetCurrent()
            for codeIdx in 0 ..< talkerConfig.numCodeGroups - 1 {
                let passStart = CFAbsoluteTimeGetCurrent()

                let codeInput: MLXArray
                if codeIdx == 0 {
                    let code0Embed = talker.getInputEmbeddings()(nextToken)
                    codeInput = concatenated([codeHidden, code0Embed], axis: 1)
                } else {
                    codeInput = talker.codePredictor.codecEmbedding[codeIdx - 1](codeTokens.last!)
                }

                let (codeLogits, _, _) = talker.codePredictor(
                    codeInput, cache: codePredictorCache, generationStep: codeIdx
                )

                // Simple temperature sampling for code predictor (no top-p sort needed)
                let nextCode = sampleToken(codeLogits, temperature: temperature, topP: 1.0)
                if profiling { eval(nextCode) }
                codeTokens.append(nextCode)

                if profiling {
                    timings.codePredictorPasses += CFAbsoluteTimeGetCurrent() - passStart
                    timings.codePredictorPassCount += 1
                }
            }
            let cpTime = CFAbsoluteTimeGetCurrent() - cpStart

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

            // Sum all code embeddings for next step
            var codecEmbed = talker.getInputEmbeddings()(nextToken)
            for (i, code) in codeTokens.dropFirst().enumerated() {
                codecEmbed = codecEmbed + talker.codePredictor.codecEmbedding[i](code)
            }

            // Sync eval the full lazy graph (code predictor 15 passes + embedding prep),
            // then free temporary GPU memory so the next step's talker gets clean GPU state.
            inputEmbeds = textEmbed + codecEmbed
            eval(inputEmbeds)
            GPU.clearCache()
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
        logger.info("\(tokenMsg, privacy: .public)")

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
            logger.info("\(summary, privacy: .public)")
        }

        guard !generatedCodes.isEmpty else {
            return MLXArray.zeros([1])
        }

        // Stack and decode
        let decodeStartTime = CFAbsoluteTimeGetCurrent()
        let codes = stacked(generatedCodes, axis: 1)  // [1, seq_len, num_code_groups]

        // Streaming decode for memory efficiency
        var audioChunks = [MLXArray]()
        for chunk in speechTokenizer.streamingDecode(codes, chunkTokens: 100) {
            audioChunks.append(chunk)
        }
        var audio = concatenated(audioChunks, axis: -1)[0]  // Remove batch dim

        // Trim to valid length
        let validLen = Int((codes[0..., 0..., 0] .> 0).sum().item(Int32.self)) * speechTokenizer.decodeUpsampleRate
        if validLen > 0 && validLen < audio.dim(0) {
            audio = audio[..<validLen]
        }

        eval(audio)

        let decodeTime = CFAbsoluteTimeGetCurrent() - decodeStartTime
        let totalTime = CFAbsoluteTimeGetCurrent() - genStartTime
        let decodeMsg = "Decode: \(String(format: "%.2f", decodeTime))s | Total: \(String(format: "%.2f", totalTime))s for \(String(format: "%.1f", audioSeconds))s audio (overall RTF=\(String(format: "%.2f", totalTime / max(audioSeconds, 0.001)))x)"
        logger.info("\(decodeMsg, privacy: .public)")

        return audio
    }

    // MARK: - Streaming VoiceDesign generation

    /// Generates speech tokens and progressively decodes/yields audio chunks.
    /// Each chunk is a 1D MLXArray of float32 audio samples at `sampleRate` Hz.
    func generateStreamingVoiceDesign(
        text: String,
        instruct: String?,
        language: String,
        temperature: Float,
        topP: Float,
        repetitionPenalty: Float,
        maxTokens: Int,
        emitEvery: Int = 8,
        decodeWindow: Int = 80,
        onAudioChunk: (MLXArray) -> Void
    ) {
        guard let speechTokenizer, let tokenizer else { return }

        let talkerConfig = config.talkerConfig!

        let (inputEmbedsInit, trailingTextHidden, ttsPadEmbed) = prepareGenerationInputs(
            text: text, language: language, instruct: instruct
        )

        let targetTokenCount = tokenizer.encode(text: text).count
        let effectiveMaxTokens = min(maxTokens, max(75, targetTokenCount * 6))

        let cache = talker.makeCache()
        var generatedCodes = [MLXArray]()
        var generatedTokenIds = [Int]()
        let eosTokenId = talkerConfig.codecEosTokenId

        let suppressTokens = (talkerConfig.vocabSize - 1024 ..< talkerConfig.vocabSize)
            .filter { $0 != eosTokenId }

        var trailingIdx = 0
        var inputEmbeds = inputEmbedsInit
        var totalEmitted = 0  // Number of tokens whose audio has been emitted

        // Pre-allocate code predictor KV caches once; reuse via trim() each step.
        let codePredictorCache = talker.codePredictor.makeCache()
        for c in codePredictorCache {
            (c as? KVCacheSimple)?.step = talkerConfig.numCodeGroups
        }

        for step in 0 ..< effectiveMaxTokens {
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
            let codeHidden = hidden[0..., (-1)..., 0...]

            for codeIdx in 0 ..< talkerConfig.numCodeGroups - 1 {
                let codeInput: MLXArray
                if codeIdx == 0 {
                    let code0Embed = talker.getInputEmbeddings()(nextToken)
                    codeInput = concatenated([codeHidden, code0Embed], axis: 1)
                } else {
                    codeInput = talker.codePredictor.codecEmbedding[codeIdx - 1](codeTokens.last!)
                }

                let (codeLogits, _, _) = talker.codePredictor(
                    codeInput, cache: codePredictorCache, generationStep: codeIdx
                )

                // Simple temperature sampling for code predictor (no top-p sort needed)
                let nextCode = sampleToken(codeLogits, temperature: temperature, topP: 1.0)
                codeTokens.append(nextCode)
            }

            let allCodes = concatenated(codeTokens, axis: 1)
            generatedCodes.append(allCodes)

            // Sliding window decode: emit audio every `emitEvery` tokens
            let pendingCount = generatedCodes.count - totalEmitted
            if pendingCount >= emitEvery {
                let samplesPerToken = speechTokenizer.decodeUpsampleRate  // 1920
                let stepSamples = emitEvery * samplesPerToken

                // Decode last min(decodeWindow, total) tokens — 85% context ratio
                let windowSize = min(decodeWindow, generatedCodes.count)
                let windowStart = generatedCodes.count - windowSize
                let windowCodes = stacked(Array(generatedCodes[windowStart..<generatedCodes.count]), axis: 1)

                // One decoder call on full window, no context trimming
                let wav = speechTokenizer.decodeSingleChunk(windowCodes, leftContextTokens: 0)
                let allSamples = wav[0]

                // Extract only the tail (newly emitted audio)
                let startSample = max(0, allSamples.dim(0) - stepSamples)
                let chunk = allSamples[startSample...]
                if chunk.dim(0) > 0 {
                    onAudioChunk(chunk)
                }
                totalEmitted += emitEvery
            }

            // Prepare next input
            let textEmbed: MLXArray
            if trailingIdx < trailingTextHidden.dim(1) {
                textEmbed = trailingTextHidden[0..., trailingIdx ..< (trailingIdx + 1), 0...]
                trailingIdx += 1
            } else {
                textEmbed = ttsPadEmbed
            }

            var codecEmbed = talker.getInputEmbeddings()(nextToken)
            for (i, code) in codeTokens.dropFirst().enumerated() {
                codecEmbed = codecEmbed + talker.codePredictor.codecEmbedding[i](code)
            }

            inputEmbeds = textEmbed + codecEmbed
            eval(inputEmbeds)
            GPU.clearCache()
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

            // Extract un-emitted tail
            let remainingSamples = remaining * samplesPerToken
            let startSample = max(0, allSamples.dim(0) - remainingSamples)
            var chunk = allSamples[startSample...]

            // Trim padding tokens (valid-length check)
            let newTokenCodes = stacked(Array(generatedCodes[totalEmitted...]), axis: 1)
            let validTokens = Int((newTokenCodes[0..., 0..., 0] .> 0).sum().item(Int32.self))
            let validSamples = validTokens * samplesPerToken
            if validSamples > 0 && validSamples < chunk.dim(0) {
                chunk = chunk[..<validSamples]
            }
            if chunk.dim(0) > 0 {
                onAudioChunk(chunk)
            }
        }
    }

    // MARK: - Prepare generation inputs

    func prepareGenerationInputs(
        text: String,
        language: String,
        instruct: String?
    ) -> (MLXArray, MLXArray, MLXArray) {
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

        // Full input embedding
        var inputEmbeds: MLXArray
        if let instructEmbed {
            inputEmbeds = concatenated([instructEmbed, roleEmbed, combinedEmbed], axis: 1)
        } else {
            inputEmbeds = concatenated([roleEmbed, combinedEmbed], axis: 1)
        }

        // Add first text token (index 3) + last codec embed
        let firstTextEmbed = textEmbed[0..., 3 ..< 4, 0...] + codecEmbed[0..., (-1)..., 0...]
        inputEmbeds = concatenated([inputEmbeds, firstTextEmbed], axis: 1)

        // Trailing text (tokens 4 to -5, plus EOS)
        let trailingTextHidden = concatenated(
            [textEmbed[0..., 4 ..< (textEmbed.dim(1) - 5), 0...], ttsEosEmbed],
            axis: 1
        )

        return (inputEmbeds, trailingTextHidden, ttsPadEmbed)
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
