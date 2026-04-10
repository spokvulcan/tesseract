import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLMCommon
import Tokenizers
import os

nonisolated final class UnsafeSendableBox<T>: @unchecked Sendable {
    let value: T

    init(_ value: T) {
        self.value = value
    }
}

/// Output of `LLMActor.makeHTTPPrefixCacheGeneration`. Bundles the lower-level MLX
/// generation handles together so the caller can drive the stream and recover the
/// final KV cache after generation completes.
nonisolated struct HTTPPrefixCacheGeneration: @unchecked Sendable {
    let stream: AsyncStream<Generation>
    let completion: Task<Void, Never>
    let finalCacheHandle: FinalizedKVCacheHandle
    /// Total prompt tokens (full conversation, ignoring slicing).
    let promptTokenCount: Int
    /// Number of leading tokens skipped because the cache already covered them.
    let skippedPrefillTokens: Int
}

/// Actor-isolated wrapper that owns the LLM model and runs inference off the MainActor.
///
/// Follows the same pattern as `WhisperActor` in `TranscriptionEngine`:
/// the `@MainActor` ``AgentEngine`` publishes UI state while delegating
/// heavy model operations to this actor.
actor LLMActor {

    /// Memory budget for the LLM stack.
    ///
    /// Math for Qwen3.5-Next-4B-paro on Apple Silicon (Mac15,9 / 48 GB unified):
    /// - Model weights (4-bit quant):      ~2.5 GB
    /// - MLX free buffer pool:             ~1.0 GB (cacheLimitMB)
    /// - Active inference KV cache:        ~1.0 GB (single in-flight 16 K context turn)
    /// - Activations / scratch:            ~1.0 GB
    /// - Subtotal (single in-flight turn): ~5.5 GB
    ///
    /// Plus the HTTP prefix cache, which retains a full KV state per stored entry.
    /// For Qwen3.5 hybrid (8 attention layers per 32, ~16 K context):
    ///   8 attn layers × 16384 tokens × 8 KV heads × 128 dim × 2 (K+V) × 2 B (fp16) ≈ 0.5 GB / entry
    /// At 32 K context this roughly doubles to ~1 GB / entry.
    ///
    /// With `httpPrefixCacheCapacity = 3`: ~3 GB cache budget → ~8.5 GB total target.
    /// With `httpPrefixCacheCapacity = 8`: ~8 GB cache budget → ~13.5 GB total — was the
    /// previous default and is what put the process at >20 GB once OpenCode subagents
    /// kept several distinct cache keys live simultaneously.
    ///
    /// Trade-off: smaller capacity → fewer parallel conversation chains kept warm.
    /// 3 is enough for "main agent + subagent + title gen" — the three keys we
    /// observe in OpenCode workloads. If you need more, raise it knowing each
    /// extra slot costs ~0.5–1 GB.
    private enum Defaults {
        static let cacheLimitMB = 2048
        static let httpPrefixCacheCapacity = 4
    }

    private var modelContainer: ModelContainer?
    private(set) var agentTokenizer: AgentTokenizer?
    private let httpPrefixCache = HTTPPrefixCacheSpikeStore(capacity: Defaults.httpPrefixCacheCapacity)
    private var promptStartsThinking = false

    var isLoaded: Bool { modelContainer != nil }

    /// Loads model weights, verifies with a 1-token generation, and resolves the tokenizer.
    ///
    /// Reads the model's `config.json` to detect the model type and configure the
    /// appropriate tool call format (e.g., `.xmlFunction` for Qwen3.5).
    ///
    /// - Parameters:
    ///   - directory: Local path containing model weights, config, and tokenizer files.
    ///   - visionMode: When `true`, loads the VLM variant of ParoQuant models (supports
    ///     image attachments but has ~3.4× slower prefill on long text prompts). When
    ///     `false`, loads the LLM variant with fast chunked prefill. Ignored for
    ///     non-ParoQuant models.
    /// - Returns: The resolved ``AgentTokenizer`` and whether the template starts inside a think block.
    @discardableResult
    func loadModel(
        from directory: URL,
        visionMode: Bool
    ) async throws -> (AgentTokenizer, promptStartsThinking: Bool) {
        let format = Self.detectToolCallFormat(directory: directory)
        Log.agent.info(
            "Loading model — visionMode=\(visionMode) "
            + "format=\(format.map { "\($0)" } ?? "json (default)")"
        )

        if isParoQuantModel(directory: directory) {
            Log.agent.info("Detected ParoQuant model — using \(visionMode ? "VLM" : "LLM") path")
            let container: ModelContainer = visionMode
                ? try await loadParoQuantVLMContainer(from: directory, toolCallFormat: format)
                : try await loadParoQuantLLMContainer(from: directory, toolCallFormat: format)
            return try await verifyAndStore(container: container, directory: directory)
        }

        let container = try await loadModelContainer(
            from: directory,
            using: #huggingFaceTokenizerLoader()
        )
        if let format {
            await container.update { context in
                context.configuration.toolCallFormat = format
            }
        }
        return try await verifyAndStore(container: container, directory: directory)
    }

    /// Prepares input and starts generation, returning the raw stream.
    ///
    /// The caller is responsible for iterating the stream and mapping
    /// `Generation` events to ``AgentGeneration``.
    func generate(
        input: sending UserInput,
        parameters: AgentGenerateParameters
    ) async throws -> AsyncStream<Generation> {
        guard let container = modelContainer else {
            throw AgentEngineError.modelNotLoaded
        }

        Memory.cacheLimit = Defaults.cacheLimitMB * 1024 * 1024

        let prepared = try await container.prepare(input: input)
        let genParams = Self.makeGenerateParameters(from: parameters)

        return try await container.generate(input: prepared, parameters: genParams)
    }

    /// Start the HTTP text-based prefix-cache path for `/v1/chat/completions`.
    ///
    /// Returns `nil` when the request shape is incompatible and the caller should fall back
    /// to the normal generation path.
    func generateServerTextCompletion(
        modelID: String,
        conversation: HTTPPrefixCacheConversation,
        toolSpecs: [ToolSpec]?,
        parameters: AgentGenerateParameters
    ) async throws -> HTTPServerGenerationStart? {
        guard let container = modelContainer else {
            throw AgentEngineError.modelNotLoaded
        }
        guard let lastMessage = conversation.lastMessage else {
            Log.agent.info("HTTP prefix cache bypass — model=\(modelID) reason=empty-conversation")
            return nil
        }
        guard lastMessage.role != .assistant else {
            Log.agent.info(
                "HTTP prefix cache bypass — model=\(modelID) reason=last-message-assistant"
            )
            return nil
        }

        let cacheKey = HTTPPrefixCacheKey(
            modelID: modelID,
            kvBits: parameters.kvBits,
            kvGroupSize: parameters.kvGroupSize,
            toolDefinitionsDigest: conversation.toolDefinitionsDigest,
            templateContextDigest: conversation.templateContextDigest
        )
        let lookup = await httpPrefixCache.lookup(conversation: conversation, key: cacheKey)
        let usableMatch = lookup.match

        if let usableMatch {
            Log.agent.info(
                "HTTP prefix cache HIT — model=\(modelID) cachedTokens=\(usableMatch.cachedTokenCount) "
                + "requestMessages=\(conversation.messages.count) cachedMessages=\(usableMatch.conversation.messages.count) "
                + "entriesForKey=\(lookup.keyedEntryCount)"
            )
            if let report = lookup.mismatchReport {
                // A strictly longer entry exists but didn't match — log why so we can
                // diagnose cases where the cache is stuck reusing a short prefix.
                // Split across lines so `log stream --style compact` (default ~200 char
                // truncation) doesn't drop the preview content.
                Self.logMismatchReport(prefix: "HTTP prefix cache HIT longer-entry mismatch", report: report)
            }
        } else {
            Log.agent.info(
                "HTTP prefix cache MISS — model=\(modelID) reason=\(lookup.reason) "
                + "requestMessages=\(conversation.messages.count) entriesForKey=\(lookup.keyedEntryCount)"
            )
            if let report = lookup.mismatchReport {
                Self.logMismatchReport(prefix: "HTTP prefix cache MISS detail", report: report)
            }
        }

        Memory.cacheLimit = Defaults.cacheLimitMB * 1024 * 1024

        let mlxStart = try await makeHTTPPrefixCacheGeneration(
            container: container,
            conversation: conversation,
            match: usableMatch,
            parameters: Self.makeGenerateParameters(from: parameters),
            toolSpecs: toolSpecs
        )
        Log.agent.info(
            "HTTP prefix cache prefill plan — model=\(modelID) "
            + "promptTokens=\(mlxStart.promptTokenCount) "
            + "skippedPrefillTokens=\(mlxStart.skippedPrefillTokens) "
            + "newTokensToPrefill=\(mlxStart.promptTokenCount - mlxStart.skippedPrefillTokens) "
            + "matchUsed=\(usableMatch != nil)"
        )

        let mlxStartBox = UnsafeSendableBox(mlxStart)
        let (stream, continuation) = AsyncThrowingStream<AgentGeneration, Error>.makeStream()
        let startsInsideThinkBlock = promptStartsThinking

        let task = Task { [httpPrefixCache, conversation, cacheKey, container, toolSpecs] in
            let mlxStart = mlxStartBox.value
            var textContent = ""
            var thinkingContent = ""
            var toolCalls: [HTTPPrefixCacheToolCall] = []
            let parser = ToolCallParser(startsInsideThinkBlock: startsInsideThinkBlock)
            var rawChunkParts: [String] = []
            var libraryParsedToolCalls = false
            var completionInfo: AgentGeneration.Info?

            do {
                func handle(_ event: AgentGeneration) {
                    switch event {
                    case .text(let text):
                        textContent += text
                    case .thinking(let chunk):
                        thinkingContent += chunk
                    case .thinkReclassify:
                        textContent += thinkingContent
                        thinkingContent = ""
                    case .toolCall(let call):
                        toolCalls.append(HTTPPrefixCacheToolCall(
                            name: call.function.name,
                            arguments: call.function.arguments
                        ))
                    case .malformedToolCall:
                        break
                    case .thinkStart, .thinkEnd, .info:
                        break
                    }

                    continuation.yield(event)
                }

                func emitParserEvents(_ events: [ToolCallParser.Event], allowToolEvents: Bool) {
                    for event in events {
                        switch event {
                        case .toolCall, .malformedToolCall where !allowToolEvents:
                            continue
                        default:
                            handle(AgentGeneration(parserEvent: event))
                        }
                    }
                }

                for await item in mlxStart.stream {
                    if Task.isCancelled {
                        break
                    }

                    switch item {
                    case .chunk(let text):
                        rawChunkParts.append(text)
                        emitParserEvents(
                            parser.processChunk(text),
                            allowToolEvents: !libraryParsedToolCalls
                        )

                    case .toolCall(let call):
                        libraryParsedToolCalls = true
                        handle(.toolCall(call))

                    case .info(let info):
                        completionInfo = .init(
                            promptTokenCount: info.promptTokenCount,
                            generationTokenCount: info.generationTokenCount,
                            promptTime: info.promptTime,
                            generateTime: info.generateTime
                        )
                    }
                }

                // Wait for the underlying iterator task to finish before extracting
                // the cache (mirrors ChatSession's pattern at vendor line 440-441).
                await mlxStart.completion.value

                emitParserEvents(
                    parser.finalize(),
                    allowToolEvents: !libraryParsedToolCalls
                )

                if let completionInfo {
                    handle(.info(completionInfo))
                    Log.agent.info(
                        "Generation complete — \(completionInfo.generationTokenCount) tokens, "
                        + "\(String(format: "%.1f", completionInfo.tokensPerSecond)) tok/s"
                    )
                    let rawChunks = rawChunkParts.joined()
                    Log.agent.debug("Raw library chunks (after ToolCallProcessor):\n\(rawChunks)")
                    if !libraryParsedToolCalls &&
                        (rawChunks.contains("tool_call") || rawChunks.contains("<function"))
                    {
                        Log.agent.warning(
                            "Raw output contains tool call markers but no .toolCall events were emitted by library"
                        )
                    }
                }

                if !Task.isCancelled,
                   let finalCache = await mlxStart.finalCacheHandle.takeFinalCache() {
                    // HTTPPrefixCacheMessage.assistant() normalizes whitespace-only
                    // content to "" so the stored entry can prefix-match echoes from
                    // clients (OpenCode) that strip such content.
                    let storedConversation = conversation.appendingAssistant(.assistant(
                        content: textContent,
                        reasoning: thinkingContent,
                        toolCalls: toolCalls
                    ))
                    let cacheOffsets = httpPrefixCacheOffsets(finalCache)
                    let fallbackTokenCount = httpPrefixCacheReportedTokenCount(finalCache)
                    if httpPrefixCacheHasReusableState(finalCache) {
                        let cachedTokenCount = await Self.measureHTTPPrefixCacheTokenCount(
                            container: container,
                            conversation: storedConversation,
                            toolSpecs: toolSpecs,
                            fallback: fallbackTokenCount
                        )

                        // If content normalization shortened the stored conversation
                        // (whitespace-only assistant content → ""), the actual cache
                        // offset is HIGHER than the re-tokenized cachedTokenCount.
                        // Trim the trimmable layers (KVCacheSimple/Quantized) so the
                        // attention offset matches what the next request will compute,
                        // and the slicing math on lookup stays consistent.
                        // Mamba layers cannot be trimmed; their state is left as-is
                        // (small recurrent divergence — acceptable for the prototype).
                        let actualOffsetBefore = fallbackTokenCount
                        if actualOffsetBefore > cachedTokenCount {
                            let trimAmount = actualOffsetBefore - cachedTokenCount
                            var trimmedLayerCount = 0
                            for layer in finalCache where layer.isTrimmable {
                                let actuallyTrimmed = layer.trim(trimAmount)
                                if actuallyTrimmed > 0 {
                                    trimmedLayerCount += 1
                                }
                            }
                            let actualOffsetAfter = httpPrefixCacheReportedTokenCount(finalCache)
                            Log.agent.info(
                                "HTTP prefix cache offset trim — model=\(modelID) "
                                + "trimAmount=\(trimAmount) "
                                + "offsetBefore=\(actualOffsetBefore) offsetAfter=\(actualOffsetAfter) "
                                + "trimmedLayers=\(trimmedLayerCount)/\(finalCache.count) "
                                + "(reason=normalized assistant content shortened conversation)"
                            )
                        }

                        await httpPrefixCache.store(
                            conversation: storedConversation,
                            key: cacheKey,
                            cachedTokenCount: cachedTokenCount,
                            cache: finalCache
                        )
                        // Release the MLX free buffer pool back to the OS so it
                        // doesn't accumulate transient prefill intermediates
                        // across requests. The next allocation pays a small
                        // cost going to the OS but steady-state RSS stays
                        // bounded. Without this we observed peakMemMB climbing
                        // monotonically (25 GB → 44 GB → OOM kill) on long
                        // prefill chains.
                        Memory.clearCache()
                        let snapshot = await httpPrefixCache.snapshot()
                        let activeMB = Float(Memory.activeMemory) / 1e6
                        let peakMB = Float(Memory.peakMemory) / 1e6
                        Log.agent.info(
                            "HTTP prefix cache STORE — model=\(modelID) cachedTokens=\(cachedTokenCount) "
                            + "conversationMessages=\(conversation.messages.count + 1) "
                            + "responseCharacters=\(textContent.count) toolCalls=\(toolCalls.count) "
                            + "reasoningCharacters=\(thinkingContent.count) "
                            + "cacheOffsets=\(cacheOffsets) "
                            + "entries=\(snapshot.entryCount)/\(snapshot.capacity) "
                            + "totalCachedTokens=\(snapshot.totalCachedTokens) "
                            + "activeMemMB=\(String(format: "%.0f", activeMB)) "
                            + "peakMemMB=\(String(format: "%.0f", peakMB))"
                        )
                    } else {
                        Log.agent.info(
                            "HTTP prefix cache store skipped — model=\(modelID) reason=no-reusable-cache-state "
                            + "conversationMessages=\(conversation.messages.count + 1) "
                            + "cacheOffsets=\(cacheOffsets)"
                        )
                    }
                } else if !Task.isCancelled {
                    Log.agent.warning(
                        "HTTP prefix cache store skipped — model=\(modelID) reason=no-final-cache"
                    )
                }

                continuation.finish()
            } catch is CancellationError {
                continuation.finish()
            } catch {
                continuation.finish(throwing: AgentEngineError.generationFailed(
                    error.localizedDescription
                ))
            }
        }

        continuation.onTermination = { _ in
            task.cancel()
        }

        return HTTPServerGenerationStart(
            stream: stream,
            cachedTokenCount: mlxStart.skippedPrefillTokens
        )
    }

    /// Renders messages and tools through the Jinja chat template, returning the exact
    /// ChatML string the model receives as input (including `<|im_start|>`, tool definitions, etc.).
    func formatRawPrompt(
        messages: [[String: any Sendable]],
        tools: [ToolSpec]?
    ) async throws -> String {
        guard let container = modelContainer else {
            throw AgentEngineError.modelNotLoaded
        }
        return try await container.perform { context in
            let tokens = try context.tokenizer.applyChatTemplate(
                messages: messages, tools: tools
            )
            return context.tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false)
        }
    }

    /// Formats and returns the raw ChatML string along with its token count in a single actor hop.
    func formatRawPromptWithCount(
        messages: [[String: any Sendable]],
        tools: [ToolSpec]?
    ) async throws -> (text: String, tokenCount: Int) {
        guard let container = modelContainer else {
            throw AgentEngineError.modelNotLoaded
        }
        return try await container.perform { context in
            let tokens = try context.tokenizer.applyChatTemplate(
                messages: messages, tools: tools
            )
            let text = context.tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false)
            return (text, tokens.count)
        }
    }

    /// Releases the model from memory.
    func unloadModel() async {
        modelContainer = nil
        agentTokenizer = nil
        promptStartsThinking = false
        await httpPrefixCache.clear()
    }

    /// Frees unreferenced MLX buffers between tool rounds.
    func clearMemoryCache() {
        Memory.clearCache()
    }

    /// Returns current MLX memory usage in MB.
    func memoryStats() -> (activeMB: Float, peakMB: Float) {
        (Float(Memory.activeMemory) / 1e6, Float(Memory.peakMemory) / 1e6)
    }

    // MARK: - Private

    /// Verifies the model with a 1-token generation, stores it, and returns the tokenizer.
    private func verifyAndStore(
        container: ModelContainer, directory: URL
    ) async throws -> (AgentTokenizer, promptStartsThinking: Bool) {
        // Wrap in withError so C++ MLX errors (e.g. matmul shape mismatches) throw
        // instead of calling fatalError via the default error handler.
        try await withError {
            let input = try await container.prepare(input: UserInput(prompt: "Hello"))
            let stream = try await container.generate(
                input: input, parameters: GenerateParameters(maxTokens: 1)
            )
            for await _ in stream {}
        }

        let tokenizer = try await AgentTokenizer(container: container)
        let startsThinking = Self.detectPromptStartsThinking(directory: directory)
        modelContainer = container
        agentTokenizer = tokenizer
        promptStartsThinking = startsThinking
        await httpPrefixCache.clear()
        return (tokenizer, startsThinking)
    }

    /// Build the lower-level MLX generation pipeline for the HTTP prefix cache path.
    ///
    /// Always renders the FULL conversation through the chat template. When a cache
    /// match is available and the cache offset is strictly less than the full prompt
    /// token count, slices the rendered tokens at `cacheOffset` so MLX prefill only
    /// processes the new suffix. The cache provides the K/V state for the leading
    /// `cacheOffset` tokens, and Qwen3.5's attention layers use `cache.offset` to
    /// position-encode (RoPE) the new tokens correctly.
    ///
    /// Bypasses `ChatSession` because its `init(cache:)` path renders only the new
    /// message and drops intermediate history, which produces incoherent output when
    /// the cached state corresponds to a strict prefix of the request rather than the
    /// most recent turn.
    private func makeHTTPPrefixCacheGeneration(
        container: ModelContainer,
        conversation: HTTPPrefixCacheConversation,
        match: HTTPPrefixCacheMatch?,
        parameters: GenerateParameters,
        toolSpecs: [ToolSpec]?
    ) async throws -> HTTPPrefixCacheGeneration {
        // Capture the Sendable conversation by value; build the non-Sendable
        // [Chat.Message] history INSIDE the closure to avoid a Sendable warning.
        let matchBox = match.map { UnsafeSendableBox($0) }

        return try await container.perform { context in
            let history = conversation.historyMessages
            let userInput = UserInput(chat: history, tools: toolSpecs)
            let fullInput = try await context.processor.prepare(input: userInput)
            // Sequence length is always the LAST dim. For LLM models tokens are
            // 1D `[seq]`, for VLM models (ParoQuant Qwen35) they are 2D
            // `[batch, seq]`. Both cases use `dim(-1)` for the seq length.
            let fullTokenCount = fullInput.text.tokens.dim(-1)
            let tokenNDim = fullInput.text.tokens.ndim

            let inputForGeneration: LMInput
            let cacheToUse: [KVCache]?
            let skippedTokens: Int

            if let match = matchBox?.value {
                let cacheOffset = httpPrefixCacheReportedTokenCount(match.cache)
                if cacheOffset > 0 && cacheOffset < fullTokenCount {
                    // Slice the sequence dim (always the last one). The Text
                    // subscript slices dim 0 by default, which is the BATCH dim
                    // for 2D inputs — that would produce an empty tensor and
                    // crash QuantizedEmbedding.reshape downstream.
                    let slicedTokens: MLXArray
                    if tokenNDim <= 1 {
                        slicedTokens = fullInput.text.tokens[cacheOffset...]
                    } else {
                        slicedTokens = fullInput.text.tokens[0..., cacheOffset...]
                    }
                    // Drop the mask on the slice — for our HTTP path the
                    // input is always pure text and downstream code recreates
                    // attention masks from the cache offset, so an
                    // out-of-bounds mask would actively confuse the model.
                    let slicedText = LMInput.Text(tokens: slicedTokens, mask: nil)
                    inputForGeneration = LMInput(text: slicedText)
                    cacheToUse = match.cache
                    skippedTokens = cacheOffset
                } else {
                    // Cache offset is 0 (no reusable state) or covers the entire
                    // request — fall back to a fresh full prefill.
                    inputForGeneration = fullInput
                    cacheToUse = nil
                    skippedTokens = 0
                }
            } else {
                inputForGeneration = fullInput
                cacheToUse = nil
                skippedTokens = 0
            }

            let iterator = try TokenIterator(
                input: inputForGeneration,
                model: context.model,
                cache: cacheToUse,
                parameters: parameters
            )

            let (stream, task, finalCacheHandle) = MLXLMCommon.generateTaskWithFinalCache(
                promptTokenCount: fullTokenCount,
                modelConfiguration: context.configuration,
                tokenizer: context.tokenizer,
                iterator: iterator
            )

            return HTTPPrefixCacheGeneration(
                stream: stream,
                completion: task,
                finalCacheHandle: finalCacheHandle,
                promptTokenCount: fullTokenCount,
                skippedPrefillTokens: skippedTokens
            )
        }
    }

    /// Emit a mismatch report as a series of short lines so the default
    /// `log stream --style compact` truncation doesn't drop the content preview.
    /// One line per logical field; an additional `previewStored=` /
    /// `previewRequest=` pair is split out for content/reasoning/toolCall fields.
    private static func logMismatchReport(prefix: String, report: HTTPPrefixCacheMismatchReport) {
        switch report {
        case .messageFieldMismatch(let i, let role, let field, let sl, let rl, let sh, let rh, let sp, let rp):
            Log.agent.info(
                "\(prefix) — message[\(i)](\(role)).\(field) storedLen=\(sl) reqLen=\(rl) storedHash=\(sh) reqHash=\(rh)"
            )
            Log.agent.info("\(prefix) — message[\(i)] storedPreview=\"\(sp)\"")
            Log.agent.info("\(prefix) — message[\(i)] reqPreview=\"\(rp)\"")
        case .toolCallArgumentsMismatch(let mi, let ti, let toolName, let sl, let rl, let sh, let rh, let sp, let rp):
            Log.agent.info(
                "\(prefix) — message[\(mi)].toolCalls[\(ti)] tool=\(toolName) storedLen=\(sl) reqLen=\(rl) storedHash=\(sh) reqHash=\(rh)"
            )
            Log.agent.info("\(prefix) — message[\(mi)].toolCalls[\(ti)] storedPreview=\"\(sp)\"")
            Log.agent.info("\(prefix) — message[\(mi)].toolCalls[\(ti)] reqPreview=\"\(rp)\"")
        default:
            Log.agent.info("\(prefix) — \(report)")
        }
    }

    private static func makeGenerateParameters(
        from parameters: AgentGenerateParameters
    ) -> GenerateParameters {
        GenerateParameters(
            maxTokens: parameters.maxTokens,
            kvBits: parameters.kvBits,
            kvGroupSize: parameters.kvGroupSize,
            temperature: parameters.temperature,
            topP: parameters.topP,
            topK: parameters.topK,
            minP: parameters.minP,
            repetitionPenalty: parameters.repetitionPenalty,
            repetitionContextSize: parameters.repetitionContextSize,
            presencePenalty: parameters.presencePenalty,
            prefillStepSize: parameters.prefillStepSize
        )
    }

    private static func measureHTTPPrefixCacheTokenCount(
        container: ModelContainer,
        conversation: HTTPPrefixCacheConversation,
        toolSpecs: [ToolSpec]?,
        fallback: Int
    ) async -> Int {
        do {
            let prepared = try await container.prepare(input: UserInput(
                chat: conversation.historyMessages,
                tools: toolSpecs
            ))
            return prepared.text.tokens.size
        } catch {
            Log.agent.warning(
                "HTTP prefix cache token count fallback — error=\(error.localizedDescription) "
                + "fallbackTokens=\(fallback)"
            )
            return fallback
        }
    }

    /// Returns `true` if the model's chat template appends `<think>` to the generation prompt.
    ///
    /// Detected models:
    /// - Qwen3.5: `enable_thinking` defaults to true, template ends with `<think>\n`
    /// - Qwen3 Thinking / Opus Distill: unconditionally append `<think>\n`
    /// - Qwen3 Instruct: no thinking in prompt → returns false
    private static func detectPromptStartsThinking(directory: URL) -> Bool {
        let templateURL = directory.appendingPathComponent("chat_template.jinja")
        guard let template = try? String(contentsOf: templateURL, encoding: .utf8) else {
            return false
        }

        // Check if the add_generation_prompt section contains <think>
        // All known thinking templates put <think> right after <|im_start|>assistant
        // in the add_generation_prompt block at the end of the template.
        guard let genPromptRange = template.range(of: "add_generation_prompt") else {
            return false
        }
        return template[genPromptRange.upperBound...].contains("<think>")
    }

    /// Reads `config.json` from the model directory to detect the model type
    /// and return the appropriate ``ToolCallFormat``.
    ///
    /// Qwen3.5 uses XML function syntax (`<function=name>...</function>`) inside
    /// `<tool_call>` tags, which requires `.xmlFunction` format.
    private static func detectToolCallFormat(directory: URL) -> ToolCallFormat? {
        let configURL = directory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let modelType = json["model_type"] as? String
        else { return nil }

        // Qwen3.5 chat template instructs XML function format for tool calls
        if modelType.hasPrefix("qwen3_5") {
            return .xmlFunction
        }

        return ToolCallFormat.infer(from: modelType)
    }
}
