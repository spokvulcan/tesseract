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

    // -- Post-generation store context (radix tree flow) --

    /// Flat token sequence for the full prompt (1D extraction from potentially 2D VLM tensor).
    let fullTokens: [Int]
    /// Snapshots captured during prefill at checkpoint offsets (e.g. stable-prefix boundary).
    let capturedSnapshots: [HybridCacheSnapshot]
    /// Partition key used for cache routing.
    let partitionKey: CachePartitionKey
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
        /// Memory budget for the prefix cache. Each snapshot costs ~200–600 MiB
        /// depending on context length. 3 GiB fits ~5–15 snapshots for typical
        /// Qwen3.5 workloads (system + leaf per concurrent conversation).
        static let prefixCacheMemoryBudgetBytes = 3 * 1024 * 1024 * 1024 // 3 GiB
    }

    private var modelContainer: ModelContainer?
    private(set) var agentTokenizer: AgentTokenizer?
    private var _prefixCache: PrefixCacheManager?
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
            Log.agent.info("Prefix cache bypass — model=\(modelID) reason=empty-conversation")
            return nil
        }
        guard lastMessage.role != .assistant else {
            Log.agent.info(
                "Prefix cache bypass — model=\(modelID) reason=last-message-assistant"
            )
            return nil
        }

        Memory.cacheLimit = Defaults.cacheLimitMB * 1024 * 1024

        let prefixCache = await ensurePrefixCache()
        // Canonicalize tools once so the leaf re-tokenization uses the same dict
        // iteration order as the prefill path inside makeHTTPPrefixCacheGeneration.
        let canonicalTools = Self.canonicalizeToolSpecs(toolSpecs)
        let mlxStart = try await makeHTTPPrefixCacheGeneration(
            container: container,
            conversation: conversation,
            modelID: modelID,
            parameters: Self.makeGenerateParameters(from: parameters),
            toolSpecs: canonicalTools,
            prefixCache: prefixCache
        )

        let mlxStartBox = UnsafeSendableBox(mlxStart)
        let (stream, continuation) = AsyncThrowingStream<AgentGeneration, Error>.makeStream()
        let startsInsideThinkBlock = promptStartsThinking

        let task = Task { [conversation, container, canonicalTools] in
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

                // -- Post-generation: store snapshots in radix tree --

                // Store mid-prefill snapshots (e.g. stable-prefix boundary) unconditionally.
                // These are captured during prefill and independent of the leaf path — if
                // final-cache recovery or leaf capture fails, the stable-prefix checkpoint
                // still saves future requests from a full re-prefill.
                if !Task.isCancelled, !mlxStart.capturedSnapshots.isEmpty {
                    await MainActor.run {
                        prefixCache.storeSnapshots(
                            promptTokens: mlxStart.fullTokens,
                            capturedSnapshots: mlxStart.capturedSnapshots,
                            partitionKey: mlxStart.partitionKey
                        )
                    }
                }

                if !Task.isCancelled,
                   let finalCache = await mlxStart.finalCacheHandle.takeFinalCache()
                {
                    let cacheOffsets = httpPrefixCacheOffsets(finalCache)
                    guard httpPrefixCacheHasReusableState(finalCache) else {
                        Log.agent.info(
                            "Prefix cache store skipped — model=\(modelID) reason=no-reusable-cache-state "
                            + "cacheOffsets=\(cacheOffsets)"
                        )
                        continuation.finish()
                        return
                    }

                    // 1. Build stored conversation (prompt + generated assistant turn).
                    let storedConversation = conversation.appendingAssistant(.assistant(
                        content: textContent,
                        reasoning: thinkingContent,
                        toolCalls: toolCalls
                    ))

                    // 3. Re-tokenize stored conversation → flat token sequence.
                    guard let storedTokens = await Self.measureStoredTokenSequence(
                        container: container,
                        conversation: storedConversation,
                        toolSpecs: canonicalTools
                    ) else {
                        Log.agent.warning(
                            "Prefix cache leaf store skipped — model=\(modelID) reason=tokenization-failed"
                        )
                        continuation.finish()
                        return
                    }

                    // 4. Offset-alignment: trim attention KV if normalization shortened
                    //    the stored conversation (whitespace-only assistant content → "").
                    //    Mamba layers cannot be trimmed; their state is left as-is
                    //    (small recurrent divergence — acceptable).
                    let actualCacheOffset = httpPrefixCacheReportedTokenCount(finalCache)
                    if actualCacheOffset > storedTokens.count {
                        let trimAmount = actualCacheOffset - storedTokens.count
                        var trimmedLayerCount = 0
                        for layer in finalCache where layer.isTrimmable {
                            let actuallyTrimmed = layer.trim(trimAmount)
                            if actuallyTrimmed > 0 {
                                trimmedLayerCount += 1
                            }
                        }
                        Log.agent.info(
                            "Prefix cache offset trim — model=\(modelID) "
                            + "trimAmount=\(trimAmount) "
                            + "offsetBefore=\(actualCacheOffset) offsetAfter=\(storedTokens.count) "
                            + "trimmedLayers=\(trimmedLayerCount)/\(finalCache.count)"
                        )
                    }

                    // 5. Capture leaf snapshot from the final (trimmed) cache.
                    if let leafSnapshot = HybridCacheSnapshot.capture(
                        cache: finalCache,
                        offset: storedTokens.count,
                        type: .leaf
                    ) {
                        await MainActor.run {
                            prefixCache.storeLeaf(
                                storedTokens: storedTokens,
                                leafSnapshot: leafSnapshot,
                                partitionKey: mlxStart.partitionKey
                            )
                        }

                        // Release the MLX free buffer pool back to the OS so it
                        // doesn't accumulate transient prefill intermediates
                        // across requests.
                        Memory.clearCache()

                        let stats = await MainActor.run { prefixCache.stats }
                        let activeMB = Float(Memory.activeMemory) / 1e6
                        let peakMB = Float(Memory.peakMemory) / 1e6
                        Log.agent.info(
                            "Prefix cache STORE — model=\(modelID) "
                            + "leafTokens=\(storedTokens.count) "
                            + "conversationMessages=\(conversation.messages.count + 1) "
                            + "responseCharacters=\(textContent.count) toolCalls=\(toolCalls.count) "
                            + "cacheOffsets=\(cacheOffsets) "
                            + "snapshots=\(stats.snapshotCount) "
                            + "partitions=\(stats.partitionCount) "
                            + "totalSnapshotMB=\(String(format: "%.0f", Float(stats.totalSnapshotBytes) / 1e6)) "
                            + "activeMemMB=\(String(format: "%.0f", activeMB)) "
                            + "peakMemMB=\(String(format: "%.0f", peakMB))"
                        )
                    } else {
                        Log.agent.info(
                            "Prefix cache leaf capture skipped — model=\(modelID) "
                            + "reason=unsupported-cache-type cacheOffsets=\(cacheOffsets)"
                        )
                    }
                } else if !Task.isCancelled {
                    Log.agent.warning(
                        "Prefix cache store skipped — model=\(modelID) reason=no-final-cache"
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
        _prefixCache = nil
    }

    /// Frees unreferenced MLX buffers between tool rounds.
    func clearMemoryCache() {
        Memory.clearCache()
    }

    /// Returns current MLX memory usage in MB.
    func memoryStats() -> (activeMB: Float, peakMB: Float) {
        (Float(Memory.activeMemory) / 1e6, Float(Memory.peakMemory) / 1e6)
    }

    // MARK: - Prefix Cache Helpers

    /// Lazily creates and returns the `PrefixCacheManager`. Initialization requires
    /// a MainActor hop because PrefixCacheManager is `@MainActor`.
    private func ensurePrefixCache() async -> PrefixCacheManager {
        if let existing = _prefixCache { return existing }
        let cache = await PrefixCacheManager(memoryBudgetBytes: Defaults.prefixCacheMemoryBudgetBytes)
        _prefixCache = cache
        return cache
    }

    /// Extract a flat `[Int]` token sequence from an MLXArray that may be 1D `[seq]`
    /// (LLM-only) or 2D `[batch, seq]` (VLM). Uses the first batch element for 2D.
    private static func extractTokenSequence(_ tokens: MLXArray) -> [Int] {
        if tokens.ndim <= 1 {
            return tokens.asArray(Int.self)
        } else {
            return tokens[0].asArray(Int.self)
        }
    }

    /// Canonicalize tool specs by round-tripping through `JSONSerialization` with
    /// `.sortedKeys`. Returns dicts with deterministic key ordering, so downstream
    /// Jinja `tojson` calls produce the same token sequence on every invocation.
    /// Without this, swift-jinja's `Value(any:)` path has non-deterministic dict
    /// iteration that makes token-level prefix caching unreliable.
    /// Internal (not private) so tests can exercise it directly.
    nonisolated static func canonicalizeToolSpecs(_ tools: [ToolSpec]?) -> [ToolSpec]? {
        guard let tools else { return nil }
        return tools.map { canonicalizeToolDict($0) }
    }

    nonisolated static func canonicalizeToolDict(_ dict: ToolSpec) -> ToolSpec {
        guard JSONSerialization.isValidJSONObject(dict),
              let data = try? JSONSerialization.data(withJSONObject: dict, options: [.sortedKeys]),
              let parsed = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return dict
        }
        return sendableDict(from: parsed)
    }

    /// Recursively rebuild a JSON-decoded `[String: Any]` as `[String: any Sendable]`.
    /// Every JSON primitive has a Sendable counterpart; we explicitly narrow each.
    private nonisolated static func sendableDict(from dict: [String: Any]) -> [String: any Sendable] {
        var result: [String: any Sendable] = [:]
        for (key, value) in dict {
            if let narrowed = sendableJSONValue(from: value) {
                result[key] = narrowed
            }
        }
        return result
    }

    private nonisolated static func sendableJSONValue(from value: Any) -> (any Sendable)? {
        if value is NSNull { return "" as String }  // JSON null → empty string (closest Sendable equivalent)
        if let v = value as? String { return v }
        if let v = value as? Bool { return v }
        if let v = value as? Int { return v }
        if let v = value as? Double { return v }
        if let v = value as? NSNumber {
            // Foundation may box booleans as NSNumber; disambiguate.
            if CFGetTypeID(v) == CFBooleanGetTypeID() { return v.boolValue }
            if v.stringValue.contains(".") { return v.doubleValue }
            return v.intValue
        }
        if let v = value as? [Any] {
            return v.compactMap { sendableJSONValue(from: $0) }
        }
        if let v = value as? [String: Any] {
            return sendableDict(from: v)
        }
        return nil
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
        _prefixCache = nil
        return (tokenizer, startsThinking)
    }

    /// Build the lower-level MLX generation pipeline using the radix-tree prefix cache.
    ///
    /// Flow: tokenize full conversation → extract flat token sequence → detect stable
    /// prefix boundary → radix tree lookup → plan checkpoints → slice suffix on hit →
    /// set checkpoint params → create TokenIterator (captures snapshots during prefill)
    /// → start generation stream.
    ///
    /// Bypasses `ChatSession` because its `init(cache:)` path renders only the new
    /// message and drops intermediate history, which produces incoherent output when
    /// the cached state corresponds to a strict prefix of the request rather than the
    /// most recent turn.
    private func makeHTTPPrefixCacheGeneration(
        container: ModelContainer,
        conversation: HTTPPrefixCacheConversation,
        modelID: String,
        parameters: GenerateParameters,
        toolSpecs: [ToolSpec]?,
        prefixCache: PrefixCacheManager
    ) async throws -> HTTPPrefixCacheGeneration {
        // Canonicalize tools once so the stable-prefix detector and the real
        // prefill tokenize against identical dict representations. Historically
        // swift-jinja <2.3.5 had non-deterministic `tojson` key ordering; the
        // canonicalization is kept as defense-in-depth and costs almost nothing.
        let canonicalTools = Self.canonicalizeToolSpecs(toolSpecs)

        // Capture promptStartsThinking for the non-MainActor closure below.
        let promptStartsThinking = self.promptStartsThinking

        return try await container.perform { context in
            // 1. Tokenize the full conversation (BEFORE cache lookup).
            let history = conversation.historyMessages
            let fullInput = try await context.processor.prepare(
                input: UserInput(chat: history, tools: canonicalTools)
            )
            // Sequence length is always the LAST dim. For LLM models tokens are
            // 1D [seq], for VLM models (ParoQuant Qwen35) they are 2D [batch, seq].
            let fullTokenCount = fullInput.text.tokens.dim(-1)
            let tokenNDim = fullInput.text.tokens.ndim

            // 2. Extract flat token sequence for radix tree operations.
            let fullTokens = Self.extractTokenSequence(fullInput.text.tokens)

            // 3. Build partition key for radix-tree routing.
            let partitionKey = CachePartitionKey(
                modelID: modelID,
                kvBits: parameters.kvBits,
                kvGroupSize: parameters.kvGroupSize
            )

            // 4a. Detect stable prefix boundary (system + tools) via two-probe technique.
            let stablePrefixOffset = try StablePrefixDetector.detect(
                systemPrompt: conversation.systemPrompt,
                toolSpecs: canonicalTools,
                fullTokens: fullTokens,
                tokenizer: context.tokenizer
            )

            // 4b. Detect the last-message boundary: the offset where the final
            // history message ends, right before the assistant-generation
            // prompt (e.g. `<|im_start|>assistant\n<think>\n` for Qwen3.5).
            //
            // WHY: templates like Qwen3.5 re-render non-latest assistant
            // messages WITHOUT their `<think>...</think>` blocks (via the
            // template's `last_query_index` logic). Turn N stored a leaf at
            // the full post-generation offset, but turn N+1 tokenizes the
            // same history with old assistants stripped of think blocks, so
            // the leaf path is unreachable. A checkpoint at the last-message
            // boundary (before the current-turn assistant prompt) IS stable
            // across turns: turn N+1 has the same prefix up to its own last
            // user message, regardless of think-block rewriting.
            //
            // HOW: the MLXLMCommon `Tokenizer` protocol doesn't expose
            // `addGenerationPrompt`, so we can't re-tokenize without the
            // suffix. Instead we compute the suffix length by encoding the
            // known generation-prompt string and subtracting. The gen prompt
            // is the fixed trailing string appended by the Jinja template.
            let genPromptStr = promptStartsThinking
                ? "<|im_start|>assistant\n<think>\n"
                : "<|im_start|>assistant\n"
            let genPromptTokens = context.tokenizer.encode(
                text: genPromptStr, addSpecialTokens: false
            )
            let lastMessageBoundaryOffset: Int?
            if genPromptTokens.count > 0,
               fullTokens.count > genPromptTokens.count,
               Array(fullTokens.suffix(genPromptTokens.count)).elementsEqual(genPromptTokens)
            {
                lastMessageBoundaryOffset = fullTokens.count - genPromptTokens.count
            } else {
                lastMessageBoundaryOffset = nil
            }

            // 5–6. Radix tree lookup + checkpoint planning (single MainActor hop).
            let (lookupResult, initialCheckpointPlan) = await MainActor.run {
                let lookup = prefixCache.lookup(tokens: fullTokens, partitionKey: partitionKey)
                let plan = prefixCache.planCheckpoints(
                    tokens: fullTokens,
                    stablePrefixOffset: stablePrefixOffset,
                    lastMessageBoundaryOffset: lastMessageBoundaryOffset,
                    partitionKey: partitionKey
                )
                return (lookup, plan)
            }
            var checkpointPlan = initialCheckpointPlan

            // 7. Determine input for generation and cache to restore.
            let inputForGeneration: LMInput
            let cacheToUse: [any KVCache]?
            let skippedTokens: Int
            let checkpointBaseOffset: Int

            if let snapshot = lookupResult.snapshot, snapshot.tokenOffset > 0,
               snapshot.tokenOffset < fullTokenCount
            {
                // HIT: restore cache, prefill only the suffix.
                let cacheOffset = snapshot.tokenOffset
                let slicedTokens: MLXArray
                if tokenNDim <= 1 {
                    slicedTokens = fullInput.text.tokens[cacheOffset...]
                } else {
                    slicedTokens = fullInput.text.tokens[0..., cacheOffset...]
                }
                // Drop the mask — for our HTTP path the input is always pure text
                // and downstream code recreates attention masks from cache offset.
                inputForGeneration = LMInput(text: LMInput.Text(tokens: slicedTokens, mask: nil))
                cacheToUse = lookupResult.restoreCache()
                skippedTokens = cacheOffset
                checkpointBaseOffset = cacheOffset
                // Only capture checkpoints in the SUFFIX (ones before snapshot already stored).
                checkpointPlan = checkpointPlan.filter { $0.offset > cacheOffset }
            } else {
                // MISS: full prefill.
                inputForGeneration = fullInput
                cacheToUse = nil
                skippedTokens = 0
                checkpointBaseOffset = 0
            }

            // 8. Set checkpoint offsets on parameters — flows into TokenIterator → prepare().
            // Planner guarantees offset uniqueness, so uniqueKeysWithValues traps loudly
            // on a planner-side invariant break instead of silently dropping a candidate.
            var genParams = parameters
            genParams.checkpoints = Dictionary(
                uniqueKeysWithValues: checkpointPlan.map { ($0.offset, $0.type) }
            )
            genParams.checkpointBaseOffset = checkpointBaseOffset

            // 9. Create TokenIterator — this calls model.prepare() internally with checkpoints.
            // NO separate prepare() call. TokenIterator owns prefill.
            let iterator = try TokenIterator(
                input: inputForGeneration,
                model: context.model,
                cache: cacheToUse,
                parameters: genParams
            )

            // 10. Read captured snapshots (populated by prepare() inside TokenIterator.init).
            let capturedSnapshots = iterator.capturedSnapshots

            // 11. Start generation stream.
            let (stream, task, finalCacheHandle) = MLXLMCommon.generateTaskWithFinalCache(
                promptTokenCount: fullTokenCount,
                modelConfiguration: context.configuration,
                tokenizer: context.tokenizer,
                iterator: iterator
            )

            Log.agent.info(
                "Prefix cache \(lookupResult.reason) — model=\(modelID) "
                + "promptTokens=\(fullTokenCount) "
                + "skippedPrefillTokens=\(skippedTokens) "
                + "newTokensToPrefill=\(fullTokenCount - skippedTokens) "
                + "sharedPrefixLength=\(lookupResult.sharedPrefixLength) "
                + "requestMessages=\(conversation.messages.count) "
                + "capturedCheckpoints=\(capturedSnapshots.count)"
            )

            return HTTPPrefixCacheGeneration(
                stream: stream,
                completion: task,
                finalCacheHandle: finalCacheHandle,
                promptTokenCount: fullTokenCount,
                skippedPrefillTokens: skippedTokens,
                fullTokens: fullTokens,
                capturedSnapshots: capturedSnapshots,
                partitionKey: partitionKey
            )
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

    /// Re-tokenize the stored conversation (prompt + generated response) and return
    /// the flat token sequence. Used for storing the leaf snapshot under the correct
    /// radix path. Returns `nil` on tokenization failure.
    private static func measureStoredTokenSequence(
        container: ModelContainer,
        conversation: HTTPPrefixCacheConversation,
        toolSpecs: [ToolSpec]?
    ) async -> [Int]? {
        do {
            return try await container.perform { context in
                let prepared = try await context.processor.prepare(
                    input: UserInput(chat: conversation.historyMessages, tools: toolSpecs)
                )
                return Self.extractTokenSequence(prepared.text.tokens)
            }
        } catch {
            Log.agent.warning(
                "Stored token sequence measurement failed — error=\(error.localizedDescription)"
            )
            return nil
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
