import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXNN
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
    let diagnosticsContext: PrefixCacheDiagnostics.Context
    let lookupMs: TimeInterval
    let restoreMs: TimeInterval
    let prefillMs: TimeInterval
    /// Total prompt tokens (full conversation, ignoring slicing).
    let promptTokenCount: Int
    /// Number of leading tokens skipped because the cache already covered them.
    let skippedPrefillTokens: Int

    // -- Post-generation store context (radix tree flow) --

    /// Flat token sequence for the full prompt (1D extraction from potentially 2D VLM tensor).
    let fullTokens: [Int]
    /// Snapshots captured during prefill at checkpoint offsets (e.g. stable-prefix boundary).
    let capturedSnapshots: [HybridCacheSnapshot]
    /// Pre-extracted `SnapshotPayload` values for each entry in
    /// ``capturedSnapshots``, produced inside the
    /// ``ModelContainer/perform(_:)`` block that captured the snapshots
    /// so that `MLXArray.asData()` runs on the Metal-affine inference
    /// thread. Positionally aligned with ``capturedSnapshots``. Empty
    /// when ``ssdEnabled`` is false; callers must tolerate the
    /// zero-length case without crashing.
    let capturedPayloads: [SnapshotPayload]
    /// SSD persistence tier gate, sampled once on `LLMActor`'s own
    /// isolation at `makeHTTPPrefixCacheGeneration` entry. Downstream
    /// post-generation sites (unstripped leaf + stripped leaf) read
    /// this through the captured `mlxStart` instead of re-sampling
    /// `self.ssdConfig`, which they cannot do without crossing the
    /// Metal-affine scope boundary.
    let ssdEnabled: Bool
    /// Partition key used for cache routing.
    let partitionKey: CachePartitionKey
    /// Offset where the final history message ends, right before the
    /// assistant generation prompt. `nil` when the generation prompt
    /// string couldn't be resolved (non-Qwen3.5 templates). Used by the
    /// stripped-leaf capture path to restore a fresh cache at a
    /// known-stable boundary and re-prefill the stripped assistant turn.
    let lastMessageBoundaryOffset: Int?
    /// Chunked prefill step size from the request's `GenerateParameters`,
    /// plumbed out so the post-generation stripped-leaf path can use the
    /// same chunk size when re-prefilling the stripped assistant residual
    /// on top of the restored last-message-boundary snapshot.
    let prefillStepSize: Int
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
    /// The HTTP prefix cache is auto-sized from the remaining unified memory
    /// after subtracting model weights and a fixed 20 GiB safety headroom:
    /// `max(0, (physicalMemory - modelWeightBytes - 20 GiB) / 2)`.
    ///
    /// The 20 GiB headroom accounts for: peak active MLX working set during
    /// a large-chunk prefill (~12–14 GiB above the cache budget on Qwen3.5
    /// 9B), macOS kernel + WindowServer + other apps (~6–8 GiB), and
    /// enough slack to avoid memory compression / swap on a typical dev
    /// machine running Xcode + browser alongside Tesseract. An earlier
    /// 4 GiB headroom produced a 19 GiB cache budget on a 48 GiB Mac,
    /// which pushed peak MLX usage to 36 GiB and triggered 14 GiB of swap.
    ///
    /// Example: on a 48 GiB machine with a 4.8 GiB model, the default cache
    /// budget becomes ~11.6 GiB. Before a model is sized (or after unload),
    /// the actor falls back to a conservative 3 GiB budget so pre-load paths
    /// and tests retain deterministic behavior. Machines too small to fit
    /// model + headroom clamp to 0 and rely on the fallback.
    private enum Defaults {
        static let cacheLimitMB = 2048
        static let prefixCacheHeadroomBytes = 20 * 1024 * 1024 * 1024 // 20 GiB
        /// Fallback budget used before load-time sizing runs. Each snapshot
        /// costs ~200–600 MiB depending on context length, so 3 GiB fits
        /// ~5–15 snapshots for typical Qwen3.5 workloads.
        static let fallbackPrefixCacheMemoryBudgetBytes = 3 * 1024 * 1024 * 1024 // 3 GiB
    }

    private var modelContainer: ModelContainer?
    private(set) var agentTokenizer: AgentTokenizer?
    private var _prefixCache: PrefixCacheManager?
    private var promptStartsThinking = false
    private var modelWeightBytes: Int64 = 0
    private var defaultPrefixCacheMemoryBudgetBytes =
        Defaults.fallbackPrefixCacheMemoryBudgetBytes

    /// Stable SHA-256 of the loaded model's weight files. Folded into every
    /// `CachePartitionKey` so a weight swap under the same `modelID`
    /// cannot surface stale persisted snapshots. `nil` before load and
    /// after `unloadModel()`.
    private var modelFingerprint: String?

    /// Snapshot of the SSD prefix-cache config captured at load time.
    /// Actor-isolated and synchronously readable from inside
    /// `container.perform`, which cannot await MainActor.
    private var ssdConfig: SSDPrefixCacheConfig?

    var isLoaded: Bool { modelContainer != nil }

    /// Internal read-only accessor for the load-time SSD config snapshot.
    /// Production reads happen via the synchronous capture in
    /// `makeHTTPPrefixCacheGeneration`; this accessor exists so tests
    /// can assert the load/unload lifecycle across the actor boundary.
    var currentSSDConfigForTesting: SSDPrefixCacheConfig? { ssdConfig }

    /// Internal read-only accessor for the load-time model fingerprint.
    var currentModelFingerprintForTesting: String? { modelFingerprint }

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
    ///   - ssdConfig: Snapshot of the SSD prefix-cache config, normally
    ///     produced by `SettingsManager.makeSSDPrefixCacheConfig()`. `nil`
    ///     disables SSD for the lifetime of this load.
    /// - Returns: The resolved ``AgentTokenizer`` and whether the template starts inside a think block.
    @discardableResult
    func loadModel(
        from directory: URL,
        visionMode: Bool,
        ssdConfig: SSDPrefixCacheConfig? = nil
    ) async throws -> (AgentTokenizer, promptStartsThinking: Bool) {
        let format = Self.detectToolCallFormat(directory: directory)
        Log.agent.info(
            "Loading model — visionMode=\(visionMode) "
            + "format=\(format.map { "\($0)" } ?? "json (default)")"
        )

        if let ssdConfig {
            Log.agent.info(
                "prefix-cache ssd enabled=\(ssdConfig.enabled) "
                + "budget=\(ssdConfig.budgetBytes) "
                + "root=\(ssdConfig.rootURL.path) "
                + "maxPendingBytes=\(ssdConfig.maxPendingBytes)"
            )
        } else {
            Log.agent.info("prefix-cache ssd enabled=false")
        }

        // Install SSD plumbing + weight fingerprint before attempting the
        // container load. A failed load leaves these fields populated
        // (harmless because nothing reads them without a loaded
        // container) and `unloadModel` is still responsible for
        // clearing them. Installing up-front keeps the full plumbing
        // chain — `resolveSSDConfig` → here — exercisable via a
        // real-path unit test even when no MLX container can be loaded.
        let fingerprint = try ModelFingerprint.computeFingerprint(modelDir: directory)
        installLoadTimeSSDState(fingerprint: fingerprint, ssdConfig: ssdConfig)

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
        sessionAffinity: String?,
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
        let requestID = UUID()
        // Canonicalize tools once so the leaf re-tokenization uses the same dict
        // iteration order as the prefill path inside makeHTTPPrefixCacheGeneration.
        let canonicalTools = Self.canonicalizeToolSpecs(toolSpecs)
        let mlxStart = try await makeHTTPPrefixCacheGeneration(
            container: container,
            conversation: conversation,
            requestID: requestID,
            modelID: modelID,
            sessionAffinity: sessionAffinity,
            parameters: Self.makeGenerateParameters(from: parameters),
            toolSpecs: canonicalTools,
            prefixCache: prefixCache
        )

        let mlxStartBox = UnsafeSendableBox(mlxStart)
        let (stream, continuation) = AsyncThrowingStream<AgentGeneration, Error>.makeStream()
        let startsInsideThinkBlock = promptStartsThinking
        let loadedModelWeightBytes = modelWeightBytes

        let task = Task { [conversation, container, canonicalTools, requestID, loadedModelWeightBytes] in
            let mlxStart = mlxStartBox.value
            let diagnosticsContext = mlxStart.diagnosticsContext
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

                func logEvictions(_ evictions: [PrefixCacheManager.EvictionEvent]) {
                    for event in evictions {
                        diagnosticsContext.log(PrefixCacheDiagnostics.EvictionEvent(event))
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
                    diagnosticsContext.log(PrefixCacheDiagnostics.TTFTEvent(
                        lookupMs: mlxStart.lookupMs,
                        restoreMs: mlxStart.restoreMs,
                        prefillMs: mlxStart.prefillMs,
                        totalPromptMs: completionInfo.promptTime
                    ))
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
                var storedSnapshotsForTuner: [HybridCacheSnapshot] = []
                if !Task.isCancelled, !mlxStart.capturedSnapshots.isEmpty {
                    let diagnostics = await MainActor.run {
                        prefixCache.storeSnapshots(
                            promptTokens: mlxStart.fullTokens,
                            capturedSnapshots: mlxStart.capturedSnapshots,
                            snapshotPayloads: mlxStart.capturedPayloads,
                            partitionKey: mlxStart.partitionKey,
                            requestID: requestID
                        )
                    }
                    logEvictions(diagnostics.evictions)
                    storedSnapshotsForTuner = mlxStart.capturedSnapshots
                }

                // Leaf store, wrapped so any skip path falls through to
                // the request-end recordRequest call below — the alpha
                // tuner needs to see every request, not just the ones
                // whose leaf store completed.
                //
                // Two leaves are captured per turn:
                //
                //  - **Unstripped leaf** (this block): represents the
                //    full conversation state with the just-generated
                //    assistant. Reachable for tool-loop continuations
                //    where the assistant remains the latest message in
                //    the next request.
                //
                //  - **Stripped leaf** (`captureStrippedLeaf` below):
                //    represents the conversation as Turn N+1 will see
                //    it after a new user message arrives — Qwen3.5's
                //    template strips the assistant's `<think>...</think>`
                //    block when it is no longer the latest assistant.
                //    Reachable for cross-new-user-turn lookups.
                //
                // Both leaves coexist in the radix tree on different
                // paths and serve different lookup shapes. Eviction
                // treats them uniformly under the LRU-with-FLOPs policy.
                var leafStoreForTuner: AlphaTuner.LeafStore? = nil
                // Hoisted out of `leafBlock` so the stripped-leaf capture
                // path below can reuse them.
                var storedConversationCaptured: HTTPPrefixCacheConversation? = nil
                var storedTokensCaptured: [Int]? = nil
                leafBlock: do {
                    guard !Task.isCancelled,
                          let finalCache = await mlxStart.finalCacheHandle.takeFinalCache()
                    else {
                        if !Task.isCancelled {
                            diagnosticsContext.logSkip(
                                stage: "store",
                                reason: "no-final-cache",
                                level: .warning
                            )
                        }
                        break leafBlock
                    }

                    let cacheOffsets = httpPrefixCacheOffsets(finalCache)
                    guard httpPrefixCacheHasReusableState(finalCache) else {
                        diagnosticsContext.logSkip(
                            stage: "store",
                            reason: "no-reusable-cache-state",
                            extraFields: [("cacheOffsets", "\(cacheOffsets)")]
                        )
                        break leafBlock
                    }

                    // 1. Build stored conversation (prompt + generated assistant turn).
                    let storedConversation = conversation.appendingAssistant(.assistant(
                        content: textContent,
                        reasoning: thinkingContent,
                        toolCalls: toolCalls
                    ))
                    storedConversationCaptured = storedConversation

                    // 3. Re-tokenize stored conversation → flat token sequence.
                    guard let storedTokens = await Self.measureStoredTokenSequence(
                        container: container,
                        conversation: storedConversation,
                        toolSpecs: canonicalTools
                    ) else {
                        diagnosticsContext.logSkip(
                            stage: "leafStore",
                            reason: "tokenization-failed",
                            level: .warning
                        )
                        break leafBlock
                    }
                    storedTokensCaptured = storedTokens

                    // 4. Offset-alignment guard: if normalization shortened the
                    //    stored conversation (whitespace-only assistant content → ""),
                    //    we can only trim attention K/V — Mamba's recurrent state
                    //    can't be unwound. Trimming the cache and capturing it as a
                    //    leaf produces a snapshot whose attention is aligned to
                    //    `storedTokens.count` but whose Mamba state is from the
                    //    full pre-trim offset. On Qwen3.5 the resulting leaf hit
                    //    perturbs raw logits by ~10 even at trim=1: argmax stays
                    //    stable (greedy decoding survives), but the rest of the
                    //    distribution drifts in a way that affects sampled
                    //    decoding. Since the HTTP server propagates the request's
                    //    `temperature`/`top_p` and we can't predict future request
                    //    sampling params at store time, the safe choice is to
                    //    skip the leaf store entirely when normalization would
                    //    require any trim. Lost cache hits on whitespace-normalized
                    //    conversations are the trade-off for sampler-agnostic
                    //    correctness. Verified by `HybridCacheCorrectnessRunner`
                    //    test 9 — see the `leafHitWithNormalizationDivergence...`
                    //    diagnostics for the empirical drift measurements.
                    let actualCacheOffset = httpPrefixCacheReportedTokenCount(finalCache)
                    if actualCacheOffset > storedTokens.count {
                        let trimAmount = actualCacheOffset - storedTokens.count
                        diagnosticsContext.logSkip(
                            stage: "leafStore",
                            reason: "normalization-trim",
                            extraFields: [
                                ("trimAmount", "\(trimAmount)"),
                                ("offsetBefore", "\(actualCacheOffset)"),
                                ("canonicalCount", "\(storedTokens.count)"),
                            ]
                        )
                        break leafBlock
                    }

                    // 5. Capture leaf snapshot AND extract its payload
                    //    inside a Metal-affine `container.perform` so
                    //    the per-array `asData()` calls run on the
                    //    inference thread. `finalCache` is non-`Sendable`
                    //    `[any KVCache]` — routed through the vendor's
                    //    `nonSendable` perform overload. The offset
                    //    guard above ensures no per-layer trimming is
                    //    needed before capture.
                    let ssdEnabled = mlxStart.ssdEnabled
                    let (maybeLeaf, leafPayload): (HybridCacheSnapshot?, SnapshotPayload?) =
                        try await container.perform(
                            nonSendable: finalCache
                        ) { _, cache in
                            guard let snap = HybridCacheSnapshot.capture(
                                cache: cache,
                                offset: storedTokens.count,
                                type: .leaf
                            ) else {
                                return (nil, nil)
                            }
                            let payload = Self.extractSnapshotPayloads(
                                [snap],
                                ssdEnabled: ssdEnabled
                            ).first
                            return (snap, payload)
                        }
                    guard let leafSnapshot = maybeLeaf else {
                        diagnosticsContext.logSkip(
                            stage: "leafCapture",
                            reason: "unsupported-cache-type",
                            extraFields: [("cacheOffsets", "\(cacheOffsets)")]
                        )
                        break leafBlock
                    }
                    diagnosticsContext.log(PrefixCacheDiagnostics.CaptureEvent(
                        offset: leafSnapshot.tokenOffset,
                        checkpointType: leafSnapshot.checkpointType,
                        bytes: leafSnapshot.memoryBytes,
                        duringPrefill: false,
                        source: "leaf"
                    ))

                    // Coalesce storeLeaf + stats read in one MainActor
                    // hop — saves one cross-actor switch on the success
                    // path (the request hot path). Includes the post-store
                    // budget/total snapshot so the admission diagnostic can
                    // be logged from this actor without another hop.
                    let (diagnostics, postStoreBudgetBytes, postStoreSnapshotBytes) =
                        await MainActor.run { () -> (PrefixCacheManager.StoreDiagnostics, Int, Int) in
                            let d = prefixCache.storeLeaf(
                                storedTokens: storedTokens,
                                leafSnapshot: leafSnapshot,
                                leafPayload: leafPayload,
                                partitionKey: mlxStart.partitionKey,
                                requestID: requestID
                            )
                            return (d, prefixCache.memoryBudgetBytes, prefixCache.totalSnapshotBytes)
                        }
                    logEvictions(diagnostics.evictions)
                    let unstrippedAdmissionEvicted = diagnostics.evictions.contains { event in
                        event.offset == leafSnapshot.tokenOffset
                            && event.checkpointType == .leaf
                    }
                    if unstrippedAdmissionEvicted {
                        diagnosticsContext.logSkip(
                            stage: "leafAdmission",
                            reason: "capturedThenEvicted",
                            level: .warning,
                            extraFields: [
                                ("offset", "\(leafSnapshot.tokenOffset)"),
                                ("bytes", "\(leafSnapshot.memoryBytes)"),
                                ("budgetBytes", "\(postStoreBudgetBytes)"),
                                ("snapshotBytesAfter", "\(postStoreSnapshotBytes)"),
                            ]
                        )
                    } else {
                        leafStoreForTuner = AlphaTuner.LeafStore(
                            storedTokens: storedTokens,
                            bytes: leafSnapshot.memoryBytes
                        )
                    }

                    // Release the MLX free buffer pool back to the OS so it
                    // doesn't accumulate transient prefill intermediates
                    // across requests.
                    Memory.clearCache()
                }

                // Stripped-leaf capture: stores a second `.leaf` snapshot
                // keyed by the token sequence that Turn N+1 WILL see when
                // it re-renders the conversation (with `a_N`'s `<think>`
                // block stripped by the Qwen3.5 template's
                // `last_query_index` logic). The unstripped leaf above is
                // unreachable across new-user turns; the stripped leaf is
                // reachable and saves a full cold prefill on every
                // subsequent cross-new-user-turn lookup. Purely additive —
                // any failure is caught and logged without affecting the
                // rest of the request lifecycle.
                if !Task.isCancelled,
                   let storedConversation = storedConversationCaptured,
                   let storedTokens = storedTokensCaptured
                {
                    await Self.captureStrippedLeaf(
                        container: container,
                        storedConversation: storedConversation,
                        storedTokens: storedTokens,
                        toolSpecs: canonicalTools,
                        capturedSnapshots: mlxStart.capturedSnapshots,
                        lastMessageBoundaryOffset: mlxStart.lastMessageBoundaryOffset,
                        partitionKey: mlxStart.partitionKey,
                        prefillStepSize: mlxStart.prefillStepSize,
                        requestID: requestID,
                        prefixCache: prefixCache,
                        diagnosticsContext: diagnosticsContext,
                        promptStartsThinking: startsInsideThinkBlock,
                        thinkingContent: thinkingContent,
                        ssdEnabled: mlxStart.ssdEnabled
                    )
                }

                // Record the request lifecycle for the alpha tuner. Fires
                // for every request, including the leaf-skipped paths
                // — the tuner needs the full workload trace, not just
                // successful leaf stores.
                let capturedSnapshots = storedSnapshotsForTuner
                let leafCapture = leafStoreForTuner
                let (finalStats, finalBudgetBytes) = await MainActor.run {
                    prefixCache.recordRequest(
                        partitionKey: mlxStart.partitionKey,
                        promptTokens: mlxStart.fullTokens,
                        capturedSnapshots: capturedSnapshots,
                        leafStore: leafCapture,
                        requestID: requestID
                    )
                    return (prefixCache.stats, prefixCache.memoryBudgetBytes)
                }
                diagnosticsContext.log(PrefixCacheDiagnostics.MemoryEvent(
                    stats: finalStats,
                    budgetBytes: finalBudgetBytes,
                    modelWeightBytes: loadedModelWeightBytes,
                    activeMlxBytes: Int64(clamping: Memory.activeMemory),
                    peakMlxBytes: Int64(clamping: Memory.peakMemory),
                    mlxCacheLimitBytes: Int64(clamping: Memory.cacheLimit)
                ))

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
        modelWeightBytes = 0
        _prefixCache = nil
        defaultPrefixCacheMemoryBudgetBytes = Defaults.fallbackPrefixCacheMemoryBudgetBytes
        modelFingerprint = nil
        ssdConfig = nil
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
    /// The production cache attaches an `AlphaTuner` so eviction `alpha`
    /// adapts to the workload after the first eviction fires. Reset the
    /// global `EvictionPolicy.alpha` when creating a fresh cache so a
    /// previous cache's tuned value doesn't leak into the new tuner.
    private func ensurePrefixCache() async -> PrefixCacheManager {
        if let existing = _prefixCache { return existing }
        await MainActor.run { EvictionPolicy.alpha = 0.0 }
        let cache = await PrefixCacheManager(
            memoryBudgetBytes: defaultPrefixCacheMemoryBudgetBytes,
            alphaTuner: AlphaTuner()
        )
        _prefixCache = cache
        return cache
    }

    /// Snapshot of the live prefix-cache state, or `nil` if the cache hasn't
    /// been instantiated. Used by the loaded-model E2E runner to verify
    /// branch-point capture and survival.
    func prefixCacheStats() async -> PrefixCacheManager.CacheStats? {
        guard let cache = _prefixCache else { return nil }
        return await cache.stats
    }

    /// Override the prefix-cache memory budget at runtime. Used by the
    /// loaded-model E2E runner to deliberately trigger eviction pressure.
    func setPrefixCacheBudgetBytes(_ bytes: Int) async {
        let cache = await ensurePrefixCache()
        await MainActor.run {
            cache.memoryBudgetBytes = bytes
            cache.evictToFitBudget()
        }
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

    /// Pre-extract `HybridCacheSnapshot` instances into pure `Sendable`
    /// ``SnapshotPayload`` value types, calling `MLXArray.asData` on
    /// every per-layer state array. Result is positionally aligned
    /// with the input; empty when `ssdEnabled` is false.
    ///
    /// **Metal-affinity contract.** Must be called from inside
    /// ``ModelContainer/perform(_:)`` on `LLMActor` — calling it
    /// outside a live Metal-affine scope risks re-issuing command-queue
    /// work on a non-inference thread. The method is `nonisolated
    /// static` so callers can invoke it synchronously from inside a
    /// `container.perform` closure without an `await`; the Metal
    /// affinity is enforced by convention, not the type system. See
    /// `docs/marconi-hybrid-prefix-cache-implementation-plan.md`
    /// "Write path" for the three call sites and their per-site
    /// scoping requirements.
    nonisolated static func extractSnapshotPayloads(
        _ snapshots: [HybridCacheSnapshot],
        ssdEnabled: Bool
    ) -> [SnapshotPayload] {
        guard ssdEnabled, !snapshots.isEmpty else { return [] }

        var result: [SnapshotPayload] = []
        result.reserveCapacity(snapshots.count)

        for snapshot in snapshots {
            var layers: [SnapshotPayload.LayerPayload] = []
            layers.reserveCapacity(snapshot.layers.count)

            for layer in snapshot.layers {
                var arrays: [SnapshotPayload.ArrayPayload] = []
                arrays.reserveCapacity(layer.state.count)
                for array in layer.state {
                    let extracted = array.asData(access: .copy)
                    arrays.append(SnapshotPayload.ArrayPayload(
                        data: extracted.data,
                        dtype: dtypeWireString(extracted.dType),
                        shape: extracted.shape
                    ))
                }
                layers.append(SnapshotPayload.LayerPayload(
                    className: layer.className,
                    state: arrays,
                    metaState: layer.metaState,
                    offset: layer.offset
                ))
            }

            result.append(SnapshotPayload(
                tokenOffset: snapshot.tokenOffset,
                checkpointType: snapshot.checkpointType,
                layers: layers
            ))
        }

        return result
    }

    /// Stable wire-format name for an MLX `DType`. Load-bearing: the
    /// result is written into the SSD snapshot header at
    /// `SSDSnapshotStore.encodePlaceholderContainer(payload:descriptor:)`,
    /// so the mapping is part of the on-disk contract. A vendor-side
    /// rename of any `DType` case label would silently corrupt files
    /// without this explicit table.
    ///
    /// `@unknown default` traps via `fatalError` rather than inventing
    /// a placeholder string, because reaching it means the vendor
    /// shipped a new case that this table hasn't audited — inventing
    /// a wire name would persist an unreadable header under a claim of
    /// success. The remediation is always "add the case", not "paper
    /// over with a sentinel." Mirrors `DType.init(_ cmlxDtype:)` at
    /// `Vendor/.../mlx-swift/Source/MLX/DType.swift:61`, which uses
    /// the same loud-failure pattern for the C → Swift direction.
    nonisolated static func dtypeWireString(_ dtype: DType) -> String {
        switch dtype {
        case .bool: return "bool"
        case .uint8: return "uint8"
        case .uint16: return "uint16"
        case .uint32: return "uint32"
        case .uint64: return "uint64"
        case .int8: return "int8"
        case .int16: return "int16"
        case .int32: return "int32"
        case .int64: return "int64"
        case .float16: return "float16"
        case .float32: return "float32"
        case .bfloat16: return "bfloat16"
        case .complex64: return "complex64"
        case .float64: return "float64"
        @unknown default:
            fatalError(
                "dtypeWireString missing case for MLX DType \(dtype) — "
                + "extend the switch to preserve the SSD wire-format contract."
            )
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

    /// Run a closure with the loaded `ModelContainer`. Used by loaded-model
    /// runners (`PrefixCacheE2ERunner`, `HybridCacheCorrectnessRunner`) that
    /// need raw forward-pass access via `container.perform { context in ... }`
    /// without going through the agent generation pipeline.
    func withModelContainer<T: Sendable>(
        _ body: @Sendable (ModelContainer) async throws -> T
    ) async throws -> T {
        guard let container = modelContainer else {
            throw AgentEngineError.modelNotLoaded
        }
        return try await body(container)
    }

    // MARK: - Private

    /// Verifies the model with a 1-token generation, stores it, and returns the tokenizer.
    private func verifyAndStore(
        container: ModelContainer,
        directory: URL
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
        let profile = Self.detectModelFlopProfile(directory: directory) ?? .qwen35_4B_PARO
        let modelWeightBytes = await container.perform { context in
            context.model.parameters().flattened().reduce(into: Int64.zero) { partial, item in
                let nbytes = Int64(clamping: item.1.nbytes)
                if partial > Int64.max - nbytes {
                    partial = Int64.max
                } else {
                    partial += nbytes
                }
            }
        }
        let totalMemoryBytes = ProcessInfo.processInfo.physicalMemory
        let prefixCacheBudgetBytes = Self.autoSizedPrefixCacheMemoryBudgetBytes(
            totalMemoryBytes: totalMemoryBytes,
            modelMemoryBytes: modelWeightBytes
        )
        await MainActor.run { EvictionPolicy.modelProfile = profile }
        Log.agent.info(
            "EvictionPolicy.modelProfile — D=\(profile.hiddenSize) "
            + "attn=\(profile.attentionLayers) ssm=\(profile.ssmLayers) "
            + "mlp=\(profile.mlpLayers) N=\(profile.ssmStateDim)"
        )
        Log.agent.info(
            "Prefix cache budget auto-sized — "
            + "totalRAMBytes=\(totalMemoryBytes) "
            + "modelWeightBytes=\(modelWeightBytes) "
            + "headroomBytes=\(Defaults.prefixCacheHeadroomBytes) "
            + "budgetBytes=\(prefixCacheBudgetBytes)"
        )
        modelContainer = container
        agentTokenizer = tokenizer
        promptStartsThinking = startsThinking
        self.modelWeightBytes = modelWeightBytes
        defaultPrefixCacheMemoryBudgetBytes = prefixCacheBudgetBytes
        _prefixCache = nil
        return (tokenizer, startsThinking)
    }

    /// Single install site for per-load SSD plumbing state. Called from
    /// `loadModel` before the container load is attempted so the state
    /// is visible even on failed loads; this lets the unit suite exercise
    /// the full `AgentEngine.loadModel` → `resolveSSDConfig` → here
    /// chain via a fake directory that trips the container load. The
    /// unload path clears both fields.
    private func installLoadTimeSSDState(
        fingerprint: String,
        ssdConfig: SSDPrefixCacheConfig?
    ) {
        self.modelFingerprint = fingerprint
        self.ssdConfig = ssdConfig
    }

    static func autoSizedPrefixCacheMemoryBudgetBytes(
        totalMemoryBytes: UInt64,
        modelMemoryBytes: Int64
    ) -> Int {
        let total = Int64(clamping: totalMemoryBytes)
        let model = max(Int64.zero, modelMemoryBytes)
        let headroom = Int64(clamping: Defaults.prefixCacheHeadroomBytes)
        let reserved = model > Int64.max - headroom ? Int64.max : model + headroom
        let available = total - reserved
        guard available > 0 else { return 0 }
        return Int(clamping: available / 2)
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
        requestID: UUID,
        modelID: String,
        sessionAffinity: String?,
        parameters: GenerateParameters,
        toolSpecs: [ToolSpec]?,
        prefixCache: PrefixCacheManager
    ) async throws -> HTTPPrefixCacheGeneration {
        // Canonicalize tools once so the stable-prefix detector and the real
        // prefill tokenize against identical dict representations. Historically
        // swift-jinja <2.3.5 had non-deterministic `tojson` key ordering; the
        // canonicalization is kept as defense-in-depth and costs almost nothing.
        let canonicalTools = Self.canonicalizeToolSpecs(toolSpecs)

        // Capture actor state for the non-MainActor closure below —
        // the closure runs on `ModelContainer`'s isolation and cannot
        // sync-read `LLMActor` properties.
        let promptStartsThinking = self.promptStartsThinking
        let modelFingerprint = self.modelFingerprint
        let ssdEnabled = self.ssdConfig?.enabled == true
        let diagnosticsContext = PrefixCacheDiagnostics.Context(
            requestID: requestID,
            modelID: modelID,
            kvBits: parameters.kvBits,
            kvGroupSize: parameters.kvGroupSize
        )

        return try await container.perform { context in
            func measure<T>(_ work: () throws -> T) rethrows -> (T, TimeInterval) {
                let started = Date.timeIntervalSinceReferenceDate
                let value = try work()
                return (value, Date.timeIntervalSinceReferenceDate - started)
            }

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
            //    sessionAffinity isolates main agent from subagents — without
            //    it, a long-running subagent's churn evicts the idle main
            //    agent's snapshots under shared budget pressure. See
            //    `CachePartitionKey` for the full rationale.
            let partitionKey = CachePartitionKey(
                modelID: modelID,
                kvBits: parameters.kvBits,
                kvGroupSize: parameters.kvGroupSize,
                sessionAffinity: sessionAffinity,
                modelFingerprint: modelFingerprint
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
            let lookupStarted = Date.timeIntervalSinceReferenceDate
            let (lookupResult, initialCheckpointPlan) = await MainActor.run {
                prefixCache.lookupAndPlanCheckpoints(
                    tokens: fullTokens,
                    stablePrefixOffset: stablePrefixOffset,
                    lastMessageBoundaryOffset: lastMessageBoundaryOffset,
                    partitionKey: partitionKey
                )
            }
            let lookupMs = Date.timeIntervalSinceReferenceDate - lookupStarted
            var checkpointPlan = initialCheckpointPlan

            // 7. Determine input for generation and cache to restore.
            let inputForGeneration: LMInput
            let cacheToUse: [any KVCache]?
            let skippedTokens: Int
            let checkpointBaseOffset: Int
            let restoreMs: TimeInterval

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
                let (restoredCache, measuredRestoreMs) = measure {
                    lookupResult.restoreCache()
                }
                cacheToUse = restoredCache
                restoreMs = measuredRestoreMs
                skippedTokens = cacheOffset
                checkpointBaseOffset = cacheOffset
                // Only capture checkpoints in the SUFFIX (ones before snapshot already stored).
                checkpointPlan = checkpointPlan.filter { $0.offset > cacheOffset }
            } else {
                // MISS: full prefill.
                inputForGeneration = fullInput
                cacheToUse = nil
                restoreMs = 0
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
            let (iterator, prefillMs) = try measure {
                try TokenIterator(
                    input: inputForGeneration,
                    model: context.model,
                    cache: cacheToUse,
                    parameters: genParams
                )
            }

            // 10. Read captured snapshots (populated by prepare() inside TokenIterator.init),
            // then extract their payloads inside this `container.perform` so
            // `MLXArray.asData()` runs on the Metal-affine thread before the
            // later MainActor store hop.
            let capturedSnapshots = iterator.capturedSnapshots
            let capturedPayloads = Self.extractSnapshotPayloads(
                capturedSnapshots,
                ssdEnabled: ssdEnabled
            )
            diagnosticsContext.log(PrefixCacheDiagnostics.LookupEvent(
                reason: lookupResult.reason,
                promptTokens: fullTokenCount,
                sharedPrefixLength: lookupResult.sharedPrefixLength,
                skippedPrefillTokens: skippedTokens,
                newTokensToPrefill: fullTokenCount - skippedTokens,
                lookupMs: lookupMs,
                restoreMs: restoreMs,
                plannedCheckpoints: checkpointPlan
            ))
            for snapshot in capturedSnapshots {
                diagnosticsContext.log(PrefixCacheDiagnostics.CaptureEvent(
                    offset: snapshot.tokenOffset,
                    checkpointType: snapshot.checkpointType,
                    bytes: snapshot.memoryBytes,
                    duringPrefill: true,
                    source: "prefill"
                ))
            }

            // 11. Start generation stream.
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
                diagnosticsContext: diagnosticsContext,
                lookupMs: lookupMs,
                restoreMs: restoreMs,
                prefillMs: prefillMs,
                promptTokenCount: fullTokenCount,
                skippedPrefillTokens: skippedTokens,
                fullTokens: fullTokens,
                capturedSnapshots: capturedSnapshots,
                capturedPayloads: capturedPayloads,
                ssdEnabled: ssdEnabled,
                partitionKey: partitionKey,
                lastMessageBoundaryOffset: lastMessageBoundaryOffset,
                prefillStepSize: parameters.prefillStepSize
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

    /// Return the stripped stored-token sequence `[sys, ..., u_N, a_N_stripped]`
    /// — the token path Turn N+1 will see when it re-renders the conversation
    /// with a new user message, triggering the Qwen3.5 template's `last_query_index`
    /// logic to strip `a_N`'s `<think>...</think>` block.
    ///
    /// Algorithm: tokenize two probe conversations that share the same
    /// `[sys, ..., u_N, a_N]` prefix and differ only in an appended dummy
    /// user message. Both tokenizations run the assistant through the template
    /// as non-latest, so its think block is stripped. The two token sequences
    /// share a common prefix up to the first token of the dummy user's
    /// CONTENT (the `<|im_start|>user\n` opener is identical). Subtract the
    /// opener length from the divergence index to land on the end of
    /// `a_N_stripped`.
    ///
    /// Returns `nil` on any unexpected condition (probes identical, opener
    /// mismatch, etc.) so the caller can skip the stripped-leaf store cleanly.
    private static func computeStrippedStoredTokens(
        context: ModelContext,
        storedConversation: HTTPPrefixCacheConversation,
        toolSpecs: [ToolSpec]?
    ) async throws -> [Int]? {
        let baseHistory = storedConversation.historyMessages

        // Build probe histories with differing dummy user content. Content
        // is chosen to start with different tokens immediately (single
        // alphabetic character + marker), so the divergence point lands
        // right at the first content token of the dummy user.
        let probeAInput = UserInput(
            chat: baseHistory + [.user("Aqkz_strip_probe")],
            tools: toolSpecs
        )
        let probeBInput = UserInput(
            chat: baseHistory + [.user("Zqkz_strip_probe")],
            tools: toolSpecs
        )
        let probeA = try await context.processor.prepare(input: probeAInput)
        let probeB = try await context.processor.prepare(input: probeBInput)
        let tokensA = Self.extractTokenSequence(probeA.text.tokens)
        let tokensB = Self.extractTokenSequence(probeB.text.tokens)

        // First divergence between the two probes.
        var divergence = 0
        let shorter = min(tokensA.count, tokensB.count)
        while divergence < shorter && tokensA[divergence] == tokensB[divergence] {
            divergence += 1
        }
        // If the two probes match entirely, something unexpected happened
        // (template ignored user content?) — bail out so we don't produce
        // bogus stripped tokens.
        guard divergence > 0, divergence < tokensA.count else { return nil }

        // Subtract the `<|im_start|>user\n` opener (fixed, template-emitted)
        // to get the offset at the end of `a_N_stripped`.
        let userOpenerTokens = context.tokenizer.encode(
            text: "<|im_start|>user\n", addSpecialTokens: false
        )
        guard userOpenerTokens.count > 0,
              divergence >= userOpenerTokens.count
        else { return nil }

        // Defense in depth: verify the opener actually precedes the
        // divergence. If the template injected something else between
        // `a_N_stripped` and the dummy user (chat history wrappers,
        // additional role markers), bail out rather than producing
        // misaligned stripped tokens.
        let openerSlice = Array(tokensA[(divergence - userOpenerTokens.count)..<divergence])
        guard openerSlice.elementsEqual(userOpenerTokens) else { return nil }

        let strippedLen = divergence - userOpenerTokens.count
        guard strippedLen > 0, strippedLen <= tokensA.count else { return nil }
        return Array(tokensA[0..<strippedLen])
    }

    /// Capture a second, "stripped" leaf snapshot for the current turn's
    /// conversation. Addresses the Qwen3.5 think-stripping cross-new-user-turn
    /// divergence: the template strips `<think>...</think>` from assistant
    /// messages that are not the latest assistant, so Turn N's unstripped
    /// leaf is unreachable from Turn N+1's lookup once a new user message
    /// arrives. The stripped leaf stores a snapshot keyed by the token
    /// sequence Turn N+1 WILL actually see, allowing a deep cache hit
    /// instead of a full cold prefill.
    ///
    /// Correctness: the stripped leaf's cache state is the result of
    /// restoring the `lastMessageBoundary` snapshot (captured mid-prefill
    /// during this turn, byte-exact representation of the cache at
    /// end-of-last-user) and running a real residual prefill over
    /// `[assistant_opener + response_content + <|im_end|>]`. No state
    /// trimming, no Mamba divergence — bit-exact continuation of the
    /// same forward pass.
    ///
    /// Additive: runs AFTER the primary (unstripped) leaf store, and any
    /// failure is caught and logged without affecting the rest of the
    /// request lifecycle. The unstripped leaf is still the authoritative
    /// reuse target for tool-loop continuation (where the assistant is
    /// still the latest message in the next request's history).
    ///
    /// Skip conditions (logged via `logSkip` unless explicitly silent):
    /// - `!promptStartsThinking`: non-thinking model, nothing to strip.
    ///   Silent — the unstripped leaf path covers this case correctly.
    /// - `thinkingContent < 50 chars`: prefill cost > expected savings.
    ///   Silent.
    /// - `lastMessageBoundaryOffset == nil`: can't identify boundary
    ///   snapshot to restore from. Logged.
    /// - No `.lastMessageBoundary` snapshot at the boundary offset in
    ///   `capturedSnapshots`: prior turn already stored the boundary
    ///   (planner deduped it), so its stripped leaf still exists. Silent.
    /// - Two-probe tokenization failed or probes identical: tokenizer
    ///   edge case. Logged.
    /// - `strippedLen <= boundaryOffset`: no residual to prefill. Logged.
    /// - `strippedLen >= storedTokens.count`: template didn't actually
    ///   strip anything (possibly non-Qwen3.5 template). Logged.
    /// - Residual prefill throws. Logged.
    /// - Snapshot capture returns nil (unsupported cache types). Logged.
    /// - The freshly stored leaf was immediately evicted by its own
    ///   `evictToFitBudget` cycle (the "capturedThenEvicted" pathology).
    ///   Logged as a warning so the issue is visible in production logs.
    private static func captureStrippedLeaf(
        container: ModelContainer,
        storedConversation: HTTPPrefixCacheConversation,
        storedTokens: [Int],
        toolSpecs: [ToolSpec]?,
        capturedSnapshots: [HybridCacheSnapshot],
        lastMessageBoundaryOffset: Int?,
        partitionKey: CachePartitionKey,
        prefillStepSize: Int,
        requestID: UUID,
        prefixCache: PrefixCacheManager,
        diagnosticsContext: PrefixCacheDiagnostics.Context,
        promptStartsThinking: Bool,
        thinkingContent: String,
        ssdEnabled: Bool
    ) async {
        // Guard 1: non-thinking model → nothing to strip.
        guard promptStartsThinking else {
            return  // silent; not an anomaly
        }

        // Guard 2: trivial think content → prefill cost would exceed savings.
        let trimmedThink = thinkingContent.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmedThink.count >= 50 else {
            return  // silent; not worth the prefill
        }

        // Guard 3: no boundary offset → can't identify where to restore from.
        guard let boundaryOffset = lastMessageBoundaryOffset else {
            diagnosticsContext.logSkip(
                stage: "strippedLeafStore",
                reason: "no-boundary-offset"
            )
            return
        }

        // Guard 4: boundary snapshot must be in capturedSnapshots.
        // When the planner skipped the boundary checkpoint because a
        // prior turn already stored one at the same offset (deduped),
        // capturedSnapshots won't contain it — in that case the prior
        // turn's stripped leaf already exists and we silently skip.
        guard let boundarySnapshot = capturedSnapshots.first(
            where: { $0.tokenOffset == boundaryOffset }
        ) else {
            return  // silent; prior turn handled it
        }

        do {
            try await container.perform { context in
                // Tokenize the stripped stored sequence via two-probe.
                guard let storedTokensStripped = try await Self.computeStrippedStoredTokens(
                    context: context,
                    storedConversation: storedConversation,
                    toolSpecs: toolSpecs
                ) else {
                    diagnosticsContext.logSkip(
                        stage: "strippedLeafStore",
                        reason: "probe-divergence-failed"
                    )
                    return
                }

                // Guard 5: no residual to prefill.
                guard storedTokensStripped.count > boundaryOffset else {
                    diagnosticsContext.logSkip(
                        stage: "strippedLeafStore",
                        reason: "no-residual",
                        extraFields: [
                            ("strippedLen", "\(storedTokensStripped.count)"),
                            ("boundaryOffset", "\(boundaryOffset)"),
                        ]
                    )
                    return
                }

                // Guard 6: template didn't actually strip anything (non-Qwen3.5?).
                // If `strippedLen == storedTokens.count`, the probe produced
                // the same length as the unstripped path — no think block was
                // removed. Skip to avoid storing a duplicate leaf.
                guard storedTokensStripped.count < storedTokens.count else {
                    diagnosticsContext.logSkip(
                        stage: "strippedLeafStore",
                        reason: "no-stripping-applied",
                        level: .warning,
                        extraFields: [
                            ("strippedLen", "\(storedTokensStripped.count)"),
                            ("storedLen", "\(storedTokens.count)"),
                        ]
                    )
                    return
                }

                // Restore the boundary snapshot into a fresh [KVCache].
                let restoredCache = boundarySnapshot.restore(
                    kvBitsHint: partitionKey.kvBits,
                    kvGroupSizeHint: partitionKey.kvGroupSize
                )

                // Compute the residual tokens: `[assistant_opener + response + <|im_end|>]`.
                let residual = Array(storedTokensStripped[boundaryOffset...])

                // Prefill the residual on top of the restored cache.
                // Mirrors `HybridCacheCorrectnessRunner.prefill` — the
                // production prefill loop, with leftover drain.
                let prefillStart = Date.timeIntervalSinceReferenceDate
                let inputArr = MLXArray(residual.map { Int32($0) })
                let lmInput = LMInput(text: .init(tokens: inputArr, mask: nil))
                let (prepareResult, _) = try context.model.prepareWithCheckpoints(
                    lmInput,
                    cache: restoredCache,
                    windowSize: prefillStepSize,
                    checkpoints: [:],
                    checkpointBaseOffset: boundaryOffset
                )
                if case .tokens(let leftover) = prepareResult, leftover.tokens.size > 0 {
                    let batched = LMInput.Text(
                        tokens: leftover.tokens.expandedDimensions(axis: 0),
                        mask: leftover.mask
                    )
                    _ = context.model(batched, cache: restoredCache, state: nil)
                    eval(restoredCache)
                }
                let prefillMs = Date.timeIntervalSinceReferenceDate - prefillStart

                // Capture the stripped leaf from the freshly prefilled cache.
                guard let strippedLeaf = HybridCacheSnapshot.capture(
                    cache: restoredCache,
                    offset: storedTokensStripped.count,
                    type: .leaf
                ) else {
                    diagnosticsContext.logSkip(
                        stage: "strippedLeafCapture",
                        reason: "unsupported-cache-type"
                    )
                    return
                }

                // Extract the stripped leaf payload inside the enclosing
                // `container.perform` so `asData()` runs on the Metal-affine
                // thread that just prefilled the residual.
                let strippedLeafPayload = Self.extractSnapshotPayloads(
                    [strippedLeaf],
                    ssdEnabled: ssdEnabled
                ).first

                diagnosticsContext.log(PrefixCacheDiagnostics.CaptureEvent(
                    offset: strippedLeaf.tokenOffset,
                    checkpointType: strippedLeaf.checkpointType,
                    bytes: strippedLeaf.memoryBytes,
                    duringPrefill: false,
                    source: "strippedLeaf"
                ))
                Log.agent.info(
                    "Stripped leaf captured — offset=\(strippedLeaf.tokenOffset) "
                    + "residualTokens=\(residual.count) "
                    + "prefillMs=\(String(format: "%.3f", prefillMs * 1000)) "
                    + "unstrippedLen=\(storedTokens.count) "
                    + "strippedLen=\(storedTokensStripped.count)"
                )

                // Store the stripped leaf on MainActor and detect immediate
                // eviction (the "capturedThenEvicted" pathology). Read the
                // post-store budget + total in the same hop.
                let (diagnostics, postStoreBudgetBytes, postStoreSnapshotBytes) =
                    await MainActor.run { () -> (PrefixCacheManager.StoreDiagnostics, Int, Int) in
                        let d = prefixCache.storeLeaf(
                            storedTokens: storedTokensStripped,
                            leafSnapshot: strippedLeaf,
                            leafPayload: strippedLeafPayload,
                            partitionKey: partitionKey,
                            requestID: requestID
                        )
                        return (d, prefixCache.memoryBudgetBytes, prefixCache.totalSnapshotBytes)
                    }
                for event in diagnostics.evictions {
                    diagnosticsContext.log(PrefixCacheDiagnostics.EvictionEvent(event))
                }
                let admissionEvicted = diagnostics.evictions.contains { event in
                    event.offset == strippedLeaf.tokenOffset
                        && event.checkpointType == .leaf
                }
                if admissionEvicted {
                    diagnosticsContext.logSkip(
                        stage: "strippedLeafAdmission",
                        reason: "capturedThenEvicted",
                        level: .warning,
                        extraFields: [
                            ("offset", "\(strippedLeaf.tokenOffset)"),
                            ("bytes", "\(strippedLeaf.memoryBytes)"),
                            ("budgetBytes", "\(postStoreBudgetBytes)"),
                            ("snapshotBytesAfter", "\(postStoreSnapshotBytes)"),
                        ]
                    )
                }

                Memory.clearCache()
            }
        } catch {
            diagnosticsContext.logSkip(
                stage: "strippedLeafStore",
                reason: "prefill-threw",
                level: .warning,
                extraFields: [("error", error.localizedDescription)]
            )
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

    /// Parse `config.json` from the model directory into a top-level dict.
    /// Returns `nil` if the file is missing or unparseable.
    private static func loadConfigJSON(directory: URL) -> [String: Any]? {
        let configURL = directory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        return json
    }

    /// Detects the chat-template tool-call format from the model's
    /// `config.json`. Qwen3.5 uses XML function syntax
    /// (`<function=name>...</function>`) inside `<tool_call>` tags, which
    /// requires `.xmlFunction`.
    private static func detectToolCallFormat(directory: URL) -> ToolCallFormat? {
        guard let json = loadConfigJSON(directory: directory),
              let modelType = json["model_type"] as? String
        else { return nil }

        if modelType.hasPrefix("qwen3_5") {
            return .xmlFunction
        }
        return ToolCallFormat.infer(from: modelType)
    }

    /// Build a `ModelFlopProfile` for a Qwen3.5 hybrid model by reading its
    /// `config.json`. Both LLM and VLM Qwen3.5 variants nest architecture
    /// fields under `text_config` (the top-level `model_type` is `qwen3_5`,
    /// the nested one is `qwen3_5_text`). Returns `nil` for non-Qwen3.5
    /// models, missing fields, or malformed configs — caller should fall
    /// back to `.qwen35_4B_PARO`.
    static func detectModelFlopProfile(directory: URL) -> ModelFlopProfile? {
        guard let root = loadConfigJSON(directory: directory),
              let topModelType = root["model_type"] as? String,
              topModelType.hasPrefix("qwen3_5")
        else { return nil }

        // VLM nests architecture fields under `text_config`; LLM-only puts
        // them at the top level.
        let textConfig = (root["text_config"] as? [String: Any]) ?? root
        guard let hiddenLayers = textConfig["num_hidden_layers"] as? Int,
              let hiddenSize = textConfig["hidden_size"] as? Int,
              let linearNumValueHeads = textConfig["linear_num_value_heads"] as? Int,
              let linearKeyHeadDim = textConfig["linear_key_head_dim"] as? Int,
              let fullAttentionInterval = textConfig["full_attention_interval"] as? Int
        else { return nil }

        return .qwen35(
            hiddenLayers: hiddenLayers,
            hiddenSize: hiddenSize,
            linearNumValueHeads: linearNumValueHeads,
            linearKeyHeadDim: linearKeyHeadDim,
            fullAttentionInterval: fullAttentionInterval
        )
    }
}
