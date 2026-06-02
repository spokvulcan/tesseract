import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
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
    /// Lookup outcome classification, surfaced for in-app observability.
    let lookupReason: PrefixCacheManager.LookupReason
    /// Shared-prefix length in tokens between the request and the best cache entry.
    let sharedPrefixLength: Int

    // -- Post-generation store context (radix tree flow) --

    /// Flat token sequence for the full prompt (1D extraction from potentially 2D VLM tensor).
    let fullTokens: [Int]
    /// Validated mid-prefill checkpoint admission, if any checkpoints survived
    /// extraction-edge path validation.
    let snapshotAdmission: SnapshotAdmission?
    /// SSD persistence tier gate, sampled once on `LLMActor`'s own
    /// isolation at `makeHTTPPrefixCacheGeneration` entry. Downstream
    /// post-generation sites (unstripped leaf + stripped leaf) read
    /// this through the captured `mlxStart` instead of re-sampling
    /// `self.ssdConfig`, which they cannot do without crossing the
    /// Metal-affine scope boundary.
    let ssdEnabled: Bool
    /// Partition key used for cache routing.
    let partitionKey: CachePartitionKey
    /// Request-local helper snapshot captured at the end of the last history
    /// message. Never stored or persisted; used to synthesize the direct
    /// tool-continuation leaf for tool-call turns.
    let transientLastMessageBoundarySnapshot: HybridCacheSnapshot?
    /// Request-local helper snapshot captured at the end of the last real
    /// user message. Never stored or persisted; used to synthesize the
    /// canonical user-continuation leaf for templates that rewrite the
    /// assistant/tool suffix after the last user.
    let transientLastUserBoundarySnapshot: HybridCacheSnapshot?
    /// Chunked prefill step size from the request's `GenerateParameters`,
    /// plumbed out so the post-generation canonical-leaf path can use the
    /// same chunk size when re-prefilling the canonical assistant residual
    /// on top of the restored last-message-boundary snapshot.
    let prefillStepSize: Int
    /// Stable prefix boundary detected for the request, if any. Reused by
    /// post-generation leaf-capture helpers so their restored-boundary
    /// prefills apply the same TriAttention prefix-protection policy as the
    /// main request prefill.
    let triAttentionStablePrefixOffset: Int?
    /// Rank of the processor-prepared token tensor (1 for pure LLMs, 2
    /// for conditional-generation models like Qwen3.5). Post-generation
    /// leaf-capture rebuilds residual inputs from a raw `[Int]` via
    /// `MLXArray(...)` (which is 1D), but the MLXVLM `prepare` indexes
    /// the tensor with two axes — passing 1D there crashes in
    /// `getRopeIndex` on `inputIds.dim(1)`.
    let tokenNDim: Int
}

nonisolated enum HTTPLeafContinuationKind: String, Sendable {
    case toolResult
    case userTurn
}

nonisolated enum HTTPLeafStoreMode: String, Sendable {
    case directToolLeaf
    case canonicalUserLeaf
    case directLeaf
}

/// Actor-isolated wrapper that owns the LLM model and runs inference off the MainActor.
///
/// Follows the same pattern as `WhisperKitSpeechRecognizer` below `TranscriptionEngine`:
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

    /// Load-time, directory-derived facts about the current model — tool-call
    /// format, Qwen3.5 family/MoE, prompt-starts-thinking, and flop profile.
    /// Installed by `installLoadTimeState`; `nil` before load and after
    /// `unloadModel()`.
    private var modelIdentity: ModelIdentity?

    /// Stable SHA-256 of the loaded model's weight files. Folded into every
    /// `CachePartitionKey` so a weight swap under the same `modelID`
    /// cannot surface stale persisted snapshots. `nil` before load and
    /// after `unloadModel()`.
    private var modelFingerprint: String?

    /// Snapshot of the SSD prefix-cache config captured at load time.
    /// Actor-isolated and synchronously readable from inside
    /// `container.perform`, which cannot await MainActor.
    private var ssdConfig: SSDPrefixCacheConfig?
    private var triAttentionRuntimeSelection: TriAttentionRuntimeSelection = .disabledDefault
    private var triAttentionConfiguration: TriAttentionConfiguration = .v1Disabled
    private let triAttentionCalibrationArtifactLoader: TriAttentionCalibrationArtifactLoader
    private var triAttentionCalibrationArtifactLookup: TriAttentionCalibrationArtifactLookupResult?

    var isLoaded: Bool { modelContainer != nil }

    init(
        triAttentionCalibrationArtifactLoader: TriAttentionCalibrationArtifactLoader? = nil
    ) {
        self.triAttentionCalibrationArtifactLoader =
            triAttentionCalibrationArtifactLoader ?? TriAttentionCalibrationArtifactLoader()
    }

    /// Internal read-only accessor for the load-time SSD config snapshot.
    /// Production reads happen via the synchronous capture in
    /// `makeHTTPPrefixCacheGeneration`; this accessor exists so tests
    /// can assert the load/unload lifecycle across the actor boundary.
    var currentSSDConfigForTesting: SSDPrefixCacheConfig? { ssdConfig }

    /// Internal read-only accessor for the load-time model fingerprint.
    var currentModelFingerprintForTesting: String? { modelFingerprint }

    /// Internal read-only accessor for the load-time model identity.
    var currentModelIdentityForTesting: ModelIdentity? { modelIdentity }

    /// Test-only: install a `ModelIdentity` without a full model load, so
    /// tests can exercise identity-derived wiring (notably the prefix cache's
    /// FLOP profile) at the actor seam. Production identity is installed only
    /// by `installLoadTimeState`.
    func setModelIdentityForTesting(_ identity: ModelIdentity?) {
        self.modelIdentity = identity
    }

    /// Test-only: the eviction configuration of the live prefix cache, or
    /// `nil` if the cache hasn't been built. Lets tests assert that
    /// `ensurePrefixCache` folds the model's `flopProfile` into the cache.
    func currentEvictionConfigForTesting() async -> EvictionConfiguration? {
        guard let cache = _prefixCache else { return nil }
        return await MainActor.run { cache.evictionConfig }
    }

    var currentTriAttentionRuntimeSelection: TriAttentionRuntimeSelection {
        triAttentionRuntimeSelection
    }

    /// Internal read-only accessor for the load-time TriAttention runtime selection.
    var currentTriAttentionRuntimeSelectionForTesting: TriAttentionRuntimeSelection {
        triAttentionRuntimeSelection
    }

    /// Internal read-only accessor for the load-time TriAttention config snapshot.
    var currentTriAttentionConfigForTesting: TriAttentionConfiguration { triAttentionConfiguration }

    /// Internal read-only accessor for the load-time TriAttention calibration lookup.
    var currentTriAttentionCalibrationArtifactLookupForTesting: TriAttentionCalibrationArtifactLookupResult? {
        triAttentionCalibrationArtifactLookup
    }

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
        ssdConfig: SSDPrefixCacheConfig? = nil,
        triAttention: TriAttentionConfiguration = .v1Disabled
    ) async throws -> (AgentTokenizer, promptStartsThinking: Bool) {
        let identity = ModelIdentity(directory: directory)
        let format = identity.toolCallFormat
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
        let isParoModel = isParoQuantModel(directory: directory)
        let isTriAttentionEligible = identity.isTriAttentionEligible
        let triAttentionRuntimeSelection = resolveTriAttentionRuntimeSelection(
            requestedConfiguration: triAttention,
            isTriAttentionEligible: isTriAttentionEligible,
            visionMode: visionMode,
            modelDirectory: directory
        )

        installLoadTimeState(
            modelIdentity: identity,
            fingerprint: fingerprint,
            ssdConfig: ssdConfig,
            triAttentionRuntimeSelection: triAttentionRuntimeSelection
        )
        logTriAttentionRuntimeSelection(
            triAttentionRuntimeSelection,
            modelFingerprint: fingerprint,
            isParoModel: isParoModel,
            isTriAttentionEligible: isTriAttentionEligible,
            visionMode: visionMode
        )

        if isParoModel {
            Log.agent.info("Detected ParoQuant model — using \(visionMode ? "VLM" : "LLM") path")
            let container: ModelContainer = visionMode
                ? try await loadParoQuantVLMContainer(from: directory, toolCallFormat: format)
                : try await loadParoQuantLLMContainer(from: directory, toolCallFormat: format)
            return try await verifyAndStore(container: container, identity: identity)
        }

        // Non-PARO Qwen3.5 checkpoints (e.g. mlx-community/Qwen3.5-*-MLX) ship
        // as `Qwen3_5ForConditionalGeneration` VLM bundles. The default factory
        // registry tries MLXVLM first and resolves them to the VLM `Qwen35`,
        // whose `prepare` indexes tokens with two axes and whose
        // `getRopeIndex` requires a batch dim — fundamentally incompatible
        // with the post-generation leaf-capture residual prefill path. V1
        // TriAttention is also text-only (see
        // `docs/tesseract-server-triattention-implementation-plan-2026-04-16.md`
        // section 2), so in non-vision mode we force the text-only
        // `MLXLLM.Qwen35Model` by invoking the LLM factory directly.
        let container: ModelContainer
        if !visionMode, identity.isQwen35 {
            Log.agent.info("Detected Qwen3.5 non-PARO model — forcing text-only LLM path")
            container = try await LLMModelFactory.shared.loadContainer(
                from: directory,
                using: #huggingFaceTokenizerLoader()
            )
        } else {
            container = try await loadModelContainer(
                from: directory,
                using: #huggingFaceTokenizerLoader()
            )
        }
        if let format {
            await container.update { context in
                context.configuration.toolCallFormat = format
            }
        }
        return try await verifyAndStore(container: container, identity: identity)
    }

    /// Start a raw text/tool generation and surface the underlying vendor task so
    /// callers can deterministically wait for model use to actually stop.
    func startRawGeneration(
        input: sending UserInput,
        toolSpecs: [ToolSpec]?,
        parameters: AgentGenerateParameters,
        progressHandler: ServerInferenceProgressHandler? = nil
    ) async throws -> HTTPServerRawGenerationStart {
        guard let container = modelContainer else {
            throw AgentEngineError.modelNotLoaded
        }

        Memory.cacheLimit = Defaults.cacheLimitMB * 1024 * 1024

        let genParams = makeGenerateParameters(from: parameters)
        // Canonicalize once so the loop-handler sees the same dict iteration
        // order the tokenizer uses for the prompt. Type-aware tool-call
        // parsing reads this schema via `XMLFunctionParser.parse(content:tools:)`.
        let canonicalTools = Self.canonicalizeToolSpecs(toolSpecs)
        return try await container.perform(nonSendable: input) { context, input in
            await progressHandler?(.cacheLookupStarted)
            let lookupStarted = Date.timeIntervalSinceReferenceDate
            let prepared = try await context.processor.prepare(input: input)
            let lookupMs = (Date.timeIntervalSinceReferenceDate - lookupStarted) * 1000
            let promptTokenCount = prepared.text.tokens.size
            await progressHandler?(.cacheLookupFinished(.init(
                reason: "standardGenerationNoPrefixCache",
                cachedTokens: 0,
                sharedPrefixLength: 0,
                promptTokens: promptTokenCount,
                newTokensToPrefill: promptTokenCount,
                lookupMs: lookupMs,
                restoreMs: 0
            )))
            await progressHandler?(.prefillStarted(.init(
                promptTokens: promptTokenCount,
                cachedTokens: 0,
                newTokensToPrefill: promptTokenCount,
                prefillMs: nil
            )))
            let prefillStarted = Date.timeIntervalSinceReferenceDate
            let iterator = try TokenIterator(
                input: prepared,
                model: context.model,
                parameters: genParams
            )
            let prefillMs = (Date.timeIntervalSinceReferenceDate - prefillStarted) * 1000
            await progressHandler?(.prefillFinished(.init(
                promptTokens: promptTokenCount,
                cachedTokens: 0,
                newTokensToPrefill: promptTokenCount,
                prefillMs: prefillMs
            )))
            let (stream, completion) = MLXLMCommon.generateTask(
                promptTokenCount: promptTokenCount,
                modelConfiguration: context.configuration,
                tokenizer: context.tokenizer,
                iterator: iterator,
                tools: canonicalTools
            )
            return HTTPServerRawGenerationStart(
                stream: stream,
                cancel: { completion.cancel() },
                waitForCompletion: { await completion.value }
            )
        }
    }

    /// Thinking-loop safeguard continuation for the HTTP prefix-cache path.
    /// We already have the fully-tokenized original prompt (captured during
    /// prefill as `HTTPPrefixCacheGeneration.fullTokens`), so re-tokenizing
    /// the `UserInput` is unnecessary work — encode the appended hand-off text
    /// straight onto the token sequence. The resulting input is fed to a
    /// fresh `TokenIterator`; the on-device KV cache for the cancelled
    /// generation is discarded by vendor cleanup.
    func startThinkingContinuationFromTokens(
        originalTokens: [Int],
        tokenNDim: Int,
        safeThinkingPrefix: String,
        injection: String,
        toolSpecs: [ToolSpec]?,
        parameters: AgentGenerateParameters
    ) async throws -> HTTPServerRawGenerationStart {
        guard let container = modelContainer else {
            throw AgentEngineError.modelNotLoaded
        }
        Memory.cacheLimit = Defaults.cacheLimitMB * 1024 * 1024
        let genParams = makeGenerateParameters(from: parameters)
        let handoff = safeThinkingPrefix + injection
        let canonicalTools = Self.canonicalizeToolSpecs(toolSpecs)

        return try await container.perform { context in
            try Self.buildThinkingContinuationStart(
                context: context,
                originalTokens: originalTokens,
                tokenNDim: tokenNDim,
                handoffText: handoff,
                tools: canonicalTools,
                parameters: genParams
            )
        }
    }

    /// Thinking-loop safeguard continuation: re-prefill `originalInput` augmented
    /// with `safeThinkingPrefix + injection` (encoded as additional tokens) so
    /// the model picks up where it left off with the degen thinking truncated
    /// and a hand-off phrase that closes `</think>`.
    ///
    /// The original chat-template prompt already ends with
    /// `<|im_start|>assistant\n<think>\n` (Qwen3.5 thinking preset), so we append
    /// tokens with `addSpecialTokens: false` — this extends the in-progress
    /// assistant turn rather than starting a new one.
    ///
    /// This intentionally bypasses the HTTP prefix cache: the continuation is
    /// rare, the original prompt is typically small, and routing through the
    /// prefix cache would require first-class partial-assistant-turn support
    /// that the Qwen3.5 chat template doesn't provide.
    func startThinkingContinuationRaw(
        originalInput: sending UserInput,
        safeThinkingPrefix: String,
        injection: String,
        toolSpecs: [ToolSpec]?,
        parameters: AgentGenerateParameters
    ) async throws -> HTTPServerRawGenerationStart {
        guard let container = modelContainer else {
            throw AgentEngineError.modelNotLoaded
        }

        Memory.cacheLimit = Defaults.cacheLimitMB * 1024 * 1024
        let genParams = makeGenerateParameters(from: parameters)
        let handoff = safeThinkingPrefix + injection
        let canonicalTools = Self.canonicalizeToolSpecs(toolSpecs)

        return try await container.perform(nonSendable: originalInput) { context, input in
            let basePrepared = try await context.processor.prepare(input: input)
            return try Self.buildThinkingContinuationStart(
                context: context,
                originalTokens: Self.extractTokenSequence(basePrepared.text.tokens),
                tokenNDim: basePrepared.text.tokens.ndim,
                handoffText: handoff,
                tools: canonicalTools,
                parameters: genParams
            )
        }
    }

    /// Build an `HTTPServerRawGenerationStart` by extending a pre-tokenized
    /// prompt with the hand-off suffix and feeding the combined sequence to
    /// a fresh `TokenIterator`. Must run inside a
    /// ``ModelContainer/perform(_:)`` closure on this actor.
    private static func buildThinkingContinuationStart(
        context: ModelContext,
        originalTokens: [Int],
        tokenNDim: Int,
        handoffText: String,
        tools: [ToolSpec]?,
        parameters: GenerateParameters
    ) throws -> HTTPServerRawGenerationStart {
        let appendedIDs = try context.tokenizer.encode(
            text: handoffText,
            addSpecialTokens: false
        )
        let combined = originalTokens + appendedIDs
        let flatArr = MLXArray(combined.map { Int32($0) })
        let tokenArr: MLXArray = tokenNDim >= 2
            ? flatArr.expandedDimensions(axis: 0)
            : flatArr
        let continued = LMInput(text: LMInput.Text(tokens: tokenArr, mask: nil))

        let iterator = try TokenIterator(
            input: continued,
            model: context.model,
            parameters: parameters
        )
        let (stream, completion) = MLXLMCommon.generateTask(
            promptTokenCount: combined.count,
            modelConfiguration: context.configuration,
            tokenizer: context.tokenizer,
            iterator: iterator,
            tools: tools
        )
        return HTTPServerRawGenerationStart(
            stream: stream,
            cancel: { completion.cancel() },
            waitForCompletion: { await completion.value }
        )
    }

    /// Start the HTTP text-based prefix-cache path for `/v1/chat/completions`.
    ///
    /// Returns `nil` when the request shape is incompatible and the caller should fall back
    /// to the normal generation path.
    func generateServerTextCompletion(
        modelID: String,
        conversation: HTTPPrefixCacheConversation,
        toolSpecs: [ToolSpec]?,
        parameters: AgentGenerateParameters,
        progressHandler: ServerInferenceProgressHandler? = nil
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
        let genParams = makeGenerateParameters(from: parameters)
        // Canonicalize tools once so the leaf re-tokenization uses the same dict
        // iteration order as the prefill path inside makeHTTPPrefixCacheGeneration.
        let canonicalTools = Self.canonicalizeToolSpecs(toolSpecs)
        let mlxStart = try await makeHTTPPrefixCacheGeneration(
            container: container,
            conversation: conversation,
            requestID: requestID,
            modelID: modelID,
            parameters: genParams,
            toolSpecs: canonicalTools,
            prefixCache: prefixCache,
            progressHandler: progressHandler
        )

        let mlxStartBox = UnsafeSendableBox(mlxStart)
        let (stream, continuation) = AsyncThrowingStream<AgentGeneration, Error>.makeStream()
        let startsInsideThinkBlock = promptStartsThinking
        let loadedModelWeightBytes = modelWeightBytes

        let safeguardConfig = parameters.thinkingSafeguard
        parameters.warnIfThinkingLoopRiskElevated(startsThinking: startsInsideThinkBlock)
        let fullTokensForContinuation = mlxStart.fullTokens
        let tokenNDimForContinuation = mlxStart.tokenNDim
        let continuationInjection = safeguardConfig.continuationHandOff
        let continuationToolSpecs = canonicalTools
        let actorRef = self
        let continuationStarter:
            @Sendable (String) async throws -> HTTPServerRawGenerationStart = { safePrefix in
            try await actorRef.startThinkingContinuationFromTokens(
                originalTokens: fullTokensForContinuation,
                tokenNDim: tokenNDimForContinuation,
                safeThinkingPrefix: safePrefix,
                injection: continuationInjection,
                toolSpecs: continuationToolSpecs,
                parameters: parameters
            )
        }

        // The loop owns the cross-swap cancel invariant (which raw handle is live
        // after an intervention swap). Its `cancelCurrent` must be wired into
        // `start.cancel` synchronously, but the loop isn't built until the task
        // starts — bridge through a late-bound cancel the task fills.
        let loopCancel = LateBoundCancel()

        let task = Task {
            [conversation, container, canonicalTools, requestID, loadedModelWeightBytes, genParams] in
            let mlxStart = mlxStartBox.value
            let diagnosticsContext = mlxStart.diagnosticsContext
            // The accumulator fold and the `.toolCall → HTTPPrefixCacheToolCall`
            // projection are the server's per-event side effects (its sink); the
            // streaming spine itself lives in `GenerationStreamLoop`.
            var accumulator = GenerationAccumulator()
            var toolCalls: [HTTPPrefixCacheToolCall] = []

            do {
                func handle(_ event: AgentGeneration) {
                    // Fold shared accumulation (text/thinking/safeguard prefix)
                    // in one place. The leaf-store tool-call projection
                    // (raw `ToolCall` → `HTTPPrefixCacheToolCall`) stays here, as
                    // does the continuation yield that drives downstream
                    // consumers (the Requests-log UI).
                    accumulator.ingest(event)
                    if case .toolCall(let call) = event {
                        toolCalls.append(HTTPPrefixCacheToolCall(
                            name: call.function.name,
                            arguments: call.function.arguments
                        ))
                    }
                    continuation.yield(event)
                }

                func logEvictions(_ evictions: [PrefixCacheManager.EvictionEvent]) {
                    for event in evictions {
                        diagnosticsContext.log(PrefixCacheDiagnostics.EvictionEvent(event))
                        // Body-drop with live Snapshot Ref → pending body-dropped
                        // or state 4→5 transition. Surface a separate
                        // `ssdBodyDrop` event so an operator scanning
                        // the eviction log can correlate the RAM
                        // freeing with the SSD-tier survival of the
                        // same node.
                        if let id = event.bodyDroppedSnapshotRefID {
                            diagnosticsContext.log(
                                PrefixCacheDiagnostics.SSDBodyDropEvent(id: id)
                            )
                        }
                    }
                }

                func logSupersededLeaves(
                    _ supersededLeaves: [PrefixCacheManager.LeafSupersession]
                ) {
                    for supersession in supersededLeaves {
                        diagnosticsContext.log(PrefixCacheDiagnostics.LeafSupersessionEvent(
                            offset: supersession.offset,
                            snapshotRefID: supersession.bodyDroppedSnapshotRefID
                        ))
                    }
                }

                // Restore-state snapshot: what cache state does this generation
                // begin from? Pair this with the silent-close warning to
                // correlate model misbehavior with cache hits (e.g. the Qwen3.6
                // hybrid-linear-attention stale-state bug, jundot/omlx#825).
                Log.agent.info(
                    "Generation starting — "
                    + "request_id=\(requestID.uuidString) "
                    + "cached=\(mlxStart.skippedPrefillTokens)/"
                    + "\(mlxStart.promptTokenCount) "
                    + "sharedPrefix=\(mlxStart.sharedPrefixLength) "
                    + "lookup=\(mlxStart.lookupReason) "
                    + "restoreMs=\(String(format: "%.1f", mlxStart.restoreMs * 1000)) "
                    + "prefillMs=\(String(format: "%.1f", mlxStart.prefillMs * 1000))"
                )

                // Drive the shared spine. `handle` is the sink (fold + project +
                // yield); the loop captures the terminal `.info` and the
                // silent-close diagnostics into its `Outcome` rather than sinking
                // them. The server always supplies a continuation starter.
                let loop = GenerationStreamLoop(
                    initial: .init(mlxStart),
                    startsInsideThinkBlock: startsInsideThinkBlock,
                    safeguard: safeguardConfig,
                    logContext: "request_id=\(requestID.uuidString)"
                )
                loopCancel.fill(loop.cancelCurrent)

                let outcome = try await loop.run(
                    continuation: { safePrefix in
                        GenerationStreamLoop.RawGenerationHandle(
                            try await continuationStarter(safePrefix)
                        )
                    },
                    sink: handle
                )

                if outcome.cancelled {
                    Memory.clearCache()
                    continuation.finish()
                    return
                }

                if let completionInfo = outcome.completionInfo {
                    // Re-yield the terminal `.info` so CompletionHandler's
                    // non-streaming and SSE paths still read final completion
                    // metrics (token counts / finish reason) from the stream,
                    // exactly as the previous EOS `handle(.info:)` did.
                    handle(.info(completionInfo))
                    diagnosticsContext.log(PrefixCacheDiagnostics.TTFTEvent(
                        lookupMs: mlxStart.lookupMs,
                        restoreMs: mlxStart.restoreMs,
                        prefillMs: mlxStart.prefillMs,
                        totalPromptMs: completionInfo.promptTime
                    ))
                    Log.agent.info(
                        "Generation complete — \(completionInfo.generationTokenCount) tokens, "
                        + "\(String(format: "%.1f", completionInfo.tokensPerSecond)) tok/s, "
                        + "stopReason=\(describeStopReason(completionInfo.stopReason))"
                    )
                    Log.agent.debug(
                        "Raw library chunks (after ToolCallProcessor):\n\(outcome.diagnostics.rawChunksJoined)"
                    )
                    if outcome.diagnostics.hasUnparsedToolCallMarkers {
                        Log.agent.warning(
                            "Raw output contains tool call markers but no .toolCall events were emitted by library"
                        )
                    }
                } else {
                    // Stream closed without an `.info` event from MLX — the case we
                    // were previously blind to (jundot/omlx#825: Qwen3.6 hybrid
                    // linear attention losing tool-calling after prefix-cache hit).
                    // The loop-owned diagnostics plus server-local cache context
                    // give the operator one correlatable log cluster.
                    let rawChunks = outcome.diagnostics.rawChunksJoined
                    let parserState = outcome.diagnostics.finalizeState
                    Log.agent.warning(
                        "Generation stream closed without .info event — "
                        + "request_id=\(requestID.uuidString) "
                        + "rawLen=\(rawChunks.count) "
                        + "libraryParsedToolCalls=\(outcome.diagnostics.libraryParsedToolCalls) "
                        + "cachedTokens=\(mlxStart.skippedPrefillTokens)/"
                        + "\(mlxStart.promptTokenCount) "
                        + "lookupReason=\(mlxStart.lookupReason) "
                        + "parserInsideThink=\(parserState.insideThinkBlock) "
                        + "parserThinkClosed=\(parserState.thinkBlockClosed) "
                        + "parserBufferLen=\(parserState.bufferLen) "
                        + "rawTail=\(String(rawChunks.suffix(200)).debugDescription)"
                    )
                }

                // -- Post-generation: store snapshots in radix tree --

                // Store mid-prefill snapshots (e.g. stable-prefix boundary) unconditionally.
                // These are captured during prefill and independent of the leaf path — if
                // final-cache recovery or leaf capture fails, the stable-prefix checkpoint
                // still saves future requests from a full re-prefill.
                var storedSnapshotsForTuner: [HybridCacheSnapshot] = []
                if !Task.isCancelled, let admission = mlxStart.snapshotAdmission {
                    let diagnostics = await MainActor.run {
                        prefixCache.admit(admission)
                    }
                    logEvictions(diagnostics.evictions)
                    storedSnapshotsForTuner = admission.snapshots
                }

                if Task.isCancelled {
                    Memory.clearCache()
                    continuation.finish()
                    return
                }

                // Leaf store, wrapped so any skip path falls through to
                // the request-end recordRequest call below — the alpha
                // tuner needs to see every request, not just the ones
                // whose leaf store completed.
                //
                // Canonical leaf policy:
                // - thinking templates store one template-canonical leaf
                //   synthesized from the transient boundary snapshot
                // - non-thinking templates store the direct post-response
                //   leaf captured from the final cache
                var leafStoreForTuner: AlphaTuner.LeafStore? = nil
                var postGenerationParams = genParams
                postGenerationParams.triAttentionStablePrefixOffset =
                    mlxStart.triAttentionStablePrefixOffset
                leafBlock: do {
                    // Skip leaf-store when a thinking-safeguard intervention
                    // fired: the continuation ran through the raw path, so the
                    // on-device KV cache no longer matches the radix-tree
                    // logical snapshot we'd compute from
                    // `textContent + thinkingContent + toolCalls`. Storing
                    // anything here would corrupt future prefix-cache hits
                    // for requests sharing this prefix. The stable-prefix
                    // snapshot captured pre-generation is still stored
                    // unconditionally earlier in this task, so future requests
                    // still benefit from partial cache reuse; only the leaf
                    // is lost for this one turn.
                    if outcome.intervened {
                        diagnosticsContext.logSkip(
                            stage: "leafStore",
                            reason: "thinking-safeguard-intervention"
                        )
                        break leafBlock
                    }

                    // 1. Build stored conversation (prompt + generated assistant turn).
                    let storedConversation = conversation.appendingAssistant(.assistant(
                        content: accumulator.text,
                        reasoning: accumulator.thinking ?? "",
                        toolCalls: toolCalls
                    ))

                    // 2. Re-tokenize stored conversation → flat token sequence.
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

                    let leafStoreMode = Self.selectHTTPLeafStoreMode(
                        promptStartsThinking: startsInsideThinkBlock,
                        emittedToolCalls: !toolCalls.isEmpty
                    )
                    diagnosticsContext.log(PrefixCacheDiagnostics.LeafModeEvent(
                        mode: leafStoreMode.rawValue,
                        continuation: toolCalls.isEmpty
                            ? HTTPLeafContinuationKind.userTurn.rawValue
                            : HTTPLeafContinuationKind.toolResult.rawValue
                    ))

                    // directLeaf snapshots the live final KV cache (below) and
                    // needs none of the builder's probe/boundary/tokenizer work;
                    // only the boundary modes route through the GPU-free plan.
                    // This mapping is the one place that knows directLeaf is the
                    // live-cache path, so a future `HTTPLeafStoreMode` surfaces as
                    // a compile error here rather than a silently missed branch.
                    let boundaryMode: BoundaryLeafMode? = switch leafStoreMode {
                    case .directToolLeaf: .directTool
                    case .canonicalUserLeaf: .canonical
                    case .directLeaf: nil
                    }
                    if let boundaryMode {
                        let transientBoundary: HybridCacheSnapshot? = switch boundaryMode {
                        case .directTool: mlxStart.transientLastMessageBoundarySnapshot
                        case .canonical: mlxStart.transientLastUserBoundarySnapshot
                        }
                        let leafTokenizer = await container.perform { $0.tokenizer }
                        let leafPlan = await LeafAdmissionBuilder.plan(
                            mode: boundaryMode,
                            storedConversation: storedConversation,
                            storedTokens: storedTokens,
                            toolSpecs: canonicalTools,
                            transientBoundary: transientBoundary,
                            tokenizer: leafTokenizer,
                            resolveBoundary: { tokens in
                                // Drive Snapshot Resolution inside `container.perform`
                                // so the SSD `loadSync` stays off-MainActor (ADR-0001).
                                await container.perform { _ in
                                    await SnapshotResolution.resolve(
                                        tokens: tokens,
                                        promptTokenCount: tokens.count,
                                        partitionKey: mlxStart.partitionKey,
                                        modelFingerprint: mlxStart.partitionKey.modelFingerprint,
                                        prefixCache: prefixCache,
                                        diagnostics: diagnosticsContext
                                    ).lookup.snapshot
                                }
                            }
                        )

                        // One exhaustive switch over the boundary plan: `.skip`
                        // logs the decidable reason; `.fromBoundary` runs the
                        // shared restore→reprefill→capture executor. Both leave
                        // the leaf block — only directLeaf reaches the live
                        // final-cache capture below.
                        switch leafPlan {
                        case .skip(let reason):
                            Self.logLeafSkip(
                                reason, mode: boundaryMode, diagnosticsContext: diagnosticsContext
                            )
                            break leafBlock
                        case .fromBoundary(let boundarySnapshot, let boundaryStoredTokens):
                            let stages = Self.leafStages(for: boundaryMode)
                            leafStoreForTuner = await Self.captureStructuredLeafFromBoundary(
                                container: container,
                                storedTokens: boundaryStoredTokens,
                                boundarySnapshot: boundarySnapshot,
                                partitionKey: mlxStart.partitionKey,
                                prefillStepSize: mlxStart.prefillStepSize,
                                tokenNDim: mlxStart.tokenNDim,
                                requestID: requestID,
                                prefixCache: prefixCache,
                                diagnosticsContext: diagnosticsContext,
                                ssdEnabled: mlxStart.ssdEnabled,
                                generateParameters: postGenerationParams,
                                storeStage: stages.store,
                                captureStage: stages.capture,
                                admissionStage: stages.admission,
                                captureSource: stages.source
                            )
                            break leafBlock
                        }
                    }

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

                    // 3. Offset-alignment guard: if normalization shortened the
                    //    stored conversation (whitespace-only assistant content → ""),
                    //    we can only trim attention K/V — Mamba's recurrent state
                    //    and TriAttention's sparse retained-position state can't
                    //    be unwound (`canTrimPromptCache` returns `false` for
                    //    both). Trimming the cache and capturing it as a leaf
                    //    produces a snapshot whose attention is aligned to
                    //    `storedTokens.count` but whose Mamba/TriAttention state
                    //    is from the full pre-trim offset. On Qwen3.5 the
                    //    resulting leaf hit perturbs raw logits by ~10 even at
                    //    trim=1: argmax stays stable (greedy decoding survives),
                    //    but the rest of the distribution drifts in a way that
                    //    affects sampled decoding. Since the HTTP server
                    //    propagates the request's `temperature`/`top_p` and we
                    //    can't predict future request sampling params at store
                    //    time, the safe choice is to skip the leaf store
                    //    entirely when normalization would require any trim.
                    //    Lost cache hits on whitespace-normalized conversations
                    //    are the trade-off for sampler-agnostic correctness.
                    //    Verified by `HybridCacheCorrectnessRunner` test 9 — see
                    //    the `leafHitWithNormalizationDivergence...` diagnostics
                    //    for the empirical drift measurements.
                    let actualCacheOffset = httpPrefixCacheReportedTokenCount(finalCache)
                    if actualCacheOffset > storedTokens.count {
                        let trimAmount = actualCacheOffset - storedTokens.count
                        let hasTriAttention = containsTriAttentionState(finalCache)
                        diagnosticsContext.logSkip(
                            stage: "leafStore",
                            reason: "normalization-trim",
                            extraFields: [
                                ("trimAmount", "\(trimAmount)"),
                                ("offsetBefore", "\(actualCacheOffset)"),
                                ("canonicalCount", "\(storedTokens.count)"),
                                ("triAttention", "\(hasTriAttention)"),
                            ]
                        )
                        break leafBlock
                    }

                    // 4. Capture leaf snapshot and derive its admission
                    //    storage inside a Metal-affine `container.perform`
                    //    so any per-array `asData()` calls run on the
                    //    inference thread. `finalCache` is non-`Sendable`
                    //    `[any KVCache]` — routed through the vendor's
                    //    `nonSendable` perform overload. The offset
                    //    guard above ensures no per-layer trimming is
                    //    needed before capture.
                    let ssdEnabled = mlxStart.ssdEnabled
                    let (maybeLeaf, maybeLeafAdmission): (HybridCacheSnapshot?, SnapshotAdmission?) =
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
                            let storage = Self.snapshotAdmissionStorage(
                                for: snap,
                                ssdEnabled: ssdEnabled
                            )
                            let leafAdmission = SnapshotAdmission.leaf(
                                storedTokens: storedTokens,
                                snapshot: snap,
                                storage: storage,
                                partitionKey: mlxStart.partitionKey,
                                requestID: requestID
                            )
                            return (snap, leafAdmission)
                        }
                    guard let leafSnapshot = maybeLeaf else {
                        diagnosticsContext.logSkip(
                            stage: "leafCapture",
                            reason: "unsupported-cache-type",
                            extraFields: [("cacheOffsets", "\(cacheOffsets)")]
                        )
                        break leafBlock
                    }
                    guard let leafAdmission = maybeLeafAdmission else {
                        diagnosticsContext.logSkip(
                            stage: "leafAdmission",
                            reason: "invalid-path",
                            extraFields: [
                                ("offset", "\(leafSnapshot.tokenOffset)"),
                                ("storedLen", "\(storedTokens.count)"),
                            ]
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

                    // Coalesce admit + stats read in one MainActor
                    // hop — saves one cross-actor switch on the success
                    // path (the request hot path). Includes the post-store
                    // budget/total snapshot so the admission diagnostic can
                    // be logged from this actor without another hop.
                    let (diagnostics, postStoreBudgetBytes, postStoreSnapshotBytes) =
                        await MainActor.run { () -> (PrefixCacheManager.StoreDiagnostics, Int, Int) in
                            let d = prefixCache.admit(leafAdmission)
                            return (d, prefixCache.memoryBudgetBytes, prefixCache.totalSnapshotBytes)
                        }
                    logEvictions(diagnostics.evictions)
                    logSupersededLeaves(diagnostics.supersededLeaves)
                    let directAdmissionEvicted = diagnostics.evictions.contains { event in
                        event.offset == leafSnapshot.tokenOffset
                            && event.checkpointType == .leaf
                    }
                    if directAdmissionEvicted {
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
            cachedTokenCount: mlxStart.skippedPrefillTokens,
            cancel: {
                // Cancel the live raw handle (whichever is current after an
                // intervention swap) to unpark a mid-generation `for await`, then
                // the driving task.
                loopCancel()
                task.cancel()
            },
            waitForCompletion: {
                // The loop awaits the live handle internally before returning, so
                // waiting on the task is sufficient.
                _ = await task.result
            },
            diagnostics: .fromSeconds(
                lookup: mlxStart.lookupMs,
                restore: mlxStart.restoreMs,
                prefill: mlxStart.prefillMs,
                cacheReason: String(describing: mlxStart.lookupReason),
                sharedPrefixLength: mlxStart.sharedPrefixLength,
                promptTokenCount: mlxStart.promptTokenCount
            )
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
        modelIdentity = nil
        modelFingerprint = nil
        ssdConfig = nil
        triAttentionRuntimeSelection = .disabledDefault
        triAttentionConfiguration = .v1Disabled
        triAttentionCalibrationArtifactLookup = nil
    }

    /// Frees unreferenced MLX buffers between tool rounds.
    func clearMemoryCache() {
        Memory.clearCache()
    }

    /// Block until any pending SSD-tier writes have drained and the
    /// manifest is durably persisted. Callers must invoke this
    /// before `unloadModel()` when they need the on-disk state to
    /// survive the teardown (benchmark restart scenarios, manual
    /// "flush before shutdown" flows). No-op when SSD is disabled
    /// or the prefix cache was never instantiated.
    func flushPrefixCache() async {
        guard let cache = _prefixCache else { return }
        await cache.flushSSDWrites()
    }

    /// Returns current MLX memory usage in MB.
    func memoryStats() -> (activeMB: Float, peakMB: Float) {
        (Float(Memory.activeMemory) / 1e6, Float(Memory.peakMemory) / 1e6)
    }

    // MARK: - Prefix Cache Helpers

    /// Lazily creates and returns the `PrefixCacheManager`. Initialization requires
    /// a MainActor hop because PrefixCacheManager is `@MainActor`.
    /// The production cache attaches an `AlphaTuner` so eviction `alpha`
    /// adapts to the workload after the first eviction fires. Each cache
    /// owns its **Eviction Configuration**, so a fresh cache starts at the
    /// LRU default (`alpha = 0`) and reads the model's `flopProfile` from
    /// **Model Identity** — there is no global to reset or leak.
    ///
    /// When `ssdConfig?.enabled == true` the manager is composed over
    /// a `TieredSnapshotStore` owning an `SSDSnapshotStore`, and
    /// `warmStart` restores the radix-tree structure from the on-disk
    /// manifest. Warm start is fingerprint-gated: partitions from a
    /// different model layout get their descriptors skipped and
    /// their directories scheduled for async cleanup.
    private func ensurePrefixCache() async -> PrefixCacheManager {
        if let existing = _prefixCache { return existing }
        let budget = defaultPrefixCacheMemoryBudgetBytes
        let ssdConfigSnapshot = self.ssdConfig
        let fingerprint = self.modelFingerprint
        let flopProfile: ModelFlopProfile
        if let identity = self.modelIdentity {
            flopProfile = identity.flopProfile
        } else {
            // Normally unreachable: the model load installs the identity and
            // `verifyAndStore` nils any pre-load cache, so the cache is built
            // (or rebuilt) once the identity is known. A nil identity here
            // means a pre-load caller (e.g. the E2E budget/alpha tooling) built
            // the cache early; it gets the shared fallback profile until the
            // next load rebuilds it.
            flopProfile = .fallback
            Log.agent.info(
                "PrefixCacheManager built before model identity is known — "
                + "using the fallback FLOP profile; the cache is rebuilt after load."
            )
        }
        let cache = await MainActor.run { () -> PrefixCacheManager in
            let tieredStore = TieredSnapshotStore(ssdConfig: ssdConfigSnapshot)
            return PrefixCacheManager(
                memoryBudgetBytes: budget,
                evictionConfig: EvictionConfiguration(flopProfile: flopProfile),
                alphaTuner: AlphaTuner(flopProfile: flopProfile),
                tieredStore: tieredStore
            )
        }
        if ssdConfigSnapshot?.enabled == true, let fingerprint {
            do {
                try await cache.warmStart(modelFingerprint: fingerprint)
            } catch {
                Log.agent.error(
                    "PrefixCacheManager.warmStart failed: \(String(describing: error))"
                )
            }
        }
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

    func promptCacheTelemetrySnapshot() async -> PromptCacheTelemetrySnapshot? {
        guard let cache = _prefixCache else { return nil }
        return await cache.makeTelemetrySnapshot()
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

    /// Override the prefix-cache eviction weighting (`alpha`) by writing
    /// the manager's **Eviction Configuration**. Used by the loaded-model
    /// E2E runner to force F/B-weighted eviction for the branch-point
    /// survival check — it replaces the retired `EvictionPolicy.alpha`
    /// global. Production code should not call this; the `AlphaTuner` owns
    /// `alpha` after warmup.
    func setEvictionAlpha(_ alpha: Double) async {
        let cache = await ensurePrefixCache()
        await MainActor.run {
            cache.evictionConfig.alpha = alpha
        }
    }

    /// Current prefix-cache eviction weighting (`alpha`), or `nil` if the
    /// cache hasn't been built. Symmetric with `setEvictionAlpha`, so the E2E
    /// runner can save and restore the weighting around its forced-pressure
    /// step instead of leaving the cache mutated.
    func evictionAlpha() async -> Double? {
        guard let cache = _prefixCache else { return nil }
        return await MainActor.run { cache.evictionConfig.alpha }
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

    /// Pre-extract checkpoint snapshots into Snapshot Admission
    /// candidates, attaching storage intent to each entry at the
    /// Metal-affine extraction edge.
    ///
    /// **Metal-affinity contract.** Must be called from inside
    /// ``ModelContainer/perform(_:)`` on `LLMActor` — calling it
    /// outside a live Metal-affine scope risks re-issuing command-queue
    /// work on a non-inference thread. The method is `nonisolated
    /// static` so callers can invoke it synchronously from inside a
    /// `container.perform` closure without an `await`; the Metal
    /// affinity is enforced by convention, not the type system.
    nonisolated static func extractCheckpointAdmissionCandidates(
        _ snapshots: [HybridCacheSnapshot],
        ssdEnabled: Bool
    ) -> [SnapshotAdmission.CheckpointCandidate] {
        snapshots.map { snapshot in
            return SnapshotAdmission.CheckpointCandidate(
                snapshot: snapshot,
                storage: snapshotAdmissionStorage(
                    for: snapshot,
                    ssdEnabled: ssdEnabled
                )
            )
        }
    }

    private nonisolated static func snapshotAdmissionStorage(
        for snapshot: HybridCacheSnapshot,
        ssdEnabled: Bool
    ) -> SnapshotAdmission.Storage {
        guard ssdEnabled else { return .ramOnly }
        return .ramAndSSD(extractSnapshotPayload(snapshot))
    }

    private nonisolated static func extractSnapshotPayload(
        _ snapshot: HybridCacheSnapshot
    ) -> SnapshotPayload {
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

        return SnapshotPayload(
            tokenOffset: snapshot.tokenOffset,
            checkpointType: snapshot.checkpointType,
            layers: layers
        )
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

    /// Inverse of ``dtypeWireString``. Must stay exhaustive against
    /// the forward table so round-tripping an SSD-resident snapshot
    /// cannot silently lose dtype information; every branch in
    /// `dtypeWireString` has a matching branch here. Returns `nil`
    /// for unknown wire strings so the `SSDSnapshotStore` decoder
    /// can distinguish a parse error from a supported dtype.
    nonisolated static func dtypeFromWireString(_ wire: String) -> DType? {
        switch wire {
        case "bool": return .bool
        case "uint8": return .uint8
        case "uint16": return .uint16
        case "uint32": return .uint32
        case "uint64": return .uint64
        case "int8": return .int8
        case "int16": return .int16
        case "int32": return .int32
        case "int64": return .int64
        case "float16": return .float16
        case "float32": return .float32
        case "bfloat16": return .bfloat16
        case "complex64": return .complex64
        case "float64": return .float64
        default: return nil
        }
    }

    /// Ensure a malformed tool-call buffer surfaced from the vendor's
    /// ToolCallProcessor is wrapped with both `<tool_call>` and `</tool_call>`
    /// tags. The vendor's in-flight buffer always starts with `<tool_call>`
    /// by construction (see `ToolCallProcessor.processTaggedChunk`), but the
    /// close tag may be missing when the model was interrupted before
    /// emitting it — in that case the HTTP client (e.g. opencode) otherwise
    /// sees only an opening tag and can't detect the tool-call attempt.
    /// Idempotent: if the buffer already has both tags, returns it unchanged.

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
        identity: ModelIdentity
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
        let startsThinking = identity.promptStartsThinking
        let profile = identity.flopProfile
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
        Log.agent.info(
            "Model FLOP profile (flows into the prefix cache's Eviction "
            + "Configuration at construction) — D=\(profile.hiddenSize) "
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

    /// Single install site for per-load snapshot state. Called from
    /// `loadModel` before the container load is attempted so the state
    /// is visible even on failed loads; this lets the unit suite exercise
    /// the full config-resolution chain via a fake directory that trips
    /// the container load. The unload path clears all of these fields.
    private func installLoadTimeState(
        modelIdentity: ModelIdentity,
        fingerprint: String,
        ssdConfig: SSDPrefixCacheConfig?,
        triAttentionRuntimeSelection: TriAttentionRuntimeSelection
    ) {
        self.modelIdentity = modelIdentity
        self.modelFingerprint = fingerprint
        self.ssdConfig = ssdConfig
        self.triAttentionRuntimeSelection = triAttentionRuntimeSelection
        self.triAttentionConfiguration = triAttentionRuntimeSelection.effectiveConfiguration
        self.triAttentionCalibrationArtifactLookup = triAttentionRuntimeSelection.calibrationArtifactLookup
    }

    private func resolveTriAttentionRuntimeSelection(
        requestedConfiguration: TriAttentionConfiguration,
        isTriAttentionEligible: Bool,
        visionMode: Bool,
        modelDirectory: URL
    ) -> TriAttentionRuntimeSelection {
        let calibrationArtifactLookup: TriAttentionCalibrationArtifactLookupResult? =
            if requestedConfiguration.enabled && isTriAttentionEligible && !visionMode {
                resolveTriAttentionCalibrationArtifactLookup(modelDirectory: modelDirectory)
            } else {
                nil
            }

        return TriAttentionRuntimeSelection.resolve(
            requestedConfiguration: requestedConfiguration,
            isTriAttentionEligible: isTriAttentionEligible,
            visionMode: visionMode,
            calibrationArtifactLookup: calibrationArtifactLookup
        )
    }

    private func logTriAttentionRuntimeSelection(
        _ triAttentionRuntimeSelection: TriAttentionRuntimeSelection,
        modelFingerprint: String,
        isParoModel: Bool,
        isTriAttentionEligible: Bool,
        visionMode: Bool
    ) {
        switch (
            triAttentionRuntimeSelection.effectiveConfiguration.enabled,
            triAttentionRuntimeSelection.calibrationArtifactLookup
        ) {
        case (true, .some(.loaded(_, let identity, let relativeResourcePath))):
            Log.agent.info(
                "TriAttention runtime enabled — modelFingerprint=\(modelFingerprint) "
                + "budget=\(triAttentionRuntimeSelection.effectiveConfiguration.budgetTokens) "
                + "artifactIdentity=\(identity.rawValue) "
                + "resourcePath=\(relativeResourcePath)"
            )
        default:
            Log.agent.info(
                "TriAttention runtime selection — requestedEnabled=\(triAttentionRuntimeSelection.requestedConfiguration.enabled) "
                + "effectiveEnabled=\(triAttentionRuntimeSelection.effectiveConfiguration.enabled) "
                + "fallbackReason=\(triAttentionRuntimeSelection.fallbackReason?.rawValue ?? "none") "
                + "isParoModel=\(isParoModel) isTriAttentionEligible=\(isTriAttentionEligible) visionMode=\(visionMode) "
                + "modelFingerprint=\(modelFingerprint)"
            )
        }
    }

    /// Artifact lookup keys on a *content* fingerprint (bytes of config,
    /// tokenizer, and safetensors) — not the mtime-based `computeFingerprint`
    /// used for the local SSD cache. mtime varies per user's HuggingFace
    /// download; content doesn't, so shipped `.pt` artifacts named by content
    /// fingerprint resolve the same across machines.
    ///
    /// Parse/read failures must not block dense model loads.
    private func resolveTriAttentionCalibrationArtifactLookup(
        modelDirectory: URL
    ) -> TriAttentionCalibrationArtifactLookupResult {
        let contentFingerprint: String
        do {
            contentFingerprint = try ModelFingerprint.computeContentFingerprint(modelDir: modelDirectory)
        } catch {
            let attemptedURL = triAttentionCalibrationArtifactLoader.expectedURL(for: "")
            Log.agent.error(
                "TriAttention content fingerprint failed for \(modelDirectory.lastPathComponent): \(error.localizedDescription)"
            )
            return .unavailable(
                expectedModelFingerprint: "",
                attemptedURL: attemptedURL,
                errorDescription: error.localizedDescription
            )
        }

        do {
            return try triAttentionCalibrationArtifactLoader.lookup(modelFingerprint: contentFingerprint)
        } catch {
            let attemptedURL = triAttentionCalibrationArtifactLoader.expectedURL(for: contentFingerprint)
            Log.agent.error(
                "TriAttention calibration artifact unavailable for fingerprint \(contentFingerprint): \(error.localizedDescription)"
            )
            return .unavailable(
                expectedModelFingerprint: contentFingerprint,
                attemptedURL: attemptedURL,
                errorDescription: error.localizedDescription
            )
        }
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
        parameters: GenerateParameters,
        toolSpecs: [ToolSpec]?,
        prefixCache: PrefixCacheManager,
        progressHandler: ServerInferenceProgressHandler?
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
        let triAttentionIdentity = TriAttentionPartitionIdentity.from(
            self.triAttentionRuntimeSelection.effectiveConfiguration
        )
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

            let triAttentionRestoreContext = Self.makeTriAttentionSnapshotRestoreContext(
                model: context.model,
                parameters: parameters
            )

            // 1. Tokenize the full conversation (BEFORE cache lookup).
            let fullInput = try await context.processor.prepare(
                input: UserInput(messages: conversation.promptMessages, tools: canonicalTools)
            )
            // Sequence length is always the LAST dim. For LLM models tokens are
            // 1D [seq], for VLM models (ParoQuant Qwen35) they are 2D [batch, seq].
            let fullTokenCount = fullInput.text.tokens.dim(-1)
            let tokenNDim = fullInput.text.tokens.ndim

            // 2. Extract flat token sequence for radix tree operations.
            let fullTokens = Self.extractTokenSequence(fullInput.text.tokens)

            // 3. Build the global Marconi partition key for this model
            //    configuration. Cross-session sharing is intentional:
            //    identical prompts under the same model config should
            //    reuse the same radix tree.
            let partitionKey = CachePartitionKey(
                modelID: modelID,
                kvBits: parameters.kvBits,
                kvGroupSize: parameters.kvGroupSize,
                modelFingerprint: modelFingerprint,
                triAttention: triAttentionIdentity
            )

            // 4. Detect the prefill boundaries (stable prefix + last-message +
            // last-user). The Prefill Planner owns this tokenizer-affine work —
            // including the generation-prompt suffix subtraction and the
            // last-user re-render — in one tested place.
            let boundaries = try PrefillPlanner.detectBoundaries(
                conversation: conversation,
                toolSpecs: canonicalTools,
                fullTokens: fullTokens,
                promptStartsThinking: promptStartsThinking,
                tokenizer: context.tokenizer
            )

            // 5–6. Resolve the best cached prefix (lookup + lazy SSD hydration),
            // then plan checkpoints against the settled tree.
            await progressHandler?(.cacheLookupStarted)
            let lookupStarted = Date.timeIntervalSinceReferenceDate
            // Resolve the best usable snapshot in one place: radix lookup plus
            // lazy SSD hydration (consumed internally — only `.hit`/miss surface
            // here). `loadSync` stays off-MainActor inside this scope per
            // ADR-0001; promote/clear hop to MainActor inside `resolve`.
            let resolved = await SnapshotResolution.resolve(
                tokens: fullTokens,
                promptTokenCount: fullTokenCount,
                partitionKey: partitionKey,
                modelFingerprint: modelFingerprint,
                prefixCache: prefixCache,
                diagnostics: diagnosticsContext
            )
            let lookupResult = resolved.lookup
            // Plan AFTER resolution, against the settled tree: any promote or
            // forgiving clear has already happened, so the post-hydration-failure
            // replan becomes the ordinary single plan. `resolved.alignmentLookup`
            // carries the SSD-hydrated-hit special case — it aligns against
            // nothing, matching the pre-carve ordering against the unhydrated
            // `.ssdHit`.
            let checkpointPlan = await MainActor.run {
                prefixCache.planCheckpoints(
                    tokens: fullTokens,
                    stablePrefixOffset: boundaries.stablePrefixOffset,
                    partitionKey: partitionKey,
                    alignTo: resolved.alignmentLookup
                )
            }
            let lookupMs = Date.timeIntervalSinceReferenceDate - lookupStarted

            // 7. Fold resolution + plan into the request's Prefill Plan:
            // restore-vs-cold, the suffix checkpoint filter, the transient
            // boundary offsets, and the single `prefillBaseOffset` (which
            // collapses the old `skippedTokens` / `checkpointBaseOffset` pair).
            let prefillPlan = PrefillPlanner.plan(
                boundaries: boundaries,
                lookupResult: lookupResult,
                checkpointPlan: checkpointPlan,
                promptTokenCount: fullTokenCount
            )

            // Execute the restore decision (Metal): on a hit, slice the suffix
            // and restore the KV cache; otherwise run cold.
            let inputForGeneration: LMInput
            let cacheToUse: [any KVCache]?
            let restoreMs: TimeInterval

            switch prefillPlan.restore {
            case .restore(let cacheOffset):
                // Suffix-only prefill is safe for both dense and TriAttention
                // partitions: TriAttention layers are restored via
                // `HybridCacheSnapshot` with their absolute logical offset
                // intact, and each layer's own `makeMask` recreates the right
                // (causal or sparse) mask for the suffix tokens.
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
                    lookupResult.restoreCache(
                        triAttentionRestoreContext: triAttentionRestoreContext
                    )
                }
                cacheToUse = restoredCache
                restoreMs = measuredRestoreMs
            case .cold:
                inputForGeneration = fullInput
                cacheToUse = nil
                restoreMs = 0
            }
            let skippedTokens = prefillPlan.prefillBaseOffset
            let newTokensToPrefill = fullTokenCount - skippedTokens
            await progressHandler?(.cacheLookupFinished(.init(
                reason: String(describing: lookupResult.reason),
                cachedTokens: skippedTokens,
                sharedPrefixLength: lookupResult.sharedPrefixLength,
                promptTokens: fullTokenCount,
                newTokensToPrefill: newTokensToPrefill,
                lookupMs: lookupMs * 1000,
                restoreMs: restoreMs * 1000
            )))
            diagnosticsContext.log(PrefixCacheDiagnostics.LookupEvent(
                reason: lookupResult.reason,
                promptTokens: fullTokenCount,
                sharedPrefixLength: lookupResult.sharedPrefixLength,
                skippedPrefillTokens: skippedTokens,
                newTokensToPrefill: newTokensToPrefill,
                lookupMs: lookupMs,
                restoreMs: restoreMs,
                plannedCheckpoints: prefillPlan.checkpointsToCapture
            ))

            // 8. Set checkpoint offsets on parameters — flows into TokenIterator → prepare().
            // Planner guarantees offset uniqueness, so uniqueKeysWithValues traps loudly
            // on a planner-side invariant break instead of silently dropping a candidate.
            var genParams = parameters
            genParams.checkpoints = Dictionary(
                uniqueKeysWithValues: prefillPlan.checkpointsToCapture.map { ($0.offset, $0.type) }
            )
            genParams.triAttentionStablePrefixOffset = prefillPlan.stablePrefixOffset
            genParams.transientCheckpointOffsets = prefillPlan.transientCheckpointOffsets
            genParams.checkpointBaseOffset = prefillPlan.prefillBaseOffset

            // 9. Create TokenIterator — this calls model.prepare() internally with checkpoints.
            // NO separate prepare() call. TokenIterator owns prefill.
            await progressHandler?(.prefillStarted(.init(
                promptTokens: fullTokenCount,
                cachedTokens: skippedTokens,
                newTokensToPrefill: newTokensToPrefill,
                prefillMs: nil
            )))
            let (iterator, prefillMs) = try measure {
                try TokenIterator(
                    input: inputForGeneration,
                    model: context.model,
                    cache: cacheToUse,
                    parameters: genParams
                )
            }
            await progressHandler?(.prefillFinished(.init(
                promptTokens: fullTokenCount,
                cachedTokens: skippedTokens,
                newTokensToPrefill: newTokensToPrefill,
                prefillMs: prefillMs * 1000
            )))

            // 10. Read captured snapshots (populated by prepare() inside TokenIterator.init),
            // then extract their payloads inside this `container.perform` so
            // `MLXArray.asData()` runs on the Metal-affine thread before the
            // later MainActor store hop.
            let capturedSnapshots = iterator.capturedSnapshots
            let transientLastMessageBoundarySnapshot = prefillPlan.transientBoundaries.lastMessage.flatMap { offset in
                iterator.transientSnapshots[offset]
                    ?? capturedSnapshots.first(where: { $0.tokenOffset == offset })
            }
            let transientLastUserBoundarySnapshot = prefillPlan.transientBoundaries.lastUser.flatMap { offset in
                iterator.transientSnapshots[offset]
                    ?? capturedSnapshots.first(where: { $0.tokenOffset == offset })
            }
            let checkpointCandidates = Self.extractCheckpointAdmissionCandidates(
                capturedSnapshots,
                ssdEnabled: ssdEnabled
            )
            let snapshotAdmission = SnapshotAdmission.checkpoints(
                fullPromptTokens: fullTokens,
                candidates: checkpointCandidates,
                partitionKey: partitionKey,
                requestID: requestID
            )
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
                iterator: iterator,
                tools: canonicalTools
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
                lookupReason: lookupResult.reason,
                sharedPrefixLength: lookupResult.sharedPrefixLength,
                fullTokens: fullTokens,
                snapshotAdmission: snapshotAdmission,
                ssdEnabled: ssdEnabled,
                partitionKey: partitionKey,
                transientLastMessageBoundarySnapshot: transientLastMessageBoundarySnapshot,
                transientLastUserBoundarySnapshot: transientLastUserBoundarySnapshot,
                prefillStepSize: parameters.prefillStepSize,
                triAttentionStablePrefixOffset: boundaries.stablePrefixOffset,
                tokenNDim: tokenNDim
            )
        }
    }

    private func makeGenerateParameters(
        from parameters: AgentGenerateParameters
    ) -> GenerateParameters {
        let calibrationArtifact: TriAttentionCalibrationArtifact? =
            if triAttentionRuntimeSelection.effectiveConfiguration.enabled,
                case .some(.loaded(let artifact, _, _)) = triAttentionCalibrationArtifactLookup
            {
                artifact
            } else {
                nil
            }
        return GenerateParameters(
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
            presenceContextSize: parameters.presenceContextSize,
            frequencyPenalty: parameters.frequencyPenalty,
            frequencyContextSize: parameters.frequencyContextSize,
            prefillStepSize: parameters.prefillStepSize,
            triAttention: triAttentionRuntimeSelection.effectiveConfiguration,
            triAttentionCalibrationArtifact: calibrationArtifact
        )
    }

    func makeGenerateParametersForTesting(
        from parameters: AgentGenerateParameters
    ) -> GenerateParameters {
        makeGenerateParameters(from: parameters)
    }

    private nonisolated static func makeTriAttentionSnapshotRestoreContext(
        model: any LanguageModel,
        parameters: GenerateParameters
    ) -> TriAttentionSnapshotRestoreContext? {
        guard let provider = model as? TriAttentionSnapshotRestoreContextProviding else {
            return nil
        }
        return provider.triAttentionSnapshotRestoreContext(
            configuration: parameters.triAttention,
            artifact: parameters.triAttentionCalibrationArtifact
        )
    }

    nonisolated static func selectHTTPLeafStoreMode(
        promptStartsThinking: Bool,
        emittedToolCalls: Bool
    ) -> HTTPLeafStoreMode {
        if emittedToolCalls {
            return .directToolLeaf
        }
        if promptStartsThinking {
            return .canonicalUserLeaf
        }
        return .directLeaf
    }

    /// The diagnostics stage labels for a `.fromBoundary` capture, by boundary
    /// leaf mode — the exact strings the dissolved `captureDirectToolLeaf` /
    /// `captureCanonicalTemplateLeaf` helpers passed to the shared executor.
    private nonisolated static func leafStages(
        for mode: BoundaryLeafMode
    ) -> (store: String, capture: String, admission: String, source: String) {
        switch mode {
        case .directTool:
            ("directToolLeafStore", "directToolLeafCapture", "directToolLeafAdmission", "directToolLeaf")
        case .canonical:
            ("canonicalLeafStore", "canonicalLeafCapture", "canonicalLeafAdmission", "canonicalLeaf")
        }
    }

    /// The exact `logSkip` record a decidable `LeafSkipReason` reproduces — the
    /// stage/reason/level/fields the dissolved capture helpers logged.
    struct LeafSkipLog: Sendable {
        let stage: String
        let reason: String
        let level: PrefixCacheDiagnostics.Level
        let extraFields: [(String, String)]
    }

    /// Map a decidable skip to its wire record. The reason carries the payload
    /// (offsets, lengths); the stage prefix follows the boundary mode, exactly as
    /// the dissolved `captureDirectToolLeaf` / `captureCanonicalTemplateLeaf`
    /// helpers did. `.info` is the `logSkip` default those untyped helpers relied
    /// on, made explicit so the level is pinned too. A pure value (no `Context`,
    /// no side effect) so `LLMActorLeafSkipLogTests` pins the byte-for-byte wire
    /// format — mirroring `ssdDropReasonString` — and any future drift (a renamed
    /// stage, a flipped level) fails a test rather than silently shifting
    /// dashboards and the diagnostics net.
    nonisolated static func leafSkipLog(
        for reason: LeafSkipReason,
        mode: BoundaryLeafMode
    ) -> LeafSkipLog {
        let stage = leafStages(for: mode).store
        switch reason {
        case .tokenizationFailed(let error):
            // The probe's chat-template render threw — today's helpers catch this
            // in the same `do/catch` as the prefill, logged as `prefill-threw`.
            return LeafSkipLog(
                stage: stage, reason: "prefill-threw", level: .warning,
                extraFields: [("error", error)]
            )
        case .probeDivergence:
            return LeafSkipLog(
                stage: stage, reason: "probe-divergence-failed", level: .info, extraFields: []
            )
        case .noTransientBoundary:
            return LeafSkipLog(
                stage: stage, reason: "no-transient-boundary-snapshot", level: .info, extraFields: []
            )
        case .noResolvedBoundary(let canonicalLen):
            return LeafSkipLog(
                stage: stage, reason: "no-canonical-restore-boundary", level: .info,
                extraFields: [("canonicalLen", "\(canonicalLen)")]
            )
        case .storedAtOrBeforeBoundary(let storedLen, let boundaryOffset):
            return LeafSkipLog(
                stage: stage, reason: "stored-at-or-before-boundary", level: .info,
                extraFields: [("storedLen", "\(storedLen)"), ("boundaryOffset", "\(boundaryOffset)")]
            )
        case .canonicalLongerThanStored(let canonicalLen, let storedLen):
            return LeafSkipLog(
                stage: stage, reason: "canonical-longer-than-stored", level: .warning,
                extraFields: [("canonicalLen", "\(canonicalLen)"), ("storedLen", "\(storedLen)")]
            )
        }
    }

    /// Emit the mapped wire record for a decidable `LeafSkipReason` the **Leaf
    /// Admission Builder** returned, so existing dashboards and the diagnostics
    /// net keep working byte-for-byte.
    private nonisolated static func logLeafSkip(
        _ reason: LeafSkipReason,
        mode: BoundaryLeafMode,
        diagnosticsContext: PrefixCacheDiagnostics.Context
    ) {
        let record = leafSkipLog(for: reason, mode: mode)
        diagnosticsContext.logSkip(
            stage: record.stage,
            reason: record.reason,
            level: record.level,
            extraFields: record.extraFields
        )
    }

    /// Re-tokenize the stored conversation (prompt + generated response) and return
    /// the flat token sequence. The HTTP prefix cache uses raw prompt messages here
    /// so assistant `reasoning_content` and `tool_calls` survive template rendering.
    /// Used for storing the leaf snapshot under the correct radix path.
    /// Returns `nil` on tokenization failure.
    private static func measureStoredTokenSequence(
        container: ModelContainer,
        conversation: HTTPPrefixCacheConversation,
        toolSpecs: [ToolSpec]?
    ) async -> [Int]? {
        do {
            return try await container.perform { context in
                try context.tokenizer.applyChatTemplate(
                    messages: conversation.promptMessages,
                    tools: toolSpecs,
                    additionalContext: ["add_generation_prompt": false]
                )
            }
        } catch {
            Log.agent.warning(
                "Stored token sequence measurement failed — error=\(error.localizedDescription)"
            )
            return nil
        }
    }

    /// Restore the boundary snapshot, prefill the residual stored-token suffix,
    /// capture a `.leaf`, and admit it under the given token path. The pure
    /// model-affine executor for a `.fromBoundary` **Leaf Capture Plan**, shared
    /// by the direct-tool and canonical-user modes so both align to the
    /// structured template render, not the raw generated bytes.
    ///
    /// TriAttention-safe by construction: the **Leaf Admission Builder** only
    /// emits `.fromBoundary` when `storedTokens.count > boundary.tokenOffset`, so
    /// the residual is non-empty and no caller-side trim is required. The leaf is
    /// captured at `storedTokens.count` after a clean extension prefill, which
    /// works identically for dense and TriAttention sparse caches because each
    /// cache type's `update(...)` extends its own state at the absolute offset.
    private static func captureStructuredLeafFromBoundary(
        container: ModelContainer,
        storedTokens: [Int],
        boundarySnapshot: HybridCacheSnapshot,
        partitionKey: CachePartitionKey,
        prefillStepSize: Int,
        tokenNDim: Int,
        requestID: UUID,
        prefixCache: PrefixCacheManager,
        diagnosticsContext: PrefixCacheDiagnostics.Context,
        ssdEnabled: Bool,
        generateParameters: GenerateParameters,
        storeStage: String,
        captureStage: String,
        admissionStage: String,
        captureSource: String
    ) async -> AlphaTuner.LeafStore? {
        // The residual is guaranteed non-empty by the builder's offset guard
        // (it only emits `.fromBoundary` when `storedTokens.count > tokenOffset`).
        let boundaryOffset = boundarySnapshot.tokenOffset

        do {
            return try await container.perform { context in
                let triAttentionRestoreContext = Self.makeTriAttentionSnapshotRestoreContext(
                    model: context.model,
                    parameters: generateParameters
                )
                let restoredCache = boundarySnapshot.restore(
                    kvBitsHint: partitionKey.kvBits,
                    kvGroupSizeHint: partitionKey.kvGroupSize,
                    triAttentionRestoreContext: triAttentionRestoreContext
                )

                let residual = Array(storedTokens[boundaryOffset...])
                var prefillParameters = generateParameters
                prefillParameters.checkpointBaseOffset = boundaryOffset
                prefillParameters.configureTriAttentionCachesForPrefill(
                    restoredCache,
                    inputTokenCount: residual.count
                )
                let prefillStart = Date.timeIntervalSinceReferenceDate
                // Qwen3.5 is a `Qwen3_5ForConditionalGeneration` (VLM)
                // whose `prepare` indexes tokens with two axes
                // (`y[0..., ..<step]`) — 1D crashes in `getRopeIndex` on
                // `inputIds.dim(1)`. Pure LLMs use the default
                // `LLMModel.prepare`, which adds the batch dim itself via
                // `.newAxis` and would promote a pre-batched 2D chunk to
                // 3D. Match the processor's original rank.
                let flatInput = MLXArray(residual.map { Int32($0) })
                let inputArr = tokenNDim >= 2
                    ? flatInput.expandedDimensions(axis: 0)
                    : flatInput
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

                guard let leaf = HybridCacheSnapshot.capture(
                    cache: restoredCache,
                    offset: storedTokens.count,
                    type: .leaf
                ) else {
                    diagnosticsContext.logSkip(
                        stage: captureStage,
                        reason: "unsupported-cache-type"
                    )
                    return nil
                }

                let storage = Self.snapshotAdmissionStorage(
                    for: leaf,
                    ssdEnabled: ssdEnabled
                )
                let leafAdmission = SnapshotAdmission.leaf(
                    storedTokens: storedTokens,
                    snapshot: leaf,
                    storage: storage,
                    partitionKey: partitionKey,
                    requestID: requestID
                )
                guard let leafAdmission else {
                    diagnosticsContext.logSkip(
                        stage: admissionStage,
                        reason: "invalid-path",
                        extraFields: [
                            ("offset", "\(leaf.tokenOffset)"),
                            ("storedLen", "\(storedTokens.count)"),
                        ]
                    )
                    return nil
                }

                diagnosticsContext.log(PrefixCacheDiagnostics.CaptureEvent(
                    offset: leaf.tokenOffset,
                    checkpointType: leaf.checkpointType,
                    bytes: leaf.memoryBytes,
                    duringPrefill: false,
                    source: captureSource
                ))
                Log.agent.info(
                    "\(captureSource) captured — offset=\(leaf.tokenOffset) "
                    + "residualTokens=\(residual.count) "
                    + "prefillMs=\(String(format: "%.3f", prefillMs * 1000)) "
                    + "storedLen=\(storedTokens.count)"
                )

                let (diagnostics, postStoreBudgetBytes, postStoreSnapshotBytes) =
                    await MainActor.run { () -> (PrefixCacheManager.StoreDiagnostics, Int, Int) in
                        let d = prefixCache.admit(leafAdmission)
                        return (d, prefixCache.memoryBudgetBytes, prefixCache.totalSnapshotBytes)
                    }
                for event in diagnostics.evictions {
                    diagnosticsContext.log(PrefixCacheDiagnostics.EvictionEvent(event))
                    if let id = event.bodyDroppedSnapshotRefID {
                        diagnosticsContext.log(
                            PrefixCacheDiagnostics.SSDBodyDropEvent(id: id)
                        )
                    }
                }
                for supersession in diagnostics.supersededLeaves {
                    diagnosticsContext.log(PrefixCacheDiagnostics.LeafSupersessionEvent(
                        offset: supersession.offset,
                        snapshotRefID: supersession.bodyDroppedSnapshotRefID
                    ))
                }
                let admissionEvicted = diagnostics.evictions.contains { event in
                    event.offset == leaf.tokenOffset
                        && event.checkpointType == .leaf
                }
                if admissionEvicted {
                    diagnosticsContext.logSkip(
                        stage: admissionStage,
                        reason: "capturedThenEvicted",
                        level: .warning,
                        extraFields: [
                            ("offset", "\(leaf.tokenOffset)"),
                            ("bytes", "\(leaf.memoryBytes)"),
                            ("budgetBytes", "\(postStoreBudgetBytes)"),
                            ("snapshotBytesAfter", "\(postStoreSnapshotBytes)"),
                        ]
                    )
                    Memory.clearCache()
                    return nil
                }

                Memory.clearCache()
                return AlphaTuner.LeafStore(
                    storedTokens: storedTokens,
                    bytes: leaf.memoryBytes
                )
            }
        } catch {
            diagnosticsContext.logSkip(
                stage: storeStage,
                reason: "prefill-threw",
                level: .warning,
                extraFields: [("error", error.localizedDescription)]
            )
            return nil
        }
    }
}
