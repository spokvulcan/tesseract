import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXNN
import Tokenizers
import os

/// Actor-isolated wrapper that owns the LLM model and runs inference off the MainActor.
///
/// Follows the same pattern as `WhisperKitSpeechRecognizer` below `TranscriptionEngine`:
/// the `@MainActor` ``AgentEngine`` publishes UI state while delegating
/// heavy model operations to this actor.
///
/// The actor carries two narrow surfaces: model lifecycle plus raw generation
/// for the agent chat path, and the **Server Completion** entry for the HTTP
/// server's cache-aware path. The server execution itself lives in the
/// actor-confined ``ServerCompletion`` module (CONTEXT.md → Server completion,
/// ADR-0006), installed at model load and dropped at unload. The
/// thinking-continuation primitives stay here as actor primitives shared by
/// both paths; the module composes them for the safeguard's continuation swap.
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
    /// the Server Completion module falls back to a conservative 3 GiB budget
    /// so pre-load paths and tests retain deterministic behavior. Machines too
    /// small to fit model + headroom clamp to 0 and rely on the fallback.
    enum Defaults {
        static let cacheLimitMB = 2048
        static let prefixCacheHeadroomBytes = 20 * 1024 * 1024 * 1024 // 20 GiB
        /// Fallback budget used before load-time sizing runs. Each snapshot
        /// costs ~200–600 MiB depending on context length, so 3 GiB fits
        /// ~5–15 snapshots for typical Qwen3.5 workloads.
        static let fallbackPrefixCacheMemoryBudgetBytes = 3 * 1024 * 1024 * 1024 // 3 GiB
    }

    private var modelContainer: ModelContainer?
    private(set) var agentTokenizer: AgentTokenizer?

    /// The **Server Completion** module — actor-confined, non-`Sendable`,
    /// owning the cache-aware HTTP completion execution, the prefix cache,
    /// and the load-time snapshot facts. Created lazily (pre-load admin
    /// callers like the E2E budget/alpha tooling) or by the load path;
    /// dropped wholesale at `unloadModel()`.
    private var serverCompletion: ServerCompletion?

    var isLoaded: Bool { modelContainer != nil }

    /// Internal read-only accessor for the load-time SSD config snapshot.
    /// Production reads happen inside the Server Completion module; this
    /// accessor exists so tests can assert the load/unload lifecycle across
    /// the actor boundary.
    var currentSSDConfigForTesting: SSDPrefixCacheConfig? {
        serverCompletion?.ssdConfig
    }

    /// Internal read-only accessor for the load-time model fingerprint.
    var currentModelFingerprintForTesting: String? {
        serverCompletion?.modelFingerprint
    }

    /// Internal read-only accessor for the load-time model identity.
    var currentModelIdentityForTesting: ModelIdentity? {
        serverCompletion?.modelIdentity
    }

    /// Test-only: install a `ModelIdentity` without a full model load, so
    /// tests can exercise identity-derived wiring (notably the prefix cache's
    /// FLOP profile) at the actor seam. Production identity is installed only
    /// by `installLoadTimeState`.
    func setModelIdentityForTesting(_ identity: ModelIdentity?) {
        ensureServerCompletion().setModelIdentityForTesting(identity)
    }

    /// Test-only: the eviction configuration of the live prefix cache, or
    /// `nil` if the cache hasn't been built.
    func currentEvictionConfigForTesting() async -> EvictionConfiguration? {
        guard let serverCompletion else { return nil }
        return await serverCompletion.currentEvictionConfigForTesting(on: self)
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
        ssdConfig: SSDPrefixCacheConfig? = nil
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

        installLoadTimeState(
            modelIdentity: identity,
            fingerprint: fingerprint,
            ssdConfig: ssdConfig
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
        // with the post-generation leaf-capture residual prefill path. So in
        // non-vision mode we force the text-only `MLXLLM.Qwen35Model` by
        // invoking the LLM factory directly.
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

        // The standard path bypasses `ServerCompletion.start`, so it preempts
        // the background speculative prefill itself — otherwise its first
        // `container.perform` could queue behind background chunks. Awaited:
        // the pass settles (bounded by ~one chunk plus one capture) so its
        // partial-leaf admission lands before this generation touches the
        // container, preserved for future cache-aware requests.
        await preemptServerSpeculativePrefill()

        let genParams = Self.makeGenerateParameters(from: parameters)
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
            // VLM-class models (2D token tensors) run upstream `prepare` as a
            // single forward pass over the whole prompt; chunk text-only
            // prompts through the app's prefill driver to keep peak memory
            // bounded (ADR-0006). Image-bearing inputs stay single-shot.
            let iterator: TokenIterator
            if prepared.text.tokens.ndim >= 2,
               prepared.image == nil, prepared.video == nil,
               prepared.text.tokens.dim(-1) > genParams.prefillStepSize {
                var cache = context.model.newCache(parameters: genParams)
                let warmed = try PrefillExecutor.run(
                    model: context.model,
                    text: prepared.text,
                    cache: cache,
                    prefillStepSize: genParams.prefillStepSize
                )
                iterator = try PrefillExecutor.makeIterator(
                    model: context.model,
                    fullText: prepared.text,
                    remainder: warmed.remainder,
                    cache: &cache,
                    parameters: genParams
                )
            } else {
                iterator = try TokenIterator(
                    input: prepared,
                    model: context.model,
                    cache: nil,
                    parameters: genParams
                )
            }
            let prefillMs = (Date.timeIntervalSinceReferenceDate - prefillStarted) * 1000
            await progressHandler?(.prefillFinished(.init(
                promptTokens: promptTokenCount,
                cachedTokens: 0,
                newTokensToPrefill: promptTokenCount,
                prefillMs: prefillMs
            )))
            let (stream, completion) = TokenGenerationLoop.start(
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
    /// prefill by the Server Completion module), so re-tokenizing the
    /// `UserInput` is unnecessary work — encode the appended hand-off text
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
        let genParams = Self.makeGenerateParameters(from: parameters)
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
        let genParams = Self.makeGenerateParameters(from: parameters)
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
        let continuedText = LMInput.Text(tokens: tokenArr, mask: nil)

        // Same VLM-class chunking as `startRawGeneration`: upstream's 2D
        // `prepare` is single-shot, so pre-chunk long token-only prompts
        // through the app driver (ADR-0006).
        let iterator: TokenIterator
        if tokenNDim >= 2, combined.count > parameters.prefillStepSize {
            var cache = context.model.newCache(parameters: parameters)
            let warmed = try PrefillExecutor.run(
                model: context.model,
                text: continuedText,
                cache: cache,
                prefillStepSize: parameters.prefillStepSize
            )
            iterator = try PrefillExecutor.makeIterator(
                model: context.model,
                fullText: continuedText,
                remainder: warmed.remainder,
                cache: &cache,
                parameters: parameters
            )
        } else {
            iterator = try TokenIterator(
                input: LMInput(text: continuedText),
                model: context.model,
                cache: nil,
                parameters: parameters
            )
        }
        let (stream, completion) = TokenGenerationLoop.start(
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

    /// Start the HTTP text-based prefix-cache path for `/v1/chat/completions` —
    /// the **Server Completion** entry (the dispatcher's cache-aware arm).
    /// Execution lives in the actor-confined ``ServerCompletion`` module.
    ///
    /// The **Completion Route** guarantees the conversation shape is servable
    /// before this is called — the module never sees a request it cannot
    /// serve (ADR-0006).
    func startServerCompletion(
        modelID: String,
        conversation: HTTPPrefixCacheConversation,
        toolSpecs: [ToolSpec]?,
        parameters: AgentGenerateParameters,
        renderContext: TemplateRenderContext = .canonical,
        progressHandler: ServerInferenceProgressHandler? = nil
    ) async throws -> HTTPServerGenerationStart {
        guard let container = modelContainer else {
            throw AgentEngineError.modelNotLoaded
        }
        return try await ensureServerCompletion().start(
            on: self,
            container: container,
            modelID: modelID,
            conversation: conversation,
            toolSpecs: toolSpecs,
            parameters: parameters,
            renderContext: renderContext,
            progressHandler: progressHandler
        )
    }

    /// Template-declared render flags of the loaded model (issue #98), read
    /// from the **Server Completion** module's load-time identity snapshot.
    /// Empty before a load installs one.
    func loadedDeclaredTemplateFlags() -> Set<TemplateRenderFlag> {
        serverCompletion?.modelIdentity?.declaredTemplateFlags ?? []
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

    /// Releases the model from memory: drains the **Server Completion**
    /// module's active completion (cancel-and-await, so no in-flight server
    /// request can touch model state during teardown), then drops the module
    /// (prefix cache, load-time snapshot facts, admin state) and the
    /// container. The GPU lease remains the primary guard; this drain is the
    /// in-actor backstop (ADR-0006).
    func unloadModel() async {
        let containerAtEntry = modelContainer
        if let serverCompletion {
            await serverCompletion.drainActiveCompletion(on: self)
        }
        // The drain suspends, so a concurrent `loadModel` can interleave and
        // install a fresh container (actor reentrancy). Tear down only the
        // state this call set out to release — never a newer load's.
        guard modelContainer === containerAtEntry else {
            Log.agent.info(
                "unloadModel skipped teardown — a newer load replaced the container during the drain"
            )
            return
        }
        modelContainer = nil
        agentTokenizer = nil
        serverCompletion = nil
    }

    /// Cancel-and-await the active **Server Completion**, leaving the model
    /// loaded. Callers that flush the prefix cache before unloading use this
    /// to stop in-flight server generations first, so snapshot admissions
    /// cannot land after the SSD manifest has been persisted.
    func drainServerCompletion() async {
        guard let serverCompletion else { return }
        await serverCompletion.drainActiveCompletion(on: self)
    }

    /// Natural-finish hook from the Server Completion driving task: drop the
    /// registry slot for `requestID` once its stream has fully completed.
    func clearFinishedServerCompletion(_ requestID: UUID) {
        serverCompletion?.clearFinishedCompletion(requestID, on: self)
    }

    /// Schedule the post-answer **Speculative Canonical Prefill** for the
    /// server completion that just finished (issue #76, ADR-0009). No-ops
    /// when the model unloaded or the module decides it is no longer idle.
    func scheduleServerSpeculativePrefill(
        seed: SpeculativeCanonicalPrefill.Seed,
        entryDrainGeneration: Int
    ) async {
        guard let container = modelContainer, let serverCompletion else {
            seed.discard()
            return
        }
        await serverCompletion.scheduleSpeculativePrefill(
            seed: seed,
            container: container,
            entryDrainGeneration: entryDrainGeneration,
            on: self
        )
    }

    /// Preempt (cancel-and-await) the background speculative prefill: the
    /// standard raw path calls this at entry, and the drain suite pins the
    /// settle handshake through it. No-ops when no pass is live.
    func preemptServerSpeculativePrefill() async {
        await serverCompletion?.preemptSpeculativePrefill(on: self)
    }

    /// Natural-finish hook from the speculative prefill task: drop its slot
    /// once the pass has fully finished.
    func clearFinishedSpeculativeServerPrefill(_ id: UUID) {
        serverCompletion?.clearFinishedSpeculativePrefill(id, on: self)
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
        guard let serverCompletion else { return }
        await serverCompletion.flushPrefixCache(on: self)
    }

    /// Returns current MLX memory usage in MB.
    func memoryStats() -> (activeMB: Float, peakMB: Float) {
        (Float(Memory.activeMemory) / 1e6, Float(Memory.peakMemory) / 1e6)
    }

    // MARK: - Prefix Cache Admin (forwarded to the Server Completion module)

    /// Snapshot of the live prefix-cache state, or `nil` if the cache hasn't
    /// been instantiated. Used by the loaded-model E2E runner to verify
    /// branch-point capture and survival.
    func prefixCacheStats() async -> PrefixCacheManager.CacheStats? {
        guard let serverCompletion else { return nil }
        return await serverCompletion.prefixCacheStats(on: self)
    }

    func promptCacheTelemetrySnapshot() async -> PromptCacheTelemetrySnapshot? {
        guard let serverCompletion else { return nil }
        return await serverCompletion.promptCacheTelemetrySnapshot(on: self)
    }

    /// Override the prefix-cache memory budget at runtime. Used by the
    /// loaded-model E2E runner to deliberately trigger eviction pressure.
    func setPrefixCacheBudgetBytes(_ bytes: Int) async {
        await ensureServerCompletion().setPrefixCacheBudgetBytes(bytes, on: self)
    }

    /// Override the prefix-cache eviction weighting (`alpha`). Used by the
    /// loaded-model E2E runner to force F/B-weighted eviction for the
    /// branch-point survival check. Production code should not call this;
    /// the `AlphaTuner` owns `alpha` after warmup.
    func setEvictionAlpha(_ alpha: Double) async {
        await ensureServerCompletion().setEvictionAlpha(alpha, on: self)
    }

    /// Current prefix-cache eviction weighting (`alpha`), or `nil` if the
    /// cache hasn't been built. Symmetric with `setEvictionAlpha`, so the E2E
    /// runner can save and restore the weighting around its forced-pressure
    /// step instead of leaving the cache mutated.
    func evictionAlpha() async -> Double? {
        guard let serverCompletion else { return nil }
        return await serverCompletion.evictionAlpha(on: self)
    }

    /// Extract a flat `[Int]` token sequence from an MLXArray that may be 1D `[seq]`
    /// (LLM-only) or 2D `[batch, seq]` (VLM). Uses the first batch element for 2D.
    /// Shared with the Server Completion module's tokenize step.
    nonisolated static func extractTokenSequence(_ tokens: MLXArray) -> [Int] {
        if tokens.ndim <= 1 {
            return tokens.asArray(Int.self)
        } else {
            return tokens[0].asArray(Int.self)
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

    /// The Server Completion module, created on first use. Pre-load admin
    /// callers (the E2E budget/alpha tooling) get an empty module whose
    /// prefix cache runs on the fallback FLOP profile until a load installs
    /// the real identity — the same semantics the actor-resident fields had.
    private func ensureServerCompletion() -> ServerCompletion {
        if let existing = serverCompletion { return existing }
        let module = ServerCompletion()
        serverCompletion = module
        return module
    }

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
        ensureServerCompletion().installLoadedModelFacts(
            promptStartsThinking: startsThinking,
            modelWeightBytes: modelWeightBytes,
            prefixCacheBudgetBytes: prefixCacheBudgetBytes
        )
        return (tokenizer, startsThinking)
    }

    /// Single install site for per-load snapshot state, delegated to the
    /// Server Completion module. Called from `loadModel` before the container
    /// load is attempted so the state is visible even on failed loads; this
    /// lets the unit suite exercise the full config-resolution chain via a
    /// fake directory that trips the container load. The unload path drops
    /// the module wholesale.
    private func installLoadTimeState(
        modelIdentity: ModelIdentity,
        fingerprint: String,
        ssdConfig: SSDPrefixCacheConfig?
    ) {
        ensureServerCompletion().installLoadTimeState(
            modelIdentity: modelIdentity,
            fingerprint: fingerprint,
            ssdConfig: ssdConfig
        )
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

    /// Shared by the actor's raw-generation entries and the Server Completion
    /// module's cache-aware path.
    nonisolated static func makeGenerateParameters(
        from parameters: AgentGenerateParameters
    ) -> GenerateParameters {
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
            prefillStepSize: parameters.prefillStepSize
        )
    }

    /// Test-only: register a fake server-completion handle so the unit suite
    /// can exercise the unload drain contract without a loaded model.
    func registerServerCompletionForTesting(
        _ handle: HTTPServerGenerationStart,
        id: UUID
    ) {
        ensureServerCompletion().registerActiveCompletionForTesting(
            handle, id: id, on: self
        )
    }

    /// Test-only: occupy the speculative-prefill slot so the unit suite can
    /// exercise its drain contract without a loaded model.
    func registerSpeculativePrefillForTesting(
        _ task: Task<Void, Never>,
        id: UUID
    ) {
        ensureServerCompletion().registerSpeculativePrefillForTesting(
            task, id: id, on: self
        )
    }

}

extension LLMActor: ServerCompletionStarting {}
