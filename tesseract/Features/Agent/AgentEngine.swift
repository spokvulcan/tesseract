import Foundation
import Observation
import MLXLLM
import MLXVLM
import MLXLMCommon
import Tokenizers
import os

// MARK: - Sendable conformance for cross-actor transfer

// UserInput is a value-type struct (strings, messages, tool specs) but MLXLMCommon
// doesn't declare Sendable conformance. We need it to cross from @MainActor to LLMActor.
extension UserInput: @retroactive @unchecked Sendable {}

/// Errors thrown by ``AgentEngine`` during generation.
enum AgentEngineError: LocalizedError {
    case modelNotLoaded
    /// Raised when loading a specific model ID fails because its weights are
    /// not present on disk. Carries the offending ID so HTTP handlers can
    /// surface it in a 404 `model_not_found` response.
    case modelNotDownloaded(modelID: String)
    case generationFailed(String)

    var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            "No model is loaded"
        case .modelNotDownloaded(let id):
            "Model '\(id)' is not downloaded"
        case .generationFailed(let description):
            "Generation failed: \(description)"
        }
    }
}

/// Thin MainActor wrapper that publishes UI state and delegates heavy model
/// operations to ``LLMActor``.
///
/// Model downloading is handled by ``ModelDownloadManager``. This engine loads
/// the already-downloaded weights into memory via the actor and verifies them.
@Observable @MainActor
final class AgentEngine {

    private(set) var isModelLoaded = false
    private(set) var isLoading = false
    private(set) var loadingStatus: String = ""
    private(set) var isGenerating = false

    private(set) var agentTokenizer: AgentTokenizer?
    private(set) var triAttentionRuntimeSelection: TriAttentionRuntimeSelection = .disabledDefault

    /// Whether the loaded model's template starts generation inside a `<think>` block.
    private(set) var promptStartsThinking = false

    /// Internal (not private) so unit tests can reach across the actor
    /// boundary to assert load/unload state transitions; production code
    /// never references this directly.
    let llmActor = LLMActor()
    @ObservationIgnored private var activeGenerationID: UUID?
    @ObservationIgnored private var activeGeneration: HTTPServerGenerationStart?

    /// Tracks the most recent `unloadModel` call's detached actor
    /// unload. Callers drain this via `awaitPendingUnload()` when
    /// they need race-free observation of the cleared state
    /// (unit tests, benchmark restart scenarios); production code
    /// is happy with fire-and-forget semantics.
    private var unloadTask: Task<Void, Never>?

    /// How the engine sources its `SSDPrefixCacheConfig` at model load.
    /// `.explicit` wins over `.settings` when both are provided to `init`.
    private enum SSDConfigSource {
        case none
        case settings(SettingsManager)
        case explicit(SSDPrefixCacheConfig)
    }

    private let ssdConfigSource: SSDConfigSource
    private let settingsManager: SettingsManager?

    /// `ssdConfig` takes precedence over `settingsManager`. If both are nil,
    /// the SSD tier is disabled for the lifetime of this engine — the
    /// default shape used by benchmarks and unit tests.
    init(
        settingsManager: SettingsManager? = nil,
        ssdConfig: SSDPrefixCacheConfig? = nil
    ) {
        self.settingsManager = settingsManager
        if let ssdConfig {
            self.ssdConfigSource = .explicit(ssdConfig)
        } else if let settingsManager {
            self.ssdConfigSource = .settings(settingsManager)
        } else {
            self.ssdConfigSource = .none
        }
    }

    /// Resolve the effective SSD config at the moment of the call. Reaches
    /// back into `SettingsManager` via `makeSSDPrefixCacheConfig()` when
    /// the source is `.settings`, so two consecutive loads with a setting
    /// mutated between them produce two different snapshots.
    ///
    /// Internal so unit tests can exercise the precedence rule and the
    /// live-reflection property without a real model load.
    func resolveSSDConfig() -> SSDPrefixCacheConfig? {
        switch ssdConfigSource {
        case .none: return nil
        case .settings(let manager): return manager.makeSSDPrefixCacheConfig()
        case .explicit(let config): return config
        }
    }

    /// Resolve the hidden TriAttention setting at the moment of the call. The
    /// zero-arg/test engine shape stays disabled-by-default.
    func resolveTriAttentionConfig() -> TriAttentionConfiguration {
        settingsManager?.makeTriAttentionConfig() ?? .v1Disabled
    }

    /// Resolve the DFlash draft load config for a target. Returns `nil`
    /// when DFlash is off, the target has no registered draft, or the
    /// draft hasn't been downloaded — silently falls back to AR. Live-
    /// reads `SettingsManager.dflashEnabled` so flipping the toggle takes
    /// effect on the next reload.
    func resolveDFlashLoadConfig(targetModelID: String) -> DFlashLoadConfig? {
        settingsManager?.makeDFlashLoadConfig(targetModelID: targetModelID)
    }

    /// Loads model weights from a local directory into memory and verifies with a 1-token generation.
    ///
    /// - Parameters:
    ///   - directory: Local path containing model weights, config, and tokenizer files
    ///     (as downloaded by ``ModelDownloadManager``).
    ///   - visionMode: When `true`, loads the VLM variant of ParoQuant models (supports
    ///     image attachments but has slower prefill). When `false`, loads the LLM variant
    ///     with fast chunked prefill. Ignored for non-ParoQuant models.
    ///   - triAttention: Explicit TriAttention request. `nil` falls back to
    ///     ``resolveTriAttentionConfig()``.
    ///   - modelID: Catalog ID of the model being loaded. Used to resolve the
    ///     DFlash draft against the *target being loaded* rather than the
    ///     settings-selected model, which can differ (e.g. an HTTP request
    ///     overrides the model). Falls back to settings when `nil`.
    func loadModel(
        from directory: URL,
        visionMode: Bool,
        triAttention: TriAttentionConfiguration? = nil,
        dflashConfig: DFlashLoadConfig? = nil,
        modelID: String? = nil
    ) async throws {
        guard !isModelLoaded, !isLoading else { return }

        isLoading = true
        loadingStatus = "Loading model…"

        do {
            let resolvedDFlash = dflashConfig
                ?? (modelID ?? settingsManager?.selectedAgentModelID).flatMap { id in
                    resolveDFlashLoadConfig(targetModelID: id)
                }
            let (tokenizer, startsThinking) = try await llmActor.loadModel(
                from: directory,
                visionMode: visionMode,
                ssdConfig: resolveSSDConfig(),
                triAttention: triAttention ?? resolveTriAttentionConfig(),
                dflashConfig: resolvedDFlash
            )

            let st = tokenizer.specialTokens
            Log.agent.info(
                "Special tokens resolved — imStart=\(st.imStart) imEnd=\(st.imEnd) "
                + "endOfText=\(st.endOfText) thinkStart=\(st.thinkStart) thinkEnd=\(st.thinkEnd) "
                + "toolCallStart=\(st.toolCallStart) toolCallEnd=\(st.toolCallEnd)"
            )

            agentTokenizer = tokenizer
            promptStartsThinking = startsThinking
            triAttentionRuntimeSelection = await llmActor.currentTriAttentionRuntimeSelection
            isModelLoaded = true
            loadingStatus = ""
            Log.agent.info("Model loaded — promptStartsThinking=\(promptStartsThinking)")
        } catch {
            loadingStatus = ""
            Log.agent.error("Failed to load model: \(error)")
            throw error
        }

        isLoading = false
    }

    /// Streams text generation from a raw prompt string.
    ///
    /// - Parameters:
    ///   - prompt: The full prompt string (caller is responsible for ChatML formatting).
    ///   - parameters: Generation parameters (temperature, maxTokens, etc.).
    /// - Returns: An async stream of ``AgentGeneration`` events.
    func generate(
        prompt: String,
        parameters: AgentGenerateParameters = .default
    ) throws -> AsyncThrowingStream<AgentGeneration, Error> {
        try startPromptInference(
            prompt: prompt,
            parameters: parameters
        ).stream
    }

    /// Streams text generation from the new `LLMMessage`-based conversation format.
    ///
    /// Bridges the new agent loop's message types to the existing MLX inference pipeline.
    /// The system prompt is prepended as a `.system` chat message.
    ///
    /// - Parameters:
    ///   - systemPrompt: The system prompt (prepended as the first message).
    ///   - messages: Conversation history as `LLMMessage` values.
    ///   - tools: Optional tool definitions for the Jinja template's `<tools>` block.
    ///   - parameters: Generation parameters (temperature, maxTokens, etc.).
    /// - Returns: An async stream of ``AgentGeneration`` events.
    func generate(
        systemPrompt: String,
        messages: [LLMMessage],
        tools: [AgentToolDefinition]?,
        parameters: AgentGenerateParameters = .default
    ) throws -> AsyncThrowingStream<AgentGeneration, Error> {
        Log.agent.debug("generate(llmMessages:) — \(messages.count) messages, \(tools?.count ?? 0) tools")
        return try generate(
            systemPrompt: systemPrompt,
            messages: messages,
            toolSpecs: tools?.map { $0.toolSpec },
            parameters: parameters
        )
    }

    /// Generate from structured messages with raw tool specs (schema-only, no execute closures).
    ///
    /// Used by the HTTP server where clients supply tool schemas for prompt rendering
    /// but the server never executes them — the model output tells the client which tools to call.
    func generate(
        systemPrompt: String,
        messages: [LLMMessage],
        toolSpecs: [ToolSpec]?,
        parameters: AgentGenerateParameters = .default
    ) throws -> AsyncThrowingStream<AgentGeneration, Error> {
        Log.agent.debug("generate(toolSpecs:) — \(messages.count) messages, \(toolSpecs?.count ?? 0) tool specs")
        return try startChatInference(
            systemPrompt: systemPrompt,
            messages: messages,
            toolSpecs: toolSpecs,
            parameters: parameters
        ).stream
    }

    /// Start HTTP-server generation, opportunistically reusing a cached prefix when the
    /// request can be canonicalized into the text-based HTTP prefix-cache shape.
    func generateServerTextCompletion(
        modelID: String,
        systemPrompt: String,
        messages: [LLMMessage],
        toolSpecs: [ToolSpec]?,
        prefixCacheConversation: HTTPPrefixCacheConversation?,
        parameters: AgentGenerateParameters = .default
    ) async throws -> HTTPServerGenerationStart {
        guard isModelLoaded else {
            throw AgentEngineError.modelNotLoaded
        }

        if let prefixCacheConversation,
           let start = try await llmActor.generateServerTextCompletion(
                modelID: modelID,
                conversation: prefixCacheConversation,
                toolSpecs: toolSpecs,
                parameters: parameters
           ) {
            Log.agent.info(
                "HTTP completion using prefix-cache path — model=\(modelID) "
                + "cachedTokens=\(start.cachedTokenCount)"
            )
            return startManagedHTTPGeneration(start)
        }

        Log.agent.info(
            "HTTP completion using standard generation path — model=\(modelID) "
            + "toolDefinitions=\(toolSpecs?.count ?? 0) prefixCacheConversation=\(prefixCacheConversation != nil)"
        )

        let input = Self.buildUserInput(
            systemPrompt: systemPrompt,
            messages: messages,
            toolSpecs: toolSpecs,
        )
        return try startManagedHTTPFallbackGeneration(
            input: input,
            toolSpecs: toolSpecs,
            parameters: parameters
        )
    }

    /// Run a closure with the loaded `ModelContainer`. Used by loaded-model
    /// runners that need raw forward-pass access via
    /// `container.perform { context in ... }` outside the agent generation
    /// pipeline (e.g. `HybridCacheCorrectnessRunner`).
    func withModelContainer<T: Sendable>(
        _ body: @Sendable (ModelContainer) async throws -> T
    ) async throws -> T {
        try await llmActor.withModelContainer(body)
    }

    /// Build a `UserInput` from a system prompt, messages, and optional raw tool specs.
    /// Extracted for testability — callers can verify tool specs are forwarded without a loaded model.
    static func buildUserInput(systemPrompt: String, messages: [LLMMessage], toolSpecs: [ToolSpec]?) -> UserInput {
        var chatMessages = [Chat.Message.system(systemPrompt)]
        chatMessages.append(contentsOf: toLLMCommonMessages(messages))
        return UserInput(chat: chatMessages, tools: toolSpecs)
    }

    /// Formats a system prompt + tools through the chat template, returning the raw ChatML string and token count.
    ///
    /// Some model templates (e.g. Qwen3.5) require at least one user message, so a placeholder
    /// is appended and stripped from the output to isolate the system prompt portion.
    func formatRawPrompt(systemPrompt: String, tools: [AgentToolDefinition]?) async throws -> (text: String, tokenCount: Int) {
        let placeholder = "__SYSTEM_PROMPT_END__"
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": systemPrompt],
            ["role": "user", "content": placeholder],
        ]
        let toolSpecs: [ToolSpec]? = tools?.map { $0.toolSpec }
        var result = try await llmActor.formatRawPromptWithCount(messages: messages, tools: toolSpecs)

        // Strip everything from the placeholder user message onward
        if let range = result.text.range(of: placeholder) {
            // Back up to the im_start tag for the user turn
            let beforePlaceholder = result.text[..<range.lowerBound]
            if let userTagRange = beforePlaceholder.range(of: "<|im_start|>user", options: .backwards) {
                result.text = String(result.text[..<userTagRange.lowerBound])
            } else {
                result.text = String(beforePlaceholder)
            }
        }

        // Recount tokens for the trimmed text
        if let tokenizer = agentTokenizer {
            result.tokenCount = await tokenizer.encode(result.text, addSpecialTokens: false).count
        }
        return result
    }

    /// Cancels any in-progress generation.
    func cancelGeneration() {
        activeGeneration?.cancel()
        activeGeneration = nil
        activeGenerationID = nil
    }

    /// Cancels any in-progress generation and waits for the underlying MLX work
    /// to stop touching model state before returning.
    func cancelGenerationAndWait() async {
        let activeGeneration = activeGeneration
        self.activeGeneration = nil
        activeGenerationID = nil
        activeGeneration?.cancel()
        await activeGeneration?.waitForCompletion()
        isGenerating = false
    }

    /// Frees unreferenced MLX buffers (safe to call between tool rounds).
    func clearMemoryCache() async {
        await llmActor.clearMemoryCache()
    }

    /// Returns current MLX memory usage in MB.
    func memoryStats() async -> (activeMB: Float, peakMB: Float) {
        await llmActor.memoryStats()
    }

    /// Snapshot of the live prefix-cache state, or `nil` if the cache has
    /// not yet been instantiated. Observer hook intended for the loaded-
    /// model E2E runner; production code should not depend on it.
    func prefixCacheStats() async -> PrefixCacheManager.CacheStats? {
        await llmActor.prefixCacheStats()
    }

    func promptCacheTelemetrySnapshot() async -> PromptCacheTelemetrySnapshot? {
        await llmActor.promptCacheTelemetrySnapshot()
    }

    /// Override the prefix-cache memory budget, triggering an immediate
    /// eviction pass. Observer hook intended for the loaded-model E2E
    /// runner; production code should not call this.
    func setPrefixCacheBudgetBytes(_ bytes: Int) async {
        await llmActor.setPrefixCacheBudgetBytes(bytes)
    }

    /// Block until any pending SSD-tier writes drain and the
    /// manifest is persisted. Exposed as a standalone primitive for
    /// callers that want to observe durability without tearing down
    /// the engine; `unloadModel()` already runs this first on the
    /// detached unload task.
    func flushPrefixCache() async {
        await llmActor.flushPrefixCache()
    }

    /// Releases the model from memory. The detached unload task
    /// drains the SSD writer + persists the manifest BEFORE
    /// clearing actor state so production unload/restart flows
    /// (normal model switching, app shutdown, `InferenceArbiter.unload`)
    /// preserve any in-flight prefix-cache snapshots. Callers that
    /// need to observe the drain synchronously await
    /// `awaitPendingUnload()`.
    func unloadModel() {
        cancelGeneration()
        agentTokenizer = nil
        promptStartsThinking = false
        triAttentionRuntimeSelection = .disabledDefault
        isModelLoaded = false
        loadingStatus = ""
        unloadTask = Task { [llmActor] in
            await llmActor.flushPrefixCache()
            await llmActor.unloadModel()
        }
        Log.agent.info("Model unloaded")
    }

    /// Wait for the most recent `unloadModel` call's detached actor
    /// unload to complete. Used by unit tests and benchmark restart
    /// scenarios that need to observe the cleared state race-free;
    /// production hot paths use fire-and-forget `unloadModel()`.
    func awaitPendingUnload() async {
        await unloadTask?.value
    }

    // MARK: - Private

    @discardableResult
    private func registerActiveGeneration(
        _ start: HTTPServerGenerationStart,
        id: UUID
    ) -> HTTPServerGenerationStart {
        activeGenerationID = id
        activeGeneration = start
        return start
    }

    private func clearActiveGeneration(id: UUID) {
        guard activeGenerationID == id else { return }
        activeGenerationID = nil
        activeGeneration = nil
    }

    /// Shared generation logic for both prompt-based and message-based entry points.
    private func startGeneration(
        input: UserInput,
        toolSpecs: [ToolSpec]?,
        parameters: AgentGenerateParameters
    ) throws -> AsyncThrowingStream<AgentGeneration, Error> {
        try startManagedGeneration(
            input: input,
            toolSpecs: toolSpecs,
            parameters: parameters
        ).stream
    }

    private func startManagedGeneration(
        input: UserInput,
        toolSpecs: [ToolSpec]?,
        parameters: AgentGenerateParameters
    ) throws -> HTTPServerGenerationStart {
        guard isModelLoaded else {
            throw AgentEngineError.modelNotLoaded
        }

        let actor = llmActor
        return wrapManagedGeneration(
            input: input,
            toolSpecs: toolSpecs,
            parameters: parameters
        ) {
            try await actor.startRawGeneration(
                input: input,
                toolSpecs: toolSpecs,
                parameters: parameters
            )
        }
    }

    private func startManagedHTTPFallbackGeneration(
        input: UserInput,
        toolSpecs: [ToolSpec]?,
        parameters: AgentGenerateParameters
    ) throws -> HTTPServerGenerationStart {
        guard isModelLoaded else {
            throw AgentEngineError.modelNotLoaded
        }

        let actor = llmActor
        return wrapManagedGeneration(
            input: input,
            toolSpecs: toolSpecs,
            parameters: parameters
        ) {
            try await actor.startRawGeneration(
                input: input,
                toolSpecs: toolSpecs,
                parameters: parameters
            )
        }
    }

    func wrapManagedGeneration(
        cachedTokenCount: Int = 0,
        input: UserInput? = nil,
        toolSpecs: [ToolSpec]? = nil,
        parameters: AgentGenerateParameters = .default,
        launch: @escaping @Sendable () async throws -> HTTPServerRawGenerationStart
    ) -> HTTPServerGenerationStart {
        isGenerating = true

        let (stream, continuation) = AsyncThrowingStream.makeStream(of: AgentGeneration.self)
        let startsThinking = promptStartsThinking
        let generationID = UUID()
        struct RawGenerationHandle: Sendable {
            let cancel: @Sendable () -> Void
            let waitForCompletion: @Sendable () async -> Void
        }
        struct RawGenerationState {
            var handle: RawGenerationHandle?
            var cancelIssued = false
        }
        let rawState = OSAllocatedUnfairLock<RawGenerationState>(initialState: .init())

        let actor = llmActor
        let safeguardConfig = parameters.thinkingSafeguard
        parameters.warnIfThinkingLoopRiskElevated(startsThinking: startsThinking)

        let task = Task { @MainActor [weak self] in
            defer {
                self?.isGenerating = false
                self?.clearActiveGeneration(id: generationID)
            }

            do {
                try Task.checkCancellation()

                var currentStart = try await launch()
                let initialCancel = currentStart.cancel
                let initialWait = currentStart.waitForCompletion
                rawState.withLock {
                    $0.handle = .init(
                        cancel: initialCancel,
                        waitForCompletion: initialWait
                    )
                }
                try Task.checkCancellation()

                var parser = ToolCallParser(startsInsideThinkBlock: startsThinking)
                var rawChunkParts: [String] = []
                var libraryParsedToolCalls = false
                let safeguard = ThinkingSafeguardObserver(config: safeguardConfig)

                // Forward a parser event, honoring the safeguard. Returns an
                // intervention payload when the safeguard fired, otherwise nil.
                func yieldOrIntervene(
                    _ event: ToolCallParser.Event
                ) -> (safePrefix: String, reason: ThinkingRepetitionDetector.Reason)? {
                    switch safeguard.observe(parserEvent: event) {
                    case .forward:
                        continuation.yield(AgentGeneration(parserEvent: event))
                        return nil
                    case .intervene(let safe, let reason):
                        // Replace the degen thinking with the clean prefix, emit the
                        // hand-off phrase, close `</think>`. Downstream consumers
                        // reset their thinking accumulators on `.thinkTruncate`.
                        continuation.yield(.thinkTruncate(safePrefix: safe))
                        continuation.yield(.thinking(safeguardConfig.injectionMessage))
                        continuation.yield(.thinkEnd)
                        Log.agent.warning(
                            "Thinking-loop intervention — reason=\(reason.rawValue) "
                            + "safe_prefix_chars=\(safe.count) generation_id=\(generationID.uuidString)"
                        )
                        return (safe, reason)
                    }
                }

                generationLoop: while true {
                    var intervention: (safePrefix: String, reason: ThinkingRepetitionDetector.Reason)? = nil

                    for await generation in currentStart.stream {
                        try Task.checkCancellation()

                        switch generation {
                        case .chunk(let text):
                            rawChunkParts.append(text)
                            // When the library's ToolCallProcessor handles tool calls
                            // (xmlFunction format), it leaks wrapper tags as chunks.
                            // Skip app-level parsing to avoid false "malformed tool
                            // call" warnings.
                            if !libraryParsedToolCalls {
                                for parserEvent in parser.processChunk(text) {
                                    if let fired = yieldOrIntervene(parserEvent) {
                                        intervention = fired
                                        break
                                    }
                                }
                                if intervention != nil { break }
                            }

                        case .info(let completionInfo):
                            // Flush any buffered text before emitting info
                            if !libraryParsedToolCalls {
                                for event in parser.finalize() {
                                    if let fired = yieldOrIntervene(event) {
                                        intervention = fired
                                        break
                                    }
                                }
                            }
                            if intervention != nil { break }
                            let info = AgentGeneration.Info(
                                promptTokenCount: completionInfo.promptTokenCount,
                                generationTokenCount: completionInfo.generationTokenCount,
                                promptTime: completionInfo.promptTime,
                                generateTime: completionInfo.generateTime,
                                stopReason: completionInfo.stopReason
                            )
                            continuation.yield(.info(info))
                            Log.agent.info(
                                "Generation complete — \(completionInfo.generationTokenCount) tokens, "
                                + "\(String(format: "%.1f", info.tokensPerSecond)) tok/s, "
                                + "stopReason=\(describeStopReason(completionInfo.stopReason))"
                            )
                            let rawChunks = rawChunkParts.joined()
                            Log.agent.debug("Raw library chunks (after ToolCallProcessor):\n\(rawChunks)")

                        case .toolCall(let call):
                            // Vendor library parsed this tool call — yield directly
                            libraryParsedToolCalls = true
                            Log.agent.info("Library parsed tool call: \(call.function.name)(\(call.function.arguments))")
                            continuation.yield(.toolCall(call))

                        case .toolCallBufferDelta(let delta):
                            // Vendor is buffering `<tool_call>…</tool_call>` and
                            // just appended `delta` characters. Forward as a
                            // progressive UI event; the final `.toolCall` still
                            // fires atomically once the close tag parses.
                            libraryParsedToolCalls = true
                            continuation.yield(.toolCallDelta(
                                name: nil,
                                argumentsDelta: delta
                            ))
                        }
                    }

                    if let fired = intervention, let originalInput = input {
                        // Cancel the current raw handle and wait for it to stop.
                        let handleToCancel = rawState.withLock {
                            (state: inout RawGenerationState) -> RawGenerationHandle? in
                            let h = state.handle
                            state.cancelIssued = true
                            return h
                        }
                        handleToCancel?.cancel()
                        await handleToCancel?.waitForCompletion()

                        // Kick off the continuation from the clean safe prefix.
                        do {
                            let newStart = try await actor.startThinkingContinuationRaw(
                                originalInput: originalInput,
                                safeThinkingPrefix: fired.safePrefix,
                                injection: safeguardConfig.continuationHandOff,
                                toolSpecs: toolSpecs,
                                parameters: parameters
                            )
                            rawState.withLock {
                                $0.handle = .init(
                                    cancel: newStart.cancel,
                                    waitForCompletion: newStart.waitForCompletion
                                )
                                $0.cancelIssued = false
                            }
                            currentStart = newStart
                            // Continuation starts OUTSIDE `<think>` — reset
                            // the parser so its output is classified as text.
                            // Do NOT reset the safeguard: `interventionsIssued
                            // >= limit` already blocks re-triggering, and
                            // resetting would erase the "we intervened" flag
                            // that any downstream consumer relies on.
                            parser = ToolCallParser(startsInsideThinkBlock: false)
                            continue generationLoop
                        } catch {
                            Log.agent.error(
                                "Thinking-safeguard continuation failed: \(error.localizedDescription) — finishing with truncated response"
                            )
                            break generationLoop
                        }
                    }

                    // Natural end (no intervention on this round).
                    break generationLoop
                }

                // Only warn if raw output has tool call markers AND the library didn't parse any
                if !libraryParsedToolCalls {
                    let rawChunks = rawChunkParts.joined()
                    if rawChunks.contains("tool_call") || rawChunks.contains("<function") {
                        Log.agent.warning("Raw output contains tool call markers but no .toolCall events were emitted by library")
                    }
                }

                // Flush any remaining buffered text
                if !libraryParsedToolCalls {
                    for event in parser.finalize() {
                        _ = yieldOrIntervene(event)
                    }
                }

                await currentStart.waitForCompletion()
                continuation.finish()
            } catch is CancellationError {
                let cancellation = rawState.withLock { state -> (handle: RawGenerationHandle?, shouldCancel: Bool) in
                    guard let handle = state.handle else {
                        return (nil, false)
                    }
                    if state.cancelIssued {
                        return (handle, false)
                    }
                    state.cancelIssued = true
                    return (handle, true)
                }
                if cancellation.shouldCancel {
                    cancellation.handle?.cancel()
                }
                await cancellation.handle?.waitForCompletion()
                continuation.finish()
            } catch {
                let cancellation = rawState.withLock { state -> (handle: RawGenerationHandle?, shouldCancel: Bool) in
                    guard let handle = state.handle else {
                        return (nil, false)
                    }
                    if state.cancelIssued {
                        return (handle, false)
                    }
                    state.cancelIssued = true
                    return (handle, true)
                }
                if cancellation.shouldCancel {
                    cancellation.handle?.cancel()
                }
                await cancellation.handle?.waitForCompletion()
                continuation.finish(throwing: AgentEngineError.generationFailed(
                    error.localizedDescription
                ))
            }
        }

        let start = HTTPServerGenerationStart(
            stream: stream,
            cachedTokenCount: cachedTokenCount,
            cancel: {
                let cancellation = rawState.withLock { state -> (handle: RawGenerationHandle?, shouldCancel: Bool) in
                    guard let handle = state.handle else {
                        return (nil, false)
                    }
                    if state.cancelIssued {
                        return (handle, false)
                    }
                    state.cancelIssued = true
                    return (handle, true)
                }
                if cancellation.shouldCancel {
                    cancellation.handle?.cancel()
                }
                task.cancel()
            },
            waitForCompletion: {
                _ = await task.result
            }
        )
        continuation.onTermination = { _ in start.cancel() }
        return registerActiveGeneration(start, id: generationID)
    }

    private func startManagedHTTPGeneration(
        _ start: HTTPServerGenerationStart
    ) -> HTTPServerGenerationStart {
        isGenerating = true

        let (stream, continuation) = AsyncThrowingStream.makeStream(of: AgentGeneration.self)
        let generationID = UUID()

        let task = Task { @MainActor [weak self] in
            defer {
                self?.isGenerating = false
                self?.clearActiveGeneration(id: generationID)
            }

            do {
                for try await event in start.stream {
                    try Task.checkCancellation()
                    continuation.yield(event)
                }
                continuation.finish()
            } catch is CancellationError {
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }

        let managedStart = HTTPServerGenerationStart(
            stream: stream,
            cachedTokenCount: start.cachedTokenCount,
            cancel: {
                task.cancel()
                start.cancel()
            },
            waitForCompletion: {
                _ = await task.result
                await start.waitForCompletion()
            },
            diagnostics: start.diagnostics
        )
        continuation.onTermination = { _ in managedStart.cancel() }
        return registerActiveGeneration(managedStart, id: generationID)
    }
}

@MainActor
extension AgentEngine: ServerInferenceEngine {
    func startPromptInference(
        prompt: String,
        parameters: AgentGenerateParameters
    ) throws -> HTTPServerGenerationStart {
        try startManagedGeneration(
            input: UserInput(prompt: prompt),
            toolSpecs: nil,
            parameters: parameters
        )
    }

    func startChatInference(
        systemPrompt: String,
        messages: [LLMMessage],
        toolSpecs: [ToolSpec]?,
        parameters: AgentGenerateParameters
    ) throws -> HTTPServerGenerationStart {
        let input = Self.buildUserInput(
            systemPrompt: systemPrompt,
            messages: messages,
            toolSpecs: toolSpecs
        )
        return try startManagedGeneration(
            input: input,
            toolSpecs: toolSpecs,
            parameters: parameters
        )
    }

    func startServerChatInference(
        modelID: String,
        systemPrompt: String,
        messages: [LLMMessage],
        toolSpecs: [ToolSpec]?,
        prefixCacheConversation: HTTPPrefixCacheConversation?,
        parameters: AgentGenerateParameters
    ) async throws -> HTTPServerGenerationStart {
        try await generateServerTextCompletion(
            modelID: modelID,
            systemPrompt: systemPrompt,
            messages: messages,
            toolSpecs: toolSpecs,
            prefixCacheConversation: prefixCacheConversation,
            parameters: parameters
        )
    }
}
