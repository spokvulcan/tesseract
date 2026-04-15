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

    /// Whether the loaded model's template starts generation inside a `<think>` block.
    private(set) var promptStartsThinking = false

    /// Internal (not private) so unit tests can reach across the actor
    /// boundary to assert load/unload state transitions; production code
    /// never references this directly.
    let llmActor = LLMActor()
    private var generationTask: Task<Void, Never>?

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

    /// `ssdConfig` takes precedence over `settingsManager`. If both are nil,
    /// the SSD tier is disabled for the lifetime of this engine — the
    /// default shape used by benchmarks and unit tests.
    init(
        settingsManager: SettingsManager? = nil,
        ssdConfig: SSDPrefixCacheConfig? = nil
    ) {
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

    /// Loads model weights from a local directory into memory and verifies with a 1-token generation.
    ///
    /// - Parameters:
    ///   - directory: Local path containing model weights, config, and tokenizer files
    ///     (as downloaded by ``ModelDownloadManager``).
    ///   - visionMode: When `true`, loads the VLM variant of ParoQuant models (supports
    ///     image attachments but has slower prefill). When `false`, loads the LLM variant
    ///     with fast chunked prefill. Ignored for non-ParoQuant models.
    func loadModel(from directory: URL, visionMode: Bool) async throws {
        guard !isModelLoaded, !isLoading else { return }

        isLoading = true
        loadingStatus = "Loading model…"

        do {
            let (tokenizer, startsThinking) = try await llmActor.loadModel(
                from: directory,
                visionMode: visionMode,
                ssdConfig: resolveSSDConfig()
            )

            let st = tokenizer.specialTokens
            Log.agent.info(
                "Special tokens resolved — imStart=\(st.imStart) imEnd=\(st.imEnd) "
                + "endOfText=\(st.endOfText) thinkStart=\(st.thinkStart) thinkEnd=\(st.thinkEnd) "
                + "toolCallStart=\(st.toolCallStart) toolCallEnd=\(st.toolCallEnd)"
            )

            agentTokenizer = tokenizer
            promptStartsThinking = startsThinking
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
        try startGeneration(input: UserInput(prompt: prompt), parameters: parameters)
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
        let input = Self.buildUserInput(systemPrompt: systemPrompt, messages: messages, toolSpecs: toolSpecs)
        return try startGeneration(input: input, parameters: parameters)
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
        return try await startManagedHTTPFallbackGeneration(
            input: input,
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
        generationTask?.cancel()
        generationTask = nil
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

    /// Shared generation logic for both prompt-based and message-based entry points.
    private func startGeneration(
        input: UserInput,
        parameters: AgentGenerateParameters
    ) throws -> AsyncThrowingStream<AgentGeneration, Error> {
        try startManagedGeneration(input: input, parameters: parameters).stream
    }

    private func startManagedGeneration(
        input: UserInput,
        parameters: AgentGenerateParameters
    ) throws -> HTTPServerGenerationStart {
        guard isModelLoaded else {
            throw AgentEngineError.modelNotLoaded
        }

        isGenerating = true

        let (stream, continuation) = AsyncThrowingStream.makeStream(of: AgentGeneration.self)
        let actor = llmActor
        let startsThinking = promptStartsThinking

        let task = Task { @MainActor [weak self] in
            defer {
                self?.isGenerating = false
                self?.generationTask = nil
            }

            do {
                try Task.checkCancellation()

                let genStream = try await actor.generate(
                    input: input, parameters: parameters
                )
                let parser = ToolCallParser(startsInsideThinkBlock: startsThinking)
                var rawChunkParts: [String] = []
                var libraryParsedToolCalls = false

                for await generation in genStream {
                    try Task.checkCancellation()

                    switch generation {
                    case .chunk(let text):
                        rawChunkParts.append(text)
                        if !libraryParsedToolCalls {
                            for event in parser.processChunk(text) {
                                continuation.yield(AgentGeneration(parserEvent: event))
                            }
                        }

                    case .info(let completionInfo):
                        if !libraryParsedToolCalls {
                            for event in parser.finalize() {
                                continuation.yield(AgentGeneration(parserEvent: event))
                            }
                        }
                        let info = AgentGeneration.Info(
                            promptTokenCount: completionInfo.promptTokenCount,
                            generationTokenCount: completionInfo.generationTokenCount,
                            promptTime: completionInfo.promptTime,
                            generateTime: completionInfo.generateTime
                        )
                        continuation.yield(.info(info))
                        Log.agent.info(
                            "Generation complete — \(completionInfo.generationTokenCount) tokens, "
                            + "\(String(format: "%.1f", info.tokensPerSecond)) tok/s"
                        )
                        let rawChunks = rawChunkParts.joined()
                        Log.agent.debug("Raw library chunks (after ToolCallProcessor):\n\(rawChunks)")

                    case .toolCall(let call):
                        libraryParsedToolCalls = true
                        Log.agent.info("Library parsed tool call: \(call.function.name)(\(call.function.arguments))")
                        continuation.yield(.toolCall(call))
                    }
                }

                if !libraryParsedToolCalls {
                    let rawChunks = rawChunkParts.joined()
                    if rawChunks.contains("tool_call") || rawChunks.contains("<function") {
                        Log.agent.warning("Raw output contains tool call markers but no .toolCall events were emitted by library")
                    }
                }

                if !libraryParsedToolCalls {
                    for event in parser.finalize() {
                        continuation.yield(AgentGeneration(parserEvent: event))
                    }
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

        generationTask = task
        continuation.onTermination = { _ in task.cancel() }

        return HTTPServerGenerationStart(
            stream: stream,
            cachedTokenCount: 0,
            cancel: { task.cancel() },
            waitForCompletion: { _ = await task.result }
        )
    }

    private func startManagedHTTPFallbackGeneration(
        input: UserInput,
        parameters: AgentGenerateParameters
    ) async throws -> HTTPServerGenerationStart {
        guard isModelLoaded else {
            throw AgentEngineError.modelNotLoaded
        }

        let start = try await llmActor.startHTTPRawGeneration(
            input: input,
            parameters: parameters
        )
        return wrapManagedGeneration(start)
    }

    private func wrapManagedGeneration(
        _ start: HTTPServerRawGenerationStart
    ) -> HTTPServerGenerationStart {
        isGenerating = true

        let (stream, continuation) = AsyncThrowingStream.makeStream(of: AgentGeneration.self)
        let startsThinking = promptStartsThinking

        let task = Task { @MainActor [weak self] in
            defer {
                self?.isGenerating = false
                self?.generationTask = nil
            }

            do {
                try Task.checkCancellation()

                let parser = ToolCallParser(startsInsideThinkBlock: startsThinking)
                var rawChunkParts: [String] = []
                var libraryParsedToolCalls = false

                for await generation in start.stream {
                    try Task.checkCancellation()

                    switch generation {
                    case .chunk(let text):
                        rawChunkParts.append(text)
                        // When the library's ToolCallProcessor handles tool calls (xmlFunction
                        // format), it leaks wrapper tags (<tool_call>/</tool_call>) as chunks.
                        // Skip app-level parsing to avoid false "malformed tool call" warnings.
                        if !libraryParsedToolCalls {
                            for event in parser.processChunk(text) {
                                continuation.yield(AgentGeneration(parserEvent: event))
                            }
                        }

                    case .info(let completionInfo):
                        // Flush any buffered text before emitting info
                        if !libraryParsedToolCalls {
                            for event in parser.finalize() {
                                continuation.yield(AgentGeneration(parserEvent: event))
                            }
                        }
                        let info = AgentGeneration.Info(
                            promptTokenCount: completionInfo.promptTokenCount,
                            generationTokenCount: completionInfo.generationTokenCount,
                            promptTime: completionInfo.promptTime,
                            generateTime: completionInfo.generateTime
                        )
                        continuation.yield(.info(info))
                        Log.agent.info(
                            "Generation complete — \(completionInfo.generationTokenCount) tokens, "
                            + "\(String(format: "%.1f", info.tokensPerSecond)) tok/s"
                        )
                        let rawChunks = rawChunkParts.joined()
                        Log.agent.debug("Raw library chunks (after ToolCallProcessor):\n\(rawChunks)")

                    case .toolCall(let call):
                        // Vendor library parsed this tool call — yield directly
                        libraryParsedToolCalls = true
                        Log.agent.info("Library parsed tool call: \(call.function.name)(\(call.function.arguments))")
                        continuation.yield(.toolCall(call))
                    }
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
                        continuation.yield(AgentGeneration(parserEvent: event))
                    }
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

        generationTask = task
        continuation.onTermination = { _ in task.cancel() }

        return HTTPServerGenerationStart(
            stream: stream,
            cachedTokenCount: 0,
            cancel: {
                task.cancel()
                start.cancel()
            },
            waitForCompletion: {
                _ = await task.result
                await start.waitForCompletion()
            }
        )
    }

    private func startManagedHTTPGeneration(
        _ start: HTTPServerGenerationStart
    ) -> HTTPServerGenerationStart {
        isGenerating = true

        let (stream, continuation) = AsyncThrowingStream.makeStream(of: AgentGeneration.self)

        let task = Task { @MainActor [weak self] in
            defer {
                self?.isGenerating = false
                self?.generationTask = nil
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

        generationTask = task
        continuation.onTermination = { _ in task.cancel() }

        return HTTPServerGenerationStart(
            stream: stream,
            cachedTokenCount: start.cachedTokenCount,
            cancel: {
                task.cancel()
                start.cancel()
            },
            waitForCompletion: {
                _ = await task.result
                await start.waitForCompletion()
            }
        )
    }
}
