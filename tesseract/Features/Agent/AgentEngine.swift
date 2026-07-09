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

    /// Whether the loaded container is the vision (VLM) variant — decides
    /// whether tool-result images (browser screenshots) are attached to the
    /// prompt or degraded to a text note (`toLLMCommonMessages`).
    private(set) var loadedVisionMode = false

    private(set) var agentTokenizer: AgentTokenizer?

    /// Whether the loaded model's template starts generation inside a `<think>` block.
    private(set) var promptStartsThinking = false

    /// The loaded model's template-declared render flags
    /// (`ModelIdentity.declaredTemplateFlags`) — cached at load like
    /// `promptStartsThinking` so the server dispatcher can read it
    /// synchronously on the MainActor (issue #98). Empty when unloaded.
    private(set) var declaredTemplateFlags: Set<TemplateRenderFlag> = []

    /// The loaded model's tool-call format (`ModelIdentity.toolCallFormat`) —
    /// cached at load like `declaredTemplateFlags` so the server's Argument
    /// Transcoder keys off the same identity the parser uses. `nil` when
    /// unloaded or when the model has no override (vendor JSON default).
    private(set) var toolCallFormat: ToolCallFormat?

    /// The shared inference actor. Created by the composition root and
    /// injected so the server dispatcher can reach the same actor (ADR-0015);
    /// benchmarks and unit tests rely on the `init` default instead.
    let llmActor: LLMActor
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
        ssdConfig: SSDPrefixCacheConfig? = nil,
        llmActor: LLMActor = LLMActor()
    ) {
        self.settingsManager = settingsManager
        self.llmActor = llmActor
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
    ///   - visionMode: When `true`, loads the VLM variant of ParoQuant models (adds
    ///     image support). Text prefill measures on par with the LLM variant (the
    ///     retired "slower prefill" claim — ADR-0013); the only standing cost is the
    ///     resident vision tower. When `false`, loads the LLM variant. Ignored for
    ///     non-ParoQuant models.
    func loadModel(
        from directory: URL,
        visionMode: Bool
    ) async throws {
        guard !isModelLoaded, !isLoading else { return }

        isLoading = true
        loadingStatus = "Loading model…"
        // Reset on every exit, including the throwing path — a stuck `true`
        // would block all future load attempts behind the reentrancy guard.
        defer { isLoading = false }

        do {
            let (tokenizer, startsThinking) = try await llmActor.loadModel(
                from: directory,
                visionMode: visionMode,
                ssdConfig: resolveSSDConfig(),
                // RAM cap (ADR-0018): same snapshot-at-load semantics as
                // the SSD config; nil (no settings source) = Automatic.
                ramBudgetCapBytes: settingsManager?.prefixCacheRAMBudgetCapBytes
            )

            let st = tokenizer.specialTokens
            Log.agent.info(
                "Special tokens resolved — imStart=\(st.imStart) imEnd=\(st.imEnd) "
                    + "endOfText=\(st.endOfText) thinkStart=\(st.thinkStart) thinkEnd=\(st.thinkEnd) "
                    + "toolCallStart=\(st.toolCallStart) toolCallEnd=\(st.toolCallEnd)"
            )

            agentTokenizer = tokenizer
            promptStartsThinking = startsThinking
            declaredTemplateFlags = await llmActor.loadedDeclaredTemplateFlags()
            toolCallFormat = await llmActor.loadedToolCallFormat()
            loadedVisionMode = visionMode
            isModelLoaded = true
            loadingStatus = ""
            Log.agent.info("Model loaded — promptStartsThinking=\(promptStartsThinking)")
        } catch {
            loadingStatus = ""
            Log.agent.error("Failed to load model: \(error)")
            throw error
        }
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
        Log.agent.debug(
            "generate(llmMessages:) — \(messages.count) messages, \(tools?.count ?? 0) tools")
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
        Log.agent.debug(
            "generate(toolSpecs:) — \(messages.count) messages, \(toolSpecs?.count ?? 0) tool specs"
        )
        return try startChatInference(
            systemPrompt: systemPrompt,
            messages: messages,
            toolSpecs: toolSpecs,
            parameters: parameters,
            progressHandler: nil
        ).stream
    }

    /// Build a `UserInput` from a system prompt, messages, and optional raw tool specs.
    /// Extracted for testability — callers can verify tool specs are forwarded without a loaded model.
    /// The render context's opt-in flags (`preserve_thinking`) ride into the
    /// template render as `additionalContext`, so the standard chat path honors
    /// them exactly as the cache-aware path does.
    static func buildUserInput(
        systemPrompt: String,
        messages: [LLMMessage],
        toolSpecs: [ToolSpec]?,
        renderContext: TemplateRenderContext = .canonical,
        visionActive: Bool = false
    ) -> UserInput {
        var chatMessages = [Chat.Message.system(systemPrompt)]
        chatMessages.append(contentsOf: toLLMCommonMessages(messages, visionActive: visionActive))
        return UserInput(
            chat: chatMessages,
            tools: toolSpecs,
            additionalContext: renderContext.additionalContext()
        )
    }

    /// Formats a system prompt + tools through the chat template, returning the raw ChatML string and token count.
    ///
    /// Some model templates (e.g. Qwen3.5) require at least one user message, so a placeholder
    /// is appended and stripped from the output to isolate the system prompt portion.
    func formatRawPrompt(systemPrompt: String, tools: [AgentToolDefinition]?) async throws -> (
        text: String, tokenCount: Int
    ) {
        let placeholder = "__SYSTEM_PROMPT_END__"
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": systemPrompt],
            ["role": "user", "content": placeholder],
        ]
        let toolSpecs: [ToolSpec]? = tools?.map { $0.toolSpec }
        var result = try await llmActor.formatRawPromptWithCount(
            messages: messages, tools: toolSpecs)

        // Strip everything from the placeholder user message onward
        if let range = result.text.range(of: placeholder) {
            // Back up to the im_start tag for the user turn
            let beforePlaceholder = result.text[..<range.lowerBound]
            if let userTagRange = beforePlaceholder.range(
                of: "<|im_start|>user", options: .backwards)
            {
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
        declaredTemplateFlags = []
        toolCallFormat = nil
        loadedVisionMode = false
        isModelLoaded = false
        loadingStatus = ""
        unloadTask = Task { [llmActor] in
            // Stop the in-flight server completion BEFORE the SSD flush:
            // admissions landing mid-flush would miss the persisted manifest,
            // and the engine no longer registers cache-aware completions, so
            // the `cancelGeneration()` above does not cover them. The drain
            // inside `unloadModel` remains the backstop.
            await llmActor.drainServerCompletion()
            await llmActor.prefixCacheAdmin.flushSSDWrites()
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
        parameters: AgentGenerateParameters,
        progressHandler: ServerInferenceProgressHandler? = nil
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
                parameters: parameters,
                progressHandler: progressHandler
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

        let actor = llmActor
        let safeguardConfig = parameters.thinkingSafeguard
        parameters.warnIfThinkingLoopRiskElevated(startsThinking: startsThinking)

        // The loop owns the cross-swap cancel invariant. Its `cancelCurrent` must
        // be wired into `start.cancel` synchronously, but the loop can't be built
        // until `launch` yields the initial handle — bridge through a late-bound
        // cancel the task fills once the loop exists.
        let loopCancel = LateBoundCancel()

        let task = Task { @MainActor [weak self] in
            defer {
                self?.isGenerating = false
                self?.clearActiveGeneration(id: generationID)
            }

            do {
                try Task.checkCancellation()

                let initialStart = try await launch()
                let loop = GenerationStreamLoop(
                    initial: .init(initialStart),
                    startsInsideThinkBlock: startsThinking,
                    safeguard: safeguardConfig,
                    logContext: "generation_id=\(generationID.uuidString)"
                )
                loopCancel.fill(loop.cancelCurrent)

                // The agent supplies a continuation starter only when it has an
                // `originalInput` to re-prefill from; otherwise `nil` ⇒ the loop
                // emits the truncation triple and stops.
                let continuationStarter: GenerationStreamLoop.ContinuationStarter?
                if let originalInput = input {
                    continuationStarter = { safePrefix in
                        let newStart = try await actor.startThinkingContinuationRaw(
                            originalInput: originalInput,
                            safeThinkingPrefix: safePrefix,
                            injection: safeguardConfig.continuationHandOff,
                            toolSpecs: toolSpecs,
                            parameters: parameters
                        )
                        return .init(newStart)
                    }
                } else {
                    continuationStarter = nil
                }

                // The sink only yields — the agent keeps no per-event side effects.
                let outcome = try await loop.run(continuation: continuationStarter) { event in
                    continuation.yield(event)
                }

                if outcome.cancelled {
                    continuation.finish()
                    return
                }

                // Re-yield the terminal `.info` the loop captured (downstream reads
                // completion metrics from the stream, not a return value).
                if let info = outcome.completionInfo {
                    continuation.yield(.info(info))
                    Log.agent.info(
                        "Generation complete — \(info.generationTokenCount) tokens, "
                            + "\(String(format: "%.1f", info.tokensPerSecond)) tok/s, "
                            + "stopReason=\(describeStopReason(info.stopReason))"
                    )
                }
                if outcome.diagnostics.hasUnparsedToolCallMarkers {
                    Log.agent.warning(
                        "Raw output contains tool call markers but no .toolCall events were emitted by library"
                    )
                }
                continuation.finish()
            } catch is CancellationError {
                loopCancel()
                continuation.finish()
            } catch {
                continuation.finish(
                    throwing: AgentEngineError.generationFailed(
                        error.localizedDescription
                    ))
            }
        }

        let start = HTTPServerGenerationStart(
            stream: stream,
            cachedTokenCount: cachedTokenCount,
            cancel: {
                // Cancel the live raw handle (unparks a mid-generation `for await`)
                // and the driving task.
                loopCancel()
                task.cancel()
            },
            waitForCompletion: {
                _ = await task.result
            }
        )
        continuation.onTermination = { _ in start.cancel() }
        return registerActiveGeneration(start, id: generationID)
    }

}

@MainActor
extension AgentEngine: ManagedInferenceStarting {
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
        parameters: AgentGenerateParameters,
        renderContext: TemplateRenderContext = .canonical,
        progressHandler: ServerInferenceProgressHandler?
    ) throws -> HTTPServerGenerationStart {
        let input = Self.buildUserInput(
            systemPrompt: systemPrompt,
            messages: messages,
            toolSpecs: toolSpecs,
            renderContext: renderContext,
            visionActive: loadedVisionMode
        )
        return try startManagedGeneration(
            input: input,
            toolSpecs: toolSpecs,
            parameters: parameters,
            progressHandler: progressHandler
        )
    }
}
