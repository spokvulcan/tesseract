import Combine
import Foundation
import MLXLLM
import MLXLMCommon
import Tokenizers
import os

/// Errors thrown by ``AgentEngine`` during generation.
enum AgentEngineError: LocalizedError {
    case modelNotLoaded
    case generationFailed(String)

    var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            "No model is loaded"
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
@MainActor
final class AgentEngine: ObservableObject {

    @Published private(set) var isModelLoaded = false
    @Published private(set) var isLoading = false
    @Published private(set) var loadingStatus: String = ""
    @Published private(set) var isGenerating = false

    private(set) var agentTokenizer: AgentTokenizer?

    private let llmActor = LLMActor()
    private var generationTask: Task<Void, Never>?

    /// Loads model weights from a local directory into memory and verifies with a 1-token generation.
    ///
    /// - Parameter directory: Local path containing model weights, config, and tokenizer files
    ///   (as downloaded by ``ModelDownloadManager``).
    func loadModel(from directory: URL) async throws {
        guard !isModelLoaded, !isLoading else { return }

        isLoading = true
        loadingStatus = "Loading model…"

        do {
            let tokenizer = try await llmActor.loadModel(from: directory)

            let st = tokenizer.specialTokens
            Log.agent.info(
                "Special tokens resolved — imStart=\(st.imStart) imEnd=\(st.imEnd) "
                + "endOfText=\(st.endOfText) thinkStart=\(st.thinkStart) thinkEnd=\(st.thinkEnd) "
                + "toolCallStart=\(st.toolCallStart) toolCallEnd=\(st.toolCallEnd)"
            )

            agentTokenizer = tokenizer
            isModelLoaded = true
            loadingStatus = ""
            Log.agent.info("Model loaded and verified successfully")
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

        // 1. Convert LLMMessage → Chat.Message, prepending the system prompt
        var chatMessages = [Chat.Message.system(systemPrompt)]
        chatMessages.append(contentsOf: toLLMCommonMessages(messages))

        // 2. Convert AgentToolDefinition → ToolSpec ([String: any Sendable])
        let toolSpecs: [ToolSpec]? = tools?.map { $0.toolSpec }

        // 3. Build UserInput and delegate to shared generation logic
        let input = UserInput(chat: chatMessages, tools: toolSpecs)
        return try startGeneration(input: input, parameters: parameters)
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

    /// Releases the model from memory.
    func unloadModel() {
        cancelGeneration()
        agentTokenizer = nil
        isModelLoaded = false
        loadingStatus = ""
        Task { await llmActor.unloadModel() }
        Log.agent.info("Model unloaded")
    }

    // MARK: - Private

    /// Shared generation logic for both prompt-based and message-based entry points.
    private func startGeneration(
        input: UserInput,
        parameters: AgentGenerateParameters
    ) throws -> AsyncThrowingStream<AgentGeneration, Error> {
        guard isModelLoaded else {
            throw AgentEngineError.modelNotLoaded
        }

        isGenerating = true

        let (stream, continuation) = AsyncThrowingStream.makeStream(of: AgentGeneration.self)
        let actor = llmActor

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
                let parser = ToolCallParser()
                var rawChunkParts: [String] = []
                var libraryParsedToolCalls = false

                for await generation in genStream {
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

        return stream
    }
}
