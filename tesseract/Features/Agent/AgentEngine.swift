import Combine
import Foundation
import MLXLLM
import MLXLMCommon
import Tokenizers
import os

/// Errors thrown by ``AgentEngine`` during generation.
enum AgentEngineError: LocalizedError {
    case modelNotLoaded
    case alreadyGenerating
    case generationFailed(String)

    var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            "No model is loaded"
        case .alreadyGenerating:
            "Generation is already in progress"
        case .generationFailed(let description):
            "Generation failed: \(description)"
        }
    }
}

/// Loads Nanbeige4.1-3B from a local directory and prepares it for inference.
///
/// Model downloading is handled by ``ModelDownloadManager``. This engine loads
/// the already-downloaded weights into memory and verifies them.
@MainActor
final class AgentEngine: ObservableObject {

    @Published private(set) var isModelLoaded = false
    @Published private(set) var isLoading = false
    @Published private(set) var loadingStatus: String = ""
    @Published private(set) var isGenerating = false

    private(set) var agentTokenizer: AgentTokenizer?
    private var modelContainer: ModelContainer?
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
            let container = try await loadModelContainer(directory: directory)

            Log.agent.info("Model loaded into memory, verifying…")
            loadingStatus = "Verifying model…"

            // Verify with a 1-token generation
            let input = try await container.prepare(
                input: UserInput(prompt: "Hello")
            )
            let parameters = GenerateParameters(maxTokens: 1)
            let stream = try await container.generate(input: input, parameters: parameters)
            for await _ in stream {}

            // Resolve special tokens from the tokenizer vocabulary
            loadingStatus = "Resolving tokenizer…"
            let tokenizer = try await AgentTokenizer(container: container)
            let st = tokenizer.specialTokens
            Log.agent.info(
                "Special tokens resolved — imStart=\(st.imStart) imEnd=\(st.imEnd) "
                + "endOfText=\(st.endOfText) thinkStart=\(st.thinkStart) thinkEnd=\(st.thinkEnd) "
                + "toolCallStart=\(st.toolCallStart) toolCallEnd=\(st.toolCallEnd)"
            )

            modelContainer = container
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

    /// Streams text generation from a structured conversation history.
    ///
    /// The messages are formatted into ChatML via the model's Jinja template pipeline.
    ///
    /// - Parameters:
    ///   - messages: Conversation history in chronological order.
    ///   - tools: Optional tool schemas for the Jinja template's `<tools>` block.
    ///   - parameters: Generation parameters (temperature, maxTokens, etc.).
    /// - Returns: An async stream of ``AgentGeneration`` events.
    func generate(
        messages: [AgentChatMessage],
        tools: [ToolSpec]? = nil,
        parameters: AgentGenerateParameters = .default
    ) throws -> AsyncThrowingStream<AgentGeneration, Error> {
        let input = AgentChatFormatter.makeUserInput(from: messages, tools: tools)
        return try startGeneration(input: input, parameters: parameters)
    }

    // MARK: - Private

    /// Shared generation logic for both prompt-based and message-based entry points.
    private func startGeneration(
        input: UserInput,
        parameters: AgentGenerateParameters
    ) throws -> AsyncThrowingStream<AgentGeneration, Error> {
        guard let container = modelContainer else {
            throw AgentEngineError.modelNotLoaded
        }
        guard !isGenerating else {
            throw AgentEngineError.alreadyGenerating
        }

        isGenerating = true

        let (stream, continuation) = AsyncThrowingStream.makeStream(of: AgentGeneration.self)

        let task = Task { [weak self] in
            defer {
                Task { @MainActor [weak self] in
                    self?.isGenerating = false
                    self?.generationTask = nil
                }
            }

            do {
                try Task.checkCancellation()

                let prepared = try await container.prepare(input: input)

                let genParams = GenerateParameters(
                    maxTokens: parameters.maxTokens,
                    temperature: parameters.temperature,
                    topP: parameters.topP,
                    repetitionPenalty: parameters.repetitionPenalty,
                    repetitionContextSize: parameters.repetitionContextSize
                )

                let genStream = try await container.generate(
                    input: prepared, parameters: genParams
                )

                for await generation in genStream {
                    try Task.checkCancellation()

                    switch generation {
                    case .chunk(let text):
                        continuation.yield(.text(text))

                    case .info(let completionInfo):
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

                    case .toolCall(let call):
                        // Stringify tool calls as text for now (task 2.2 handles proper parsing)
                        let text = "<tool_call>\(call.function.name)(\(call.function.arguments))</tool_call>"
                        continuation.yield(.text(text))
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

    /// Cancels any in-progress generation.
    func cancelGeneration() {
        generationTask?.cancel()
        generationTask = nil
    }

    /// Releases the model from memory.
    func unloadModel() {
        cancelGeneration()
        modelContainer = nil
        agentTokenizer = nil
        isModelLoaded = false
        loadingStatus = ""
        Log.agent.info("Model unloaded")
    }
}
