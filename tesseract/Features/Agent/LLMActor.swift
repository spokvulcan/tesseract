import Foundation
import MLX
import MLXLMCommon
import Tokenizers
import os

/// Actor-isolated wrapper that owns the LLM model and runs inference off the MainActor.
///
/// Follows the same pattern as `WhisperActor` in `TranscriptionEngine`:
/// the `@MainActor` ``AgentEngine`` publishes UI state while delegating
/// heavy model operations to this actor.
actor LLMActor {

    private enum Defaults {
        static let cacheLimitMB = 128
    }

    private var modelContainer: ModelContainer?
    private(set) var agentTokenizer: AgentTokenizer?

    var isLoaded: Bool { modelContainer != nil }

    /// Loads model weights, verifies with a 1-token generation, and resolves the tokenizer.
    ///
    /// Reads the model's `config.json` to detect the model type and configure the
    /// appropriate tool call format (e.g., `.xmlFunction` for Qwen3.5).
    ///
    /// - Returns: The resolved ``AgentTokenizer`` and whether the template starts inside a think block.
    @discardableResult
    func loadModel(from directory: URL) async throws -> (AgentTokenizer, promptStartsThinking: Bool) {
        let format = Self.detectToolCallFormat(directory: directory)
        Log.agent.info("Tool call format: \(format.map { "\($0)" } ?? "json (default)")")
        let config = ModelConfiguration(directory: directory, toolCallFormat: format)
        let container = try await loadModelContainer(configuration: config)

        // Verify with a 1-token generation
        let input = try await container.prepare(input: UserInput(prompt: "Hello"))
        let parameters = GenerateParameters(maxTokens: 1)
        let stream = try await container.generate(input: input, parameters: parameters)
        for await _ in stream {}

        let tokenizer = try await AgentTokenizer(container: container)

        modelContainer = container
        agentTokenizer = tokenizer
        return (tokenizer, Self.detectPromptStartsThinking(directory: directory))
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

        let genParams = GenerateParameters(
            maxTokens: parameters.maxTokens,
            temperature: parameters.temperature,
            topP: parameters.topP,
            repetitionPenalty: parameters.repetitionPenalty,
            repetitionContextSize: parameters.repetitionContextSize
        )

        return try await container.generate(input: prepared, parameters: genParams)
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
            return context.tokenizer.decode(tokens: tokens, skipSpecialTokens: false)
        }
    }

    /// Releases the model from memory.
    func unloadModel() {
        modelContainer = nil
        agentTokenizer = nil
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

    /// Returns `true` if the model's chat template appends `<think>` to the generation prompt.
    ///
    /// Detected models:
    /// - Qwen3.5: `enable_thinking` defaults to true, template ends with `<think>\n`
    /// - Qwen3 Thinking / Opus Distill: unconditionally append `<think>\n`
    /// - Qwen3 Instruct / Nanbeige: no thinking in prompt → returns false
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
