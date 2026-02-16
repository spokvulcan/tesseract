import Foundation
import MLXLMCommon
import os

/// Actor-isolated wrapper that owns the LLM model and runs inference off the MainActor.
///
/// Follows the same pattern as `WhisperActor` in `TranscriptionEngine`:
/// the `@MainActor` ``AgentEngine`` publishes UI state while delegating
/// heavy model operations to this actor.
actor LLMActor {

    private var modelContainer: ModelContainer?
    private(set) var agentTokenizer: AgentTokenizer?

    var isLoaded: Bool { modelContainer != nil }

    /// Loads model weights, verifies with a 1-token generation, and resolves the tokenizer.
    ///
    /// - Returns: The resolved ``AgentTokenizer`` (caller can stash it on MainActor).
    @discardableResult
    func loadModel(from directory: URL) async throws -> AgentTokenizer {
        let container = try await loadModelContainer(directory: directory)

        // Verify with a 1-token generation
        let input = try await container.prepare(input: UserInput(prompt: "Hello"))
        let parameters = GenerateParameters(maxTokens: 1)
        let stream = try await container.generate(input: input, parameters: parameters)
        for await _ in stream {}

        let tokenizer = try await AgentTokenizer(container: container)

        modelContainer = container
        agentTokenizer = tokenizer
        return tokenizer
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

    /// Releases the model from memory.
    func unloadModel() {
        modelContainer = nil
        agentTokenizer = nil
    }
}
