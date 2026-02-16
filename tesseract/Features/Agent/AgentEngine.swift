import Combine
import Foundation
import MLXLLM
import MLXLMCommon
import os

/// Downloads and loads Nanbeige4.1-3B for local LLM inference.
@MainActor
final class AgentEngine: ObservableObject {

    @Published private(set) var isModelLoaded = false
    @Published private(set) var isLoading = false
    @Published private(set) var loadProgress: Double = 0
    @Published private(set) var loadingStatus: String = ""

    private(set) var agentTokenizer: AgentTokenizer?
    private var modelContainer: ModelContainer?

    private enum Defaults {
        static let modelId = "mlx-community/Nanbeige4.1-3B-8bit"
    }

    /// Downloads (if needed) and loads the model, then verifies with a 1-token generation.
    func loadModel() async throws {
        guard !isModelLoaded, !isLoading else { return }

        isLoading = true
        loadProgress = 0
        loadingStatus = "Downloading model…"

        do {
            let configuration = ModelConfiguration(id: Defaults.modelId)

            let container = try await loadModelContainer(
                configuration: configuration
            ) { [weak self] progress in
                Task { @MainActor in
                    guard let self else { return }
                    self.loadProgress = progress.fractionCompleted
                    if progress.fractionCompleted < 1.0 {
                        self.loadingStatus = "Downloading model… \(Int(progress.fractionCompleted * 100))%"
                    } else {
                        self.loadingStatus = "Loading model…"
                    }
                }
            }

            Log.agent.info("Model downloaded and loaded, verifying…")
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
            loadProgress = 0
            Log.agent.error("Failed to load model: \(error)")
            throw error
        }

        isLoading = false
    }

    /// Releases the model from memory.
    func unloadModel() {
        modelContainer = nil
        agentTokenizer = nil
        isModelLoaded = false
        loadingStatus = ""
        loadProgress = 0
        Log.agent.info("Model unloaded")
    }
}
