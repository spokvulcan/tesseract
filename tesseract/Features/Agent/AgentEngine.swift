import Combine
import Foundation
import MLXLLM
import MLXLMCommon
import os

/// Loads Nanbeige4.1-3B from a local directory and prepares it for inference.
///
/// Model downloading is handled by ``ModelDownloadManager``. This engine loads
/// the already-downloaded weights into memory and verifies them.
@MainActor
final class AgentEngine: ObservableObject {

    @Published private(set) var isModelLoaded = false
    @Published private(set) var isLoading = false
    @Published private(set) var loadingStatus: String = ""

    private(set) var agentTokenizer: AgentTokenizer?
    private var modelContainer: ModelContainer?

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

    /// Releases the model from memory.
    func unloadModel() {
        modelContainer = nil
        agentTokenizer = nil
        isModelLoaded = false
        loadingStatus = ""
        Log.agent.info("Model unloaded")
    }
}
