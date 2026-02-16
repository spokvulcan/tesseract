import Combine
import Foundation
import os

/// Manages agent chat state — message history, streaming generation, and cancellation.
///
/// Mirrors the `SpeechCoordinator` pattern: owns a `[AgentChatMessage]` conversation,
/// delegates inference to ``AgentEngine``, and publishes streaming text for the view.
@MainActor
final class AgentCoordinator: ObservableObject {

    @Published private(set) var messages: [AgentChatMessage] = []
    @Published private(set) var streamingText: String = ""
    @Published private(set) var isGenerating: Bool = false
    @Published var error: String?

    private let agentEngine: AgentEngine
    private let debugLogger = AgentDebugLogger()
    private var generationTask: Task<Void, Never>?

    init(agentEngine: AgentEngine) {
        self.agentEngine = agentEngine
    }

    /// Sends a user message and streams the assistant response.
    func sendMessage(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        Log.agent.info("User message (\(trimmed.count) chars): \(trimmed)")

        messages.append(.user(trimmed))
        error = nil
        isGenerating = true
        streamingText = ""

        let prompt: [AgentChatMessage] = [.system("You are a helpful assistant.")] + messages

        Log.agent.info("Sending \(prompt.count) messages to model (system + \(self.messages.count) history)")

        // Start debug session on first turn
        if messages.count == 1 {
            debugLogger.startSession()
        }
        debugLogger.logPrompt(messages: prompt, parameters: .default)

        var generationInfo: AgentGeneration.Info?

        generationTask = Task { [weak self] in
            guard let self else { return }

            do {
                let stream = try agentEngine.generate(messages: prompt)
                Log.agent.debug("Generation stream started")

                for try await event in stream {
                    switch event {
                    case .text(let chunk):
                        streamingText += chunk
                    case .toolCall, .malformedToolCall:
                        // Tool execution handled by the agent loop (task 2.3)
                        break
                    case .info(let info):
                        generationInfo = info
                    }
                }

                let response = streamingText
                Log.agent.info("Generation complete — \(response.count) chars")

                debugLogger.logResponse(
                    rawOutput: response,
                    displayOutput: response,
                    info: generationInfo
                )

                messages.append(.assistant(response))
                streamingText = ""
                isGenerating = false
            } catch is CancellationError {
                Log.agent.info("Generation cancelled — \(self.streamingText.count) chars generated")
                if !streamingText.isEmpty {
                    messages.append(.assistant(streamingText))
                }
                streamingText = ""
                isGenerating = false
            } catch {
                Log.agent.error("Generation failed: \(error)")
                debugLogger.logError(error.localizedDescription)
                self.error = error.localizedDescription
                if !streamingText.isEmpty {
                    messages.append(.assistant(streamingText))
                }
                streamingText = ""
                isGenerating = false
            }

            generationTask = nil
        }
    }

    /// Cancels in-progress generation.
    func cancelGeneration() {
        generationTask?.cancel()
        generationTask = nil
        agentEngine.cancelGeneration()
    }

    /// Clears conversation history and cancels any in-progress generation.
    func clearConversation() {
        cancelGeneration()
        messages = []
        streamingText = ""
        error = nil
        debugLogger.reset()
        Log.agent.info("Conversation cleared")
    }
}
