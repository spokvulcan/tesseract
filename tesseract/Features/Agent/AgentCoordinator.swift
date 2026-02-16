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
    private var generationTask: Task<Void, Never>?

    init(agentEngine: AgentEngine) {
        self.agentEngine = agentEngine
    }

    /// Sends a user message and streams the assistant response.
    func sendMessage(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        messages.append(.user(trimmed))
        error = nil
        isGenerating = true
        streamingText = ""

        let prompt: [AgentChatMessage] = [.system("You are a helpful assistant.")] + messages

        generationTask = Task { [weak self] in
            guard let self else { return }

            do {
                let stream = try agentEngine.generate(messages: prompt)

                for try await event in stream {
                    switch event {
                    case .text(let chunk):
                        streamingText += chunk
                    case .info:
                        break
                    }
                }

                let response = streamingText
                messages.append(.assistant(response))
                streamingText = ""
                isGenerating = false
            } catch is CancellationError {
                // Append partial response if any
                if !streamingText.isEmpty {
                    messages.append(.assistant(streamingText))
                }
                streamingText = ""
                isGenerating = false
            } catch {
                Log.agent.error("Generation failed: \(error)")
                self.error = error.localizedDescription
                // Append partial response if any
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
    }
}
