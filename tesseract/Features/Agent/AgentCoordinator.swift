import Combine
import Foundation
import os

/// Manages agent chat state — message history, streaming generation, and cancellation.
///
/// Mirrors the `SpeechCoordinator` pattern: owns a `[AgentChatMessage]` conversation,
/// delegates inference to ``AgentRunner`` (which handles tool loops internally),
/// and publishes streaming text for the view.
@MainActor
final class AgentCoordinator: ObservableObject {

    @Published private(set) var messages: [AgentChatMessage] = []
    @Published private(set) var streamingText: String = ""
    @Published private(set) var streamingThinking: String = ""
    @Published private(set) var isThinking: Bool = false
    @Published private(set) var isGenerating: Bool = false
    @Published var error: String?

    private let agentRunner: AgentRunner
    private let debugLogger = AgentDebugLogger()
    private var generationTask: Task<Void, Never>?

    init(agentRunner: AgentRunner) {
        self.agentRunner = agentRunner
    }

    /// Sends a user message and streams the assistant response (with tool loops).
    func sendMessage(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        Log.agent.info("User message (\(trimmed.count) chars): \(trimmed)")

        messages.append(.user(trimmed))
        error = nil
        isGenerating = true
        streamingText = ""
        streamingThinking = ""
        isThinking = false

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
                let stream = try agentRunner.run(messages: prompt)
                Log.agent.debug("Agent runner stream started")

                for try await event in stream {
                    switch event {
                    case .text(let chunk):
                        streamingText += chunk
                    case .thinkStart:
                        isThinking = true
                    case .thinking(let chunk):
                        streamingThinking += chunk
                    case .thinkEnd:
                        isThinking = false
                        Log.agent.debug("Think block (\(self.streamingThinking.count) chars)")
                    case .toolStart(let name):
                        Log.agent.info("Tool start: \(name)")
                    case .toolResult(let name, let result):
                        Log.agent.info("Tool result [\(name)]: \(result.prefix(200))")
                        // Clear streaming text between rounds so the next
                        // generation round starts fresh in the UI
                        streamingText = ""
                        streamingThinking = ""
                        isThinking = false
                    case .toolError(let raw):
                        Log.agent.warning("Tool error: \(raw.prefix(200))")
                    case .info(let info):
                        generationInfo = info
                    case .completed(let newMessages):
                        messages.append(contentsOf: newMessages)

                        let lastAssistant = newMessages.last { $0.role == .assistant }
                        debugLogger.logResponse(
                            rawOutput: lastAssistant?.content ?? "",
                            displayOutput: lastAssistant?.content ?? "",
                            thinking: lastAssistant?.thinking,
                            info: generationInfo
                        )
                    }
                }

                Log.agent.info("Agent run complete — \(self.messages.count) total messages")
                streamingText = ""
                streamingThinking = ""
                isThinking = false
                isGenerating = false
            } catch is CancellationError {
                Log.agent.info("Generation cancelled — \(self.streamingText.count) chars generated")
                if !streamingText.isEmpty || !streamingThinking.isEmpty {
                    messages.append(.assistant(
                        streamingText,
                        thinking: streamingThinking.isEmpty ? nil : streamingThinking
                    ))
                }
                streamingText = ""
                streamingThinking = ""
                isThinking = false
                isGenerating = false
            } catch {
                Log.agent.error("Generation failed: \(error)")
                debugLogger.logError(error.localizedDescription)
                self.error = error.localizedDescription
                if !streamingText.isEmpty || !streamingThinking.isEmpty {
                    messages.append(.assistant(
                        streamingText,
                        thinking: streamingThinking.isEmpty ? nil : streamingThinking
                    ))
                }
                streamingText = ""
                streamingThinking = ""
                isThinking = false
                isGenerating = false
            }

            generationTask = nil
        }
    }

    /// Cancels in-progress generation.
    func cancelGeneration() {
        generationTask?.cancel()
        generationTask = nil
        agentRunner.cancelGeneration()
    }

    /// Clears conversation history and cancels any in-progress generation.
    func clearConversation() {
        cancelGeneration()
        messages = []
        streamingText = ""
        streamingThinking = ""
        isThinking = false
        error = nil
        debugLogger.reset()
        Log.agent.info("Conversation cleared")
    }
}
