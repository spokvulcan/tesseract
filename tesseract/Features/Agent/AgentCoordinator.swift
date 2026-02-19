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
    @Published private(set) var voiceState: AgentVoiceState = .idle
    @Published var error: String?

    private let agentRunner: AgentRunner
    private let conversationStore: AgentConversationStore
    private let audioCapture: (any AudioCapturing)?
    private let transcriptionEngine: (any Transcribing)?
    private let settings: SettingsManager?
    private let postProcessor = TranscriptionPostProcessor()
    private let speechCoordinator: SpeechCoordinator?
    private let prepareForInference: (@MainActor () -> Void)?
    private let loadAgentModel: (@MainActor () async throws -> Void)?
    private let debugLogger = AgentDebugLogger()
    private var generationTask: Task<Void, Never>?
    private var voiceErrorResetTask: Task<Void, Never>?

    private enum Defaults {
        static let contextLimit = 20
        static let minimumRecordingDuration: TimeInterval = 0.5
        static let errorAutoResetDelay: Duration = .seconds(3)
    }

    init(
        agentRunner: AgentRunner,
        conversationStore: AgentConversationStore,
        audioCapture: (any AudioCapturing)? = nil,
        transcriptionEngine: (any Transcribing)? = nil,
        settings: SettingsManager? = nil,
        prepareForInference: (@MainActor () -> Void)? = nil,
        loadAgentModel: (@MainActor () async throws -> Void)? = nil,
        speechCoordinator: SpeechCoordinator? = nil
    ) {
        self.agentRunner = agentRunner
        self.conversationStore = conversationStore
        self.audioCapture = audioCapture
        self.transcriptionEngine = transcriptionEngine
        self.settings = settings
        self.prepareForInference = prepareForInference
        self.loadAgentModel = loadAgentModel
        self.speechCoordinator = speechCoordinator

        // Load the most recent conversation (or create a fresh one)
        conversationStore.loadMostRecent()
        if let current = conversationStore.currentConversation {
            messages = current.messages
        }
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

        // Context window: send system prompt + only the most recent messages to LLM
        let systemPrompt = SystemPromptBuilder.build()
        let recentMessages = Array(messages.suffix(Defaults.contextLimit))
        let prompt: [AgentChatMessage] = [.system(systemPrompt)] + recentMessages

        Log.agent.info("Sending \(prompt.count) messages to model (\(recentMessages.count) of \(self.messages.count) history)")

        // Free memory from other engines before inference
        prepareForInference?()

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
                    case .toolStart(let name, _):
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
                persistCurrentConversation()

                // Auto-speak the final assistant response if enabled
                if let settings, settings.agentAutoSpeak,
                   let lastAssistant = self.messages.last(where: { $0.role == .assistant }),
                   !lastAssistant.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    Log.agent.info("Auto-speaking response (\(lastAssistant.content.count) chars)")
                    speechCoordinator?.speakText(lastAssistant.content)
                }
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
                persistCurrentConversation()
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
                persistCurrentConversation()
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

    /// Creates a new conversation, saving the current one first.
    func newConversation() {
        cancelGeneration()
        conversationStore.createNew()
        messages = []
        streamingText = ""
        streamingThinking = ""
        isThinking = false
        error = nil
        debugLogger.reset()
        Log.agent.info("New conversation created")
    }

    /// Loads a past conversation by ID.
    func loadConversation(_ id: UUID) {
        cancelGeneration()
        conversationStore.load(id: id)
        messages = conversationStore.currentConversation?.messages ?? []
        streamingText = ""
        streamingThinking = ""
        isThinking = false
        error = nil
        debugLogger.reset()
        Log.agent.info("Loaded conversation \(id) with \(self.messages.count) messages")
    }

    /// Deletes a conversation. If it's the current one, switches to a new conversation.
    func deleteConversation(_ id: UUID) {
        let wasCurrent = conversationStore.currentConversation?.id == id
        conversationStore.delete(id: id)
        if wasCurrent {
            messages = conversationStore.currentConversation?.messages ?? []
            debugLogger.reset()
        }
    }

    /// Clears conversation history and cancels any in-progress generation.
    func clearConversation() {
        newConversation()
    }

    // MARK: - Voice Input

    func startVoiceInput() {
        guard voiceState == .idle else { return }
        guard let audioCapture else {
            setVoiceError("Voice input not available")
            return
        }
        guard !audioCapture.isCapturing else {
            setVoiceError("Microphone in use")
            return
        }

        do {
            try audioCapture.startCapture()
            voiceState = .recording
            Log.agent.info("Voice input started")
        } catch {
            setVoiceError("Mic error: \(error.localizedDescription)")
        }
    }

    func stopVoiceInputAndSend() {
        guard voiceState == .recording else { return }
        guard let audioCapture, let transcriptionEngine else {
            cancelVoiceInput()
            return
        }

        let audioData = audioCapture.stopCapture()

        guard let audioData, audioData.duration >= Defaults.minimumRecordingDuration else {
            setVoiceError("Recording too short")
            return
        }

        voiceState = .transcribing
        Log.agent.info("Voice input stopped, transcribing \(String(format: "%.1f", audioData.duration))s audio")

        Task {
            do {
                let language = settings?.language ?? "en"
                let result = try await transcriptionEngine.transcribe(audioData, language: language)
                let processedText = postProcessor.process(result.text)

                guard !processedText.isEmpty else {
                    setVoiceError("No speech detected")
                    return
                }

                Log.agent.info("Voice transcribed: \(processedText)")
                voiceState = .idle

                // Ensure agent LLM is loaded before sending
                try await loadAgentModel?()
                sendMessage(processedText)
            } catch {
                setVoiceError("Transcription failed")
                Log.agent.error("Voice transcription error: \(error)")
            }
        }
    }

    func cancelVoiceInput() {
        if let audioCapture, audioCapture.isCapturing {
            _ = audioCapture.stopCapture()
        }
        transcriptionEngine?.cancelTranscription()
        voiceState = .idle
        Log.agent.info("Voice input cancelled")
    }

    private func setVoiceError(_ message: String) {
        voiceState = .error(message)
        Log.agent.warning("Voice error: \(message)")

        voiceErrorResetTask?.cancel()
        voiceErrorResetTask = Task {
            try? await Task.sleep(for: Defaults.errorAutoResetDelay)
            if case .error = voiceState {
                voiceState = .idle
            }
        }
    }

    // MARK: - Voice Output

    func speakMessage(_ message: AgentChatMessage) {
        guard message.role == .assistant,
              !message.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else { return }
        speechCoordinator?.speakText(message.content)
    }

    func stopSpeaking() {
        speechCoordinator?.stop()
    }

    // MARK: - Private

    private func persistCurrentConversation() {
        conversationStore.updateCurrentMessages(messages)
        conversationStore.saveCurrent()
    }
}
