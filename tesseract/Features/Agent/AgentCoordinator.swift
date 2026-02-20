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
    private let notchController: AgentNotchPanelController?
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
        speechCoordinator: SpeechCoordinator? = nil,
        notchController: AgentNotchPanelController? = nil
    ) {
        self.agentRunner = agentRunner
        self.conversationStore = conversationStore
        self.audioCapture = audioCapture
        self.transcriptionEngine = transcriptionEngine
        self.settings = settings
        self.prepareForInference = prepareForInference
        self.loadAgentModel = loadAgentModel
        self.speechCoordinator = speechCoordinator
        self.notchController = notchController

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
        let voiceMode = settings?.agentAutoSpeak ?? false
        let systemPrompt = SystemPromptBuilder.build(voiceMode: voiceMode)
        let recentMessages = Array(messages.suffix(Defaults.contextLimit))
        let prompt: [AgentChatMessage] = [.system(systemPrompt)] + recentMessages

        // Select model-specific generation parameters
        let modelID = settings?.selectedAgentModelID ?? "qwen3-4b-instruct-2507"
        let generateParams = AgentGenerateParameters.forModel(modelID)

        Log.agent.info("Sending \(prompt.count) messages to model \(modelID) (\(recentMessages.count) of \(self.messages.count) history)")

        // Free memory from other engines before inference
        prepareForInference?()

        // Start debug session on first turn
        if messages.count == 1 {
            debugLogger.startSession()
        }
        debugLogger.logPrompt(messages: prompt, parameters: generateParams)

        var generationInfo: AgentGeneration.Info?

        // Show thinking in notch if it's visible (voice-initiated)
        notchController?.updatePhase(.thinking)

        generationTask = Task { [weak self] in
            guard let self else { return }
            var notchTextLength = 0

            do {
                let stream = try agentRunner.run(messages: prompt, parameters: generateParams)
                Log.agent.debug("Agent runner stream started")

                for try await event in stream {
                    switch event {
                    case .text(let chunk):
                        streamingText += chunk
                        // Throttle notch updates — every 10 chars
                        if notchController?.isShowing == true && streamingText.count - notchTextLength >= 10 {
                            notchTextLength = streamingText.count
                            notchController?.updatePhase(.responding(text: streamingText))
                        }
                    case .thinkStart:
                        isThinking = true
                        notchController?.updatePhase(.thinking)
                    case .thinking(let chunk):
                        streamingThinking += chunk
                    case .thinkEnd:
                        if !isThinking {
                            // Model omitted <think> tag — text streamed so far is thinking
                            streamingThinking = streamingText
                            streamingText = ""
                        }
                        isThinking = false
                        Log.agent.debug("Think block (\(self.streamingThinking.count) chars)")
                    case .toolStart(let name, _):
                        Log.agent.info("Tool start: \(name)")
                        notchController?.updatePhase(.toolCall(name: name))
                    case .toolResult(let name, let result):
                        Log.agent.info("Tool result [\(name)]: \(result.prefix(200))")
                        // Clear streaming text between rounds so the next
                        // generation round starts fresh in the UI
                        streamingText = ""
                        streamingThinking = ""
                        isThinking = false
                        notchTextLength = 0
                        notchController?.updatePhase(.thinking)
                    case .toolError(let raw):
                        Log.agent.warning("Tool error: \(raw.prefix(200))")
                    case .roundStart:
                        break
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

                // Show complete state in notch with response preview
                if let notchController, notchController.isShowing,
                   let lastAssistant = self.messages.last(where: { $0.role == .assistant }) {
                    notchController.updatePhase(.complete(text: lastAssistant.content))
                }

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
                notchController?.dismiss()
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
                notchController?.updatePhase(.error(error.localizedDescription))
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
            notchController?.show()
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
        notchController?.updatePhase(.transcribing(preview: ""))
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
                notchController?.updatePhase(.transcribing(preview: processedText))
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
        notchController?.dismiss()
        Log.agent.info("Voice input cancelled")
    }

    private func setVoiceError(_ message: String) {
        voiceState = .error(message)
        Log.agent.warning("Voice error: \(message)")

        if notchController?.isShowing == true {
            notchController?.updatePhase(.error(message))
        }

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
