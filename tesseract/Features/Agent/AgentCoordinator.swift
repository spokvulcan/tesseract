import Combine
import Foundation
import os

/// Thin UI bridge that delegates to ``Agent`` (new core loop).
///
/// `agent.state.messages` is the **single source of truth**. The coordinator
/// derives `messages: [AgentChatMessage]` for view consumption — it never
/// appends to this array directly.
@MainActor
final class AgentCoordinator: ObservableObject {

    // MARK: - Published UI State

    /// Derived display messages. Refreshed from `agent.state.messages`.
    @Published private(set) var messages: [AgentChatMessage] = []
    @Published private(set) var streamingText: String = ""
    @Published private(set) var streamingThinking: String = ""
    @Published private(set) var isThinking: Bool = false
    @Published private(set) var isGenerating: Bool = false
    @Published private(set) var voiceState: AgentVoiceState = .idle
    @Published var error: String?

    // System prompt transparency
    @Published private(set) var assembledSystemPrompt: String = ""
    @Published private(set) var rawChatMLPrompt: String?
    @Published private(set) var systemPromptTokenCount: Int?

    // MARK: - Dependencies

    private let agent: Agent
    private let conversationStore: AgentConversationStore
    private let audioCapture: (any AudioCapturing)?
    private let transcriptionEngine: (any Transcribing)?
    private let settings: SettingsManager?
    private let postProcessor = TranscriptionPostProcessor()
    private let speechCoordinator: SpeechCoordinator?
    private let notchController: AgentNotchPanelController?
    private let prepareForInference: (@MainActor () -> Void)?
    private let loadAgentModel: (@MainActor () async throws -> Void)?
    private let formatRawPrompt: (@MainActor (String, [AgentToolDefinition]?) async throws -> (text: String, tokenCount: Int))?
    private let debugLogger = AgentDebugLogger()
    private var voiceErrorResetTask: Task<Void, Never>?
    private var rawPromptFetchTask: Task<Void, Never>?
    private var notchTextLength: Int = 0

    /// Unsubscribe closure for agent event subscription.
    private var unsubscribe: (@MainActor () -> Void)?

    private enum Defaults {
        nonisolated static let minimumRecordingDuration: TimeInterval = 0.5
        nonisolated static let errorAutoResetDelay: Duration = .seconds(3)
    }

    // MARK: - Init

    init(
        agent: Agent,
        conversationStore: AgentConversationStore,
        audioCapture: (any AudioCapturing)? = nil,
        transcriptionEngine: (any Transcribing)? = nil,
        settings: SettingsManager? = nil,
        prepareForInference: (@MainActor () -> Void)? = nil,
        loadAgentModel: (@MainActor () async throws -> Void)? = nil,
        formatRawPrompt: (@MainActor (String, [AgentToolDefinition]?) async throws -> (text: String, tokenCount: Int))? = nil,
        speechCoordinator: SpeechCoordinator? = nil,
        notchController: AgentNotchPanelController? = nil
    ) {
        self.agent = agent
        self.conversationStore = conversationStore
        self.audioCapture = audioCapture
        self.transcriptionEngine = transcriptionEngine
        self.settings = settings
        self.prepareForInference = prepareForInference
        self.loadAgentModel = loadAgentModel
        self.formatRawPrompt = formatRawPrompt
        self.speechCoordinator = speechCoordinator
        self.notchController = notchController

        // Sync assembled system prompt immediately (agent has it from init)
        assembledSystemPrompt = agent.state.systemPrompt

        // Subscribe to agent events
        subscribeToAgentEvents()

        // Load the most recent conversation (or create a fresh one)
        conversationStore.loadMostRecent()
        if let current = conversationStore.currentConversation {
            // Restore agent state from persisted messages
            agent.loadMessages(current.messages)
            refreshDisplayMessages()
        }
    }

    deinit {
        // Capture the closure to call from nonisolated deinit via MainActor.
        let unsub = unsubscribe
        if let unsub {
            MainActor.assumeIsolated { unsub() }
        }
    }

    // MARK: - Send Message

    /// Sends a user message via the Agent. The agent owns the message lifecycle.
    func sendMessage(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        Log.agent.info("User message (\(trimmed.count) chars): \(trimmed)")

        error = nil
        isGenerating = true
        streamingText = ""
        streamingThinking = ""
        isThinking = false

        // Free memory from other engines before inference
        prepareForInference?()

        // Start debug session on first turn
        if agent.state.messages.isEmpty {
            debugLogger.startSession()
            debugLogger.logSystemPrompt(agent.state.systemPrompt, tools: agent.state.tools)
        }

        // Show thinking in notch if it's visible (voice-initiated)
        notchController?.updatePhase(.thinking)

        // Create user message and prompt the agent — the agent appends it to context
        let userMessage = CoreMessage.user(UserMessage.create(trimmed))
        agent.prompt(userMessage)
    }

    /// Cancels in-progress generation.
    func cancelGeneration() {
        agent.abort()
    }

    // MARK: - Conversation Management

    /// Creates a new conversation, saving the current one first.
    func newConversation() {
        cancelGeneration()
        persistCurrentConversation()
        conversationStore.createNew()
        agent.resetMessages([])
        messages = []
        streamingText = ""
        streamingThinking = ""
        isThinking = false
        error = nil
        rawChatMLPrompt = nil
        systemPromptTokenCount = nil
        debugLogger.reset()
        Log.agent.info("New conversation created")
    }

    /// Loads a past conversation by ID.
    func loadConversation(_ id: UUID) {
        cancelGeneration()
        persistCurrentConversation()
        conversationStore.load(id: id)
        if let current = conversationStore.currentConversation {
            agent.loadMessages(current.messages)
        } else {
            agent.resetMessages([])
        }
        refreshDisplayMessages()
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
            if let current = conversationStore.currentConversation {
                agent.loadMessages(current.messages)
            } else {
                agent.resetMessages([])
            }
            refreshDisplayMessages()
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

    // MARK: - Event Subscription

    private func subscribeToAgentEvents() {
        unsubscribe = agent.subscribe { [weak self] event in
            Task { @MainActor in
                self?.handleAgentEvent(event)
            }
        }
    }

    private func handleAgentEvent(_ event: AgentEvent) {
        switch event {
        case .agentStart:
            isGenerating = true
            refreshDisplayMessages()

        case .contextTransformStart(let reason):
            updatePhaseIndicator(reason)

        case .contextTransformEnd(let reason, let didMutate, _):
            clearPhaseIndicator(reason: reason, didMutate: didMutate)
            if didMutate { refreshDisplayMessages() }

        case .messageUpdate(_, let delta):
            if let text = delta.textDelta {
                streamingText += text
                // Throttle notch updates — every 10 chars
                if notchController?.isShowing == true,
                   streamingText.count - notchTextLength >= 10 {
                    notchTextLength = streamingText.count
                    notchController?.updatePhase(.responding(text: streamingText))
                }
            }
            if let thinking = delta.thinkingDelta {
                if !isThinking {
                    isThinking = true
                    notchController?.updatePhase(.thinking)
                }
                streamingThinking += thinking
            }

        case .toolExecutionStart(_, let toolName, _):
            Log.agent.info("Tool start: \(toolName)")
            notchController?.updatePhase(.toolCall(name: toolName))
            // Clear streaming state between tool rounds
            streamingText = ""
            streamingThinking = ""
            isThinking = false
            notchTextLength = 0

        case .toolExecutionEnd(_, let toolName, let result, _):
            let text = result.content.textContent
            Log.agent.info("Tool result [\(toolName)]: \(text.prefix(200))")
            notchController?.updatePhase(.thinking)

        case .turnEnd(let message, let toolResults, let contextMessages):
            debugLogger.logTurn(
                message: message,
                toolResults: toolResults,
                messageCount: contextMessages.count
            )

            // Save after each turn for crash resilience
            refreshDisplayMessages()
            persistCurrentConversation()

        case .agentEnd(_):
            isGenerating = false
            isThinking = false
            streamingText = ""
            streamingThinking = ""
            notchTextLength = 0
            refreshDisplayMessages()
            // turnEnd already saved — only save again if no turns occurred
            autoSpeakIfEnabled()

            // Show complete state in notch
            if let notchController, notchController.isShowing,
               let lastAssistant = messages.last(where: { $0.role == .assistant }) {
                notchController.updatePhase(.complete(text: lastAssistant.content))
            }

        case .turnStart, .messageStart, .messageEnd, .toolExecutionUpdate:
            break
        }
    }

    // MARK: - System Prompt

    /// Fetches the raw ChatML-formatted system prompt from the model's tokenizer.
    func fetchRawSystemPrompt() {
        assembledSystemPrompt = agent.state.systemPrompt
        Log.agent.info("fetchRawSystemPrompt — prompt length=\(assembledSystemPrompt.count)")

        let tools = agent.state.tools
        guard let formatRawPrompt else {
            Log.agent.warning("fetchRawSystemPrompt — formatRawPrompt closure is nil")
            return
        }

        Log.agent.info("fetchRawSystemPrompt — calling closure with \(tools.count) tools")
        rawPromptFetchTask?.cancel()
        rawPromptFetchTask = Task {
            do {
                let result = try await formatRawPrompt(assembledSystemPrompt, tools)
                guard !Task.isCancelled else { return }
                Log.agent.info("fetchRawSystemPrompt — success, raw length=\(result.text.count), tokens=\(result.tokenCount)")
                rawChatMLPrompt = result.text
                systemPromptTokenCount = result.tokenCount
            } catch is CancellationError {
                Log.agent.debug("fetchRawSystemPrompt — cancelled")
            } catch {
                Log.agent.error("fetchRawSystemPrompt — error: \(error)")
            }
        }
    }

    // MARK: - Private Helpers

    /// Derive UI messages from the agent's authoritative state.
    private func refreshDisplayMessages() {
        messages = agent.state.messages.map { msg in
            if let chat = msg as? AgentChatMessage { return chat.normalizedForDisplay() }
            return AgentChatMessage(from: msg)
        }
    }

    private func persistCurrentConversation() {
        guard !agent.state.messages.isEmpty else { return }
        conversationStore.updateCurrentMessages(agent.state.messages.map { $0 as any AgentMessageProtocol & Sendable })
        conversationStore.saveCurrent()
    }

    private func autoSpeakIfEnabled() {
        guard let settings, settings.agentAutoSpeak,
              let lastAssistant = messages.last(where: { $0.role == .assistant }),
              !lastAssistant.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else { return }
        Log.agent.info("Auto-speaking response (\(lastAssistant.content.count) chars)")
        speechCoordinator?.speakText(lastAssistant.content)
    }

    private func updatePhaseIndicator(_ reason: ContextTransformReason) {
        switch reason {
        case .compaction:
            notchController?.updatePhase(.thinking)
        case .extensionTransform(let name):
            Log.agent.info("Extension transform: \(name)")
        }
    }

    private func clearPhaseIndicator(reason: ContextTransformReason, didMutate: Bool) {
        // Phase indicator is transient — clears automatically when next event arrives
        if didMutate {
            switch reason {
            case .compaction:
                Log.agent.info("Context compaction applied")
            case .extensionTransform(let name):
                Log.agent.info("Extension transform applied: \(name)")
            }
        }
    }
}
