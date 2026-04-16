import Foundation
import Observation
import os
import MLXLMCommon

/// Thin UI bridge that delegates to ``Agent`` (new core loop).
///
/// `agent.state.messages` is the **single source of truth**. The coordinator
/// derives `rows: [ChatRow]` for view consumption via `rebuildRows()`, which
/// walks messages once and directly produces flat, equatable rows for SwiftUI's List.
@Observable @MainActor
final class AgentCoordinator {

    // MARK: - Observable UI State

    /// Flat, pre-computed rows for the chat List. Equatable elements enable SwiftUI skip-rendering.
    private(set) var rows: [ChatRow] = []
    private(set) var isGenerating: Bool = false

    /// Bumped on each streaming row update — cheap change signal for scroll tracking.
    private(set) var streamingRowVersion: Int = 0

    /// Progressive streaming message. Reads through to agent's @Observable state.
    var streamMessage: AssistantMessage? { agent.state.streamMessage }
    private(set) var voiceState: AgentVoiceState = .idle
    var error: String?

    // Background session viewing (read-only mode for notification deep-links)
    private(set) var isViewingBackgroundSession: Bool = false
    private(set) var viewingSessionId: UUID?
    private(set) var viewingSessionName: String?

    // System prompt transparency
    private(set) var assembledSystemPrompt: String = ""
    private(set) var rawChatMLPrompt: String?
    private(set) var systemPromptTokenCount: Int?

    // MARK: - UI Interaction State

    /// Expanded turn headers (step timeline visible).
    @ObservationIgnored private var expandedTurns: Set<UUID> = []
    /// Expanded tool call details (arguments/results visible), keyed by row ID.
    @ObservationIgnored private var expandedDetails: Set<String> = []
    /// Throttle streaming row updates.
    @ObservationIgnored private var lastStreamingUpdate: ContinuousClock.Instant = .now
    /// Turn that was auto-expanded for generation — auto-collapse on agentEnd.
    @ObservationIgnored private var autoExpandedTurnID: UUID?
    /// Tracks if user manually collapsed the streaming header during this generation.
    @ObservationIgnored private var streamingManuallyCollapsed: Bool = false
    /// Index in `rows` where the active (last) turn starts — used for incremental streaming patches.
    @ObservationIgnored private var activeTurnRowIndex: Int = 0
    /// Stable ID for the streaming turn header — survives rebuilds so toggle state persists.
    private static let streamingTurnID = UUID(uuidString: "00000000-0000-0000-0000-000000000001")!

    // MARK: - Dependencies

    private let agent: Agent
    private let conversationStore: AgentConversationStore
    private let audioCapture: (any AudioCapturing)?
    private let transcriptionEngine: (any Transcribing)?
    private let settings: SettingsManager?
    private let toolRegistry: ToolRegistry?
    private let postProcessor = TranscriptionPostProcessor()
    private let speechCoordinator: SpeechCoordinator?
    private let arbiter: InferenceArbiter?
    private let formatRawPrompt: (@MainActor (String, [AgentToolDefinition]?) async throws -> (text: String, tokenCount: Int))?
    @ObservationIgnored var loadBackgroundSessionById: (@MainActor (UUID) async throws -> (messages: [any AgentMessageProtocol & Sendable], name: String))?
    private let debugLogger = AgentDebugLogger()

    // MARK: - Slash Commands

    private(set) var commandRegistry = SlashCommandRegistry()
    var showCommandPopup: Bool = false
    var commandSelectedIndex: Int = 0
    var commandFilteredResults: [SlashCommand] = []
    private let extensionHost: ExtensionHost?
    private let packageRegistry: PackageRegistry?
    private let contextManager: ContextManager?
    private let contextWindow: Int
    private let summarize: (@Sendable (String) async throws -> String)?
    /// The task that holds the arbiter lease for the current agent run.
    /// Cancelled by `cancelGeneration()` to abort both queued waits and active runs.
    @ObservationIgnored private var sendTask: Task<Void, Never>?
    @ObservationIgnored private var voiceErrorResetTask: Task<Void, Never>?
    @ObservationIgnored private var rawPromptFetchTask: Task<Void, Never>?

    /// Unsubscribe closure for agent event subscription.
    @ObservationIgnored private var unsubscribe: (@MainActor () -> Void)?

    /// Shared date formatter for timestamps — avoids creating formatters per row.
    @ObservationIgnored private let timeFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateStyle = .none
        f.timeStyle = .short
        return f
    }()

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
        arbiter: InferenceArbiter? = nil,
        formatRawPrompt: (@MainActor (String, [AgentToolDefinition]?) async throws -> (text: String, tokenCount: Int))? = nil,
        speechCoordinator: SpeechCoordinator? = nil,
        toolRegistry: ToolRegistry? = nil,
        extensionHost: ExtensionHost? = nil,
        packageRegistry: PackageRegistry? = nil,
        contextManager: ContextManager? = nil,
        contextWindow: Int = 262_144,
        summarize: (@Sendable (String) async throws -> String)? = nil
    ) {
        self.agent = agent
        self.conversationStore = conversationStore
        self.audioCapture = audioCapture
        self.transcriptionEngine = transcriptionEngine
        self.settings = settings
        self.arbiter = arbiter
        self.formatRawPrompt = formatRawPrompt
        self.speechCoordinator = speechCoordinator
        self.toolRegistry = toolRegistry
        self.extensionHost = extensionHost
        self.packageRegistry = packageRegistry
        self.contextManager = contextManager
        self.contextWindow = contextWindow
        self.summarize = summarize

        assembledSystemPrompt = agent.state.systemPrompt

        subscribeToAgentEvents()
        rebuildCommandRegistry()

        conversationStore.loadMostRecent()
        if let current = conversationStore.currentConversation {
            agent.loadMessages(current.messages)
            rebuildRows()
        }
    }

    deinit {
        let unsub = unsubscribe
        if let unsub {
            MainActor.assumeIsolated { unsub() }
        }
    }

    // MARK: - Send Message

    func sendMessage(_ text: String, images: [ImageAttachment] = [], bypassCommandParsing: Bool = false) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty || !images.isEmpty else { return }

        if !bypassCommandParsing && images.isEmpty {
            let parseResult = SlashCommandParser.parse(trimmed, registry: commandRegistry)
            switch parseResult {
            case .matched(let command, let arguments):
                executeCommand(command, arguments: arguments)
                return
            case .unknown(let name):
                error = "Unknown command: /\(name)"
                return
            case .notACommand, .partial:
                break
            }
        }

        Log.agent.info("User message (\(trimmed.count) chars, \(images.count) images): \(trimmed)")

        error = nil
        isGenerating = true

        if agent.state.messages.isEmpty {
            debugLogger.startSession()
            debugLogger.logSystemPrompt(agent.state.systemPrompt, tools: agent.state.tools)
        }

        let userMessage = CoreMessage.user(UserMessage(content: trimmed, images: images))

        // Sync active tools based on current web access setting
        syncToolsForWebAccess()

        if let arbiter {
            sendTask = Task {
                do {
                    try await arbiter.withExclusiveGPU(.llm) {
                        self.agent.prompt(userMessage)
                        await self.agent.waitForIdle()
                    }
                } catch is CancellationError {
                    // Cancelled while queued or during run — clean up
                    self.isGenerating = false
                } catch {
                    self.error = error.localizedDescription
                    self.isGenerating = false
                }
                self.sendTask = nil
            }
        } else {
            agent.prompt(userMessage)
        }
    }

    /// Filter active tools based on webAccessEnabled setting.
    /// Called before each prompt so the LLM sees the current tool set.
    private func syncToolsForWebAccess() {
        guard let toolRegistry else { return }
        let allTools = toolRegistry.allTools
        if settings?.webAccessEnabled == true {
            agent.updateTools(allTools)
        } else {
            agent.updateTools(allTools.filter { $0.name != "web_search" && $0.name != "web_fetch" })
        }
    }

    func cancelGeneration() {
        sendTask?.cancel()
        sendTask = nil
        agent.abort()
    }

    // MARK: - Vision Mode Toggle

    /// Handles a user-initiated vision mode toggle from the composer.
    ///
    /// Flips `SettingsManager.visionModeEnabled`, then triggers an eager model
    /// reload via the arbiter's empty-body lease flow — `withExclusiveGPU(.llm)`
    /// calls `ensureLoaded(.llm)` which detects the visionMode mismatch and
    /// reloads into the new mode. The UI is responsible for disabling the
    /// toggle button while `isGenerating` or `agentEngine.isLoading`, so this
    /// method mostly needs to handle the reload lifecycle.
    ///
    /// On failure: reverts the setting, attempts a reload to the previous mode
    /// (graceful degradation), and surfaces an error message.
    func setVisionModeEnabled(_ enabled: Bool) {
        guard !isGenerating else {
            Log.agent.info("Vision mode toggle ignored — generation in progress")
            return
        }
        guard let arbiter, let settings else {
            Log.agent.warning("Vision mode toggle ignored — arbiter or settings unavailable")
            return
        }
        guard settings.visionModeEnabled != enabled else { return }

        let previous = settings.visionModeEnabled
        settings.visionModeEnabled = enabled
        Log.agent.info("Vision mode toggle → \(enabled); triggering reload")

        Task { @MainActor in
            do {
                // Empty-body lease trick: withExclusiveGPU runs ensureLoaded(.llm)
                // before the body, which detects the visionMode change and reloads.
                try await arbiter.withExclusiveGPU(.llm) { }
                Log.agent.info("Vision mode switched to \(enabled)")
            } catch {
                Log.agent.error("Vision mode switch failed — \(error.localizedDescription)")
                // Revert the setting and attempt to reload with the previous value
                // so the user isn't left with an unloaded model.
                settings.visionModeEnabled = previous
                do {
                    try await arbiter.withExclusiveGPU(.llm) { }
                    self.error = "Vision mode unavailable: \(error.localizedDescription)"
                } catch let fallbackError {
                    self.error = "Model reload failed: \(fallbackError.localizedDescription)"
                }
            }
        }
    }

    // MARK: - Slash Command Execution

    /// Update popup state based on current input text.
    func updateCommandPopup(for inputText: String) {
        if let prefix = SlashCommandParser.autocompletePrefix(inputText) {
            if !showCommandPopup { showCommandPopup = true }
            commandFilteredResults = commandRegistry.filter(prefix: prefix)
        } else {
            if showCommandPopup { showCommandPopup = false }
        }
        if commandSelectedIndex != 0 {
            commandSelectedIndex = 0
        }
    }

    func dismissCommandPopup() {
        showCommandPopup = false
        commandFilteredResults = []
        commandSelectedIndex = 0
    }

    /// Autocomplete a command into the input text (used by both keyboard and click).
    func autocompleteCommand(_ command: SlashCommand) -> String {
        dismissCommandPopup()
        return "/\(command.name) "
    }

    /// Rebuild the command registry from current skills and extensions.
    private func rebuildCommandRegistry() {
        let agentRoot = PathSandbox.defaultRoot
        let skillsDir = agentRoot.appendingPathComponent("skills")
        let packageSkillFiles: [URL]
        if let packageRegistry {
            packageSkillFiles = PackageBootstrap.cachedSkillPaths(from: packageRegistry, agentRoot: agentRoot)
        } else {
            packageSkillFiles = []
        }
        let skills = SkillRegistry.discover(locations: [skillsDir], packageSkillFiles: packageSkillFiles)
        commandRegistry.rebuild(skills: skills, extensionHost: extensionHost)
    }

    func executeCommand(_ command: SlashCommand, arguments: String = "") {
        Log.agent.info("Slash command: /\(command.name) \(arguments)")

        switch command.source {
        case .builtIn:
            executeBuiltIn(command.name, arguments: arguments)
        case .skill(let filePath):
            executeSkill(filePath: filePath, skillName: command.name, arguments: arguments)
        case .extension(let extensionPath):
            executeExtensionCommand(command.name, extensionPath: extensionPath, arguments: arguments)
        }
    }

    private func executeBuiltIn(_ name: String, arguments: String) {
        switch name {
        case "compact":
            triggerCompaction(instructions: arguments.isEmpty ? nil : arguments)
        case "new", "clear":
            newConversation()
        default:
            error = "Unknown built-in command: /\(name)"
        }
    }

    private func triggerCompaction(instructions: String?) {
        guard !agent.state.messages.isEmpty else {
            error = "Nothing to compact"
            return
        }
        guard let contextManager, let summarize else {
            error = "Compaction not available"
            return
        }

        isGenerating = true
        agent.forceCompact(
            contextManager: contextManager,
            contextWindow: contextWindow,
            summarize: summarize
        )
    }

    private func executeSkill(filePath: String, skillName: String, arguments: String) {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: filePath)),
              let fullText = String(data: data, encoding: .utf8) else {
            error = "Failed to load skill: \(filePath)"
            return
        }

        let body = SkillRegistry.bodyContent(of: fullText)
        let skillDir = URL(fileURLWithPath: filePath).deletingLastPathComponent().path

        var message = """
        <skill name="\(skillName)" location="\(filePath)">
        References are relative to \(skillDir).

        \(body)
        </skill>
        """

        if !arguments.isEmpty {
            message += "\n\n\(arguments)"
        }

        sendMessage(message, bypassCommandParsing: true)
    }

    private func executeExtensionCommand(
        _ name: String,
        extensionPath: String,
        arguments: String
    ) {
        guard let ext = extensionHost?.getExtension(path: extensionPath),
              let cmd = ext.commands[name] else {
            error = "Extension command not found: /\(name)"
            return
        }
        Task {
            do {
                let context = StubExtensionContext(cwd: PathSandbox.defaultRoot.path)
                try await cmd.execute(arguments, context)
            } catch {
                self.error = "Command failed: \(error.localizedDescription)"
            }
        }
    }

    // MARK: - Conversation Management

    func newConversation() {
        cancelGeneration()
        if isViewingBackgroundSession {
            // Exit read-only mode without persisting background messages
            isViewingBackgroundSession = false
            viewingSessionName = nil
        } else {
            persistCurrentConversation()
        }
        conversationStore.createNew()
        agent.resetMessages([])
        resetState()
        error = nil
        rawChatMLPrompt = nil
        systemPromptTokenCount = nil
        debugLogger.reset()
        Log.agent.info("New conversation created")
    }

    func loadConversation(_ id: UUID) {
        cancelGeneration()
        if isViewingBackgroundSession {
            isViewingBackgroundSession = false
            viewingSessionName = nil
        } else {
            persistCurrentConversation()
        }
        conversationStore.load(id: id)
        if let current = conversationStore.currentConversation {
            agent.loadMessages(current.messages)
        } else {
            agent.resetMessages([])
        }
        resetState()
        rebuildRows()
        error = nil
        debugLogger.reset()
        Log.agent.info("Loaded conversation \(id) with \(self.rows.count) rows")
    }

    func deleteConversation(_ id: UUID) {
        let wasCurrent = conversationStore.currentConversation?.id == id
        conversationStore.delete(id: id)
        if wasCurrent {
            if let current = conversationStore.currentConversation {
                agent.loadMessages(current.messages)
            } else {
                agent.resetMessages([])
            }
            resetState()
            rebuildRows()
            debugLogger.reset()
        }
    }

    func clearConversation() {
        newConversation()
    }

    // MARK: - Background Session Viewing

    @discardableResult
    func openBackgroundSession(id: UUID) async -> Bool {
        guard let loader = loadBackgroundSessionById else {
            error = "Background session loading not available"
            return false
        }
        do {
            let (messages, name) = try await loader(id)
            cancelGeneration()
            if !isViewingBackgroundSession {
                persistCurrentConversation()
            }
            agent.loadMessages(messages)
            resetState()
            rebuildRows()
            isViewingBackgroundSession = true
            viewingSessionId = id
            viewingSessionName = name
            error = nil
            debugLogger.reset()
            Log.agent.info("Viewing background session '\(name)' with \(self.rows.count) rows")
            return true
        } catch {
            self.error = "Failed to load background session"
            Log.agent.error("Failed to load background session \(id): \(error)")
            return false
        }
    }

    func dismissBackgroundSession() {
        isViewingBackgroundSession = false
        viewingSessionId = nil
        viewingSessionName = nil
        error = nil
        // Restore the previous conversation
        if let current = conversationStore.currentConversation {
            agent.loadMessages(current.messages)
        } else {
            agent.resetMessages([])
        }
        resetState()
        rebuildRows()
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

    /// Called when voice transcription completes, to populate the input bar.
    @ObservationIgnored var onVoiceTranscription: ((String) -> Void)?

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

                self.onVoiceTranscription?(processedText)
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

    func speakMessage(_ messageID: UUID) {
        for rawMsg in agent.state.messages {
            if let asst = rawMsg.asAssistant, asst.id == messageID {
                let text = asst.content.trimmingCharacters(in: .whitespacesAndNewlines)
                if !text.isEmpty { speechCoordinator?.speakText(text) }
                return
            }
        }
    }

    func stopSpeaking() {
        speechCoordinator?.stop()
    }

    // MARK: - Expand/Collapse

    func toggleTurnExpanded(_ turnID: UUID) {
        expandedTurns.formSymmetricDifference([turnID])
        // Track manual collapse of the streaming header
        if turnID == Self.streamingTurnID && !expandedTurns.contains(Self.streamingTurnID) {
            streamingManuallyCollapsed = true
        }
        rebuildRows()
    }

    func toggleDetailExpanded(_ rowID: String) {
        let beforeCount = rows.count
        expandedDetails.formSymmetricDifference([rowID])
        if let idx = rows.firstIndex(where: { $0.id == rowID }),
           case .toolCall(let data) = rows[idx].kind {
            rows[idx] = ChatRow(id: rowID, kind: .toolCall(data.togglingDetail()))
        }
        Log.agent.debug("[Perf] toggleDetailExpanded: \(rowID) | rows \(beforeCount) → \(rows.count)")
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
            rebuildRows()

        case .contextTransformStart(let reason):
            if case .extensionTransform(let name) = reason {
                Log.agent.info("Extension transform: \(name)")
            }

        case .contextTransformEnd(let reason, let didMutate, _):
            if didMutate {
                switch reason {
                case .compaction:
                    Log.agent.info("Context compaction applied")
                case .extensionTransform(let name):
                    Log.agent.info("Extension transform applied: \(name)")
                }
                rebuildRows()
            }
            // Reset isGenerating for standalone compaction (not part of an agent loop)
            if case .compaction = reason, agent.state.phase == .idle {
                isGenerating = false
            }

        case .messageUpdate:
            updateStreamingRows()

        case .toolExecutionStart(_, let toolName, _):
            Log.agent.info("Tool start: \(toolName)")

        case .toolExecutionEnd(_, let toolName, let result, _):
            let text = result.content.textContent
            Log.agent.info("Tool result [\(toolName)]: \(text.prefix(200))")

        case .turnEnd(let message, let toolResults, let contextMessages):
            debugLogger.logTurn(
                message: message,
                toolResults: toolResults,
                messageCount: contextMessages.count
            )
            // Auto-expand on first turnEnd — by now the user message is committed
            // and the assistant turn exists in agent.state.messages.
            if autoExpandedTurnID == nil {
                autoExpandLastTurn()
            }
            rebuildRows()
            persistCurrentConversation()

        case .agentEnd(_):
            isGenerating = false
            streamingManuallyCollapsed = false
            expandedTurns.remove(Self.streamingTurnID)
            autoCollapseIfNeeded()
            rebuildRows()
            autoSpeakIfEnabled()

        case .messageEnd:
            rebuildRows()

        case .turnStart, .messageStart, .toolExecutionUpdate, .malformedToolCall:
            break
        }
    }

    // MARK: - Row Building

    /// Single-pass row builder: groups messages into turns and directly produces `[ChatRow]`.
    private func rebuildRows() {
        let perfState = ChatViewPerf.signposter.beginInterval("rebuildRows")

        let allMessages = agent.state.messages

        // Group messages into turns — each turn starts with a user/system message.
        struct Turn {
            let id: UUID
            var messages: [any AgentMessageProtocol]
            var userContent: String?
            var userImages: [ImageAttachment]
            var userTimestamp: String?
            var compactionText: String?
        }

        var turns: [Turn] = []
        var currentTurn: Turn?

        for msg in allMessages {
            let user = msg.asUser
            let compaction = msg as? CompactionSummaryMessage

            if user != nil || compaction != nil {
                if let t = currentTurn { turns.append(t) }
                let turnID = msg.messageUUID
                currentTurn = Turn(
                    id: turnID, messages: [msg],
                    userContent: user?.content,
                    userImages: user?.images ?? [],
                    userTimestamp: user.map { timeFormatter.string(from: $0.timestamp) },
                    compactionText: compaction.map { "[Context compacted — \($0.tokensBefore) tokens summarized]" }
                )
            } else {
                if currentTurn == nil {
                    currentTurn = Turn(id: msg.messageUUID, messages: [], userContent: nil, userImages: [], userTimestamp: nil, compactionText: nil)
                }
                currentTurn?.messages.append(msg)
            }
        }
        if let t = currentTurn { turns.append(t) }

        // Prune expansion state against active turns
        let activeTurnIDs = Set(turns.map(\.id))
        var validTurnIDs = activeTurnIDs
        if isGenerating { validTurnIDs.insert(Self.streamingTurnID) }
        expandedTurns = expandedTurns.intersection(validTurnIDs)

        // Build rows directly
        var newRows: [ChatRow] = []
        let needsDetailPruning = !expandedDetails.isEmpty
        var allToolCallRowIDs: Set<String> = []

        for (turnIndex, turn) in turns.enumerated() {
            let isLastTurn = turnIndex == turns.count - 1
            if isLastTurn { activeTurnRowIndex = newRows.count }

            // User row
            if let content = turn.userContent, let timestamp = turn.userTimestamp {
                newRows.append(ChatRow(
                    id: turn.id.uuidString,
                    kind: .user(UserRow(
                        content: content,
                        images: turn.userImages,
                        timestamp: timestamp,
                        messageID: turn.id
                    ))
                ))
            }

            // System (compaction) row
            if let text = turn.compactionText {
                newRows.append(ChatRow(
                    id: turn.id.uuidString + "-system",
                    kind: .system(SystemRow(content: text))
                ))
            }

            // Process assistant messages into steps + final answer
            var toolResultMap: [String: ToolResultMessage] = [:]
            for msg in turn.messages {
                if let tr = msg.asToolResult { toolResultMap[tr.toolCallId] = tr }
            }
            let lastAssistantMsg = turn.messages.last(where: { $0.asAssistant != nil })?.asAssistant

            var steps: [ChatRow] = []
            var answerContent: String?
            var answerTimestamp: String?
            var answerMessageID: UUID?

            for msg in turn.messages {
                if msg is CompactionSummaryMessage { continue }
                if msg.asUser != nil || msg.asToolResult != nil { continue }
                guard let asst = msg.asAssistant else { continue }

                // Thinking
                if let thinking = asst.thinking?.trimmingCharacters(in: .whitespacesAndNewlines),
                   !thinking.isEmpty {
                    let rowID = "\(asst.id)-thinking"
                    steps.append(ChatRow(id: rowID, kind: .thinking(ThinkingRow(content: thinking, isLast: false))))
                }

                let isFinalAnswer = asst.id == lastAssistantMsg?.id && asst.toolCalls.isEmpty
                let trimmedContent = asst.content.trimmingCharacters(in: .whitespacesAndNewlines)

                // Intermediate text (not the final answer)
                if !trimmedContent.isEmpty && !isFinalAnswer {
                    let rowID = "\(asst.id)-text"
                    steps.append(ChatRow(id: rowID, kind: .toolText(ToolTextRow(content: trimmedContent, isLast: false))))
                }

                // Tool calls
                for (index, info) in asst.toolCalls.enumerated() {
                    let props = ToolDisplayHelpers.displayProps(for: info)
                    let result = toolResultMap[info.id]
                    let rowID = "\(asst.id)-tool-\(index)"
                    if needsDetailPruning { allToolCallRowIDs.insert(rowID) }
                    steps.append(ChatRow(id: rowID, kind: .toolCall(ToolCallRow(
                        displayTitle: props.title,
                        iconName: props.icon,
                        argumentsFormatted: props.argsFormatted,
                        resultContent: result?.content.textContent,
                        isError: result?.isError ?? false,
                        isLast: false,
                        isDetailExpanded: expandedDetails.contains(rowID),
                        filePath: props.filePath
                    ))))
                }

                // Final answer
                if isFinalAnswer && !trimmedContent.isEmpty {
                    answerContent = trimmedContent
                    answerTimestamp = timeFormatter.string(from: asst.timestamp)
                    answerMessageID = asst.id
                }
            }

            let isActiveTurn = isGenerating && isLastTurn
            let isExpanded = expandedTurns.contains(turn.id)
            let liveSteps = isActiveTurn ? streamingStepCount() : 0

            // Streaming header for active turn
            if isActiveTurn {
                appendStreamingHeader(to: &newRows, totalStepCount: steps.count + liveSteps)
            }

            // Turn header for completed turns
            if !steps.isEmpty && !isActiveTurn {
                newRows.append(ChatRow(
                    id: "\(turn.id)-header",
                    kind: .turnHeader(TurnHeaderRow(
                        stepCount: steps.count,
                        isGenerating: false,
                        turnID: turn.id,
                        isExpanded: isExpanded
                    ))
                ))
            }

            // Step rows (if expanded)
            if !steps.isEmpty {
                let showSteps = isActiveTurn
                    ? expandedTurns.contains(Self.streamingTurnID)
                    : isExpanded
                if showSteps {
                    let hasLiveStepsAfter = liveSteps > 0
                    for (index, step) in steps.enumerated() {
                        let isLast = index == steps.count - 1 && !hasLiveStepsAfter
                        newRows.append(isLast ? step.withIsLast(true) : step)
                    }
                }
            }

            // Live streaming step rows
            if isActiveTurn {
                appendStreamingStepRows(to: &newRows)
            }

            // Answer row
            if let content = answerContent,
               let timestamp = answerTimestamp,
               let messageID = answerMessageID {
                newRows.append(ChatRow(
                    id: messageID.uuidString + "-answer",
                    kind: .assistantText(AssistantTextRow(
                        content: content,
                        timestamp: timestamp,
                        messageID: messageID,
                        hasStepsAbove: !steps.isEmpty
                    ))
                ))
            }
        }

        // Streaming rows for turns with no committed assistant messages yet
        if isGenerating {
            let lastTurnHasAssistant = turns.last.map { turn in
                turn.messages.contains(where: { $0.asAssistant != nil })
            } ?? false
            if !lastTurnHasAssistant {
                let liveSteps = streamingStepCount()
                if liveSteps > 0 {
                    appendStreamingHeader(to: &newRows, totalStepCount: liveSteps)
                    appendStreamingStepRows(to: &newRows)
                } else if agent.state.streamMessage == nil {
                    newRows.append(ChatRow(id: "streaming-indicator", kind: .streamingIndicator))
                } else {
                    appendStreamingStepRows(to: &newRows)
                    if newRows.last?.id.hasPrefix("streaming-") != true {
                        newRows.append(ChatRow(id: "streaming-indicator", kind: .streamingIndicator))
                    }
                }
            }
        }

        // Prune expandedDetails against actual tool call row IDs
        if needsDetailPruning {
            expandedDetails = expandedDetails.intersection(allToolCallRowIDs)
        }

        rows = newRows

        Log.agent.debug("[Perf] rebuildRows: \(allMessages.count) msgs → \(turns.count) turns → \(newRows.count) rows")
        ChatViewPerf.signposter.endInterval("rebuildRows", perfState)
    }

    /// Counts live streaming steps from the current `streamMessage`.
    private func streamingStepCount() -> Int {
        guard let stream = agent.state.streamMessage else { return 0 }
        var count = 0
        if let thinking = stream.thinking, !thinking.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            count += 1
        }
        count += stream.toolCalls.count
        let trimmed = stream.content.trimmingCharacters(in: .whitespacesAndNewlines)
        if !stream.toolCalls.isEmpty && !trimmed.isEmpty { count += 1 }
        return count
    }

    /// Emits the unified streaming header row. Returns whether the header was emitted.
    @discardableResult
    private func appendStreamingHeader(to newRows: inout [ChatRow], totalStepCount: Int) -> Bool {
        guard totalStepCount > 0 else { return false }
        if !streamingManuallyCollapsed {
            expandedTurns.insert(Self.streamingTurnID)
        }
        newRows.append(ChatRow(
            id: "streaming-header",
            kind: .turnHeader(TurnHeaderRow(
                stepCount: totalStepCount,
                isGenerating: true,
                turnID: Self.streamingTurnID,
                isExpanded: expandedTurns.contains(Self.streamingTurnID)
            ))
        ))
        return true
    }

    /// Emits live streaming step rows (thinking, tool calls, text). No header.
    private func appendStreamingStepRows(to newRows: inout [ChatRow]) {
        guard let stream = agent.state.streamMessage else { return }
        let trimmedStreamContent = stream.content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard expandedTurns.contains(Self.streamingTurnID) else {
            // Collapsed — only emit final streaming answer (no tool calls = plain text)
            if !trimmedStreamContent.isEmpty && stream.toolCalls.isEmpty {
                newRows.append(ChatRow(
                    id: "streaming-answer",
                    kind: .streamingText(StreamingTextRow(content: trimmedStreamContent))
                ))
            }
            return
        }

        // Streaming thinking
        if let thinking = stream.thinking?.trimmingCharacters(in: .whitespacesAndNewlines),
           !thinking.isEmpty {
            newRows.append(ChatRow(
                id: "streaming-thinking",
                kind: .thinking(ThinkingRow(
                    content: thinking,
                    isLast: stream.toolCalls.isEmpty && trimmedStreamContent.isEmpty
                ))
            ))
        }

        // Streaming tool calls
        for (index, info) in stream.toolCalls.enumerated() {
            let props = ToolDisplayHelpers.displayProps(for: info)

            newRows.append(ChatRow(
                id: "streaming-tool-\(index)",
                kind: .toolCall(ToolCallRow(
                    displayTitle: props.title,
                    iconName: props.icon,
                    argumentsFormatted: props.argsFormatted,
                    resultContent: nil,
                    isError: false,
                    isLast: index == stream.toolCalls.count - 1 && trimmedStreamContent.isEmpty,
                    isDetailExpanded: false,
                    filePath: props.filePath
                ))
            ))
        }

        // Streaming text
        if !trimmedStreamContent.isEmpty {
            if stream.toolCalls.isEmpty {
                newRows.append(ChatRow(
                    id: "streaming-answer",
                    kind: .streamingText(StreamingTextRow(content: trimmedStreamContent))
                ))
            } else {
                newRows.append(ChatRow(
                    id: "streaming-text",
                    kind: .toolText(ToolTextRow(content: trimmedStreamContent, isLast: true))
                ))
            }
        }
    }

    /// Throttled streaming update — patches only the active turn's rows.
    private func updateStreamingRows() {
        let now = ContinuousClock.now
        guard now - lastStreamingUpdate >= .milliseconds(50) else { return }
        lastStreamingUpdate = now

        ChatViewPerf.signposter.emitEvent("updateStreamingRows")
        patchStreamingTail()
        streamingRowVersion &+= 1
    }

    /// Fast path: rebuilds only the active (last) turn's rows + streaming rows.
    /// Prior turns are immutable during streaming, so we splice onto the stable prefix.
    private func patchStreamingTail() {
        guard isGenerating, activeTurnRowIndex <= rows.count else {
            rebuildRows()
            return
        }

        let allMessages = agent.state.messages

        // Find the last turn's messages — walk back from the end to find the turn boundary
        var lastTurnStart = allMessages.count
        for i in stride(from: allMessages.count - 1, through: 0, by: -1) {
            let msg = allMessages[i]
            if msg.asUser != nil || msg is CompactionSummaryMessage {
                lastTurnStart = i
                break
            }
        }

        let lastTurnMessages = Array(allMessages[lastTurnStart...])
        let user = lastTurnMessages.first?.asUser
        let compaction = lastTurnMessages.first as? CompactionSummaryMessage
        let turnID = lastTurnMessages.first?.messageUUID ?? UUID()

        // Build the tail rows for the active turn
        var tailRows: [ChatRow] = []

        // User row
        if let user, !user.content.isEmpty {
            tailRows.append(ChatRow(
                id: turnID.uuidString,
                kind: .user(UserRow(
                    content: user.content,
                    images: user.images,
                    timestamp: timeFormatter.string(from: user.timestamp),
                    messageID: turnID
                ))
            ))
        }

        // System (compaction) row
        if let compaction {
            tailRows.append(ChatRow(
                id: turnID.uuidString + "-system",
                kind: .system(SystemRow(content: "[Context compacted — \(compaction.tokensBefore) tokens summarized]"))
            ))
        }

        // Process assistant messages for the active turn
        var toolResultMap: [String: ToolResultMessage] = [:]
        for msg in lastTurnMessages {
            if let tr = msg.asToolResult { toolResultMap[tr.toolCallId] = tr }
        }
        let lastAssistantMsg = lastTurnMessages.last(where: { $0.asAssistant != nil })?.asAssistant

        var steps: [ChatRow] = []
        var answerContent: String?
        var answerTimestamp: String?
        var answerMessageID: UUID?

        for msg in lastTurnMessages {
            if msg is CompactionSummaryMessage { continue }
            if msg.asUser != nil || msg.asToolResult != nil { continue }
            guard let asst = msg.asAssistant else { continue }

            if let thinking = asst.thinking?.trimmingCharacters(in: .whitespacesAndNewlines),
               !thinking.isEmpty {
                let rowID = "\(asst.id)-thinking"
                steps.append(ChatRow(id: rowID, kind: .thinking(ThinkingRow(content: thinking, isLast: false))))
            }

            let isFinalAnswer = asst.id == lastAssistantMsg?.id && asst.toolCalls.isEmpty
            let trimmedContent = asst.content.trimmingCharacters(in: .whitespacesAndNewlines)

            if !trimmedContent.isEmpty && !isFinalAnswer {
                let rowID = "\(asst.id)-text"
                steps.append(ChatRow(id: rowID, kind: .toolText(ToolTextRow(content: trimmedContent, isLast: false))))
            }

            for (index, info) in asst.toolCalls.enumerated() {
                let props = ToolDisplayHelpers.displayProps(for: info)
                let result = toolResultMap[info.id]
                let rowID = "\(asst.id)-tool-\(index)"
                steps.append(ChatRow(id: rowID, kind: .toolCall(ToolCallRow(
                    displayTitle: props.title, iconName: props.icon,
                    argumentsFormatted: props.argsFormatted,
                    resultContent: result?.content.textContent,
                    isError: result?.isError ?? false, isLast: false,
                    isDetailExpanded: expandedDetails.contains(rowID),
                    filePath: props.filePath
                ))))
            }

            if isFinalAnswer && !trimmedContent.isEmpty {
                answerContent = trimmedContent
                answerTimestamp = timeFormatter.string(from: asst.timestamp)
                answerMessageID = asst.id
            }
        }

        let liveSteps = streamingStepCount()
        let hasCommittedContent = !steps.isEmpty || answerContent != nil

        if hasCommittedContent {
            // Streaming header + committed steps + streaming steps + answer
            appendStreamingHeader(to: &tailRows, totalStepCount: steps.count + liveSteps)

            if !steps.isEmpty {
                let showSteps = expandedTurns.contains(Self.streamingTurnID)
                if showSteps {
                    let hasLiveStepsAfter = liveSteps > 0
                    for (index, step) in steps.enumerated() {
                        let isLast = index == steps.count - 1 && !hasLiveStepsAfter
                        tailRows.append(isLast ? step.withIsLast(true) : step)
                    }
                }
            }

            appendStreamingStepRows(to: &tailRows)

            if let content = answerContent, let timestamp = answerTimestamp, let messageID = answerMessageID {
                tailRows.append(ChatRow(
                    id: messageID.uuidString + "-answer",
                    kind: .assistantText(AssistantTextRow(
                        content: content, timestamp: timestamp,
                        messageID: messageID, hasStepsAbove: !steps.isEmpty
                    ))
                ))
            }
        } else {
            // No committed assistant messages yet — same logic as rebuildRows()
            if liveSteps > 0 {
                appendStreamingHeader(to: &tailRows, totalStepCount: liveSteps)
                appendStreamingStepRows(to: &tailRows)
            } else if agent.state.streamMessage == nil {
                tailRows.append(ChatRow(id: "streaming-indicator", kind: .streamingIndicator))
            } else {
                appendStreamingStepRows(to: &tailRows)
                if tailRows.last?.id.hasPrefix("streaming-") != true {
                    tailRows.append(ChatRow(id: "streaming-indicator", kind: .streamingIndicator))
                }
            }
        }

        // Splice: keep stable prefix, replace active turn tail
        let spliceIndex = min(activeTurnRowIndex, rows.count)
        rows.replaceSubrange(spliceIndex..., with: tailRows)
    }

    // MARK: - System Prompt

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

    private func resetState() {
        expandedTurns.removeAll()
        expandedDetails.removeAll()
        autoExpandedTurnID = nil
        streamingManuallyCollapsed = false
        activeTurnRowIndex = 0
        rows = []
        isGenerating = false
    }

    /// Auto-expand the last assistant turn when generation starts.
    private func autoExpandLastTurn() {
        for msg in agent.state.messages.reversed() {
            if let u = msg.asUser {
                if !expandedTurns.contains(u.id) {
                    expandedTurns.insert(u.id)
                    autoExpandedTurnID = u.id
                }
                return
            }
        }
    }

    /// Auto-collapse the turn that was auto-expanded, unless user manually toggled it.
    private func autoCollapseIfNeeded() {
        if let turnID = autoExpandedTurnID, expandedTurns.contains(turnID) {
            expandedTurns.remove(turnID)
        }
        autoExpandedTurnID = nil
    }

    /// Finds the last assistant message content directly from agent state.
    private func lastAssistantContent() -> String? {
        for msg in agent.state.messages.reversed() {
            if let asst = msg.asAssistant {
                let text = asst.content.trimmingCharacters(in: .whitespacesAndNewlines)
                return text.isEmpty ? nil : text
            }
        }
        return nil
    }

    private func persistCurrentConversation() {
        guard !agent.state.messages.isEmpty else { return }
        conversationStore.updateCurrentMessages(agent.state.messages.map { $0 as any AgentMessageProtocol & Sendable })
        conversationStore.saveCurrent()
    }

    private func autoSpeakIfEnabled() {
        guard let settings, settings.agentAutoSpeak,
              let content = lastAssistantContent()
        else { return }
        Log.agent.info("Auto-speaking response (\(content.count) chars)")
        speechCoordinator?.speakText(content)
    }

}

// MARK: - StubExtensionContext

/// Minimal ExtensionContext for slash command execution.
/// A full implementation will be added when the extension runner is wired.
private struct StubExtensionContext: ExtensionContext, Sendable {
    let cwd: String
    var model: AgentModelRef? { nil }
    func isIdle() -> Bool { true }
    func abort() {}
    func getSystemPrompt() -> String { "" }
    func getContextUsage() -> ContextUsage? { nil }
    func compact(options: CompactOptions?) {}
}
