import Foundation
import Observation
import os
import MLXLMCommon

/// Thin UI bridge that delegates to ``Agent`` (new core loop).
///
/// `agent.state.messages` is the **single source of truth**. The coordinator
/// derives `rows: [ChatRow]` for view consumption via a two-tier cache:
/// 1. Immutable `TurnDisplayData` keyed by message IDs (computed once per cache miss)
/// 2. Overlay state (expansion) stamped during assembly without recomputing strings
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

    // MARK: - Turn Cache

    /// Immutable content cache keyed by turnID. Stores pre-computed display strings.
    @ObservationIgnored private var turnDisplayCache: [UUID: TurnDisplayData] = [:]
    /// Ordered turn references — computed once in `rebuildRows()`, reused by `assembleRowsFromCache()`.
    @ObservationIgnored private var turnOrder: [TurnRef] = []
    /// Expanded turn headers (step timeline visible).
    @ObservationIgnored private var expandedTurns: Set<UUID> = []
    /// Expanded tool call details (arguments/results visible), keyed by row ID.
    @ObservationIgnored private var expandedDetails: Set<String> = []
    /// Cached measured row heights — stabilizes List's height estimation for off-screen cells.
    @ObservationIgnored private var rowHeightCache: [String: CGFloat] = [:]
    /// Throttle streaming row updates.
    @ObservationIgnored private var lastStreamingUpdate: ContinuousClock.Instant = .now
    /// Turn that was auto-expanded for generation — auto-collapse on agentEnd.
    @ObservationIgnored private var autoExpandedTurnID: UUID?
    /// Cached streaming tool args to avoid re-parsing unchanged JSON on each tick.
    @ObservationIgnored private var lastStreamingToolArgs: [String] = []
    @ObservationIgnored private var lastStreamingToolDisplay: [(title: String, icon: String, argsFormatted: String)] = []
    /// Tracks if user manually collapsed the streaming header during this generation.
    @ObservationIgnored private var streamingManuallyCollapsed: Bool = false
    /// Stable ID for the streaming turn header — survives rebuilds so toggle state persists.
    private static let streamingTurnID = UUID(uuidString: "00000000-0000-0000-0000-000000000001")!

    // MARK: - Cache Data Types

    /// Lightweight turn reference for ordered iteration — no message content, just metadata for row assembly.
    private struct TurnRef {
        let id: UUID
        let userContent: String?
        let userImages: [ImageAttachment]
        let userTimestamp: String?
        let compactionText: String?
    }

    /// Immutable content for one assistant turn. No overlay state — that's stamped during assembly.
    private struct TurnDisplayData {
        let turnID: UUID
        let messageIDs: [UUID]
        let headerStepCount: Int
        let steps: [StepDisplayData]
        let answerContent: String?
        let answerTimestamp: String?
        let answerMessageID: UUID?
    }

    /// Pure content for one step — no expansion state.
    private struct StepDisplayData {
        let rowID: String
        let kind: StepKind

        enum StepKind {
            case thinking(content: String)
            case toolCall(displayTitle: String, iconName: String,
                          argumentsFormatted: String, resultContent: String?, isError: Bool)
            case text(content: String)
        }
    }

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
        toolRegistry: ToolRegistry? = nil
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

        assembledSystemPrompt = agent.state.systemPrompt

        subscribeToAgentEvents()

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

    func sendMessage(_ text: String, images: [ImageAttachment] = []) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty || !images.isEmpty else { return }

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
        clearAllCaches()
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
        clearAllCaches()
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
            clearAllCaches()
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
            clearAllCaches()
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
        clearAllCaches()
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

                // Model loading is handled by the arbiter inside sendMessage
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
        let beforeCount = rows.count
        let expanding = !expandedTurns.contains(turnID)
        expandedTurns.formSymmetricDifference([turnID])
        // Track manual collapse of the streaming header
        if turnID == Self.streamingTurnID && !expandedTurns.contains(Self.streamingTurnID) {
            streamingManuallyCollapsed = true
        }
        assembleRowsFromCache()
        Log.agent.debug("[Perf] toggleTurnExpanded: \(expanding ? "EXPAND" : "COLLAPSE") turn \(turnID) | rows \(beforeCount) → \(rows.count) (Δ\(rows.count - beforeCount))")
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

    // MARK: - Row Height Cache

    /// Returns the cached height for a row, or nil if not yet measured.
    func cachedHeight(for rowID: String) -> CGFloat? {
        rowHeightCache[rowID]
    }

    /// Caches a measured row height. Called from the view layer via onGeometryChange.
    func cacheRowHeight(_ height: CGFloat, for rowID: String) {
        rowHeightCache[rowID] = height
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

        case .messageUpdate:
            updateStreamingRows()

        case .toolExecutionStart(_, let toolName, _):
            Log.agent.info("Tool start: \(toolName)")
            lastStreamingToolArgs = []
            lastStreamingToolDisplay = []

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
            lastStreamingToolArgs = []
            lastStreamingToolDisplay = []
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

    /// Full rebuild: groups messages into turns, validates cache, stores turn order, assembles rows.
    private func rebuildRows() {
        let perfState = ChatViewPerf.signposter.beginInterval("rebuildRows")

        let allMessages = agent.state.messages

        // Group messages into turns — each turn starts with a user/system message.
        struct Turn {
            let id: UUID
            var messages: [any AgentMessageProtocol]
            var ref: TurnRef
        }

        var turns: [Turn] = []
        var currentTurn: Turn?

        for msg in allMessages {
            let user = msg.asUser
            let compaction = msg as? CompactionSummaryMessage

            if user != nil || compaction != nil {
                if let t = currentTurn { turns.append(t) }
                let turnID = msg.messageUUID
                let ref = TurnRef(
                    id: turnID,
                    userContent: user?.content,
                    userImages: user?.images ?? [],
                    userTimestamp: user.map { timeFormatter.string(from: $0.timestamp) },
                    compactionText: compaction.map { "[Context compacted — \($0.tokensBefore) tokens summarized]" }
                )
                currentTurn = Turn(id: turnID, messages: [msg], ref: ref)
            } else {
                if currentTurn == nil {
                    let turnID = msg.messageUUID
                    let ref = TurnRef(id: turnID, userContent: nil, userImages: [], userTimestamp: nil, compactionText: nil)
                    currentTurn = Turn(id: turnID, messages: [], ref: ref)
                }
                currentTurn?.messages.append(msg)
            }
        }
        if let t = currentTurn { turns.append(t) }

        // Validate cache for each turn
        var activeTurnIDs: Set<UUID> = []
        var newTurnOrder: [TurnRef] = []
        var cacheHits = 0
        var cacheMisses = 0

        for turn in turns {
            activeTurnIDs.insert(turn.id)
            newTurnOrder.append(turn.ref)

            // Compaction-only turns don't need cache — they only emit a .system row
            if turn.ref.compactionText != nil && turn.ref.userContent == nil { continue }

            let messageIDs = turn.messages.map { $0.messageUUID }

            if let cached = turnDisplayCache[turn.id],
               cached.messageIDs == messageIDs {
                cacheHits += 1
                continue // Cache hit
            }

            cacheMisses += 1
            turnDisplayCache[turn.id] = computeTurnDisplayData(
                turnID: turn.id, messages: turn.messages, messageIDs: messageIDs
            )
        }

        // Prune stale entries
        let staleTurnIDs = Set(turnDisplayCache.keys).subtracting(activeTurnIDs)
        for id in staleTurnIDs {
            turnDisplayCache.removeValue(forKey: id)
        }
        var validTurnIDs = activeTurnIDs
        // Preserve streaming turn expansion state during generation — it's not a committed turn
        if isGenerating { validTurnIDs.insert(Self.streamingTurnID) }
        expandedTurns = expandedTurns.intersection(validTurnIDs)

        if !expandedDetails.isEmpty {
            let allStepRowIDs = Set(turnDisplayCache.values.flatMap { $0.steps.map(\.rowID) })
            expandedDetails = expandedDetails.intersection(allStepRowIDs)
        }

        turnOrder = newTurnOrder
        assembleRowsFromCache()

        Log.agent.debug("[Perf] rebuildRows: \(allMessages.count) msgs → \(turns.count) turns → \(rows.count) rows | cache \(cacheHits) hit / \(cacheMisses) miss | pruned \(staleTurnIDs.count)")
        ChatViewPerf.signposter.endInterval("rebuildRows", perfState)
    }

    /// Computes immutable display data for one turn from protocol messages.
    private func computeTurnDisplayData(turnID: UUID, messages: [any AgentMessageProtocol], messageIDs: [UUID]) -> TurnDisplayData {
        let perfState = ChatViewPerf.signposter.beginInterval("computeTurnDisplayData")
        defer { ChatViewPerf.signposter.endInterval("computeTurnDisplayData", perfState) }
        var steps: [StepDisplayData] = []
        var answerContent: String?
        var answerTimestamp: String?
        var answerMessageID: UUID?

        // Build tool result lookup: toolCallId → ToolResultMessage
        var toolResultMap: [String: ToolResultMessage] = [:]
        for msg in messages {
            if let tr = msg.asToolResult {
                toolResultMap[tr.toolCallId] = tr
            }
        }

        // Find last assistant message for final answer detection
        let lastAssistantMsg = messages.last(where: { $0.asAssistant != nil })?.asAssistant

        for msg in messages {
            // Skip compaction (handled as .system row), user, and bare tool results
            if msg is CompactionSummaryMessage { continue }
            if msg.asUser != nil || msg.asToolResult != nil { continue }

            guard let asst = msg.asAssistant else { continue }

            // Thinking
            if let thinking = asst.thinking?.trimmingCharacters(in: .whitespacesAndNewlines),
               !thinking.isEmpty {
                steps.append(StepDisplayData(
                    rowID: "\(asst.id)-thinking",
                    kind: .thinking(content: thinking)
                ))
            }

            let isFinalAnswer = asst.id == lastAssistantMsg?.id && asst.toolCalls.isEmpty
            let trimmedContent = asst.content.trimmingCharacters(in: .whitespacesAndNewlines)

            if !trimmedContent.isEmpty && !isFinalAnswer {
                steps.append(StepDisplayData(
                    rowID: "\(asst.id)-text",
                    kind: .text(content: trimmedContent)
                ))
            }

            // Tool calls — match results by toolCallId
            for (index, info) in asst.toolCalls.enumerated() {
                let props = ToolDisplayHelpers.displayProps(for: info)
                let result = toolResultMap[info.id]

                steps.append(StepDisplayData(
                    rowID: "\(asst.id)-tool-\(index)",
                    kind: .toolCall(
                        displayTitle: props.title,
                        iconName: props.icon,
                        argumentsFormatted: props.argsFormatted,
                        resultContent: result?.content.textContent,
                        isError: result?.isError ?? false
                    )
                ))
            }

            // Final answer
            if isFinalAnswer && !trimmedContent.isEmpty {
                answerContent = trimmedContent
                answerTimestamp = timeFormatter.string(from: asst.timestamp)
                answerMessageID = asst.id
            }
        }

        return TurnDisplayData(
            turnID: turnID,
            messageIDs: messageIDs,
            headerStepCount: steps.count,
            steps: steps,
            answerContent: answerContent,
            answerTimestamp: answerTimestamp,
            answerMessageID: answerMessageID
        )
    }

    /// Stamps overlay state (expansion) onto immutable cached content to produce `rows`.
    /// Iterates stored `turnOrder` — no message walk needed.
    private func assembleRowsFromCache() {
        let perfState = ChatViewPerf.signposter.beginInterval("assembleRowsFromCache")
        let oldRowIDs = Set(rows.map(\.id))
        var newRows: [ChatRow] = []

        for turnRef in turnOrder {
            // User row
            if let content = turnRef.userContent, let timestamp = turnRef.userTimestamp {
                newRows.append(ChatRow(
                    id: turnRef.id.uuidString,
                    kind: .user(UserRow(
                        content: content,
                        images: turnRef.userImages,
                        timestamp: timestamp,
                        messageID: turnRef.id
                    ))
                ))
            }

            // System (compaction) row
            if let text = turnRef.compactionText {
                newRows.append(ChatRow(
                    id: turnRef.id.uuidString + "-system",
                    kind: .system(SystemRow(content: text))
                ))
            }

            guard let cached = turnDisplayCache[turnRef.id] else { continue }

            let isExpanded = expandedTurns.contains(turnRef.id)
            let isLastTurn = turnRef.id == turnOrder.last?.id

            let isActiveTurn = isGenerating && isLastTurn

            // For the active turn during generation, emit the streaming header FIRST,
            // then committed steps under it, then live streaming steps — one unified section.
            if isActiveTurn {
                let (liveSteps, _) = streamingStepCount()
                appendStreamingHeader(to: &newRows, totalStepCount: cached.headerStepCount + liveSteps)
            }

            // Turn header — for non-active turns only (active turn uses the streaming header above)
            if cached.headerStepCount > 0 && !isActiveTurn {
                newRows.append(ChatRow(
                    id: "\(turnRef.id)-header",
                    kind: .turnHeader(TurnHeaderRow(
                        stepCount: cached.headerStepCount,
                        isGenerating: false,
                        turnID: turnRef.id,
                        isExpanded: isExpanded
                    ))
                ))
            }

            // Step rows
            if cached.headerStepCount > 0 {
                let showSteps = isActiveTurn
                    ? expandedTurns.contains(Self.streamingTurnID)
                    : isExpanded
                if showSteps {
                    // When live streaming steps follow, the last committed step is not truly last
                    let (liveSteps, _) = isActiveTurn ? streamingStepCount() : (0, "")
                    let hasLiveStepsAfter = isActiveTurn && liveSteps > 0
                    for (index, step) in cached.steps.enumerated() {
                        let isLast = index == cached.steps.count - 1 && !hasLiveStepsAfter
                        switch step.kind {
                        case .thinking(let content):
                            newRows.append(ChatRow(
                                id: step.rowID,
                                kind: .thinking(ThinkingRow(content: content, isLast: isLast))
                            ))
                        case .toolCall(let title, let icon, let args, let result, let isError):
                            newRows.append(ChatRow(
                                id: step.rowID,
                                kind: .toolCall(ToolCallRow(
                                    displayTitle: title,
                                    iconName: icon,
                                    argumentsFormatted: args,
                                    resultContent: result,
                                    isError: isError,
                                    isLast: isLast,
                                    isDetailExpanded: expandedDetails.contains(step.rowID)
                                ))
                            ))
                        case .text(let content):
                            newRows.append(ChatRow(
                                id: step.rowID,
                                kind: .toolText(ToolTextRow(content: content, isLast: isLast))
                            ))
                        }
                    }
                }
            }

            // Live streaming step rows — after committed steps, before answer
            if isActiveTurn {
                appendStreamingStepRows(to: &newRows)
            }

            // Answer row
            if let content = cached.answerContent,
               let timestamp = cached.answerTimestamp,
               let messageID = cached.answerMessageID {
                newRows.append(ChatRow(
                    id: messageID.uuidString + "-answer",
                    kind: .assistantText(AssistantTextRow(
                        content: content,
                        timestamp: timestamp,
                        messageID: messageID,
                        hasStepsAbove: cached.headerStepCount > 0
                    ))
                ))
            }
        }

        // Streaming rows for turns with no committed cache (e.g. first message, no turnEnd yet)
        if isGenerating && turnOrder.last.flatMap({ turnDisplayCache[$0.id] }) == nil {
            let (liveSteps, _) = streamingStepCount()
            if liveSteps > 0 {
                appendStreamingHeader(to: &newRows, totalStepCount: liveSteps)
                appendStreamingStepRows(to: &newRows)
            } else if agent.state.streamMessage == nil {
                newRows.append(ChatRow(id: "streaming-indicator", kind: .streamingIndicator))
            } else {
                // Has streamMessage but no steps — plain text streaming answer
                appendStreamingStepRows(to: &newRows)
                if newRows.last?.id.hasPrefix("streaming-") != true {
                    newRows.append(ChatRow(id: "streaming-indicator", kind: .streamingIndicator))
                }
            }
        }

        // Row diff analysis
        let newRowIDs = Set(newRows.map(\.id))
        let inserted = newRowIDs.subtracting(oldRowIDs)
        let removed = oldRowIDs.subtracting(newRowIDs)
        if !inserted.isEmpty || !removed.isEmpty {
            Log.agent.debug("[Perf] assembleRows: \(rows.count) → \(newRows.count) rows | +\(inserted.count) -\(removed.count)")
            if !inserted.isEmpty, inserted.count <= 20 {
                Log.agent.debug("[Perf]   inserted: \(inserted.sorted().joined(separator: ", "))")
            }
            if !removed.isEmpty, removed.count <= 20 {
                Log.agent.debug("[Perf]   removed: \(removed.sorted().joined(separator: ", "))")
            }
        }

        rows = newRows
        ChatViewPerf.signposter.endInterval("assembleRowsFromCache", perfState)
    }

    /// Counts live streaming steps from the current `streamMessage`.
    private func streamingStepCount() -> (steps: Int, trimmedContent: String) {
        guard let stream = agent.state.streamMessage else { return (0, "") }
        let trimmed = stream.content.trimmingCharacters(in: .whitespacesAndNewlines)
        var count = 0
        if let thinking = stream.thinking, !thinking.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            count += 1
        }
        count += stream.toolCalls.count
        if !stream.toolCalls.isEmpty && !trimmed.isEmpty { count += 1 }
        return (count, trimmed)
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
            let props: (title: String, icon: String, argsFormatted: String)
            if index < lastStreamingToolArgs.count, lastStreamingToolArgs[index] == info.argumentsJSON {
                props = lastStreamingToolDisplay[index]
            } else {
                props = ToolDisplayHelpers.displayProps(for: info)
                if index < lastStreamingToolArgs.count {
                    lastStreamingToolArgs[index] = info.argumentsJSON
                    lastStreamingToolDisplay[index] = props
                } else {
                    lastStreamingToolArgs.append(info.argumentsJSON)
                    lastStreamingToolDisplay.append(props)
                }
            }

            newRows.append(ChatRow(
                id: "streaming-tool-\(index)",
                kind: .toolCall(ToolCallRow(
                    displayTitle: props.title,
                    iconName: props.icon,
                    argumentsFormatted: props.argsFormatted,
                    resultContent: nil,
                    isError: false,
                    isLast: index == stream.toolCalls.count - 1 && trimmedStreamContent.isEmpty,
                    isDetailExpanded: false
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

    /// Throttled streaming update: replaces only streaming rows at the tail.
    private func updateStreamingRows() {
        let now = ContinuousClock.now
        guard now - lastStreamingUpdate >= .milliseconds(50) else { return }
        lastStreamingUpdate = now

        ChatViewPerf.signposter.emitEvent("updateStreamingRows")

        // Streaming header is interleaved with committed steps, so we reassemble
        // the full row list. assembleRowsFromCache is O(turns) with no string work.
        assembleRowsFromCache()
        streamingRowVersion &+= 1
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

    private func clearAllCaches() {
        turnDisplayCache.removeAll()
        turnOrder = []
        expandedTurns.removeAll()
        expandedDetails.removeAll()
        rowHeightCache.removeAll()
        autoExpandedTurnID = nil
        streamingManuallyCollapsed = false
        lastStreamingToolArgs = []
        lastStreamingToolDisplay = []
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
