import Foundation
import Observation

// MARK: - ChatRunPhase

/// What the run is doing right now — drives the composer's status affordances
/// and the transcript's live indicator.
enum ChatRunPhase: Equatable, Sendable {
    case idle
    case streaming
    /// The model is writing a tool call (the Open Tool Call is live) — the
    /// tool row owns the live indicator, so the transcript's quiet spinner
    /// stays hidden. Associated value: the tool name.
    case writingTool(String)
    case executingTool(String)
    case transformingContext(ContextTransformReason)
}

// MARK: - ChatItem

/// One committed transcript entry — a pure value row source. Assistant items
/// render one row per Content Part (identity = message id + part index);
/// tool-result items render nothing standalone (their content shows under the
/// matching tool-call row via ``ChatSession/toolResult(for:)``).
enum ChatItem: Identifiable, Equatable {
    case user(UserMessage)
    case assistant(AssistantMessage)
    case toolResult(ToolResultMessage)
    case system(id: UUID, text: String)

    var id: UUID {
        switch self {
        case .user(let msg): return msg.id
        case .assistant(let msg): return msg.id
        case .toolResult(let msg): return msg.id
        case .system(let id, _): return id
        }
    }
}

// MARK: - ChatSession

/// The **Chat Session** (ADR-0024): the single `@Observable` store holding the
/// agent-event fold — committed messages, the Live Part, the run phase — as
/// the *only* agent-event subscriber. Everything not derived from agent events
/// (composer draft, voice input, skill pills, command palette) lives in leaf
/// controllers owned by the views that use them; no central dispatcher exists.
///
/// The fold (`handle(_:)`) is the primary test seam: scripted event sequences
/// in, folded `items` / Live Part lifecycle / run-phase transitions out.
@Observable @MainActor
final class ChatSession {

    // MARK: - Observable state (the event fold)

    /// Committed transcript items, in order. Pure values — a commit replaces
    /// the array element-wise; per-token updates never touch this.
    private(set) var items: [ChatItem] = []

    /// The in-flight assistant message: its *committed* parts. Updated only at
    /// part boundaries (start/end/toolcall) — never per delta.
    private(set) var liveMessage: AssistantMessage?

    /// The one observable box for the currently-streaming part, if any.
    private(set) var livePart: LivePart?

    /// The **Pending Row**: the just-sent user message, rendered from send
    /// until the event spine commits the same message — so the transcript
    /// shows it instantly even while the run sits queued behind the lease
    /// (cold-start model load). Ephemeral derived view-state like the Live
    /// Part, never agent state. If the run settles before the commit (cancel
    /// while queued, load failure), it vanishes and its content is restored
    /// to the composer via `restoreComposerDraft`.
    private(set) var pendingUserMessage: UserMessage?

    /// The run phase, folded from lifecycle + tool events.
    private(set) var runPhase: ChatRunPhase = .idle

    /// The single notice feed for the in-composer banner slot. Set by
    /// `generationError` and command failures; cleared on the next send or by
    /// the banner's dismiss.
    var error: String?

    /// "A foreground run is queued or active" — includes lease-queue time the
    /// event fold can't see. The composer's send/cancel switch keys off this.
    var isGenerating: Bool { agentRun.isGenerating }

    /// Whether the **Waiting Row** shows: the run is waiting on the model with
    /// nothing streaming — queued behind the lease (cold start) or in a turn
    /// prefill (first turn and after every tool batch). Deliberately *not*
    /// shown between parts mid-stream (the live message already has content —
    /// a row there would flicker) or during tool execution (tool rows own
    /// their spinners). The view layer adds the `promptStartsThinking` gate.
    var showsWaitingRow: Bool {
        guard isGenerating, livePart == nil else { return false }
        guard liveMessage?.content.isEmpty ?? true else { return false }
        switch runPhase {
        case .idle, .streaming: return true
        case .writingTool, .executingTool, .transformingContext: return false
        }
    }

    // MARK: - Non-observable fold state

    /// Tool results keyed by tool-call id, for O(1) lookup under tool rows.
    /// Observable — a tool row re-renders when its result lands.
    private var toolResultsByCallID: [String: ToolResultMessage] = [:]

    /// Tool calls currently executing (run-phase bookkeeping).
    @ObservationIgnored private var pendingToolCalls: Set<String> = []

    /// The Tool Clock: one wall clock per tool call, from `toolcallStart`
    /// (name-lock, when the row first appears) through writing, waiting, and
    /// execution to `toolExecutionEnd` — the frozen badge reads "time from
    /// first visible to result". `toolStartInstants` is the ticking clock's
    /// anchor; `toolExecutionStart` never resets an existing instant.
    private var toolDurations: [String: Duration] = [:]
    @ObservationIgnored private var toolStartInstants: [String: ContinuousClock.Instant] = [:]

    /// A thinking part's identity: parts carry no IDs, so message id + content
    /// index — stable across the `turnEnd` resync — key its duration.
    struct ThinkingPartKey: Hashable {
        let messageID: UUID
        let partIndex: Int
    }

    /// Wall-clock streaming time per finished thinking part — shown in the
    /// expanded thought row. Session-scoped like `toolDurations` (gone on
    /// conversation reload). Start instants are recorded on `thinkingStart`.
    private var thinkingDurations: [ThinkingPartKey: Duration] = [:]
    @ObservationIgnored private var thinkingStartInstants:
        [ThinkingPartKey: ContinuousClock.Instant] = [:]

    /// Throttle for the Live Part's markdown republish (~10 Hz in production;
    /// `.zero` in tests for determinism).
    @ObservationIgnored private let liveThrottle: Duration

    // MARK: - Dependencies

    private let agent: Agent
    let agentRun: AgentRunController
    private let conversationStore: any AgentConversationStoring
    private let settings: SettingsManager?
    private let speechCoordinator: SpeechCoordinator?
    private let contextManager: ContextManager?
    private let contextWindow: Int
    private let summarize: (@Sendable (String) async throws -> String)?
    /// The slash-command registry, provided by the view-owned Command Palette
    /// leaf so parsing here can never drift from the popup's listing.
    private let commandRegistry: @MainActor () -> SlashCommandRegistry
    /// Skill-invocation assembly + usage recording, provided by the view-owned
    /// Skill Pill leaf (identity/no-op when absent, e.g. in tests).
    private let assembleSkillArguments:
        @MainActor (_ skillName: String, _ userText: String) -> String
    private let recordSkillInvocation: @MainActor (_ skillName: String) -> Void
    /// `/clear` discards the view-owned composer draft through this hook.
    private let clearComposerDraft: @MainActor () -> Void
    /// A run that settles before its user message committed (cancel while
    /// queued, load failure) hands the Pending Row's content back to the
    /// view-owned composer through this hook — the message was never sent,
    /// so silently discarding it is not an option.
    private let restoreComposerDraft: @MainActor (String, [ImageAttachment]) -> Void
    /// Fired on every conversation boundary (new / load / delete-current) so
    /// the view-owned leaves can react: System Prompt Inspector re-probe,
    /// Skill Pill ranking recompute, ephemeral composer state reset. The
    /// Composer Draft's *content* deliberately rides across the switch.
    private let onConversationSwitch: @MainActor () -> Void
    private let debugLogger = AgentDebugLogger()

    @ObservationIgnored private var unsubscribe: (@MainActor () -> Void)?

    // MARK: - Init

    init(
        agent: Agent,
        conversationStore: any AgentConversationStoring,
        arbiter: any InferenceArbitrating,
        toolRegistry: ToolRegistry? = nil,
        settings: SettingsManager? = nil,
        speechCoordinator: SpeechCoordinator? = nil,
        contextManager: ContextManager? = nil,
        contextWindow: Int = 262_144,
        summarize: (@Sendable (String) async throws -> String)? = nil,
        commandRegistry: @MainActor @escaping () -> SlashCommandRegistry = {
            SlashCommandRegistry()
        },
        assembleSkillArguments: @MainActor @escaping (String, String) -> String = { _, text in text
        },
        recordSkillInvocation: @MainActor @escaping (String) -> Void = { _ in },
        clearComposerDraft: @MainActor @escaping () -> Void = {},
        restoreComposerDraft: @MainActor @escaping (String, [ImageAttachment]) -> Void = { _, _ in
        },
        onConversationSwitch: @MainActor @escaping () -> Void = {},
        liveMarkdownThrottle: Duration = .milliseconds(100)
    ) {
        self.agent = agent
        self.conversationStore = conversationStore
        self.settings = settings
        self.speechCoordinator = speechCoordinator
        self.contextManager = contextManager
        self.contextWindow = contextWindow
        self.summarize = summarize
        self.commandRegistry = commandRegistry
        self.assembleSkillArguments = assembleSkillArguments
        self.recordSkillInvocation = recordSkillInvocation
        self.clearComposerDraft = clearComposerDraft
        self.restoreComposerDraft = restoreComposerDraft
        self.onConversationSwitch = onConversationSwitch
        self.liveThrottle = liveMarkdownThrottle
        self.agentRun = AgentRunController(
            agent: agent, arbiter: arbiter, toolRegistry: toolRegistry, settings: settings
        )

        agentRun.setReportError { [weak self] message in self?.error = message }
        // A settle with the Pending Row still up means the user message never
        // entered the agent context — hand it back to the composer.
        agentRun.setOnRunSettled { [weak self] in self?.settlePendingUserMessage() }

        unsubscribe = agent.subscribe { [weak self] event in
            Task { @MainActor in
                self?.handle(event)
            }
        }

        conversationStore.loadMostRecent()
        if let current = conversationStore.currentConversation {
            agent.loadMessages(current.messages)
            resync(from: current.messages)
        }
    }

    deinit {
        let unsub = unsubscribe
        if let unsub {
            MainActor.assumeIsolated { unsub() }
        }
    }

    // MARK: - The event fold (test seam 1)

    /// Fold one agent event into the session. Total over `AgentEvent` — adding
    /// a case forces a decision here.
    func handle(_ event: AgentEvent) {
        switch event {
        case .agentStart:
            agentRun.markStarted()
            runPhase = .streaming

        case .agentEnd:
            agentRun.finish()
            clearLiveState()
            runPhase = .idle
            pendingToolCalls.removeAll()
            toolStartInstants.removeAll()
            thinkingStartInstants.removeAll()
            autoSpeakIfEnabled()

        case .generationError(let message):
            // Sticky until the user acts: `sendMessage` clears it on the next
            // turn, and the banner has a manual dismiss.
            error = message

        case .turnStart:
            break

        case .turnEnd(let message, let toolResults, let contextMessages):
            debugLogger.logTurn(
                message: message,
                toolResults: toolResults,
                messageCount: contextMessages.count
            )
            // Authoritative resync — the loop's context snapshot carries tool
            // results and commits the streaming path may have raced.
            resync(from: contextMessages)
            persistCurrentConversation()

        case .contextTransformStart(let reason):
            runPhase = .transformingContext(reason)

        case .contextTransformEnd(_, let didMutate, let messages):
            if didMutate, let messages {
                resync(from: messages)
            }
            runPhase = .streaming

        case .messageStart(let message):
            if let assistant = message.asAssistant {
                liveMessage = assistant
            }

        case .messageUpdate(_, let streamEvent):
            fold(streamEvent)

        case .messageEnd(let message):
            commit(message)

        case .malformedToolCall:
            // The Open Tool Call was retracted with no stream event of its
            // own — leave the writing phase now rather than waiting for the
            // next part boundary.
            endWritingToolPhaseIfNeeded()

        case .toolExecutionStart(let id, let name, _):
            pendingToolCalls.insert(id)
            anchorToolClock(id)
            runPhase = .executingTool(name)

        case .toolExecutionUpdate:
            break

        case .toolExecutionEnd(let id, _, _, _):
            pendingToolCalls.remove(id)
            if let start = toolStartInstants.removeValue(forKey: id) {
                toolDurations[id] = .now - start
            }
            if pendingToolCalls.isEmpty {
                runPhase = .streaming
            }
        }
    }

    /// Tool result for a tool-call part, if it has arrived.
    func toolResult(for toolCallID: String) -> ToolResultMessage? {
        toolResultsByCallID[toolCallID]
    }

    /// The Tool Clock's frozen reading for a finished tool call, if measured
    /// this session: name-lock through execution end.
    func toolDuration(for toolCallID: String) -> Duration? {
        toolDurations[toolCallID]
    }

    /// The Tool Clock's anchor for a still-active tool call — the live badge
    /// derives its ticking elapsed time from this. Nil once the call finishes
    /// (the frozen `toolDuration` takes over) or for calls not started this
    /// session. Not observable; the badge's `TimelineView` drives its own
    /// updates.
    func toolStartInstant(for toolCallID: String) -> ContinuousClock.Instant? {
        toolStartInstants[toolCallID]
    }

    /// Streaming duration for a committed thinking part, if measured this
    /// session.
    func thinkingDuration(messageID: UUID, partIndex: Int) -> Duration? {
        thinkingDurations[ThinkingPartKey(messageID: messageID, partIndex: partIndex)]
    }

    // MARK: - The assistant stream fold

    private func fold(_ event: AssistantMessageEvent) {
        // Only the Open Tool Call's own start/deltas sustain the writing
        // phase; any other event means the call committed (`toolcallEnd`) or
        // was retracted without an event of its own (text/thinking opening
        // after an unclosed block, terminal done/error).
        switch event {
        case .toolcallStart, .toolcallDelta: break
        default: endWritingToolPhaseIfNeeded()
        }

        switch event {
        case .start(let partial):
            liveMessage = partial

        case .textStart(let index, let partial):
            liveMessage = partial
            livePart = LivePart(
                messageID: partial.id, partIndex: index, kind: .text,
                initial: textOfPart(at: index, in: partial), throttle: liveThrottle
            )

        case .thinkingStart(let index, let partial):
            liveMessage = partial
            thinkingStartInstants[
                ThinkingPartKey(messageID: partial.id, partIndex: index)] = .now
            livePart = LivePart(
                messageID: partial.id, partIndex: index, kind: .thinking,
                initial: textOfPart(at: index, in: partial), throttle: liveThrottle
            )

        case .textDelta(let index, let delta, let partial),
            .thinkingDelta(let index, let delta, let partial):
            if let livePart, livePart.partIndex == index {
                livePart.append(delta)
            } else {
                // Missed the part start (shouldn't happen) — resync from the
                // snapshot the event carries.
                rebuildLivePart(at: index, from: partial)
            }

        case .textEnd(_, _, let partial):
            liveMessage = partial
            livePart = nil

        case .thinkingEnd(let index, _, let partial):
            liveMessage = partial
            livePart = nil
            let key = ThinkingPartKey(messageID: partial.id, partIndex: index)
            if let start = thinkingStartInstants.removeValue(forKey: key) {
                thinkingDurations[key] = .now - start
            }

        case .toolcallStart(let index, let partial):
            // The Open Tool Call is born (name-lock): the row appears, the
            // Tool Clock starts, and the writing phase hides the transcript's
            // quiet spinner. No Live Part — argument deltas are not rendered.
            liveMessage = partial
            if partial.content.indices.contains(index),
                case .toolCall(let part) = partial.content[index]
            {
                anchorToolClock(part.id)
                runPhase = .writingTool(part.name)
            }

        case .toolcallDelta:
            // Deliberately a rendering no-op: the row shows status only
            // (title, spinner, ticking badge); `liveMessage` republishes at
            // part boundaries. The Tool Clock ticks via its own TimelineView.
            break

        case .toolcallEnd(_, _, let partial):
            liveMessage = partial

        case .done(_, let message):
            liveMessage = message

        case .error(_, let message):
            // The partial is preserved; `messageEnd` commits it. The banner is
            // fed by the distinct `generationError` agent event.
            liveMessage = message
        }
    }

    /// Anchor the Tool Clock for a call id, never resetting an existing
    /// instant: the clock runs continuously from `toolcallStart` (name-lock)
    /// across the execution boundary. `toolExecutionStart` anchors only when
    /// the writing phase was never observed (e.g. resync mid-turn).
    private func anchorToolClock(_ toolCallID: String) {
        if toolStartInstants[toolCallID] == nil {
            toolStartInstants[toolCallID] = .now
        }
    }

    /// Leave `.writingTool` for `.streaming`. The fold fires this for every
    /// stream event that isn't the Open Tool Call's own start/delta; the
    /// malformed close (which emits no stream event) fires it from `handle`.
    private func endWritingToolPhaseIfNeeded() {
        if case .writingTool = runPhase {
            runPhase = .streaming
        }
    }

    private func commit(_ message: any AgentMessageProtocol) {
        if let user = message.asUser {
            // The agent accepted a message — the Pending Row's job is done.
            // Only one send is in flight at a time, so no id check is needed.
            pendingUserMessage = nil
            items.append(.user(user))
            return
        }
        if let toolResult = message.asToolResult {
            toolResultsByCallID[toolResult.toolCallId] = toolResult
            items.append(.toolResult(toolResult))
            return
        }
        if let assistant = message.asAssistant {
            clearLiveState()
            // Drop empty assistant turns from cancel/error paths — the same
            // `hasContent` rule `runLoop` folds against on persist.
            guard assistant.hasContent else { return }
            items.append(.assistant(assistant))
            return
        }
        if let compaction = message as? CompactionSummaryMessage {
            items.append(.system(id: UUID(), text: compaction.displayText))
        }
    }

    private func clearLiveState() {
        livePart = nil
        liveMessage = nil
    }

    private func textOfPart(at index: Int, in message: AssistantMessage) -> String {
        guard message.content.indices.contains(index) else { return "" }
        switch message.content[index] {
        case .text(let t): return t.text
        case .thinking(let t): return t.thinking
        case .toolCall: return ""
        }
    }

    private func rebuildLivePart(at index: Int, from partial: AssistantMessage) {
        liveMessage = partial
        guard partial.content.indices.contains(index) else { return }
        let kind: LivePart.Kind
        switch partial.content[index] {
        case .text: kind = .text
        case .thinking: kind = .thinking
        case .toolCall: return
        }
        if kind == .thinking {
            // Missed-start rebuild: begin timing now (undercounts, but the
            // part still gets a duration instead of none).
            let key = ThinkingPartKey(messageID: partial.id, partIndex: index)
            if thinkingStartInstants[key] == nil {
                thinkingStartInstants[key] = .now
            }
        }
        livePart = LivePart(
            messageID: partial.id, partIndex: index, kind: kind,
            initial: textOfPart(at: index, in: partial), throttle: liveThrottle
        )
    }

    /// Rebuild the committed items from an authoritative message log.
    private func resync(from messages: [any AgentMessageProtocol]) {
        // An authoritative snapshot supersedes the Pending Row: either the
        // message is in it (committed) or it doesn't belong here (switch).
        pendingUserMessage = nil
        var rebuilt: [ChatItem] = []
        var results: [String: ToolResultMessage] = [:]
        for message in messages {
            if let user = message.asUser {
                rebuilt.append(.user(user))
            } else if let assistant = message.asAssistant {
                rebuilt.append(.assistant(assistant))
            } else if let toolResult = message.asToolResult {
                results[toolResult.toolCallId] = toolResult
                rebuilt.append(.toolResult(toolResult))
            } else if let compaction = message as? CompactionSummaryMessage {
                rebuilt.append(.system(id: compaction.id, text: compaction.displayText))
            }
        }
        items = rebuilt
        toolResultsByCallID = results
    }

    // MARK: - Send / cancel

    func sendMessage(
        _ text: String, images: [ImageAttachment] = [], bypassCommandParsing: Bool = false
    ) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty || !images.isEmpty else { return }

        if !bypassCommandParsing && images.isEmpty {
            let parseResult = SlashCommandParser.parse(trimmed, registry: commandRegistry())
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

        if agent.state.messages.isEmpty {
            debugLogger.startSession()
            debugLogger.logSystemPrompt(agent.state.systemPrompt, tools: agent.state.tools)
        }

        let user = UserMessage(content: trimmed, images: images)
        // Raise the Pending Row before the send: the run may sit queued behind
        // the lease (cold-start model load) for seconds, and the transcript
        // must show the message immediately. The event-spine commit of the
        // same message lowers it.
        pendingUserMessage = user
        agentRun.send(CoreMessage.user(user))
    }

    func cancelGeneration() {
        agentRun.cancel()
        // Settle synchronously: a cancel while the run is still queued (the
        // Pending Row is up) must hand the message back to the composer *now*,
        // not when the cancelled task's catch runs — the callers that switch
        // conversations right after would otherwise resync the row away and
        // lose the content. The async `onRunSettled` backstop then finds the
        // row already down and does nothing.
        settlePendingUserMessage()
    }

    func cancelGenerationAndWait() async {
        settlePendingUserMessage()
        await agentRun.cancelAndWait()
    }

    /// The run settled (completed, cancelled, or failed) — if the Pending Row
    /// is still up, its message never entered the agent context: lower it and
    /// restore the content to the composer.
    ///
    /// Known, accepted race: events reach the fold one main-actor hop after
    /// emission, so a cancel landing in the instant between the loop
    /// committing the user message and the fold processing it restores a copy
    /// to the composer *and* keeps the committed item. Worst case is a
    /// spurious composer copy — never a lost message, never a transcript that
    /// disagrees with the agent context.
    private func settlePendingUserMessage() {
        guard let pending = pendingUserMessage else { return }
        pendingUserMessage = nil
        restoreComposerDraft(pending.content, pending.images)
    }

    // MARK: - Slash commands

    func executeCommand(_ command: SlashCommand, arguments: String = "") {
        Log.agent.info("Slash command: /\(command.name) \(arguments)")

        switch command.source {
        case .builtIn:
            executeBuiltIn(command.name, arguments: arguments)
        case .skill(let filePath):
            let sent = executeSkill(
                filePath: filePath, skillName: command.name,
                arguments: assembleSkillArguments(command.name, arguments))
            if sent {
                recordSkillInvocation(command.name)
            }
        case .extension:
            error = "Extension commands are not wired yet: /\(command.name)"
        }
    }

    private func executeBuiltIn(_ name: String, arguments: String) {
        switch name {
        case "compact":
            triggerCompaction()
        case "new":
            newConversation()
        case "clear":
            newConversation()
            clearComposerDraft()
        default:
            error = "Unknown built-in command: /\(name)"
        }
    }

    /// `/compact` observes the same arbiter-lease contract as a regular turn by
    /// reusing Agent Run's `runUnderLease`, so the lease/flag/cancel logic is
    /// written once.
    private func triggerCompaction() {
        guard !agent.state.messages.isEmpty else {
            error = "Nothing to compact"
            return
        }
        guard let contextManager, let summarize else {
            error = "Compaction not available"
            return
        }

        let contextWindow = self.contextWindow
        agentRun.runUnderLease { [agent] in
            await agent.forceCompact(
                contextManager: contextManager,
                contextWindow: contextWindow,
                summarize: summarize
            )
        }
    }

    // MARK: - Skill execution

    /// Inject the skill body as the user message (the established `<skill>`
    /// wrapper — a user-message injection, never a system-prompt mutation, so
    /// the prefix cache's stable prefix is untouched). Returns whether the
    /// injection was sent — false when the skill file failed to load, so
    /// callers can skip usage counting and restore their draft.
    @discardableResult
    func executeSkill(
        filePath: String, skillName: String, arguments: String,
        images: [ImageAttachment] = []
    ) -> Bool {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: filePath)),
            let fullText = String(data: data, encoding: .utf8)
        else {
            error = "Failed to load skill: \(filePath)"
            return false
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

        sendMessage(message, images: images, bypassCommandParsing: true)
        return true
    }

    /// The Skill Pill instant action (PRD #174): fire `pill`'s skill now with
    /// the drained composer text and images riding along. Returns whether the
    /// fire was sent — the caller (the pill row, which owns the draft) restores
    /// the draft whole on failure.
    @discardableResult
    func fireSkillPill(_ pill: SkillPill, draftText: String, images: [ImageAttachment]) -> Bool {
        guard !agentRun.isGenerating else { return false }
        let sent = executeSkill(
            filePath: pill.filePath,
            skillName: pill.name,
            arguments: assembleSkillArguments(pill.name, draftText),
            images: images
        )
        if sent {
            recordSkillInvocation(pill.name)
        }
        return sent
    }

    // MARK: - Conversation lifecycle

    func newConversation() {
        switchConversation { conversationStore.createNew() }
        Log.agent.info("New conversation created")
    }

    func loadConversation(_ id: UUID) {
        switchConversation { conversationStore.load(id: id) }
        Log.agent.info("Loaded conversation \(id) with \(self.items.count) items")
    }

    func deleteConversation(_ id: UUID) {
        guard conversationStore.currentConversation?.id == id else {
            conversationStore.delete(id: id)
            return
        }
        // Deleting the current conversation is a switch to whatever the
        // store installs next; the outgoing conversation is the one being
        // deleted, so it is not persisted first.
        switchConversation(persistingOutgoing: false) {
            conversationStore.delete(id: id)
        }
    }

    /// The one conversation-switch sequence. Every path that changes which
    /// conversation is current — new, load, delete-current — settles the
    /// outgoing conversation, runs its store mutation, then adopts whatever
    /// the store reports as current: the destination is data, not a code
    /// path. Divergence between the switch paths was exactly the bug class
    /// this prevents (delete-current used to leave the outgoing
    /// conversation's error banner up).
    private func switchConversation(
        persistingOutgoing: Bool = true, mutateStore: () -> Void
    ) {
        // Stop the run and settle a still-queued Pending Row (restore to
        // the composer, never silently drop) before the resync below would
        // clear it.
        cancelGeneration()
        if persistingOutgoing {
            persistCurrentConversation()
        }
        mutateStore()
        let destination = conversationStore.currentConversation?.messages ?? []
        if destination.isEmpty {
            agent.resetMessages([])
        } else {
            agent.loadMessages(destination)
        }
        resync(from: destination)
        agentRun.finish()  // clear isGenerating synchronously
        clearLiveState()
        runPhase = .idle
        pendingToolCalls.removeAll()
        toolStartInstants.removeAll()
        error = nil
        debugLogger.reset()
        onConversationSwitch()
    }

    private func persistCurrentConversation() {
        guard !agent.state.messages.isEmpty else { return }
        conversationStore.updateCurrentMessages(
            agent.state.messages.map { $0 as any AgentMessageProtocol & Sendable })
        conversationStore.saveCurrent()
    }

    // MARK: - Edit & resend

    /// Truncate the conversation to the turns before `messageID` and return the
    /// edited user message's draft (text + images) for the caller to restore
    /// into the composer. Returns nil (no-op) while generating, or if the id is
    /// missing / not a user message.
    func beginEditingMessage(_ messageID: UUID) -> (text: String, images: [ImageAttachment])? {
        guard !agentRun.isGenerating else { return nil }
        let messages = agent.context.messages
        guard let index = messages.firstIndex(where: { $0.messageUUID == messageID }),
            let user = messages[index].asUser
        else { return nil }

        let head = Array(messages.prefix(index))
        agent.loadMessages(head)
        // `loadMessages` no-ops unless the agent is idle; the `isGenerating`
        // gate above is only a proxy for that. Verify the truncation applied
        // before mutating the store, so a stale gate can never split the
        // on-disk conversation from the live transcript.
        guard agent.context.messages.count == head.count else {
            Log.agent.error(
                "beginEditingMessage: agent not idle, truncation did not apply — "
                    + "aborting to avoid store/transcript divergence")
            return nil
        }
        // Persist the truncation durably. `saveCurrent` SKIPS a head the store
        // won't persist (no user message), so such a head must DELETE the
        // stored conversation rather than write a head that silently drops.
        if AgentConversationStore.persists(head) {
            conversationStore.updateCurrentMessages(head)
            conversationStore.saveCurrent()
        } else if let id = conversationStore.currentConversation?.id {
            conversationStore.delete(id: id)
        }

        resync(from: head)
        // Leave `error` in place: a rejection banner's guidance stays visible
        // while the user trims; `sendMessage` clears it on the next turn.
        Log.agent.info(
            "Editing message \(messageID): truncated to \(head.count) message(s)")
        return (user.content, user.images)
    }

    // MARK: - Voice output

    func speakMessage(_ messageID: UUID) {
        for item in items {
            if case .assistant(let assistant) = item, assistant.id == messageID {
                let text = assistant.text.trimmingCharacters(in: .whitespacesAndNewlines)
                if !text.isEmpty { speechCoordinator?.speakText(text) }
                return
            }
        }
    }

    func stopSpeaking() {
        speechCoordinator?.stop()
    }

    private func autoSpeakIfEnabled() {
        guard let settings, settings.agentAutoSpeak else { return }
        for item in items.reversed() {
            if case .assistant(let assistant) = item {
                let text = assistant.text.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !text.isEmpty else { return }
                Log.agent.info("Auto-speaking response (\(text.count) chars)")
                speechCoordinator?.speakText(text)
                return
            }
        }
    }
}
