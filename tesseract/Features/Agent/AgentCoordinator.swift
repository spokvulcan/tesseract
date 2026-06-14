import Foundation
import Observation
import UniformTypeIdentifiers

/// Thin **dispatcher spine** over ``Agent`` and five publisher-agnostic
/// sub-modules. The coordinator owns the single agent-event subscription and
/// *sequences* typed intent calls to the sub-modules — each owning its own state
/// and dependencies, each testable at its own interface:
///
/// - ``AgentRunController`` (`agentRun`) — the foreground-run envelope: the
///   `isGenerating` busy flag, the `InferenceArbiter` lease, send, cancellation.
/// - ``ChatTranscriptController`` (`transcript`) — the stateful driver of the
///   pure ``ChatTranscript`` fold: rows, expansion, the streaming throttle/splice.
/// - ``AgentVoiceInputController`` (`voiceInput`) — push-to-talk capture→emit.
/// - ``AgentSystemPromptInspector`` (`systemPromptInspector`) — the cancellable
///   raw-prompt / token-count transparency panel.
/// - ``SlashCommandPaletteController`` (`commandPalette`) — slash-command popup.
///
/// The coordinator retains only what is genuinely cross-cutting: event
/// sequencing, conversation-lifecycle orchestration, the shared `error` banner,
/// command-execution routing, and the thin voice-output passthroughs. Command
/// execution and voice output are deliberate non-carves (see `CONTEXT.md` —
/// *Agent coordinator leaves*).
@Observable @MainActor
final class AgentCoordinator {

    // MARK: - Sub-modules

    let agentRun: AgentRunController
    let transcript: ChatTranscriptController
    let voiceInput: AgentVoiceInputController
    let systemPromptInspector: AgentSystemPromptInspector
    let commandPalette: SlashCommandPaletteController

    // MARK: - Shared view-facing state

    /// The shared error banner. Sub-modules that fail report here via an injected
    /// closure; voice input is the exception — its errors stay in `voiceState`.
    var error: String?

    /// The pending Quick Look open request (slice #114). Set by `openQuickLook`,
    /// presented by the `QuickLookContainer` host, cleared when the panel closes.
    var quickLookRequest: QuickLookRequest?

    // MARK: - Hot-read passthroughs

    /// Re-exposed so existing call sites and Observation tracking are unchanged —
    /// the reads resolve through the nested `@Observable` sub-modules.
    var rows: [ChatRow] { transcript.rows }
    var isGenerating: Bool { agentRun.isGenerating }
    var streamingRowVersion: Int { transcript.streamingRowVersion }

    // MARK: - Spine dependencies

    private let agent: Agent
    private let conversationStore: any AgentConversationStoring
    private let settings: SettingsManager?
    private let arbiter: any InferenceArbitrating
    private let speechCoordinator: SpeechCoordinator?
    private let extensionHost: ExtensionHost?
    private let contextManager: ContextManager?
    private let contextWindow: Int
    private let summarize: (@Sendable (String) async throws -> String)?
    private let debugLogger = AgentDebugLogger()
    /// Digest-keyed temp-file cache backing the Quick Look viewer (slice #114).
    @ObservationIgnored private let imagePreviewCache = ImagePreviewFileCache()

    /// Unsubscribe closure for agent event subscription.
    @ObservationIgnored private var unsubscribe: (@MainActor () -> Void)?

    // MARK: - Init

    init(
        agent: Agent,
        conversationStore: any AgentConversationStoring,
        audioCapture: (any AudioCapturing)? = nil,
        transcriptionEngine: (any Transcribing)? = nil,
        settings: SettingsManager? = nil,
        arbiter: any InferenceArbitrating,
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
        self.settings = settings
        self.arbiter = arbiter
        self.speechCoordinator = speechCoordinator
        self.extensionHost = extensionHost
        self.contextManager = contextManager
        self.contextWindow = contextWindow
        self.summarize = summarize

        // Construct the publisher-agnostic sub-modules from the injected deps.
        self.agentRun = AgentRunController(
            agent: agent, arbiter: arbiter, toolRegistry: toolRegistry, settings: settings
        )
        self.transcript = ChatTranscriptController()
        self.voiceInput = AgentVoiceInputController(
            audioCapture: audioCapture, transcriptionEngine: transcriptionEngine, settings: settings
        )
        self.systemPromptInspector = AgentSystemPromptInspector(
            promptSource: { [agent] in (agent.state.systemPrompt, agent.state.tools) },
            formatRawPrompt: formatRawPrompt
        )
        self.commandPalette = SlashCommandPaletteController(
            extensionHost: extensionHost, packageRegistry: packageRegistry
        )

        // Now that `self` is fully initialized, route sub-module failures to the
        // shared error banner.
        agentRun.setReportError { [weak self] message in self?.error = message }

        subscribeToAgentEvents()

        conversationStore.loadMostRecent()
        if let current = conversationStore.currentConversation {
            agent.loadMessages(current.messages)
            rebuildTranscript()
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
            let parseResult = SlashCommandParser.parse(trimmed, registry: commandPalette.commandRegistry)
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

        let userMessage = CoreMessage.user(UserMessage(content: trimmed, images: images))
        agentRun.send(userMessage)
    }

    func cancelGeneration() {
        agentRun.cancel()
    }

    func cancelGenerationAndWait() async {
        await agentRun.cancelAndWait()
    }

    // MARK: - Slash Command Execution
    //
    // Command *execution* is a deliberate non-carve: it routes `/compact` into
    // Agent Run's lease, `/new`·`/clear` into the conversation orchestration, and
    // skills back into `agentRun.send` — a standalone module would only forward.

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

    /// `/compact` observes the same arbiter-lease contract as a regular turn by
    /// reusing Agent Run's `runUnderLease`, so the lease/flag/cancel logic is
    /// written once. Without the lease, compaction could race a foreground turn
    /// and reach the LLM with a stale model state.
    private func triggerCompaction(instructions: String?) {
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
    //
    // The conversation lifecycle stays on the spine: new/load/delete orchestrate
    // resets across Agent Run (cancel/finish), the Chat Transcript Controller
    // (reset/rebuild), the inspector, and the store.

    func newConversation() {
        cancelGeneration()
        persistCurrentConversation()
        conversationStore.createNew()
        agent.resetMessages([])
        resetState()
        error = nil
        systemPromptInspector.reset()
        debugLogger.reset()
        Log.agent.info("New conversation created")
    }

    func loadConversation(_ id: UUID) {
        cancelGeneration()
        persistCurrentConversation()
        conversationStore.load(id: id)
        if let current = conversationStore.currentConversation {
            agent.loadMessages(current.messages)
        } else {
            agent.resetMessages([])
        }
        resetState()
        rebuildTranscript()
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
            rebuildTranscript()
            debugLogger.reset()
        }
    }

    func clearConversation() {
        newConversation()
    }

    // MARK: - Quick Look Image Viewer (slice #114)

    /// Open the full-size Quick Look viewer on the clicked image, navigable
    /// across every image in the conversation in order. Builds the preview-set
    /// projection, materializes digest-keyed temp files (reused when pre-warmed),
    /// and hands the host an ordered URL set + start index. A no-op if the clicked
    /// image is no longer in the conversation or nothing materialized.
    func openQuickLook(clicked id: UUID, includingPending pending: [ImageAttachment] = []) {
        let all = conversationImages() + pending
        guard let set = ImagePreviewSet.project(all: all, clicked: id) else { return }
        let urls = imagePreviewCache.materialize(set.attachments)
        // Re-derive the start index from the clicked image's URL so a (rare)
        // dropped write can't misalign the opening index.
        let clicked = set.attachments[set.startIndex]
        guard let clickedURL = try? imagePreviewCache.url(for: clicked),
              let startIndex = urls.firstIndex(of: clickedURL)
        else { return }
        quickLookRequest = QuickLookRequest(urls: urls, startIndex: startIndex)
    }

    /// Pre-warm one image's temp file (called as the transcript decodes it) so a
    /// later click opens near-instantly.
    func prewarmImagePreview(_ attachment: ImageAttachment) {
        imagePreviewCache.prewarm(attachment)
    }

    /// Clear the pending request when the panel closes so the same image can be
    /// reopened later.
    func dismissQuickLook() {
        quickLookRequest = nil
    }

    // MARK: - Composer Image Draft (lifted for full-window drop, slice #117)

    /// Images queued in the composer awaiting send. Lifted from the input bar so
    /// a full-window drop — hosted on the content view, above the composer — lands
    /// in the same queue. The composer renders/removes them and `send` consumes
    /// them.
    var pendingImages: [ImageAttachment] = []

    /// Whether the selected model can serve images, synced from the composer
    /// (which owns the capability probe). The window-level drop reads this to
    /// decide between attaching and surfacing the switch hint.
    var imageInputAvailable = false

    /// Drives the composer's "switch to a vision model" hint — set when an image is
    /// pasted or dropped onto a model that can't see images.
    var showImageSwitchHint = false

    /// Max images queued in the composer at once (PRD #112 cap).
    static let maxPendingImages = 8

    /// Append ingested images, capped to the room remaining under `maxPendingImages`.
    func attachImages(_ attachments: [ImageAttachment]) {
        let added = ImageIngest.capBatch(
            attachments, alreadyQueued: pendingImages.count, limit: Self.maxPendingImages
        )
        pendingImages.append(contentsOf: added)
    }

    /// Handle image item providers dropped anywhere on the window (slice #117).
    /// When the selected model can't see images, surface the switch hint instead
    /// of attaching. Each provider is ingested via the shared `ImageIngest` core
    /// and appended cap-respecting. Returns whether the drop was accepted.
    @discardableResult
    func handleWindowImageDrop(_ providers: [NSItemProvider]) -> Bool {
        guard imageInputAvailable else {
            showImageSwitchHint = true
            return false
        }
        for provider in providers {
            let preferredUTI = provider.registeredTypeIdentifiers.first {
                UTType($0)?.conforms(to: .image) == true
            } ?? UTType.image.identifier
            provider.loadDataRepresentation(forTypeIdentifier: UTType.image.identifier) { [weak self] data, _ in
                guard let self, let data else { return }
                guard case .success(let attachment) = ImageIngest.ingest(
                    data: data, typeIdentifier: preferredUTI, filename: "dropped-image"
                ) else { return }
                Task { @MainActor in self.attachImages([attachment]) }
            }
        }
        return true
    }

    /// Every committed image in the conversation, in message order — user
    /// attachments and tool-result images (slice #116) — the navigable set the
    /// preview-set projection pages through. Pending composer images are merged
    /// in by `openQuickLook(clicked:includingPending:)` since they aren't yet in
    /// `agent.state.messages`.
    private func conversationImages() -> [ImageAttachment] {
        agent.state.messages.flatMap { message -> [ImageAttachment] in
            if let user = message.asUser { return user.images }
            if let tool = message.asToolResult { return tool.content.imageAttachments }
            return []
        }
    }

    // MARK: - Voice Output
    //
    // A deliberate non-carve: thin, stateless calls already sitting over the
    // seamed `SpeechCoordinator`.

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
    //
    // Thin feeders: the view has no `Agent`, so the coordinator supplies the
    // current message log to the Chat Transcript Controller.

    func toggleTurnExpanded(_ turnID: UUID) {
        transcript.toggleTurnExpanded(
            turnID, messages: agent.state.messages,
            stream: agent.state.streamMessage, isGenerating: agentRun.isGenerating
        )
    }

    func toggleDetailExpanded(_ rowID: String) {
        transcript.toggleDetailExpanded(rowID)
    }

    // MARK: - Event Subscription (the sequencing dispatcher)

    private func subscribeToAgentEvents() {
        unsubscribe = agent.subscribe { [weak self] event in
            Task { @MainActor in
                self?.handleAgentEvent(event)
            }
        }
    }

    /// Translates each `AgentEvent` into ordered intent calls on the sub-modules.
    ///
    /// The `isGenerating`-before-rebuild ordering invariant lives here: on
    /// `.agentStart`/`.agentEnd` we flip Agent Run's busy state *before* asking
    /// the Chat Transcript Controller to rebuild, because the transcript
    /// projection reads `isGenerating` (the streaming-header auto-expand guards on
    /// it). Reversing the order would leak the streaming indicator into the
    /// committed transcript.
    private func handleAgentEvent(_ event: AgentEvent) {
        switch event {
        case .agentStart:
            agentRun.markStarted()
            rebuildTranscript()

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
                rebuildTranscript()
            }

        case .messageUpdate:
            transcript.patchStreamingTail(
                messages: agent.state.messages,
                stream: agent.state.streamMessage,
                isGenerating: agentRun.isGenerating
            )

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
            transcript.onTurnEnd(
                messages: agent.state.messages,
                stream: agent.state.streamMessage,
                isGenerating: agentRun.isGenerating
            )
            persistCurrentConversation()

        case .agentEnd(_):
            // Terminal transition: flip the busy flag so the view passthroughs
            // settle. `onAgentEnded` rebuilds as not-generating on its own, so the
            // committed transcript no longer depends on `finish()` landing first.
            agentRun.finish()
            transcript.onAgentEnded(
                messages: agent.state.messages,
                stream: agent.state.streamMessage
            )
            autoSpeakIfEnabled()

        case .messageEnd:
            rebuildTranscript()

        case .turnStart, .messageStart, .toolExecutionUpdate, .malformedToolCall:
            break
        }
    }

    // MARK: - Private Helpers

    /// Feed the Chat Transcript Controller a full rebuild from current state.
    private func rebuildTranscript() {
        transcript.rebuild(
            messages: agent.state.messages,
            stream: agent.state.streamMessage,
            isGenerating: agentRun.isGenerating
        )
    }

    private func resetState() {
        transcript.reset()
        agentRun.finish()   // clear isGenerating synchronously
        quickLookRequest = nil
        imagePreviewCache.clear()   // drop the previous conversation's temp previews
        pendingImages = []
        showImageSwitchHint = false
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
