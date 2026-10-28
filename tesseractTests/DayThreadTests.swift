//
//  DayThreadTests.swift
//  tesseractTests
//
//  The Day Thread's store over a scratch conversation store: one
//  conversation per day under an id derived from the day, saved through the
//  shared store's index, never where the Agent page opens at launch. And the
//  Today chat shows a moment turn, request and reply, as one row, and the
//  thread compacts in the selected model's window.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct DayThreadTests {

    private func scratchStore() -> AgentConversationStore {
        AgentConversationStore(directory: makeTempDir("day-thread"))
    }

    @Test func eachDayHasOneThreadUnderItsOwnID() {
        let day = DayKey(rawValue: "2026-09-30")
        #expect(AgentConversation.dayThreadID(for: day) == AgentConversation.dayThreadID(for: day))
        #expect(
            AgentConversation.dayThreadID(for: day)
                != AgentConversation.dayThreadID(for: DayKey(rawValue: "2026-10-01")))

        let backing = scratchStore()
        let store = DayThreadStore(backing: backing, day: day)
        store.loadMostRecent()
        #expect(store.currentConversation?.id == AgentConversation.dayThreadID(for: day))
        #expect(store.currentConversation?.isDayThread == true)
        #expect(store.currentConversation?.messages.isEmpty == true)
    }

    @Test func aSavedThreadComesBackAndNeverOpensTheAgentPage() {
        let backing = scratchStore()
        let day = DayKey(rawValue: "2026-09-30")
        let store = DayThreadStore(backing: backing, day: day)
        store.loadMostRecent()
        store.updateCurrentMessages([UserMessage(content: "[Day Opening]", turnOrigin: .moment)])
        store.saveCurrent()

        let reopened = DayThreadStore(backing: backing, day: day)
        reopened.loadMostRecent()
        #expect(reopened.currentConversation?.messages.count == 1)
        #expect(backing.conversations.first?.turnOrigin == .dayThread)

        backing.loadMostRecent()
        #expect(backing.currentConversation?.isDayThread != true)
    }

    @Test func aMomentTurnIsOneRowInTheTodayChat() {
        let session = ChatSession(
            agent: makeNoOpAgent(modelID: "day-thread-test"),
            conversationStore: InMemoryAgentConversationStore(),
            arbiter: InMemoryInferenceArbiter(),
            foldsMoments: true,
            liveMarkdownThrottle: .zero)
        let request = UserMessage(content: "[Morning Plan]\nPlan the day.", turnOrigin: .moment)
        let reply = AssistantMessage.create(content: #"{"line": "A calm start."}"#)
        #expect(session.appendCommitted([request, reply]))
        #expect(session.items.count == 1)
        guard case .moment(let shown, let answer) = session.items.first else {
            Issue.record("a moment turn must fold into one row")
            return
        }
        let moment = ThreadMoment(request: shown, reply: answer)
        #expect(moment.kind == .moment(.morningPlan))
        #expect(moment.line == "A calm start.")

        #expect(session.appendCommitted([UserMessage(content: "move the gym to 20:00")]))
        #expect(session.items.count == 2)
    }

    // MARK: - Compaction in the selected model's window

    /// The thread compacts past its ceiling in the selected model's window,
    /// read when it checks: with a 32K model picked, the 64K ceiling gives
    /// way to the window.
    @Test func theCeilingGivesWayToTheSelectedModelsWindow() {
        let settings = SettingsManager(store: InMemorySettingsStore())
        let qwen = DayThread.compactionWindow(settings: settings)
        #expect(qwen == CompactionWindow(contextLength: 262_144, ceiling: 64_000))
        #expect(CompactionSettings.standard.compactionPoint(in: qwen) == 64_000)

        settings.selectedAgentModelID = "lfm3-audio-2b"
        let audio = DayThread.compactionWindow(settings: settings)
        #expect(audio == CompactionWindow(contextLength: 32_768, ceiling: 64_000))
        #expect(CompactionSettings.standard.compactionPoint(in: audio) == 23_247)
    }

    /// A moment over a 24,000-token thread, under its 64K ceiling, compacts
    /// the thread first with a 32K model selected, before the window fills,
    /// and leaves it whole with Qwen's.
    @Test func aMomentCompactsTheThreadBeforeASmallWindowFills() async {
        let qwen = CompactingThread(modelID: ModelDefinition.defaultAgentModelID)
        _ = await qwen.thread.runMoment(CompactingThread.moment)
        #expect(qwen.summaries.value == 0)
        #expect(!(qwen.agent.state.messages.first is CompactionSummaryMessage))

        let audio = CompactingThread(modelID: "lfm3-audio-2b")
        _ = await audio.thread.runMoment(CompactingThread.moment)
        #expect(audio.summaries.value == 1)
        #expect(audio.agent.state.messages.first is CompactionSummaryMessage)
    }
}

/// A Day Thread over twelve 2,000-token turns (24,000 tokens, under the 64K
/// ceiling) with `modelID` selected, counting the summaries it asks for.
@MainActor
private struct CompactingThread {
    static let moment = MomentRequest(
        kind: .breakpoint, trigger: .presenceReturned, text: "[Breakpoint]\nWhile you were away")

    let thread: DayThread
    let agent: Agent
    let summaries = Locked(0)

    init(modelID: String) {
        let summaries = summaries
        let settings = SettingsManager(store: InMemorySettingsStore())
        settings.selectedAgentModelID = modelID
        let agent = makeNoOpAgent(modelID: modelID)
        self.agent = agent
        thread = DayThread(
            agent: agent,
            store: DayThreadStore(
                backing: AgentConversationStore(directory: makeTempDir("day-thread-compaction")),
                day: DayKey(rawValue: "2026-10-28")),
            arbiter: InMemoryInferenceArbiter(),
            inferenceService: ServerInferenceService(
                completionStarter: RecordingCompletionStarter(), engine: UnusedManagedInference(),
                modelStateProvider: { nil }),
            toolRegistry: ToolRegistry(
                sandbox: PathSandbox(root: makeTempDir("day-thread-compaction-sandbox")),
                extensionHost: ExtensionHost()),
            settings: settings,
            speechCoordinator: nil,
            contextManager: ContextManager(settings: .standard),
            summarize: { _ in
                summaries.value += 1
                return "## Goal\nThe day so far."
            },
            trace: scratchTrace())
        // Through the thread's chat, which holds the agent's messages from here.
        thread.chat.appendCommitted(
            (0..<12).map { index in
                let text = String(repeating: "\(index % 10)", count: 8_000)
                let turn: any AgentMessageProtocol & Sendable =
                    index.isMultiple(of: 2)
                    ? UserMessage(content: text) : AssistantMessage(content: text)
                return turn
            })
    }
}
