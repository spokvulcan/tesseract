//
//  DayThreadTests.swift
//  tesseractTests
//
//  The Day Thread's store over a scratch conversation store: one
//  conversation per day under an id derived from the day, saved through the
//  shared store's index, never where the Agent page opens at launch. And the
//  Today chat shows a moment turn as one quiet line.
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

    @Test func aMomentTurnReadsAsOneLineInTheTodayChat() {
        let session = ChatSession(
            agent: makeNoOpAgent(modelID: "day-thread-test"),
            conversationStore: InMemoryAgentConversationStore(),
            arbiter: InMemoryInferenceArbiter(),
            momentSummary: { request, reply in
                MomentTranscript.summary(request: request, reply: reply)
            },
            liveMarkdownThrottle: .zero)
        let request = UserMessage(content: "[Morning Plan]\nPlan the day.", turnOrigin: .moment)
        let reply = AssistantMessage.create(content: #"{"line": "A calm start."}"#)
        #expect(session.appendCommitted([request, reply]))
        #expect(session.items.count == 1)
        guard case .system(_, let text) = session.items.first else {
            Issue.record("a moment turn must render as one line")
            return
        }
        #expect(text == "Morning Plan · A calm start.")

        #expect(session.appendCommitted([UserMessage(content: "move the gym to 20:00")]))
        #expect(session.items.count == 2)
    }
}
