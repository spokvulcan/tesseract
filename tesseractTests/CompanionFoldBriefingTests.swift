//
//  CompanionFoldBriefingTests.swift
//  tesseractTests
//
//  The Fold Briefing (ADR-0052) at its seam: every owner conversation opens
//  as the one mind, re-briefs only when the fold advanced, and a reopened
//  transcript carries its own stamp across relaunch. Plus the briefing-id
//  resolution `revise_wake`/`cancel_wake` lean on.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
@Suite struct CompanionFoldBriefingTests {

    // MARK: - Fixtures

    /// A fold conversation with one turn: a conclusion, a "Silence." turn
    /// (never briefed), and a notify delivery.
    private func foldFixture(updatedAt: Date = Date()) -> AgentConversation {
        AgentConversation(
            id: AgentConversation.missionControlID,
            messages: [
                UserMessage(content: "<turn>fold</turn>", turnOrigin: .event),
                AssistantMessage(content: "Silence."),
                AssistantMessage(
                    content: "He hasn't engaged — folding the journal into tomorrow.",
                    toolCalls: [
                        ToolCallInfo(
                            id: "call-1", name: "notify",
                            argumentsJSON:
                                #"{"title":"Jarvis","body":"Evening journal is open, sir."}"#
                        )
                    ]),
            ],
            updatedAt: updatedAt,
            origin: .missionControl)
    }

    private func briefing(
        store: MemoryStore, fold: @escaping () -> AgentConversation?,
        enabled: Bool = true
    ) -> CompanionFoldBriefing {
        CompanionFoldBriefing(
            store: store, missionControl: fold, isEnabled: { enabled })
    }

    // MARK: - Injection

    @Test func briefsTheFirstMessageWithTheFoldsRecentLife() async throws {
        let store = try scratchStore()
        let booked = CompanionWake(
            content: "Morning planning", due: Date().addingTimeInterval(3600),
            wakeClass: .rhythm)
        try await store.upsertWake(booked)
        var fired = CompanionWake(
            content: "Evening journal", due: Date().addingTimeInterval(-1800),
            wakeClass: .rhythm, state: .delivered)
        fired.firedAt = Date().addingTimeInterval(-1800)
        try await store.upsertWake(fired)

        let sut = briefing(store: store, fold: { self.foldFixture() })
        let decorated = await sut.decorate(UserMessage(content: "hey"), transcript: [])

        let injected = decorated.injectedContext ?? ""
        #expect(injected.contains("<fold-briefing at=\""))
        #expect(injected.contains("Morning planning"))
        #expect(injected.contains("[id \(booked.shortID)]"))
        #expect(injected.contains("Evening journal"))
        #expect(injected.contains(", unheard"))
        #expect(injected.contains("notify: \"Evening journal is open, sir.\""))
        #expect(injected.contains("folding the journal into tomorrow"))
        #expect(!injected.contains("- Silence."))
        #expect(injected.contains("report_back"))
    }

    @Test func rebriefsOnlyWhenTheFoldAdvances() async throws {
        let store = try scratchStore()
        var fold = foldFixture(updatedAt: Date().addingTimeInterval(-60))
        let sut = briefing(store: store, fold: { fold })

        let first = await sut.decorate(UserMessage(content: "one"), transcript: [])
        #expect(first.injectedContext?.contains("<fold-briefing") == true)

        // The fold has not moved: the next message rides clean.
        let second = await sut.decorate(UserMessage(content: "two"), transcript: [first])
        #expect(second.injectedContext == nil)

        // A fold turn landed meanwhile: the chat re-briefs.
        fold.updatedAt = Date()
        let third = await sut.decorate(UserMessage(content: "three"), transcript: [first])
        #expect(third.injectedContext?.contains("<fold-briefing") == true)
    }

    /// Relaunch loses the in-memory stamp; the transcript's own block is the
    /// durable half, so a reopened conversation is not re-briefed for a fold
    /// that never moved.
    @Test func reopenedTranscriptCarriesItsOwnStamp() async throws {
        let store = try scratchStore()
        let fold = foldFixture(updatedAt: Date().addingTimeInterval(-60))

        let before = briefing(store: store, fold: { fold })
        let briefed = await before.decorate(UserMessage(content: "one"), transcript: [])

        let relaunched = briefing(store: store, fold: { fold })
        let after = await relaunched.decorate(
            UserMessage(content: "two"), transcript: [briefed])
        #expect(after.injectedContext == nil)
    }

    @Test func disabledOrEmptyFoldInjectsNothing() async throws {
        let store = try scratchStore()

        let disabled = briefing(store: store, fold: { self.foldFixture() }, enabled: false)
        let off = await disabled.decorate(UserMessage(content: "hey"), transcript: [])
        #expect(off.injectedContext == nil)

        let emptyFold = briefing(
            store: store,
            fold: { AgentConversation(origin: .missionControl) })
        let empty = await emptyFold.decorate(UserMessage(content: "hey"), transcript: [])
        #expect(empty.injectedContext == nil)
    }

    // MARK: - Fold-transcript extraction

    @Test func extractionCapsAndFiltersMechanically() {
        let old = Date().addingTimeInterval(-48 * 3600)
        var messages: [any AgentMessageProtocol & Sendable] = [
            AssistantMessage(
                content: "",
                toolCalls: [
                    ToolCallInfo(
                        id: "old", name: "notify", argumentsJSON: #"{"body":"stale"}"#)
                ],
                timestamp: old)
        ]
        for index in 1...5 {
            messages.append(AssistantMessage(content: "Conclusion \(index)"))
        }
        for index in 1...7 {
            messages.append(
                AssistantMessage(
                    content: "Silence.",
                    toolCalls: [
                        ToolCallInfo(
                            id: "call-\(index)", name: "speak",
                            argumentsJSON: #"{"text":"line \#(index)"}"#)
                    ]))
        }

        let activity = CompanionFoldBriefing.extractActivity(
            from: messages, deliveriesSince: Date().addingTimeInterval(-24 * 3600))
        // The stale delivery is outside the window; the last five spoken
        // lines survive the cap.
        #expect(activity.deliveries.count == 5)
        #expect(!activity.deliveries.contains { $0.contains("stale") })
        #expect(activity.deliveries.last?.contains("speak: \"line 7\"") == true)
        // Conclusions keep the last three, "Silence." never among them.
        #expect(activity.conclusions.count == 3)
        #expect(activity.conclusions.first?.contains("Conclusion 3") == true)
        #expect(activity.conclusions.last?.contains("Conclusion 5") == true)
    }
}

// MARK: - Briefing-id resolution (ADR-0052 — the 22:41 cancel_wake failure)

@Suite struct WakeShortIDResolutionTests {

    @Test func resolvesFullUUIDAndShortPrefix() async throws {
        let store = try scratchStore()
        let wake = CompanionWake(content: "Midday pulse", due: Date().addingTimeInterval(600))
        try await store.upsertWake(wake)

        let byUUID = try await store.openWake(matching: wake.id.uuidString)
        #expect(byUUID?.id == wake.id)
        let byShort = try await store.openWake(matching: wake.shortID)
        #expect(byShort?.id == wake.id)
        // Case-insensitive: the briefing renders lowercase, UUIDs are upper.
        let byUpper = try await store.openWake(matching: wake.shortID.uppercased())
        #expect(byUpper?.id == wake.id)
    }

    @Test func ambiguousOrUnknownPrefixesResolveToNil() async throws {
        let store = try scratchStore()
        let twinA = CompanionWake(
            id: UUID(uuidString: "AAAAAA01-0000-4000-8000-000000000001")!,
            content: "A", due: Date().addingTimeInterval(600))
        let twinB = CompanionWake(
            id: UUID(uuidString: "AAAAAA02-0000-4000-8000-000000000002")!,
            content: "B", due: Date().addingTimeInterval(600))
        try await store.upsertWake(twinA)
        try await store.upsertWake(twinB)

        #expect(try await store.openWake(matching: "aaaaaa") == nil)
        #expect(try await store.openWake(matching: "bbbbbb") == nil)
        // Too short to be an id at all.
        #expect(try await store.openWake(matching: "aa") == nil)
    }

    /// Short ids name what a tool can act on — open (booked or fired) wakes;
    /// a terminal wake stops matching by prefix but stays reachable by its
    /// full UUID (the exact-id path resolves any state, callers guard).
    @Test func terminalWakesLeaveTheShortNamespace() async throws {
        let store = try scratchStore()
        var done = CompanionWake(
            content: "Delivered already", due: Date().addingTimeInterval(-600),
            state: .delivered)
        done.firedAt = Date().addingTimeInterval(-600)
        try await store.upsertWake(done)

        #expect(try await store.openWake(matching: done.shortID) == nil)
        #expect(try await store.openWake(matching: done.id.uuidString)?.id == done.id)
    }
}
