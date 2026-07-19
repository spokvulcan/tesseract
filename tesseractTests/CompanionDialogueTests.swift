//
//  CompanionDialogueTests.swift
//  tesseractTests
//
//  Report-Back (ADR-0046 #372): the deposit door and the dialogue ledger.
//  The tool tests pin the deposit's landing (an Event in the queue, exactly
//  the wake-table discipline); the ledger tests pin the one-nudge conduct —
//  a dialogue ending or going quiet without a deposit is asked exactly once,
//  on the record, and never nagged.
//

import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

// MARK: - The deposit door

@Suite struct CompanionReportBackToolTests {

    @MainActor
    @Test func aDepositLandsAsAPendingEvent() async throws {
        let store = try scratchStore()
        let recorder = scratchRecorder()
        let dialogueID = UUID()
        var landed: UUID?
        let tool = createReportBackTool(
            store: store, recorder: recorder,
            currentConversationID: { dialogueID },
            depositLanded: { landed = $0 })

        // The tool exists in every owner chat and never in the headless
        // loop (ADR-0052) — the audience IS the rule.
        #expect(tool.audience == .chatOnly)

        let reply = try await toolText(
            tool, ["report": .string("He decided to move the dentist to Thursday.")])
        #expect(reply.contains("Deposited"))
        #expect(landed == dialogueID)

        let pending = try await store.pendingEvents()
        #expect(pending.count == 1)
        #expect(pending[0].kind == .reportBack)
        #expect(pending[0].content == "He decided to move the dentist to Thursday.")
        #expect(pending[0].payload?.contains(dialogueID.uuidString) == true)

        recorder.flushForTesting()
        let records = recorder.records(since: Date().addingTimeInterval(-60))
        #expect(records.contains { $0.event == "report-back.deposited" })
    }

    @MainActor
    @Test func twoMilestonesAreTwoEvents() async throws {
        let store = try scratchStore()
        let tool = createReportBackTool(
            store: store, recorder: scratchRecorder(),
            currentConversationID: { nil }, depositLanded: { _ in })

        _ = try await toolText(tool, ["report": .string("First: the plan is set.")])
        _ = try await toolText(tool, ["report": .string("Second: he promised a walk at 6.")])
        #expect(try await store.pendingEvents().count == 2)
    }

    @MainActor
    @Test func anEmptyReportFailsLoudly() async throws {
        let store = try scratchStore()
        let tool = createReportBackTool(
            store: store, recorder: scratchRecorder(),
            currentConversationID: { nil }, depositLanded: { _ in })

        await #expect(throws: CompanionToolError.self) {
            _ = try await tool.execute("test-call", ["report": .string("   ")], nil, nil)
        }
        await #expect(throws: CompanionToolError.self) {
            _ = try await tool.execute("test-call", [:], nil, nil)
        }
        #expect(try await store.pendingEvents().isEmpty)
    }
}

// MARK: - The dialogue ledger

@MainActor
@Suite struct CompanionDialogueLedgerTests {

    /// A ledger over closure spies: the dialogue conversation is a fixed id,
    /// the agent is never busy, and every nudge send is captured.
    @MainActor
    private final class Harness {
        let recorder = scratchRecorder()
        let dialogueID = UUID()
        var currentID: UUID?
        var sent: [String] = []
        private(set) var ledger: CompanionDialogue!

        init() {
            currentID = dialogueID
            ledger = CompanionDialogue(
                recorder: recorder,
                openDialogue: { [weak self] _ in self?.dialogueID },
                isAgentBusy: { false },
                currentConversationID: { [weak self] in self?.currentID },
                sendNudge: { [weak self] text in self?.sent.append(text) })
        }

        func events(_ name: String) -> Int {
            recorder.flushForTesting()
            return recorder.records(since: Date().addingTimeInterval(-60))
                .filter { $0.event == name }.count
        }
    }

    @Test func endingWithoutADepositGetsExactlyOneNudge() async {
        let harness = Harness()
        harness.ledger.begin(line: "Evening journal, sir.", via: "test")
        harness.ledger.activity(in: harness.dialogueID)

        await harness.ledger.voiceSessionEnded().value
        await harness.ledger.voiceSessionEnded().value

        #expect(harness.sent == [CompanionDialogue.nudgeMessage])
        #expect(harness.events("dialogue.nudged") == 1)
        #expect(harness.events("dialogue.began") == 1)
    }

    @Test func aDepositSettlesTheDebt() async {
        let harness = Harness()
        harness.ledger.begin(line: "Evening journal, sir.", via: "test")
        harness.ledger.activity(in: harness.dialogueID)
        harness.ledger.depositLanded(in: harness.dialogueID)

        await harness.ledger.voiceSessionEnded().value
        #expect(harness.sent.isEmpty)
        #expect(harness.events("dialogue.nudged") == 0)
    }

    @Test func anEmptyEngagementOwesNothing() async {
        let harness = Harness()
        harness.ledger.begin(line: "Evening journal, sir.", via: "test")
        // He engaged and said nothing — no exchange, no debt, no nudge.
        await harness.ledger.voiceSessionEnded().value
        #expect(harness.sent.isEmpty)
    }

    @Test func aSwitchedAwayDialogueRecordsTheMissAndLetsGo() async {
        let harness = Harness()
        harness.ledger.begin(line: "Evening journal, sir.", via: "test")
        harness.ledger.activity(in: harness.dialogueID)
        harness.currentID = UUID()  // he opened another chat

        await harness.ledger.voiceSessionEnded().value
        #expect(harness.sent.isEmpty)
        #expect(harness.events("dialogue.nudge-missed") == 1)
        // The one nudge is burned — no retry, no nag.
        await harness.ledger.voiceSessionEnded().value
        #expect(harness.events("dialogue.nudge-missed") == 1)
    }

    @Test func aReopenedDialogueArmsOnActivity() async {
        let harness = Harness()
        // No begin(): he reopened yesterday's dialogue from history and spoke.
        harness.ledger.activity(in: harness.dialogueID)
        await harness.ledger.voiceSessionEnded().value
        #expect(harness.sent == [CompanionDialogue.nudgeMessage])
    }
}

// MARK: - The dialogue mint (ChatSession.beginDialogue)

@MainActor
@Suite struct CompanionDialogueMintTests {

    @Test func beginDialogueMintsTheSummonedChat() {
        let store = InMemoryAgentConversationStore()
        let session = makeChatSession(store: store)

        let id = session.beginDialogue(line: "It is time for the evening journal, sir.")

        let current = store.currentConversation
        #expect(current?.id == id)
        #expect(current?.origin == .dialogue)
        #expect(session.isDialogueOpen)
        #expect(!session.isMissionControlOpen)
        // The summons line seeds the chat as the entity's own first words —
        // the dialogue agent's context for why it summoned.
        #expect(
            current?.messages.first?.asAssistant?.text
                == "It is time for the evening journal, sir.")
    }

    @Test func aBlankLineSeedsNothing() {
        let store = InMemoryAgentConversationStore()
        let session = makeChatSession(store: store)
        session.beginDialogue(line: "   ")
        #expect(store.currentConversation?.messages.isEmpty == true)
        #expect(store.currentConversation?.origin == .dialogue)
    }
}
