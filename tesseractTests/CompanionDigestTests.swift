//
//  CompanionDigestTests.swift
//  tesseractTests
//
//  Mission Control's fold-down (ADR-0046 #373). The splice suite pins the
//  pure math directly — correctness, tail preservation, idempotence, the
//  re-anchor against a fresh read. The engine suite drives the practice with
//  a scripted model: the night gates, the day stamp, the record, and the
//  failure paths that must leave the conversation untouched.
//

import Foundation
import Testing

@testable import Tesseract_Agent

// MARK: - Fixtures

/// N fold turns of Mission Control: opening (user, origin-tagged) + reply.
@MainActor
private func foldMessages(turns: Int) -> [any AgentMessageProtocol & Sendable] {
    var messages: [any AgentMessageProtocol & Sendable] = []
    for index in 1...turns {
        messages.append(UserMessage(content: "opening \(index)", turnOrigin: .event))
        messages.append(AssistantMessage(content: "reply \(index)"))
    }
    return messages
}

// MARK: - The pure splice

@MainActor
@Suite struct CompanionDigestSpliceTests {

    @Test func planCutsAtTheLastKTurns() {
        let messages = foldMessages(turns: 10)
        let plan = CompanionDigestSplice.plan(messages, keepLastTurns: 6)!
        // Older = the first 4 turns (8 messages); the cut is turn 5's opening.
        #expect(plan.older.count == 8)
        #expect(plan.cutID == messages[8].messageUUID)
        #expect(plan.estimatedTokensBefore > 0)
    }

    @Test func fewerThanKPlusOneTurnsIsNothingToFold() {
        #expect(CompanionDigestSplice.plan(foldMessages(turns: 6), keepLastTurns: 6) == nil)
        #expect(CompanionDigestSplice.plan([], keepLastTurns: 6) == nil)
    }

    @Test func spliceLandsDigestPlusVerbatimTail() {
        let messages = foldMessages(turns: 10)
        let plan = CompanionDigestSplice.plan(messages, keepLastTurns: 6)!
        let spliced = CompanionDigestSplice.splice(
            messages, digest: "The digest.", cutID: plan.cutID, tokensBefore: 1_000)!

        // Head is the digest; the tail is the last 6 turns, verbatim.
        let head = spliced[0] as? CompactionSummaryMessage
        #expect(head?.summary == "The digest.")
        #expect(head?.tokensBefore == 1_000)
        #expect(spliced.count == 1 + 12)
        for (offset, original) in messages[8...].enumerated() {
            #expect(spliced[1 + offset].messageUUID == original.messageUUID)
        }
    }

    @Test func spliceIsIdempotent() {
        let messages = foldMessages(turns: 10)
        let plan = CompanionDigestSplice.plan(messages, keepLastTurns: 6)!
        let spliced = CompanionDigestSplice.splice(
            messages, digest: "The digest.", cutID: plan.cutID, tokensBefore: 1_000)!
        // After a fold, everything is digest + tail — there is nothing left
        // to fold until new turns append past K again.
        #expect(CompanionDigestSplice.plan(spliced, keepLastTurns: 6) == nil)
    }

    @Test func spliceReanchorsOverTurnsThatLandedMidGeneration() {
        let messages = foldMessages(turns: 10)
        let plan = CompanionDigestSplice.plan(messages, keepLastTurns: 6)!
        // Two turns land while the digest is being authored: the fresh read
        // is longer, and they stay verbatim — the cut id anchors the tail.
        let fresh = messages + foldMessages(turns: 2)
        let spliced = CompanionDigestSplice.splice(
            fresh, digest: "The digest.", cutID: plan.cutID, tokensBefore: 1_000)!
        #expect(spliced.count == 1 + 12 + 4)
        #expect(spliced.last?.messageUUID == fresh.last?.messageUUID)
    }

    @Test func aVanishedCutIsARefusal() {
        let messages = foldMessages(turns: 10)
        let spliced = CompanionDigestSplice.splice(
            messages, digest: "d", cutID: UUID(), tokensBefore: 0)
        #expect(spliced == nil)
    }

    @Test func estimatedTokensScaleWithContent() {
        let small = CompanionDigestSplice.estimatedTokens(foldMessages(turns: 2))
        let large = CompanionDigestSplice.estimatedTokens(foldMessages(turns: 40))
        #expect(small > 0)
        #expect(large > small * 10)
    }

    @Test func openingsRenderWithoutTheInstructionsBlock() {
        let opening = """
            <companion-instructions version="3" author="entity">
            # IDENTITY
            Everything here repeats every turn.
            </companion-instructions>

            <situation>He is at his desk.</situation>
            """
        let stripped = CompanionDigestSplice.strippingInstructionsBlock(opening)
        #expect(!stripped.contains("companion-instructions"))
        #expect(stripped.contains("<situation>He is at his desk.</situation>"))

        let rendered = CompanionDigestSplice.renderForDigest([
            UserMessage(content: opening, turnOrigin: .wake),
            AssistantMessage(content: "Noted, sir."),
        ])
        #expect(!rendered.contains("Everything here repeats every turn."))
        #expect(rendered.contains("[wake] OPENING:"))
        #expect(rendered.contains("YOU:\nNoted, sir."))
    }

    @Test func aPreviousDigestRendersFirstAndWhole() {
        let previous = CompactionSummaryMessage(
            summary: "Previously: the dentist saga.", tokensBefore: 90_000)
        let rendered = CompanionDigestSplice.renderForDigest(
            [previous as any AgentMessageProtocol & Sendable] + foldMessages(turns: 2))
        #expect(rendered.hasPrefix("YOUR PREVIOUS DIGEST:\nPreviously: the dentist saga."))
    }
}

// MARK: - The practice

@MainActor
@Suite struct CompanionDigestEngineTests {

    private final class ScriptedModel: @unchecked Sendable {
        var reply = "Digest: contracts in flight, promises, his patterns."
        var failure: Error?
        private(set) var prompts: [String] = []
        var complete: @Sendable (String) async throws -> String {
            { [self] prompt in
                try await MainActor.run {
                    self.prompts.append(prompt)
                    if let failure { throw failure }
                    return reply
                }
            }
        }
    }

    private struct ScriptedFailure: Error {}

    private func makeEngine(
        _ conversations: AgentConversationStore, _ store: MemoryStore,
        _ recorder: CompanionFlightRecorder, _ model: ScriptedModel, enabled: Bool = true
    ) -> CompanionDigest {
        CompanionDigest(
            conversationStore: conversations, store: store, recorder: recorder,
            arbiter: InMemoryInferenceArbiter(), complete: model.complete,
            isEnabled: { enabled })
    }

    /// A Mission Control of `turns` fold turns, saved to a scratch store.
    private func seededConversations(turns: Int) -> AgentConversationStore {
        let store = AgentConversationStore(directory: makeTempDir("scratch-conversations"))
        var missionControl = store.missionControl()
        missionControl.messages = foldMessages(turns: turns)
        store.save(missionControl)
        return store
    }

    /// 23:00 tonight — inside the night window.
    private var night: Date {
        Calendar.current.date(bySettingHour: 23, minute: 0, second: 0, of: Date())!
    }
    /// 14:00 — outside it.
    private var afternoon: Date {
        Calendar.current.date(bySettingHour: 14, minute: 0, second: 0, of: Date())!
    }

    @Test func nightlyFoldLandsDigestPlusTailAndStampsTheNight() async throws {
        let conversations = seededConversations(turns: 10)
        let store = try scratchStore()
        let recorder = scratchRecorder()
        let model = ScriptedModel()
        let engine = makeEngine(conversations, store, recorder, model)

        await engine.nightlyFold(now: night)

        let folded = conversations.missionControl()
        #expect((folded.messages.first as? CompactionSummaryMessage)?.summary == model.reply)
        #expect(folded.messages.count == 1 + 12)
        #expect(model.prompts.count == 1)
        #expect(model.prompts[0].contains("opening 1"))
        // The prompt covers only the older history — the tail is not re-told.
        #expect(!model.prompts[0].contains("opening 10"))

        recorder.flushForTesting()
        let events = recorder.records(since: Date().addingTimeInterval(-60))
        #expect(
            events.contains {
                $0.event == "digest.folded" && $0.snapshot?["reason"] == "nightly"
            })

        // Stamped under the night's key: the same night never folds twice.
        let day = try await store.loopDayState(CompanionDigest.nightKey(for: night))
        #expect(day.digestFoldAt != nil)
        await engine.nightlyFold(now: night)
        #expect(model.prompts.count == 1)
    }

    @Test func theAfternoonNeverFoldsNightly() async throws {
        let conversations = seededConversations(turns: 10)
        let model = ScriptedModel()
        let engine = makeEngine(conversations, try scratchStore(), scratchRecorder(), model)

        await engine.nightlyFold(now: afternoon)
        #expect(model.prompts.isEmpty)
        #expect(conversations.missionControl().messages.count == 20)
    }

    @Test func theCeilingFoldRunsAtAnyHourOnTheRecord() async throws {
        let conversations = seededConversations(turns: 10)
        let recorder = scratchRecorder()
        let model = ScriptedModel()
        let engine = makeEngine(conversations, try scratchStore(), recorder, model)

        await engine.earlyFold(now: afternoon)

        #expect(conversations.missionControl().messages.count == 1 + 12)
        recorder.flushForTesting()
        let events = recorder.records(since: Date().addingTimeInterval(-60))
        #expect(
            events.contains {
                $0.event == "digest.folded" && $0.snapshot?["reason"] == "ceiling"
            })
    }

    @Test func aFailedGenerationLeavesEverythingUntouchedAndUnstamped() async throws {
        let conversations = seededConversations(turns: 10)
        let store = try scratchStore()
        let model = ScriptedModel()
        model.failure = ScriptedFailure()
        let engine = makeEngine(conversations, store, scratchRecorder(), model)

        await engine.nightlyFold(now: night)

        #expect(conversations.missionControl().messages.count == 20)
        // Unstamped — the next idle pass retries.
        let day = try await store.loopDayState(CompanionDigest.nightKey(for: night))
        #expect(day.digestFoldAt == nil)
    }

    @Test func anOversizeDigestIsRejectedAsNoise() async throws {
        let conversations = seededConversations(turns: 10)
        let model = ScriptedModel()
        model.reply = String(repeating: "x", count: CompanionDigest.digestMaxLength + 1)
        let engine = makeEngine(conversations, try scratchStore(), scratchRecorder(), model)

        await engine.earlyFold(now: night)
        #expect(conversations.missionControl().messages.count == 20)
    }

    @Test func aSmallConversationStampsTheNightWithoutAGeneration() async throws {
        let conversations = seededConversations(turns: 3)
        let store = try scratchStore()
        let model = ScriptedModel()
        let engine = makeEngine(conversations, store, scratchRecorder(), model)

        await engine.nightlyFold(now: night)

        #expect(model.prompts.isEmpty)
        #expect(conversations.missionControl().messages.count == 6)
        // Nothing to fold IS tonight's outcome — no re-planning every pass.
        let day = try await store.loopDayState(CompanionDigest.nightKey(for: night))
        #expect(day.digestFoldAt != nil)
    }
}
