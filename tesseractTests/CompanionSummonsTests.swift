//
//  CompanionSummonsTests.swift
//  tesseractTests
//
//  The overlay summons is choreography over the loop's one **Reaction**
//  door (#391): an engage or dismiss is *reported*, never hand-written —
//  the fold reducer decides the wake stamps and the dialogue mint. These
//  are the first tests to construct `CompanionSummons`; they pin the
//  report's correlation (the wakeID the pre-#391 path dropped), the
//  report-before-voice ordering, and the unanswered fallback.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct CompanionSummonsTests {

    /// One summons under test: scripted overlay outcome, recording doors.
    private final class Harness {
        var reported: [CompanionPingReaction] = []
        var voiceEntries: [String] = []
        var fallbacks: [(line: String, wakeID: UUID?, conversationID: UUID?)] = []
        /// Interleaving proof: every door appends its name in call order.
        var sequence: [String] = []
        let summons: CompanionSummons
        let context = CompanionTurnContext()

        @MainActor
        init(outcome: CompanionBeatSummonsOutcome) {
            let recorder = scratchRecorder()
            var capture: ((CompanionPingReaction) async -> Void)!
            var voice: ((String) -> Void)!
            var fallback: ((String, UUID?, UUID?) async -> Void)!
            summons = CompanionSummons(
                settings: SettingsManager(store: InMemorySettingsStore()),
                presence: CompanionPresence(recorder: recorder),
                recorder: recorder,
                context: context,
                speakPlain: { _ in },
                speakUnderOverlay: { _ in },
                summonOverlay: { _, _ in outcome },
                reportReaction: { reaction in await capture(reaction) },
                enterVoiceSession: { via in voice(via) },
                postFallbackBanner: { line, wakeID, conversationID in
                    await fallback(line, wakeID, conversationID)
                }
            )
            capture = { [weak self] reaction in
                self?.reported.append(reaction)
                self?.sequence.append("report")
            }
            voice = { [weak self] via in
                self?.voiceEntries.append(via)
                self?.sequence.append("voice")
            }
            fallback = { [weak self] line, wakeID, conversationID in
                self?.fallbacks.append((line, wakeID, conversationID))
                self?.sequence.append("fallback")
            }
        }
    }

    /// An engage reports a Reaction carrying the snapshotted correlation —
    /// the wakeID the pre-#391 path dropped, which is what the resurfacing
    /// ladder keys `heardAt` on — and enters the voice session only after
    /// the report, so the engage's minted dialogue is the one it rides.
    @Test func engageReportsTheReactionWithItsWakeThenEntersVoice() async {
        let harness = Harness(outcome: .engaged)
        let wakeID = UUID()
        let conversationID = UUID()

        await harness.summons.conclude(
            outcome: .engaged, line: "Standup in ten, sir.",
            wakeID: wakeID, conversationID: conversationID)

        #expect(harness.reported.count == 1)
        let reaction = harness.reported[0]
        #expect(reaction.outcome == .engaged)
        #expect(reaction.wakeID == wakeID)
        #expect(reaction.conversationID == conversationID)
        #expect(reaction.line == "Standup in ten, sir.")
        #expect(reaction.note == "overlay")
        #expect(harness.voiceEntries == ["summons-engage"])
        #expect(harness.sequence == ["report", "voice"])
        #expect(harness.fallbacks.isEmpty)
    }

    /// A wave-off is still a Reaction — heard is proof of reach (#309) — but
    /// no voice session and no fallback follow.
    @Test func dismissReportsTheReactionAndNothingElse() async {
        let harness = Harness(outcome: .dismissed)
        let wakeID = UUID()

        await harness.summons.conclude(
            outcome: .dismissed, line: "Lunch, sir?", wakeID: wakeID, conversationID: nil)

        #expect(harness.reported.count == 1)
        #expect(harness.reported[0].outcome == .dismissed)
        #expect(harness.reported[0].wakeID == wakeID)
        #expect(harness.voiceEntries.isEmpty)
        #expect(harness.fallbacks.isEmpty)
    }

    /// Unanswered reached no one — not a Reaction. §11 guarantee 1: the
    /// line lands as a banner with its full correlation, so a late click
    /// still routes to the wake.
    @Test func unansweredPostsTheFallbackBannerAndReportsNoReaction() async {
        let harness = Harness(outcome: .unanswered)
        let wakeID = UUID()
        let conversationID = UUID()

        await harness.summons.conclude(
            outcome: .unanswered, line: "Journal, sir.",
            wakeID: wakeID, conversationID: conversationID)

        #expect(harness.reported.isEmpty)
        #expect(harness.voiceEntries.isEmpty)
        #expect(harness.fallbacks.count == 1)
        #expect(harness.fallbacks[0].line == "Journal, sir.")
        #expect(harness.fallbacks[0].wakeID == wakeID)
        #expect(harness.fallbacks[0].conversationID == conversationID)
    }

    /// `summon` snapshots the turn's correlation before the overlay stands,
    /// speaks under the overlay, and routes the outcome through `conclude` —
    /// end to end through the real Task choreography.
    @Test func summonRoutesTheOverlayOutcomeWithTheSnapshottedCorrelation() async {
        let harness = Harness(outcome: .dismissed)
        let wakeID = UUID()
        let conversationID = UUID()
        harness.context.begin(
            turnID: UUID(), wakeIDs: [wakeID], conversationID: conversationID,
            origin: .wake)

        harness.summons.summon(line: "Evening review, sir.")
        // The unstructured Task hops the main actor twice (overlay await,
        // conclude); yield until the report lands.
        for _ in 0..<100 where harness.reported.isEmpty {
            await Task.yield()
        }

        #expect(harness.reported.count == 1)
        #expect(harness.reported[0].wakeID == wakeID)
        #expect(harness.reported[0].conversationID == conversationID)
    }
}
