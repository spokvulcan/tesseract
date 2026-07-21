//
//  CompanionFoldReducerTests.swift
//  tesseractTests
//
//  The **Companion Fold Reducer** at its own seam (ADR-0051): decision
//  tables over the Event Fold's write side — presentation, the settlement
//  invariant (consumed only by a completed turn; anything less
//  re-presents), the retry ladder's banner fallback, and reaction writes.
//  Before the cut these decisions lived inline in `CompanionLoop`, the one
//  fold module no test constructs; here each table hands the reducer a
//  batch and asserts the exact ordered effect values.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct CompanionFoldReducerTests {

    // MARK: - Fixtures

    private let now = Date(timeIntervalSinceReferenceDate: 800_000_000)
    private let turnID = UUID()
    private let conversationID = UUID()

    private func makeWake(
        _ content: String = "wake", firedAt: Date? = nil
    ) -> CompanionWake {
        CompanionWake(content: content, due: now, wakeClass: .promise, firedAt: firedAt)
    }

    private func makeEvent(_ content: String = "event") -> CompanionEvent {
        CompanionEvent(kind: .wakeDue, content: content, occurredAt: now)
    }

    // MARK: - begin: presentation

    /// Nothing drained and nothing due — no turn at all.
    @Test func emptyBatchAndWakesSkipTheTurn() {
        let reducer = CompanionFoldReducer()
        let plan = reducer.begin(batch: [], dueWakes: [], carriesBeat: true, now: now)
        #expect(plan == .skip)
    }

    /// Every due wake fires (state flipped, `firedAt` stamped once), in
    /// order, and a granted beat appends the resurfacing pass last.
    @Test func beginFiresDueWakesThenRunsTheBeatPass() {
        let reducer = CompanionFoldReducer()
        let fresh = makeWake("fresh")
        let earlier = Date(timeIntervalSinceReferenceDate: 700_000_000)
        let refired = makeWake("refired", firedAt: earlier)

        guard
            case .present(let effects) = reducer.begin(
                batch: [], dueWakes: [fresh, refired], carriesBeat: true, now: now)
        else {
            Issue.record("expected a present plan")
            return
        }

        var expectedFresh = fresh
        expectedFresh.state = .fired
        expectedFresh.firedAt = now
        var expectedRefired = refired
        expectedRefired.state = .fired

        #expect(
            effects == [
                .fireWake(expectedFresh),
                .fireWake(expectedRefired),
                .runResurfacingPass,
            ])
    }

    /// An event-only turn presents with no wake to fire and no beat — the
    /// plan is present-with-nothing-to-do, never a skip.
    @Test func eventOnlyTurnPresentsWithoutFiring() {
        let reducer = CompanionFoldReducer()
        let plan = reducer.begin(
            batch: [makeEvent()], dueWakes: [], carriesBeat: false, now: now)
        #expect(plan == .present([]))
    }

    // MARK: - settle: the invariant

    /// The success half: a completed turn consumes the batch and delivers
    /// its wakes — events first, then each wake's guarded delivery.
    @Test func completedTurnConsumesEventsAndDeliversWakes() {
        var reducer = CompanionFoldReducer()
        let events = [makeEvent("a"), makeEvent("b")]
        let wake = makeWake()

        let effects = reducer.settle(
            batch: events, wakes: [wake], outcome: (turnID, conversationID), now: now)

        #expect(
            effects == [
                .consumeEvents(ids: events.map(\.id), turnID: turnID),
                .deliverFiredWake(
                    id: wake.id, turnID: turnID, conversationID: conversationID),
            ])
    }

    /// The failure half, retries remaining: everything re-presents — wakes
    /// flipped back to booked, events back to pending — and nothing is
    /// consumed.
    @Test func failedTurnRepresentsEverythingWhileRetriesRemain() {
        var reducer = CompanionFoldReducer()
        let events = [makeEvent()]
        let wake = makeWake()

        let effects = reducer.settle(batch: events, wakes: [wake], outcome: nil, now: now)

        var rebooked = wake
        rebooked.state = .booked
        #expect(
            effects == [
                .rebookWake(rebooked),
                .representEvents(ids: events.map(\.id)),
            ])
    }

    /// Retries exhausted: each wake's own line falls back to a banner
    /// (delivered, consumed-stamped — never-silent-give-up), while the
    /// events deliberately stay presented for launch recovery.
    @Test func exhaustedRetriesFallBackToBannersAndLeaveEventsPresented() {
        var reducer = CompanionFoldReducer()
        let events = [makeEvent()]
        let wake = makeWake()

        let first = reducer.settle(batch: events, wakes: [wake], outcome: nil, now: now)
        #expect(first.count == 2)  // rebook + represent

        var delivered = wake
        delivered.state = .delivered
        delivered.consumedAt = now
        let second = reducer.settle(batch: events, wakes: [wake], outcome: nil, now: now)
        #expect(second == [.fallbackBanner(delivered)])

        // The ledger cleared with the fallback: a later failure of the same
        // batch starts the ladder over instead of banner-spamming.
        let third = reducer.settle(batch: events, wakes: [wake], outcome: nil, now: now)
        #expect(third.count == 2)
    }

    /// The stamp survives the ladder: threading the presentation's fired
    /// copies through settlement — as the loop does — keeps `firedAt` at
    /// its first value across the rebook, the re-fire, and the exhausted
    /// fallback. Stamped once, never re-stamped, never dropped.
    @Test func firedAtSurvivesTheRetryLadder() {
        var reducer = CompanionFoldReducer()
        let wake = makeWake()

        // First presentation stamps the wake.
        guard
            case .present(let firstEffects) = reducer.begin(
                batch: [], dueWakes: [wake], carriesBeat: false, now: now)
        else {
            Issue.record("expected a present plan")
            return
        }
        let fired = firedWakes(in: firstEffects)
        #expect(fired.map(\.firedAt) == [now])

        // First failure: the rebooked copy keeps the stamp.
        var rebooked = fired[0]
        rebooked.state = .booked
        let firstSettle = reducer.settle(batch: [], wakes: fired, outcome: nil, now: now)
        #expect(firstSettle == [.rebookWake(rebooked)])
        #expect(rebooked.firedAt == now)

        // The retry fires later — the stamp stays the first one's.
        let later = now.addingTimeInterval(60)
        guard
            case .present(let retryEffects) = reducer.begin(
                batch: [], dueWakes: [rebooked], carriesBeat: false, now: later)
        else {
            Issue.record("expected a present plan")
            return
        }
        let refired = firedWakes(in: retryEffects)
        #expect(refired.map(\.firedAt) == [now])

        // Exhaustion: the banner fallback still carries the first stamp.
        var delivered = refired[0]
        delivered.state = .delivered
        delivered.consumedAt = later
        let fallback = reducer.settle(batch: [], wakes: refired, outcome: nil, now: later)
        #expect(fallback == [.fallbackBanner(delivered)])
        #expect(delivered.firedAt == now)
    }

    /// The loop's own extraction: the plan's fired copies become the
    /// turn's wake values for everything after presentation.
    private func firedWakes(
        in effects: [CompanionFoldReducer.Effect]
    ) -> [CompanionWake] {
        effects.compactMap {
            guard case .fireWake(let wake) = $0 else { return nil }
            return wake
        }
    }

    /// The attempt ledger keys by the earliest wake, falling back to the
    /// earliest event for an event-only batch; a success clears the whole
    /// ledger.
    @Test func attemptLedgerKeysAndSuccessClearing() {
        var reducer = CompanionFoldReducer()
        let events = [makeEvent()]

        // Event-only batch: same key across failures reaches the fallback —
        // which for a wakeless batch has no banner to post.
        _ = reducer.settle(batch: events, wakes: [], outcome: nil, now: now)
        let exhausted = reducer.settle(batch: events, wakes: [], outcome: nil, now: now)
        #expect(exhausted == [])

        // A success wipes every key: the next failure counts from one.
        _ = reducer.settle(batch: events, wakes: [], outcome: nil, now: now)
        _ = reducer.settle(
            batch: events, wakes: [], outcome: (turnID, conversationID), now: now)
        let afterSuccess = reducer.settle(batch: events, wakes: [], outcome: nil, now: now)
        #expect(afterSuccess == [.representEvents(ids: events.map(\.id))])
    }

    /// An empty settlement (nothing drained, no wakes) decides nothing —
    /// there is no key to count an attempt under.
    @Test func emptySettlementIsInert() {
        var reducer = CompanionFoldReducer()
        let effects = reducer.settle(batch: [], wakes: [], outcome: nil, now: now)
        #expect(effects == [])
    }

    // MARK: - reactions

    /// Heard is stamped first for every outcome — a wave-off is still
    /// proof the delivery reached him; nothing else happens on dismissal.
    @Test func dismissalStampsHeardAndNothingElse() {
        let reducer = CompanionFoldReducer()
        let wakeID = UUID()
        let effects = reducer.reaction(
            outcome: .dismissed, wakeID: wakeID, conversationID: nil, note: nil,
            surface: .banner)
        #expect(effects == [.stampWakeHeard(id: wakeID)])
    }

    /// Engaging the banner upgrades the wake, spares the resurfacing
    /// agenda, and opens the correlated conversation — in that order.
    @Test func engagementUpgradesTheWakeAndOpensTheConversation() {
        let reducer = CompanionFoldReducer()
        let wakeID = UUID()
        let effects = reducer.reaction(
            outcome: .engaged, wakeID: wakeID, conversationID: conversationID, note: nil,
            surface: .banner)
        #expect(
            effects == [
                .stampWakeHeard(id: wakeID),
                .engageWake(id: wakeID),
                .stampResurfacedHeard,
                .openConversation(id: conversationID),
            ])
    }

    /// An engage correlated to Mission Control — read-only, never a click
    /// destination — mints a dialogue seeded with the banner's line instead
    /// (ADR-0052), and so does an engage with no correlation at all (the
    /// pre-ADR-0052 dead click).
    @Test func engagementNeverOpensTheFoldItMintsADialogue() {
        let reducer = CompanionFoldReducer()
        let wakeID = UUID()
        let toFold = reducer.reaction(
            outcome: .engaged, wakeID: wakeID,
            conversationID: AgentConversation.missionControlID,
            line: "Evening journal is open, sir.", note: nil, surface: .banner)
        #expect(
            toFold == [
                .stampWakeHeard(id: wakeID),
                .engageWake(id: wakeID),
                .stampResurfacedHeard,
                .beginDialogue(line: "Evening journal is open, sir.", via: "banner-engage"),
            ])

        let uncorrelated = reducer.reaction(
            outcome: .engaged, wakeID: nil, conversationID: nil,
            line: "You're in, sir.", note: nil, surface: .banner)
        #expect(
            uncorrelated == [
                .stampResurfacedHeard,
                .beginDialogue(line: "You're in, sir.", via: "banner-engage"),
            ])
    }

    /// A reply becomes a followup wake due now, carrying his words and the
    /// conversation, and accelerates the loop; an empty reply just stamps.
    @Test func replyBooksTheFollowupAndAccelerates() {
        let reducer = CompanionFoldReducer()
        let effects = reducer.reaction(
            outcome: .replied, wakeID: nil, conversationID: conversationID, note: "on my way",
            surface: .banner)
        #expect(
            effects == [
                .stampResurfacedHeard,
                .bookReplyFollowup(
                    content: "He replied to your notification: \"on my way\" — respond.",
                    conversationID: conversationID),
                .accelerateEvaluation,
            ])

        let empty = reducer.reaction(
            outcome: .replied, wakeID: nil, conversationID: nil, note: "",
            surface: .banner)
        #expect(empty == [.stampResurfacedHeard])
    }

    /// The overlay summons reports through the same table (#391): an engage
    /// stamps heard, upgrades the wake, and mints the dialogue — identical
    /// writes to a banner engage, only the dialogue's provenance differs.
    @Test func summonsEngageDecidesTheSameWritesAsABannerEngage() {
        let reducer = CompanionFoldReducer()
        let wakeID = UUID()
        let effects = reducer.reaction(
            outcome: .engaged, wakeID: wakeID,
            conversationID: AgentConversation.missionControlID,
            line: "Standup in ten, sir.", note: "overlay", surface: .overlaySummons)
        #expect(
            effects == [
                .stampWakeHeard(id: wakeID),
                .engageWake(id: wakeID),
                .stampResurfacedHeard,
                .beginDialogue(line: "Standup in ten, sir.", via: "summons-engage"),
            ])
    }

    /// A summons wave-off is still a heard delivery — the stamp the
    /// pre-#391 summons path skipped, letting an answered promise die
    /// delivered-unheard in the resurfacing ladder.
    @Test func summonsDismissalStampsHeard() {
        let reducer = CompanionFoldReducer()
        let wakeID = UUID()
        let effects = reducer.reaction(
            outcome: .dismissed, wakeID: wakeID, conversationID: nil, note: "overlay",
            surface: .overlaySummons)
        #expect(effects == [.stampWakeHeard(id: wakeID)])
    }
}
