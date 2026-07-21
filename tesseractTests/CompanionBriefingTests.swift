//
//  CompanionBriefingTests.swift
//  tesseractTests
//
//  The Situation Briefing (ADR-0040) at its composition seam (#403): the
//  `<situation>` block the entity reads at the start of every turn, built by
//  splicing the shared Companion Fold Render primitives with the turn-only
//  sections (presence, yesterday, the weekly numbers). Rows here pin what the
//  block carries and in what shape — the weekly-numbers splice among them,
//  previously unexercised.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
@Suite struct CompanionBriefingTests {

    /// A minimal situation: owner present, nothing booked, no contract. Rows
    /// override only the field under test.
    private func inputs(
        now: Date = Date(),
        today: DayRecord? = nil,
        dueWakes: [CompanionWake] = [],
        upcomingWakes: [CompanionWake] = [],
        weeklyNumbers: String? = nil
    ) -> CompanionBriefing.Inputs {
        CompanionBriefing.Inputs(
            now: now, ownerPresent: true, screenLocked: false, frontmostApp: "Xcode",
            onACPower: true, today: today, yesterday: nil, dueWakes: dueWakes,
            upcomingWakes: upcomingWakes, weeklyNumbers: weeklyNumbers)
    }

    // MARK: - Envelope

    @Test func alwaysWrappedInTheSituationTags() {
        let text = CompanionBriefing.render(inputs())
        #expect(text.hasPrefix("<situation>"))
        #expect(text.hasSuffix("</situation>"))
    }

    // MARK: - Contract line

    @Test func aChainRendersTheContractSummary() {
        var today = DayRecord(date: TrackingDay.key(for: Date()))
        today.chain = [ContractStep(title: "Ship the render", status: .active)]
        let text = CompanionBriefing.render(inputs(today: today))
        #expect(text.contains(today.chainSummary))
        #expect(text.contains("Ship the render"))
    }

    @Test func noChainReadsNoContract() {
        #expect(CompanionBriefing.render(inputs()).contains("No contract for today yet."))
    }

    // MARK: - Due wakes

    @Test func dueWakesRenderUnderDueNowEmphasisWithLateness() {
        let now = Date()
        let due = CompanionWake(
            content: "morning planning", due: now.addingTimeInterval(-8 * 60),
            wakeClass: .rhythm, state: .fired)
        let text = CompanionBriefing.render(inputs(now: now, dueWakes: [due]))
        #expect(text.contains("DUE NOW — the wakes that triggered this turn:"))
        #expect(
            text.contains("[rhythm] morning planning [id \(due.shortID)] (overdue by 8 min)"))
    }

    @Test func noDueWakesRendersNoDueSection() {
        #expect(!CompanionBriefing.render(inputs()).contains("DUE NOW"))
    }

    // MARK: - Upcoming wakes

    @Test func upcomingWakesRenderUnderBookedAhead() {
        let upcoming = CompanionWake(
            content: "evening journal", due: Date().addingTimeInterval(6 * 3600),
            wakeClass: .rhythm)
        let text = CompanionBriefing.render(inputs(upcomingWakes: [upcoming]))
        #expect(text.contains("Booked ahead:"))
        #expect(text.contains("evening journal [id \(upcoming.shortID)]"))
    }

    @Test func anEmptyFutureDemandsARhythm() {
        let text = CompanionBriefing.render(inputs())
        #expect(text.contains("You have NOTHING booked ahead"))
        #expect(!text.contains("Booked ahead:"))
    }

    // MARK: - Weekly-numbers splice (#313)

    @Test func weeklyNumbersSpliceRidesAfterABlankLineAtTheTail() {
        let weekly = "WEEKLY REVIEW\nkept 5/7, promises 9/10"
        let text = CompanionBriefing.render(inputs(weeklyNumbers: weekly))
        // The splice is separated from the situation body by a blank line and
        // sits at the tail, just inside the closing tag.
        #expect(text.contains("\n\n\(weekly)\n</situation>"))
    }

    @Test func noWeeklyNumbersLeavesNoTrailingBlank() {
        let text = CompanionBriefing.render(inputs())
        #expect(!text.contains("\n\n</situation>"))
    }
}
