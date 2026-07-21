//
//  CompanionFoldRenderTests.swift
//  tesseractTests
//
//  The line primitives both briefings compose (#403): one rendering per
//  concept, pinned here as a decision table so the Situation Briefing and the
//  Fold Briefing can never drift apart by hand again.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@Suite struct CompanionFoldRenderTests {

    // MARK: - contractLine

    @Test func contractLineWithoutADayReadsNoContract() {
        #expect(CompanionFoldRender.contractLine(today: nil) == "No contract for today yet.")
    }

    @Test func contractLineWithAnEmptyChainReadsNoContract() {
        let empty = DayRecord(date: "2026-07-21")
        #expect(
            CompanionFoldRender.contractLine(today: empty) == "No contract for today yet.")
    }

    @Test func contractLineWithAChainIsTheDaysSummary() {
        var day = DayRecord(date: "2026-07-21")
        day.chain = [ContractStep(title: "Ship the render", status: .active)]
        let line = CompanionFoldRender.contractLine(today: day)
        #expect(line == day.chainSummary)
        #expect(line.contains("Contract for 2026-07-21:"))
        #expect(line.contains("Ship the render"))
    }

    // MARK: - dueHeading

    @Test func dueHeadingCarriesTheTurnTriggerClauseOnlyForTheTurn() {
        #expect(
            CompanionFoldRender.dueHeading(triggeredThisTurn: true)
                == "DUE NOW — the wakes that triggered this turn:")
        #expect(CompanionFoldRender.dueHeading(triggeredThisTurn: false) == "DUE NOW:")
        // The "DUE NOW" emphasis rides both surfaces.
        #expect(CompanionFoldRender.dueHeading(triggeredThisTurn: true).contains("DUE NOW"))
        #expect(CompanionFoldRender.dueHeading(triggeredThisTurn: false).contains("DUE NOW"))
    }

    // MARK: - dueWakeLine

    @Test func dueWakeLineWithoutANowOmitsLateness() {
        let wake = CompanionWake(
            content: "midday pulse", due: Date().addingTimeInterval(-3600),
            wakeClass: .rhythm, state: .fired)
        let line = CompanionFoldRender.dueWakeLine(wake)
        #expect(line == "- \(wake.briefingLine)")
        #expect(!line.contains("overdue"))
    }

    @Test func dueWakeLineReportsLatenessPastTheGrace() {
        let now = Date()
        let wake = CompanionWake(
            content: "morning planning", due: now.addingTimeInterval(-8 * 60),
            wakeClass: .rhythm, state: .fired)
        #expect(
            CompanionFoldRender.dueWakeLine(wake, now: now).contains("(overdue by 8 min)"))
    }

    @Test func dueWakeLineWithinTheGraceReportsNoLateness() {
        let now = Date()
        let wake = CompanionWake(
            content: "morning planning", due: now.addingTimeInterval(-3 * 60),
            wakeClass: .rhythm, state: .fired)
        #expect(!CompanionFoldRender.dueWakeLine(wake, now: now).contains("overdue"))
    }

    @Test func dueWakeLineCarriesTheHandleOnlyWhileOpen() {
        let open = CompanionWake(
            content: "open one", due: Date(), wakeClass: .promise, state: .fired)
        var closed = CompanionWake(
            content: "done one", due: Date(), wakeClass: .promise, state: .delivered)
        closed.firedAt = Date()
        #expect(CompanionFoldRender.dueWakeLine(open).contains("[id \(open.shortID)]"))
        #expect(!CompanionFoldRender.dueWakeLine(closed).contains("[id \(closed.shortID)]"))
    }

    // MARK: - upcomingHeading / upcomingWakeLine

    @Test func upcomingHeadingIsTheChosenLeanForm() {
        #expect(CompanionFoldRender.upcomingHeading == "Booked ahead:")
    }

    @Test func upcomingWakeLineDatesTheBriefingLine() {
        let wake = CompanionWake(
            content: "evening journal", due: Date().addingTimeInterval(6 * 3600),
            wakeClass: .rhythm)
        let line = CompanionFoldRender.upcomingWakeLine(wake)
        #expect(line.hasPrefix("- "))
        #expect(line.contains(wake.briefingLine))
        #expect(line.contains("[id \(wake.shortID)]"))
    }

    // MARK: - firedWakeLine

    @Test func firedWakeLineFlagsAnUnheardDelivery() {
        var wake = CompanionWake(
            content: "evening journal", due: Date().addingTimeInterval(-1800),
            wakeClass: .rhythm, state: .delivered)
        wake.firedAt = Date().addingTimeInterval(-1800)
        let line = CompanionFoldRender.firedWakeLine(wake)
        #expect(line.contains("evening journal"))
        #expect(line.contains("(delivered, unheard)"))
        // A delivered wake is terminal — no actionable handle.
        #expect(!line.contains("[id \(wake.shortID)]"))
    }

    @Test func firedWakeLineDropsTheUnheardFlagOnceHeard() {
        var wake = CompanionWake(
            content: "midday pulse", due: Date().addingTimeInterval(-600),
            wakeClass: .promise, state: .engaged)
        wake.firedAt = Date().addingTimeInterval(-600)
        wake.heardAt = Date().addingTimeInterval(-300)
        let line = CompanionFoldRender.firedWakeLine(wake)
        #expect(line.contains("(engaged)"))
        #expect(!line.contains("unheard"))
    }

    @Test func firedWakeLineWithoutAFireTimeRendersAPlaceholder() {
        let wake = CompanionWake(
            content: "orphan", due: Date(), wakeClass: .promise, state: .fired)
        // A still-open fired wake keeps its handle even in this section.
        let line = CompanionFoldRender.firedWakeLine(wake)
        #expect(line.hasPrefix("- ? "))
        #expect(line.contains("[id \(wake.shortID)]"))
    }
}
