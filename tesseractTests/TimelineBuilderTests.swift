//
//  TimelineBuilderTests.swift
//  tesseractTests
//
//  The Today Timeline from a fixture day: time order, free gaps of half an
//  hour or more, the Now line, "slid" instead of "missed", done items
//  dimmed, and the counts in the header.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct TimelineBuilderTests {

    static func local(_ day: Int, _ hour: Int, _ minute: Int = 0) -> Date {
        Calendar.current.date(
            from: DateComponents(year: 2026, month: 9, day: day, hour: hour, minute: minute))!
    }

    static let standup = AgendaEvent(
        id: "E1", title: "Standup", start: local(30, 9, 30), end: local(30, 10),
        calendarID: "c", calendarTitle: "Work")
    static let oneOnOne = AgendaEvent(
        id: "E2", title: "1:1", start: local(30, 13), end: local(30, 13, 45),
        calendarID: "c", calendarTitle: "Work")

    static func facts(now: Date, plan: [Placement] = [], mustDo: String? = nil) -> DayFacts {
        DayFacts(
            now: now, events: [standup, oneOnOne],
            dueOrOverdue: [
                AgendaReminder(
                    id: "dentist", title: "Call the dentist", listID: "health", listTitle: "Health",
                    due: local(30, 11), dueHasTime: true),
                AgendaReminder(
                    id: "rent", title: "Pay rent", listID: "life", listTitle: "Life",
                    due: local(30, 0)),
                AgendaReminder(
                    id: "passport", title: "Renew passport", listID: "life", listTitle: "Life",
                    due: local(28, 0)),
            ],
            undated: [
                AgendaReminder(id: "gym", title: "Gym", listID: "health", listTitle: "Health")
            ],
            doneToday: [
                AgendaReminder(
                    id: "mail", title: "Answer mail", listID: "work", listTitle: "Work",
                    due: local(30, 0), isCompleted: true, completedAt: local(30, 8))
            ],
            areas: [
                Area(id: "health", name: "Health"), Area(id: "life", name: "Life"),
                Area(id: "work", name: "Work"),
            ],
            mustDoID: mustDo, plan: plan)
    }

    @Test func theDayReadsInTimeOrderWithTheNowLine() {
        let timeline = TimelineBuilder.build(facts: Self.facts(now: Self.local(30, 10, 30)))
        let kinds = timeline.rows.map { row -> String in
            switch row.kind {
            case .event(let event): event.title
            case .task(let task): task.reminder.title
            case .free(let minutes): "free \(minutes)"
            case .now: "now"
            }
        }
        // A gap of exactly half an hour counts as free time.
        #expect(
            kinds == [
                "Standup", "now", "free 30", "Call the dentist", "free 105", "1:1", "free 495",
            ])
        #expect(timeline.rows.first?.isPast == true)
    }

    @Test func anUnfinishedPastTaskSlidesAndIsNeverMissed() throws {
        let timeline = TimelineBuilder.build(facts: Self.facts(now: Self.local(30, 12)))
        let dentist = try #require(
            timeline.rows.compactMap { row -> TimelineTask? in
                if case .task(let task) = row.kind { task } else { nil }
            }.first)
        #expect(dentist.isSlid)
        #expect(!dentist.isDone)
    }

    @Test func plannedTasksTakeTheirSlotAndTheRestWaitAnytime() {
        let plan = [Placement(reminderID: "gym", start: Self.local(30, 17), minutes: 60)]
        let timeline = TimelineBuilder.build(
            facts: Self.facts(now: Self.local(30, 8), plan: plan, mustDo: "gym"))
        let timed = timeline.rows.compactMap { row -> String? in
            if case .task(let task) = row.kind { task.reminder.title } else { nil }
        }
        #expect(timed == ["Call the dentist", "Gym"])
        #expect(
            timeline.anytime.map(\.reminder.title) == ["Pay rent", "Renew passport", "Answer mail"])
        #expect(timeline.anytime.first { $0.id == "passport" }?.isCarried == true)
        #expect(timeline.mustDo?.id == "gym")
        // Today's tasks: dentist, gym, rent, mail (the carried passport is not today's).
        #expect(timeline.totalCount == 4)
        #expect(timeline.doneCount == 1)
    }

    @Test func freeGapsSkipShortOnes() {
        let gaps = TimelineBuilder.freeGaps(
            from: Self.local(30, 9), to: Self.local(30, 12),
            busy: [
                DateInterval(start: Self.local(30, 9, 20), end: Self.local(30, 10)),
                DateInterval(start: Self.local(30, 10, 20), end: Self.local(30, 11)),
            ])
        #expect(gaps == [DateInterval(start: Self.local(30, 11), end: Self.local(30, 12))])
    }

    @Test func findATimeTakesTheNextQuarterHourThatFits() {
        let facts = Self.facts(now: Self.local(30, 10, 7))
        #expect(TimelineBuilder.firstFreeSlot(minutes: 30, facts: facts) == Self.local(30, 10, 15))
    }
}

struct CardParserTests {

    static let facts = TimelineBuilderTests.facts(now: TimelineBuilderTests.local(30, 7, 30))

    @Test(arguments: [
        #"{"line": "Hi", "must_do": null, "plan": [], "suggestions": []}"#,
        "Sure.\n```json\n{\"line\": \"Hi\"}\n```\nDone.",
        "I think {\"line\": \"Hi\", \"plan\": []} works",
    ])
    func findsTheCardInsideTheReply(_ reply: String) {
        guard case .card(.morningPlan(let card)) = CardParser.morningPlan(reply, facts: Self.facts)
        else {
            Issue.record("expected a Morning Plan card")
            return
        }
        #expect(card.line == "Hi")
    }

    @Test(arguments: ["", "No JSON here.", #"{"plan": []}"#, #"{"line": "   "}"#, "{broken"])
    func rejectsWhatIsNotACard(_ reply: String) {
        if case .card = CardParser.morningPlan(reply, facts: Self.facts) {
            Issue.record("\(reply) must not parse")
        }
    }

    @Test func dropsUnknownIdsAndBadTimesButKeepsTheCard() {
        let reply =
            #"{"line": "Ok", "must_do": "nope", "plan": [{"id": "nope", "at": "08:00"}, {"id": "gym", "at": "25:00"}, {"id": "rent", "at": "08:15", "minutes": 1}], "suggestions": ["a", "b", "c", "d"]}"#
        guard case .card(.morningPlan(let card)) = CardParser.morningPlan(reply, facts: Self.facts)
        else {
            Issue.record("expected a Morning Plan card")
            return
        }
        #expect(card.mustDoID == nil)
        #expect(card.placements.map(\.reminderID) == ["rent"])
        #expect(card.placements.first?.minutes == 5)
        #expect(card.suggestions.count == 3)
    }

    @Test func aLongLineIsTrimmed() {
        let line = String(repeating: "a", count: 500)
        #expect(CardParser.cleanLine(line)?.count == 318)
    }
}
