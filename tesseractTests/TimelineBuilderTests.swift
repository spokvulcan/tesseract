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

    // MARK: Tomorrow

    static let work = AgendaEvent(
        id: "W", title: "Work", start: local(31, 9), end: local(31, 13), calendarID: "c",
        calendarTitle: "Work")
    static let birthday = AgendaEvent(
        id: "B", title: "Mom's birthday", start: local(31, 0), end: local(32, 0), isAllDay: true,
        calendarID: "c", calendarTitle: "Family")
    static let bins = AgendaReminder(
        id: "bins", title: "Put the bins out", listID: "life", listTitle: "Life",
        due: local(31, 7, 30), dueHasTime: true)
    static let invoice = AgendaReminder(
        id: "invoice", title: "Send the invoice", listID: "work", listTitle: "Work",
        due: local(31, 0))

    /// The fixture day with tomorrow on it: work, a birthday, the bins at
    /// 07:30 and the invoice at no set time.
    static func twoDays(now: Date) -> DayFacts {
        var facts = facts(now: now)
        facts.events += [work, birthday]
        facts.dueTomorrow = [invoice, bins]
        return facts
    }

    @Test func tomorrowFollowsWithItsEventsAndTasks() {
        let tomorrow = TimelineBuilder.build(facts: Self.twoDays(now: Self.local(30, 10, 30)))
            .tomorrow
        #expect(tomorrow.date == Self.local(31, 0))
        let rows = tomorrow.rows.map { row -> String in
            switch row.kind {
            case .event(let event): event.title
            case .task(let task): task.reminder.title
            case .free, .now: "?"
            }
        }
        // Not planned yet: no free time and no Now line.
        #expect(rows == ["Put the bins out", "Work"])
        #expect(tomorrow.anytime.map(\.reminder.title) == ["Send the invoice"])
        #expect(tomorrow.allDayEvents.map(\.title) == ["Mom's birthday"])
    }

    @Test func aTaskDueTomorrowAndDoneEarlyStaysWithTomorrow() {
        var facts = Self.twoDays(now: Self.local(30, 20))
        facts.dueTomorrow = [Self.bins]
        var invoice = Self.invoice
        invoice.isCompleted = true
        invoice.completedAt = Self.local(30, 19)
        facts.doneToday.append(invoice)
        let timeline = TimelineBuilder.build(facts: facts)
        #expect(timeline.tomorrow.anytime.map(\.id) == ["invoice"])
        #expect(timeline.tomorrow.anytime.first?.isDone == true)
        // Today's count is today's: dentist, rent and the mail done this morning.
        #expect(!timeline.anytime.contains { $0.id == "invoice" })
        #expect(timeline.totalCount == 3)
        #expect(timeline.doneCount == 1)
    }

    @Test func untilFourTheSmallHoursStillBelongToTheDayThatIsEnding() {
        let facts = Self.twoDays(now: Self.local(31, 0, 40))
        #expect(facts.startOfToday == Self.local(30, 0))
        let timeline = TimelineBuilder.build(facts: facts)
        let today = timeline.rows.map { row -> String in
            switch row.kind {
            case .event(let event): event.title
            case .task(let task): task.reminder.title
            case .free: "free"
            case .now: "now"
            }
        }
        // The 30th's day, past its end: no free time left, the Now line last.
        #expect(today == ["Standup", "Call the dentist", "1:1", "now"])
        #expect(timeline.tomorrow.date == Self.local(31, 0))
        #expect(timeline.tomorrow.rows.count == 2)
    }

    @Test func theSnapshotSplitsTasksIntoTodayTomorrowAndUndated() {
        let snapshot = AgendaSnapshot(
            takenAt: Self.local(31, 0, 40), access: .full, events: [],
            open: [
                AgendaReminder(
                    id: "rent", title: "Pay rent", listID: "life", listTitle: "Life",
                    due: Self.local(30, 0)),
                Self.invoice, Self.bins,
                AgendaReminder(
                    id: "later", title: "Book flights", listID: "life", listTitle: "Life",
                    due: Self.local(32, 0)),
                AgendaReminder(id: "gym", title: "Gym", listID: "health", listTitle: "Health"),
            ], doneToday: [], lists: [], calendars: [])
        // At 00:40 the 31st is still tomorrow: the day rolls over at 04:00.
        let facts = DayFacts(
            snapshot: snapshot, areas: [], inboxListID: nil, now: Self.local(31, 0, 40))
        #expect(facts.dueOrOverdue.map(\.id) == ["rent"])
        #expect(facts.dueTomorrow.map(\.id) == ["invoice", "bins"])
        #expect(facts.undated.map(\.id) == ["gym"])
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

    @Test func aTimeToLeaveIsKeptOnlyJustBeforeAnEventTheRequestListed() {
        // As the request listed them: e1 the standup at 09:30, e2 the 1:1 at 13:00.
        let reply =
            #"{"line": "Ok", "leave": [{"event": "e2", "at": "08:00"}, {"event": "e1", "at": "09:45"}, {"event": "e9", "at": "10:00"}, {"event": "e2", "at": "12:30"}, {"event": "e2", "at": "12:40"}]}"#
        guard
            case .card(.morningPlan(let card)) = CardParser.morningPlan(
                reply, facts: Self.facts, eventIDs: ["E1", "E2"])
        else {
            Issue.record("expected a Morning Plan card")
            return
        }
        // Hours ahead, after the start, unknown, and a second time: dropped.
        #expect(card.departures.map(\.eventID) == ["E2"])
        #expect(card.departures.first?.at == TimelineBuilderTests.local(30, 12, 30))
        #expect(card.departures.first?.eventStart == TimelineBuilderTests.local(30, 13))
    }

    @Test func aCardSavedBeforeDeparturesStillLoads() throws {
        let json = #"{"line": "Hi", "placements": [], "suggestions": []}"#
        let card = try JSONDecoder().decode(MorningPlanCard.self, from: Data(json.utf8))
        #expect(card.departures.isEmpty)
    }
}
