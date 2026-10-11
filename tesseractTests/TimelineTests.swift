//
//  TimelineTests.swift
//  tesseractTests
//
//  The Timeline, the day's one read (ADR-0091), from a fixture day's
//  snapshot and state: time order, free gaps of half an hour or more, the
//  Now line, "slid" instead of "missed", done items dimmed, the counts in
//  the header, tomorrow after today and the small hours past midnight; the
//  step the day is on, by its one rule, read the same on the Now Card, the
//  Day Line, the menu bar and the engine; a time to leave only while its
//  event stands; the Inbox.
//

import Foundation
import Testing

@testable import Tesseract_Agent

/// The fixture day, Wednesday 30 September 2026: a standup (09:30) and a
/// 1:1 (13:00), blocks of the owner's own; the dentist due at 11:00, rent
/// due today, the passport carried from the 28th, the gym with no date; the
/// mail done at 08:00.
enum DayFixture {

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
    static let dentist = AgendaReminder(
        id: "dentist", title: "Call the dentist", listID: "health", listTitle: "Health",
        due: local(30, 11), dueHasTime: true)
    static let rent = AgendaReminder(
        id: "rent", title: "Pay rent", listID: "life", listTitle: "Life", due: local(30, 0))
    static let passport = AgendaReminder(
        id: "passport", title: "Renew passport", listID: "life", listTitle: "Life",
        due: local(28, 0))
    static let gym = AgendaReminder(
        id: "gym", title: "Gym", listID: "health", listTitle: "Health")
    static let mail = AgendaReminder(
        id: "mail", title: "Answer mail", listID: "work", listTitle: "Work", due: local(30, 0),
        isCompleted: true, completedAt: local(30, 8))
    static let areas = [
        Area(id: "health", name: "Health"), Area(id: "life", name: "Life"),
        Area(id: "work", name: "Work"),
    ]

    /// The fixture day's snapshot at `now`.
    static func snapshot(
        now: Date, events: [AgendaEvent] = [standup, oneOnOne],
        open: [AgendaReminder] = [dentist, rent, passport, gym], done: [AgendaReminder] = [mail],
        settings: DaySettings = DaySettings()
    ) -> DaySnapshot {
        var agenda = AgendaSnapshot.empty
        agenda.access = .full
        agenda.events = events
        agenda.open = open
        agenda.doneToday = done
        return DaySnapshot(
            now: now, settings: settings, agenda: agenda, areas: areas, inboxListID: "inbox")
    }

    /// The fixture day's state: planned at 08:00, with `plan` and `mustDo`.
    static func state(plan: [Step] = [], mustDo: String? = nil) -> DayState {
        var state = DayState(day: DayKey(rawValue: "2026-09-30"))
        state.morningPlanAt = local(30, 8)
        state.plan = plan
        state.mustDoID = mustDo
        return state
    }

    /// The fixture day at `now`.
    static func day(now: Date, plan: [Step] = [], mustDo: String? = nil) -> Timeline {
        Timeline(snapshot: snapshot(now: now), state: state(plan: plan, mustDo: mustDo))
    }

    /// The facts a moment is asked about on the fixture day.
    static func facts(now: Date, plan: [Placement] = []) -> DayFacts {
        snapshot(now: now).facts(state: state(plan: plan.map(Step.init)))
    }
}

struct TimelineTests {

    static func local(_ day: Int, _ hour: Int, _ minute: Int = 0) -> Date {
        DayFixture.local(day, hour, minute)
    }

    static func names(_ rows: [TimelineRow]) -> [String] {
        rows.map { row in
            switch row.kind {
            case .event(let event): event.title
            case .task(let task): task.reminder.title
            case .free(let minutes): "free \(minutes)"
            case .now: "now"
            }
        }
    }

    @Test func theDayReadsInTimeOrderWithTheNowLine() {
        let day = DayFixture.day(now: Self.local(30, 10, 30))
        // A gap of exactly half an hour counts as free time.
        #expect(
            Self.names(day.rows) == [
                "Standup", "now", "free 30", "Call the dentist", "free 105", "1:1", "free 495",
            ])
        #expect(day.rows.first?.isPast == true)
    }

    @Test func aGapShorterThanHalfAnHourIsNotFreeTime() {
        // Twenty minutes before each of two meetings: only after them is free.
        let first = AgendaEvent(
            id: "a", title: "Review", start: Self.local(30, 9, 20), end: Self.local(30, 10),
            calendarID: "c", calendarTitle: "Work")
        let second = AgendaEvent(
            id: "b", title: "Planning", start: Self.local(30, 10, 20), end: Self.local(30, 11),
            calendarID: "c", calendarTitle: "Work")
        let day = Timeline(
            snapshot: DayFixture.snapshot(
                now: Self.local(30, 9), events: [first, second], open: [], done: []),
            state: DayFixture.state())
        #expect(Self.names(day.rows) == ["now", "Review", "Planning", "free 660"])
    }

    @Test func anUnfinishedPastTaskSlidesAndIsNeverMissed() throws {
        let day = DayFixture.day(now: Self.local(30, 12))
        let dentist = try #require(
            day.rows.compactMap { row -> TimelineTask? in
                if case .task(let task) = row.kind { task } else { nil }
            }.first)
        #expect(dentist.isSlid)
        #expect(!dentist.isDone)
    }

    @Test func plannedTasksTakeTheirSlotAndTheRestWaitAnytime() {
        let day = DayFixture.day(
            now: Self.local(30, 8), plan: [Step("gym", at: Self.local(30, 17), minutes: 60)],
            mustDo: "gym")
        let timed = day.rows.compactMap { row -> String? in
            if case .task(let task) = row.kind { task.reminder.title } else { nil }
        }
        #expect(timed == ["Call the dentist", "Gym"])
        #expect(day.anytime.map(\.reminder.title) == ["Pay rent", "Renew passport", "Answer mail"])
        #expect(day.anytime.first { $0.id == "passport" }?.isCarried == true)
        #expect(day.mustDo?.id == "gym")
        // Today's tasks: dentist, gym, rent, mail (the carried passport is not today's).
        #expect(day.totalCount == 4)
        #expect(day.doneCount == 1)
    }

    @Test func theWayToAnEventIsBusyOnlyWhileTheEventStands() {
        func freeBeforeTheOneOnOne(_ departure: Departure) -> [Int] {
            var state = DayFixture.state()
            state.departures = [departure]
            let day = Timeline(
                snapshot: DayFixture.snapshot(now: Self.local(30, 10, 30)), state: state)
            return day.rows.compactMap { row in
                guard case .free(let minutes) = row.kind, row.start >= Self.local(30, 11),
                    row.start < Self.local(30, 13)
                else { return nil }
                return minutes
            }
        }
        // Leave at 12:30 for the 1:1 at 13:00: the way there is not free.
        let leave = Departure(
            eventID: "E2", title: "1:1", at: Self.local(30, 12, 30),
            eventStart: Self.local(30, 13))
        #expect(freeBeforeTheOneOnOne(leave) == [75])
        // Planned for a start the 1:1 no longer has, or for an event since
        // cancelled: the time is free again.
        var moved = leave
        moved.eventStart = Self.local(30, 13, 30)
        #expect(freeBeforeTheOneOnOne(moved) == [105])
        var cancelled = leave
        cancelled.eventID = "E9"
        #expect(freeBeforeTheOneOnOne(cancelled) == [105])
    }

    @Test func findATimeTakesTheNextQuarterHourThatFits() {
        #expect(
            DayFixture.day(now: Self.local(30, 10, 7)).firstFreeSlot(minutes: 30)
                == Self.local(30, 10, 15))
    }

    // MARK: The step the day is on

    /// "Work", 09:00–13:00: no one else, nowhere to go — a block of the
    /// owner's own.
    static let block = AgendaEvent(
        id: "work", title: "Work", start: local(30, 9), end: local(30, 13), calendarID: "c",
        calendarTitle: "Work")
    /// A meeting inside it, with other people.
    static let sync = AgendaEvent(
        id: "sync", title: "Sync", start: local(30, 10), end: local(30, 10, 30), calendarID: "c",
        calendarTitle: "Work", hasOtherAttendees: true)
    /// Somewhere to be inside it: no one invited, but a place.
    static let dentistVisit = AgendaEvent(
        id: "visit", title: "Dentist", start: local(30, 10), end: local(30, 10, 30),
        calendarID: "c", calendarTitle: "Home", location: "Dental Clinic, Main St 4")
    static let letter = AgendaReminder(
        id: "letter", title: "Write to the case worker", listID: "life", listTitle: "Life",
        due: local(30, 0))
    /// The letter's step, 10:00–10:30.
    static let letterStep = Step("letter", at: local(30, 10), minutes: 30)

    struct UnderWayRow: Sendable, CustomTestStringConvertible {
        let name: String
        let events: [AgendaEvent]
        let plan: [Step]
        var done = false
        /// The step the day is on at 10:10, by its row on the Day Line.
        let expected: (kind: UnderWay.Kind, row: String)?
        var testDescription: String { name }
    }

    static let underWayRows: [UnderWayRow] = [
        UnderWayRow(
            name: "a step the owner started, inside their block", events: [block],
            plan: [letterStep.started(at: local(30, 10))], expected: (.started, "task-letter")),
        UnderWayRow(
            name: "a step the owner started, over a meeting", events: [block, sync],
            plan: [letterStep.started(at: local(30, 10))], expected: (.started, "task-letter")),
        UnderWayRow(
            name: "a meeting inside a block", events: [block, sync], plan: [letterStep],
            expected: (.meeting, "event-sync")),
        UnderWayRow(
            name: "somewhere to be inside a block", events: [block, dentistVisit], plan: [],
            expected: (.meeting, "event-visit")),
        UnderWayRow(
            name: "a step not started, inside a block", events: [block], plan: [letterStep],
            expected: (.block, "event-work")),
        UnderWayRow(
            name: "a step not started, alone", events: [], plan: [letterStep],
            expected: (.notStarted, "task-letter")),
        UnderWayRow(
            name: "a step done early", events: [], plan: [letterStep.started(at: local(30, 10))],
            done: true, expected: nil),
        UnderWayRow(name: "nothing on", events: [], plan: [], expected: nil),
    ]

    @Test(arguments: underWayRows)
    func theStepTheDayIsOnIsOneRule(_ row: UnderWayRow) {
        var letter = Self.letter
        if row.done {
            letter.isCompleted = true
            letter.completedAt = Self.local(30, 10, 5)
        }
        let day = Timeline(
            snapshot: DayFixture.snapshot(
                now: Self.local(30, 10, 10), events: row.events, open: row.done ? [] : [letter],
                done: row.done ? [letter] : []),
            state: DayFixture.state(plan: row.plan))
        #expect(day.underWay?.kind == row.expected?.kind)
        #expect(day.underWay?.row.id == row.expected?.row)
    }

    /// A step the owner started inside a block of their own is the step the
    /// day is on everywhere: the Now Card drains its time left, the Day Line
    /// tints that one row, the menu bar counts it down, and the engine holds
    /// cues back for it. The Now Card led with the block, and the Day Line
    /// tinted both.
    @Test @MainActor func aStepStartedInsideABlockReadsTheSameEverywhere() {
        let day = Timeline(
            snapshot: DayFixture.snapshot(
                now: Self.local(30, 10, 10), events: [Self.block], open: [Self.letter],
                done: []),
            state: DayFixture.state(plan: [Self.letterStep.started(at: Self.local(30, 10))]))
        let card = NowCardBuilder.build(day, companionOn: true)
        #expect(card.headline == "Write to the case worker")
        #expect(card.detail == "20 min left, until 10:30")
        #expect(card.span == DateInterval(start: Self.local(30, 10), end: Self.local(30, 10, 30)))
        #expect(card.actions.map(\.kind) == [.complete(reminderID: "letter")])
        #expect(DaySteps.steps(day).filter(\.isUnderWay).map(\.id) == ["task-letter"])
        #expect(
            day.clock
                == MenuBarClock(
                    kind: .focus, title: "Write to the case worker", until: Self.local(30, 10, 30)
                ))
        #expect(day.focus?.reminderID == "letter")
    }

    /// A meeting under way inside a block is the step the day is on,
    /// whichever started first: the Now Card led with the block.
    @Test @MainActor func aMeetingInsideABlockIsTheStepTheDayIsOn() {
        let day = Timeline(
            snapshot: DayFixture.snapshot(
                now: Self.local(30, 10, 10), events: [Self.block, Self.sync], open: [], done: []),
            state: DayFixture.state())
        #expect(NowCardBuilder.build(day, companionOn: true).headline == "Sync")
        #expect(DaySteps.steps(day).filter(\.isUnderWay).map(\.id) == ["event-sync"])
        #expect(day.focus == nil)
    }

    /// A step the owner started is the step the day is on over a meeting it
    /// runs into, and over an earlier task, not started, whose slot holds
    /// now too. The Now Card led with the meeting, and else offered the
    /// earlier task's Start now.
    @Test @MainActor func aStartedStepLeadsTheNowCardOverAMeetingAndAnEarlierTask() {
        let started = Self.letterStep.started(at: Self.local(30, 10))
        let overMeeting = Timeline(
            snapshot: DayFixture.snapshot(
                now: Self.local(30, 10, 10), events: [Self.sync], open: [Self.letter], done: []),
            state: DayFixture.state(plan: [started]))
        #expect(
            NowCardBuilder.build(overMeeting, companionOn: true).headline
                == "Write to the case worker")
        let earlier = Step("rent", at: Self.local(30, 9, 45), minutes: 60)
        let overTask = Timeline(
            snapshot: DayFixture.snapshot(
                now: Self.local(30, 10, 10), events: [], open: [Self.letter, DayFixture.rent],
                done: []),
            state: DayFixture.state(plan: [earlier, started]))
        let card = NowCardBuilder.build(overTask, companionOn: true)
        #expect(card.headline == "Write to the case worker")
        #expect(card.span == DateInterval(start: Self.local(30, 10), end: Self.local(30, 10, 30)))
        #expect(DaySteps.steps(overTask).filter(\.isUnderWay).map(\.id) == ["task-letter"])
    }

    /// A leftover's Another Day: the day after tomorrow to a week on, from
    /// the owner's today — at 01:00 still the day that is ending.
    @Test func anotherDayIsTheDayAfterTomorrowToAWeekOn() {
        let days = [2, 3, 4, 5, 6, 7].map {
            Calendar.current.date(from: DateComponents(year: 2026, month: 10, day: $0))!
        }
        for now in [Self.local(30, 22), Self.local(31, 1)] {
            let facts = DayFacts(snapshot: DayFixture.snapshot(now: now), state: DayFixture.state())
            #expect(facts.anotherDays == days)
        }
    }

    // MARK: The Inbox

    @Test func theInboxHoldsUndatedCapturesUntilTheyArePlanned() {
        let chain = AgendaReminder(
            id: "chain", title: "Order a new bike chain", listID: "inbox", listTitle: "Inbox")
        let snapshot = DayFixture.snapshot(
            now: Self.local(30, 10, 7), open: [DayFixture.dentist, DayFixture.gym, chain])
        #expect(
            Timeline(snapshot: snapshot, state: DayFixture.state()).inbox.map(\.id) == ["chain"])
        let planned = DayFixture.state(plan: [Step("chain", at: Self.local(30, 15), minutes: 30)])
        #expect(Timeline(snapshot: snapshot, state: planned).inbox.isEmpty)
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
    static func twoDays(
        now: Date, tomorrow: [AgendaReminder] = [invoice, bins],
        done: [AgendaReminder] = [DayFixture.mail], mustDo: String? = nil
    ) -> Timeline {
        Timeline(
            snapshot: DayFixture.snapshot(
                now: now,
                events: [DayFixture.standup, DayFixture.oneOnOne, work, birthday],
                open: [DayFixture.dentist, DayFixture.rent, DayFixture.passport, DayFixture.gym]
                    + tomorrow, done: done),
            state: DayFixture.state(mustDo: mustDo))
    }

    @Test func tomorrowFollowsWithItsEventsAndTasks() {
        let tomorrow = Self.twoDays(now: Self.local(30, 10, 30)).tomorrow
        #expect(tomorrow.date == Self.local(31, 0))
        // Not planned yet: no free time and no Now line.
        #expect(Self.names(tomorrow.rows) == ["Put the bins out", "Work"])
        #expect(tomorrow.anytime.map(\.reminder.title) == ["Send the invoice"])
        #expect(tomorrow.allDayEvents.map(\.title) == ["Mom's birthday"])
    }

    @Test func aTaskDueTomorrowAndDoneEarlyStaysWithTomorrow() {
        var invoice = Self.invoice
        invoice.isCompleted = true
        invoice.completedAt = Self.local(30, 19)
        let day = Self.twoDays(
            now: Self.local(30, 20), tomorrow: [Self.bins], done: [DayFixture.mail, invoice])
        #expect(day.tomorrow.anytime.map(\.id) == ["invoice"])
        #expect(day.tomorrow.anytime.first?.isDone == true)
        // Today's count is today's: dentist, rent and the mail done this morning.
        #expect(!day.anytime.contains { $0.id == "invoice" })
        #expect(day.totalCount == 3)
        #expect(day.doneCount == 1)
    }

    /// The must-do is today's. Ticked off, a repeating one moves on to its
    /// next date (Reminders keeps the done one as a copy), and that next one,
    /// due tomorrow, doesn't wear today's star.
    @Test func tomorrowsTasksDontWearTodaysMustDoStar() {
        let invoiceDay = Self.twoDays(now: Self.local(30, 20), mustDo: Self.invoice.id)
        #expect(invoiceDay.tomorrow.anytime.map(\.id) == ["invoice"])
        #expect(invoiceDay.tomorrow.anytime.contains(where: \.isMustDo) == false)
        // Timed ones too.
        let rows = Self.twoDays(now: Self.local(30, 20), mustDo: Self.bins.id).tomorrow.rows
        #expect(
            rows.contains { row in
                if case .task(let task) = row.kind { task.isMustDo } else { false }
            } == false)
    }

    @Test func untilFourTheSmallHoursStillBelongToTheDayThatIsEnding() {
        let day = Self.twoDays(now: Self.local(31, 0, 40))
        #expect(day.facts.startOfToday == Self.local(30, 0))
        // The 30th's day, past its end: no free time left, the Now line last.
        #expect(Self.names(day.rows) == ["Standup", "Call the dentist", "1:1", "now"])
        #expect(day.tomorrow.date == Self.local(31, 0))
        #expect(day.tomorrow.rows.count == 2)
    }

    @Test func theSnapshotSplitsTasksIntoTodayTomorrowAndUndated() {
        let later = AgendaReminder(
            id: "later", title: "Book flights", listID: "life", listTitle: "Life",
            due: Self.local(32, 0))
        // At 00:40 the 31st is still tomorrow: the day rolls over at 04:00.
        let facts = DayFixture.snapshot(
            now: Self.local(31, 0, 40), events: [],
            open: [DayFixture.rent, Self.invoice, Self.bins, later, DayFixture.gym], done: []
        ).facts(state: DayFixture.state())
        #expect(facts.dueOrOverdue.map(\.id) == ["rent"])
        #expect(facts.dueTomorrow.map(\.id) == ["invoice", "bins"])
        #expect(facts.undated.map(\.id) == ["gym"])
    }
}

struct CardParserTests {

    static let facts = DayFixture.facts(now: DayFixture.local(30, 7, 30))

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
        #expect(card.departures.first?.at == DayFixture.local(30, 12, 30))
        #expect(card.departures.first?.eventStart == DayFixture.local(30, 13))
    }

    @Test func theWeeksLookBackKeepsAShortFocusOnlyOnItsDay() {
        let reply =
            #"{"line": "A good week.", "leftovers": [], "week": "Steady work and the streak held.", "focus": "Send three applications and keep the evenings for the Companion work, every single day of it"}"#
        guard
            case .card(.eveningWrapUp(let sunday)) = CardParser.eveningWrapUp(
                reply, facts: MomentPromptsTests.weekFacts(day: 4), leftovers: [])
        else {
            Issue.record("expected a wrap-up card")
            return
        }
        #expect(sunday.week == "Steady work and the streak held.")
        #expect(
            sunday.focus
                == "Send three applications and keep the evenings for the Companion work, every…")
        guard
            case .card(.eveningWrapUp(let saturday)) = CardParser.eveningWrapUp(
                reply, facts: MomentPromptsTests.weekFacts(day: 3), leftovers: [])
        else {
            Issue.record("expected a wrap-up card")
            return
        }
        #expect(saturday.week == nil)
        #expect(saturday.focus == nil)
    }

    @Test func aWrapUpSavedBeforeTheLookBackStillLoads() throws {
        let json = #"{"line": "Done.", "done": [], "leftovers": []}"#
        let card = try JSONDecoder().decode(EveningWrapUpCard.self, from: Data(json.utf8))
        #expect(card.week == nil)
        #expect(card.focus == nil)
    }

    @Test func aBrokenEntryIsDroppedNotTheWholePlan() {
        let reply =
            #"{"line": "Ok", "must_do": "rent", "plan": [{"id": "rent", "at": null}, {"id": "rent", "at": "08:15"}], "suggestions": ["a", 3], "leave": {}}"#
        guard case .card(.morningPlan(let card)) = CardParser.morningPlan(reply, facts: Self.facts)
        else {
            Issue.record("a stray entry must not cost the card")
            return
        }
        #expect(card.mustDoID == "rent")
        #expect(card.placements.map(\.reminderID) == ["rent"])
        #expect(card.suggestions == ["a"])
        #expect(card.departures.isEmpty)
    }

    @Test func theNightProposesTasksTheDayShowedOnlyOnce() {
        // "Pay rent" is already an open task: not proposed again.
        let reply =
            #"{"carry_over": "A good day.", "tasks": [{"title": "Send Anna the request", "when": "tomorrow"}, {"title": "pay rent"}, {"title": "Look into evening classes", "when": "later"}, {"title": "Send Anna the request"}, {"title": 7}]}"#
        guard
            case .card(.reflection(let card)) = CardParser.nightReflection(
                reply, facts: Self.facts)
        else {
            Issue.record("expected a Night Reflection card")
            return
        }
        #expect(card.tasks.map(\.title) == ["Send Anna the request", "Look into evening classes"])
        #expect(card.tasks.first?.due == Self.facts.endOfToday)
        #expect(card.tasks.last?.due == nil)
    }

    @Test func aReflectionSavedBeforeTasksStillLoads() throws {
        let json = #"{"carryOver": "Night.", "tomorrow": [], "proposals": []}"#
        let card = try JSONDecoder().decode(ReflectionCard.self, from: Data(json.utf8))
        #expect(card.tasks.isEmpty)
    }

    @Test func aCardSavedBeforeDeparturesStillLoads() throws {
        let json = #"{"line": "Hi", "placements": [], "suggestions": []}"#
        let card = try JSONDecoder().decode(MorningPlanCard.self, from: Data(json.utf8))
        #expect(card.departures.isEmpty)
    }
}
