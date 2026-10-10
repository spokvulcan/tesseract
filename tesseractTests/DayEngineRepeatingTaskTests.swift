//
//  DayEngineRepeatingTaskTests.swift
//  tesseractTests
//
//  A repeating task ticked off. Reminders keeps the done occurrence as a
//  copy with an id of its own and moves the series, under the old id, on to
//  its next date. The day follows the done occurrence: the must-do counts as
//  done, the step stays done where it was planned, the minutes it took join
//  the plan's pace, and the series' next date stays tomorrow's. A task moved
//  on without being done, one that doesn't repeat, and an occurrence done
//  another day are left as they are.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct DayEngineRepeatingTaskTests {

    static func at(_ hour: Int, _ minute: Int = 0, day: Int = 30) -> Date {
        Calendar.current.date(
            from: DateComponents(year: 2026, month: 9, day: day, hour: hour, minute: minute))!
    }

    /// The daily step's series, moved on to tomorrow once ticked off today.
    static let series = AgendaReminder(
        id: "daily", title: "Implement the Companion", listID: "daily", listTitle: "Daily",
        due: at(0, day: 31), repeats: true)
    /// Today's occurrence, done at 11:15: a copy with an id of its own.
    static let occurrence = AgendaReminder(
        id: "daily-0930", title: "Implement the Companion", listID: "daily",
        listTitle: "Daily", due: at(0), isCompleted: true, completedAt: at(11, 15))
    static let rent = AgendaReminder(
        id: "rent", title: "Pay rent", listID: "life", listTitle: "Life", due: at(0))
    static let slot = Placement(reminderID: "daily", start: at(10), minutes: 90)

    static func snapshot(
        at now: Date, open: [AgendaReminder] = [series, rent],
        done: [AgendaReminder] = [occurrence], doneEarlier: [AgendaReminder] = []
    ) -> DaySnapshot {
        var agenda = AgendaSnapshot.empty
        agenda.access = .full
        agenda.open = open
        agenda.doneToday = done
        agenda.doneThisWeek = doneEarlier + done
        return DaySnapshot(
            now: now, settings: DaySettings(), agenda: agenda,
            areas: [Area(id: "daily", name: "Daily"), Area(id: "life", name: "Life")],
            ownerPresent: false)
    }

    /// The must-do, planned 10:00–11:30, started on its cue, put off once.
    static func state() -> DayState {
        var state = DayState(day: DayKey(rawValue: "2026-09-30"))
        state.syncedNudgeIDs = []
        state.morningPlanAt = at(8)
        state.mustDoID = "daily"
        state.plan = [slot]
        let key = StepCue.key(slot)
        state.cuedSteps[key] = at(10)
        state.startedSteps = [key]
        state.startedMinutes[key] = 90
        state.putOff["daily"] = 1
        return state
    }

    @Test func theDayFollowsTheDoneOccurrence() {
        let state = DayEngine.decide(
            .agendaChanged, snapshot: Self.snapshot(at: Self.at(11, 16)), state: Self.state()
        ).state
        let followed = Placement(reminderID: "daily-0930", start: Self.at(10), minutes: 90)
        #expect(state.mustDoID == "daily-0930")
        #expect(state.mustDoDoneAt == Self.at(11, 16))
        #expect(state.plan == [followed])
        #expect(state.startedSteps == [StepCue.key(followed)])
        #expect(state.cuedSteps == [StepCue.key(followed): Self.at(10)])
        #expect(state.putOff == ["daily-0930": 1])
        // Started at 10:00 with 90 minutes, done at 11:15: the pace hears it.
        #expect(state.startedMinutes.isEmpty)
        #expect(state.stepRuns.map(\.planned) == [90])
        #expect(state.stepRuns.map(\.actual) == [75])
        // The week's look-back counts the day's must-do done.
        #expect(
            state.rolledOver(to: DayKey(rawValue: "2026-10-01")).mustDoDays == [
                "2026-09-30": true
            ])
    }

    /// On Today: done where it was planned, wearing the star, and the evening
    /// counts it; tomorrow's occurrence waits under the series' id.
    @Test func todayShowsTheMustDoDoneInItsSlot() {
        let evening = Self.at(21, 30)
        let state = DayEngine.decide(
            .agendaChanged, snapshot: Self.snapshot(at: evening), state: Self.state()
        ).state
        let facts = Self.snapshot(at: evening).facts(state: state)
        let timeline = TimelineBuilder.build(facts: facts)
        let steps = timeline.rows.compactMap { row -> TimelineTask? in
            if case .task(let task) = row.kind { task } else { nil }
        }
        #expect(steps.map(\.id) == ["daily-0930"])
        #expect(steps.first?.start == Self.at(10))
        #expect(steps.first?.isDone == true)
        #expect(steps.first?.isMustDo == true)
        #expect(timeline.mustDo?.isDone == true)
        #expect(timeline.tomorrow.anytime.map(\.id) == ["daily"])
        #expect(timeline.tomorrow.anytime.first?.isMustDo == false)
        let card = NowCardBuilder.build(
            timeline: timeline, facts: facts,
            context: NowCardBuilder.Context(
                companionOn: true, planned: true, wrappedUp: false, eveningMinutes: 21 * 60))
        #expect(card.headline == "1 of 2 done today, the must-do among them.")
    }

    /// Ticked off on the phone at 23:00 while the Mac slept through 04:00:
    /// the must-do counts for the day it was set.
    @Test func aRepeatingMustDoDoneWhileTheMacSleptCountsForItsDay() {
        var occurrence = Self.occurrence
        occurrence.completedAt = Self.at(23)
        let morning = DayEngine.decide(
            .tick,
            snapshot: Self.snapshot(at: Self.at(8, day: 31), done: [], doneEarlier: [occurrence]),
            state: Self.state())
        #expect(morning.state.day == DayKey(rawValue: "2026-10-01"))
        #expect(morning.state.mustDoDays == ["2026-09-30": true])
    }

    /// Moved on by hand, a repeating task has no done occurrence to follow.
    @Test func aRepeatingTaskMovedOnUndoneIsLeftAsItIs() {
        let state = DayEngine.decide(
            .agendaChanged, snapshot: Self.snapshot(at: Self.at(11, 16), done: []),
            state: Self.state()
        ).state
        #expect(state.mustDoID == "daily")
        #expect(state.plan == [Self.slot])
        #expect(state.mustDoDoneAt == nil)
    }

    /// A task that doesn't repeat, moved to tomorrow, and another of the same
    /// name done today: not an occurrence of it.
    @Test func aTaskThatDoesntRepeatIsNeverTakenForAnotherOfTheSameName() {
        var oneOff = Self.series
        oneOff.repeats = false
        let state = DayEngine.decide(
            .agendaChanged, snapshot: Self.snapshot(at: Self.at(11, 16), open: [oneOff]),
            state: Self.state()
        ).state
        #expect(state.mustDoID == "daily")
        #expect(state.mustDoDoneAt == nil)
    }

    /// Two daily reminders of one name, morning and evening: the morning's
    /// done occurrence isn't the evening's, moved on by hand.
    @Test func theTimeOfDayTellsTwoHabitsOfOneNameApart() {
        var evening = Self.series
        evening.title = "Take pills"
        evening.due = Self.at(20, day: 31)
        evening.dueHasTime = true
        var morning = Self.occurrence
        morning.title = "Take pills"
        morning.due = Self.at(8)
        morning.dueHasTime = true
        let state = DayEngine.decide(
            .agendaChanged,
            snapshot: Self.snapshot(at: Self.at(11, 16), open: [evening], done: [morning]),
            state: Self.state()
        ).state
        #expect(state.mustDoID == "daily")
        #expect(state.mustDoDoneAt == nil)
        // The evening's own occurrence, done at 20:00, is followed.
        var done = morning
        done.due = Self.at(20)
        done.completedAt = Self.at(20, 5)
        let followed = DayEngine.decide(
            .agendaChanged,
            snapshot: Self.snapshot(at: Self.at(20, 6), open: [evening], done: [done]),
            state: Self.state()
        ).state
        #expect(followed.mustDoID == "daily-0930")
    }

    /// Yesterday's done occurrence isn't today's.
    @Test func anOccurrenceDoneAnotherDayIsntTodays() {
        var yesterdays = Self.occurrence
        yesterdays.completedAt = Self.at(18, day: 29)
        let state = DayEngine.decide(
            .agendaChanged,
            snapshot: Self.snapshot(at: Self.at(11, 16), done: [], doneEarlier: [yesterdays]),
            state: Self.state()
        ).state
        #expect(state.mustDoID == "daily")
        #expect(state.mustDoDoneAt == nil)
    }

    /// The in-memory store ticks a repeating reminder off as Reminders does,
    /// and the day follows what it keeps.
    @Test @MainActor func theStoreKeepsTheDoneOccurrenceAndMovesTheSeriesOn() async throws {
        var today = Self.series
        today.due = Self.at(0)
        let store = InMemoryAgendaStore(
            lists: [AgendaList(id: "daily", title: "Daily", isDefault: true)],
            reminders: [today], now: { Self.at(11, 15) })
        let agenda = Agenda(store: store, now: { Self.at(11, 15) })
        try await agenda.updateReminder(id: "daily", completed: true, source: "test")
        await agenda.refresh()
        #expect(agenda.snapshot.open.map(\.id) == ["daily"])
        #expect(agenda.snapshot.open.first?.due == Self.at(0, day: 31))
        let done = try #require(agenda.snapshot.doneToday.first)
        #expect(done.id != "daily")
        #expect(done.title == "Implement the Companion")
        #expect(done.due == Self.at(0))
        #expect(done.completedAt == Self.at(11, 15))
        #expect(!done.repeats)

        let snapshot = DaySnapshot(
            now: Self.at(11, 16), settings: DaySettings(), agenda: agenda.snapshot,
            ownerPresent: false)
        let state = DayEngine.decide(.agendaChanged, snapshot: snapshot, state: Self.state()).state
        #expect(state.mustDoID == done.id)
        #expect(state.mustDoDoneAt == Self.at(11, 16))
    }
}
