//
//  DayEngineRepeatingTaskTests.swift
//  tesseractTests
//
//  A repeating task ticked off (ADR-0092). Reminders keeps the done
//  occurrence as a copy with an id of its own and moves the series, under
//  its own id, on to its next date. The day keeps the series' id, and the
//  Agenda answers for that day's occurrence: the must-do counts as done, the
//  step is done in its slot, the minutes it took join the plan's pace,
//  Today's Chat shows it done, and the series' next date stays tomorrow's.
//  Taken back — by Undo, by unticking the task or its copy — Reminders is as
//  it was: the series back on the occurrence, the copy gone, one open
//  reminder, an occurrence overdue and ticked off late too. Only what the
//  Agenda ticked off itself is taken back so: a one-off of the series'
//  name, a copy ticked off elsewhere, and a series moved since are only
//  ever opened again. A task moved on without being done, one that doesn't
//  repeat, and an occurrence done another day are left as they are.
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
        var step = Step(slot).started(at: at(10))
        step.putOff = 1
        state.plan = [step]
        return state
    }

    @Test func theDayKeepsTheSeriesAndCountsTodaysOccurrenceDone() {
        let state = DayEngine.decide(
            .agendaChanged, snapshot: Self.snapshot(at: Self.at(11, 16)), state: Self.state()
        ).state
        // The series' id, its step and its life, as they were.
        #expect(state.mustDoID == "daily")
        #expect(state.plan.map(\.slot) == [Self.slot])
        #expect(state.step("daily")?.isStarted == true)
        #expect(state.step("daily")?.putOff == 1)
        // Its occurrence done at 11:15: the must-do counts as done.
        #expect(state.mustDoDoneAt == Self.at(11, 16))
        // Started at 10:00 with 90 minutes, done at 11:15: the pace hears it.
        #expect(state.step("daily")?.timedMinutes == nil)
        #expect(state.stepRuns.map(\.planned) == [90])
        #expect(state.stepRuns.map(\.actual) == [75])
        // The week's look-back counts the day's must-do done.
        #expect(
            state.rolledOver(to: DayKey(rawValue: "2026-10-01")).mustDoDays == [
                "2026-09-30": true
            ])
    }

    /// On Today: done where it was planned, wearing the star, shown once,
    /// and the evening counts it; tomorrow's occurrence waits under the
    /// series' id.
    @Test @MainActor func todayShowsTheMustDoDoneInItsSlot() {
        let evening = Self.at(21, 30)
        let state = DayEngine.decide(
            .agendaChanged, snapshot: Self.snapshot(at: evening), state: Self.state()
        ).state
        let timeline = Timeline(snapshot: Self.snapshot(at: evening), state: state)
        let steps = timeline.rows.compactMap { row -> TimelineTask? in
            if case .task(let task) = row.kind { task } else { nil }
        }
        #expect(steps.map(\.id) == ["daily"])
        #expect(steps.first?.reminder.id == "daily-0930")
        #expect(steps.first?.start == Self.at(10))
        #expect(steps.first?.isDone == true)
        #expect(steps.first?.isMustDo == true)
        #expect(!timeline.anytime.contains { $0.reminder.id == "daily-0930" })
        #expect(timeline.mustDo?.isDone == true)
        #expect(timeline.tomorrow.anytime.map(\.id) == ["daily"])
        #expect(timeline.tomorrow.anytime.first?.isMustDo == false)
        let card = NowCardBuilder.build(timeline, companionOn: true)
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

    /// Moved on by hand, a repeating task has no done occurrence.
    @Test func aRepeatingTaskMovedOnUndoneIsLeftAsItIs() {
        let state = DayEngine.decide(
            .agendaChanged, snapshot: Self.snapshot(at: Self.at(11, 16), done: []),
            state: Self.state()
        ).state
        #expect(state.mustDoID == "daily")
        #expect(state.plan.map(\.slot) == [Self.slot])
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
        #expect(state.mustDoDoneAt == nil)
        // The evening's own occurrence, done at 20:05, is its occurrence.
        var done = morning
        done.due = Self.at(20)
        done.completedAt = Self.at(20, 5)
        let counted = DayEngine.decide(
            .agendaChanged,
            snapshot: Self.snapshot(at: Self.at(20, 6), open: [evening], done: [done]),
            state: Self.state()
        ).state
        #expect(counted.mustDoID == "daily")
        #expect(counted.mustDoDoneAt == Self.at(20, 6))
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
        #expect(state.mustDoDoneAt == nil)
    }

    // MARK: Through the Agenda

    /// The daily step's series in the in-memory store, open on `due`
    /// (today's occurrence), beside any `others`.
    @MainActor
    struct Store {
        let store: InMemoryAgendaStore
        let agenda: Agenda

        init(
            at now: Date = DayEngineRepeatingTaskTests.at(11, 15),
            due: Date = DayEngineRepeatingTaskTests.at(0), others: [AgendaReminder] = []
        ) async {
            var series = DayEngineRepeatingTaskTests.series
            series.due = due
            store = InMemoryAgendaStore(
                lists: [AgendaList(id: "daily", title: "Daily", isDefault: true)],
                reminders: [series] + others, now: { now })
            agenda = Agenda(store: store, now: { now })
            await agenda.refresh()
        }

        /// Reminders as the owner would see them: every reminder, done or open.
        var reminders: [AgendaReminder] {
            store.reminders.values.sorted { $0.id < $1.id }
        }
    }

    /// The in-memory store ticks a repeating reminder off as Reminders does,
    /// and the day sees its occurrence done.
    @Test @MainActor func theStoreKeepsTheDoneOccurrenceAndMovesTheSeriesOn() async throws {
        let reminders = await Store()
        try await reminders.agenda.updateReminder(id: "daily", completed: true, source: "test")
        await reminders.agenda.refresh()
        let snapshot = reminders.agenda.snapshot
        #expect(snapshot.open.map(\.id) == ["daily"])
        #expect(snapshot.open.first?.due == Self.at(0, day: 31))
        let done = try #require(snapshot.doneToday.first)
        #expect(done.id != "daily")
        #expect(done.title == "Implement the Companion")
        #expect(done.due == Self.at(0))
        #expect(done.completedAt == Self.at(11, 15))
        #expect(!done.repeats)
        #expect(snapshot.doneOccurrence(of: "daily", on: DayKey(rawValue: "2026-09-30")) == done)

        let state = DayEngine.decide(
            .agendaChanged,
            snapshot: DaySnapshot(
                now: Self.at(11, 16), settings: DaySettings(), agenda: snapshot,
                ownerPresent: false),
            state: Self.state()
        ).state
        #expect(state.mustDoID == "daily")
        #expect(state.mustDoDoneAt == Self.at(11, 16))
    }

    /// Today's Chat's Morning Plan shows each task as today sees it: a
    /// repeating one ticked off is today's occurrence, done — not its series
    /// moved on to tomorrow, open, whose tick would have ticked off
    /// tomorrow's.
    @Test @MainActor func theMorningPlanShowsTodaysOccurrenceDone() async throws {
        let reminders = await Store()
        let today = DayKey(rawValue: "2026-09-30")
        #expect(reminders.agenda.snapshot.occurrence(of: "daily", on: today)?.isCompleted == false)
        try await reminders.agenda.apply(.setDone(reminderID: "daily", done: true, onToday: true))
        await reminders.agenda.refresh()
        let shown = try #require(reminders.agenda.snapshot.occurrence(of: "daily", on: today))
        #expect(shown.isCompleted)
        #expect(shown.due == Self.at(0))
        #expect(shown.completedAt == Self.at(11, 15))
    }

    /// Ticked off and taken back, Reminders is as it was: the series on the
    /// occurrence's date, no copy, one reminder — whether by the change's
    /// Undo, by unticking the task (a cue's Undo, Today's ring), or by
    /// unticking the copy; and for an occurrence overdue from yesterday,
    /// ticked off late, whose series moves on only to today. Reopening the
    /// copy made two open reminders, and the Undo reopened the series,
    /// already open, leaving the copy done.
    @Test(arguments: ["undo", "the task", "the copy"], [false, true])
    @MainActor func takenBackTheOccurrenceLeavesRemindersAsItWas(_ how: String, overdue: Bool)
        async throws
    {
        let reminders = await Store(due: Self.at(0, day: overdue ? 29 : 30))
        let before = reminders.reminders
        let (_, done) = try await reminders.agenda.updateReminder(
            id: "daily", completed: true, source: "test")
        #expect(reminders.reminders.count == 2)
        #expect(reminders.store.reminders["daily"]?.due == Self.at(0, day: overdue ? 30 : 31))
        switch how {
        case "undo":
            try await reminders.agenda.undo(done)
        case "the task":
            try await reminders.agenda.apply(.setDone(reminderID: "daily", done: false))
        default:
            let copy = try #require(reminders.reminders.first { $0.id != "daily" })
            try await reminders.agenda.apply(
                .setDone(reminderID: copy.id, done: false, onToday: true))
        }
        #expect(reminders.reminders == before)
        await reminders.agenda.refresh()
        #expect(
            reminders.agenda.snapshot.doneOccurrence(
                of: "daily", on: DayKey(rawValue: "2026-09-30")) == nil)
    }

    /// The day sees its occurrence open again: the must-do isn't done.
    @Test @MainActor func aMustDoTakenBackIsNotDoneAfterAll() async throws {
        let reminders = await Store()
        let (_, done) = try await reminders.agenda.updateReminder(
            id: "daily", completed: true, source: "test")
        await reminders.agenda.refresh()
        func decide(_ state: DayState) -> DayState {
            DayEngine.decide(
                .agendaChanged,
                snapshot: DaySnapshot(
                    now: Self.at(11, 16), settings: DaySettings(),
                    agenda: reminders.agenda.snapshot, ownerPresent: false),
                state: state
            ).state
        }
        let ticked = decide(Self.state())
        #expect(ticked.mustDoDoneAt != nil)
        try await reminders.agenda.undo(done)
        await reminders.agenda.refresh()
        #expect(decide(ticked).mustDoDoneAt == nil)
    }

    /// A one-off of the series' name in its list — today's call beside the
    /// weekly one, next due in a week — looks like a copy of the series.
    /// Ticked off and unticked, it is only ever opened again: never
    /// deleted, the series never moved, and the untick's Undo ticks the
    /// one-off off again, not the series.
    @Test @MainActor func aOneOffOfTheSeriesNameIsOnlyEverOpenedAgain() async throws {
        let call = AgendaReminder(
            id: "call", title: "Implement the Companion", listID: "daily", listTitle: "Daily",
            due: Self.at(0))
        let nextWeek = try #require(Calendar.current.date(byAdding: .day, value: 7, to: Self.at(0)))
        let reminders = await Store(due: nextWeek, others: [call])
        let before = reminders.reminders
        try await reminders.agenda.apply(.setDone(reminderID: "call", done: true, onToday: true))
        #expect(reminders.store.reminders["call"]?.isCompleted == true)
        let (_, opened) = try await reminders.agenda.updateReminder(
            id: "call", completed: false, source: "test")
        #expect(reminders.reminders == before)
        try await reminders.agenda.undo(opened)
        #expect(reminders.store.reminders["call"]?.isCompleted == true)
        #expect(reminders.store.reminders["daily"] == before.first { $0.id == "daily" })
    }

    /// A copy the Agenda didn't make — the occurrence ticked off on the
    /// phone — is never deleted. Unticked, it opens again beside its series:
    /// two open reminders, the risk ADR-0092 takes rather than delete what
    /// it can't be sure of. Unticking the series changes nothing.
    @Test @MainActor func aCopyTickedOffElsewhereIsOnlyOpenedAgain() async throws {
        let reminders = await Store(due: Self.at(0, day: 31), others: [Self.occurrence])
        try await reminders.agenda.apply(.setDone(reminderID: "daily", done: false))
        #expect(reminders.store.reminders["daily"]?.due == Self.at(0, day: 31))
        #expect(reminders.store.reminders["daily-0930"]?.isCompleted == true)
        try await reminders.agenda.apply(
            .setDone(reminderID: "daily-0930", done: false, onToday: true))
        #expect(reminders.store.reminders["daily-0930"]?.isCompleted == false)
        #expect(reminders.store.reminders["daily"]?.due == Self.at(0, day: 31))
        #expect(reminders.reminders.count == 2)
    }

    /// The series moved since it was ticked off (by the owner, here or on
    /// the phone): it isn't where the tick left it, so unticking the copy
    /// only opens the copy again and moves nothing back.
    @Test @MainActor func anOccurrenceChangedSinceItsTickIsOnlyOpenedAgain() async throws {
        let reminders = await Store()
        try await reminders.agenda.apply(.setDone(reminderID: "daily", done: true, onToday: true))
        let copy = try #require(reminders.reminders.first { $0.id != "daily" })
        let later = Self.at(9, day: 31)
        _ = try reminders.store.updateReminder(
            id: "daily", ReminderChange(due: .set(later, hasTime: true)))
        try await reminders.agenda.apply(.setDone(reminderID: copy.id, done: false, onToday: true))
        #expect(reminders.store.reminders[copy.id]?.isCompleted == false)
        #expect(reminders.store.reminders["daily"]?.due == later)
    }
}
