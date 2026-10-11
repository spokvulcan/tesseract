//
//  TodayRedatingTests.swift
//  tesseractTests
//
//  A task re-dated from Today and its cards, by the runtime's clock. A
//  repeating reminder is one item for its whole series, so a new date or
//  time is every occurrence's: nothing meant for one day may change it by
//  accident. Tomorrow on Today is the Step Cue's: off today's plan and due
//  on the owner's tomorrow at its own time of day. Move to Today takes a
//  one-off task into today's Anytime tasks and never re-dates a repeating
//  one (done early, it is ticked off where it is). Later on a repeating
//  reminder that came due is a step in today's plan half an hour on, cued
//  then; its series stays as it was. Through Today's actions, the runtime,
//  the Agenda and the in-memory store, which ticks a repeating reminder off
//  as Reminders does (a daily series) but keeps no alarms: what EventKit
//  does to a series' alarm is read from the adapter, not tested (ADR-0073).
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct TodayRedatingTests {

    static func at(_ hour: Int, _ minute: Int = 0, day: Int) -> Date {
        Calendar.current.date(
            from: DateComponents(year: 2026, month: 10, day: day, hour: hour, minute: minute))!
    }

    /// Every day at 08:00; this morning's slid by.
    static let plants = AgendaReminder(
        id: "plants", title: "Water the plants", listID: "home", listTitle: "Home",
        due: at(8, day: 1), dueHasTime: true, repeats: true)
    /// A capture waiting in the Inbox, undated.
    static let plumber = AgendaReminder(
        id: "plumber", title: "Call the plumber", listID: "inbox", listTitle: "Inbox")
    /// Once, tomorrow at 10:00.
    static let dentist = AgendaReminder(
        id: "dentist", title: "Call the dentist", listID: "home", listTitle: "Home",
        due: at(10, day: 2), dueHasTime: true)
    /// Every day at 09:00; it came due while the owner was away.
    static let pills = AgendaReminder(
        id: "pills", title: "Take the pills", listID: "home", listTitle: "Home",
        due: at(9, day: 1), dueHasTime: true, repeats: true)
    /// Once, at 09:30; it came due too.
    static let bank = AgendaReminder(
        id: "bank", title: "Call the bank", listID: "home", listTitle: "Home",
        due: at(9, 30, day: 1), dueHasTime: true)

    /// Today's buttons over the app's wiring: the in-memory Agenda, and the
    /// runtime with the Companion off (Today works without it) on the
    /// test's clock, never the real one, from a saved day if one is given.
    @MainActor
    struct Today {
        let container: DependencyContainer
        let store: InMemoryAgendaStore
        let agenda: Agenda
        let runtime: CompanionRuntime
        let actions: TodayActions

        init(_ reminders: [AgendaReminder], now: Date, day: DayState? = nil) async {
            container = DependencyContainer()
            store = InMemoryAgendaStore(
                lists: [
                    AgendaList(id: "inbox", title: "Inbox", isDefault: true),
                    AgendaList(id: "home", title: "Home", isDefault: false),
                ],
                reminders: reminders, now: { now })
            agenda = Agenda(store: store, now: { now })
            await agenda.refresh()
            var stateStore = DayStateStore(url: nil)
            if let day {
                stateStore = DayStateStore(
                    url: StorageEnvironment.applicationSupport.appendingPathComponent(
                        "today-redating-\(UUID().uuidString).json"))
                stateStore.save(day)
            }
            runtime = CompanionRuntime(
                settings: container.settingsManager, agenda: agenda,
                notifier: container.companionNotifier, trace: container.companionTrace,
                idleMonitor: container.idleMonitor, presence: container.companionPresence,
                thread: container.dayThread, stateStore: stateStore,
                frontmost: container.frontmostApp, power: container.powerMonitor,
                delivery: CompanionDelivery(), now: { now })
            actions = TodayActions(agenda: agenda, runtime: runtime)
        }

        /// Waits for the Agenda to see `id` dated anew.
        func moved(_ id: String, from due: Date?) async {
            await observe(until: { [agenda] in
                agenda.snapshot.open.contains { $0.id == id && $0.due != due }
            })
        }
    }

    // MARK: Tomorrow

    /// The engine's half: off today's plan, and the cue's mutation.
    @Test func tomorrowTakesItOffTodaysPlanAsTheCueDoes() {
        var state = DayState(day: DayKey(rawValue: "2026-10-01"))
        state.syncedNudgeIDs = []
        state.morningPlanAt = Self.at(7, day: 1)
        state.plan = [Placement(reminderID: "plants", start: Self.at(8, day: 1), minutes: 15)]
        let decision = DayEngine.decide(
            .cardAction(.tomorrow(reminderID: "plants")),
            snapshot: Self.snapshot(at: Self.at(10, day: 1), open: [Self.plants]), state: state)
        #expect(!decision.state.plan.contains { $0.reminderID == "plants" })
        #expect(decision.effects.contains(.mutateAgenda(.dueTomorrow(reminderID: "plants"))))
    }

    @Test func aTimedRepeatingReminderKeepsItsTimeForEveryOccurrence() async throws {
        let today = await Today([Self.plants], now: Self.at(10, day: 1))
        today.actions.moveToTomorrow("plants")
        await today.moved("plants", from: Self.plants.due)
        let moved = try #require(today.store.reminders["plants"])
        #expect(moved.due == Self.at(8, day: 2))
        #expect(moved.dueHasTime)

        // Tomorrow's occurrence ticked off: the series goes on at 08:00.
        today.store.now = { Self.at(8, 5, day: 2) }
        try await today.agenda.updateReminder(id: "plants", completed: true, source: "test")
        let done = try #require(today.store.reminders.values.first { $0.isCompleted })
        #expect(done.due == Self.at(8, day: 2))
        #expect(done.dueHasTime)
        let series = try #require(today.store.reminders["plants"])
        #expect(series.due == Self.at(8, day: 3))
        #expect(series.dueHasTime)
    }

    /// The Inbox's Tomorrow: a capture without a time gets none.
    @Test func anUndatedTaskIsDueTomorrowWithoutATime() async throws {
        let today = await Today([Self.plumber], now: Self.at(10, day: 1))
        today.actions.moveToTomorrow("plumber")
        await today.moved("plumber", from: nil)
        let moved = try #require(today.store.reminders["plumber"])
        #expect(moved.due == Self.at(0, day: 2))
        #expect(!moved.dueHasTime)
    }

    /// Past midnight the owner's day is still the one ending, so tomorrow is
    /// today's date.
    @Test func pastMidnightTomorrowIsTodaysDate() async throws {
        let today = await Today([Self.plants], now: Self.at(1, 30, day: 2))
        today.actions.moveToTomorrow("plants")
        await today.moved("plants", from: Self.plants.due)
        let moved = try #require(today.store.reminders["plants"])
        #expect(moved.due == Self.at(8, day: 2))
        #expect(moved.dueHasTime)
    }

    // MARK: Move to Today

    /// The engine never re-dates a repeating reminder to today (the Day Line
    /// doesn't offer it); a one-off task is due today.
    @Test func moveToTodayLeavesARepeatingReminderAlone() {
        var state = DayState(day: DayKey(rawValue: "2026-10-01"))
        state.syncedNudgeIDs = []
        state.morningPlanAt = Self.at(7, day: 1)
        var tomorrows = Self.plants
        tomorrows.due = Self.at(8, day: 2)
        let snapshot = Self.snapshot(at: Self.at(10, day: 1), open: [tomorrows, Self.dentist])
        let repeating = DayEngine.decide(
            .cardAction(.today(reminderID: "plants")), snapshot: snapshot, state: state)
        #expect(Self.mutations(repeating.effects).isEmpty)
        let oneOff = DayEngine.decide(
            .cardAction(.today(reminderID: "dentist")), snapshot: snapshot, state: state)
        #expect(Self.mutations(oneOff.effects) == [.dueToday(reminderID: "dentist")])
    }

    /// A one-off task due tomorrow joins today's Anytime tasks: today, by
    /// the runtime's clock, at no set time.
    @Test func moveToTodayTakesAOneOffTaskIntoTodayWithoutATime() async throws {
        let today = await Today([Self.dentist], now: Self.at(10, day: 1))
        today.actions.moveToToday("dentist")
        await today.moved("dentist", from: Self.dentist.due)
        let moved = try #require(today.store.reminders["dentist"])
        #expect(moved.due == Self.at(0, day: 1))
        #expect(!moved.dueHasTime)
    }

    // MARK: Later

    /// Back at 10:05 from a morning away: the Breakpoint holds both
    /// reminders that came due.
    static func back() -> DayState {
        var state = DayState(day: DayKey(rawValue: "2026-10-01"))
        state.syncedNudgeIDs = []
        state.morningPlanAt = at(7, day: 1)
        state.lastPresentAt = at(10, 5, day: 1)
        state.lastTickAt = at(10, 5, day: 1)
        state.cards = [
            DayCard(
                id: "breakpoint", kind: .breakpoint, createdAt: at(10, 5, day: 1),
                isFallback: false,
                body: .breakpoint(
                    BreakpointCard(
                        awayFrom: at(8, 30, day: 1), awayUntil: at(10, 5, day: 1),
                        line: "Welcome back.",
                        needsYou: [pills, bank].map {
                            WaitingItem(
                                id: "reminder:\($0.id)", kind: .reminder, title: $0.title,
                                detail: "Came due", app: "Reminders")
                        },
                        next: [], canWait: [])))
        ]
        return state
    }

    static func snapshot(
        at now: Date, open: [AgendaReminder] = [pills, bank], events: [AgendaEvent] = []
    ) -> DaySnapshot {
        var agenda = AgendaSnapshot.empty
        agenda.access = .full
        agenda.open = open
        agenda.events = events
        return DaySnapshot(
            now: now, settings: DaySettings(), agenda: agenda, ownerPresent: true,
            frontmostAppName: "Safari", frontmostBundleID: "com.apple.Safari")
    }

    static func later(_ id: String, at now: Date, events: [AgendaEvent] = [])
        -> DayEngine.Decision
    {
        DayEngine.decide(
            .cardAction(.itemLater(cardID: "breakpoint", itemID: "reminder:\(id)")),
            snapshot: snapshot(at: now, events: events), state: back())
    }

    static func mutations(_ effects: [DayEffect]) -> [AgendaMutation] {
        effects.compactMap { if case .mutateAgenda(let change) = $0 { change } else { nil } }
    }

    /// Its time is its whole series', so "Remind me in half an hour" is a
    /// step in today's plan, cued when it starts.
    @Test func laterOnARepeatingReminderIsAStepHalfAnHourOn() throws {
        let later = Self.later("pills", at: Self.at(10, 7, day: 1))
        #expect(Self.mutations(later.effects).isEmpty)
        #expect(
            later.state.plan == [
                Placement(reminderID: "pills", start: Self.at(10, 37, day: 1), minutes: 15)
            ])
        let cued = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.at(10, 37, day: 1)), state: later.state)
        let cue = try #require(DayEngineStepCueTests.cues(cued.effects).first)
        #expect(cue.reminderID == "pills")
        #expect(cue.start == Self.at(10, 37, day: 1))
    }

    /// A step can't be cued in a meeting: it starts when the meeting is over.
    @Test func laterIntoAMeetingStartsWhenItsOver() {
        let review = AgendaEvent(
            id: "review", title: "Design review", start: Self.at(10, 30, day: 1),
            end: Self.at(11, 15, day: 1), calendarID: "work", calendarTitle: "Work",
            hasOtherAttendees: true)
        let later = Self.later("pills", at: Self.at(10, 7, day: 1), events: [review])
        #expect(Self.mutations(later.effects).isEmpty)
        #expect(
            later.state.plan == [
                Placement(reminderID: "pills", start: Self.at(11, 15, day: 1), minutes: 15)
            ])
    }

    /// A one-off reminder is still moved half an hour on: Reminders rings it
    /// again, on every device.
    @Test func laterOnAOneOffReminderStillMovesItHalfAnHourOn() {
        let later = Self.later("bank", at: Self.at(10, 7, day: 1))
        #expect(
            Self.mutations(later.effects) == [
                .dueAt(reminderID: "bank", at: Self.at(10, 37, day: 1))
            ])
        #expect(later.state.plan.isEmpty)
    }

    /// Through the store: the series keeps its 09:00, today's and every
    /// day's after.
    @Test func laterLeavesARepeatingSeriesAtItsTime() async throws {
        let today = await Today(
            [Self.pills, Self.bank], now: Self.at(10, 7, day: 1), day: Self.back())
        today.runtime.act(.itemLater(cardID: "breakpoint", itemID: "reminder:pills"))
        await observe(until: { [runtime = today.runtime, agenda = today.agenda] in
            runtime.state.plan.contains { $0.reminderID == "pills" }
                || agenda.snapshot.open.contains { $0.id == "pills" && $0.due != Self.pills.due }
        })
        let kept = try #require(today.store.reminders["pills"])
        #expect(kept.due == Self.at(9, day: 1))
        #expect(kept.dueHasTime)
        #expect(
            today.runtime.state.plan == [
                Placement(reminderID: "pills", start: Self.at(10, 37, day: 1), minutes: 15)
            ])

        // Taken at its cue: the series goes on tomorrow at 09:00.
        today.store.now = { Self.at(10, 40, day: 1) }
        try await today.agenda.updateReminder(id: "pills", completed: true, source: "test")
        let series = try #require(today.store.reminders["pills"])
        #expect(series.due == Self.at(9, day: 2))
        #expect(series.dueHasTime)
    }
}
