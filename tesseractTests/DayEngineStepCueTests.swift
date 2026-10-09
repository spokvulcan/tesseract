//
//  DayEngineStepCueTests.swift
//  tesseractTests
//
//  The plan keeps its own time, as decision tables: a planned step is put on
//  the Jarvis Panel when its slot starts and the owner is at the Mac — once,
//  never while away, in quiet hours, a call, a game or a meeting, never over
//  a panel that is up, and not when its reminder rings at the same minute —
//  and each of the owner's four choices (and closing it) does what it says.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct DayEngineStepCueTests {

    static func local(_ hour: Int, _ minute: Int = 0, day: Int = 30) -> Date {
        Calendar.current.date(
            from: DateComponents(year: 2026, month: 9, day: day, hour: hour, minute: minute))!
    }

    static let letter = AgendaReminder(
        id: "letter", title: "Write to the case worker", listID: "inbox", listTitle: "Reminders",
        due: local(0))
    static let deck = AgendaReminder(
        id: "deck", title: "Reply to Anna about the deck", listID: "work", listTitle: "Work")
    static let dentist = AgendaReminder(
        id: "dentist", title: "Call the dentist", listID: "health", listTitle: "Health",
        due: local(16, 30), dueHasTime: true)

    static func snapshot(
        at now: Date, present: Bool = true, frontmost: String? = "com.apple.Safari",
        game: Bool = false, panelUp: Bool = false, open: [AgendaReminder]? = nil,
        done: [AgendaReminder] = [], events: [AgendaEvent] = [], quiet: (Int, Int)? = nil
    ) -> DaySnapshot {
        var agenda = AgendaSnapshot.empty
        agenda.access = .full
        agenda.open = open ?? [letter, deck, dentist]
        agenda.doneToday = done
        agenda.events = events
        var settings = DaySettings()
        if let quiet {
            settings.quietStartMinutes = quiet.0
            settings.quietEndMinutes = quiet.1
        }
        return DaySnapshot(
            now: now, settings: settings, agenda: agenda,
            areas: [Area(id: "work", name: "Work"), Area(id: "health", name: "Health")],
            inboxListID: "inbox", ownerPresent: present, frontmostAppName: "Safari",
            frontmostBundleID: frontmost, frontmostIsGame: game, panelUp: panelUp)
    }

    /// The morning's plan: the letter at 11:10, the deck at 11:35, the
    /// dentist at the time its own reminder rings.
    static func state() -> DayState {
        var state = DayState(day: DayKey(rawValue: "2026-09-30"))
        state.syncedNudgeIDs = []
        state.morningPlanAt = local(8)
        state.lastPresentAt = local(11, 9)
        state.lastTickAt = local(11, 9)
        state.plan = [
            Placement(reminderID: "letter", start: local(11, 10), minutes: 20),
            Placement(reminderID: "deck", start: local(11, 35), minutes: 50),
            Placement(reminderID: "dentist", start: local(16, 30), minutes: 15),
        ]
        return state
    }

    static func cues(_ effects: [DayEffect]) -> [StepCue] {
        effects.compactMap { if case .presentStep(let cue) = $0 { cue } else { nil } }
    }

    static func traced(_ event: CompanionTraceEvent, in effects: [DayEffect])
        -> [String: CompanionTraceValue]?
    {
        for effect in effects {
            if case .trace(event, let fields) = effect { return fields }
        }
        return nil
    }

    // MARK: The cue

    @Test func aPlannedStepIsCuedWhenItsSlotStarts() throws {
        let decision = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(11, 10)), state: Self.state())
        let cue = try #require(Self.cues(decision.effects).first)
        #expect(cue.reminderID == "letter")
        #expect(cue.title == "Write to the case worker")
        #expect(cue.start == Self.local(11, 10))
        #expect(cue.end == Self.local(11, 30))
        #expect(cue.areaName == "Inbox")
        #expect(cue.next == "Reply to Anna about the deck at 11:35")
        #expect(!cue.isMustDo)
        let fields = try #require(Self.traced(.cuePresented, in: decision.effects))
        #expect(fields["late"] == .int(0))
        #expect(fields["minutes"] == .int(20))
        #expect(decision.state.cuedSteps.count == 1)
    }

    @Test func theMustDoSaysSo() throws {
        var state = Self.state()
        state.mustDoID = "letter"
        let decision = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(11, 10)), state: state)
        #expect(try #require(Self.cues(decision.effects).first).isMustDo)
    }

    @Test func eachSlotIsCuedOnce() {
        let first = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(11, 10)), state: Self.state())
        let second = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(11, 11)), state: first.state)
        #expect(Self.cues(second.effects).isEmpty)
    }

    struct QuietRow: Sendable, CustomTestStringConvertible {
        let name: String
        let snapshot: DaySnapshot
        var testDescription: String { name }
    }

    static let quietRows: [QuietRow] = [
        QuietRow(name: "before the slot starts", snapshot: snapshot(at: local(11, 9))),
        QuietRow(name: "the owner is away", snapshot: snapshot(at: local(11, 10), present: false)),
        QuietRow(
            name: "quiet hours", snapshot: snapshot(at: local(11, 10), quiet: (11 * 60, 12 * 60))),
        QuietRow(name: "a game in front", snapshot: snapshot(at: local(11, 10), game: true)),
        QuietRow(
            name: "a call in front", snapshot: snapshot(at: local(11, 10), frontmost: "us.zoom.xos")
        ),
        QuietRow(
            name: "the panel is up with something else",
            snapshot: snapshot(at: local(11, 10), panelUp: true)),
        QuietRow(
            name: "a meeting is under way",
            snapshot: snapshot(
                at: local(11, 10),
                events: [
                    AgendaEvent(
                        id: "sync", title: "Sync", start: local(11), end: local(11, 30),
                        calendarID: "c", calendarTitle: "Work", hasOtherAttendees: true)
                ])),
        QuietRow(
            name: "the task is already done",
            snapshot: snapshot(at: local(11, 10), open: [deck, dentist], done: [letter])),
        QuietRow(
            name: "the slot started too long ago (the Mac slept)",
            snapshot: snapshot(at: local(11, 21))),
    ]

    @Test(arguments: quietRows)
    func noCue(_ row: QuietRow) {
        let decision = DayEngine.decide(.tick, snapshot: row.snapshot, state: Self.state())
        #expect(Self.cues(decision.effects).isEmpty)
    }

    @Test func aStepWaitsForAPanelThatIsUpAndIsCuedOnceItCloses() throws {
        let busy = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(11, 10), panelUp: true),
            state: Self.state())
        #expect(Self.cues(busy.effects).isEmpty)
        let free = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(11, 13)), state: busy.state)
        let cue = try #require(Self.cues(free.effects).first)
        #expect(cue.reminderID == "letter")
        #expect(Self.traced(.cuePresented, in: free.effects)?["late"] == .int(180))
    }

    @Test func aCardThatTakesThePanelOnTheSameTickGoesFirst() {
        // The standup ends at 11:10 with a coding agent waiting: its
        // Breakpoint card takes the panel; the letter waits a tick.
        var state = Self.state()
        state.agents = [
            AgentSignal(
                id: "s1", kind: .waiting, agent: "Claude Code", project: "tesseract",
                directory: "/tmp/tesseract", message: "Needs approval", at: Self.local(10, 50))
        ]
        let standup = AgendaEvent(
            id: "standup", title: "Standup", start: Self.local(10, 45), end: Self.local(11, 10),
            calendarID: "c", calendarTitle: "Work", hasOtherAttendees: true)
        let decision = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(11, 10), events: [standup]),
            state: state)
        #expect(
            decision.effects.contains {
                if case .presentCard(_, .panel) = $0 { true } else { false }
            })
        #expect(Self.cues(decision.effects).isEmpty)
    }

    @Test func aSlotWhoseReminderRingsAtTheSameMinuteIsLeftToReminders() {
        var state = Self.state()
        state.lastTickAt = Self.local(16, 29)
        let decision = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(16, 30)), state: state)
        #expect(Self.cues(decision.effects).isEmpty)
        #expect(decision.state.cuedSteps[StepCue.key(state.plan[2])] != nil)
    }

    @Test func aNewDayForgetsYesterdaysCues() {
        let cued = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(11, 10)), state: Self.state())
        #expect(!cued.state.cuedSteps.isEmpty)
        let morning = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(9, day: 31), present: false),
            state: cued.state)
        #expect(morning.state.cuedSteps.isEmpty)
    }

    // MARK: The owner's choice

    /// The letter's cue, up at 11:10.
    static func cued() -> DayState {
        DayEngine.decide(
            .tick, snapshot: snapshot(at: local(11, 10)), state: state()
        ).state
    }

    static func choose(_ choice: StepChoice, at now: Date) -> DayEngine.Decision {
        DayEngine.decide(
            .cardAction(.step(reminderID: "letter", choice)), snapshot: snapshot(at: now),
            state: cued())
    }

    @Test func startBeginsTheSlotNowAndNeedsNoSecondCue() throws {
        let started = Self.choose(.start, at: Self.local(11, 13))
        let slot = try #require(started.state.plan.first { $0.reminderID == "letter" })
        #expect(slot.start == Self.local(11, 13))
        #expect(slot.minutes == 20)
        let fields = try #require(Self.traced(.cueReaction, in: started.effects))
        #expect(fields["action"] == .string("start"))
        #expect(fields["secondsToReact"] == .double(180))
        let next = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(11, 14)), state: started.state)
        #expect(Self.cues(next.effects).isEmpty)
    }

    @Test func laterMovesTheSlotAQuarterHourOnAndCuesItThen() throws {
        let later = Self.choose(.later, at: Self.local(11, 12))
        let slot = try #require(later.state.plan.first { $0.reminderID == "letter" })
        #expect(slot.start == Self.local(11, 27))
        #expect(slot.minutes == 20)
        let quiet = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(11, 20)), state: later.state)
        #expect(Self.cues(quiet.effects).isEmpty)
        let again = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(11, 27)), state: later.state)
        #expect(Self.cues(again.effects).first?.start == Self.local(11, 27))
    }

    @Test func tomorrowTakesItOffTodayAndDatesItTomorrow() {
        let tomorrow = Self.choose(.tomorrow, at: Self.local(11, 11))
        #expect(!tomorrow.state.plan.contains { $0.reminderID == "letter" })
        #expect(tomorrow.effects.contains(.mutateAgenda(.dueTomorrow(reminderID: "letter"))))
        #expect(Self.traced(.cueReaction, in: tomorrow.effects)?["action"] == .string("tomorrow"))
    }

    @Test func doneCompletesTheReminderAndKeepsItsPlaceInTheDay() {
        let done = Self.choose(.done, at: Self.local(11, 11))
        #expect(done.effects.contains(.mutateAgenda(.complete(reminderID: "letter"))))
        #expect(done.state.plan == Self.cued().plan)
    }

    @Test func closingTheCueChangesNothing() {
        let closed = Self.choose(.dismiss, at: Self.local(11, 11))
        #expect(closed.state.plan == Self.cued().plan)
        #expect(!closed.effects.contains { if case .mutateAgenda = $0 { true } else { false } })
        #expect(Self.traced(.cueReaction, in: closed.effects)?["action"] == .string("dismiss"))
    }

    // MARK: Saved state

    @Test func aStateSavedBeforeCuesStillLoads() throws {
        let json = #"{"day": "2026-09-30", "plan": []}"#
        let state = try JSONDecoder().decode(DayState.self, from: Data(json.utf8))
        #expect(state.cuedSteps.isEmpty)
    }
}
