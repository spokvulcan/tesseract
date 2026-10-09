//
//  DayEngineStepCueTests.swift
//  tesseractTests
//
//  The plan keeps its own time, as decision tables: a planned step is put on
//  the Jarvis Panel when its slot starts and the owner is at the Mac — once,
//  never while away, in quiet hours, a call, a game or a meeting, never over
//  a panel that is up, and not when its reminder rings at the same minute —
//  and each of the owner's choices (and closing it) does what it says. A
//  step the owner started checks in when its time is up; "Start now" on
//  Today counts as started and needs no cue.
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
        done: [AgendaReminder] = [], events: [AgendaEvent] = [], quiet: (Int, Int)? = nil,
        stepCues: Bool = true
    ) -> DaySnapshot {
        var agenda = AgendaSnapshot.empty
        agenda.access = .full
        agenda.open = open ?? [letter, deck, dentist]
        agenda.doneToday = done
        agenda.events = events
        var settings = DaySettings()
        settings.stepCues = stepCues
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
        QuietRow(
            name: "the owner switched Step Cues off",
            snapshot: snapshot(at: local(11, 10), stepCues: false)),
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

    // MARK: The end of a started step

    /// The letter started from its cue at 11:12: 11:12–11:32.
    static func started() -> DayState { choose(.start, at: local(11, 12)).state }

    static func tick(_ state: DayState, at now: Date, panelUp: Bool = false) -> DayEngine.Decision {
        DayEngine.decide(.tick, snapshot: snapshot(at: now, panelUp: panelUp), state: state)
    }

    @Test func aStartedStepChecksInWhenItsTimeIsUp() throws {
        #expect(Self.cues(Self.tick(Self.started(), at: Self.local(11, 31)).effects).isEmpty)
        let up = Self.tick(Self.started(), at: Self.local(11, 32))
        let cue = try #require(Self.cues(up.effects).first)
        #expect(cue.phase == .end)
        #expect(cue.reminderID == "letter")
        #expect(cue.end == Self.local(11, 32))
        #expect(Self.traced(.cuePresented, in: up.effects)?["phase"] == .string("end"))
        let after = Self.tick(up.state, at: Self.local(11, 33))
        #expect(Self.cues(after.effects).isEmpty)
    }

    @Test func theStepTheOwnerStartedIsTheMenuBarsFocusWhileItRuns() throws {
        let started = Self.started()
        let focus = try #require(
            DayEngine.focus(snapshot: Self.snapshot(at: Self.local(11, 20)), state: started))
        #expect(focus.reminderID == "letter")
        #expect(focus.end == Self.local(11, 32))
        #expect(
            DayEngine.focus(snapshot: Self.snapshot(at: Self.local(11, 32)), state: started) == nil)
        let doneEarly = Self.snapshot(
            at: Self.local(11, 20), open: [Self.deck, Self.dentist], done: [Self.letter])
        #expect(DayEngine.focus(snapshot: doneEarly, state: started) == nil)
        // A slot that was only cued, never started, is no focus.
        #expect(
            DayEngine.focus(snapshot: Self.snapshot(at: Self.local(11, 15)), state: Self.cued())
                == nil)
    }

    @Test func theMenuBarSaysTheTimeLeftShort() {
        let now = Self.local(11, 0)
        #expect(MenuBarFocusText.timeLeft(until: Self.local(11, 25), now: now) == "25m")
        #expect(MenuBarFocusText.timeLeft(until: Self.local(12, 5), now: now) == "1h 5m")
        #expect(MenuBarFocusText.timeLeft(until: Self.local(12, 0), now: now) == "1h")
        #expect(MenuBarFocusText.timeLeft(until: now.addingTimeInterval(30), now: now) == "1m")
        #expect(MenuBarFocusText.spoken(until: Self.local(11, 25), now: now) == "25 min left")
    }

    @Test func startNowOnTodayIsStartedAndNeedsNoCue() throws {
        let placed = DayEngine.decide(
            .cardAction(.place(reminderID: "deck", start: Self.local(11, 15), minutes: 25)),
            snapshot: Self.snapshot(at: Self.local(11, 15)), state: Self.cued())
        let next = Self.tick(placed.state, at: Self.local(11, 16))
        #expect(Self.cues(next.effects).isEmpty)
        let up = Self.tick(placed.state, at: Self.local(11, 40))
        let cue = try #require(Self.cues(up.effects).first)
        #expect(cue.phase == .end)
        #expect(cue.reminderID == "deck")
    }

    @Test func aStartHeldBackByAFocusSessionComesOnceTheOwnerIsFree() throws {
        // The letter, started at 11:30, runs until 11:50 over the deck's 11:35.
        let started = Self.choose(.start, at: Self.local(11, 30)).state
        let held = Self.tick(started, at: Self.local(11, 35))
        #expect(Self.cues(held.effects).isEmpty)
        // The letter's check-in names the deck, which is already under way…
        let up = Self.tick(held.state, at: Self.local(11, 50))
        let checkIn = try #require(Self.cues(up.effects).first)
        #expect(checkIn.phase == .end)
        #expect(checkIn.next == "Reply to Anna about the deck")
        // …and once it is answered, the deck is cued, though its start is
        // more than ten minutes gone.
        let answered = DayEngine.decide(
            .cardAction(.step(reminderID: "letter", .done)),
            snapshot: Self.snapshot(at: Self.local(11, 51)), state: up.state)
        let deck = try #require(
            Self.cues(
                DayEngine.decide(
                    .tick,
                    snapshot: Self.snapshot(
                        at: Self.local(11, 52), open: [Self.deck, Self.dentist],
                        done: [Self.letter]),
                    state: answered.state
                ).effects
            ).first)
        #expect(deck.reminderID == "deck")
        #expect(deck.phase == .start)
    }

    @Test func aStartedStepIsNotInterruptedByTheNextStart() {
        // The letter, started at 11:30, runs until 11:50 over the deck's 11:35.
        let started = Self.choose(.start, at: Self.local(11, 30)).state
        #expect(Self.cues(Self.tick(started, at: Self.local(11, 35)).effects).isEmpty)
        // Done early: the deck's start is cued after all.
        let doneEarly = DayEngine.decide(
            .tick,
            snapshot: Self.snapshot(
                at: Self.local(11, 38), open: [Self.deck, Self.dentist], done: [Self.letter]),
            state: started)
        #expect(Self.cues(doneEarly.effects).first?.reminderID == "deck")
    }

    @Test func aStepThatWasNeverStartedDoesNotCheckIn() {
        let closed = Self.choose(.dismiss, at: Self.local(11, 11))
        #expect(Self.cues(Self.tick(closed.state, at: Self.local(11, 30)).effects).isEmpty)
    }

    @Test func aStepDoneBeforeItsTimeIsUpDoesNotCheckIn() {
        let decision = DayEngine.decide(
            .tick,
            snapshot: Self.snapshot(
                at: Self.local(11, 32), open: [Self.deck, Self.dentist], done: [Self.letter]),
            state: Self.started())
        #expect(Self.cues(decision.effects).isEmpty)
    }

    @Test func fifteenMoreMinutesAnsweredLateCountsFromNow() throws {
        // The check-in went up at 11:32; it is answered only at 11:50.
        let up = Self.tick(Self.started(), at: Self.local(11, 32))
        let longer = DayEngine.decide(
            .cardAction(.step(reminderID: "letter", .extend)),
            snapshot: Self.snapshot(at: Self.local(11, 50)), state: up.state)
        let slot = try #require(longer.state.plan.first { $0.reminderID == "letter" })
        #expect(slot.start == Self.local(11, 12))
        #expect(slot.minutes == 53)
        #expect(Self.cues(Self.tick(longer.state, at: Self.local(11, 51)).effects).isEmpty)
        let again = try #require(
            Self.cues(Self.tick(longer.state, at: Self.local(12, 5)).effects).first)
        #expect(again.phase == .end)
    }

    @Test func afterAnEarlySitDownTheMorningsQuietHoursDontHoldThePlanBack() throws {
        // Quiet until 08:00; the owner sat down at 06:12 and the plan put the
        // letter at 06:30.
        var state = Self.state()
        state.plan = [Placement(reminderID: "letter", start: Self.local(6, 30), minutes: 20)]
        state.lastTickAt = Self.local(6, 29)
        let asleep = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(6, 30), quiet: (23 * 60, 8 * 60)),
            state: state)
        #expect(Self.cues(asleep.effects).isEmpty)
        state.satDownAt = Self.local(6, 12)
        let up = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(6, 30), quiet: (23 * 60, 8 * 60)),
            state: state)
        #expect(try #require(Self.cues(up.effects).first).reminderID == "letter")
        // The evening's quiet hours still hold.
        state.plan = [Placement(reminderID: "letter", start: Self.local(23, 30), minutes: 20)]
        let night = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(23, 30), quiet: (23 * 60, 8 * 60)),
            state: state)
        #expect(Self.cues(night.effects).isEmpty)
    }

    @Test func fifteenMoreMinutesRunsItLongerAndChecksInAgain() throws {
        let up = Self.tick(Self.started(), at: Self.local(11, 32))
        let longer = DayEngine.decide(
            .cardAction(.step(reminderID: "letter", .extend)),
            snapshot: Self.snapshot(at: Self.local(11, 33)), state: up.state)
        // Answered at 11:33, a minute after its end: fifteen from now.
        let slot = try #require(longer.state.plan.first { $0.reminderID == "letter" })
        #expect(slot.start == Self.local(11, 12))
        #expect(slot.minutes == 36)
        #expect(Self.traced(.cueReaction, in: longer.effects)?["action"] == .string("extend"))
        #expect(Self.cues(Self.tick(longer.state, at: Self.local(11, 40)).effects).isEmpty)
        let again = try #require(
            Self.cues(Self.tick(longer.state, at: Self.local(11, 48)).effects).first)
        #expect(again.phase == .end)
        #expect(again.end == Self.local(11, 48))
    }

    @Test func aStepThatEndsAsTheNextBeginsChecksInFirst() throws {
        // The letter, started at 11:15, ends as the deck's slot begins.
        let started = Self.choose(.start, at: Self.local(11, 15)).state
        let both = Self.tick(started, at: Self.local(11, 35))
        let first = try #require(Self.cues(both.effects).first)
        #expect(first.phase == .end)
        #expect(first.reminderID == "letter")
        #expect(Self.cues(both.effects).count == 1)
        let waiting = Self.tick(both.state, at: Self.local(11, 36), panelUp: true)
        #expect(Self.cues(waiting.effects).isEmpty)
        let next = try #require(
            Self.cues(Self.tick(waiting.state, at: Self.local(11, 37)).effects).first)
        #expect(next.phase == .start)
        #expect(next.reminderID == "deck")
    }

    // MARK: Saved state

    @Test func aStateSavedBeforeCuesStillLoads() throws {
        let json = #"{"day": "2026-09-30", "plan": []}"#
        let state = try JSONDecoder().decode(DayState.self, from: Data(json.utf8))
        #expect(state.cuedSteps.isEmpty)
        #expect(state.startedSteps.isEmpty)
    }
}
