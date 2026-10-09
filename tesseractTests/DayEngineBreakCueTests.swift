//
//  DayEngineBreakCueTests.swift
//  tesseractTests
//
//  The body keeps time too, as decision tables: two hours at the Mac with no
//  break of five minutes or more puts a Break Cue on the Jarvis Panel, by
//  the Step Cue's manners — never while away, in quiet hours, a call, a game
//  or a meeting, never over a panel that is up, between a step's check-in
//  and the next step's start, and into a step the owner started only when
//  that runs on for half an hour or more — and each answer does what it
//  says. Five minutes away is a break, whatever the
//  cue said: the count starts again, and a cue still up comes down.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct DayEngineBreakCueTests {

    static func local(_ hour: Int, _ minute: Int = 0) -> Date {
        Calendar.current.date(
            from: DateComponents(year: 2026, month: 9, day: 30, hour: hour, minute: minute))!
    }

    static let letter = AgendaReminder(
        id: "letter", title: "Write to the case worker", listID: "inbox", listTitle: "Reminders")

    static func snapshot(
        at now: Date, present: Bool = true, frontmost: String? = "com.apple.Safari",
        game: Bool = false, panelUp: Bool = false, events: [AgendaEvent] = [],
        quiet: (Int, Int)? = nil, breakCues: Bool = true
    ) -> DaySnapshot {
        var agenda = AgendaSnapshot.empty
        agenda.access = .full
        agenda.open = [letter]
        agenda.events = events
        var settings = DaySettings()
        settings.breakCues = breakCues
        if let quiet {
            settings.quietStartMinutes = quiet.0
            settings.quietEndMinutes = quiet.1
        }
        return DaySnapshot(
            now: now, settings: settings, agenda: agenda, inboxListID: "inbox",
            ownerPresent: present, frontmostAppName: "Safari", frontmostBundleID: frontmost,
            frontmostIsGame: game, panelUp: panelUp)
    }

    /// At the Mac since 09:00, the day planned, nothing on the panel.
    static func state() -> DayState {
        var state = DayState(day: DayKey(rawValue: "2026-09-30"))
        state.syncedNudgeIDs = []
        state.morningPlanAt = local(8)
        state.sittingSince = local(9)
        state.lastPresentAt = local(10, 59)
        state.lastActiveAt = local(10, 59)
        state.lastTickAt = local(10, 59)
        return state
    }

    static func tick(_ state: DayState, at now: Date, panelUp: Bool = false)
        -> DayEngine.Decision
    {
        DayEngine.decide(.tick, snapshot: snapshot(at: now, panelUp: panelUp), state: state)
    }

    static func answer(_ choice: BreakChoice, _ state: DayState, at now: Date)
        -> DayEngine.Decision
    {
        DayEngine.decide(
            .cardAction(.breakCue(choice)), snapshot: snapshot(at: now, panelUp: true),
            state: state)
    }

    /// The cue up at 11:00.
    static func cued() -> DayState { tick(state(), at: local(11)).state }

    static func breaks(_ effects: [DayEffect]) -> [BreakCue] {
        effects.compactMap { if case .presentBreak(let cue) = $0 { cue } else { nil } }
    }

    static func steps(_ effects: [DayEffect]) -> [StepCue] {
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

    @Test func twoHoursAtTheMacPutABreakCueUp() throws {
        let decision = Self.tick(Self.state(), at: Self.local(11))
        let cue = try #require(Self.breaks(decision.effects).first)
        #expect(cue.since == Self.local(9))
        #expect(cue.minutes == 120)
        #expect(cue.number == 1)
        #expect(cue.step == nil)
        let fields = try #require(Self.traced(.cuePresented, in: decision.effects))
        #expect(fields["phase"] == .string("break"))
        #expect(fields["minutes"] == .int(120))
        #expect(fields["late"] == .int(0))
        #expect(decision.state.breakCuedAt == Self.local(11))
    }

    @Test func notBeforeTwoHours() {
        #expect(Self.breaks(Self.tick(Self.state(), at: Self.local(10, 59)).effects).isEmpty)
    }

    struct QuietRow: Sendable, CustomTestStringConvertible {
        let name: String
        let snapshot: DaySnapshot
        var testDescription: String { name }
    }

    static let quietRows: [QuietRow] = [
        QuietRow(name: "the owner is away", snapshot: snapshot(at: local(11), present: false)),
        QuietRow(name: "quiet hours", snapshot: snapshot(at: local(11), quiet: (10 * 60, 12 * 60))),
        QuietRow(name: "a game in front", snapshot: snapshot(at: local(11), game: true)),
        QuietRow(
            name: "a call in front", snapshot: snapshot(at: local(11), frontmost: "us.zoom.xos")),
        QuietRow(
            name: "the panel is up with something else",
            snapshot: snapshot(at: local(11), panelUp: true)),
        QuietRow(
            name: "a meeting is under way",
            snapshot: snapshot(
                at: local(11),
                events: [
                    AgendaEvent(
                        id: "sync", title: "Sync", start: local(10, 45), end: local(11, 30),
                        calendarID: "c", calendarTitle: "Work", hasOtherAttendees: true)
                ])),
        QuietRow(
            name: "the owner switched Break Cues off",
            snapshot: snapshot(at: local(11), breakCues: false)),
    ]

    @Test(arguments: quietRows)
    func noCue(_ row: QuietRow) {
        let decision = DayEngine.decide(.tick, snapshot: row.snapshot, state: Self.state())
        #expect(Self.breaks(decision.effects).isEmpty)
        #expect(decision.state.breakCuedAt == nil)
    }

    @Test func aCueWaitsWhileJarvisIsThinking() {
        // The Evening Wrap-up is being written: its card may take the panel
        // any moment; the break waits for it rather than go up and be replaced.
        var state = Self.state()
        state.running = .eveningWrapUp
        #expect(Self.breaks(Self.tick(state, at: Self.local(11)).effects).isEmpty)
        state.running = nil
        #expect(Self.breaks(Self.tick(state, at: Self.local(11, 1)).effects).count == 1)
    }

    @Test func theFirstTickAtTheMacStartsTheCount() {
        var state = Self.state()
        state.sittingSince = nil
        let away = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(11), present: false), state: state)
        #expect(away.state.sittingSince == nil)
        let here = Self.tick(state, at: Self.local(11))
        #expect(here.state.sittingSince == Self.local(11))
        #expect(Self.breaks(here.effects).isEmpty)
    }

    // MARK: Between steps

    /// The letter, started at 10:45, runs to 11:15 — or, `minutes` long,
    /// later.
    static func onAStep(minutes: Int = 30) -> DayState {
        var state = state()
        let slot = Placement(reminderID: "letter", start: local(10, 45), minutes: minutes)
        state.plan = [slot]
        state.startedSteps = [StepCue.key(slot)]
        state.cuedSteps[StepCue.key(slot)] = local(10, 45)
        return state
    }

    @Test func aStepEndingWithinHalfAnHourHoldsTheBreakItsCheckInComesFirst() throws {
        #expect(Self.breaks(Self.tick(Self.onAStep(), at: Self.local(11)).effects).isEmpty)
        let end = Self.tick(Self.onAStep(), at: Self.local(11, 15))
        #expect(Self.steps(end.effects).first?.phase == .end)
        #expect(Self.breaks(end.effects).isEmpty)
        let done = DayEngine.decide(
            .cardAction(.step(reminderID: "letter", .done)),
            snapshot: Self.snapshot(at: Self.local(11, 16), panelUp: true), state: end.state)
        let rest = try #require(
            Self.breaks(Self.tick(done.state, at: Self.local(11, 17)).effects).first)
        #expect(rest.minutes == 137)
    }

    @Test func aLongStepGetsTheBreakMidwayAndRunsOn() throws {
        // Started at 10:45 for two and a half hours: to 13:15.
        let midway = Self.tick(Self.onAStep(minutes: 150), at: Self.local(11))
        let rest = try #require(Self.breaks(midway.effects).first)
        #expect(rest.minutes == 120)
        #expect(rest.step == "Write to the case worker")
        #expect(rest.stepEnd == Self.local(13, 15))
        #expect(Self.steps(midway.effects).isEmpty)
        let taken = Self.answer(.taking, midway.state, at: Self.local(11, 1))
        #expect(taken.state.plan == Self.onAStep(minutes: 150).plan)
        #expect(taken.state.startedSteps == Self.onAStep(minutes: 150).startedSteps)
        // The step still checks in at its end.
        let end = Self.tick(taken.state, at: Self.local(13, 15))
        #expect(Self.steps(end.effects).first?.phase == .end)
        #expect(Self.breaks(end.effects).isEmpty)
    }

    @Test func aBreakComesBeforeTheNextStepStarts() throws {
        var state = Self.state()
        state.plan = [Placement(reminderID: "letter", start: Self.local(11), minutes: 30)]
        let first = Self.tick(state, at: Self.local(11))
        #expect(Self.breaks(first.effects).count == 1)
        #expect(Self.steps(first.effects).isEmpty)
        let taken = Self.answer(.taking, first.state, at: Self.local(11, 1))
        let step = try #require(
            Self.steps(Self.tick(taken.state, at: Self.local(11, 2)).effects).first)
        #expect(step.reminderID == "letter")
        #expect(step.phase == .start)
    }

    // MARK: Answers

    @Test func takingFiveStartsTheCountAgain() throws {
        let taken = Self.answer(.taking, Self.cued(), at: Self.local(11, 1))
        #expect(taken.state.sittingSince == Self.local(11, 1))
        #expect(taken.state.breakCuedAt == nil)
        let fields = try #require(Self.traced(.cueReaction, in: taken.effects))
        #expect(fields["phase"] == .string("break"))
        #expect(fields["action"] == .string("taking"))
        #expect(fields["secondsToReact"] == .double(60))
        #expect(Self.breaks(Self.tick(taken.state, at: Self.local(13)).effects).isEmpty)
        let next = try #require(
            Self.breaks(Self.tick(taken.state, at: Self.local(13, 1)).effects).first)
        #expect(next.since == Self.local(11, 1))
        #expect(next.minutes == 120)
        #expect(next.number == 2)
    }

    @Test func inThirtyMinutesAsksAgainThen() throws {
        let later = Self.answer(.later, Self.cued(), at: Self.local(11, 1))
        #expect(later.state.sittingSince == Self.local(9))
        #expect(Self.breaks(Self.tick(later.state, at: Self.local(11, 30)).effects).isEmpty)
        let again = try #require(
            Self.breaks(Self.tick(later.state, at: Self.local(11, 31)).effects).first)
        #expect(again.minutes == 151)
        #expect(again.number == 2)
    }

    @Test func closingItHoldsItForTwoHours() throws {
        let closed = Self.answer(.dismiss, Self.cued(), at: Self.local(11, 1))
        #expect(Self.traced(.cueReaction, in: closed.effects)?["action"] == .string("dismiss"))
        #expect(Self.breaks(Self.tick(closed.state, at: Self.local(13)).effects).isEmpty)
        // Still the same sitting: the cue counts all of it.
        let again = try #require(
            Self.breaks(Self.tick(closed.state, at: Self.local(13, 1)).effects).first)
        #expect(again.minutes == 241)
    }

    // MARK: Breaks taken

    @Test func fiveMinutesAwayIsABreak() {
        let back = DayEngine.decide(
            .presenceReturned(awayFrom: Self.local(10, 50)),
            snapshot: Self.snapshot(at: Self.local(10, 56)), state: Self.state())
        #expect(back.state.sittingSince == Self.local(10, 56))
        #expect(Self.breaks(Self.tick(back.state, at: Self.local(11)).effects).isEmpty)
        #expect(Self.breaks(Self.tick(back.state, at: Self.local(12, 56)).effects).count == 1)
    }

    @Test func aBreakIsTracedWithTheSittingBeforeIt() throws {
        let back = DayEngine.decide(
            .presenceReturned(awayFrom: Self.local(10, 50)),
            snapshot: Self.snapshot(at: Self.local(10, 56)), state: Self.state())
        let fields = try #require(Self.traced(.breakTaken, in: back.effects))
        #expect(fields["minutesAtMac"] == .int(110))
        #expect(fields["minutesAway"] == .int(6))
        // The night is night.ended's; the app starting measures nothing.
        let night = DayEngine.decide(
            .presenceReturned(awayFrom: Self.local(2)),
            snapshot: Self.snapshot(at: Self.local(10, 56)), state: Self.state())
        #expect(Self.traced(.breakTaken, in: night.effects) == nil)
        var closed = Self.state()
        closed.lastPresentAt = Self.local(10, 30)
        let launch = DayEngine.decide(
            .companionEnabled, snapshot: Self.snapshot(at: Self.local(10, 56)), state: closed)
        #expect(Self.traced(.breakTaken, in: launch.effects) == nil)
        #expect(launch.state.sittingSince == Self.local(10, 56))
    }

    @Test func aShorterAbsenceIsNoBreak() {
        let back = DayEngine.decide(
            .presenceReturned(awayFrom: Self.local(10, 52)),
            snapshot: Self.snapshot(at: Self.local(10, 56)), state: Self.state())
        #expect(back.state.sittingSince == Self.local(9))
        #expect(Self.breaks(Self.tick(back.state, at: Self.local(11)).effects).count == 1)
    }

    @Test func aCueStillUpComesDownWhenTheOwnerTakesABreak() throws {
        let back = DayEngine.decide(
            .presenceReturned(awayFrom: Self.local(11, 2)),
            snapshot: Self.snapshot(at: Self.local(11, 9), panelUp: true), state: Self.cued())
        #expect(back.effects.contains(.retractBreak))
        let fields = try #require(Self.traced(.cueReaction, in: back.effects))
        #expect(fields["action"] == .string("away"))
        #expect(fields["secondsToReact"] == .double(120))
        #expect(back.state.breakCuedAt == nil)
        #expect(back.state.sittingSince == Self.local(11, 9))
    }

    @Test func aBreakAfterInThirtyMinutesClearsTheWait() {
        let later = Self.answer(.later, Self.cued(), at: Self.local(11, 1))
        let back = DayEngine.decide(
            .presenceReturned(awayFrom: Self.local(11, 7)),
            snapshot: Self.snapshot(at: Self.local(11, 15)), state: later.state)
        #expect(back.state.breakNotBefore == nil)
        #expect(!back.effects.contains(.retractBreak))
        #expect(Self.breaks(Self.tick(back.state, at: Self.local(11, 31)).effects).isEmpty)
    }

    // MARK: The panel

    @Test func aCardThatTakesThePanelHoldsTheCueUntilItCloses() throws {
        // The cue is up; the standup ends at 11:10 with a coding agent
        // waiting: its Breakpoint card takes the panel.
        var state = Self.cued()
        state.agents = [
            AgentSignal(
                id: "s1", kind: .waiting, agent: "Claude Code", project: "tesseract",
                directory: "/tmp/tesseract", message: "Needs approval", at: Self.local(11, 5))
        ]
        let standup = AgendaEvent(
            id: "standup", title: "Standup", start: Self.local(11, 2), end: Self.local(11, 10),
            calendarID: "c", calendarTitle: "Work", hasOtherAttendees: true)
        let card = DayEngine.decide(
            .tick,
            snapshot: Self.snapshot(at: Self.local(11, 10), panelUp: true, events: [standup]),
            state: state)
        #expect(
            card.effects.contains { if case .presentCard(_, .panel) = $0 { true } else { false } })
        #expect(card.state.breakCuedAt == nil)
        #expect(
            Self.breaks(Self.tick(card.state, at: Self.local(11, 15), panelUp: true).effects)
                .isEmpty)
        let again = try #require(
            Self.breaks(Self.tick(card.state, at: Self.local(11, 20)).effects).first)
        #expect(again.minutes == 140)
    }

    @Test func aCueOnThePanelWhenTheAppQuitsComesBackAfterTheRelaunch() {
        let relaunched = Self.cued().relaunched()
        #expect(relaunched.breakCuedAt == nil)
        #expect(Self.breaks(Self.tick(relaunched, at: Self.local(11, 2)).effects).count == 1)
    }

    @Test func switchingTheCompanionOffTakesTheCueWithIt() {
        let off = DayEngine.decide(
            .companionDisabled, snapshot: Self.snapshot(at: Self.local(11, 1)), state: Self.cued())
        #expect(off.state.breakCuedAt == nil)
    }

    // MARK: Days

    @Test func theSittingCarriesAcrossTheRollover() {
        var state = Self.state()
        state.breakNotBefore = Self.local(11, 30)
        state.breakCues = 2
        let next = state.rolledOver(to: DayKey(rawValue: "2026-10-01"))
        #expect(next.sittingSince == Self.local(9))
        #expect(next.breakNotBefore == Self.local(11, 30))
        #expect(next.breakCues == 0)
    }

    @Test func aStateSavedBeforeBreakCuesStillLoads() throws {
        let json = #"{"day": "2026-09-30", "plan": []}"#
        let state = try JSONDecoder().decode(DayState.self, from: Data(json.utf8))
        #expect(state.sittingSince == nil)
        #expect(state.breakCuedAt == nil)
        #expect(state.breakCues == 0)
    }
}
