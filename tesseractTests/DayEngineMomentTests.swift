//
//  DayEngineMomentTests.swift
//  tesseractTests
//
//  The Day Engine's moments as decision tables: when the Morning Plan and the
//  Evening Wrap-up run, that they never run without new input, how a reply
//  becomes a card (or one retry, or the fallback), and what the owner's card
//  actions do. Signals and snapshots in, effects out: no model, no EventKit.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct DayEngineMomentTests {

    static func local(_ day: Int, _ hour: Int, _ minute: Int = 0) -> Date {
        Calendar.current.date(
            from: DateComponents(year: 2026, month: 9, day: day, hour: hour, minute: minute))!
    }

    static let review = AgendaReminder(
        id: "R1", title: "Review PR 42", listID: "work", listTitle: "Work",
        due: local(30, 0), dueHasTime: false)
    static let gym = AgendaReminder(id: "R2", title: "Gym", listID: "health", listTitle: "Health")
    static let standup = AgendaEvent(
        id: "E1", title: "Standup", start: local(30, 9, 30), end: local(30, 10),
        calendarID: "c", calendarTitle: "Work")

    static func snapshot(
        at now: Date, present: Bool = true, chatBusy: Bool = false,
        open: [AgendaReminder] = [review, gym], done: [AgendaReminder] = [],
        power: PowerState = .nominal
    ) -> DaySnapshot {
        var agenda = AgendaSnapshot.empty
        agenda.access = .full
        agenda.events = [standup]
        agenda.open = open
        agenda.doneToday = done
        return DaySnapshot(
            now: now, settings: DaySettings(), agenda: agenda,
            areas: [Area(id: "work", name: "Work"), Area(id: "health", name: "Health")],
            inboxListID: "inbox", ownerPresent: present, chatBusy: chatBusy, power: power)
    }

    static func state(_ day: Int = 30) -> DayState {
        var state = DayState(day: DayKey(rawValue: "2026-09-\(day)"))
        state.syncedNudgeIDs = []
        return state
    }

    static func moments(_ effects: [DayEffect]) -> [MomentRequest] {
        effects.compactMap { if case .runMoment(let request) = $0 { request } else { nil } }
    }

    // MARK: Morning Plan triggers

    struct MorningRow: Sendable, CustomTestStringConvertible {
        let name: String
        let now: Date
        let awayFrom: Date
        let runs: Bool
        var testDescription: String { name }
    }

    static let morningRows: [MorningRow] = [
        MorningRow(
            name: "first sit-down after the night", now: local(30, 7, 40), awayFrom: local(29, 23),
            runs: true),
        MorningRow(
            name: "a short break is not a morning", now: local(30, 7, 40), awayFrom: local(30, 5),
            runs: false),
        MorningRow(
            name: "before 04:00 is still last night", now: local(30, 3, 30),
            awayFrom: local(29, 20), runs: false),
        MorningRow(
            name: "a day that starts after the morning window", now: local(30, 14, 43),
            awayFrom: local(29, 23), runs: true),
        MorningRow(
            name: "back after a long afternoon away", now: local(30, 18),
            awayFrom: local(30, 13), runs: true),
    ]

    @Test(arguments: morningRows)
    func morningPlanRunsOnTheFirstSitDown(_ row: MorningRow) {
        let decision = DayEngine.decide(
            .presenceReturned(awayFrom: row.awayFrom), snapshot: Self.snapshot(at: row.now),
            state: Self.state(Calendar.current.component(.hour, from: row.now) < 4 ? 29 : 30))
        let requests = Self.moments(decision.effects)
        #expect(requests.map(\.kind) == (row.runs ? [.morningPlan] : []))
        if row.runs {
            #expect(decision.state.running == .morningPlan)
            #expect(requests.first?.text.contains("R1 · Work · today — Review PR 42") == true)
        }
    }

    @Test func aPlanPreparedThatMorningIsMadeAgainForALateSitDown() throws {
        // Prepared at 07:00 while the owner slept in; they sit down at 14:43.
        let prepared = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(30, 7), present: false),
            state: {
                var state = Self.state()
                state.lastPresentAt = Self.local(29, 23)
                return state
            }())
        #expect(Self.moments(prepared.effects).first?.trigger == .prepared)
        var made = prepared.state
        made.running = nil
        made.plan = [Placement(reminderID: "R2", start: Self.local(30, 8), minutes: 30)]
        let late = DayEngine.decide(
            .presenceReturned(awayFrom: Self.local(29, 23)),
            snapshot: Self.snapshot(at: Self.local(30, 14, 43)), state: made)
        let request = try #require(Self.moments(late.effects).first)
        #expect(request.kind == .morningPlan)
        #expect(request.trigger == .firstPresence)
        #expect(late.state.plan.isEmpty)
        #expect(
            late.effects.contains { if case .presentCard(_, .panel) = $0 { true } else { false } })
        #expect(late.effects.contains { if case .retractCard = $0 { true } else { false } })
    }

    @Test func aPlanSeenThatMorningIsNotMadeAgainAfterALongAbsence() {
        // Planned at 08:00 with the owner there; away from 09:00 to 14:00.
        var state = Self.state()
        state.morningPlanAt = Self.local(30, 8)
        state.cards = [
            DayCard(
                id: "morningPlan-1", kind: .morningPlan, createdAt: Self.local(30, 8),
                isFallback: false,
                body: .morningPlan(
                    MorningPlanCard(
                        line: "A calm day.", mustDoID: nil, placements: [], suggestions: [])))
        ]
        let back = DayEngine.decide(
            .presenceReturned(awayFrom: Self.local(30, 9)),
            snapshot: Self.snapshot(at: Self.local(30, 14)), state: state)
        #expect(Self.moments(back.effects).isEmpty)
    }

    @Test func afterANightPastMidnightThePlanIsAskedToKeepTheDayLight() throws {
        // At the Mac until 01:20; the plan is made ahead at 05:30.
        var late = Self.state()
        late.lastPresentAt = Self.local(30, 1, 20)
        late.lastActiveAt = Self.local(30, 1, 20)
        let prepared = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(30, 5, 30), present: false),
            state: late)
        let request = try #require(Self.moments(prepared.effects).first)
        #expect(request.kind == .morningPlan)
        #expect(
            request.text.contains("Last night they were at the Mac until 01:20, past midnight."))
        #expect(request.text.contains("Keep today light"))
        // Off to bed at 23:30: an ordinary plan.
        var early = Self.state()
        early.lastPresentAt = Self.local(29, 23, 30)
        early.lastActiveAt = Self.local(29, 23, 30)
        let ordinary = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(30, 5, 30), present: false),
            state: early)
        #expect(Self.moments(ordinary.effects).first?.text.contains("Last night") == false)
        // Locked at 23:45, the Mac asleep at 00:05 (sleep reads as a return,
        // stamping the last presence): no tick saw the owner past midnight.
        var locked = early
        locked.lastActiveAt = Self.local(29, 23, 45)
        locked.lastPresentAt = Self.local(30, 0, 5)
        let asleep = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(30, 5, 30), present: false),
            state: locked)
        #expect(Self.moments(asleep.effects).first?.text.contains("Last night") == false)
        // Up until 04:30 is up late too, though it is already the new day.
        var dawn = Self.state()
        dawn.lastActiveAt = Self.local(30, 4, 30)
        #expect(
            DaySnapshot(
                now: Self.local(30, 9), settings: DaySettings(), agenda: .empty,
                ownerPresent: true
            ).facts(state: dawn).upLateUntil == Self.local(30, 4, 30))
    }

    @Test func theDaysFirstSitDownMeasuresTheNight() throws {
        // At the Mac until 01:20, back at 07:40: 6 h 20 min away, up late.
        var state = Self.state()
        state.lastPresentAt = Self.local(30, 1, 20)
        state.lastActiveAt = Self.local(30, 1, 20)
        let sitDown = DayEngine.decide(
            .presenceReturned(awayFrom: Self.local(30, 1, 20)),
            snapshot: Self.snapshot(at: Self.local(30, 7, 40)), state: state)
        #expect(
            sitDown.effects.contains(
                .trace(.nightEnded, ["minutesAway": .int(380), "upLate": .bool(true)])))
        // A long afternoon away is not a night.
        let afternoon = DayEngine.decide(
            .presenceReturned(awayFrom: Self.local(30, 12)),
            snapshot: Self.snapshot(at: Self.local(30, 17)), state: sitDown.state)
        #expect(
            !afternoon.effects.contains {
                if case .trace(.nightEnded, _) = $0 { true } else { false }
            })
    }

    @Test func aFirstSitDownsPlanWaitsForABusyModelAndThenRuns() throws {
        // The owner sits down at 07:40 while last night's reflection still runs.
        var state = Self.state()
        state.running = .triage
        let busy = DayEngine.decide(
            .presenceReturned(awayFrom: Self.local(29, 23)),
            snapshot: Self.snapshot(at: Self.local(30, 7, 40)), state: state)
        #expect(Self.moments(busy.effects).isEmpty)
        #expect(busy.state.morningPlanWaiting)
        // The model is free a minute later: the plan comes.
        var free = busy.state
        free.running = nil
        let later = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(30, 7, 41)), state: free)
        let request = try #require(Self.moments(later.effects).first)
        #expect(request.kind == .morningPlan)
        #expect(request.trigger == .firstPresence)
        #expect(!later.state.morningPlanWaiting)
        // Opening Today first starts it there, and nothing plans the day twice.
        var opened = free
        opened =
            DayEngine.decide(
                .todayOpened, snapshot: Self.snapshot(at: Self.local(30, 7, 41)), state: opened
            ).state
        #expect(!opened.morningPlanWaiting)
        opened.running = nil
        let next = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(30, 7, 42)), state: opened)
        #expect(Self.moments(next.effects).isEmpty)
    }

    @Test func aLateReplanWaitsForAModelThatIsFree() {
        // Prepared at 07:00, the owner sits down at 14:43 while Triage runs.
        var state = Self.state()
        state.morningPlanAt = Self.local(30, 7)
        state.running = .triage
        state.plan = [Placement(reminderID: "R2", start: Self.local(30, 8), minutes: 30)]
        state.cards = [
            DayCard(
                id: "morningPlan-1", kind: .morningPlan, createdAt: Self.local(30, 7),
                isFallback: false,
                body: .morningPlan(
                    MorningPlanCard(
                        line: "A day.", mustDoID: nil, placements: state.plan, suggestions: [])))
        ]
        let back = DayEngine.decide(
            .presenceReturned(awayFrom: Self.local(29, 23)),
            snapshot: Self.snapshot(at: Self.local(30, 14, 43)), state: state)
        #expect(Self.moments(back.effects).isEmpty)
        #expect(back.state.plan == state.plan)
    }

    @Test func aReplanKeepsTheStepTheOwnerIsIn() {
        // The gym, started at 09:00 for an hour; Jarvis's plan lands at 09:10.
        var state = Self.state()
        let gym = Placement(reminderID: "R2", start: Self.local(30, 9), minutes: 60)
        state.plan = [gym]
        state.startedSteps = [StepCue.key(gym)]
        let card = MorningPlanCard(
            line: "Review first.", mustDoID: "R1",
            placements: [Placement(reminderID: "R1", start: Self.local(30, 10, 30), minutes: 30)],
            suggestions: [])
        _ = DayEngine.accept(
            .morningPlan(card), kind: .morningPlan, fallback: false,
            snapshot: Self.snapshot(at: Self.local(30, 9, 10)), state: &state)
        #expect(state.plan.map(\.reminderID) == ["R2", "R1"])
        #expect(
            DayEngine.focus(snapshot: Self.snapshot(at: Self.local(30, 9, 20)), state: state)?
                .reminderID == "R2")
    }

    @Test func morningPlanRunsOncePerDay() {
        var state = Self.state()
        state.morningPlanAt = Self.local(30, 7)
        let decision = DayEngine.decide(
            .presenceReturned(awayFrom: Self.local(29, 22)),
            snapshot: Self.snapshot(at: Self.local(30, 9)),
            state: state)
        #expect(Self.moments(decision.effects).isEmpty)
    }

    @Test func openingTodayPlansTheDayUntilTheEvening() {
        let morning = DayEngine.decide(
            .todayOpened, snapshot: Self.snapshot(at: Self.local(30, 14)), state: Self.state())
        #expect(Self.moments(morning.effects).map(\.kind) == [.morningPlan])
        #expect(Self.moments(morning.effects).first?.trigger == .todayOpened)

        let evening = DayEngine.decide(
            .todayOpened, snapshot: Self.snapshot(at: Self.local(30, 22)), state: Self.state())
        #expect(Self.moments(evening.effects).isEmpty)
    }

    @Test func aMomentWaitsWhileTheOwnerIsChatting() {
        let decision = DayEngine.decide(
            .presenceReturned(awayFrom: Self.local(29, 23)),
            snapshot: Self.snapshot(at: Self.local(30, 8), chatBusy: true), state: Self.state())
        #expect(Self.moments(decision.effects).isEmpty)
    }

    // MARK: Evening Wrap-up triggers

    @Test(arguments: [
        (21, 5, true, true), (20, 30, true, false), (21, 5, false, false), (23, 50, true, true),
    ])
    func eveningWrapUpOnTheClock(hour: Int, minute: Int, present: Bool, runs: Bool) {
        let decision = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(30, hour, minute), present: present),
            state: Self.state())
        #expect(Self.moments(decision.effects).map(\.kind) == (runs ? [.eveningWrapUp] : []))
    }

    @Test func eveningWrapUpWaitsForTheNextPresenceUntilThree() {
        // Away at 21:00; back at 01:30, which still belongs to the 30th.
        let decision = DayEngine.decide(
            .presenceReturned(awayFrom: Self.local(30, 20, 40)),
            snapshot: Self.snapshot(at: Self.local(31, 1, 30)), state: Self.state())
        #expect(Self.moments(decision.effects).map(\.kind) == [.eveningWrapUp])
    }

    @Test func noMomentRunsWithoutNewInput() {
        let first = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(30, 21, 5)), state: Self.state())
        #expect(Self.moments(first.effects).count == 1)
        // The same inputs a minute later: the moment is already running.
        let second = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(30, 21, 6)), state: first.state)
        #expect(Self.moments(second.effects).isEmpty)
    }

    // MARK: Outcomes

    private func request(_ kind: MomentKind = .morningPlan, attempt: Int = 0) -> MomentRequest {
        MomentRequest(kind: kind, trigger: .firstPresence, text: "[Morning Plan]", attempt: attempt)
    }

    private let measure = MomentMeasure(
        promptTokens: 1800, outputTokens: 240, prefillSeconds: 0.4, generateSeconds: 6,
        latencySeconds: 7, hitCap: false, modelID: "test-model")

    private func running(_ kind: MomentKind) -> DayState {
        var state = Self.state()
        state.running = kind
        return state
    }

    @Test func aValidMorningPlanBecomesTheDaysPlan() throws {
        let reply = """
            Here's the plan:
            ```json
            {"line": "A calm start: two small things, then the standup.", "must_do": "R1",
             "plan": [{"id": "R2", "at": "08:00", "minutes": 30}, {"id": "R1", "at": "09:40", "minutes": 20}],
             "suggestions": ["Reply to Anna first"]}
            ```
            """
        let decision = DayEngine.decide(
            .momentOutcome(request(), .reply(reply, measure)),
            snapshot: Self.snapshot(at: Self.local(30, 7, 45)), state: running(.morningPlan))
        #expect(decision.state.running == nil)
        #expect(decision.state.morningPlanAt == Self.local(30, 7, 45))
        #expect(decision.state.mustDoID == "R1")
        // The placement over the standup is dropped; the free one stays.
        #expect(decision.state.plan.map(\.reminderID) == ["R2"])
        let card = try #require(decision.state.cards.last)
        #expect(card.kind == .morningPlan)
        #expect(!card.isFallback)
        #expect(card.line == "A calm start: two small things, then the standup.")
        #expect(decision.effects.contains(.presentCard(card, .today)))
        #expect(
            decision.effects.contains {
                if case .trace(.momentFinished, let fields) = $0 {
                    fields["promptTokens"] == .int(1800)
                } else {
                    false
                }
            })
    }

    @Test func anUnreadableReplyGetsOneRetryThenTheFallback() throws {
        let first = DayEngine.decide(
            .momentOutcome(request(), .reply("Sure! Let me think about your day.", measure)),
            snapshot: Self.snapshot(at: Self.local(30, 7, 45)), state: running(.morningPlan))
        let retry = try #require(Self.moments(first.effects).first)
        #expect(retry.attempt == 1)
        #expect(retry.text.hasPrefix("That wasn't the card."))
        #expect(first.state.cards.isEmpty)

        let second = DayEngine.decide(
            .momentOutcome(retry, .reply("{not json", measure)),
            snapshot: Self.snapshot(at: Self.local(30, 7, 46)), state: first.state)
        #expect(Self.moments(second.effects).isEmpty)
        let card = try #require(second.state.cards.last)
        #expect(card.isFallback)
        #expect(card.line.hasPrefix("Here's your day"))
        #expect(second.state.morningPlanAt != nil)
    }

    @Test func aFailedCallRetriesTheSameRequest() throws {
        let decision = DayEngine.decide(
            .momentOutcome(request(), .failed("model not loaded", nil)),
            snapshot: Self.snapshot(at: Self.local(30, 7, 45)), state: running(.morningPlan))
        let retry = try #require(Self.moments(decision.effects).first)
        #expect(retry.text == "[Morning Plan]")
        #expect(retry.attempt == 1)
    }

    @Test func hittingTheOutputCapIsAFailure() throws {
        var capped = measure
        capped.hitCap = true
        let decision = DayEngine.decide(
            .momentOutcome(request(attempt: 1), .reply(#"{"line": "cut"#, capped)),
            snapshot: Self.snapshot(at: Self.local(30, 7, 45)), state: running(.morningPlan))
        #expect(try #require(decision.state.cards.last).isFallback)
    }

    // MARK: A moment that goes quiet

    /// The plan Jarvis began at the day's first sit-down, 07:40.
    static func planStarted() -> DayEngine.Decision {
        DayEngine.decide(
            .presenceReturned(awayFrom: local(29, 23)), snapshot: snapshot(at: local(30, 7, 40)),
            state: state())
    }

    /// A tick a minute, the Mac awake, from `from` through `to`.
    static func ticks(_ state: DayState, from: Date, through to: Date) -> (DayState, [DayEffect]) {
        var state = state
        var effects: [DayEffect] = []
        var now = from
        while now <= to {
            let decision = DayEngine.decide(.tick, snapshot: snapshot(at: now), state: state)
            state = decision.state
            effects += decision.effects
            now = now.addingTimeInterval(60)
        }
        return (state, effects)
    }

    static func failure(_ effects: [DayEffect]) -> [String: CompanionTraceValue]? {
        for effect in effects {
            if case .trace(.momentFailed, let fields) = effect { return fields }
        }
        return nil
    }

    @Test func aMomentQuietForAQuarterHourIsStoppedAndAskedOnceMore() throws {
        let started = Self.planStarted()
        let first = try #require(Self.moments(started.effects).first)
        #expect(started.state.runningRequest == first)
        let (waiting, quiet) = Self.ticks(
            started.state, from: Self.local(30, 7, 41), through: Self.local(30, 7, 54))
        #expect(!quiet.contains(.cancelMoment))
        #expect(waiting.running == .morningPlan)

        let given = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(30, 7, 55)), state: waiting)
        #expect(given.effects.first == .cancelMoment)
        #expect(Self.failure(given.effects)?["reason"] == .string("no answer in 15 min"))
        let retry = try #require(Self.moments(given.effects).first)
        #expect(retry.attempt == 1)
        #expect(retry.trigger == .retry)
        #expect(given.state.runningRequest == retry)

        // The first run answering after all changes nothing: the retry runs.
        let late = DayEngine.decide(
            .momentOutcome(first, .reply(#"{"line": "Late."}"#, measure)),
            snapshot: Self.snapshot(at: Self.local(30, 7, 56)), state: given.state)
        #expect(late.effects.isEmpty)
        #expect(late.state.runningRequest == retry)
        #expect(late.state.running == .morningPlan)
    }

    @Test func aRetryThatGoesQuietTooLeavesTheCardCodeBuilt() throws {
        let started = Self.planStarted()
        let (gaveUp, _) = Self.ticks(
            started.state, from: Self.local(30, 7, 41), through: Self.local(30, 7, 55))
        let (final, effects) = Self.ticks(
            gaveUp, from: Self.local(30, 7, 56), through: Self.local(30, 8, 10))
        #expect(effects.contains(.cancelMoment))
        #expect(Self.failure(effects)?["fallback"] == .bool(true))
        #expect(final.running == nil)
        #expect(final.runningRequest == nil)
        let card = try #require(final.cards.last { $0.kind == .morningPlan })
        #expect(card.isFallback)
        #expect(!card.isRefining)
        // Free again: what comes next can run.
        #expect(Self.moments(effects).isEmpty)
    }

    @Test func timeTheMacSleptDoesNotCount() throws {
        let started = Self.planStarted()
        var state = started.state
        state.lastTickAt = Self.local(30, 7, 45)
        // The lid closed at 07:45 and opened at 09:00: the run slept with it.
        let woke = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(30, 9)), state: state)
        #expect(!woke.effects.contains(.cancelMoment))
        #expect(woke.state.running == .morningPlan)
        #expect(woke.state.runningSince == Self.local(30, 8, 55))
        // Awake ten more minutes: fifteen in all, given up.
        let (_, effects) = Self.ticks(
            woke.state, from: Self.local(30, 9, 1), through: Self.local(30, 9, 10))
        #expect(effects.contains(.cancelMoment))
    }

    // MARK: Evening Wrap-up and its actions

    @Test func theWrapUpOffersLeftoversAndTheirActionsChangeReminders() throws {
        let reply =
            #"{"line": "Good day: the PR is reviewed.", "leftovers": [{"id": "R1", "suggest": "drop"}]}"#
        var state = running(.eveningWrapUp)
        state.plan = [Placement(reminderID: "R2", start: Self.local(30, 18), minutes: 45)]
        let decided = DayEngine.decide(
            .momentOutcome(
                MomentRequest(kind: .eveningWrapUp, trigger: .eveningTime, text: "x"),
                .reply(reply, measure)),
            snapshot: Self.snapshot(at: Self.local(30, 21, 5)), state: state)
        let card = try #require(decided.state.cards.last)
        guard case .eveningWrapUp(let wrapUp) = card.body else {
            Issue.record("expected an Evening Wrap-up card")
            return
        }
        // R1 was due today; R2 was planned today.
        #expect(Set(wrapUp.leftovers.map(\.reminderID)) == ["R1", "R2"])
        #expect(wrapUp.leftovers.first { $0.reminderID == "R1" }?.suggestion == .drop)
        #expect(wrapUp.leftovers.first { $0.reminderID == "R2" }?.suggestion == .tomorrow)
        #expect(decided.effects.contains(.setWaiting(2)))

        let one = DayEngine.decide(
            .cardAction(.leftover(cardID: card.id, reminderID: "R2", .later)),
            snapshot: Self.snapshot(at: Self.local(30, 21, 10)), state: decided.state)
        #expect(one.effects.contains(.mutateAgenda(.clearDue(reminderID: "R2"))))
        #expect(one.state.plan.isEmpty)

        let all = DayEngine.decide(
            .cardAction(.allLeftovers(cardID: card.id)),
            snapshot: Self.snapshot(at: Self.local(30, 21, 11)), state: one.state)
        #expect(all.effects.contains(.mutateAgenda(.delete(reminderID: "R1"))))
        #expect(all.effects.contains(.setWaiting(0)))
    }

    @Test func aRepeatingLeftoverSkipsToTomorrowInsteadOfLosingItsSeries() throws {
        // A daily habit is one reminder for its whole series.
        let daily = AgendaReminder(
            id: "R1", title: "Duolingo", listID: "daily", listTitle: "Daily",
            due: Self.local(30, 0), repeats: true)
        let reply =
            #"{"line": "A good day.", "leftovers": [{"id": "R1", "suggest": "drop"}]}"#
        let decided = DayEngine.decide(
            .momentOutcome(
                MomentRequest(kind: .eveningWrapUp, trigger: .eveningTime, text: "x"),
                .reply(reply, measure)),
            snapshot: Self.snapshot(at: Self.local(30, 21, 5), open: [daily]),
            state: running(.eveningWrapUp))
        let card = try #require(decided.state.cards.last)
        for choice in [Leftover.Suggestion.drop, .later] {
            let decision = DayEngine.decide(
                .cardAction(.leftover(cardID: card.id, reminderID: "R1", choice)),
                snapshot: Self.snapshot(at: Self.local(30, 21, 10), open: [daily]),
                state: decided.state)
            #expect(decision.effects.contains(.mutateAgenda(.dueTomorrow(reminderID: "R1"))))
            #expect(!decision.effects.contains(.mutateAgenda(.delete(reminderID: "R1"))))
            #expect(!decision.effects.contains(.mutateAgenda(.clearDue(reminderID: "R1"))))
        }
    }

    @Test func nothingIsEverCalledMissed() {
        let facts = Self.snapshot(at: Self.local(30, 21)).facts(state: Self.state())
        let card = FallbackCards.eveningWrapUp(facts: facts, leftovers: [Self.review])
        for text in [card.line] + card.leftovers.map(\.title) {
            #expect(!text.lowercased().contains("missed"))
            #expect(!text.lowercased().contains("failed"))
        }
    }

    // MARK: Plan actions and rollover

    @Test func findATimePlacesATaskAndTheMustDoFollowsTheOwner() {
        let placed = DayEngine.decide(
            .cardAction(.place(reminderID: "R2", start: Self.local(30, 16), minutes: 30)),
            snapshot: Self.snapshot(at: Self.local(30, 11)), state: Self.state())
        #expect(
            placed.state.plan == [
                Placement(reminderID: "R2", start: Self.local(30, 16), minutes: 30)
            ])
        let mustDo = DayEngine.decide(
            .cardAction(.setMustDo(reminderID: "R2")),
            snapshot: Self.snapshot(at: Self.local(30, 11)),
            state: placed.state)
        #expect(mustDo.state.mustDoID == "R2")
    }

    @Test func aNewDayCarriesTheNoteAndForgetsYesterdaysPlan() {
        var yesterday = Self.state(29)
        yesterday.carryOverForNextDay = "Pick up the spec where you left it."
        yesterday.mustDoID = "R1"
        yesterday.plan = [Placement(reminderID: "R2", start: Self.local(29, 16), minutes: 30)]
        yesterday.lastPresentAt = Self.local(29, 23)
        let decision = DayEngine.decide(
            .tick, snapshot: Self.snapshot(at: Self.local(30, 6)), state: yesterday)
        #expect(decision.state.day == DayKey(rawValue: "2026-09-30"))
        #expect(decision.state.carryOver == "Pick up the spec where you left it.")
        #expect(decision.state.mustDoID == nil)
        #expect(decision.state.plan.isEmpty)
    }
}

struct WeekFocusTests {

    @Test func theLookBacksFocusRidesTheWeekAndThenGoes() throws {
        var state = DayState(day: DayKey(rawValue: "2026-10-04"))
        state.syncedNudgeIDs = []
        let reply =
            #"{"line": "A good week.", "leftovers": [], "week": "Steady.", "focus": "the job search"}"#
        let measure = MomentMeasure(
            promptTokens: 1000, outputTokens: 100, prefillSeconds: 1, generateSeconds: 2,
            latencySeconds: 3, hitCap: false, modelID: "m")
        let calendar = MomentPromptsTests.mondayFirst
        let at = { (d: Int, h: Int) in
            calendar.date(from: DateComponents(year: 2026, month: 10, day: d, hour: h))!
        }
        var agenda = AgendaSnapshot.empty
        agenda.access = .full
        func snapshot(_ now: Date) -> DaySnapshot {
            DaySnapshot(
                now: now, calendar: calendar, settings: DaySettings(), agenda: agenda,
                ownerPresent: false)
        }
        state.running = .eveningWrapUp
        let accepted = DayEngine.decide(
            .momentOutcome(
                MomentRequest(kind: .eveningWrapUp, trigger: .eveningTime, text: "x"),
                .reply(reply, measure)),
            snapshot: snapshot(at(4, 21)), state: state)
        #expect(accepted.state.weekFocus == "the job search")
        // Monday and the rest of the week: it holds.
        let monday = DayEngine.decide(.tick, snapshot: snapshot(at(5, 9)), state: accepted.state)
        #expect(monday.state.weekFocus == "the job search")
        #expect(monday.state.day == DayKey(rawValue: "2026-10-05"))
        // Set after midnight, it still belongs to Sunday's look-back: through
        // the next Monday, not a day more.
        var late = accepted.state
        late.weekFocusSetAt = at(5, 1)
        late.day = DayKey(rawValue: "2026-10-12")
        #expect(late.rolledOver(to: DayKey(rawValue: "2026-10-12")).weekFocus == "the job search")
        #expect(late.rolledOver(to: DayKey(rawValue: "2026-10-13")).weekFocus == nil)
        // Ten days on, with no new look-back: it is gone.
        var later = monday.state
        later.day = DayKey(rawValue: "2026-10-13")
        let gone = DayEngine.decide(.tick, snapshot: snapshot(at(14, 9)), state: later)
        #expect(gone.state.weekFocus == nil)
    }
}

struct KeptCardWaitingTests {

    @Test func aKeptWrapUpWaitsOnNobodyAndANewDayClearsTheGlyph() {
        var state = DayState(day: DayKey(rawValue: "2026-09-30"))
        state.syncedNudgeIDs = []
        state.cards = [
            DayCard(
                id: "w", kind: .eveningWrapUp, createdAt: DayEngineMomentTests.local(30, 21),
                isFallback: false,
                body: .eveningWrapUp(
                    EveningWrapUpCard(
                        line: "Done.", done: [],
                        leftovers: [
                            Leftover(reminderID: "R1", title: "Spec", suggestion: .tomorrow)
                        ],
                        tomorrowFirst: nil)))
        ]
        #expect(DayEngine.waitingCount(state, now: DayEngineMomentTests.local(30, 21)) == 1)
        let kept = DayEngine.decide(
            .cardAction(.keep(cardID: "w")),
            snapshot: DayEngineMomentTests.snapshot(at: DayEngineMomentTests.local(30, 22)),
            state: state)
        #expect(kept.effects.contains(.setWaiting(0)))
        // Not kept: the 04:00 rollover clears the card, and the glyph with it.
        let morning = DayEngine.decide(
            .tick,
            snapshot: DayEngineMomentTests.snapshot(
                at: DayEngineMomentTests.local(31, 9), present: false),
            state: state)
        #expect(morning.effects.contains(.setWaiting(0)))
    }
}

/// Closing the panel (its ×) takes a card in, as Looks Good does: off the
/// panel, still in Today with what it holds to settle. It used to dismiss
/// the card: an evening's leftovers, closed in seconds as the panel came
/// up, were gone from Today too.
struct PanelCloseTests {

    @Test func aWrapUpClosedOnThePanelKeepsItsLeftoversInToday() throws {
        var state = DayState(day: DayKey(rawValue: "2026-09-30"))
        state.syncedNudgeIDs = []
        state.cards = [
            DayCard(
                id: "w", kind: .eveningWrapUp, createdAt: DayEngineMomentTests.local(30, 21),
                isFallback: false,
                body: .eveningWrapUp(
                    EveningWrapUpCard(
                        line: "Done.", done: [],
                        leftovers: [
                            Leftover(reminderID: "R1", title: "Spec", suggestion: .tomorrow)
                        ],
                        tomorrowFirst: nil)))
        ]
        let closed = DayEngine.decide(
            .cardAction(.close(cardID: "w")),
            snapshot: DayEngineMomentTests.snapshot(at: DayEngineMomentTests.local(30, 21, 1)),
            state: state)
        let card = try #require(closed.state.openCards.first)
        #expect(card.id == "w")
        #expect(card.kept)
        #expect(closed.effects.contains(.setWaiting(0)))
        #expect(!closed.effects.contains(.retractCard(cardID: "w")))
        #expect(
            closed.effects.contains {
                if case .trace(.cardReaction, let fields) = $0 {
                    fields["action"] == .string("closed")
                } else {
                    false
                }
            })
        // A leftover settled later, from Today.
        let settled = DayEngine.decide(
            .cardAction(.leftover(cardID: "w", reminderID: "R1", .tomorrow)),
            snapshot: DayEngineMomentTests.snapshot(at: DayEngineMomentTests.local(30, 21, 50)),
            state: closed.state)
        #expect(settled.effects.contains(.mutateAgenda(.dueTomorrow(reminderID: "R1"))))
    }
}

struct MustDoDaysTests {

    @Test func eachDaysMustDoIsKeptForTheWeek() {
        var state = DayState(day: DayKey(rawValue: "2026-09-30"))
        state.syncedNudgeIDs = []
        state.mustDoID = "R1"
        var agenda = AgendaSnapshot.empty
        agenda.access = .full
        agenda.doneToday = [
            AgendaReminder(
                id: "R1", title: "Spec", listID: "w", listTitle: "Work", isCompleted: true,
                completedAt: DayEngineMomentTests.local(30, 15))
        ]
        let seen = DayEngine.decide(
            .tick,
            snapshot: DaySnapshot(
                now: DayEngineMomentTests.local(30, 15, 1), settings: DaySettings(),
                agenda: agenda, ownerPresent: false),
            state: state)
        #expect(seen.state.mustDoDoneAt == DayEngineMomentTests.local(30, 15, 1))
        let next = seen.state.rolledOver(to: DayKey(rawValue: "2026-10-01"))
        #expect(next.mustDoDays == ["2026-09-30": true])
        #expect(next.mustDoDoneAt == nil)
        // A week and more later, the day has left the record.
        var later = next
        later.day = DayKey(rawValue: "2026-10-08")
        #expect(later.rolledOver(to: DayKey(rawValue: "2026-10-09")).mustDoDays.isEmpty)
    }

    static func spec(doneAt: Date) -> AgendaReminder {
        AgendaReminder(
            id: "R1", title: "Spec", listID: "w", listTitle: "Work", isCompleted: true,
            completedAt: doneAt)
    }

    @Test func aDifferentMustDoStartsUndone() {
        var state = DayState(day: DayKey(rawValue: "2026-09-30"))
        state.syncedNudgeIDs = []
        state.mustDoID = "R1"
        state.mustDoDoneAt = DayEngineMomentTests.local(30, 11)
        var agenda = AgendaSnapshot.empty
        agenda.access = .full
        agenda.doneToday = [Self.spec(doneAt: DayEngineMomentTests.local(30, 11))]
        let changed = DayEngine.decide(
            .cardAction(.setMustDo(reminderID: "R2")),
            snapshot: DaySnapshot(
                now: DayEngineMomentTests.local(30, 13), settings: DaySettings(), agenda: agenda,
                ownerPresent: true),
            state: state)
        #expect(changed.state.mustDoID == "R2")
        #expect(changed.state.mustDoDoneAt == nil)
        let next = changed.state.rolledOver(to: DayKey(rawValue: "2026-10-01"))
        #expect(next.mustDoDays == ["2026-09-30": false])
    }

    @Test func aMustDoDoneOnThePhoneWhileTheMacSleptCountsForItsDay() {
        var state = DayState(day: DayKey(rawValue: "2026-09-30"))
        state.syncedNudgeIDs = []
        state.mustDoID = "R1"
        // Done at 23:00; the Mac wakes the next morning.
        var agenda = AgendaSnapshot.empty
        agenda.access = .full
        agenda.doneThisWeek = [Self.spec(doneAt: DayEngineMomentTests.local(30, 23))]
        let morning = DayEngine.decide(
            .tick,
            snapshot: DaySnapshot(
                now: DayEngineMomentTests.local(31, 8), settings: DaySettings(), agenda: agenda,
                ownerPresent: false),
            state: state)
        #expect(morning.state.day == DayKey(rawValue: "2026-10-01"))
        #expect(morning.state.mustDoDays == ["2026-09-30": true])
        // Done only the next morning, before the Mac woke: not that day's.
        agenda.doneThisWeek = [Self.spec(doneAt: DayEngineMomentTests.local(31, 8, 30))]
        let late = DayEngine.decide(
            .tick,
            snapshot: DaySnapshot(
                now: DayEngineMomentTests.local(31, 9), settings: DaySettings(), agenda: agenda,
                ownerPresent: false),
            state: state)
        #expect(late.state.mustDoDays == ["2026-09-30": false])
    }
}

struct DayStateStoreTests {

    @Test @MainActor func aSavedDayComesBackAndOldFilesStillLoad() throws {
        let url = makeTempDir("day-state").appendingPathComponent("day-state.json")
        let store = DayStateStore(url: url)
        var state = DayState(day: DayKey(rawValue: "2026-09-30"))
        state.mustDoID = "R1"
        state.carryOverForNextDay = "Start with the spec."
        store.save(state)
        #expect(store.load() == state)

        // A file written before the ledger and agents existed.
        try Data(#"{"day": "2026-09-30", "mustDoID": "R2"}"#.utf8).write(to: url)
        let old = try #require(store.load())
        #expect(old.mustDoID == "R2")
        #expect(old.agents.isEmpty)
    }
}
