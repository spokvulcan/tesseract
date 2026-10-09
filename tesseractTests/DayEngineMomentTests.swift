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
            name: "after the morning window", now: local(30, 13), awayFrom: local(29, 23),
            runs: false),
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
