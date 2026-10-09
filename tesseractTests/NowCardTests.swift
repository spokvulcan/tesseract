//
//  NowCardTests.swift
//  tesseractTests
//
//  The Now Card from the Timeline's fixture day: the meeting the owner is
//  in, the task whose slot is now, a task that slid and the slot offered for
//  it, free time and what fits in it, the next step, what's left, the
//  evening closing the day, a done day and how tomorrow starts, and the plan
//  and wrap-up offers. Also the
//  Inbox's slot offer and how long a card's line stays fresh.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct NowCardTests {

    static func local(_ day: Int, _ hour: Int, _ minute: Int = 0) -> Date {
        TimelineBuilderTests.local(day, hour, minute)
    }

    static func context(
        companionOn: Bool = true, planned: Bool = true, wrappedUp: Bool = false, inbox: Int = 0
    ) -> NowCardBuilder.Context {
        NowCardBuilder.Context(
            companionOn: companionOn, planned: planned, wrappedUp: wrappedUp,
            eveningMinutes: 21 * 60, inboxCount: inbox)
    }

    static func card(_ facts: DayFacts, _ context: NowCardBuilder.Context = context()) -> NowCard {
        NowCardBuilder.build(
            timeline: TimelineBuilder.build(facts: facts), facts: facts, context: context)
    }

    static func day(now: Date, mustDo: String? = nil) -> DayFacts {
        TimelineBuilderTests.facts(now: now, mustDo: mustDo)
    }

    @Test func inAMeetingItSaysHowMuchIsLeftAndWhatsNext() {
        let card = Self.card(Self.day(now: Self.local(30, 9, 40)))
        #expect(card.headline == "Standup")
        #expect(card.detail == "20 min left, until 10:00 · then Call the dentist at 11:00")
        #expect(card.actions.isEmpty)
        #expect(card.span?.end == Self.local(30, 10))
    }

    @Test func aTaskWhoseSlotIsNowIsOneClickFromDone() {
        let card = Self.card(Self.day(now: Self.local(30, 11, 5)))
        #expect(card.headline == "Call the dentist")
        #expect(card.detail == "10 min left, until 11:15 · then 1:1 at 13:00")
        #expect(card.actions.map(\.kind) == [.complete(reminderID: "dentist")])
        #expect(card.span == DateInterval(start: Self.local(30, 11), end: Self.local(30, 11, 15)))
    }

    @Test func theTimeLeftRoundsUpToTheMinute() {
        // 09:59:30: half a minute of the standup is left, said as a minute.
        let now = Self.local(30, 9, 59).addingTimeInterval(30)
        let card = Self.card(Self.day(now: now))
        #expect(card.detail?.hasPrefix("1 min left, until 10:00") == true)
    }

    @Test func anEventInPersonSaysWhenToLeave() {
        let leave = Departure(
            eventID: "E2", title: "1:1", at: Self.local(30, 12, 45),
            eventStart: Self.local(30, 13))
        let facts = { (now: Date) in
            var facts = DayFacts(
                now: now, events: [TimelineBuilderTests.standup, TimelineBuilderTests.oneOnOne])
            facts.departures = [leave]
            return facts
        }
        let context = Self.context()
        // Free until it's time to leave, not until the 1:1.
        let free = Self.card(facts(Self.local(30, 11, 30)), context)
        #expect(free.headline == "Free until 12:45, when you leave for 1:1")
        // Too close to fit anything: the next step is leaving.
        let soon = Self.card(facts(Self.local(30, 12, 35)), context)
        #expect(soon.headline == "1:1")
        #expect(soon.detail == "Leave at 12:45, in 10 min. It starts at 13:00.")
        let now = Self.card(facts(Self.local(30, 12, 46)), context)
        #expect(now.detail == "Time to leave. It starts at 13:00.")
        // A task still running doesn't hide it.
        var busy = facts(Self.local(30, 12, 50))
        busy.undated = [
            AgendaReminder(id: "gym", title: "Gym", listID: "health", listTitle: "Health")
        ]
        busy.plan = [Placement(reminderID: "gym", start: Self.local(30, 12, 15), minutes: 45)]
        #expect(Self.card(busy, context).detail == "Time to leave. It starts at 13:00.")
        // And the way there is no free time to offer.
        #expect(
            TimelineBuilder.firstFreeSlot(minutes: 30, facts: facts(Self.local(30, 12, 20)))
                .map { $0 >= Self.local(30, 13, 45) } != false)
    }

    @Test func freeTimeHasNoSpan() {
        #expect(Self.card(Self.day(now: Self.local(30, 12))).span == nil)
    }

    @Test func aTaskThatSlidIsOfferedTheNextFreeSlot() {
        let card = Self.card(Self.day(now: Self.local(30, 12)))
        #expect(card.headline == "Call the dentist")
        #expect(card.detail == "Slid past 11:00.")
        #expect(card.actions.map(\.title) == ["Do it at 12:00", "Done", "Tomorrow"])
        #expect(
            card.actions.first?.kind
                == .place(reminderID: "dentist", start: Self.local(30, 12), minutes: 15))
    }

    @Test func inTheEveningTheDayClosesInsteadOfASlidTask() {
        let card = Self.card(Self.day(now: Self.local(30, 21, 30)))
        #expect(card.headline == "1 of 3 done today.")
        #expect(card.detail == "Still open: Call the dentist, Pay rent.")
        #expect(card.actions.map(\.title) == ["Wrap up the day"])
    }

    @Test func onceWrappedUpTheEveningLooksAtTomorrow() {
        let card = Self.card(Self.day(now: Self.local(30, 23, 50)), Self.context(wrappedUp: true))
        #expect(card.headline == "1 of 3 done today.")
        #expect(card.detail == "Nothing on tomorrow yet.")
        #expect(card.actions.isEmpty)
    }

    @Test func aStepStillAheadTonightComesFirst() {
        var facts = Self.day(now: Self.local(30, 21, 30))
        facts.dueOrOverdue.append(
            AgendaReminder(
                id: "lesson", title: "Language lesson", listID: "life", listTitle: "Life",
                due: Self.local(30, 22), dueHasTime: true))
        let card = Self.card(facts)
        #expect(card.headline == "Language lesson")
        #expect(card.detail == "At 22:00, in 30 min.")
        #expect(card.actions.map(\.title) == ["Start now", "Done", "Wrap up the day"])
    }

    @Test func freeTimeOffersTheMustDoFirst() {
        let card = Self.card(Self.day(now: Self.local(30, 10, 30), mustDo: "gym"))
        #expect(card.headline == "Gym")
        #expect(card.detail == "Your must-do. You're free until 11:00.")
        #expect(
            card.actions.map(\.kind) == [
                .place(reminderID: "gym", start: Self.local(30, 10, 30), minutes: 15),
                .complete(reminderID: "gym"),
            ])
        #expect(card.actions.map(\.title) == ["Start now", "Done"])
    }

    @Test func onceTheMustDoIsDoneTheRestIsABonus() {
        // Answering mail was the must-do, and it is done.
        let free = Self.card(Self.day(now: Self.local(30, 10, 30), mustDo: "mail"))
        #expect(free.headline == "Pay rent")
        #expect(free.detail == "The must-do is done; this one's a bonus. You're free until 11:00.")
        let evening = Self.card(Self.day(now: Self.local(30, 21, 30), mustDo: "mail"))
        #expect(evening.headline == "1 of 3 done today, the must-do among them.")
    }

    @Test func freeTimeOffersTodaysOpenTaskBeforeACarriedOne() {
        let card = Self.card(Self.day(now: Self.local(30, 10, 30)))
        #expect(card.headline == "Pay rent")
        #expect(card.detail == "You're free until 11:00.")
    }

    @Test func freeTimeWithNothingToFitSaysUntilWhenAndWhatFollows() {
        let facts = DayFacts(
            now: Self.local(30, 10, 30),
            events: [TimelineBuilderTests.standup, TimelineBuilderTests.oneOnOne])
        let card = Self.card(facts, Self.context(inbox: 2))
        #expect(card.headline == "Free until 13:00")
        #expect(card.detail == "Then 1:1. 2 Inbox items could use a time.")
        #expect(card.actions.isEmpty)
    }

    @Test func withNoRoomToSpareTheNextStepTakesTheCard() {
        // Twenty minutes before the standup: too short to count as free.
        let card = Self.card(Self.day(now: Self.local(30, 9, 10)))
        #expect(card.headline == "Standup")
        #expect(card.detail == "At 09:30, in 20 min.")
    }

    @Test func aTaskComingUpCanStartEarly() {
        // Twenty minutes before the dentist: no free half hour, so the call is next.
        let card = Self.card(Self.day(now: Self.local(30, 10, 40)))
        #expect(card.headline == "Call the dentist")
        #expect(card.detail == "At 11:00, in 20 min.")
        #expect(
            card.actions.map(\.kind) == [
                .place(reminderID: "dentist", start: Self.local(30, 10, 40), minutes: 15),
                .complete(reminderID: "dentist"),
            ])
    }

    @Test func pastTheDaysEndItCountsWhatIsLeft() {
        let facts = DayFacts(
            now: Self.local(30, 22, 30),
            dueOrOverdue: [
                AgendaReminder(
                    id: "rent", title: "Pay rent", listID: "life", listTitle: "Life",
                    due: Self.local(30, 0)),
                AgendaReminder(
                    id: "laundry", title: "Do the laundry", listID: "life", listTitle: "Life",
                    due: Self.local(30, 0)),
            ])
        let card = Self.card(facts)
        #expect(card.headline == "2 left for today")
        #expect(card.detail == "Do the laundry, Pay rent.")
        #expect(card.actions.map(\.kind) == [.wrapUp])
    }

    @Test func aDoneDayLooksAtTomorrow() {
        let work = AgendaEvent(
            id: "W", title: "Work", start: Self.local(31, 9), end: Self.local(31, 13),
            calendarID: "c", calendarTitle: "Work")
        let facts = DayFacts(
            now: Self.local(30, 20), events: [TimelineBuilderTests.standup, work],
            doneToday: [
                AgendaReminder(
                    id: "mail", title: "Answer mail", listID: "work", listTitle: "Work",
                    due: Self.local(30, 0), isCompleted: true, completedAt: Self.local(30, 8))
            ])
        let card = Self.card(facts)
        #expect(card.headline == "All done for today.")
        #expect(card.detail == "Next: Work, tomorrow at 09:00.")
        // Past midnight it is still the same day, and tomorrow still starts
        // with work — now within half a day, it says how far off.
        var night = facts
        night.now = Self.local(31, 0, 40)
        #expect(Self.card(night).headline == "All done for today.")
        #expect(
            Self.card(night).detail == "Next: Work, tomorrow at 09:00 — 8 h 20 min from now.")
        // With nothing on tomorrow, it says so.
        var free = facts
        free.events = [TimelineBuilderTests.standup]
        #expect(Self.card(free).detail == "Nothing on tomorrow yet.")
    }

    @Test func aClearDayInvitesACapture() {
        let card = Self.card(DayFacts(now: Self.local(30, 10)))
        #expect(card.headline == "A clear day.")
        #expect(card.actions.isEmpty)
    }

    @Test func theDayIsOfferedAPlanUntilItHasOne() {
        let facts = Self.day(now: Self.local(30, 8))
        #expect(Self.card(facts, Self.context(planned: false)).actions.last?.kind == .planDay)
        #expect(!Self.card(facts).actions.contains { $0.kind == .planDay })
        // With Jarvis off, nothing would make the plan.
        #expect(
            !Self.card(facts, Self.context(companionOn: false, planned: false)).actions.contains {
                $0.kind == .planDay
            })
    }

    @Test func theInboxIsOfferedTodaysNextSlotThenTomorrow() {
        let morning = Self.day(now: Self.local(30, 10, 7))
        #expect(InboxSlot.suggest(facts: morning, evening: false) == .today(Self.local(30, 10, 15)))
        #expect(InboxSlot.suggest(facts: morning, evening: true) == .tomorrow)
        // Ten minutes before the day's end: no half hour left.
        let late = Self.day(now: Self.local(30, 21, 50))
        #expect(InboxSlot.suggest(facts: late, evening: false) == .tomorrow)
    }

    @Test func eachInboxItemIsOfferedItsOwnSlot() {
        var facts = Self.day(now: Self.local(30, 10, 7))
        facts.undated.append(
            AgendaReminder(id: "mom", title: "Call mom", listID: "inbox", listTitle: "Inbox"))
        let slots = InboxSlot.suggest(for: ["gym", "mom"], facts: facts, evening: false)
        #expect(slots["gym"] == .today(Self.local(30, 10, 15)))
        // 10:45 to the dentist at 11:00 is too short; after his call it fits.
        #expect(slots["mom"] == .today(Self.local(30, 11, 15)))
    }

    @Test func theInboxKeepsClearOfTheCardsOwnOffer() {
        // Rent was planned for 10:30 and slid, and so did the dentist at
        // 11:00: the card fits both in from 12:00.
        let facts = TimelineBuilderTests.facts(
            now: Self.local(30, 12),
            plan: [Placement(reminderID: "rent", start: Self.local(30, 10, 30), minutes: 15)])
        let card = Self.card(facts)
        #expect(card.headline == "Pay rent")
        #expect(card.detail == "Slid past 10:30. One more slid too.")
        #expect(card.actions.map(\.title) == ["Fit both in", "Done", "Tomorrow"])
        #expect(
            card.offeredPlacements == [
                Placement(reminderID: "rent", start: Self.local(30, 12), minutes: 15),
                Placement(reminderID: "dentist", start: Self.local(30, 12, 15), minutes: 15),
            ])
        // The offer replaces their old slots, so the gym goes after them.
        let slots = InboxSlot.suggest(for: ["gym"], facts: card.reserving(facts), evening: false)
        #expect(slots["gym"] == .today(Self.local(30, 12, 30)))
    }

    @Test func severalTasksThatSlidAreFittedIntoTheDayInOneClick() {
        // At 12:30, rent (10:30), the gym (an hour from 10:45) and the
        // dentist (11:00) have all slid; half an hour is free before the 1:1.
        let facts = TimelineBuilderTests.facts(
            now: Self.local(30, 12, 30),
            plan: [
                Placement(reminderID: "rent", start: Self.local(30, 10, 30), minutes: 15),
                Placement(reminderID: "gym", start: Self.local(30, 10, 45), minutes: 60),
            ])
        let card = Self.card(facts)
        #expect(card.detail == "Slid past 10:30. 2 more slid too.")
        #expect(card.actions.first?.title == "Fit all 3 in")
        // Each at the next free slot after the ones before it: rent now, the
        // gym after the 1:1, the dentist after the gym (the quarter hour left
        // before the 1:1 is too short to count as free).
        #expect(
            card.offeredPlacements == [
                Placement(reminderID: "rent", start: Self.local(30, 12, 30), minutes: 15),
                Placement(reminderID: "gym", start: Self.local(30, 13, 45), minutes: 60),
                Placement(reminderID: "dentist", start: Self.local(30, 14, 45), minutes: 15),
            ])
        // One yes puts them all in the plan, traced as one decision.
        var state = DayState(day: DayKey(rawValue: "2026-09-30"))
        state.plan = facts.plan
        let fitted = DayEngine.decide(
            .cardAction(.placeAll(card.offeredPlacements)),
            snapshot: DaySnapshot(
                now: Self.local(30, 12, 30), settings: DaySettings(), agenda: .empty,
                ownerPresent: true),
            state: state)
        #expect(fitted.state.plan.map(\.reminderID) == ["rent", "gym", "dentist"])
        #expect(
            fitted.effects.contains(.trace(.cardReaction, ["action": "fitted", "count": .int(3)])))
        // Rent's slot starts now: it is under way.
        #expect(fitted.state.startedSteps.contains(StepCue.key(fitted.state.plan[0])))
    }

    @Test func aPlansLineGoesStaleButTheEveningsHolds() {
        let plan = DayCard(
            id: "p", kind: .morningPlan, createdAt: Self.local(30, 8), isFallback: false,
            body: .morningPlan(
                MorningPlanCard(line: "Hi", mustDoID: nil, placements: [], suggestions: [])))
        #expect(plan.isFresh(at: Self.local(30, 11)))
        #expect(!plan.isFresh(at: Self.local(30, 12, 30)))
        let reflection = DayCard(
            id: "r", kind: .nightReflection, createdAt: Self.local(30, 21), isFallback: false,
            body: .reflection(ReflectionCard(carryOver: "Night", tomorrow: [], proposals: [])))
        #expect(reflection.isFresh(at: Self.local(31, 2)))
    }

    @Test func aBreakpointWithNothingForTheOwnerLeavesThePlansWordStanding() {
        let plan = DayCard(
            id: "plan", kind: .morningPlan, createdAt: Self.local(30, 8), isFallback: false,
            body: .morningPlan(
                MorningPlanCard(line: "A calm day.", mustDoID: nil, placements: [], suggestions: [])
            ))
        func breakpoint(needsYou: [WaitingItem]) -> DayCard {
            DayCard(
                id: "back", kind: .breakpoint, createdAt: Self.local(30, 10), isFallback: false,
                body: .breakpoint(
                    BreakpointCard(
                        awayFrom: Self.local(30, 9, 30), awayUntil: Self.local(30, 10),
                        line: "Nothing needs you right now.", needsYou: needsYou, next: [],
                        whereYouWere: "Xcode",
                        canWait: [QuietGroup(app: "YouTube", lines: ["A new video"])])))
        }
        let at = Self.local(30, 10, 5)
        let quiet = breakpoint(needsYou: [])
        #expect(DayCard.word(in: [plan, quiet], at: at)?.id == "plan")
        let anna = WaitingItem(
            id: "n1", kind: .notification, title: "Anna", detail: "The deck?", app: "Slack")
        #expect(DayCard.word(in: [plan, breakpoint(needsYou: [anna])], at: at)?.id == "back")
        // Dismissed or stale, the word goes; a quiet card never stands in.
        var closed = plan
        closed.dismissed = true
        #expect(DayCard.word(in: [closed, quiet], at: at) == nil)
        #expect(DayCard.word(in: [plan, quiet], at: Self.local(30, 12, 30)) == nil)
    }
}
