//
//  NowCardTests.swift
//  tesseractTests
//
//  The Now Card from the Timeline's fixture day: the meeting the owner is
//  in, the task whose slot is now, a task that slid and the slot offered for
//  it, free time and what fits in it, the next step, what's left, a done
//  day and how tomorrow starts, and the plan and wrap-up offers. Also the
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

    @Test func inTheEveningASlidTaskIsOfferedTomorrowAndTheWrapUp() {
        let card = Self.card(Self.day(now: Self.local(30, 21, 30)))
        #expect(card.headline == "Call the dentist")
        #expect(card.actions.map(\.title) == ["Tomorrow", "Done", "Wrap up the day"])
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
        // Past midnight it is still the same day, and tomorrow still starts with work.
        var night = facts
        night.now = Self.local(31, 0, 40)
        #expect(Self.card(night).headline == "All done for today.")
        #expect(Self.card(night).detail == "Next: Work, tomorrow at 09:00.")
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
        // Rent was planned for 10:30 and slid; the card offers it 12:00.
        let facts = TimelineBuilderTests.facts(
            now: Self.local(30, 12),
            plan: [Placement(reminderID: "rent", start: Self.local(30, 10, 30), minutes: 15)])
        let card = Self.card(facts)
        #expect(card.headline == "Pay rent")
        #expect(card.detail == "Slid past 10:30. One more slid too.")
        #expect(
            card.offeredPlacement
                == Placement(reminderID: "rent", start: Self.local(30, 12), minutes: 15))
        // The offer replaces rent's old slot, so the gym goes after it.
        let slots = InboxSlot.suggest(for: ["gym"], facts: card.reserving(facts), evening: false)
        #expect(slots["gym"] == .today(Self.local(30, 12, 15)))
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
        // While Jarvis judges what came in, the card speaks.
        var judging = quiet
        judging.isRefining = true
        #expect(DayCard.word(in: [plan, judging], at: at)?.id == "back")
        // Dismissed or stale, the word goes; a quiet card never stands in.
        var closed = plan
        closed.dismissed = true
        #expect(DayCard.word(in: [closed, quiet], at: at) == nil)
        #expect(DayCard.word(in: [plan, quiet], at: Self.local(30, 12, 30)) == nil)
    }
}
