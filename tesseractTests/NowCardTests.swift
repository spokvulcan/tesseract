//
//  NowCardTests.swift
//  tesseractTests
//
//  The Now Card, the decision table on the Timeline, from the fixture day's
//  snapshot and state: the step the day is on (the meeting the owner is in,
//  the step they started, a task whose slot is now), a task that slid and
//  the slot offered for it, free time and what fits in it, the next step,
//  what's left, the evening closing the day, a done day and how tomorrow
//  starts, and the plan and wrap-up offers. Also the Inbox's slot offers
//  and how long a card's line stays fresh.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct NowCardTests {

    static func local(_ day: Int, _ hour: Int, _ minute: Int = 0) -> Date {
        DayFixture.local(day, hour, minute)
    }

    /// The card on the fixture day at `now`.
    static func card(now: Date, plan: [Step] = [], mustDo: String? = nil) -> NowCard {
        NowCardBuilder.build(
            DayFixture.day(now: now, plan: plan, mustDo: mustDo), companionOn: true)
    }

    /// The card over `snapshot`, with the fixture day's state unless given.
    static func card(
        _ snapshot: DaySnapshot, _ state: DayState = DayFixture.state(), companionOn: Bool = true
    ) -> NowCard {
        NowCardBuilder.build(
            Timeline(snapshot: snapshot, state: state), companionOn: companionOn)
    }

    static func events(_ events: [AgendaEvent], at now: Date) -> DaySnapshot {
        DayFixture.snapshot(now: now, events: events, open: [], done: [])
    }

    @Test func inAMeetingItSaysHowMuchIsLeftAndWhatsNext() {
        let card = Self.card(now: Self.local(30, 9, 40))
        #expect(card.headline == "Standup")
        #expect(card.detail == "20 min left, until 10:00 · then Call the dentist at 11:00")
        #expect(card.actions.isEmpty)
        #expect(card.span?.end == Self.local(30, 10))
    }

    @Test func aStartedStepSaysHowMuchIsLeftAndIsOneClickFromDone() {
        let started = Step("dentist", at: Self.local(30, 11), minutes: 15)
            .started(at: Self.local(30, 11))
        let card = Self.card(now: Self.local(30, 11, 5), plan: [started])
        #expect(card.headline == "Call the dentist")
        #expect(card.detail == "10 min left, until 11:15 · then 1:1 at 13:00")
        #expect(card.actions.map(\.kind) == [.complete(reminderID: "dentist")])
        #expect(card.span == DateInterval(start: Self.local(30, 11), end: Self.local(30, 11, 15)))
    }

    @Test func aStepWhoseTimeIsNowButNotStartedIsOneClickFromStarting() {
        // Its cue was closed, missed, or answered by mistake: Today starts it.
        let card = Self.card(now: Self.local(30, 11, 5))
        #expect(card.headline == "Call the dentist")
        #expect(card.detail == "11:00–11:15 · then 1:1 at 13:00")
        #expect(
            card.actions.map(\.kind) == [
                .place(reminderID: "dentist", start: Self.local(30, 11, 5), minutes: 15),
                .complete(reminderID: "dentist"),
            ])
        #expect(card.actions.first?.title == "Start now")
        // Nothing is running: no time draining.
        #expect(card.span == nil)
    }

    @Test func theMustDoWearsItsStarAndSaysWhenItSlid() {
        // The dentist (11:00–11:15) is the must-do.
        let now = Self.card(now: Self.local(30, 11, 5), mustDo: "dentist")
        #expect(now.isMustDo)
        let slid = Self.card(now: Self.local(30, 11, 30), mustDo: "dentist")
        #expect(slid.headline == "Call the dentist")
        #expect(slid.detail == "Your must-do slid past 11:00.")
        #expect(slid.isMustDo)
        // Another task that slid says only that.
        let other = Self.card(now: Self.local(30, 11, 30))
        #expect(other.detail == "Slid past 11:00.")
        #expect(!other.isMustDo)
    }

    @Test func aCallUnderWayOrAboutToStartIsOneClickToJoin() throws {
        let link = try #require(URL(string: "https://meet.google.com/abc-defg-hij"))
        let call = AgendaEvent(
            id: "sync", title: "Sync", start: Self.local(30, 15), end: Self.local(30, 15, 30),
            calendarID: "c", calendarTitle: "Work",
            notes: "Join with Google Meet: https://meet.google.com/abc-defg-hij",
            hasOtherAttendees: true)
        func card(_ hour: Int, _ minute: Int) -> NowCard {
            Self.card(Self.events([call], at: Self.local(30, hour, minute)))
        }
        #expect(card(15, 10).actions.map(\.kind) == [.join(link)])
        #expect(card(14, 50).headline == "Sync")
        #expect(card(14, 50).actions.map(\.kind) == [.join(link)])
        // Hours ahead, nothing to join yet.
        #expect(!card(12, 0).actions.contains { $0.kind == .join(link) })
    }

    @Test func theTimeLeftRoundsUpToTheMinute() {
        // 09:59:30: half a minute of the standup is left, said as a minute.
        let card = Self.card(now: Self.local(30, 9, 59).addingTimeInterval(30))
        #expect(card.detail?.hasPrefix("1 min left, until 10:00") == true)
    }

    @Test func anEventInPersonSaysWhenToLeave() {
        var state = DayFixture.state()
        state.departures = [
            Departure(
                eventID: "E2", title: "1:1", at: Self.local(30, 12, 45),
                eventStart: Self.local(30, 13))
        ]
        let events = [DayFixture.standup, DayFixture.oneOnOne]
        // Free until it's time to leave, not until the 1:1.
        let free = Self.card(Self.events(events, at: Self.local(30, 11, 30)), state)
        #expect(free.headline == "Free until 12:45, when you leave for 1:1")
        // Too close to fit anything: the next step is leaving.
        let soon = Self.card(Self.events(events, at: Self.local(30, 12, 35)), state)
        #expect(soon.headline == "1:1")
        #expect(soon.detail == "Leave at 12:45, in 10 min. It starts at 13:00.")
        let now = Self.card(Self.events(events, at: Self.local(30, 12, 46)), state)
        #expect(now.detail == "Time to leave. It starts at 13:00.")
        // A task still running doesn't hide it.
        var busy = state
        busy.plan = [Step("gym", at: Self.local(30, 12, 15), minutes: 45)]
        let gym = DayFixture.snapshot(
            now: Self.local(30, 12, 50), events: events, open: [DayFixture.gym], done: [])
        #expect(Self.card(gym, busy).detail == "Time to leave. It starts at 13:00.")
        // And the way there is no free time to offer.
        let early = Timeline(
            snapshot: Self.events(events, at: Self.local(30, 12, 20)), state: state)
        #expect(early.firstFreeSlot(minutes: 30) == Self.local(30, 13, 45))
    }

    @Test func freeTimeHasNoSpan() {
        #expect(Self.card(now: Self.local(30, 12)).span == nil)
    }

    @Test func aTaskThatSlidIsOfferedTheNextFreeSlot() {
        let card = Self.card(now: Self.local(30, 12))
        #expect(card.headline == "Call the dentist")
        #expect(card.detail == "Slid past 11:00.")
        #expect(card.actions.map(\.title) == ["Do it at 12:00", "Done", "Tomorrow"])
        #expect(
            card.actions.first?.kind
                == .place(reminderID: "dentist", start: Self.local(30, 12), minutes: 15))
    }

    @Test func inTheEveningTheDayClosesInsteadOfASlidTask() {
        let card = Self.card(now: Self.local(30, 21, 30))
        #expect(card.headline == "1 of 3 done today.")
        #expect(card.detail == "Still open: Call the dentist, Pay rent.")
        #expect(card.actions.map(\.title) == ["Wrap up the day"])
    }

    @Test func onceWrappedUpTheEveningLooksAtTomorrow() {
        var state = DayFixture.state()
        state.eveningWrapUpAt = Self.local(30, 21)
        let card = Self.card(DayFixture.snapshot(now: Self.local(30, 23, 50)), state)
        #expect(card.headline == "1 of 3 done today.")
        #expect(card.detail == "Nothing on tomorrow yet.")
        #expect(card.actions.isEmpty)
    }

    @Test func aStepStillAheadTonightComesFirst() {
        let lesson = AgendaReminder(
            id: "lesson", title: "Language lesson", listID: "life", listTitle: "Life",
            due: Self.local(30, 22), dueHasTime: true)
        let card = Self.card(
            DayFixture.snapshot(
                now: Self.local(30, 21, 30),
                open: [
                    DayFixture.dentist, DayFixture.rent, DayFixture.passport, DayFixture.gym,
                    lesson,
                ]))
        #expect(card.headline == "Language lesson")
        #expect(card.detail == "At 22:00, in 30 min.")
        #expect(card.actions.map(\.title) == ["Start now", "Done", "Wrap up the day"])
    }

    @Test func freeTimeOffersTheMustDoFirst() {
        let card = Self.card(now: Self.local(30, 10, 30), mustDo: "gym")
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
        let free = Self.card(now: Self.local(30, 10, 30), mustDo: "mail")
        #expect(free.headline == "Pay rent")
        #expect(free.detail == "The must-do is done; this one's a bonus. You're free until 11:00.")
        let evening = Self.card(now: Self.local(30, 21, 30), mustDo: "mail")
        #expect(evening.headline == "1 of 3 done today, the must-do among them.")
    }

    @Test func freeTimeOffersTodaysOpenTaskBeforeACarriedOne() {
        let card = Self.card(now: Self.local(30, 10, 30))
        #expect(card.headline == "Pay rent")
        #expect(card.detail == "You're free until 11:00.")
    }

    @Test func freeTimeWithNothingToFitSaysUntilWhenAndWhatFollows() {
        // Two captures wait in the Inbox, undated: neither is today's task.
        let inbox = ["Call mom", "Order a new bike chain"].enumerated().map {
            AgendaReminder(
                id: "inbox\($0.offset)", title: $0.element, listID: "inbox", listTitle: "Inbox")
        }
        let card = Self.card(
            DayFixture.snapshot(
                now: Self.local(30, 10, 30), events: [DayFixture.standup, DayFixture.oneOnOne],
                open: inbox, done: []))
        #expect(card.headline == "Free until 13:00")
        #expect(card.detail == "Then 1:1. 2 Inbox items could use a time.")
        #expect(card.actions.isEmpty)
    }

    @Test func withNoRoomToSpareTheNextStepTakesTheCard() {
        // Twenty minutes before the standup: too short to count as free.
        let card = Self.card(now: Self.local(30, 9, 10))
        #expect(card.headline == "Standup")
        #expect(card.detail == "At 09:30, in 20 min.")
    }

    @Test func aTaskComingUpCanStartEarly() {
        // Twenty minutes before the dentist: no free half hour, so the call is next.
        let card = Self.card(now: Self.local(30, 10, 40))
        #expect(card.headline == "Call the dentist")
        #expect(card.detail == "At 11:00, in 20 min.")
        #expect(
            card.actions.map(\.kind) == [
                .place(reminderID: "dentist", start: Self.local(30, 10, 40), minutes: 15),
                .complete(reminderID: "dentist"),
            ])
    }

    @Test func pastTheDaysEndItCountsWhatIsLeft() {
        let laundry = AgendaReminder(
            id: "laundry", title: "Do the laundry", listID: "life", listTitle: "Life",
            due: Self.local(30, 0))
        let card = Self.card(
            DayFixture.snapshot(
                now: Self.local(30, 22, 30), events: [], open: [DayFixture.rent, laundry],
                done: []))
        #expect(card.headline == "2 left for today")
        #expect(card.detail == "Do the laundry, Pay rent.")
        #expect(card.actions.map(\.kind) == [.wrapUp])
    }

    @Test func aDoneDayLooksAtTomorrow() {
        let work = AgendaEvent(
            id: "W", title: "Work", start: Self.local(31, 9), end: Self.local(31, 13),
            calendarID: "c", calendarTitle: "Work")
        func done(at now: Date, events: [AgendaEvent] = [DayFixture.standup, work]) -> NowCard {
            Self.card(DayFixture.snapshot(now: now, events: events, open: []))
        }
        let card = done(at: Self.local(30, 20))
        #expect(card.headline == "All done for today.")
        #expect(card.detail == "Next: Work, tomorrow at 09:00.")
        // Past midnight it is still the same day, and tomorrow still starts
        // with work — now within half a day, it says how far off.
        let night = done(at: Self.local(31, 0, 40))
        #expect(night.headline == "All done for today.")
        #expect(night.detail == "Next: Work, tomorrow at 09:00 — 8 h 20 min from now.")
        // With nothing on tomorrow, it says so.
        #expect(
            done(at: Self.local(30, 20), events: [DayFixture.standup]).detail
                == "Nothing on tomorrow yet.")
    }

    @Test func aClearDayInvitesACapture() {
        let card = Self.card(Self.events([], at: Self.local(30, 10)))
        #expect(card.headline == "A clear day.")
        #expect(card.actions.isEmpty)
    }

    @Test func theDayIsOfferedAPlanUntilItHasOne() {
        let snapshot = DayFixture.snapshot(now: Self.local(30, 8))
        var unplanned = DayFixture.state()
        unplanned.morningPlanAt = nil
        #expect(Self.card(snapshot, unplanned).actions.last?.kind == .planDay)
        #expect(!Self.card(snapshot).actions.contains { $0.kind == .planDay })
        // With Jarvis off, nothing would make the plan.
        #expect(
            !Self.card(snapshot, unplanned, companionOn: false).actions.contains {
                $0.kind == .planDay
            })
    }

    // MARK: The Inbox's offers

    @Test func theInboxIsOfferedTodaysNextSlotThenTomorrow() {
        let morning = DayFixture.day(now: Self.local(30, 10, 7))
        #expect(morning.inboxOffers(for: ["gym"]) == ["gym": .today(Self.local(30, 10, 15))])
        // In the evening, tomorrow: here the evening starts at ten.
        var settings = DaySettings()
        settings.eveningMinutes = 10 * 60
        let evening = Timeline(
            snapshot: DayFixture.snapshot(now: Self.local(30, 10, 7), settings: settings),
            state: DayFixture.state())
        #expect(evening.inboxOffers(for: ["gym"]) == ["gym": .tomorrow])
        // Ten minutes before the day's end: no half hour left.
        let late = DayFixture.day(now: Self.local(30, 21, 50))
        #expect(late.inboxOffers(for: ["gym"]) == ["gym": .tomorrow])
    }

    @Test func eachInboxItemIsOfferedItsOwnSlot() {
        let mom = AgendaReminder(id: "mom", title: "Call mom", listID: "inbox", listTitle: "Inbox")
        let day = Timeline(
            snapshot: DayFixture.snapshot(
                now: Self.local(30, 10, 7),
                open: [
                    DayFixture.dentist, DayFixture.rent, DayFixture.passport, DayFixture.gym, mom,
                ]),
            state: DayFixture.state())
        let slots = day.inboxOffers(for: ["gym", "mom"])
        #expect(slots["gym"] == .today(Self.local(30, 10, 15)))
        // 10:45 to the dentist at 11:00 is too short; after his call it fits.
        #expect(slots["mom"] == .today(Self.local(30, 11, 15)))
    }

    @Test func theInboxKeepsClearOfTheCardsOwnOffer() {
        // Rent was planned for 10:30 and slid, and so did the dentist at
        // 11:00: the card fits both in from 12:00.
        let day = DayFixture.day(
            now: Self.local(30, 12), plan: [Step("rent", at: Self.local(30, 10, 30), minutes: 15)])
        let card = NowCardBuilder.build(day, companionOn: true)
        #expect(card.headline == "Pay rent")
        #expect(card.detail == "Slid past 10:30. One more slid too.")
        #expect(card.actions.map(\.title) == ["Fit both in", "Done", "Tomorrow"])
        #expect(card.actions.first?.help == "Pay rent at 12:00 · Call the dentist at 12:15")
        #expect(
            card.offeredPlacements == [
                Placement(reminderID: "rent", start: Self.local(30, 12), minutes: 15),
                Placement(reminderID: "dentist", start: Self.local(30, 12, 15), minutes: 15),
            ])
        // The offer replaces their old slots, so the gym goes after them.
        let slots = day.inboxOffers(for: ["gym"], reserving: card.offeredPlacements)
        #expect(slots["gym"] == .today(Self.local(30, 12, 30)))
    }

    @Test func severalTasksThatSlidAreFittedIntoTheDayInOneClick() {
        // At 12:30, rent (10:30), the gym (an hour from 10:45) and the
        // dentist (11:00) have all slid; half an hour is free before the 1:1.
        let plan = [
            Step("rent", at: Self.local(30, 10, 30), minutes: 15),
            Step("gym", at: Self.local(30, 10, 45), minutes: 60),
        ]
        let card = Self.card(now: Self.local(30, 12, 30), plan: plan)
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
        let fitted = DayEngine.decide(
            .cardAction(.placeAll(card.offeredPlacements)),
            snapshot: DaySnapshot(
                now: Self.local(30, 12, 30), settings: DaySettings(), agenda: .empty,
                ownerPresent: true),
            state: DayFixture.state(plan: plan))
        #expect(fitted.state.plan.map(\.reminderID) == ["rent", "gym", "dentist"])
        #expect(
            fitted.effects.contains(.trace(.cardReaction, ["action": "fitted", "count": .int(3)])))
        // Rent's slot starts now: it is under way.
        #expect(fitted.state.plan[0].isStarted)
    }

    // MARK: Jarvis's line

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
