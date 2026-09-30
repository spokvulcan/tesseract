//
//  NudgePlannerTests.swift
//  tesseractTests
//
//  Event nudges, planned purely and handed to the OS by the Day Engine only
//  when the set changes.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct NudgePlannerTests {

    static func local(_ day: Int, _ hour: Int, _ minute: Int = 0) -> Date {
        Calendar.current.date(
            from: DateComponents(year: 2026, month: 9, day: day, hour: hour, minute: minute))!
    }

    static let now = local(30, 9)

    static func event(_ id: String, _ start: Date, minutes: Int = 30, allDay: Bool = false)
        -> AgendaEvent
    {
        AgendaEvent(
            id: id, title: "Event \(id)", start: start,
            end: start.addingTimeInterval(TimeInterval(minutes * 60)), isAllDay: allDay,
            calendarID: "c", calendarTitle: "Home", location: "Room 2")
    }

    @Test func oneNudgePerTimedEventAheadOfItsLead() {
        let events = [
            Self.event("past", Self.local(30, 8)),
            Self.event("tooSoon", Self.local(30, 9, 5)),
            Self.event("allDay", Self.local(30, 0), minutes: 1440, allDay: true),
            Self.event("standup", Self.local(30, 10)),
            Self.event("beyond", Self.local(33, 10)),
        ]
        let nudges = NudgePlanner.plan(events: events, now: Self.now, leadMinutes: 10)
        #expect(nudges.map(\.eventID) == ["standup"])
        #expect(nudges.first?.fireAt == Self.local(30, 9, 50))
        #expect(nudges.first?.body == "In 10 min · 10:00–10:30 · Room 2")
    }

    @Test func aChangedTitleOrLeadGetsANewID() {
        let event = Self.event("x", Self.local(30, 12))
        var renamed = event
        renamed.title = "Renamed"
        let a = NudgePlanner.plan(events: [event], now: Self.now, leadMinutes: 10)
        let b = NudgePlanner.plan(events: [renamed], now: Self.now, leadMinutes: 10)
        let c = NudgePlanner.plan(events: [event], now: Self.now, leadMinutes: 5)
        #expect(a.first?.id != b.first?.id)
        #expect(a.first?.id != c.first?.id)
        #expect(
            a.first?.id
                == NudgePlanner.plan(events: [event], now: Self.now, leadMinutes: 10).first?.id)
    }

    @Test func diffAddsAndWithdraws() {
        let desired = NudgePlanner.plan(
            events: [Self.event("a", Self.local(30, 12)), Self.event("b", Self.local(30, 13))],
            now: Self.now, leadMinutes: 10)
        let (add, remove) = NudgePlanner.diff(
            desired: desired, scheduled: [desired[0].id, "nudge.event.gone.1"])
        #expect(add.map(\.eventID) == ["b"])
        #expect(remove == ["nudge.event.gone.1"])
    }
}

struct DayEngineNudgeTests {

    private func snapshot(events: [AgendaEvent], access: AgendaAccess = .full) -> DaySnapshot {
        var agenda = AgendaSnapshot.empty
        agenda.access = access
        agenda.events = events
        return DaySnapshot(now: NudgePlannerTests.now, settings: DaySettings(), agenda: agenda)
    }

    @Test func aTickSyncsOnceThenStaysQuietUntilTheAgendaChanges() {
        let events = [NudgePlannerTests.event("a", NudgePlannerTests.local(30, 12))]
        let state = DayState(day: DayKey(for: NudgePlannerTests.now))
        let first = DayEngine.decide(.tick, snapshot: snapshot(events: events), state: state)
        guard case .syncNudges(let nudges) = first.effects.first else {
            Issue.record("the first tick must sync nudges")
            return
        }
        #expect(nudges.count == 1)

        let second = DayEngine.decide(.tick, snapshot: snapshot(events: events), state: first.state)
        #expect(second.effects.isEmpty)

        let moved = [NudgePlannerTests.event("a2", NudgePlannerTests.local(30, 14))]
        let third = DayEngine.decide(
            .agendaChanged, snapshot: snapshot(events: moved), state: second.state)
        #expect(third.effects.count == 1)
    }

    @Test func switchingTheCompanionOffWithdrawsEveryNudge() {
        let events = [NudgePlannerTests.event("a", NudgePlannerTests.local(30, 12))]
        let on = DayEngine.decide(
            .tick, snapshot: snapshot(events: events),
            state: DayState(day: DayKey(for: NudgePlannerTests.now)))
        let off = DayEngine.decide(
            .companionDisabled, snapshot: snapshot(events: events), state: on.state)
        #expect(off.effects == [.syncNudges([])])
    }

    @Test func withoutCalendarAccessNothingIsScheduled() {
        let events = [NudgePlannerTests.event("a", NudgePlannerTests.local(30, 12))]
        let decision = DayEngine.decide(
            .tick, snapshot: snapshot(events: events, access: .none),
            state: DayState(day: DayKey(for: NudgePlannerTests.now)))
        #expect(decision.effects.isEmpty)
    }
}
