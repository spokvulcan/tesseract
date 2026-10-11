//
//  DayFacts+Time.swift
//  tesseract
//
//  How the day's time is taken, said once (ADR-0091). An event is a
//  meeting — other people invited, or a place to be — or a block of the
//  owner's own, no one else and nowhere to go ("Work", 09:00–13:00), where
//  steps happen. A time to leave the Morning Plan set stands while its event
//  is still on the calendar under the same id at the same start: moved or
//  cancelled, it no longer applies, and the way there is free again.
//
//  Three readers build on this, each its own way, by name: the Day Line's
//  free time (`Timeline`: every event and open task, and the way to an event
//  in person), a step's boundaries (`DayEngine+Steps`: today's meetings and
//  the times to leave for them, never a block), and a plan's placements
//  (`isClearOfEvents`: off every event, a block included).
//

import Foundation

nonisolated extension AgendaEvent {
    /// A meeting: other people, or a place to be. It ends a step five
    /// minutes before it, and while it runs it is the step the day is on,
    /// over a block. Only one with other people keeps a Step Cue off the
    /// panel (`hasOtherAttendees`).
    var isMeeting: Bool { hasOtherAttendees || !(location ?? "").isEmpty }
}

nonisolated extension Departure {
    /// Its event, while the time to leave still stands: on the calendar,
    /// under the same id, at the start the plan set it for.
    func event(in events: [AgendaEvent]) -> AgendaEvent? {
        events.first { $0.id == eventID && $0.start == eventStart }
    }
}

nonisolated extension DayFacts {

    /// A time to leave that still stands, with its event.
    struct StandingDeparture: Sendable, Equatable {
        var departure: Departure
        var event: AgendaEvent

        /// The way there: from the time to leave until the event starts.
        var wayThere: DateInterval? {
            departure.at < event.start
                ? DateInterval(start: departure.at, end: event.start) : nil
        }
    }

    /// The day's times to leave that still stand.
    var standingDepartures: [StandingDeparture] {
        departures.compactMap { departure in
            departure.event(in: events).map { StandingDeparture(departure: departure, event: $0) }
        }
    }

    /// The way to today's events in person: busy time on the Day Line.
    var waysThereToday: [DateInterval] {
        let today = eventsToday
        return standingDepartures.filter { today.contains($0.event) }.compactMap(\.wayThere)
    }

    /// Today's meetings, past ones included: what a step ends before.
    var meetingsToday: [AgendaEvent] { eventsToday.filter(\.isMeeting) }

    /// The times to leave for today's meetings that still stand: a step
    /// ends before those too.
    var departuresToMeetings: [StandingDeparture] {
        let meetings = meetingsToday
        return standingDepartures.filter { meetings.contains($0.event) }
    }

    /// A plan's placement keeps off every event today, a block of the
    /// owner's own as much as a meeting (touching one is fine): Jarvis plans
    /// around the calendar.
    func isClearOfEvents(_ slot: DateInterval) -> Bool {
        !eventsToday.contains { event in
            let busy = DateInterval(start: event.start, end: event.end)
            return busy.intersects(slot) && busy.end != slot.start && slot.end != busy.start
        }
    }
}
