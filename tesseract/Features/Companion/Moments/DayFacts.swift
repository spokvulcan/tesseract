//
//  DayFacts.swift
//  tesseract
//
//  The facts a moment is asked about and its card is checked against: the
//  events and tasks of the day, as plain values. The same facts write the
//  request, validate the reply (a card may only name ids it was shown) and
//  build the fallback card when the model fails.
//

import Foundation

nonisolated struct DayFacts: Sendable, Equatable {
    var now: Date
    var calendar: Calendar
    /// Timed and all-day events from the start of today to the end of tomorrow.
    var events: [AgendaEvent]
    /// Open reminders due today or earlier.
    var dueOrOverdue: [AgendaReminder]
    /// Open reminders with no date (Inbox and Areas).
    var undated: [AgendaReminder]
    /// Completed today.
    var doneToday: [AgendaReminder]
    var areas: [Area]
    var inboxListID: String?
    /// Today's must-do and plan, as they stand.
    var mustDoID: String?
    var plan: [Placement]

    init(
        now: Date, calendar: Calendar = .current, events: [AgendaEvent] = [],
        dueOrOverdue: [AgendaReminder] = [], undated: [AgendaReminder] = [],
        doneToday: [AgendaReminder] = [], areas: [Area] = [], inboxListID: String? = nil,
        mustDoID: String? = nil, plan: [Placement] = []
    ) {
        self.now = now
        self.calendar = calendar
        self.events = events
        self.dueOrOverdue = dueOrOverdue
        self.undated = undated
        self.doneToday = doneToday
        self.areas = areas
        self.inboxListID = inboxListID
        self.mustDoID = mustDoID
        self.plan = plan
    }

    /// Build from the Agenda's snapshot.
    init(
        snapshot: AgendaSnapshot, areas: [Area], inboxListID: String?, now: Date,
        calendar: Calendar = .current, mustDoID: String? = nil, plan: [Placement] = []
    ) {
        let endOfToday =
            calendar.date(byAdding: .day, value: 1, to: calendar.startOfDay(for: now)) ?? now
        self.init(
            now: now, calendar: calendar, events: snapshot.events,
            dueOrOverdue: snapshot.open.filter { ($0.due ?? .distantFuture) < endOfToday },
            undated: snapshot.open.filter { $0.due == nil },
            doneToday: snapshot.doneToday, areas: areas, inboxListID: inboxListID,
            mustDoID: mustDoID, plan: plan)
    }

    var startOfToday: Date { calendar.startOfDay(for: now) }
    var endOfToday: Date { calendar.date(byAdding: .day, value: 1, to: startOfToday) ?? now }

    /// Timed events still ahead today.
    var remainingEventsToday: [AgendaEvent] {
        events.filter { !$0.isAllDay && $0.end > now && $0.start < endOfToday }
    }

    /// Today's timed events, past ones included.
    var eventsToday: [AgendaEvent] {
        events.filter { !$0.isAllDay && $0.start < endOfToday && $0.end > startOfToday }
    }

    var tomorrowEvents: [AgendaEvent] {
        let end = calendar.date(byAdding: .day, value: 1, to: endOfToday) ?? endOfToday
        return events.filter { !$0.isAllDay && $0.start >= endOfToday && $0.start < end }
    }

    /// Every open task a card may name.
    var openTasks: [AgendaReminder] { dueOrOverdue + undated }

    func task(_ id: String) -> AgendaReminder? { openTasks.first { $0.id == id } }

    func areaName(of reminder: AgendaReminder) -> String {
        if reminder.listID == inboxListID { return "Inbox" }
        return areas.first { $0.id == reminder.listID }?.name ?? reminder.listTitle
    }

    /// Free time from now to the first remaining event today (or nil when
    /// the next event is less than 10 minutes away or there is none).
    var freeBeforeFirstEvent: DateInterval? {
        guard let first = remainingEventsToday.first(where: { $0.start > now }) else { return nil }
        guard first.start.timeIntervalSince(now) >= 600 else { return nil }
        return DateInterval(start: now, end: first.start)
    }
}
