//
//  DayFacts.swift
//  tesseract
//
//  The facts a moment is asked about and its card is checked against: the
//  events and tasks of the day, as plain values. The same facts write the
//  request, validate the reply (a card may only name ids it was shown) and
//  build the fallback card when the model fails. "Today" is the owner's day,
//  which rolls over at 04:00 (DayKey): after midnight the small hours still
//  belong to the day that is ending.
//

import Foundation

nonisolated struct DayFacts: Sendable, Equatable {
    var now: Date
    var calendar: Calendar
    /// Timed and all-day events from the start of today to the end of tomorrow.
    var events: [AgendaEvent]
    /// Open reminders due today or earlier.
    var dueOrOverdue: [AgendaReminder]
    /// Open reminders due tomorrow.
    var dueTomorrow: [AgendaReminder]
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
        dueOrOverdue: [AgendaReminder] = [], dueTomorrow: [AgendaReminder] = [],
        undated: [AgendaReminder] = [], doneToday: [AgendaReminder] = [], areas: [Area] = [],
        inboxListID: String? = nil, mustDoID: String? = nil, plan: [Placement] = []
    ) {
        self.now = now
        self.calendar = calendar
        self.events = events
        self.dueOrOverdue = dueOrOverdue
        self.dueTomorrow = dueTomorrow
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
        let startOfToday = Self.startOfDay(for: now, calendar: calendar)
        let endOfToday = calendar.date(byAdding: .day, value: 1, to: startOfToday) ?? now
        let endOfTomorrow = calendar.date(byAdding: .day, value: 1, to: endOfToday) ?? endOfToday
        self.init(
            now: now, calendar: calendar, events: snapshot.events,
            dueOrOverdue: snapshot.open.filter { ($0.due ?? .distantFuture) < endOfToday },
            dueTomorrow: snapshot.open.filter { reminder in
                guard let due = reminder.due else { return false }
                return due >= endOfToday && due < endOfTomorrow
            },
            undated: snapshot.open.filter { $0.due == nil },
            doneToday: snapshot.doneToday, areas: areas, inboxListID: inboxListID,
            mustDoID: mustDoID, plan: plan)
    }

    /// Midnight on the owner's day: before 04:00, the previous date's.
    static func startOfDay(for now: Date, calendar: Calendar) -> Date {
        DayKey(for: now, calendar: calendar).date(calendar: calendar)
            ?? calendar.startOfDay(for: now)
    }

    var startOfToday: Date { Self.startOfDay(for: now, calendar: calendar) }
    var endOfToday: Date { calendar.date(byAdding: .day, value: 1, to: startOfToday) ?? now }
    var endOfTomorrow: Date {
        calendar.date(byAdding: .day, value: 1, to: endOfToday) ?? endOfToday
    }

    /// Timed events still ahead today.
    var remainingEventsToday: [AgendaEvent] {
        let end = endOfToday
        return events.filter { !$0.isAllDay && $0.end > now && $0.start < end }
    }

    /// Today's timed events, past ones included.
    var eventsToday: [AgendaEvent] {
        let (start, end) = (startOfToday, endOfToday)
        return events.filter { !$0.isAllDay && $0.start < end && $0.end > start }
    }

    var tomorrowEvents: [AgendaEvent] {
        let (start, end) = (endOfToday, endOfTomorrow)
        return events.filter { !$0.isAllDay && $0.start >= start && $0.start < end }
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
