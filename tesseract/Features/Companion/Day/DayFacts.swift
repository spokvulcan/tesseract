//
//  DayFacts.swift
//  tesseract
//
//  The facts a moment is asked about and its card is checked against: the
//  events and tasks of the day, as plain values, read once from the
//  Agenda's snapshot and the Day Engine's state (`init(snapshot:state:)`,
//  the one projection of the two). The same facts write the request,
//  validate the reply (a card may only name ids it was shown), build the
//  fallback card when the model fails, and underlie the Timeline. "Today" is
//  the owner's day, which rolls over at 04:00 (DayKey): after midnight the
//  small hours still belong to the day that is ending.
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
    /// Completed in the last seven days, today included.
    var doneThisWeek: [AgendaReminder]
    /// The week's one focus, set by the last week's look-back.
    var weekFocus: String?
    /// The last days' must-dos, today's included: done or not, by day.
    var mustDoDays: [String: Bool] = [:]
    /// Last night the owner was at the Mac past midnight, until then: the
    /// day's plan is asked to stay light.
    var upLateUntil: Date?
    /// When to leave for the day's events in person: the way there is busy.
    var departures: [Departure] = []
    /// How long the owner's started steps took against their plan, lately.
    var stepRuns: [StepRun] = []

    /// Actual over planned minutes across at least five recent steps: how
    /// far the owner's days run past (or short of) their plans.
    var stepPace: (ratio: Double, count: Int)? {
        let planned = stepRuns.reduce(0) { $0 + $1.planned }
        guard stepRuns.count >= 5, planned > 0 else { return nil }
        return (Double(stepRuns.reduce(0) { $0 + $1.actual }) / Double(planned), stepRuns.count)
    }

    init(
        now: Date, calendar: Calendar = .current, events: [AgendaEvent] = [],
        dueOrOverdue: [AgendaReminder] = [], dueTomorrow: [AgendaReminder] = [],
        undated: [AgendaReminder] = [], doneToday: [AgendaReminder] = [], areas: [Area] = [],
        inboxListID: String? = nil, mustDoID: String? = nil, plan: [Placement] = [],
        doneThisWeek: [AgendaReminder] = [], weekFocus: String? = nil
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
        self.doneThisWeek = doneThisWeek
        self.weekFocus = weekFocus
    }

    /// The day as the loop gathered it (`snapshot`: the Agenda's snapshot,
    /// the Areas, the clock) and the engine keeps it (`state`: the must-do,
    /// the plan, the departures, the week).
    init(snapshot: DaySnapshot, state: DayState) {
        let (now, calendar, agenda) = (snapshot.now, snapshot.calendar, snapshot.agenda)
        let startOfToday = Self.startOfDay(for: now, calendar: calendar)
        let endOfToday = calendar.date(byAdding: .day, value: 1, to: startOfToday) ?? now
        let endOfTomorrow = calendar.date(byAdding: .day, value: 1, to: endOfToday) ?? endOfToday
        self.init(
            now: now, calendar: calendar, events: agenda.events,
            dueOrOverdue: agenda.open.filter { ($0.due ?? .distantFuture) < endOfToday },
            dueTomorrow: agenda.open.filter { reminder in
                guard let due = reminder.due else { return false }
                return due >= endOfToday && due < endOfTomorrow
            },
            undated: agenda.open.filter { $0.due == nil },
            doneToday: agenda.doneToday, areas: snapshot.areas,
            inboxListID: snapshot.inboxListID, mustDoID: state.mustDoID,
            plan: state.plan.map(\.slot), doneThisWeek: agenda.doneThisWeek,
            weekFocus: state.weekFocus)
        mustDoDays = state.mustDoDays
        departures = state.departures
        stepRuns = state.stepRuns
        // At the Mac past midnight (until 05:00) on this day's date.
        if let last = state.lastActiveAt, let midnight = state.day.date(calendar: calendar),
            last >= midnight, last < midnight.addingTimeInterval(5 * 3600),
            last <= now, now.timeIntervalSince(last) < 12 * 3600
        {
            upLateUntil = last
        }
        if state.mustDoID != nil {
            mustDoDays[state.day.rawValue] = state.mustDoDoneAt != nil
        }
    }

    /// The owner's day is the week's last (the day before the calendar's
    /// first weekday): the Evening Wrap-up looks back on the week.
    var isWeekReview: Bool {
        let weekday = calendar.component(.weekday, from: startOfToday)
        return weekday == (calendar.firstWeekday + 5) % 7 + 1
    }

    /// "The must-do got done on 4 of the 6 days it was set."
    var mustDoLine: String? {
        guard !mustDoDays.isEmpty else { return nil }
        let done = mustDoDays.values.filter { $0 }.count
        let set = mustDoDays.count
        return "The must-do got done on \(done) of the \(set) day\(set == 1 ? "" : "s") it was set."
    }

    /// The week's done reminders by Area, the most first.
    var doneThisWeekByArea: [(area: String, count: Int)] {
        var counts: [String: Int] = [:]
        for reminder in doneThisWeek { counts[areaName(of: reminder), default: 0] += 1 }
        return counts.map { ($0.key, $0.value) }.sorted {
            ($0.count, $1.area) > ($1.count, $0.area)
        }
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

    /// Overdue and not in today's plan: what the week's look-back asks about
    /// as waiting since an earlier day. A task planned today is today's,
    /// whatever its date.
    func isWaiting(_ reminder: AgendaReminder) -> Bool {
        (reminder.due ?? .distantFuture) < startOfToday
            && !plan.contains { $0.reminderID == reminder.id }
    }

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
