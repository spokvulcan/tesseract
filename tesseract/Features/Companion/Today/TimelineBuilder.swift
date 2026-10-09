//
//  TimelineBuilder.swift
//  tesseract
//
//  The Today page's Timeline, chosen from the prototype: the day as one
//  time-ordered list. Calendar events and tasks share it; free gaps of half
//  an hour or more show as "Free · 1 h 30 min"; a Now line marks the time;
//  past done items are dimmed and unfinished past items read "slid", never
//  "missed". Tomorrow follows, its events and the tasks due then, so the
//  day always shows what comes after it. Pure: the day's facts in, rows out.
//

import Foundation

nonisolated struct TimelineTask: Sendable, Equatable, Identifiable {
    var id: String { reminder.id }
    var reminder: AgendaReminder
    var start: Date?
    var minutes: Int
    var areaName: String
    var areaColorHex: String?
    var isMustDo: Bool
    var isDone: Bool
    /// Past its slot and still open.
    var isSlid: Bool
    var isPlanned: Bool
    /// Due before today and still open.
    var isCarried: Bool
}

nonisolated struct TimelineRow: Sendable, Equatable, Identifiable {
    enum Kind: Sendable, Equatable {
        case event(AgendaEvent)
        case task(TimelineTask)
        case free(minutes: Int)
        case now
    }

    let id: String
    var start: Date
    var end: Date?
    var kind: Kind
    var isPast: Bool
}

nonisolated struct TodayTimeline: Sendable, Equatable {
    var rows: [TimelineRow]
    /// Today's tasks with no time and no slot, then carried-over ones.
    var anytime: [TimelineTask]
    var allDayEvents: [AgendaEvent]
    var doneCount: Int
    var totalCount: Int
    var mustDo: TimelineTask?
    var tomorrow: TomorrowTimeline

    static let empty = TodayTimeline(
        rows: [], anytime: [], allDayEvents: [], doneCount: 0, totalCount: 0, mustDo: nil,
        tomorrow: TomorrowTimeline(date: .distantPast, rows: [], anytime: [], allDayEvents: []))
}

/// Tomorrow, after today on the Day Line: its events and the tasks due
/// then. Not planned yet, so no free time; the morning's plan finds it.
nonisolated struct TomorrowTimeline: Sendable, Equatable {
    /// Tomorrow, at midnight.
    var date: Date
    /// Timed events and tasks, in time order.
    var rows: [TimelineRow]
    /// Tomorrow's tasks with no time.
    var anytime: [TimelineTask]
    var allDayEvents: [AgendaEvent]

    var isEmpty: Bool { rows.isEmpty && anytime.isEmpty && allDayEvents.isEmpty }
}

nonisolated enum TimelineBuilder {

    /// Gaps shorter than this are not worth showing.
    static let minimumFreeMinutes = 30
    /// A timed reminder without a planned slot takes this long on the line.
    static let defaultTaskMinutes = 15
    /// The latest a day's free time is shown to, unless events run later.
    static let dayEndHour = 22

    static func build(facts: DayFacts) -> TodayTimeline {
        let calendar = facts.calendar
        let now = facts.now
        let startOfToday = facts.startOfToday
        let endOfToday = facts.endOfToday
        // The owner's day runs until 04:00 (DayKey), so a slot in the small
        // hours is still today's.
        let rollover = DayKey(for: now, calendar: calendar).end(calendar: calendar) ?? endOfToday
        let plan = Dictionary(
            facts.plan.map { ($0.reminderID, $0) }, uniquingKeysWith: { first, _ in first })
        func task(_ reminder: AgendaReminder, start: Date?, minutes: Int, planned: Bool)
            -> TimelineTask
        {
            timelineTask(reminder, start: start, minutes: minutes, planned: planned, facts: facts)
        }

        // Every reminder that belongs to today: open ones due today or
        // earlier, open planned ones, and today's done ones (one due
        // tomorrow and done early stays with tomorrow).
        var byID: [String: AgendaReminder] = [:]
        for reminder in facts.dueOrOverdue + facts.doneToday where !isDueTomorrow(reminder, facts) {
            byID[reminder.id] = reminder
        }
        for reminder in facts.undated where plan[reminder.id] != nil {
            byID[reminder.id] = reminder
        }
        if let mustDo = facts.mustDoID, let reminder = facts.task(mustDo) {
            byID[mustDo] = reminder
        }

        var timed: [TimelineTask] = []
        var anytime: [TimelineTask] = []
        for reminder in byID.values {
            if let placement = plan[reminder.id], placement.start >= startOfToday,
                placement.start < rollover
            {
                timed.append(
                    task(
                        reminder, start: placement.start, minutes: placement.minutes, planned: true)
                )
            } else if reminder.dueHasTime, let due = reminder.due, due >= startOfToday,
                due < endOfToday
            {
                timed.append(
                    task(reminder, start: due, minutes: defaultTaskMinutes, planned: false))
            } else {
                anytime.append(
                    task(reminder, start: nil, minutes: defaultTaskMinutes, planned: false))
            }
        }

        let events = facts.eventsToday
        var rows: [TimelineRow] = events.map {
            TimelineRow(
                id: "event-\($0.id)", start: $0.start, end: $0.end, kind: .event($0),
                isPast: $0.end <= now)
        }
        rows += timed.map { item in
            let start = item.start ?? now
            let end = start.addingTimeInterval(TimeInterval(item.minutes * 60))
            return TimelineRow(
                id: "task-\(item.id)", start: start, end: end, kind: .task(item),
                isPast: item.isDone || end <= now)
        }

        // Free time from now to the end of the day, around what is booked.
        let dayEnd = max(
            calendar.date(bySettingHour: dayEndHour, minute: 0, second: 0, of: startOfToday)
                ?? endOfToday,
            events.map(\.end).max() ?? .distantPast)
        let busy =
            rows.compactMap { row -> DateInterval? in
                guard let end = row.end, end > row.start else { return nil }
                if case .task(let item) = row.kind, item.isDone { return nil }
                return DateInterval(start: row.start, end: end)
            }
            // The way to an event in person is not free time.
            + facts.departures.compactMap { departure in
                departure.at < departure.eventStart
                    ? DateInterval(start: departure.at, end: departure.eventStart) : nil
            }
        rows += freeGaps(from: now, to: dayEnd, busy: busy).map { gap in
            TimelineRow(
                id: "free-\(Int(gap.start.timeIntervalSince1970))", start: gap.start, end: gap.end,
                kind: .free(minutes: Int(gap.duration / 60)), isPast: false)
        }
        rows.append(TimelineRow(id: "now", start: now, end: nil, kind: .now, isPast: false))
        rows.sort(by: order)

        let todaysTasks = timed + anytime.filter { !$0.isCarried || $0.isMustDo }
        anytime.sort { a, b in
            if a.isDone != b.isDone { return !a.isDone }
            if a.isCarried != b.isCarried { return !a.isCarried }
            return a.reminder.title < b.reminder.title
        }
        let allDay = facts.events.filter {
            $0.isAllDay && $0.start < endOfToday && $0.end > startOfToday
        }
        let mustDo = (timed + anytime).first(where: \.isMustDo)
        return TodayTimeline(
            rows: rows, anytime: anytime, allDayEvents: allDay,
            doneCount: todaysTasks.filter(\.isDone).count, totalCount: todaysTasks.count,
            mustDo: mustDo, tomorrow: tomorrow(facts: facts))
    }

    /// Tomorrow's events and the tasks due then, open or already done.
    static func tomorrow(facts: DayFacts) -> TomorrowTimeline {
        let (start, end) = (facts.endOfToday, facts.endOfTomorrow)
        var rows: [TimelineRow] = facts.tomorrowEvents.map {
            TimelineRow(
                id: "event-\($0.id)", start: $0.start, end: $0.end, kind: .event($0),
                isPast: $0.end <= facts.now)
        }
        var anytime: [TimelineTask] = []
        let doneEarly = facts.doneToday.filter { isDueTomorrow($0, facts) }
        for reminder in facts.dueTomorrow + doneEarly {
            if reminder.dueHasTime, let due = reminder.due {
                let item = timelineTask(
                    reminder, start: due, minutes: defaultTaskMinutes, planned: false,
                    facts: facts)
                let finish = due.addingTimeInterval(TimeInterval(item.minutes * 60))
                rows.append(
                    TimelineRow(
                        id: "task-\(item.id)", start: due, end: finish, kind: .task(item),
                        isPast: item.isDone || finish <= facts.now))
            } else {
                anytime.append(
                    timelineTask(
                        reminder, start: nil, minutes: defaultTaskMinutes, planned: false,
                        facts: facts))
            }
        }
        rows.sort(by: order)
        anytime.sort { a, b in
            if a.isDone != b.isDone { return !a.isDone }
            return a.reminder.title < b.reminder.title
        }
        let allDay = facts.events.filter { $0.isAllDay && $0.start < end && $0.end > start }
        return TomorrowTimeline(date: start, rows: rows, anytime: anytime, allDayEvents: allDay)
    }

    private static func isDueTomorrow(_ reminder: AgendaReminder, _ facts: DayFacts) -> Bool {
        guard let due = reminder.due else { return false }
        return due >= facts.endOfToday && due < facts.endOfTomorrow
    }

    private static func timelineTask(
        _ reminder: AgendaReminder, start: Date?, minutes: Int, planned: Bool, facts: DayFacts
    ) -> TimelineTask {
        let area = facts.areas.first { $0.id == reminder.listID }
        let done = reminder.isCompleted || facts.doneToday.contains { $0.id == reminder.id }
        var slid = false
        if !done, let start {
            slid = start.addingTimeInterval(TimeInterval(minutes * 60)) <= facts.now
        }
        let carried = !done && (reminder.due.map { $0 < facts.startOfToday } ?? false)
        return TimelineTask(
            reminder: reminder, start: start, minutes: minutes,
            areaName: reminder.listID == facts.inboxListID
                ? "Inbox" : area?.name ?? reminder.listTitle,
            areaColorHex: area?.colorHex ?? reminder.colorHex,
            isMustDo: facts.mustDoID == reminder.id, isDone: done, isSlid: slid,
            isPlanned: planned, isCarried: carried)
    }

    /// Free intervals of at least half an hour in `[from, to)` around `busy`.
    static func freeGaps(from: Date, to: Date, busy: [DateInterval]) -> [DateInterval] {
        guard to > from else { return [] }
        var gaps: [DateInterval] = []
        var cursor = from
        for interval in busy.sorted(by: { $0.start < $1.start }) where interval.end > cursor {
            if interval.start > cursor {
                gaps.append(DateInterval(start: cursor, end: min(interval.start, to)))
            }
            cursor = max(cursor, interval.end)
            if cursor >= to { break }
        }
        if cursor < to { gaps.append(DateInterval(start: cursor, end: to)) }
        return gaps.filter { $0.duration >= TimeInterval(minimumFreeMinutes * 60) }
    }

    /// The first free slot today of at least `minutes`, from now on.
    static func firstFreeSlot(minutes: Int, facts: DayFacts) -> Date? {
        let timeline = build(facts: facts)
        for row in timeline.rows {
            if case .free(let free) = row.kind, free >= minutes {
                // Start on the next quarter hour.
                let calendar = facts.calendar
                let minute = calendar.component(.minute, from: row.start)
                let rounded = (minute + 14) / 15 * 15
                let start =
                    calendar.date(byAdding: .minute, value: rounded - minute, to: row.start)?
                    .addingTimeInterval(-TimeInterval(calendar.component(.second, from: row.start)))
                    ?? row.start
                if let end = row.end, start.addingTimeInterval(TimeInterval(minutes * 60)) <= end {
                    return start
                }
            }
        }
        return nil
    }

    private static func order(_ a: TimelineRow, _ b: TimelineRow) -> Bool {
        if a.start != b.start { return a.start < b.start }
        // At the same minute: the Now line first, then events, then tasks.
        func rank(_ row: TimelineRow) -> Int {
            switch row.kind {
            case .now: 0
            case .event: 1
            case .task: 2
            case .free: 3
            }
        }
        return rank(a) < rank(b)
    }
}
