//
//  Timeline.swift
//  tesseract
//
//  The day as the owner lives it, read once (ADR-0091): the Agenda's
//  snapshot and the Day Engine's state at one moment, as the Timeline —
//  events and tasks on one Day Line in time order, free gaps of half an hour
//  or more ("Free · 1 h 30 min"), the Now line, today's Anytime tasks, then
//  tomorrow, its events and the tasks due then, so the day always shows what
//  comes after it — with the answers every surface asks of it: the step the
//  day is on, the next step, free time and the Inbox's offers in it, when to
//  leave for an event, what waits on the owner, the menu bar's countdown.
//
//  Today draws it and its Now Card decides on it; the Jarvis Panel lists the
//  plan from it; the Day Engine reads its focus, next step and tomorrow from
//  it, and the menu bar its clock. Past done items are dimmed and unfinished
//  past items read "slid", never "missed". Pure: a snapshot and the day's
//  state in, the day out.
//

import Foundation

nonisolated struct TimelineTask: Sendable, Equatable, Identifiable {
    /// The task: its reminder's id (a repeating reminder's, its series').
    let id: String
    /// The reminder shown: the task itself — or, for a repeating task done
    /// today, the copy Reminders kept of its occurrence.
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
    /// The owner started its step, at this slot (Start on a cue, Start now).
    var isStarted = false
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

/// The step the day is on: what the Now Card leads with, the one row the
/// Day Line tints, the menu bar's countdown and the Day Engine's focus. One
/// rule, the first that matches (ADR-0091):
///
/// 1. a step the owner started whose slot holds now — over a block of their
///    own, or a meeting it runs into;
/// 2. a meeting under way: other people, or a place to be;
/// 3. a block of the owner's own under way;
/// 4. a task whose slot or time holds now, not started: one click from
///    starting.
nonisolated struct UnderWay: Sendable, Equatable {
    enum Kind: Sendable, Equatable {
        case started
        case meeting
        case block
        case notStarted
    }

    var kind: Kind
    /// Its row on the Day Line: today's, an event or a task.
    var row: TimelineRow

    var task: TimelineTask? {
        if case .task(let task) = row.kind { task } else { nil }
    }

    var event: AgendaEvent? {
        if case .event(let event) = row.kind { event } else { nil }
    }
}

nonisolated struct Timeline: Sendable, Equatable {

    /// Gaps shorter than this are not worth showing.
    static let minimumFreeMinutes = 30
    /// A timed reminder without a planned slot takes this long on the line.
    static let defaultTaskMinutes = 15
    /// The latest a day's free time is shown to, unless events run later.
    static let dayEndHour = 22
    /// How far ahead the menu bar counts down to what comes next.
    static let clockLead: TimeInterval = 30 * 60

    /// What the day was read from: the facts a moment is asked about.
    let facts: DayFacts
    /// Today on the Day Line, in time order: events, tasks with a time or a
    /// slot, free gaps, the Now line.
    let rows: [TimelineRow]
    /// Today's tasks with no time and no slot, then carried-over ones.
    let anytime: [TimelineTask]
    let allDayEvents: [AgendaEvent]
    let doneCount: Int
    let totalCount: Int
    let mustDo: TimelineTask?
    let tomorrow: TomorrowTimeline
    /// The step the day is on.
    let underWay: UnderWay?
    /// From the evening time until 03:00: the day closes.
    let isEvening: Bool
    /// The Morning Plan ran today.
    let isPlanned: Bool
    /// The Evening Wrap-up ran today.
    let isWrappedUp: Bool
    /// What waits on the owner.
    let waiting: WaitingOnYou

    /// The day at `snapshot`'s moment: its agenda, Areas and settings, and
    /// the day's state — its plan, must-do, departures and cards.
    init(snapshot: DaySnapshot, state: DayState) {
        let facts = snapshot.facts(state: state)
        let today = Self.today(facts: facts, steps: state.plan)
        self.facts = facts
        rows = today.rows
        anytime = today.anytime
        allDayEvents = today.allDayEvents
        doneCount = today.doneCount
        totalCount = today.totalCount
        mustDo = today.mustDo
        tomorrow = Self.tomorrow(facts: facts)
        underWay = Self.underWay(today.rows, now: facts.now)
        isEvening = snapshot.isEvening
        isPlanned = state.morningPlanAt != nil
        isWrappedUp = state.eveningWrapUpAt != nil
        waiting = WaitingOnYou(state: state, now: snapshot.now)
    }

    // MARK: - Answers

    /// Events and open tasks still ahead, in time order.
    var ahead: [TimelineRow] {
        rows.filter { row in
            guard row.start > facts.now else { return false }
            switch row.kind {
            case .event: return true
            case .task(let task): return !task.isDone
            case .free, .now: return false
            }
        }
    }

    /// The step the owner started that is running now, its task still open.
    var focus: StepFocus? {
        guard let underWay, underWay.kind == .started, let task = underWay.task,
            let end = underWay.row.end
        else { return nil }
        return StepFocus(reminderID: task.id, title: task.reminder.title, end: end)
    }

    /// When to leave for `event`, while that still stands.
    func departure(for event: AgendaEvent) -> Departure? {
        facts.departures.first { $0.event(in: [event]) != nil }
    }

    /// The menu bar's countdown: whichever comes first of the started
    /// step's end, the time to leave for an event in person and an event's
    /// start — the last two only within half an hour. A meeting coming up
    /// shows even over a step that would run into it.
    var clock: MenuBarClock? {
        let now = facts.now
        let horizon = now.addingTimeInterval(Self.clockLead)
        var times: [MenuBarClock] = []
        if let focus {
            times.append(MenuBarClock(kind: .focus, title: focus.title, until: focus.end))
        }
        let ahead = facts.events.filter { !$0.isAllDay && $0.start > now && $0.start <= horizon }
        if let event = ahead.min(by: { $0.start < $1.start }) {
            times.append(MenuBarClock(kind: .event, title: event.title, until: event.start))
        }
        for standing in facts.standingDepartures
        where standing.departure.at > now && standing.departure.at <= horizon {
            times.append(
                MenuBarClock(
                    kind: .leave, title: standing.event.title, until: standing.departure.at))
        }
        return times.min { $0.until < $1.until }
    }

    /// The day's next step once `slot` ends, and when it starts: an open
    /// task already under way (one a focus session held back; no time),
    /// else the next event or task.
    func nextStep(after slot: Placement) -> (title: String, at: Date?, link: URL?)? {
        let slotEnd = slot.start.addingTimeInterval(TimeInterval(slot.minutes * 60))
        for row in rows {
            let at = AgendaTime.clock(row.start, calendar: facts.calendar)
            switch row.kind {
            case .event(let event) where row.start >= slotEnd:
                return ("\(event.title) at \(at)", row.start, event.meetingLink)
            case .task(let task) where !task.isDone && task.id != slot.reminderID:
                if row.start >= slotEnd {
                    return ("\(task.reminder.title) at \(at)", row.start, nil)
                }
                if (row.end ?? row.start) > slotEnd { return (task.reminder.title, nil, nil) }
            default:
                continue
            }
        }
        return nil
    }

    // MARK: - Free time

    /// The first free slot today of at least `minutes`, from now on, on the
    /// next quarter hour.
    func firstFreeSlot(minutes: Int) -> Date? {
        Self.firstFreeSlot(minutes: minutes, rows: rows, calendar: facts.calendar)
    }

    /// Each task, in order, at the next free slot after the ones before it —
    /// what "Fit them in" places for tasks that slid. Stops at the first that
    /// no longer fits today.
    func fit(_ tasks: [TimelineTask]) -> [Placement] {
        var plan = facts.plan
        var placed: [Placement] = []
        for task in tasks {
            guard let start = freeSlot(minutes: task.minutes, plan: plan) else { break }
            let placement = Placement(reminderID: task.id, start: start, minutes: task.minutes)
            plan.removeAll { $0.reminderID == task.id }
            plan.append(placement)
            placed.append(placement)
        }
        return placed
    }

    // MARK: - The Inbox

    /// Undated captures in the Inbox. One that has a slot in today's plan
    /// has left the Inbox for the day's steps.
    var inbox: [AgendaReminder] {
        let planned = Set(facts.plan.map(\.reminderID))
        return facts.undated.filter { $0.listID == facts.inboxListID && !planned.contains($0.id) }
    }

    /// Jarvis's offer for each Inbox item, in order: the next free half hour
    /// today, clear of the ones before it and of what is already planned —
    /// `reserving` (the Now Card's own offer) in place of those tasks' slots,
    /// as a yes would leave them — so saying yes to all of them never stacks
    /// two things in one slot; tomorrow once today has no room, and always
    /// in the evening.
    func inboxOffers(for ids: [String], reserving: [Placement] = []) -> [String: InboxSlot] {
        var plan = facts.plan
        func reserve(_ placement: Placement) {
            plan.removeAll { $0.reminderID == placement.reminderID }
            plan.append(placement)
        }
        reserving.forEach(reserve)
        var slots: [String: InboxSlot] = [:]
        for id in ids {
            guard !isEvening, let start = freeSlot(minutes: InboxSlot.minutes, plan: plan) else {
                slots[id] = .tomorrow
                continue
            }
            slots[id] = .today(start)
            reserve(Placement(reminderID: id, start: start, minutes: InboxSlot.minutes))
        }
        return slots
    }

    /// The first free slot of `minutes` with today's plan as `plan`.
    private func freeSlot(minutes: Int, plan: [Placement]) -> Date? {
        var facts = facts
        facts.plan = plan
        let rows = Self.today(facts: facts, steps: []).rows
        return Self.firstFreeSlot(minutes: minutes, rows: rows, calendar: facts.calendar)
    }

    private static func firstFreeSlot(minutes: Int, rows: [TimelineRow], calendar: Calendar)
        -> Date?
    {
        for row in rows {
            if case .free(let free) = row.kind, free >= minutes {
                // Start on the next quarter hour.
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

    /// Free intervals of at least half an hour in `[from, to)` around `busy`.
    private static func freeGaps(from: Date, to: Date, busy: [DateInterval]) -> [DateInterval] {
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

    // MARK: - Building

    private struct Today {
        var rows: [TimelineRow]
        var anytime: [TimelineTask]
        var allDayEvents: [AgendaEvent]
        var doneCount: Int
        var totalCount: Int
        var mustDo: TimelineTask?
    }

    /// Today's rows and tasks; `steps` say which slots the owner started.
    private static func today(facts: DayFacts, steps: [Step]) -> Today {
        let calendar = facts.calendar
        let now = facts.now
        let startOfToday = facts.startOfToday
        let endOfToday = facts.endOfToday
        // The owner's day runs until 04:00 (DayKey), so a slot in the small
        // hours is still today's.
        let rollover = DayKey(for: now, calendar: calendar).end(calendar: calendar) ?? endOfToday
        let plan = Dictionary(
            facts.plan.map { ($0.reminderID, $0) }, uniquingKeysWith: { first, _ in first })
        func task(
            _ id: String, _ reminder: AgendaReminder, start: Date?, minutes: Int, planned: Bool
        ) -> TimelineTask {
            var task = timelineTask(
                id, reminder, start: start, minutes: minutes, planned: planned,
                isMustDo: facts.mustDoID == id, facts: facts)
            task.isStarted =
                planned
                && steps.contains { $0.reminderID == id && $0.start == start && $0.isStarted }
            return task
        }

        // Every task that belongs to today, by its id: open ones due today or
        // earlier, open planned ones, and today's done ones (one due
        // tomorrow and done early stays with tomorrow). A repeating task the
        // day holds, done today, is its occurrence — the copy Reminders kept,
        // shown once, in the task's own slot.
        let copies = Set(facts.doneOccurrences.values.map(\.id))
        var byID: [String: AgendaReminder] = [:]
        for reminder in facts.dueOrOverdue + facts.doneToday
        where !isDueTomorrow(reminder, facts) && !copies.contains(reminder.id) {
            byID[reminder.id] = reminder
        }
        for reminder in facts.undated where plan[reminder.id] != nil {
            byID[reminder.id] = reminder
        }
        if let mustDo = facts.mustDoID, let reminder = facts.task(mustDo) {
            byID[mustDo] = reminder
        }
        byID.merge(facts.doneOccurrences) { _, occurrence in occurrence }

        var timed: [TimelineTask] = []
        var anytime: [TimelineTask] = []
        for (id, reminder) in byID {
            if let placement = plan[id], placement.start >= startOfToday,
                placement.start < rollover
            {
                timed.append(
                    task(
                        id, reminder, start: placement.start, minutes: placement.minutes,
                        planned: true))
            } else if reminder.dueHasTime, let due = reminder.due, due >= startOfToday,
                due < endOfToday
            {
                timed.append(
                    task(id, reminder, start: due, minutes: defaultTaskMinutes, planned: false))
            } else {
                anytime.append(
                    task(id, reminder, start: nil, minutes: defaultTaskMinutes, planned: false))
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

        // Free time from now to the end of the day, around what is booked:
        // every event (a meeting or a block), every open task's slot or
        // time, and the way to an event in person.
        let dayEnd = max(
            calendar.date(bySettingHour: dayEndHour, minute: 0, second: 0, of: startOfToday)
                ?? endOfToday,
            events.map(\.end).max() ?? .distantPast)
        let busy =
            rows.compactMap { row -> DateInterval? in
                guard let end = row.end, end > row.start else { return nil }
                if case .task(let item) = row.kind, item.isDone { return nil }
                return DateInterval(start: row.start, end: end)
            } + facts.waysThereToday
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
        return Today(
            rows: rows, anytime: anytime, allDayEvents: allDay,
            doneCount: todaysTasks.filter(\.isDone).count, totalCount: todaysTasks.count,
            mustDo: (timed + anytime).first(where: \.isMustDo))
    }

    /// Tomorrow's events and the tasks due then, open or already done. The
    /// must-do is today's: tomorrow's plan picks its own, so nothing here
    /// wears its star.
    private static func tomorrow(facts: DayFacts) -> TomorrowTimeline {
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
                    reminder.id, reminder, start: due, minutes: defaultTaskMinutes,
                    planned: false, isMustDo: false, facts: facts)
                let finish = due.addingTimeInterval(TimeInterval(item.minutes * 60))
                rows.append(
                    TimelineRow(
                        id: "task-\(item.id)", start: due, end: finish, kind: .task(item),
                        isPast: item.isDone || finish <= facts.now))
            } else {
                anytime.append(
                    timelineTask(
                        reminder.id, reminder, start: nil, minutes: defaultTaskMinutes,
                        planned: false, isMustDo: false, facts: facts))
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

    /// The step the day is on, among today's rows (`UnderWay`).
    private static func underWay(_ rows: [TimelineRow], now: Date) -> UnderWay? {
        let current = rows.filter { row in
            guard let end = row.end, !row.isPast, row.start <= now, end > now else { return false }
            switch row.kind {
            case .event, .task: return true
            case .free, .now: return false
            }
        }
        func first(_ kind: UnderWay.Kind, _ matches: (TimelineRow.Kind) -> Bool) -> UnderWay? {
            current.first { matches($0.kind) }.map { UnderWay(kind: kind, row: $0) }
        }
        return first(.started) { if case .task(let task) = $0 { task.isStarted } else { false } }
            ?? first(.meeting) { if case .event(let event) = $0 { event.isMeeting } else { false } }
            ?? first(.block) { if case .event(let event) = $0 { !event.isMeeting } else { false } }
            ?? first(.notStarted) {
                if case .task(let task) = $0 { !task.isStarted } else { false }
            }
    }

    private static func isDueTomorrow(_ reminder: AgendaReminder, _ facts: DayFacts) -> Bool {
        guard let due = reminder.due else { return false }
        return due >= facts.endOfToday && due < facts.endOfTomorrow
    }

    private static func timelineTask(
        _ id: String, _ reminder: AgendaReminder, start: Date?, minutes: Int, planned: Bool,
        isMustDo: Bool, facts: DayFacts
    ) -> TimelineTask {
        let area = facts.areas.first { $0.id == reminder.listID }
        let done = reminder.isCompleted || facts.doneToday.contains { $0.id == reminder.id }
        var slid = false
        if !done, let start {
            slid = start.addingTimeInterval(TimeInterval(minutes * 60)) <= facts.now
        }
        let carried = !done && (reminder.due.map { $0 < facts.startOfToday } ?? false)
        return TimelineTask(
            id: id, reminder: reminder, start: start, minutes: minutes,
            areaName: facts.areaName(of: reminder),
            areaColorHex: area?.colorHex ?? reminder.colorHex,
            isMustDo: isMustDo, isDone: done, isSlid: slid, isPlanned: planned,
            isCarried: carried)
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

/// Where an Inbox item goes when the owner takes Jarvis's offer: the next
/// free slot today, or tomorrow once today has no room left (and always in
/// the evening).
nonisolated enum InboxSlot: Sendable, Equatable {
    case today(Date)
    case tomorrow

    /// How long an Inbox item gets in the day.
    static let minutes = 30
}
