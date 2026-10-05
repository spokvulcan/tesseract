//
//  NowCard.swift
//  tesseract
//
//  The top of Today: where the owner is in the day and the one step that
//  moves it on. Built by code from the Timeline, so it is there the moment
//  Today opens and never waits on the model: the meeting they're in, the
//  task whose slot is now, a task that slid and the next free slot for it,
//  free time and what fits in it, what's next, and once the day is done,
//  how tomorrow starts. Jarvis proposes; the owner says yes in one click.
//  Pure: the day's facts in, a card out.
//

import Foundation

nonisolated struct NowCard: Sendable, Equatable {
    /// The step (an event or task title) or the day's state ("Free until
    /// 13:00", "All done for today.").
    var headline: String
    /// When it ends, what slid, what comes next.
    var detail: String?
    /// One-click proposals, the main one first.
    var actions: [NowAction]
}

nonisolated struct NowAction: Sendable, Equatable, Identifiable {
    enum Kind: Sendable, Equatable {
        case complete(reminderID: String)
        /// A slot in today's plan: "Start now", "Do it at 16:30".
        case place(reminderID: String, start: Date, minutes: Int)
        /// Due tomorrow; tomorrow's plan finds it a time.
        case tomorrow(reminderID: String)
        case planDay
        case wrapUp
    }

    var kind: Kind
    var title: String
    var id: String { title }
}

nonisolated enum NowCardBuilder {

    /// What the card needs to know besides the day's facts.
    struct Context: Sendable, Equatable {
        var companionOn: Bool
        /// The Morning Plan ran today.
        var planned: Bool
        /// The Evening Wrap-up ran today.
        var wrappedUp: Bool
        var eveningMinutes: Int
        var inboxCount: Int = 0
    }

    static let maxActions = 3

    static func build(timeline: TodayTimeline, facts: DayFacts, context: Context) -> NowCard {
        let evening = isEvening(
            facts.now, eveningMinutes: context.eveningMinutes, calendar: facts.calendar)
        var card = focus(
            timeline: timeline, facts: facts, evening: evening, inboxCount: context.inboxCount)
        if context.companionOn, !context.planned, !evening {
            card.actions.append(NowAction(kind: .planDay, title: "Plan my day"))
        }
        if context.companionOn, !context.wrappedUp, evening {
            card.actions.append(NowAction(kind: .wrapUp, title: "Wrap up the day"))
        }
        card.actions = Array(card.actions.prefix(maxActions))
        return card
    }

    /// The Evening Wrap-up's window: from the evening time until 03:00.
    static func isEvening(_ now: Date, eveningMinutes: Int, calendar: Calendar) -> Bool {
        let parts = calendar.dateComponents([.hour, .minute], from: now)
        let minute = (parts.hour ?? 0) * 60 + (parts.minute ?? 0)
        return minute >= eveningMinutes || minute < 3 * 60
    }

    // MARK: - The focus

    private static func focus(
        timeline: TodayTimeline, facts: DayFacts, evening: Bool, inboxCount: Int
    ) -> NowCard {
        let now = facts.now
        func clock(_ date: Date) -> String { AgendaTime.clock(date, calendar: facts.calendar) }

        // Events and open tasks still ahead, in time order.
        let ahead = timeline.rows.filter { row in
            guard row.start > now else { return false }
            switch row.kind {
            case .event: return true
            case .task(let task): return !task.isDone
            case .free, .now: return false
            }
        }
        func then(after date: Date) -> String? {
            ahead.first { $0.start >= date }.map { "then \(title(of: $0)) at \(clock($0.start))" }
        }

        // In a meeting or a block.
        for row in timeline.rows {
            guard case .event(let event) = row.kind, event.start <= now, event.end > now else {
                continue
            }
            return NowCard(
                headline: event.title,
                detail: joined("Until \(clock(event.end))", then(after: event.end)), actions: [])
        }

        let timed = timeline.rows.compactMap { row -> TimelineTask? in
            if case .task(let task) = row.kind, !task.isDone { task } else { nil }
        }

        // A task whose slot is now.
        if let task = timed.first(where: { task in
            guard let start = task.start else { return false }
            return start <= now && end(of: task) > now
        }) {
            return NowCard(
                headline: task.reminder.title,
                detail: joined("Until \(clock(end(of: task)))", then(after: end(of: task))),
                actions: [NowAction(kind: .complete(reminderID: task.id), title: "Done")])
        }

        // A task that slid: offer the next free slot (tomorrow, in the evening).
        let slid = timed.filter(\.isSlid)
        if let task = slid.first, let start = task.start {
            var detail = "Slid past \(clock(start))."
            if slid.count == 2 { detail += " One more slid too." }
            if slid.count > 2 { detail += " \(slid.count - 1) more slid too." }
            let done = NowAction(kind: .complete(reminderID: task.id), title: "Done")
            let tomorrow = NowAction(kind: .tomorrow(reminderID: task.id), title: "Tomorrow")
            var actions = [tomorrow, done]
            if !evening,
                let slot = TimelineBuilder.firstFreeSlot(minutes: task.minutes, facts: facts)
            {
                let place = NowAction(
                    kind: .place(reminderID: task.id, start: slot, minutes: task.minutes),
                    title: "Do it at \(clock(slot))")
                actions = [place, done, tomorrow]
            }
            return NowCard(headline: task.reminder.title, detail: detail, actions: actions)
        }

        let openAnytime = timeline.anytime.filter { !$0.isDone }

        // Free time now: what fits in it. With nothing to fit and nothing
        // after it, the day is done, not free.
        let freeNow = timeline.rows.contains { row in
            if case .free = row.kind, row.start <= now { true } else { false }
        }
        let candidate =
            openAnytime.first(where: \.isMustDo) ?? openAnytime.first { !$0.isCarried }
            ?? openAnytime.first
        if freeNow, candidate != nil || !ahead.isEmpty {
            let free =
                ahead.first.map { "free until \(clock($0.start))" }
                ?? "free for the rest of the day"
            if let task = candidate {
                return NowCard(
                    headline: task.reminder.title,
                    detail: task.isMustDo ? "Your must-do. You're \(free)." : "You're \(free).",
                    actions: [
                        NowAction(
                            kind: .place(
                                reminderID: task.id, start: minute(now, facts.calendar),
                                minutes: task.minutes),
                            title: "Start now"),
                        NowAction(kind: .complete(reminderID: task.id), title: "Done"),
                    ])
            }
            let next = ahead.first.map { "Then \(title(of: $0))." }
            let inbox =
                inboxCount > 0
                ? " \(inboxCount) Inbox item\(inboxCount == 1 ? "" : "s") could use a time." : ""
            return NowCard(
                headline: free.prefix(1).uppercased() + free.dropFirst(),
                detail: next.map { $0 + inbox }, actions: [])
        }

        // Not free, nothing on now: the next step. A task can start early.
        if let next = ahead.first {
            let minutes = max(1, Int(next.start.timeIntervalSince(now) / 60))
            var actions: [NowAction] = []
            if case .task(let task) = next.kind {
                actions = [
                    NowAction(
                        kind: .place(
                            reminderID: task.id, start: minute(now, facts.calendar),
                            minutes: task.minutes),
                        title: "Start now"),
                    NowAction(kind: .complete(reminderID: task.id), title: "Done"),
                ]
            }
            return NowCard(
                headline: title(of: next),
                detail: "At \(clock(next.start)), in \(MomentPrompts.minutesText(minutes)).",
                actions: actions)
        }

        // Nothing timed is left: what's still open today.
        let left = openAnytime.filter { !$0.isCarried || $0.isMustDo }
        if !left.isEmpty {
            let names = left.prefix(2).map(\.reminder.title).joined(separator: ", ")
            let more = left.count > 2 ? " and \(left.count - 2) more" : ""
            return NowCard(
                headline: "\(left.count) left for today", detail: names + more + ".", actions: [])
        }

        let hadADay = timeline.totalCount > 0 || !facts.eventsToday.isEmpty
        guard hadADay else {
            return NowCard(
                headline: "A clear day.",
                detail: "Add a task with + below, or ask Jarvis to plan with you.", actions: [])
        }
        return NowCard(
            headline: "All done for today.", detail: lookAhead(timeline.tomorrow, clock: clock),
            actions: [])
    }

    /// How tomorrow begins: its first step, steps below on the Day Line.
    private static func lookAhead(_ tomorrow: TomorrowTimeline, clock: (Date) -> String)
        -> String
    {
        if let first = tomorrow.rows.first {
            return "Next: \(title(of: first)), tomorrow at \(clock(first.start))."
        }
        if !tomorrow.allDayEvents.isEmpty {
            return "Tomorrow: " + tomorrow.allDayEvents.map(\.title).joined(separator: ", ") + "."
        }
        let tasks = tomorrow.anytime.filter { !$0.isDone }.count
        if tasks > 0 {
            return "Tomorrow has \(tasks) task\(tasks == 1 ? "" : "s"), none at a set time."
        }
        return "Nothing on tomorrow yet."
    }

    private static func title(of row: TimelineRow) -> String {
        switch row.kind {
        case .event(let event): event.title
        case .task(let task): task.reminder.title
        case .free, .now: ""
        }
    }

    private static func end(of task: TimelineTask) -> Date {
        (task.start ?? .distantPast).addingTimeInterval(TimeInterval(task.minutes * 60))
    }

    /// The current minute, seconds dropped, so a task started now reads "now".
    private static func minute(_ date: Date, _ calendar: Calendar) -> Date {
        calendar.dateInterval(of: .minute, for: date)?.start ?? date
    }

    private static func joined(_ parts: String?..., separator: String = " · ") -> String? {
        let present = parts.compactMap { $0 }
        return present.isEmpty ? nil : present.joined(separator: separator)
    }
}

// MARK: - Inbox

/// Where an Inbox item goes when the owner takes Jarvis's offer: the next
/// free slot today, or tomorrow once today has no room left (and always in
/// the evening).
nonisolated enum InboxSlot: Sendable, Equatable {
    case today(Date)
    case tomorrow

    /// How long an Inbox item gets in the day.
    static let minutes = 30

    static func suggest(facts: DayFacts, evening: Bool) -> InboxSlot {
        guard !evening, let start = TimelineBuilder.firstFreeSlot(minutes: minutes, facts: facts)
        else { return .tomorrow }
        return .today(start)
    }

    /// One slot each, in order: every offer keeps clear of the ones before
    /// it and of what's already planned (the Now Card's own offer included),
    /// so saying yes to all of them never stacks two things in one slot.
    static func suggest(for ids: [String], facts: DayFacts, evening: Bool) -> [String: InboxSlot] {
        var facts = facts
        var slots: [String: InboxSlot] = [:]
        for id in ids {
            let slot = suggest(facts: facts, evening: evening)
            slots[id] = slot
            if case .today(let start) = slot {
                facts.plan.removeAll { $0.reminderID == id }
                facts.plan.append(Placement(reminderID: id, start: start, minutes: minutes))
            }
        }
        return slots
    }
}

nonisolated extension NowCard {
    /// The slot this card offers in today's plan.
    var offeredPlacement: Placement? {
        for action in actions {
            if case .place(let id, let start, let minutes) = action.kind {
                return Placement(reminderID: id, start: start, minutes: minutes)
            }
        }
        return nil
    }

    /// The day with this card's offer taken (in place of the task's old
    /// slot, as a yes would), so the Inbox's offers keep clear of it.
    func reserving(_ facts: DayFacts) -> DayFacts {
        guard let offered = offeredPlacement else { return facts }
        var facts = facts
        facts.plan.removeAll { $0.reminderID == offered.reminderID }
        facts.plan.append(offered)
        return facts
    }
}

// MARK: - Jarvis's line

nonisolated extension DayCard {
    /// How long a card's line stays Jarvis's word on Today: a plan or a
    /// welcome back goes stale as the day moves on; the evening's cards hold
    /// until the day ends.
    var freshFor: TimeInterval? {
        switch kind {
        case .morningPlan: 4 * 3600
        case .breakpoint, .triage: 3600
        case .eveningWrapUp, .nightReflection: nil
        }
    }

    func isFresh(at now: Date) -> Bool {
        guard let freshFor else { return true }
        return now.timeIntervalSince(createdAt) < freshFor
    }
}
