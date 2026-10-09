//
//  NowCard.swift
//  tesseract
//
//  The top of Today: where the owner is in the day and the one step that
//  moves it on. Built by code from the Timeline, so it is there the moment
//  Today opens and never waits on the model: the meeting they're in, the
//  task whose slot is now (under way, or one click from starting), a task
//  that slid and the next free slot for it,
//  free time and what fits in it, what's next; in the evening, the day
//  closing (never a slid task at midnight); and once the day is done, how
//  tomorrow starts. Jarvis proposes; the owner says yes in one click.
//  Pure: the day's facts in, a card out.
//

import Foundation

nonisolated struct NowCard: Sendable, Equatable {
    /// The step (an event or task title) or the day's state ("Free until
    /// 13:00", "All done for today.").
    var headline: String
    /// How much is left and when it ends, what slid, what comes next.
    var detail: String?
    /// One-click proposals, the main one first.
    var actions: [NowAction]
    /// The step under way, start to end: Today shows how much of it is
    /// left, so the time can be seen and not only read.
    var span: DateInterval? = nil
    /// The step is the day's must-do: it wears Today's star.
    var isMustDo = false
}

nonisolated struct NowAction: Sendable, Equatable, Identifiable {
    enum Kind: Sendable, Equatable {
        case complete(reminderID: String)
        /// A slot in today's plan: "Start now", "Do it at 16:30".
        case place(reminderID: String, start: Date, minutes: Int)
        /// Slots for tasks that slid, one after another: "Fit all 3 in".
        case placeAll([Placement])
        /// Due tomorrow; tomorrow's plan finds it a time.
        case tomorrow(reminderID: String)
        case planDay
        case wrapUp
    }

    var kind: Kind
    var title: String
    /// What the click will do, in full, for the tooltip.
    var help: String? = nil
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
        /// The slots the owner started, by `StepCue.key`: one whose time
        /// is now is under way; one not started is offered to start.
        var startedSteps: Set<String> = []
    }

    static let maxActions = 3

    static func build(timeline: TodayTimeline, facts: DayFacts, context: Context) -> NowCard {
        let evening = isEvening(
            facts.now, eveningMinutes: context.eveningMinutes, calendar: facts.calendar)
        var card = focus(
            timeline: timeline, facts: facts, evening: evening, context: context)
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
        timeline: TodayTimeline, facts: DayFacts, evening: Bool, context: Context
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
        // Time left first: what a glance at the clock can't tell.
        func left(until end: Date) -> String {
            let minutes = max(1, Int((end.timeIntervalSince(now) / 60).rounded(.up)))
            return "\(MomentPrompts.minutesText(minutes)) left, until \(clock(end))"
        }
        // When to leave for an event in person, if the plan set a time.
        func departure(for row: TimelineRow) -> Departure? {
            guard case .event(let event) = row.kind else { return nil }
            return facts.departures.first {
                $0.eventID == event.id && $0.eventStart == event.start
            }
        }

        // In a meeting or a block.
        for row in timeline.rows {
            guard case .event(let event) = row.kind, event.start <= now, event.end > now else {
                continue
            }
            return NowCard(
                headline: event.title,
                detail: joined(left(until: event.end), then(after: event.end)), actions: [],
                span: DateInterval(start: event.start, end: event.end))
        }

        let timed = timeline.rows.compactMap { row -> TimelineTask? in
            if case .task(let task) = row.kind, !task.isDone { task } else { nil }
        }

        // Time to leave for an event in person: that is the step now, over
        // any task still running.
        if let next = ahead.first(where: { departure(for: $0).map { $0.at <= now } ?? false }) {
            return NowCard(
                headline: title(of: next),
                detail: "Time to leave. It starts at \(clock(next.start)).", actions: [])
        }

        // A task whose slot is now: under way once the owner started it
        // (its time left drains); else its time, one click from starting —
        // a cue closed, missed or answered by mistake leaves Today the
        // place to start it.
        if let task = timed.first(where: { task in
            guard let start = task.start else { return false }
            return start <= now && end(of: task) > now
        }) {
            let start = task.start ?? now
            let slot = Placement(reminderID: task.id, start: start, minutes: task.minutes)
            guard context.startedSteps.contains(StepCue.key(slot)) else {
                return NowCard(
                    headline: task.reminder.title,
                    detail: joined(
                        "\(clock(start))–\(clock(end(of: task)))", then(after: end(of: task))),
                    actions: [
                        NowAction(
                            kind: .place(
                                reminderID: task.id, start: minute(now, facts.calendar),
                                minutes: task.minutes),
                            title: "Start now",
                            help: "Start it now; Jarvis checks in when its time is up"),
                        NowAction(kind: .complete(reminderID: task.id), title: "Done"),
                    ], isMustDo: task.isMustDo)
            }
            return NowCard(
                headline: task.reminder.title,
                detail: joined(left(until: end(of: task)), then(after: end(of: task))),
                actions: [NowAction(kind: .complete(reminderID: task.id), title: "Done")],
                span: DateInterval(start: task.start ?? now, end: end(of: task)),
                isMustDo: task.isMustDo)
        }

        let openAnytime = timeline.anytime.filter { !$0.isDone }
        // The day's one thing that mattered most is done: the rest is a
        // bonus, and the card says so.
        let mustDoDone = timeline.mustDo?.isDone == true

        // The next step. A task can start early; an event in person says
        // when to leave for it.
        func nextStep(_ next: TimelineRow) -> NowCard {
            let minutes = max(1, Int(next.start.timeIntervalSince(now) / 60))
            if let leave = departure(for: next) {
                let untilLeave = max(1, Int(leave.at.timeIntervalSince(now) / 60))
                return NowCard(
                    headline: title(of: next),
                    detail:
                        "Leave at \(clock(leave.at)), in \(MomentPrompts.minutesText(untilLeave)). It starts at \(clock(next.start)).",
                    actions: [])
            }
            var actions: [NowAction] = []
            var isMustDo = false
            if case .task(let task) = next.kind {
                isMustDo = task.isMustDo
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
                actions: actions, isMustDo: isMustDo)
        }

        // The evening closes the day: what is still ahead tonight, or else
        // what got done — never a slid task at the top at midnight. What is
        // still open goes to the Evening Wrap-up.
        if evening {
            if let next = ahead.first { return nextStep(next) }
            let open = timed + openAnytime.filter { !$0.isCarried || $0.isMustDo }
            if !open.isEmpty {
                let names = open.prefix(2).map(\.reminder.title).joined(separator: ", ")
                let more = open.count > 2 ? " and \(open.count - 2) more" : ""
                let done = timeline.doneCount
                // Wrapped up, what's open has its place: look ahead instead.
                let detail =
                    context.wrappedUp
                    ? lookAhead(timeline.tomorrow, now: now, clock: clock)
                    : done > 0 ? "Still open: \(names)\(more)." : "\(names)\(more)."
                let count = "\(done) of \(timeline.totalCount) done today"
                return NowCard(
                    headline: done > 0
                        ? (mustDoDone ? "\(count), the must-do among them." : "\(count).")
                        : "\(open.count) left for today",
                    detail: detail, actions: [])
            }
        }

        // A task that slid: offer the next free slot, or tomorrow. Several
        // that slid are fitted into the day in one click, in order — one
        // decision, not one per task, and no tidying the plan at midnight.
        let slid = timed.filter(\.isSlid)
        if let task = slid.first, let start = task.start {
            // The day's one goal slipping is said so, not as one more task.
            var detail = (task.isMustDo ? "Your must-do slid" : "Slid") + " past \(clock(start))."
            if slid.count == 2 { detail += " One more slid too." }
            if slid.count > 2 { detail += " \(slid.count - 1) more slid too." }
            let done = NowAction(kind: .complete(reminderID: task.id), title: "Done")
            let tomorrow = NowAction(kind: .tomorrow(reminderID: task.id), title: "Tomorrow")
            var actions = [tomorrow, done]
            let fitted = slid.count > 1 ? TimelineBuilder.fit(slid, facts: facts) : []
            if fitted.count > 1 {
                let all =
                    fitted.count < slid.count
                    ? "\(fitted.count)" : fitted.count == 2 ? "both" : "all \(fitted.count)"
                let titles = Dictionary(
                    slid.map { ($0.id, $0.reminder.title) }, uniquingKeysWith: { first, _ in first }
                )
                let help = fitted.map { "\(titles[$0.reminderID] ?? "") at \(clock($0.start))" }
                    .joined(separator: " · ")
                let fit = NowAction(kind: .placeAll(fitted), title: "Fit \(all) in", help: help)
                actions = [fit, done, tomorrow]
            } else if let slot = TimelineBuilder.firstFreeSlot(minutes: task.minutes, facts: facts)
            {
                let place = NowAction(
                    kind: .place(reminderID: task.id, start: slot, minutes: task.minutes),
                    title: "Do it at \(clock(slot))")
                actions = [place, done, tomorrow]
            }
            return NowCard(
                headline: task.reminder.title, detail: detail, actions: actions,
                isMustDo: task.isMustDo)
        }

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
                ahead.first.map { row in
                    // Free until it's time to leave, not until the event.
                    departure(for: row).map {
                        "free until \(clock($0.at)), when you leave for \(title(of: row))"
                    } ?? "free until \(clock(row.start))"
                } ?? "free for the rest of the day"
            if let task = candidate {
                let detail =
                    task.isMustDo
                    ? "Your must-do. You're \(free)."
                    : mustDoDone
                        ? "The must-do is done; this one's a bonus. You're \(free)."
                        : "You're \(free)."
                return NowCard(
                    headline: task.reminder.title,
                    detail: detail,
                    actions: [
                        NowAction(
                            kind: .place(
                                reminderID: task.id, start: minute(now, facts.calendar),
                                minutes: task.minutes),
                            title: "Start now"),
                        NowAction(kind: .complete(reminderID: task.id), title: "Done"),
                    ], isMustDo: task.isMustDo)
            }
            let next = ahead.first.map { "Then \(title(of: $0))." }
            let inboxCount = context.inboxCount
            let inbox =
                inboxCount > 0
                ? " \(inboxCount) Inbox item\(inboxCount == 1 ? "" : "s") could use a time." : ""
            return NowCard(
                headline: free.prefix(1).uppercased() + free.dropFirst(),
                detail: next.map { $0 + inbox }, actions: [])
        }

        // Not free, nothing on now: the next step.
        if let next = ahead.first { return nextStep(next) }

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
            headline: "All done for today.",
            detail: lookAhead(timeline.tomorrow, now: now, clock: clock), actions: [])
    }

    /// How tomorrow begins: its first step, steps below on the Day Line.
    /// How tomorrow starts. Within half a day, how far off that is too — the
    /// wind-down's word, there whenever Today is looked at late.
    private static func lookAhead(
        _ tomorrow: TomorrowTimeline, now: Date, clock: (Date) -> String
    ) -> String {
        if let first = tomorrow.rows.first {
            let next = "Next: \(title(of: first)), tomorrow at \(clock(first.start))"
            let minutes = Int(first.start.timeIntervalSince(now) / 60)
            guard minutes > 0, minutes < 12 * 60 else { return next + "." }
            return next + " — \(MomentPrompts.minutesText(minutes)) from now."
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
    var offeredPlacement: Placement? { offeredPlacements.first }

    /// Every slot this card offers: one ("Do it at 16:30"), or one per task
    /// that slid ("Fit all 3 in").
    var offeredPlacements: [Placement] {
        actions.flatMap { action -> [Placement] in
            switch action.kind {
            case .place(let id, let start, let minutes):
                [Placement(reminderID: id, start: start, minutes: minutes)]
            case .placeAll(let placements): placements
            case .complete, .tomorrow, .planDay, .wrapUp: []
            }
        }
    }

    /// The day with this card's offer taken (in place of the tasks' old
    /// slots, as a yes would), so the Inbox's offers keep clear of it.
    func reserving(_ facts: DayFacts) -> DayFacts {
        var facts = facts
        for offered in offeredPlacements {
            facts.plan.removeAll { $0.reminderID == offered.reminderID }
            facts.plan.append(offered)
        }
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

    /// It has something to say on Today. A Breakpoint with nothing that
    /// needs the owner says nothing: "nothing needs you" is no news, and it
    /// would push the plan's word off the card after every break.
    var hasWord: Bool {
        if case .breakpoint(let breakpoint) = body { return !breakpoint.needsYou.isEmpty }
        return true
    }

    /// Jarvis's word on the Now Card: the latest open card with something to
    /// say, while it is fresh.
    static func word(in cards: [DayCard], at now: Date) -> DayCard? {
        cards.last { !$0.dismissed && $0.hasWord && $0.isFresh(at: now) }
    }
}
