//
//  DayEngine+Steps.swift
//  tesseract
//
//  The plan keeps its own time. A Morning Plan's slots and the owner's own
//  "Do it at 16:30" live in the day's state, not in Reminders, so no alarm
//  marks them. When a planned step's slot starts and the owner is at the
//  Mac, code puts it on the Jarvis Panel — the Step Cue — with one-click
//  choices: Start, In 15 min, Tomorrow, Done. A step the owner started (Start
//  on a cue, or Start now on Today) checks in when its time is up: Done,
//  15 more min, Tomorrow. Help lands at the moment of doing, not in a list
//  the owner has to remember to read, and a started step never just slides.
//
//  Starting is the hard part: a step put off twice ("In 15 min") is offered
//  as five minutes ("Start 5 min"), and five minutes in, the check-in asks
//  to keep going.
//
//  A meeting ends a step, not the other way round: a step started or moved
//  ends five minutes before the next meeting (or the time to leave for one)
//  it would run into, so its check-in is the heads-up to wrap up and get
//  there, not a question held until the meeting is over. Where a quarter
//  hour more — or later — would run into a meeting, the cue offers to go on
//  once it is over instead, with the time that was cut.
//
//  No model. Each start and each end is cued once, as soon as the owner can
//  see it: never while they are away, in quiet hours, a call, a game or a
//  meeting, never over a panel they haven't closed, and nothing while a step
//  they started is running. What came due meanwhile waits for them — a start
//  while its slot still runs, an end that day — and says it is late. A slot
//  whose reminder rings at the same minute is left to Reminders. Closing a
//  cue changes nothing.
//

import Foundation

nonisolated extension DayEngine {

    /// A cue shown this long after its moment is late, and says so: the
    /// owner was away, in a meeting or a game, or behind another panel.
    /// Dropping it instead let the plan slide unseen.
    static let stepCueLate: TimeInterval = 10 * 60
    /// "In 15 min" moves a slot this far; "15 more min" makes it this much
    /// longer.
    static let stepLaterMinutes = 15
    /// "Start 5 min": a step put off twice is offered this small a start.
    static let smallStartMinutes = 5
    /// Started from a cue whose slot went meanwhile (a re-plan): this long.
    static let lostSlotMinutes = 30
    /// A step ends this long before the next meeting, or the time to leave
    /// for one: time to wrap up and get there.
    static let transitionMinutes = 5

    // MARK: - The cue

    /// A Step Cue or a Break Cue, whichever is due, by the same rules.
    static func cueIfDue(snapshot: DaySnapshot, state: inout DayState) -> [DayEffect] {
        // Once the owner sat down to start the day, the morning's end of
        // quiet hours doesn't hold the plan back.
        let started = DeliveryLadder.dayStarted(state.satDownAt, snapshot: snapshot)
        guard !snapshot.panelUp,
            DeliveryLadder.rungs(for: .normal, snapshot: snapshot, sittingDown: started)
                .contains(.panel),
            !inMeeting(snapshot),
            // A step the owner started is still running: a focus session isn't
            // interrupted. What comes due meanwhile waits, its check-in names
            // what is under way, and it is cued once the owner is free.
            focus(snapshot: snapshot, state: state) == nil
        else { return [] }
        let steps = snapshot.settings.stepCues
        // A step that is over comes first: it closes what the next one opens.
        // A break comes between it and the next step: the time to take one.
        if steps, let end = stepEndIfDue(snapshot: snapshot, state: &state) { return end }
        if let rest = breakCueIfDue(snapshot: snapshot, state: &state) { return rest }
        return steps ? stepStartIfDue(snapshot: snapshot, state: &state) ?? [] : []
    }

    /// The first slot under way, for an open task not yet cued.
    private static func stepStartIfDue(snapshot: DaySnapshot, state: inout DayState)
        -> [DayEffect]?
    {
        let now = snapshot.now
        let facts = snapshot.facts(state: state)
        let starting = state.plan.filter { slot in
            slot.start <= now && now < end(of: slot)
                && state.cuedSteps[StepCue.key(slot)] == nil
        }
        for slot in starting.sorted(by: { $0.start < $1.start }) {
            // Done, or gone from Reminders: nothing to start.
            guard let task = facts.task(slot.reminderID) else { continue }
            state.cuedSteps[StepCue.key(slot)] = now
            // Its own alarm rings now: Reminders has it.
            if task.dueHasTime, let due = task.due, abs(due.timeIntervalSince(slot.start)) < 60 {
                continue
            }
            state.cueOnPanel = StepCue.key(slot)
            return present(
                cue(task, slot: slot, phase: .start, facts: facts, state: state, now: now),
                late: now.timeIntervalSince(slot.start))
        }
        return nil
    }

    /// A card took the panel from the cue on it: the cue is uncued, to come
    /// back by the rules above once the panel is free again.
    static func holdCueUnderCard(
        _ effects: [DayEffect], snapshot: DaySnapshot, state: inout DayState
    ) {
        guard state.cueOnPanel != nil || state.breakCuedAt != nil, snapshot.panelUp,
            effects.contains(where: { if case .presentCard(_, .panel) = $0 { true } else { false } }
            )
        else { return }
        if let key = state.cueOnPanel { state.cuedSteps[key] = nil }
        state.cueOnPanel = nil
        state.breakCuedAt = nil
    }

    /// A slot the owner started whose time is up, its task still open: the
    /// one that ended last first, the step they were just on.
    private static func stepEndIfDue(snapshot: DaySnapshot, state: inout DayState)
        -> [DayEffect]?
    {
        let now = snapshot.now
        let facts = snapshot.facts(state: state)
        let ended = state.plan.filter { slot in
            state.startedSteps.contains(StepCue.key(slot)) && end(of: slot) <= now
                && state.cuedSteps[StepCue.endKey(slot)] == nil
        }
        for slot in ended.sorted(by: { end(of: $0) > end(of: $1) }) {
            guard let task = facts.task(slot.reminderID) else { continue }
            state.cuedSteps[StepCue.endKey(slot)] = now
            state.cueOnPanel = StepCue.endKey(slot)
            return present(
                cue(task, slot: slot, phase: .end, facts: facts, state: state, now: now),
                late: now.timeIntervalSince(end(of: slot)))
        }
        return nil
    }

    private static func present(_ cue: StepCue, late: TimeInterval) -> [DayEffect] {
        [
            .presentStep(cue),
            .trace(
                .cuePresented,
                [
                    "phase": .string(cue.phase.rawValue), "late": .int(Int(late)),
                    "minutes": .int(cue.minutes), "mustDo": .bool(cue.isMustDo),
                ]),
        ]
    }

    private static func cue(
        _ task: AgendaReminder, slot: Placement, phase: StepCue.Phase, facts: DayFacts,
        state: DayState, now: Date
    ) -> StepCue {
        let moment = phase == .start ? slot.start : end(of: slot)
        let next = nextStep(after: slot, facts: facts)
        return StepCue(
            reminderID: task.id, title: task.title, start: slot.start, minutes: slot.minutes,
            areaName: facts.areaName(of: task), isMustDo: state.mustDoID == task.id,
            next: next?.0, nextAt: next?.1, phase: phase,
            late: now.timeIntervalSince(moment) >= stepCueLate,
            putOff: state.putOff[task.id] ?? 0,
            small: state.smallStarts.contains(StepCue.key(slot)),
            resumeAt: resumeTime(for: slot, started: phase == .end, now: now, facts: facts))
    }

    private static func end(of slot: Placement) -> Date {
        slot.start.addingTimeInterval(TimeInterval(slot.minutes * 60))
    }

    // MARK: - Clear of meetings

    /// What ends a step: a meeting with other people, or somewhere to be. A
    /// block of the owner's own ("Work", 09:00–13:00) is where steps happen.
    private static func endsAStep(_ event: AgendaEvent) -> Bool {
        event.hasOtherAttendees || !(event.location ?? "").isEmpty
    }

    /// How long a step from `start` gets so it keeps clear of what's fixed
    /// after it: it ends `transitionMinutes` before the next meeting, or the
    /// time to leave for one, it would run into — right at it when that
    /// leaves too little; with not even five minutes before it, as wanted.
    static func clearMinutes(from start: Date, wanted: Int, facts: DayFacts) -> Int {
        let end = start.addingTimeInterval(TimeInterval(wanted * 60))
        let events = facts.eventsToday.filter(endsAStep)
        let fixed =
            events.map(\.start)
            + facts.departures.compactMap { departure in
                events.contains { $0.id == departure.eventID && $0.start == departure.eventStart }
                    ? departure.at : nil
            }
        guard let next = fixed.filter({ $0 > start && $0 <= end }).min() else { return wanted }
        let room = Int(next.timeIntervalSince(start) / 60)
        if room - transitionMinutes >= smallStartMinutes { return room - transitionMinutes }
        return room >= smallStartMinutes ? room : wanted
    }

    /// The task's slot, cut to keep clear of the next meeting; what was cut
    /// is kept, for going on after it.
    static func keepClear(_ reminderID: String, snapshot: DaySnapshot, state: inout DayState) {
        guard let index = state.plan.firstIndex(where: { $0.reminderID == reminderID }) else {
            return
        }
        let slot = state.plan[index]
        let minutes = clearMinutes(
            from: slot.start, wanted: slot.minutes, facts: snapshot.facts(state: state))
        guard minutes < slot.minutes else { return }
        state.plan[index].minutes = minutes
        state.cutShort[reminderID] = slot.minutes - minutes
    }

    /// When a step can go on if a meeting, or the way to one, would cut into
    /// what the cue offers: "15 more min" past a started step's end, or "In
    /// 15 min" with less than a quarter hour of work before the meeting. The
    /// end of the meeting, and of any straight after it; nil when the next
    /// quarter hour is free.
    static func resumeTime(for slot: Placement, started: Bool, now: Date, facts: DayFacts)
        -> Date?
    {
        let quarter = TimeInterval(stepLaterMinutes * 60)
        let transition = TimeInterval(transitionMinutes * 60)
        let limit =
            started
            ? max(end(of: slot), now).addingTimeInterval(quarter + transition)
            : now.addingTimeInterval(2 * quarter + transition)
        let events = facts.eventsToday.filter(endsAStep)
        var coming = events.filter { $0.start > now && $0.start < limit }.map {
            (at: $0.start, end: $0.end)
        }
        for departure in facts.departures where departure.at > now && departure.at < limit {
            if let event = events.first(where: {
                $0.id == departure.eventID && $0.start == departure.eventStart
            }) {
                coming.append((at: departure.at, end: event.end))
            }
        }
        guard var end = coming.min(by: { $0.at < $1.at })?.end else { return nil }
        // Back to back: after the last of the run.
        while let next = events.first(where: { $0.start <= end && $0.end > end }) {
            end = next.end
        }
        return end
    }

    /// A meeting with other people is under way.
    private static func inMeeting(_ snapshot: DaySnapshot) -> Bool {
        snapshot.agenda.events.contains {
            $0.hasOtherAttendees && !$0.isAllDay && $0.start <= snapshot.now
                && $0.end > snapshot.now
        }
    }

    /// The day's next step once this slot ends, and when it starts: an open
    /// task already under way (one a focus session held back; no time),
    /// else the next event or task.
    private static func nextStep(after slot: Placement, facts: DayFacts) -> (String, Date?)? {
        let slotEnd = end(of: slot)
        for row in TimelineBuilder.build(facts: facts).rows {
            let at = AgendaTime.clock(row.start, calendar: facts.calendar)
            switch row.kind {
            case .event(let event) where row.start >= slotEnd:
                return ("\(event.title) at \(at)", row.start)
            case .task(let task) where !task.isDone && task.id != slot.reminderID:
                if row.start >= slotEnd { return ("\(task.reminder.title) at \(at)", row.start) }
                if (row.end ?? row.start) > slotEnd { return (task.reminder.title, nil) }
            default:
                continue
            }
        }
        return nil
    }

    // MARK: - Focus

    /// The step the owner started that is running now, its task still open:
    /// the menu bar shows its time left, so the time can be seen from any app.
    static func focus(snapshot: DaySnapshot, state: DayState) -> StepFocus? {
        let now = snapshot.now
        let facts = snapshot.facts(state: state)
        for slot in state.plan.sorted(by: { $0.start < $1.start })
        where state.startedSteps.contains(StepCue.key(slot)) && slot.start <= now
            && now < end(of: slot)
        {
            guard let task = facts.task(slot.reminderID) else { continue }
            return StepFocus(reminderID: task.id, title: task.title, end: end(of: slot))
        }
        return nil
    }

    // MARK: - The menu bar's clock

    /// How far ahead the menu bar counts down to what comes next.
    static let clockLead: TimeInterval = 30 * 60

    /// The menu bar's countdown: whichever comes first of the started step's
    /// end, the time to leave for an event in person and an event's start —
    /// the last two only within half an hour. A meeting coming up shows even
    /// over a step that would run into it.
    static func clock(snapshot: DaySnapshot, state: DayState) -> MenuBarClock? {
        let now = snapshot.now
        let horizon = now.addingTimeInterval(clockLead)
        var times: [MenuBarClock] = []
        if let focus = focus(snapshot: snapshot, state: state) {
            times.append(MenuBarClock(kind: .focus, title: focus.title, until: focus.end))
        }
        let ahead = snapshot.agenda.events.filter {
            !$0.isAllDay && $0.start > now && $0.start <= horizon
        }
        if let event = ahead.min(by: { $0.start < $1.start }) {
            times.append(MenuBarClock(kind: .event, title: event.title, until: event.start))
        }
        for departure in state.departures where departure.at > now && departure.at <= horizon {
            // Still on the calendar at the time the plan set it for.
            guard
                let event = snapshot.agenda.events.first(where: {
                    $0.id == departure.eventID && $0.start == departure.eventStart
                })
            else { continue }
            times.append(MenuBarClock(kind: .leave, title: event.title, until: departure.at))
        }
        return times.min { $0.until < $1.until }
    }

    // MARK: - Started

    /// A slot given to a task from now ("Start now" on Today) is started:
    /// it needs no cue of its own, and its end checks in.
    static func markStartedIfNow(_ slot: Placement, snapshot: DaySnapshot, state: inout DayState) {
        guard slot.start <= snapshot.now.addingTimeInterval(60) else { return }
        state.startedSteps.insert(StepCue.key(slot))
        state.cuedSteps[StepCue.key(slot)] = snapshot.now
    }

    // MARK: - The owner's choice

    static func stepChosen(
        _ reminderID: String, _ choice: StepChoice, snapshot: DaySnapshot, state: inout DayState
    ) -> [DayEffect] {
        let minute =
            snapshot.calendar.dateInterval(of: .minute, for: snapshot.now)?.start ?? snapshot.now
        let cuedAt = state.cuedSteps.filter { $0.key.hasPrefix("\(reminderID)@") }.values.max()
        if state.cueOnPanel?.hasPrefix("\(reminderID)@") == true { state.cueOnPanel = nil }
        var effects: [DayEffect] = []
        // A start whose slot went meanwhile (a re-plan) still starts: the
        // task, if still open, gets a slot from now.
        if choice == .start || choice == .startSmall,
            !state.plan.contains(where: { $0.reminderID == reminderID }),
            snapshot.facts(state: state).task(reminderID) != nil
        {
            state.plan.append(
                Placement(reminderID: reminderID, start: minute, minutes: lostSlotMinutes))
        }
        // Going on after a meeting that moved meanwhile: as the quarter hour
        // would.
        var applied = choice
        var resumeAt: Date?
        if choice == .resume {
            let slot = state.plan.first { $0.reminderID == reminderID }
            let started = slot.map { state.startedSteps.contains(StepCue.key($0)) } ?? false
            resumeAt = slot.flatMap {
                resumeTime(
                    for: $0, started: started, now: minute, facts: snapshot.facts(state: state))
            }
            if resumeAt == nil { applied = started ? .extend : .later }
        }
        switch applied {
        case .start:
            if moveSlot(reminderID, to: minute, state: &state) != nil {
                keepClear(reminderID, snapshot: snapshot, state: &state)
                if let slot = state.plan.first(where: { $0.reminderID == reminderID }) {
                    markStartedIfNow(slot, snapshot: snapshot, state: &state)
                }
            }
        case .startSmall:
            // Five minutes from now; its end asks to keep going.
            if let index = state.plan.firstIndex(where: { $0.reminderID == reminderID }) {
                state.plan[index].minutes = smallStartMinutes
            }
            if let slot = moveSlot(reminderID, to: minute, state: &state) {
                markStartedIfNow(slot, snapshot: snapshot, state: &state)
                state.smallStarts.insert(StepCue.key(slot))
            }
        case .later:
            let later = minute.addingTimeInterval(TimeInterval(stepLaterMinutes * 60))
            _ = moveSlot(reminderID, to: later, state: &state)
            keepClear(reminderID, snapshot: snapshot, state: &state)
            state.putOff[reminderID, default: 0] += 1
        case .resume:
            // When the meeting is over, cued again then: a started step for
            // the time cut to end before it (a quarter hour at least), one not
            // begun whole. Not put off: the meeting came first.
            if let resumeAt,
                let index = state.plan.firstIndex(where: { $0.reminderID == reminderID })
            {
                let slot = state.plan[index]
                let cut = state.cutShort.removeValue(forKey: reminderID) ?? 0
                state.plan[index].start = resumeAt
                state.plan[index].minutes =
                    state.startedSteps.contains(StepCue.key(slot))
                    ? max(cut, stepLaterMinutes) : slot.minutes + cut
                state.plan.sort { $0.start < $1.start }
                keepClear(reminderID, snapshot: snapshot, state: &state)
            }
        case .extend:
            // A quarter of an hour on from its end, or from now if that has
            // gone by: a late answer doesn't make it already over. Answered
            // late — the owner was away — it is a fresh block from now, not
            // one stretched back to its first start.
            if let index = state.plan.firstIndex(where: { $0.reminderID == reminderID }) {
                let slot = state.plan[index]
                let quarter = TimeInterval(stepLaterMinutes * 60)
                // Going on past five minutes: no longer a small start.
                state.smallStarts.remove(StepCue.key(slot))
                if minute.timeIntervalSince(end(of: slot)) >= stepCueLate {
                    state.plan[index].start = minute
                    state.plan[index].minutes = stepLaterMinutes
                    markStartedIfNow(state.plan[index], snapshot: snapshot, state: &state)
                    state.plan.sort { $0.start < $1.start }
                } else {
                    let until = max(end(of: slot), minute).addingTimeInterval(quarter)
                    state.plan[index].minutes = Int(until.timeIntervalSince(slot.start) / 60)
                }
            }
        case .tomorrow:
            state.plan.removeAll { $0.reminderID == reminderID }
            effects.append(.mutateAgenda(.dueTomorrow(reminderID: reminderID)))
        case .done:
            effects.append(.mutateAgenda(.complete(reminderID: reminderID)))
        case .dismiss:
            break
        }
        var fields: [String: CompanionTraceValue] = ["action": .string(choice.rawValue)]
        if let cuedAt {
            fields["secondsToReact"] = .double(snapshot.now.timeIntervalSince(cuedAt))
        }
        return effects + [.trace(.cueReaction, fields)]
    }

    /// The task's slot, keeping its length, starting at `start`.
    private static func moveSlot(_ reminderID: String, to start: Date, state: inout DayState)
        -> Placement?
    {
        guard let index = state.plan.firstIndex(where: { $0.reminderID == reminderID }) else {
            return nil
        }
        state.plan[index].start = start
        state.plan.sort { $0.start < $1.start }
        return state.plan.first { $0.reminderID == reminderID }
    }
}
