//
//  DayEngine+Steps.swift
//  tesseract
//
//  The plan keeps its own time. A Morning Plan's slots and the owner's own
//  "Do it at 16:30" live in the day's state, not in Reminders, so no alarm
//  marks them. When a planned step's slot starts and the owner is at the
//  Mac, code puts it on the Jarvis Panel — the Step Cue — with one-click
//  choices: Start, In 15 min, Tomorrow, Did it. A step the owner started (Start
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
//  once it is over instead, with the time that was cut. A meeting that lands
//  on the plan later moves a step not begun to the first free time after
//  it, and ends a started one five minutes before it.
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
            // Jarvis is thinking: his card may take the panel any moment, and
            // a cue up first would be replaced, then come back after it.
            state.running == nil,
            DeliveryLadder.rungs(for: .normal, snapshot: snapshot, sittingDown: started)
                .contains(.panel),
            !inMeeting(snapshot)
        else { return [] }
        // A step the owner started is still running: a focus session isn't
        // interrupted. What comes due meanwhile waits, its check-in names
        // what is under way, and it is cued once the owner is free — but a
        // break isn't held past half an hour: a long step gets one midway.
        if let focus = Timeline(snapshot: snapshot, state: state).focus {
            let left = focus.end.timeIntervalSince(snapshot.now)
            guard left >= TimeInterval(breakHeldMinutes * 60) else { return [] }
            return breakCueIfDue(snapshot: snapshot, state: &state, during: focus) ?? []
        }
        let steps = snapshot.settings.stepCues
        // A step that is over comes first: it closes what the next one opens.
        // A break comes between it and the next step: the time to take one.
        if steps, let end = stepEndIfDue(snapshot: snapshot, state: &state) { return end }
        if let rest = breakCueIfDue(snapshot: snapshot, state: &state) { return rest }
        return steps ? stepStartIfDue(snapshot: snapshot, state: &state) ?? [] : []
    }

    /// The first step whose slot is under way, for an open task not yet cued.
    private static func stepStartIfDue(snapshot: DaySnapshot, state: inout DayState)
        -> [DayEffect]?
    {
        let now = snapshot.now
        let day = Timeline(snapshot: snapshot, state: state)
        let starting = state.plan.filter { $0.spans(now) && $0.cuedAt == nil }
        for step in starting.sorted(by: { $0.start < $1.start }) {
            // Done, or gone from Reminders: nothing to start.
            guard let task = day.facts.task(step.reminderID),
                let index = state.plan.firstIndex(where: { $0.reminderID == step.reminderID })
            else { continue }
            state.plan[index].cuedAt = now
            // Its own alarm rings now: Reminders has it.
            if task.dueHasTime, let due = task.due, abs(due.timeIntervalSince(step.start)) < 60 {
                continue
            }
            state.cueOnPanel = CueOnPanel(reminderID: step.reminderID, phase: .start)
            return present(
                cue(task, step: step, phase: .start, day: day, state: state),
                late: now.timeIntervalSince(step.start))
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
        state.uncuePanel()
        state.breakCuedAt = nil
    }

    /// A step the owner started whose time is up, its task still open: the
    /// one that ended last first, the step they were just on.
    private static func stepEndIfDue(snapshot: DaySnapshot, state: inout DayState)
        -> [DayEffect]?
    {
        let now = snapshot.now
        let day = Timeline(snapshot: snapshot, state: state)
        let ended = state.plan.filter { $0.isStarted && $0.end <= now && $0.checkedInAt == nil }
        for step in ended.sorted(by: { $0.end > $1.end }) {
            guard let task = day.facts.task(step.reminderID),
                let index = state.plan.firstIndex(where: { $0.reminderID == step.reminderID })
            else { continue }
            state.plan[index].checkedInAt = now
            state.cueOnPanel = CueOnPanel(reminderID: step.reminderID, phase: .end)
            return present(
                cue(task, step: step, phase: .end, day: day, state: state),
                late: now.timeIntervalSince(step.end))
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
        _ task: AgendaReminder, step: Step, phase: StepCue.Phase, day: Timeline, state: DayState
    ) -> StepCue {
        let (facts, now) = (day.facts, day.facts.now)
        let moment = phase == .start ? step.start : step.end
        let next = day.nextStep(after: step.slot)
        return StepCue(
            reminderID: task.id, title: task.title, start: step.start, minutes: step.minutes,
            areaName: facts.areaName(of: task), isMustDo: state.mustDoID == task.id,
            next: next?.title, nextAt: next?.at, nextLink: next?.link, phase: phase,
            late: now.timeIntervalSince(moment) >= stepCueLate, putOff: step.putOff,
            small: step.isSmall,
            resumeAt: resumeTime(for: step.slot, started: phase == .end, now: now, facts: facts))
    }

    private static func end(of slot: Placement) -> Date {
        slot.start.addingTimeInterval(TimeInterval(slot.minutes * 60))
    }

    // MARK: - Clear of meetings

    // A meeting ends a step (`AgendaEvent.isMeeting`: other people, or a
    // place to be), and so does the time to leave for one; a block of the
    // owner's own ("Work", 09:00–13:00) is where steps happen, and ends none.

    /// How long a step from `start` gets so it keeps clear of what's fixed
    /// after it: it ends `transitionMinutes` before the next meeting, or the
    /// time to leave for one, it would run into — right at it when that
    /// leaves too little; with not even five minutes before it, as wanted.
    static func clearMinutes(from start: Date, wanted: Int, facts: DayFacts) -> Int {
        let end = start.addingTimeInterval(TimeInterval(wanted * 60))
        let fixed =
            facts.meetingsToday.map(\.start) + facts.departuresToMeetings.map(\.departure.at)
        guard let next = fixed.filter({ $0 > start && $0 <= end }).min() else { return wanted }
        let room = Int(next.timeIntervalSince(start) / 60)
        if room - transitionMinutes >= smallStartMinutes { return room - transitionMinutes }
        return room >= smallStartMinutes ? room : wanted
    }

    /// The task's step, cut to keep clear of the next meeting; what was cut
    /// is kept, for going on after it.
    static func keepClear(_ reminderID: String, snapshot: DaySnapshot, state: inout DayState) {
        guard let index = state.plan.firstIndex(where: { $0.reminderID == reminderID }) else {
            return
        }
        let step = state.plan[index]
        let minutes = clearMinutes(
            from: step.start, wanted: step.minutes, facts: snapshot.facts(state: state))
        guard minutes < step.minutes else { return }
        state.plan[index].resize(to: minutes)
        state.plan[index].cutMinutes = step.minutes - minutes
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
        let meetings = facts.meetingsToday
        var coming = meetings.filter { $0.start > now && $0.start < limit }.map {
            (at: $0.start, end: $0.end)
        }
        for leaving in facts.departuresToMeetings
        where leaving.departure.at > now && leaving.departure.at < limit {
            coming.append((at: leaving.departure.at, end: leaving.event.end))
        }
        guard var end = coming.min(by: { $0.at < $1.at })?.end else { return nil }
        // Back to back: after the last of the run.
        while let next = meetings.first(where: { $0.start <= end && $0.end > end }) {
            end = next.end
        }
        return end
    }

    /// The agenda changed. A meeting (or somewhere to be) now on a planned
    /// step not yet begun moves the step to the first free time after it,
    /// as long — clear of other meetings, the way to one and the plan's
    /// other steps — before the evening (an evening step, before midnight);
    /// with none, it stays, to slide where Today and the wrap-up see it. In
    /// a meeting at its time, the owner would never get its cue. A step
    /// the owner started now ends five minutes before one, as a start does.
    static func clearStepsOfMeetings(snapshot: DaySnapshot, state: inout DayState)
        -> [DayEffect]
    {
        let day = Timeline(snapshot: snapshot, state: state)
        let facts = day.facts
        let meetings = facts.meetingsToday
        guard !meetings.isEmpty else { return [] }
        if let focus = day.focus {
            keepClear(focus.reminderID, snapshot: snapshot, state: &state)
        }
        let calendar = snapshot.calendar
        let minutes = snapshot.settings.eveningMinutes
        guard
            let evening = calendar.date(
                bySettingHour: minutes / 60, minute: minutes % 60, second: 0, of: snapshot.now),
            let midnight = calendar.date(
                byAdding: .day, value: 1, to: calendar.startOfDay(for: snapshot.now))
        else { return [] }
        let away = facts.departuresToMeetings.compactMap(\.wayThere)
        var effects: [DayEffect] = []
        for step in state.plan.sorted(by: { $0.start < $1.start }) {
            let span = DateInterval(start: step.start, end: step.end)
            guard step.start > snapshot.now, !step.isStarted, step.cuedAt == nil,
                facts.task(step.reminderID) != nil,
                let hit = meetings.filter({ $0.start < span.end && $0.end > span.start })
                    .max(by: { $0.end < $1.end })
            else { continue }
            let others = state.plan.filter { $0.reminderID != step.reminderID }
                .map { DateInterval(start: $0.start, end: $0.end) }
            let busy = meetings.map { DateInterval(start: $0.start, end: $0.end) } + away + others
            guard
                let start = firstFree(
                    from: hit.end, minutes: step.minutes, busy: busy,
                    until: step.start < evening ? evening : midnight, calendar: calendar),
                let index = state.plan.firstIndex(where: { $0.reminderID == step.reminderID })
            else { continue }
            state.plan[index].move(to: start)
            effects.append(
                .trace(
                    .cueMoved,
                    [
                        "minutes": .int(Int(start.timeIntervalSince(step.start) / 60)),
                        "length": .int(step.minutes),
                    ]))
        }
        state.plan.sort { $0.start < $1.start }
        return effects
    }

    /// The first start from `from` — on the next five minutes — where
    /// `minutes` fit clear of `busy`, ending by `until`.
    static func firstFree(
        from: Date, minutes: Int, busy: [DateInterval], until: Date, calendar: Calendar
    ) -> Date? {
        var start = onTheFive(from, calendar: calendar)
        let length = TimeInterval(minutes * 60)
        // Each pass moves past a block, so the plan's handful ends it soon.
        for _ in 0..<64 {
            let span = DateInterval(start: start, duration: length)
            guard span.end <= until else { return nil }
            guard
                let block = busy.filter({ $0.start < span.end && $0.end > span.start })
                    .max(by: { $0.end < $1.end })
            else { return start }
            start = onTheFive(block.end, calendar: calendar)
        }
        return nil
    }

    private static func onTheFive(_ date: Date, calendar: Calendar) -> Date {
        let minute = calendar.dateInterval(of: .minute, for: date)?.start ?? date
        let over = calendar.component(.minute, from: minute) % 5
        if minute == date, over == 0 { return date }
        return minute.addingTimeInterval(TimeInterval((5 - over) * 60))
    }

    /// A meeting with other people is under way.
    private static func inMeeting(_ snapshot: DaySnapshot) -> Bool {
        snapshot.agenda.events.contains {
            $0.hasOtherAttendees && !$0.isAllDay && $0.start <= snapshot.now
                && $0.end > snapshot.now
        }
    }

    // MARK: - Started

    /// A step given a slot from now ("Start now" on Today) is started: it
    /// needs no cue of its own, and its end checks in. One started whole
    /// (`timed`: not five minutes, not a late quarter hour more) is timed.
    static func markStartedIfNow(
        _ reminderID: String, snapshot: DaySnapshot, state: inout DayState, timed: Bool = true
    ) {
        guard let index = state.plan.firstIndex(where: { $0.reminderID == reminderID }),
            state.plan[index].start <= snapshot.now.addingTimeInterval(60)
        else { return }
        state.plan[index] = state.plan[index].started(at: snapshot.now, timed: timed)
    }

    /// The task's step, its slot from now, started: cut clear of the next
    /// meeting first, then started, so a whole start is timed against the
    /// minutes it has. Today's Start now and a cue's Start are this one
    /// operation.
    static func startNow(_ reminderID: String, snapshot: DaySnapshot, state: inout DayState) {
        keepClear(reminderID, snapshot: snapshot, state: &state)
        markStartedIfNow(reminderID, snapshot: snapshot, state: &state)
    }

    /// A step started whole, now seen done: how long it took against the
    /// minutes it had, kept for the plan's sizing — unless done well after
    /// its time was up (the owner away when it checked in), when how long
    /// it took is unknown.
    static func noteStepRuns(snapshot: DaySnapshot, state: inout DayState) -> [DayEffect] {
        var effects: [DayEffect] = []
        for index in state.plan.indices {
            let step = state.plan[index]
            // A repeating task's step is done when its occurrence today is.
            guard let planned = step.timedMinutes,
                let completed = snapshot.agenda.doneOccurrence(
                    of: step.reminderID, on: state.day, calendar: snapshot.calendar)?.completedAt,
                completed >= step.start
            else { continue }
            state.plan[index].timedMinutes = nil
            guard completed <= step.end.addingTimeInterval(stepCueLate) else { continue }
            let actual = max(1, Int(completed.timeIntervalSince(step.start) / 60))
            state.stepRuns.append(StepRun(planned: planned, actual: actual, at: completed))
            state.stepRuns = Array(state.stepRuns.suffix(30))
            effects.append(.trace(.cueTimed, ["planned": .int(planned), "actual": .int(actual)]))
        }
        return effects
    }

    // MARK: - The owner's choice

    static func stepChosen(
        _ reminderID: String, _ choice: StepChoice, snapshot: DaySnapshot, state: inout DayState
    ) -> [DayEffect] {
        let minute =
            snapshot.calendar.dateInterval(of: .minute, for: snapshot.now)?.start ?? snapshot.now
        let cuedAt = state.step(reminderID)?.latestCue?.at
        if state.cueOnPanel?.reminderID == reminderID { state.cueOnPanel = nil }
        var effects: [DayEffect] = []
        // A start whose slot went meanwhile (a re-plan) still starts: the
        // task, if still open, gets a slot from now.
        if choice == .start || choice == .startSmall, state.step(reminderID) == nil,
            snapshot.facts(state: state).task(reminderID) != nil
        {
            state.plan.append(Step(reminderID, at: minute, minutes: lostSlotMinutes))
        }
        // Going on after a meeting that moved meanwhile: as the quarter hour
        // would.
        var applied = choice
        var resumeAt: Date?
        if choice == .resume {
            let step = state.step(reminderID)
            let started = step?.isStarted ?? false
            resumeAt = step.flatMap {
                resumeTime(
                    for: $0.slot, started: started, now: minute,
                    facts: snapshot.facts(state: state))
            }
            if resumeAt == nil { applied = started ? .extend : .later }
        }
        let index = state.plan.firstIndex { $0.reminderID == reminderID }
        switch applied {
        case .start:
            if move(reminderID, to: minute, state: &state) {
                startNow(reminderID, snapshot: snapshot, state: &state)
            }
        case .startSmall:
            // Five minutes from now; its end asks to keep going.
            if let index { state.plan[index].resize(to: smallStartMinutes) }
            if move(reminderID, to: minute, state: &state) {
                markStartedIfNow(reminderID, snapshot: snapshot, state: &state, timed: false)
                if let index = state.plan.firstIndex(where: { $0.reminderID == reminderID }) {
                    state.plan[index].isSmall = true
                }
            }
        case .later:
            let later = minute.addingTimeInterval(TimeInterval(stepLaterMinutes * 60))
            if let index { state.plan[index].putOff += 1 }
            _ = move(reminderID, to: later, state: &state)
            keepClear(reminderID, snapshot: snapshot, state: &state)
        case .resume:
            // When the meeting is over, cued again then: a started step for
            // the time cut to end before it (a quarter hour at least), one not
            // begun whole. Not put off: the meeting came first.
            if let resumeAt, let index {
                let step = state.plan[index]
                state.plan[index].cutMinutes = 0
                state.plan[index].reslot(
                    Placement(
                        reminderID: reminderID, start: resumeAt,
                        minutes: step.isStarted
                            ? max(step.cutMinutes, stepLaterMinutes)
                            : step.minutes + step.cutMinutes))
                state.plan.sort { $0.start < $1.start }
                keepClear(reminderID, snapshot: snapshot, state: &state)
            }
        case .extend:
            // A quarter of an hour on from its end, or from now if that has
            // gone by: a late answer doesn't make it already over. Answered
            // late — the owner was away — it is a fresh block from now, not
            // one stretched back to its first start.
            if let index {
                let step = state.plan[index]
                let quarter = TimeInterval(stepLaterMinutes * 60)
                // Going on past five minutes: no longer a small start.
                state.plan[index].isSmall = false
                if minute.timeIntervalSince(step.end) >= stepCueLate {
                    state.plan[index].reslot(
                        Placement(
                            reminderID: reminderID, start: minute, minutes: stepLaterMinutes))
                    markStartedIfNow(reminderID, snapshot: snapshot, state: &state, timed: false)
                    state.plan.sort { $0.start < $1.start }
                } else {
                    let until = max(step.end, minute).addingTimeInterval(quarter)
                    state.plan[index].resize(to: Int(until.timeIntervalSince(step.start) / 60))
                }
            }
        case .tomorrow:
            state.plan.removeAll { $0.reminderID == reminderID }
            effects.append(.mutateAgenda(.dueTomorrow(reminderID: reminderID)))
        case .done:
            effects.append(.mutateAgenda(.setDone(reminderID: reminderID, done: true)))
        case .undo:
            effects.append(.mutateAgenda(.setDone(reminderID: reminderID, done: false)))
            // The cue the owner answered is back on the panel.
            state.cueOnPanel = state.step(reminderID)?.latestCue.map {
                CueOnPanel(reminderID: reminderID, phase: $0.phase)
            }
        case .dismiss:
            break
        }
        var fields: [String: CompanionTraceValue] = ["action": .string(choice.rawValue)]
        if let cuedAt {
            fields["secondsToReact"] = .double(snapshot.now.timeIntervalSince(cuedAt))
        }
        return effects + [.trace(.cueReaction, fields)]
    }

    /// The task's step, keeping its length, at `start`; false when the plan
    /// has none.
    private static func move(_ reminderID: String, to start: Date, state: inout DayState) -> Bool {
        guard let index = state.plan.firstIndex(where: { $0.reminderID == reminderID }) else {
            return false
        }
        state.plan[index].move(to: start)
        state.plan.sort { $0.start < $1.start }
        return true
    }
}
