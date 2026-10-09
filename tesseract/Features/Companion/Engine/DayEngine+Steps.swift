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
//  No model. Each start and each end is cued once, within ten minutes;
//  never while the owner is away, in quiet hours, a call, a game or a
//  meeting, never over a panel the owner hasn't closed, and no start while a
//  step the owner started is running. A slot whose reminder rings at the
//  same minute is left to Reminders. Closing a cue changes nothing.
//

import Foundation

nonisolated extension DayEngine {

    /// A cue is for a slot's start or end: past this, the Now Card on Today
    /// has the step, and a panel would only interrupt.
    static let stepCueWindow: TimeInterval = 10 * 60
    /// "In 15 min" moves a slot this far; "15 more min" makes it this much
    /// longer.
    static let stepLaterMinutes = 15

    // MARK: - The cue

    static func stepCueIfDue(snapshot: DaySnapshot, state: inout DayState) -> [DayEffect] {
        // Once the owner sat down to start the day, the morning's end of
        // quiet hours doesn't hold the plan back.
        let started = DeliveryLadder.dayStarted(state.satDownAt, snapshot: snapshot)
        guard snapshot.settings.stepCues, !snapshot.panelUp,
            DeliveryLadder.rungs(for: .normal, snapshot: snapshot, sittingDown: started)
                .contains(.panel),
            !inMeeting(snapshot)
        else { return [] }
        // A step that is over comes first: it closes what the next one opens.
        return stepEndIfDue(snapshot: snapshot, state: &state)
            ?? stepStartIfDue(snapshot: snapshot, state: &state) ?? []
    }

    /// The first slot starting now, for an open task not yet cued.
    private static func stepStartIfDue(snapshot: DaySnapshot, state: inout DayState)
        -> [DayEffect]?
    {
        let now = snapshot.now
        let facts = snapshot.facts(state: state)
        // A slot starting now, or one held back by a focus session and still
        // running.
        let starting = state.plan.filter { slot in
            slot.start <= now && now < end(of: slot)
                && (now.timeIntervalSince(slot.start) < stepCueWindow
                    || state.heldSteps.contains(StepCue.key(slot)))
                && state.cuedSteps[StepCue.key(slot)] == nil
        }
        // A step the owner started is still running: a focus session isn't
        // interrupted. What would start meanwhile is held, its check-in names
        // it, and it is cued once the owner is free.
        let focused = state.plan.contains { slot in
            state.startedSteps.contains(StepCue.key(slot)) && slot.start <= now
                && now < end(of: slot) && facts.task(slot.reminderID) != nil
        }
        guard !focused else {
            for slot in starting where facts.task(slot.reminderID) != nil {
                state.heldSteps.insert(StepCue.key(slot))
            }
            return nil
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
                cue(task, slot: slot, phase: .start, facts: facts, state: state),
                late: now.timeIntervalSince(slot.start))
        }
        return nil
    }

    /// A card took the panel from the cue on it: the cue is held, to come
    /// back by the rules above once the panel is free again.
    static func holdCueUnderCard(
        _ effects: [DayEffect], snapshot: DaySnapshot, state: inout DayState
    ) {
        guard let key = state.cueOnPanel, snapshot.panelUp,
            effects.contains(where: { if case .presentCard(_, .panel) = $0 { true } else { false } }
            )
        else { return }
        state.cueOnPanel = nil
        state.cuedSteps[key] = nil
        state.heldSteps.insert(key)
    }

    /// The first slot the owner started whose time is up, its task still open.
    private static func stepEndIfDue(snapshot: DaySnapshot, state: inout DayState)
        -> [DayEffect]?
    {
        let now = snapshot.now
        let facts = snapshot.facts(state: state)
        let ended = state.plan.filter { slot in
            state.startedSteps.contains(StepCue.key(slot)) && end(of: slot) <= now
                && (now.timeIntervalSince(end(of: slot)) < stepCueWindow
                    || state.heldSteps.contains(StepCue.endKey(slot)))
                && state.cuedSteps[StepCue.endKey(slot)] == nil
        }
        for slot in ended.sorted(by: { $0.start < $1.start }) {
            guard let task = facts.task(slot.reminderID) else { continue }
            state.cuedSteps[StepCue.endKey(slot)] = now
            state.cueOnPanel = StepCue.endKey(slot)
            return present(
                cue(task, slot: slot, phase: .end, facts: facts, state: state),
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
        state: DayState
    ) -> StepCue {
        StepCue(
            reminderID: task.id, title: task.title, start: slot.start, minutes: slot.minutes,
            areaName: facts.areaName(of: task), isMustDo: state.mustDoID == task.id,
            next: nextStep(after: slot, facts: facts), phase: phase)
    }

    private static func end(of slot: Placement) -> Date {
        slot.start.addingTimeInterval(TimeInterval(slot.minutes * 60))
    }

    /// A meeting with other people is under way.
    private static func inMeeting(_ snapshot: DaySnapshot) -> Bool {
        snapshot.agenda.events.contains {
            $0.hasOtherAttendees && !$0.isAllDay && $0.start <= snapshot.now
                && $0.end > snapshot.now
        }
    }

    /// The day's next step once this slot ends: an open task already under
    /// way (one a focus session held back), else the next event or task.
    private static func nextStep(after slot: Placement, facts: DayFacts) -> String? {
        let slotEnd = end(of: slot)
        for row in TimelineBuilder.build(facts: facts).rows {
            let at = AgendaTime.clock(row.start, calendar: facts.calendar)
            switch row.kind {
            case .event(let event) where row.start >= slotEnd:
                return "\(event.title) at \(at)"
            case .task(let task) where !task.isDone && task.id != slot.reminderID:
                if row.start >= slotEnd { return "\(task.reminder.title) at \(at)" }
                if (row.end ?? row.start) > slotEnd { return task.reminder.title }
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
        switch choice {
        case .start:
            if let slot = moveSlot(reminderID, to: minute, state: &state) {
                markStartedIfNow(slot, snapshot: snapshot, state: &state)
            }
        case .later:
            let later = minute.addingTimeInterval(TimeInterval(stepLaterMinutes * 60))
            _ = moveSlot(reminderID, to: later, state: &state)
        case .extend:
            // A quarter of an hour on from now, or from its end if that is
            // still ahead: a late answer doesn't make it already over.
            if let index = state.plan.firstIndex(where: { $0.reminderID == reminderID }) {
                let slot = state.plan[index]
                let from = max(end(of: slot), minute)
                let until = from.addingTimeInterval(TimeInterval(stepLaterMinutes * 60))
                state.plan[index].minutes = Int(until.timeIntervalSince(slot.start) / 60)
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
