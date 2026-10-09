//
//  DayEngine+Steps.swift
//  tesseract
//
//  The plan keeps its own time. A Morning Plan's slots and the owner's own
//  "Do it at 16:30" live in the day's state, not in Reminders, so no alarm
//  marks them. When a planned step's slot starts and the owner is at the
//  Mac, code puts it on the Jarvis Panel — the Step Cue — with one-click
//  choices: Start, In 15 min, Tomorrow, Done. Help lands at the moment of
//  doing, not in a list the owner has to remember to read.
//
//  No model. Each slot is cued once, at its start; never while the owner is
//  away, in quiet hours, a call, a game or a meeting, and never over a panel
//  the owner hasn't closed. A slot whose reminder rings at the same minute
//  is left to Reminders. Closing the cue changes nothing.
//

import Foundation

nonisolated extension DayEngine {

    /// A cue is for a slot's start: past this, the Now Card on Today has the
    /// step, and a panel would only interrupt.
    static let stepCueWindow: TimeInterval = 10 * 60
    /// "In 15 min" moves a slot this far.
    static let stepLaterMinutes = 15

    // MARK: - The cue

    static func stepCueIfDue(snapshot: DaySnapshot, state: inout DayState) -> [DayEffect] {
        guard !snapshot.panelUp,
            DeliveryLadder.rungs(for: .normal, snapshot: snapshot).contains(.panel),
            !inMeeting(snapshot)
        else { return [] }
        let now = snapshot.now
        let facts = snapshot.facts(state: state)
        let starting = state.plan.filter { slot in
            let end = slot.start.addingTimeInterval(TimeInterval(slot.minutes * 60))
            return slot.start <= now && now < end
                && now.timeIntervalSince(slot.start) < stepCueWindow
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
            let cue = StepCue(
                reminderID: task.id, title: task.title, start: slot.start,
                minutes: slot.minutes, areaName: facts.areaName(of: task),
                isMustDo: state.mustDoID == task.id, next: nextStep(after: slot, facts: facts))
            return [
                .presentStep(cue),
                .trace(
                    .cuePresented,
                    [
                        "late": .int(Int(now.timeIntervalSince(slot.start))),
                        "minutes": .int(slot.minutes), "mustDo": .bool(cue.isMustDo),
                    ]),
            ]
        }
        return []
    }

    /// A meeting with other people is under way.
    private static func inMeeting(_ snapshot: DaySnapshot) -> Bool {
        snapshot.agenda.events.contains {
            $0.hasOtherAttendees && !$0.isAllDay && $0.start <= snapshot.now
                && $0.end > snapshot.now
        }
    }

    /// The day's next event or open task once this slot ends.
    private static func nextStep(after slot: Placement, facts: DayFacts) -> String? {
        let end = slot.start.addingTimeInterval(TimeInterval(slot.minutes * 60))
        for row in TimelineBuilder.build(facts: facts).rows where row.start >= end {
            let at = AgendaTime.clock(row.start, calendar: facts.calendar)
            switch row.kind {
            case .event(let event): return "\(event.title) at \(at)"
            case .task(let task) where !task.isDone && task.id != slot.reminderID:
                return "\(task.reminder.title) at \(at)"
            default: continue
            }
        }
        return nil
    }

    // MARK: - The owner's choice

    static func stepChosen(
        _ reminderID: String, _ choice: StepChoice, snapshot: DaySnapshot, state: inout DayState
    ) -> [DayEffect] {
        let minute =
            snapshot.calendar.dateInterval(of: .minute, for: snapshot.now)?.start ?? snapshot.now
        let cuedAt = state.cuedSteps.filter { $0.key.hasPrefix("\(reminderID)@") }.values.max()
        var effects: [DayEffect] = []
        switch choice {
        case .start:
            if let slot = moveSlot(reminderID, to: minute, state: &state) {
                // Started: this start needs no cue of its own.
                state.cuedSteps[StepCue.key(slot)] = snapshot.now
            }
        case .later:
            let later = minute.addingTimeInterval(TimeInterval(stepLaterMinutes * 60))
            _ = moveSlot(reminderID, to: later, state: &state)
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
