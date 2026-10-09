//
//  DayEngine+Breaks.swift
//  tesseract
//
//  The body keeps time too. Deep in something, the owner forgets water,
//  food and standing up — hyperfocus is the other side of ADHD — and the
//  Companion counted the day's steps but never the hours at the Mac. After
//  two hours at it with no break of five minutes or more, code puts a Break
//  Cue on the Jarvis Panel: how long, since when, one small thing to do
//  (stand up, water, look away), with Taking 5 and In 30 min.
//
//  No model. It keeps the Step Cue's manners — never while away, in quiet
//  hours, a call, a game or a meeting, never over a panel that is up — and
//  lands between steps: a step's check-in comes before it and the next
//  step's start after it. A step the owner started holds it back only when
//  it ends within half an hour (its check-in is the break); a longer one
//  gets it midway: a Morning Plan gave the must-do a two-and-a-half-hour
//  slot, and holding the break for all of it would make four hours at the
//  Mac. Five minutes away is a break, whatever the cue said, and a cue
//  still on the panel then comes down. Taking 5 starts the count again, In
//  30 min asks again then, and closing it holds it for two hours.
//

import Foundation

nonisolated extension DayEngine {

    /// At the Mac this long without a break: a Break Cue.
    static let breakAfterMinutes = 120
    /// Away this long is a break: the time at the Mac starts again.
    static let breakAwayMinutes = 5
    /// "In 30 min" asks again this much later.
    static let breakLaterMinutes = 30
    /// A step the owner started holds a due break back if it ends within
    /// this long; a longer one gets the break midway.
    static let breakHeldMinutes = 30

    /// Back from `awayFrom`: five minutes or more is a break — the time at
    /// the Mac starts again, and a Break Cue still on the panel comes down.
    static func backFromAway(awayFrom: Date, snapshot: DaySnapshot, state: inout DayState)
        -> [DayEffect]
    {
        guard snapshot.now.timeIntervalSince(awayFrom) >= TimeInterval(breakAwayMinutes * 60)
        else { return [] }
        state.sittingSince = snapshot.now
        state.breakNotBefore = nil
        guard let cuedAt = state.breakCuedAt else { return [] }
        state.breakCuedAt = nil
        // Up from the Mac with the cue on the panel: it was taken.
        return [
            .retractBreak,
            .trace(
                .cueReaction,
                [
                    "phase": "break", "action": "away",
                    "secondsToReact": .double(max(0, awayFrom.timeIntervalSince(cuedAt))),
                ]),
        ]
    }

    /// Two hours at the Mac with no break, and nothing asked to wait: a
    /// Break Cue — naming the step it comes `during`, which runs on. Called
    /// once the panel is free (`cueIfDue`): one that left the panel
    /// unanswered comes back.
    static func breakCueIfDue(
        snapshot: DaySnapshot, state: inout DayState, during focus: StepFocus? = nil
    ) -> [DayEffect]? {
        guard snapshot.settings.breakCues, let since = state.sittingSince else { return nil }
        let due = max(
            since.addingTimeInterval(TimeInterval(breakAfterMinutes * 60)),
            state.breakNotBefore ?? .distantPast)
        guard snapshot.now >= due else { return nil }
        state.breakCuedAt = snapshot.now
        state.breakCues += 1
        let minutes = Int(snapshot.now.timeIntervalSince(since) / 60)
        let cue = BreakCue(
            since: since, minutes: minutes, number: state.breakCues, step: focus?.title,
            stepEnd: focus?.end)
        return [
            .presentBreak(cue),
            .trace(
                .cuePresented,
                [
                    "phase": "break", "minutes": .int(minutes),
                    "late": .int(Int(snapshot.now.timeIntervalSince(due))),
                ]),
        ]
    }

    static func breakChosen(_ choice: BreakChoice, snapshot: DaySnapshot, state: inout DayState)
        -> [DayEffect]
    {
        var fields: [String: CompanionTraceValue] = [
            "phase": "break", "action": .string(choice.rawValue),
        ]
        if let cuedAt = state.breakCuedAt {
            fields["secondsToReact"] = .double(snapshot.now.timeIntervalSince(cuedAt))
        }
        state.breakCuedAt = nil
        switch choice {
        case .taking:
            state.sittingSince = snapshot.now
            state.breakNotBefore = nil
        case .later:
            state.breakNotBefore = snapshot.now.addingTimeInterval(
                TimeInterval(breakLaterMinutes * 60))
        case .dismiss:
            state.breakNotBefore = snapshot.now.addingTimeInterval(
                TimeInterval(breakAfterMinutes * 60))
        }
        return [.trace(.cueReaction, fields)]
    }
}
