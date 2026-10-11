//
//  Step.swift
//  tesseract
//
//  A step: one task's slot in today's plan, and its life that day — cued
//  at its start, started, checked in at its end, put off, cut short before
//  a meeting. The plan holds at most one step per task, so a step is known
//  by its task (a reminder's id; a repeating reminder's, its series').
//
//  Its life is bound three ways, and moving the step says which part goes:
//  what happened at its start (cued, started, five minutes, timed) belongs
//  to that start, so a step moved to a new start is cued anew; its end's
//  check-in belongs to that end, so a step made longer or cut short checks
//  in again; how often it was put off, and the minutes a meeting cut, go
//  with the task all day, wherever its step moves. A repeating task's step
//  is its series': the Agenda answers whether that day's occurrence is done.
//

import Foundation

nonisolated struct Step: Sendable, Equatable, Codable {

    /// Which task, from when, how long.
    private(set) var slot: Placement

    // Its start's: a step moved to a new start begins anew.

    /// When its start was cued — or the owner started it, which counts.
    var cuedAt: Date?
    /// The owner started it (Start on a cue, Start now on Today): its end
    /// checks in, and while it runs nothing else is cued. Set by `start(at:)`.
    var isStarted = false
    /// Started as five minutes ("Start 5 min"): its end asks to keep going.
    var isSmall = false
    /// Started whole (not five minutes, not a late quarter hour more), with
    /// these minutes: once seen done, how long it took joins the plan's pace.
    var timedMinutes: Int?

    // Its end's: a step whose end moved checks in again.

    /// When its end checked in.
    var checkedInAt: Date?

    // The task's, all day: carried wherever the step moves.

    /// How often today the owner put it off ("In 15 min"): from the second
    /// time, its start is offered as five minutes.
    var putOff = 0
    /// Minutes taken off it so it ends before the next meeting: going on
    /// after the meeting gives them back.
    var cutMinutes = 0

    init(_ slot: Placement) {
        self.slot = slot
    }

    init(_ reminderID: String, at start: Date, minutes: Int) {
        self.init(Placement(reminderID: reminderID, start: start, minutes: minutes))
    }

    var reminderID: String { slot.reminderID }
    var start: Date { slot.start }
    var minutes: Int { slot.minutes }
    var end: Date { start.addingTimeInterval(TimeInterval(minutes * 60)) }

    /// Its slot holds `date`: from its start, until its end.
    func spans(_ date: Date) -> Bool { start <= date && date < end }

    /// The owner started it and its slot holds `date`.
    func isUnderWay(at date: Date) -> Bool { isStarted && spans(date) }

    // MARK: Moving

    /// The step at another start, keeping its length: a new start is cued
    /// anew, and its end is new too.
    mutating func move(to start: Date) {
        reslot(Placement(reminderID: reminderID, start: start, minutes: minutes))
    }

    /// The step longer or shorter, from the same start: a new end checks
    /// in again.
    mutating func resize(to minutes: Int) {
        reslot(Placement(reminderID: reminderID, start: start, minutes: minutes))
    }

    /// The step in `slot`, its task the same: what its old start and end
    /// held goes with them; what the task carried stays.
    mutating func reslot(_ slot: Placement) {
        if slot.start != start {
            cuedAt = nil
            isStarted = false
            isSmall = false
            timedMinutes = nil
        }
        if slot.start != start || slot.minutes != minutes { checkedInAt = nil }
        self.slot = Placement(reminderID: reminderID, start: slot.start, minutes: slot.minutes)
    }

    // MARK: Its life

    /// The owner started it now: it counts as cued, its end checks in, and
    /// one started `timed` (whole) is timed once seen done.
    mutating func start(at now: Date, timed: Bool = true) {
        isStarted = true
        cuedAt = now
        if timed { timedMinutes = minutes }
    }

    /// The step as started at `now`: a fixture's one line, and the engine's
    /// own word for it.
    func started(at now: Date, timed: Bool = true) -> Step {
        var step = self
        step.start(at: now, timed: timed)
        return step
    }

    /// The cue at its `phase` came off the panel unanswered: it is cued
    /// again by its rules.
    mutating func uncue(_ phase: StepCue.Phase) {
        switch phase {
        case .start: cuedAt = nil
        case .end: checkedInAt = nil
        }
    }

    /// Its latest cue — the start's or the end's check-in — and when.
    var latestCue: (phase: StepCue.Phase, at: Date)? {
        switch (cuedAt, checkedInAt) {
        case (let start?, let end?): end >= start ? (.end, end) : (.start, start)
        case (let start?, nil): (.start, start)
        case (nil, let end?): (.end, end)
        case (nil, nil): nil
        }
    }

    // MARK: Saved

    private enum CodingKeys: String, CodingKey {
        case reminderID, start, minutes, cuedAt, isStarted, isSmall, timedMinutes, checkedInAt
        case putOff, cutMinutes
    }

    /// A plan saved before steps had a life (its slots alone) still loads:
    /// every part of the life is optional on disk.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        slot = Placement(
            reminderID: try c.decode(String.self, forKey: .reminderID),
            start: try c.decode(Date.self, forKey: .start),
            minutes: try c.decode(Int.self, forKey: .minutes))
        cuedAt = try c.decodeIfPresent(Date.self, forKey: .cuedAt)
        isStarted = try c.decodeIfPresent(Bool.self, forKey: .isStarted) ?? false
        isSmall = try c.decodeIfPresent(Bool.self, forKey: .isSmall) ?? false
        timedMinutes = try c.decodeIfPresent(Int.self, forKey: .timedMinutes)
        checkedInAt = try c.decodeIfPresent(Date.self, forKey: .checkedInAt)
        putOff = try c.decodeIfPresent(Int.self, forKey: .putOff) ?? 0
        cutMinutes = try c.decodeIfPresent(Int.self, forKey: .cutMinutes) ?? 0
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(reminderID, forKey: .reminderID)
        try c.encode(start, forKey: .start)
        try c.encode(minutes, forKey: .minutes)
        try c.encodeIfPresent(cuedAt, forKey: .cuedAt)
        if isStarted { try c.encode(isStarted, forKey: .isStarted) }
        if isSmall { try c.encode(isSmall, forKey: .isSmall) }
        try c.encodeIfPresent(timedMinutes, forKey: .timedMinutes)
        try c.encodeIfPresent(checkedInAt, forKey: .checkedInAt)
        if putOff > 0 { try c.encode(putOff, forKey: .putOff) }
        if cutMinutes > 0 { try c.encode(cutMinutes, forKey: .cutMinutes) }
    }
}

/// Which step's cue is on the Jarvis Panel, until the owner answers it: the
/// task's, at its start or its end.
nonisolated struct CueOnPanel: Sendable, Equatable, Codable {
    var reminderID: String
    var phase: StepCue.Phase
}
