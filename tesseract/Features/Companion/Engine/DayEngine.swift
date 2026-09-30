//
//  DayEngine.swift
//  tesseract
//
//  The Day Engine: the Companion's pure decider, in the gather → decide →
//  perform shape. The loop gathers a snapshot of the world (agenda, presence,
//  time, power) and feeds the engine one signal at a time; the engine returns
//  the day's next state and an ordered list of effects, and the loop performs
//  them. The engine performs no I/O, reads no clock and holds no references,
//  so every rule is a row in a decision table.
//
//  "Jarvis thinks at moments; code keeps the promises": the engine decides
//  when a moment is due and what the owner should see, and every promise it
//  makes (a nudge, a reminder, a card) is kept by code and the OS.
//

import Foundation

// MARK: - Signals

nonisolated enum DaySignal: Sendable, Equatable {
    /// The loop's clock, about once a minute.
    case tick
    /// Reminders or Calendar changed.
    case agendaChanged
    /// The Companion was switched off: everything it scheduled is withdrawn.
    case companionDisabled
}

// MARK: - Effects

nonisolated enum DayEffect: Sendable, Equatable {
    /// Make the OS have exactly these event nudges scheduled.
    case syncNudges([Nudge])
    /// Write one Companion Trace event.
    case trace(CompanionTraceEvent, [String: CompanionTraceValue])
}

// MARK: - Snapshot

/// What the loop gathered before this signal.
nonisolated struct DaySnapshot: Sendable, Equatable {
    var now: Date
    var calendar: Calendar
    var settings: DaySettings
    var agenda: AgendaSnapshot

    init(now: Date, calendar: Calendar = .current, settings: DaySettings, agenda: AgendaSnapshot) {
        self.now = now
        self.calendar = calendar
        self.settings = settings
        self.agenda = agenda
    }
}

/// The owner's Companion settings, as the engine reads them.
nonisolated struct DaySettings: Sendable, Equatable {
    var nudgeLeadMinutes: Int = 10
    var morningStartHour: Int = 4
    var morningEndHour: Int = 12
    var eveningMinutes: Int = 21 * 60
    var breakpointAwayMinutes: Int = 10
    var speaks: Bool = true
    var quietStartMinutes: Int = 23 * 60
    var quietEndMinutes: Int = 8 * 60
}

// MARK: - State

/// What the engine remembers between signals.
nonisolated struct DayState: Sendable, Equatable, Codable {
    var day: DayKey
    /// The nudges last handed to the OS, by id.
    var syncedNudgeIDs: Set<String>?

    init(day: DayKey, syncedNudgeIDs: Set<String>? = nil) {
        self.day = day
        self.syncedNudgeIDs = syncedNudgeIDs
    }
}

// MARK: - Engine

nonisolated enum DayEngine {

    struct Decision: Sendable, Equatable {
        var state: DayState
        var effects: [DayEffect]
    }

    static func decide(_ signal: DaySignal, snapshot: DaySnapshot, state: DayState) -> Decision {
        var state = state
        var effects: [DayEffect] = []
        let today = DayKey(for: snapshot.now, calendar: snapshot.calendar)
        if state.day != today {
            state = DayState(day: today, syncedNudgeIDs: state.syncedNudgeIDs)
        }

        switch signal {
        case .tick, .agendaChanged:
            effects += syncNudgesIfChanged(snapshot: snapshot, state: &state)
        case .companionDisabled:
            if state.syncedNudgeIDs != [] {
                effects.append(.syncNudges([]))
                state.syncedNudgeIDs = []
            }
        }
        return Decision(state: state, effects: effects)
    }

    // MARK: Nudges

    private static func syncNudgesIfChanged(snapshot: DaySnapshot, state: inout DayState)
        -> [DayEffect]
    {
        guard snapshot.agenda.access.canUseCalendar else { return [] }
        let desired = NudgePlanner.plan(
            events: snapshot.agenda.events, now: snapshot.now,
            leadMinutes: snapshot.settings.nudgeLeadMinutes, calendar: snapshot.calendar)
        let ids = Set(desired.map(\.id))
        guard ids != state.syncedNudgeIDs else { return [] }
        state.syncedNudgeIDs = ids
        return [.syncNudges(desired)]
    }
}
