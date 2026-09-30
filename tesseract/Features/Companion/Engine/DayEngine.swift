//
//  DayEngine.swift
//  tesseract
//
//  The Day Engine: the Companion's pure decider, in the gather → decide →
//  perform shape. The loop gathers a snapshot of the world (agenda, presence,
//  the app in front, power) and feeds the engine one signal at a time; the
//  engine returns the day's next state and an ordered list of effects, and
//  the loop performs them. The engine performs no I/O, reads no clock and
//  holds no references, so every rule is a row in a decision table.
//
//  "Jarvis thinks at moments; code keeps the promises": the engine decides
//  when a moment is due and what the owner should see, and every promise it
//  makes (a nudge, a reminder, a card) is kept by code and the OS. No moment
//  runs without new input: each runs once per trigger, and its outcome — a
//  card, a fallback card, or nothing needing the owner — stands until the
//  inputs change.
//
//  Split by concern: this file dispatches signals and holds the triggers;
//  `DayEngine+Moments` runs moments and turns replies into cards;
//  `DayEngine+Breakpoints` holds Breakpoints, Triage, notifications,
//  coding agents and the governor.
//

import Foundation

nonisolated enum DayEngine {

    struct Decision: Sendable, Equatable {
        var state: DayState
        var effects: [DayEffect]
    }

    static func decide(_ signal: DaySignal, snapshot: DaySnapshot, state: DayState) -> Decision {
        var state = state
        var effects: [DayEffect] = []
        let today = DayKey(for: snapshot.now, calendar: snapshot.calendar)
        if state.day != today { state = state.rolledOver(to: today) }
        let waitingBefore = waitingCount(state, now: snapshot.now)

        switch signal {
        case .tick:
            effects += syncNudgesIfChanged(snapshot: snapshot, state: &state)
            state.ledger.prune(now: snapshot.now)
            if snapshot.ownerPresent {
                state.lastPresentAt = snapshot.now
                effects += eveningIfDue(snapshot: snapshot, state: &state)
                effects += meetingEnded(snapshot: snapshot, state: &state)
                effects += triageIfDue(snapshot: snapshot, state: &state)
                effects += speakForWaitingAgents(snapshot: snapshot, state: &state)
            }
            effects += nightReflectionIfDue(snapshot: snapshot, state: &state)
            effects += runDeferred(snapshot: snapshot, state: &state)
            state.lastTickAt = snapshot.now

        case .agendaChanged:
            effects += syncNudgesIfChanged(snapshot: snapshot, state: &state)

        case .companionEnabled:
            effects += syncNudgesIfChanged(snapshot: snapshot, state: &state)
            if snapshot.ownerPresent {
                // Starting up counts as sitting down: measure the gap from
                // the last time the owner was seen.
                effects += ownerReturned(
                    awayFrom: state.lastPresentAt ?? .distantPast, snapshot: snapshot,
                    state: &state)
                state.lastPresentAt = snapshot.now
            }
            state.lastTickAt = snapshot.now

        case .companionDisabled:
            if state.syncedNudgeIDs?.isEmpty != true {
                effects.append(.syncNudges([]))
                state.syncedNudgeIDs = []
            }

        case .presenceReturned(let awayFrom):
            effects += ownerReturned(awayFrom: awayFrom, snapshot: snapshot, state: &state)
            state.lastPresentAt = snapshot.now

        case .presenceLeft:
            state.lastPresentAt = snapshot.now
            state.whereYouWere = snapshot.frontmostAppName

        case .todayOpened:
            if state.morningPlanAt == nil, state.running == nil,
                snapshot.minuteOfDay < snapshot.settings.eveningMinutes,
                snapshot.minuteOfDay >= snapshot.settings.morningStartHour * 60
            {
                effects += run(
                    .morningPlan, trigger: .todayOpened, snapshot: snapshot, state: &state)
            }

        case .notificationArrived(let notification):
            effects += notificationArrived(notification, snapshot: snapshot, state: &state)

        case .appActivated(let name, let bundleID):
            effects += appActivated(
                name: name, bundleID: bundleID, snapshot: snapshot, state: &state)

        case .agentSignal(let agent):
            effects += agentSignal(agent, snapshot: snapshot, state: &state)

        case .powerChanged:
            effects += runDeferred(snapshot: snapshot, state: &state)

        case .momentOutcome(let request, let outcome):
            effects += momentFinished(request, outcome, snapshot: snapshot, state: &state)

        case .cardAction(let action):
            effects += cardAction(action, snapshot: snapshot, state: &state)
        }

        let waitingAfter = waitingCount(state, now: snapshot.now)
        if waitingAfter != waitingBefore { effects.append(.setWaiting(waitingAfter)) }
        return Decision(state: state, effects: effects)
    }

    // MARK: - Triggers

    static func ownerReturned(
        awayFrom: Date, snapshot: DaySnapshot, state: inout DayState
    ) -> [DayEffect] {
        let away = snapshot.now.timeIntervalSince(awayFrom)
        let hour = snapshot.minuteOfDay / 60
        if state.morningPlanAt == nil, away >= DaySettings.overnightGap,
            hour >= snapshot.settings.morningStartHour, hour < snapshot.settings.morningEndHour
        {
            return run(.morningPlan, trigger: .firstPresence, snapshot: snapshot, state: &state)
        }
        let evening = eveningIfDue(snapshot: snapshot, state: &state)
        if !evening.isEmpty { return evening }
        guard away >= TimeInterval(snapshot.settings.breakpointAwayMinutes * 60) else { return [] }
        return breakpoint(
            awayFrom: awayFrom, trigger: .presenceReturned, snapshot: snapshot, state: &state)
    }

    /// The Evening Wrap-up is due from the evening time until 03:00, once.
    static func eveningIfDue(snapshot: DaySnapshot, state: inout DayState) -> [DayEffect] {
        guard state.eveningWrapUpAt == nil, isEvening(snapshot) else { return [] }
        return run(.eveningWrapUp, trigger: .eveningTime, snapshot: snapshot, state: &state)
    }

    /// The Night Reflection: once per night, at least half an hour after the
    /// Evening Wrap-up (leftovers decided), before the 04:00 rollover. The
    /// governor holds it for power and a cool Mac.
    static func nightReflectionIfDue(snapshot: DaySnapshot, state: inout DayState) -> [DayEffect] {
        guard state.nightReflectionAt == nil, state.running == nil,
            !state.deferred.contains(.nightReflection),
            let wrapUp = state.eveningWrapUpAt,
            snapshot.now.timeIntervalSince(wrapUp) >= 30 * 60, isEvening(snapshot)
        else { return [] }
        return run(
            .nightReflection, trigger: .night, snapshot: snapshot, state: &state,
            text: MomentPrompts.nightReflection(
                facts: snapshot.facts(state: state), profile: snapshot.profile))
    }

    static func isEvening(_ snapshot: DaySnapshot) -> Bool {
        let minute = snapshot.minuteOfDay
        return minute >= snapshot.settings.eveningMinutes || minute < 3 * 60
    }

    /// A meeting the owner attended just ended (since the last tick).
    private static func meetingEnded(snapshot: DaySnapshot, state: inout DayState) -> [DayEffect] {
        guard let last = state.lastTickAt else { return [] }
        let ended = snapshot.agenda.events.filter {
            $0.hasOtherAttendees && !$0.isAllDay && $0.end > last && $0.end <= snapshot.now
        }
        guard let meeting = ended.max(by: { $0.end < $1.end }) else { return [] }
        return breakpoint(
            awayFrom: meeting.start, trigger: .meetingEnded, snapshot: snapshot, state: &state)
    }

    // MARK: - Waiting

    /// What is waiting on the owner, for the glyph: waiting agents, and the
    /// open cards' items.
    static func waitingCount(_ state: DayState, now: Date) -> Int {
        let cards = state.openCards.reduce(0) { count, card in
            switch card.body {
            case .eveningWrapUp(let wrapUp): count + wrapUp.leftovers.count
            case .breakpoint(let breakpoint):
                count + breakpoint.needsYou.filter { $0.kind != .agent }.count
            case .triage(let triage): count + triage.raise.count
            case .morningPlan, .reflection: count
            }
        }
        return cards + state.agentsWaiting(now: now).count
    }

    // MARK: - Nudges

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
