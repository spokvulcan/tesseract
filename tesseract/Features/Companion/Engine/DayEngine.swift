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
//  coding agents and the governor; `DayEngine+Steps` and `DayEngine+Breaks`
//  put Step Cues and Break Cues on the panel.
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
        // Counted before the rollover, so cards a new day clears clear the glyph.
        let waitingBefore = waitingCount(state, now: snapshot.now)
        if state.day != today {
            // Done on the phone while the Mac slept through 04:00: it counts
            // for the day it was the must-do of — if done within that day.
            noteMustDoDone(
                snapshot: snapshot, state: &state,
                before: state.day.end(calendar: snapshot.calendar))
            state = state.rolledOver(to: today)
        }

        switch signal {
        case .tick:
            effects += giveUpOnQuietMoment(snapshot: snapshot, state: &state)
            effects += syncNudgesIfChanged(snapshot: snapshot, state: &state)
            state.ledger.prune(now: snapshot.now)
            if snapshot.ownerPresent {
                state.lastPresentAt = snapshot.now
                state.lastActiveAt = snapshot.now
                if state.sittingSince == nil { state.sittingSince = snapshot.now }
                effects += waitingPlanIfFree(snapshot: snapshot, state: &state)
                effects += eveningIfDue(snapshot: snapshot, state: &state)
                effects += meetingEnded(snapshot: snapshot, state: &state)
                effects += triageIfDue(snapshot: snapshot, state: &state)
                effects += speakForWaitingAgents(snapshot: snapshot, state: &state)
                // A card that just took the panel keeps it; a cue waits a tick.
                if !effects.contains(where: \.takesPanel) {
                    effects += cueIfDue(snapshot: snapshot, state: &state)
                }
                effects += windDownIfDue(snapshot: snapshot, state: &state)
            } else {
                effects += prepareMorningPlanIfDue(snapshot: snapshot, state: &state)
            }
            effects += nightReflectionIfDue(snapshot: snapshot, state: &state)
            effects += runDeferred(snapshot: snapshot, state: &state)
            state.lastTickAt = snapshot.now

        case .agendaChanged:
            effects += syncNudgesIfChanged(snapshot: snapshot, state: &state)
            effects += clearStepsOfMeetings(snapshot: snapshot, state: &state)

        case .companionEnabled:
            effects += syncNudgesIfChanged(snapshot: snapshot, state: &state)
            // The calendar may have changed while the app was closed.
            effects += clearStepsOfMeetings(snapshot: snapshot, state: &state)
            effects += resumeInterrupted(snapshot: snapshot, state: &state)
            if snapshot.ownerPresent {
                // Starting up counts as sitting down: measure the gap from
                // the last time the owner was seen.
                let awayFrom = state.lastPresentAt ?? .distantPast
                effects += backFromAway(
                    awayFrom: awayFrom, snapshot: snapshot, state: &state, measured: false)
                effects += ownerReturned(awayFrom: awayFrom, snapshot: snapshot, state: &state)
                state.lastPresentAt = snapshot.now
            }
            state.lastTickAt = snapshot.now

        case .companionDisabled:
            if state.syncedNudgeIDs?.isEmpty != true {
                effects.append(.syncNudges([]))
                state.syncedNudgeIDs = []
            }
            // The panel closes with the Companion: its cue comes back by its
            // rules once the Companion is on again.
            if let key = state.cueOnPanel {
                state.cuedSteps[key] = nil
                state.cueOnPanel = nil
            }
            state.breakCuedAt = nil

        case .presenceReturned(let awayFrom):
            effects += backFromAway(awayFrom: awayFrom, snapshot: snapshot, state: &state)
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
                effects += morningPlan(trigger: .todayOpened, snapshot: snapshot, state: &state)
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

        case .nudgesDelivered(let delivered):
            effects += nudgesDelivered(delivered, state: &state)
        }

        // After the signal: a must-do set by it may be done already.
        noteMustDoDone(snapshot: snapshot, state: &state)
        holdCueUnderCard(effects, snapshot: snapshot, state: &state)
        let waitingAfter = waitingCount(state, now: snapshot.now)
        if waitingAfter != waitingBefore { effects.append(.setWaiting(waitingAfter)) }
        return Decision(state: state, effects: effects)
    }

    /// The must-do seen done, for the week's look-back; with `before`, only
    /// if it was done by then. Seen open again (an undo, or unticked in
    /// Reminders), it isn't done after all.
    private static func noteMustDoDone(
        snapshot: DaySnapshot, state: inout DayState, before: Date? = nil
    ) {
        if before == nil, state.mustDoDoneAt != nil,
            snapshot.agenda.open.contains(where: { $0.id == state.mustDoID })
        {
            state.mustDoDoneAt = nil
            return
        }
        guard let mustDo = state.mustDoID, state.mustDoDoneAt == nil,
            (snapshot.agenda.doneToday + snapshot.agenda.doneThisWeek).contains(where: {
                $0.id == mustDo && ($0.completedAt ?? .distantPast) < (before ?? .distantFuture)
            })
        else { return }
        state.mustDoDoneAt = snapshot.now
    }

    // MARK: - Triggers

    static func ownerReturned(
        awayFrom: Date, snapshot: DaySnapshot, state: inout DayState
    ) -> [DayEffect] {
        // The night, measured once, at the day's first sit-down after an
        // absence that began in the night (before 06:00 of the day's date):
        // how long the Mac was left since it was last in use.
        var night: [DayEffect] = []
        let left = state.lastActiveAt ?? awayFrom
        let away = snapshot.now.timeIntervalSince(left)
        if !state.nightMeasured, away >= DaySettings.overnightGap,
            let midnight = state.day.date(calendar: snapshot.calendar),
            left < midnight.addingTimeInterval(6 * 3600)
        {
            state.nightMeasured = true
            let upLate = snapshot.facts(state: state).upLateUntil != nil
            night = [
                .trace(
                    .nightEnded, ["minutesAway": .int(Int(away / 60)), "upLate": .bool(upLate)])
            ]
        }
        return night + sittingDown(awayFrom: awayFrom, snapshot: snapshot, state: &state)
    }

    private static func sittingDown(
        awayFrom: Date, snapshot: DaySnapshot, state: inout DayState
    ) -> [DayEffect] {
        let away = snapshot.now.timeIntervalSince(awayFrom)
        let hour = snapshot.minuteOfDay / 60
        let afterTheNight =
            away >= DaySettings.overnightGap && hour >= snapshot.settings.morningStartHour
        if afterTheNight, hour < snapshot.settings.morningEndHour {
            state.satDownAt = snapshot.now
            if state.morningPlanAt == nil {
                return planTheDayOrWait(snapshot: snapshot, state: &state)
            }
            // Planned while they were away: it comes forward now.
            let prepared = presentPreparedMorningPlan(
                awayFrom: awayFrom, snapshot: snapshot, state: state)
            if !prepared.isEmpty { return prepared }
        } else if afterTheNight, snapshot.minuteOfDay < snapshot.settings.eveningMinutes,
            noPlanSeen(awayFrom: awayFrom, state: state)
        {
            // A day that starts late — a weekend, a long night (4 October:
            // first at the Mac at 14:43, a plan only on asking) — still gets
            // its plan at the first sit-down, made now: one prepared that
            // morning is hours out of date, its slots gone by.
            state.satDownAt = snapshot.now
            return planTheDayOrWait(snapshot: snapshot, state: &state)
        }
        let evening = eveningIfDue(snapshot: snapshot, state: &state)
        if !evening.isEmpty { return evening }
        guard away >= TimeInterval(snapshot.settings.breakpointAwayMinutes * 60) else { return [] }
        return breakpoint(
            awayFrom: awayFrom, trigger: .presenceReturned, snapshot: snapshot, state: &state)
    }

    /// The first sit-down's plan, made now (one prepared while the owner was
    /// away is made again, its stale slots cleared) — or, with the model
    /// busy (a moment running, a chat), at the next tick it is free.
    private static func planTheDayOrWait(snapshot: DaySnapshot, state: inout DayState)
        -> [DayEffect]
    {
        guard state.running == nil, !snapshot.chatBusy else {
            state.morningPlanWaiting = true
            return []
        }
        state.morningPlanWaiting = false
        if state.morningPlanAt != nil { state.plan = [] }
        return morningPlan(trigger: .firstPresence, snapshot: snapshot, state: &state)
    }

    /// A first sit-down's plan that waited for the model, once it is free.
    static func waitingPlanIfFree(snapshot: DaySnapshot, state: inout DayState)
        -> [DayEffect]
    {
        guard state.morningPlanWaiting, !isEvening(snapshot) else { return [] }
        return planTheDayOrWait(snapshot: snapshot, state: &state)
    }

    /// The owner hasn't seen a plan today: none was made, or only one made
    /// while they were away and never closed or taken in.
    private static func noPlanSeen(awayFrom: Date, state: DayState) -> Bool {
        guard state.morningPlanAt != nil else { return true }
        guard let card = state.cards.last(where: { $0.kind == .morningPlan }) else { return false }
        return !card.dismissed && !card.kept && card.createdAt >= awayFrom
    }

    /// The Evening Wrap-up is due from the evening time until 03:00, once —
    /// after the step the owner started, if one is running: like a Step
    /// Cue, it doesn't interrupt a focus session.
    static func eveningIfDue(snapshot: DaySnapshot, state: inout DayState) -> [DayEffect] {
        guard state.eveningWrapUpAt == nil, isEvening(snapshot),
            focus(snapshot: snapshot, state: state) == nil
        else { return [] }
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

    /// As quiet hours begin with the owner still at the Mac, one banner, once
    /// a night: when tomorrow starts, and how far off that is. A friend's word
    /// to rest, not a rule; then quiet hours hold everything of Jarvis's.
    static func windDownIfDue(snapshot: DaySnapshot, state: inout DayState) -> [DayEffect] {
        let settings = snapshot.settings
        // Quiet hours that start at night (a daytime window is no bedtime),
        // still on, once a night (across the 04:00 rollover too).
        guard settings.windDown, DeliveryLadder.quietHoursAreNight(settings),
            DeliveryLadder.isQuietHours(snapshot),
            state.windDownAt.map({ snapshot.now.timeIntervalSince($0) >= 12 * 3600 }) ?? true,
            !snapshot.frontmostIsGame,
            !DeliveryLadder.interruptionFreeApps.contains(snapshot.frontmostBundleID ?? "")
        else { return [] }
        // Within the first hour of quiet hours, midnight or not.
        let sinceStart = (snapshot.minuteOfDay - settings.quietStartMinutes + 24 * 60) % (24 * 60)
        guard sinceStart < 60 else { return [] }
        state.windDownAt = snapshot.now
        let tomorrow = TimelineBuilder.tomorrow(facts: snapshot.facts(state: state))
        var fields: [String: CompanionTraceValue] = [:]
        if let first = tomorrow.rows.first {
            fields["minutesUntil"] = .int(Int(first.start.timeIntervalSince(snapshot.now) / 60))
        }
        return [
            .postBanner(
                title: "Time to wind down",
                body: windDownLine(tomorrow, now: snapshot.now, calendar: snapshot.calendar)),
            .trace(.windDown, fields),
        ]
    }

    /// When tomorrow starts: its first event or timed task and how far off it
    /// is, or what it holds without a time.
    static func windDownLine(_ tomorrow: TomorrowTimeline, now: Date, calendar: Calendar)
        -> String
    {
        if let first = tomorrow.rows.first {
            let title: String =
                switch first.kind {
                case .event(let event): event.title
                case .task(let task): task.reminder.title
                case .free, .now: ""
                }
            let minutes = max(0, Int(first.start.timeIntervalSince(now) / 60))
            let clock = AgendaTime.clock(first.start, calendar: calendar)
            return
                "Tomorrow starts with \(title) at \(clock) — \(MomentPrompts.minutesText(minutes)) from now."
        }
        if !tomorrow.allDayEvents.isEmpty {
            let names = tomorrow.allDayEvents.map(\.title).joined(separator: ", ")
            return "Tomorrow: \(names), nothing at a set time. Rest well."
        }
        return "Nothing is set for tomorrow yet. Rest well."
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
        // A card the owner took in waits on them no more.
        let cards = state.openCards.filter { !$0.kept }.reduce(0) { count, card in
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

    /// Record each nudge macOS delivered, once. Notification Center keeps a
    /// nudge until the owner clears it, so the record keeps only ids still
    /// there.
    static func nudgesDelivered(_ delivered: [DeliveredNudge], state: inout DayState)
        -> [DayEffect]
    {
        let fresh = delivered.filter { !state.firedNudgeIDs.contains($0.id) }
        state.firedNudgeIDs = Set(delivered.map(\.id))
        return fresh.sorted { $0.at < $1.at }.map { nudge in
            .trace(
                .nudgeFired,
                [
                    "id": .string(nudge.id), "title": .string(nudge.title),
                    "deliveredAt": .double(nudge.at.timeIntervalSince1970),
                ])
        }
    }

    private static func syncNudgesIfChanged(snapshot: DaySnapshot, state: inout DayState)
        -> [DayEffect]
    {
        guard snapshot.agenda.access.canUseCalendar else { return [] }
        let desired =
            NudgePlanner.plan(
                events: snapshot.agenda.events, now: snapshot.now,
                leadMinutes: snapshot.settings.nudgeLeadMinutes, calendar: snapshot.calendar)
            + NudgePlanner.plan(
                departures: state.departures, events: snapshot.agenda.events,
                now: snapshot.now, calendar: snapshot.calendar)
        let ids = Set(desired.map(\.id))
        guard ids != state.syncedNudgeIDs else { return [] }
        state.syncedNudgeIDs = ids
        return [.syncNudges(desired)]
    }
}

nonisolated extension DayEffect {
    /// It puts something on the Jarvis Panel, or starts a moment that may.
    var takesPanel: Bool {
        switch self {
        case .presentCard(_, .panel), .presentStep, .presentBreak, .runMoment: true
        default: false
        }
    }
}
