//
//  DayEngine.swift
//  tesseract
//
//  The Day Engine: the Companion's pure decider, in the gather → decide →
//  perform shape. The loop gathers a snapshot of the world (agenda, presence,
//  time) and feeds the engine one signal at a time; the engine returns the
//  day's next state and an ordered list of effects, and the loop performs
//  them. The engine performs no I/O, reads no clock and holds no references,
//  so every rule is a row in a decision table.
//
//  "Jarvis thinks at moments; code keeps the promises": the engine decides
//  when a moment is due and what the owner should see, and every promise it
//  makes (a nudge, a reminder, a card) is kept by code and the OS. No moment
//  runs without new input: each runs once per trigger, and its outcome — a
//  card, or a fallback card — stands until the inputs change.
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
        let waitingBefore = waitingCount(state)

        switch signal {
        case .tick:
            effects += syncNudgesIfChanged(snapshot: snapshot, state: &state)
            if snapshot.ownerPresent {
                state.lastPresentAt = snapshot.now
                effects += eveningIfDue(snapshot: snapshot, state: &state)
            }

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

        case .todayOpened:
            if state.morningPlanAt == nil, state.running == nil,
                snapshot.minuteOfDay < snapshot.settings.eveningMinutes,
                snapshot.minuteOfDay >= snapshot.settings.morningStartHour * 60
            {
                effects += run(
                    .morningPlan, trigger: .todayOpened, snapshot: snapshot, state: &state)
            }

        case .momentOutcome(let request, let outcome):
            effects += momentFinished(request, outcome, snapshot: snapshot, state: &state)

        case .cardAction(let action):
            effects += cardAction(action, snapshot: snapshot, state: &state)
        }

        let waitingAfter = waitingCount(state)
        if waitingAfter != waitingBefore { effects.append(.setWaiting(waitingAfter)) }
        return Decision(state: state, effects: effects)
    }

    // MARK: - Triggers

    private static func ownerReturned(
        awayFrom: Date, snapshot: DaySnapshot, state: inout DayState
    ) -> [DayEffect] {
        let away = snapshot.now.timeIntervalSince(awayFrom)
        let hour = snapshot.minuteOfDay / 60
        if state.morningPlanAt == nil, away >= DaySettings.overnightGap,
            hour >= snapshot.settings.morningStartHour, hour < snapshot.settings.morningEndHour
        {
            return run(.morningPlan, trigger: .firstPresence, snapshot: snapshot, state: &state)
        }
        return eveningIfDue(snapshot: snapshot, state: &state)
    }

    /// The Evening Wrap-up is due from the evening time until 03:00, once.
    private static func eveningIfDue(snapshot: DaySnapshot, state: inout DayState) -> [DayEffect] {
        guard state.eveningWrapUpAt == nil, isEvening(snapshot) else { return [] }
        return run(.eveningWrapUp, trigger: .eveningTime, snapshot: snapshot, state: &state)
    }

    static func isEvening(_ snapshot: DaySnapshot) -> Bool {
        let minute = snapshot.minuteOfDay
        return minute >= snapshot.settings.eveningMinutes || minute < 3 * 60
    }

    // MARK: - Running moments

    private static func run(
        _ kind: MomentKind, trigger: MomentTrigger, snapshot: DaySnapshot, state: inout DayState,
        attempt: Int = 0, text: String? = nil
    ) -> [DayEffect] {
        // One moment at a time, and never under the owner's own Today turn.
        guard state.running == nil || attempt > 0, !snapshot.chatBusy else { return [] }
        let facts = snapshot.facts(state: state)
        let body: String
        switch kind {
        case .morningPlan: body = text ?? MomentPrompts.morningPlan(facts: facts)
        case .eveningWrapUp:
            body = text ?? MomentPrompts.eveningWrapUp(facts: facts, leftovers: leftovers(facts))
        case .breakpoint, .triage, .nightReflection:
            return []  // Later slices.
        }
        state.running = kind
        let request = MomentRequest(kind: kind, trigger: trigger, text: body, attempt: attempt)
        return [
            .trace(
                .momentStarted,
                [
                    "moment": .string(kind.rawValue), "trigger": .string(trigger.rawValue),
                    "attempt": .int(attempt),
                ]),
            .runMoment(request),
        ]
    }

    /// Open tasks that belonged to today: due today, or planned today.
    static func leftovers(_ facts: DayFacts) -> [AgendaReminder] {
        let planned = Set(facts.plan.map(\.reminderID))
        return facts.openTasks.filter { reminder in
            if planned.contains(reminder.id) { return true }
            guard let due = reminder.due else { return false }
            return due >= facts.startOfToday && due < facts.endOfToday
        }
    }

    private static func momentFinished(
        _ request: MomentRequest, _ outcome: MomentOutcome, snapshot: DaySnapshot,
        state: inout DayState
    ) -> [DayEffect] {
        state.running = nil
        let facts = snapshot.facts(state: state)
        var failure: String
        var measure: MomentMeasure?
        switch outcome {
        case .reply(let text, let measured):
            measure = measured
            if measured.hitCap {
                failure = "hit the output cap"
            } else {
                switch parse(request.kind, text, facts: facts) {
                case .card(let body):
                    var effects = accept(
                        body, request: request, fallback: false, snapshot: snapshot, state: &state)
                    effects.insert(
                        .trace(.momentFinished, traceFields(request, measure: measured)), at: 0)
                    return effects
                case .invalid(let reason):
                    failure = reason
                }
            }
        case .failed(let reason, let measured):
            measure = measured
            failure = reason
        }
        var fields = traceFields(request, measure: measure)
        fields["reason"] = .string(failure)
        if request.attempt == 0 {
            // One retry. After an unreadable reply the thread holds it, so the
            // retry asks again for just the card.
            let retryText: String? =
                if case .reply = outcome {
                    "That wasn't the card. Reply again with only the JSON object described above."
                } else { request.text }
            var effects: [DayEffect] = [
                .trace(.momentFailed, fields.merging(["retry": true]) { $1 })
            ]
            effects += run(
                request.kind, trigger: .retry, snapshot: snapshot, state: &state, attempt: 1,
                text: retryText)
            if !effects.contains(where: { if case .runMoment = $0 { true } else { false } }) {
                effects += accept(
                    fallbackBody(request.kind, facts: facts), request: request, fallback: true,
                    snapshot: snapshot, state: &state)
            }
            return effects
        }
        var effects: [DayEffect] = [
            .trace(.momentFailed, fields.merging(["fallback": true]) { $1 })
        ]
        effects += accept(
            fallbackBody(request.kind, facts: facts), request: request, fallback: true,
            snapshot: snapshot, state: &state)
        return effects
    }

    static func parse(_ kind: MomentKind, _ reply: String, facts: DayFacts) -> CardParse {
        switch kind {
        case .morningPlan: CardParser.morningPlan(reply, facts: facts)
        case .eveningWrapUp:
            CardParser.eveningWrapUp(reply, facts: facts, leftovers: leftovers(facts))
        case .breakpoint, .triage, .nightReflection: .invalid("not handled yet")
        }
    }

    static func fallbackBody(_ kind: MomentKind, facts: DayFacts) -> DayCard.Body {
        switch kind {
        case .morningPlan: .morningPlan(FallbackCards.morningPlan(facts: facts))
        case .eveningWrapUp:
            .eveningWrapUp(FallbackCards.eveningWrapUp(facts: facts, leftovers: leftovers(facts)))
        case .breakpoint, .triage, .nightReflection:
            .morningPlan(FallbackCards.morningPlan(facts: facts))
        }
    }

    private static func accept(
        _ body: DayCard.Body, request: MomentRequest, fallback: Bool, snapshot: DaySnapshot,
        state: inout DayState
    ) -> [DayEffect] {
        switch body {
        case .morningPlan(let card):
            state.morningPlanAt = snapshot.now
            if let mustDo = card.mustDoID { state.mustDoID = mustDo }
            if !card.placements.isEmpty { state.plan = card.placements }
        case .eveningWrapUp:
            state.eveningWrapUpAt = snapshot.now
        case .breakpoint, .triage, .reflection:
            break
        }
        // A newer card of the same kind replaces the older one.
        for index in state.cards.indices where state.cards[index].kind == request.kind {
            state.cards[index].dismissed = true
        }
        let card = DayCard(
            id: "\(request.kind.rawValue)-\(state.day.rawValue)-\(state.cards.count)",
            kind: request.kind, createdAt: snapshot.now, isFallback: fallback, body: body)
        state.cards.append(card)
        return [
            .presentCard(card, .today),
            .trace(
                .cardPresented,
                [
                    "card": .string(card.id), "moment": .string(card.kind.rawValue),
                    "rung": .string(DeliveryRung.today.rawValue), "fallback": .bool(fallback),
                ]),
        ]
    }

    private static func traceFields(_ request: MomentRequest, measure: MomentMeasure?)
        -> [String: CompanionTraceValue]
    {
        var fields: [String: CompanionTraceValue] = [
            "moment": .string(request.kind.rawValue), "trigger": .string(request.trigger.rawValue),
            "attempt": .int(request.attempt),
        ]
        if let measure {
            fields["promptTokens"] = .int(measure.promptTokens)
            fields["outputTokens"] = .int(measure.outputTokens)
            fields["prefillSeconds"] = .double(measure.prefillSeconds)
            fields["generateSeconds"] = .double(measure.generateSeconds)
            fields["latencySeconds"] = .double(measure.latencySeconds)
            fields["hitCap"] = .bool(measure.hitCap)
            fields["model"] = .string(measure.modelID)
        }
        return fields
    }

    // MARK: - Card actions

    private static func cardAction(
        _ action: CardAction, snapshot: DaySnapshot, state: inout DayState
    ) -> [DayEffect] {
        switch action {
        case .dismiss(let cardID):
            guard let index = state.cards.firstIndex(where: { $0.id == cardID }) else { return [] }
            state.cards[index].dismissed = true
            return [reaction("dismissed", card: state.cards[index], snapshot: snapshot)]

        case .setMustDo(let reminderID):
            state.mustDoID = reminderID
            return [.trace(.cardReaction, ["action": "mustDo", "set": .bool(reminderID != nil)])]

        case .removeFromPlan(let reminderID):
            state.plan.removeAll { $0.reminderID == reminderID }
            return [.trace(.cardReaction, ["action": "unplanned"])]

        case .place(let reminderID, let start, let minutes):
            state.plan.removeAll { $0.reminderID == reminderID }
            state.plan.append(Placement(reminderID: reminderID, start: start, minutes: minutes))
            state.plan.sort { $0.start < $1.start }
            return [.trace(.cardReaction, ["action": "placed", "minutes": .int(minutes)])]

        case .leftover(let cardID, let reminderID, let suggestion):
            guard let index = state.cards.firstIndex(where: { $0.id == cardID }),
                case .eveningWrapUp(var card) = state.cards[index].body,
                card.leftovers.contains(where: { $0.reminderID == reminderID })
            else { return [] }
            card.leftovers.removeAll { $0.reminderID == reminderID }
            state.cards[index].body = .eveningWrapUp(card)
            state.plan.removeAll { $0.reminderID == reminderID }
            return [
                .mutateAgenda(mutation(for: suggestion, reminderID: reminderID)),
                reaction(
                    "leftover.\(suggestion.rawValue)", card: state.cards[index], snapshot: snapshot),
            ]

        case .allLeftovers(let cardID):
            guard let index = state.cards.firstIndex(where: { $0.id == cardID }),
                case .eveningWrapUp(var card) = state.cards[index].body
            else { return [] }
            let mutations = card.leftovers.map {
                DayEffect.mutateAgenda(mutation(for: $0.suggestion, reminderID: $0.reminderID))
            }
            let moved = Set(card.leftovers.map(\.reminderID))
            card.leftovers = []
            state.cards[index].body = .eveningWrapUp(card)
            state.plan.removeAll { moved.contains($0.reminderID) }
            return mutations + [
                reaction("leftovers.all", card: state.cards[index], snapshot: snapshot)
            ]

        case .planNow:
            return run(.morningPlan, trigger: .ownerAsked, snapshot: snapshot, state: &state)

        case .wrapUpNow:
            return run(.eveningWrapUp, trigger: .ownerAsked, snapshot: snapshot, state: &state)
        }
    }

    private static func mutation(for suggestion: Leftover.Suggestion, reminderID: String)
        -> AgendaMutation
    {
        switch suggestion {
        case .tomorrow: .dueTomorrow(reminderID: reminderID)
        case .later: .clearDue(reminderID: reminderID)
        case .drop: .delete(reminderID: reminderID)
        }
    }

    private static func reaction(_ action: String, card: DayCard, snapshot: DaySnapshot)
        -> DayEffect
    {
        .trace(
            .cardReaction,
            [
                "card": .string(card.id), "moment": .string(card.kind.rawValue),
                "action": .string(action),
                "secondsToReact": .double(snapshot.now.timeIntervalSince(card.createdAt)),
            ])
    }

    // MARK: - Waiting

    /// What is waiting on the owner, for the glyph: undecided leftovers.
    static func waitingCount(_ state: DayState) -> Int {
        state.openCards.reduce(0) { count, card in
            switch card.body {
            case .eveningWrapUp(let wrapUp): count + wrapUp.leftovers.count
            case .breakpoint(let breakpoint): count + breakpoint.needsYou.count
            case .triage(let triage): count + triage.raise.count
            case .morningPlan, .reflection: count
            }
        }
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
