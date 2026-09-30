//
//  DayEngine+Moments.swift
//  tesseract
//
//  Running a moment and turning its reply into a card: one request, one
//  retry, then the deterministic card. And what the owner's card actions do.
//

import Foundation

nonisolated extension DayEngine {

    // MARK: - Running

    /// Ask the loop to run a moment. Morning Plan and Evening Wrap-up build
    /// their own request; Breakpoint and Triage pass theirs in.
    static func run(
        _ kind: MomentKind, trigger: MomentTrigger, snapshot: DaySnapshot, state: inout DayState,
        attempt: Int = 0, text: String? = nil, context: MomentContext = .init()
    ) -> [DayEffect] {
        // One moment at a time, and never under the owner's own Today turn.
        guard state.running == nil || attempt > 0, !snapshot.chatBusy else { return [] }
        if kind.isDeferrable, let why = Governor.deferral(for: kind, power: snapshot.power) {
            guard !state.deferred.contains(kind) else { return [] }
            state.deferred.insert(kind)
            return [
                .trace(
                    .governorDeferred, ["moment": .string(kind.rawValue), "reason": .string(why)])
            ]
        }
        state.deferred.remove(kind)
        let facts = snapshot.facts(state: state)
        let body: String
        switch kind {
        case .morningPlan:
            body = text ?? MomentPrompts.morningPlan(facts: facts)
        case .eveningWrapUp:
            body = text ?? MomentPrompts.eveningWrapUp(facts: facts, leftovers: leftovers(facts))
        case .breakpoint, .triage, .nightReflection:
            guard let text else { return [] }
            body = text
        }
        state.running = kind
        let request = MomentRequest(
            kind: kind, trigger: trigger, text: body, attempt: attempt, context: context)
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

    // MARK: - Outcomes

    static func momentFinished(
        _ request: MomentRequest, _ outcome: MomentOutcome, snapshot: DaySnapshot,
        state: inout DayState
    ) -> [DayEffect] {
        state.running = nil
        var fields = traceFields(request, outcome: outcome, power: snapshot.power)
        var failure = "unknown"
        if case .reply(let text, let measure) = outcome {
            if measure.hitCap {
                failure = "hit the output cap"
            } else if let effects = accepted(
                request, reply: text, snapshot: snapshot, state: &state)
            {
                return [.trace(.momentFinished, fields)] + effects
            } else {
                failure = "no readable card"
            }
        } else if case .failed(let reason, _) = outcome {
            failure = reason
        }
        fields["reason"] = .string(failure)

        if request.attempt == 0 {
            // One retry. After an unreadable reply the thread holds it, so
            // the retry asks again for just the card.
            let retryText: String =
                if case .reply = outcome {
                    "That wasn't the card. Reply again with only the JSON object described above."
                } else { request.text }
            let retry = run(
                request.kind, trigger: .retry, snapshot: snapshot, state: &state, attempt: 1,
                text: retryText, context: request.context)
            if retry.contains(where: { if case .runMoment = $0 { true } else { false } }) {
                return [.trace(.momentFailed, fields.merging(["retry": true]) { $1 })] + retry
            }
        }
        return [.trace(.momentFailed, fields.merging(["fallback": true]) { $1 })]
            + fallback(request, snapshot: snapshot, state: &state)
    }

    /// The effects of a readable reply, or nil when there is no card in it.
    private static func accepted(
        _ request: MomentRequest, reply: String, snapshot: DaySnapshot, state: inout DayState
    ) -> [DayEffect]? {
        let facts = snapshot.facts(state: state)
        switch request.kind {
        case .morningPlan:
            guard case .card(let body) = CardParser.morningPlan(reply, facts: facts) else {
                return nil
            }
            return accept(
                body, kind: .morningPlan, fallback: false, snapshot: snapshot, state: &state)
        case .eveningWrapUp:
            guard
                case .card(let body) = CardParser.eveningWrapUp(
                    reply, facts: facts, leftovers: leftovers(facts))
            else { return nil }
            return accept(
                body, kind: .eveningWrapUp, fallback: false, snapshot: snapshot, state: &state)
        case .breakpoint:
            return breakpointReplied(request, reply: reply, snapshot: snapshot, state: &state)
        case .triage:
            return triageReplied(request, reply: reply, snapshot: snapshot, state: &state)
        case .nightReflection:
            guard case .card(let body) = CardParser.nightReflection(reply) else { return nil }
            return accept(
                body, kind: .nightReflection, fallback: false, snapshot: snapshot, state: &state)
        }
    }

    private static func fallback(
        _ request: MomentRequest, snapshot: DaySnapshot, state: inout DayState
    ) -> [DayEffect] {
        let facts = snapshot.facts(state: state)
        switch request.kind {
        case .morningPlan:
            return accept(
                .morningPlan(FallbackCards.morningPlan(facts: facts)), kind: .morningPlan,
                fallback: true, snapshot: snapshot, state: &state)
        case .eveningWrapUp:
            return accept(
                .eveningWrapUp(
                    FallbackCards.eveningWrapUp(facts: facts, leftovers: leftovers(facts))),
                kind: .eveningWrapUp, fallback: true, snapshot: snapshot, state: &state)
        case .breakpoint:
            // The code-built card is already up; it stands.
            state.ledger.markPresented(request.context.notificationIDs, at: snapshot.now)
            return []
        case .triage:
            // Nothing is raised; everything waits for the next Breakpoint.
            state.ledger.markTriaged(request.context.notificationIDs, at: snapshot.now)
            return []
        case .nightReflection:
            return accept(
                .reflection(FallbackCards.nightReflection(facts: facts)), kind: .nightReflection,
                fallback: true, snapshot: snapshot, state: &state)
        }
    }

    /// Put a new card on the day and deliver it.
    static func accept(
        _ body: DayCard.Body, kind: MomentKind, fallback: Bool, snapshot: DaySnapshot,
        state: inout DayState, importance: Importance = .normal, cardID: String? = nil
    ) -> [DayEffect] {
        switch body {
        case .morningPlan(let card):
            state.morningPlanAt = snapshot.now
            if let mustDo = card.mustDoID { state.mustDoID = mustDo }
            if !card.placements.isEmpty { state.plan = card.placements }
        case .eveningWrapUp:
            state.eveningWrapUpAt = snapshot.now
        case .reflection(let card):
            state.nightReflectionAt = snapshot.now
            state.carryOverForNextDay = card.carryOver
        case .breakpoint, .triage:
            break
        }
        // Refining a card in place (a Breakpoint's model version).
        if let cardID, let index = state.cards.firstIndex(where: { $0.id == cardID }) {
            guard !state.cards[index].dismissed else { return [] }
            state.cards[index].body = body
            state.cards[index].isFallback = fallback
            let card = state.cards[index]
            let rungs = DeliveryLadder.rungs(for: importance, snapshot: snapshot).filter {
                $0 == .panel || $0 == .today
            }
            return rungs.map { .presentCard(card, $0) }
        }
        // A newer card of the same kind replaces the older one.
        var effects: [DayEffect] = []
        for index in state.cards.indices
        where state.cards[index].kind == kind && !state.cards[index].dismissed {
            state.cards[index].dismissed = true
            effects.append(.retractCard(cardID: state.cards[index].id))
        }
        let card = DayCard(
            id: "\(kind.rawValue)-\(state.day.rawValue)-\(state.cards.count)",
            kind: kind, createdAt: snapshot.now, isFallback: fallback, body: body)
        state.cards.append(card)
        // The night's reflection waits in Today; every other card takes the
        // ladder (the panel when the owner is at the Mac).
        let rungs: [DeliveryRung] =
            kind == .nightReflection
            ? [.today] : DeliveryLadder.rungs(for: importance, snapshot: snapshot)
        for rung in rungs {
            if rung == .voice {
                effects.append(.speak(card.line))
            } else {
                effects.append(.presentCard(card, rung))
            }
        }
        effects.append(
            .trace(
                .cardPresented,
                [
                    "card": .string(card.id), "moment": .string(kind.rawValue),
                    "rungs": .string(rungs.map(\.rawValue).joined(separator: ",")),
                    "fallback": .bool(fallback), "importance": .string(importance.rawValue),
                ]))
        if case .reflection(let reflection) = body, !reflection.proposals.isEmpty {
            effects.append(.proposeFacts(reflection.proposals))
        }
        return effects
    }

    private static func traceFields(
        _ request: MomentRequest, outcome: MomentOutcome, power: PowerState
    ) -> [String: CompanionTraceValue] {
        var fields: [String: CompanionTraceValue] = [
            "moment": .string(request.kind.rawValue), "trigger": .string(request.trigger.rawValue),
            "attempt": .int(request.attempt), "thermal": .string("\(power.thermal)"),
            "onACPower": .bool(power.onACPower),
        ]
        if let battery = power.batteryPercent { fields["batteryPercent"] = .int(battery) }
        let measure: MomentMeasure? =
            switch outcome {
            case .reply(_, let measure): measure
            case .failed(_, let measure): measure
            }
        if let measure {
            fields["promptTokens"] = .int(measure.promptTokens)
            fields["cachedTokens"] = .int(measure.cachedTokens)
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

    static func cardAction(
        _ action: CardAction, snapshot: DaySnapshot, state: inout DayState
    ) -> [DayEffect] {
        switch action {
        case .dismiss(let cardID):
            guard let index = state.cards.firstIndex(where: { $0.id == cardID }) else { return [] }
            state.cards[index].dismissed = true
            return [
                .retractCard(cardID: cardID),
                reaction("dismissed", card: state.cards[index], snapshot: snapshot),
            ]

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

        case .openItem(let cardID, let itemID):
            guard let item = removeItem(itemID, fromCard: cardID, state: &state) else { return [] }
            var effects: [DayEffect] = []
            if let app = item.app { effects.append(.openApp(name: app)) }
            effects += resolve(item, snapshot: snapshot, state: &state, completing: false)
            if let card = state.cards.first(where: { $0.id == cardID }) {
                effects.append(
                    reaction("open.\(item.kind.rawValue)", card: card, snapshot: snapshot))
            }
            return effects

        case .itemDone(let cardID, let itemID):
            guard let item = removeItem(itemID, fromCard: cardID, state: &state) else { return [] }
            var effects = resolve(item, snapshot: snapshot, state: &state, completing: true)
            if let card = state.cards.first(where: { $0.id == cardID }) {
                effects.append(
                    reaction("done.\(item.kind.rawValue)", card: card, snapshot: snapshot))
            }
            return effects

        case .itemLater(let cardID, let itemID):
            guard let item = removeItem(itemID, fromCard: cardID, state: &state) else { return [] }
            let later = snapshot.now.addingTimeInterval(30 * 60)
            var effects: [DayEffect] = []
            switch item.kind {
            case .notification:
                state.ledger.markSeen([item.id], at: snapshot.now)
                effects.append(
                    .mutateAgenda(.followUp(title: "Follow up: \(item.title)", at: later)))
            case .reminder:
                let reminderID = String(item.id.dropFirst("reminder:".count))
                effects.append(.mutateAgenda(.dueAt(reminderID: reminderID, at: later)))
            case .agent:
                break  // It stays in Waiting on you.
            }
            if let card = state.cards.first(where: { $0.id == cardID }) {
                effects.append(
                    reaction("later.\(item.kind.rawValue)", card: card, snapshot: snapshot))
            }
            return effects

        case .leftoverOn(let cardID, let reminderID, let day):
            guard let index = state.cards.firstIndex(where: { $0.id == cardID }),
                case .eveningWrapUp(var card) = state.cards[index].body,
                card.leftovers.contains(where: { $0.reminderID == reminderID })
            else { return [] }
            card.leftovers.removeAll { $0.reminderID == reminderID }
            state.cards[index].body = .eveningWrapUp(card)
            state.plan.removeAll { $0.reminderID == reminderID }
            return [
                .mutateAgenda(.dueOn(reminderID: reminderID, day: day)),
                reaction("leftover.day", card: state.cards[index], snapshot: snapshot),
            ]

        case .agentHandled(let agentID):
            state.agents.removeAll { $0.id == agentID }
            for index in state.cards.indices {
                if case .breakpoint(var card) = state.cards[index].body {
                    card.needsYou.removeAll { $0.id == "agent:\(agentID)" }
                    state.cards[index].body = .breakpoint(card)
                }
            }
            return [.trace(.cardReaction, ["action": "agentHandled"])]

        case .planNow:
            return run(.morningPlan, trigger: .ownerAsked, snapshot: snapshot, state: &state)

        case .wrapUpNow:
            return run(.eveningWrapUp, trigger: .ownerAsked, snapshot: snapshot, state: &state)
        }
    }

    /// Take an item off a Breakpoint or Triage card.
    private static func removeItem(_ itemID: String, fromCard cardID: String, state: inout DayState)
        -> WaitingItem?
    {
        guard let index = state.cards.firstIndex(where: { $0.id == cardID }) else { return nil }
        switch state.cards[index].body {
        case .breakpoint(var card):
            guard let item = card.needsYou.first(where: { $0.id == itemID }) else { return nil }
            card.needsYou.removeAll { $0.id == itemID }
            state.cards[index].body = .breakpoint(card)
            return item
        case .triage(var card):
            guard let item = card.raise.first(where: { $0.id == itemID }) else { return nil }
            card.raise.removeAll { $0.id == itemID }
            state.cards[index].body = .triage(card)
            if card.raise.isEmpty { state.cards[index].dismissed = true }
            return item
        case .morningPlan, .eveningWrapUp, .reflection:
            return nil
        }
    }

    /// An item the owner handled: a notification is seen, an agent is dealt
    /// with, a reminder is done.
    private static func resolve(
        _ item: WaitingItem, snapshot: DaySnapshot, state: inout DayState, completing: Bool
    ) -> [DayEffect] {
        switch item.kind {
        case .notification:
            state.ledger.markSeen([item.id], at: snapshot.now)
            return [.trace(.notificationSeen, ["via": "card"])]
        case .agent:
            let agentID = String(item.id.dropFirst("agent:".count))
            state.agents.removeAll { $0.id == agentID }
            return []
        case .reminder:
            guard completing else { return [] }
            let reminderID = String(item.id.dropFirst("reminder:".count))
            return [.mutateAgenda(.complete(reminderID: reminderID))]
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

    static func reaction(_ action: String, card: DayCard, snapshot: DaySnapshot) -> DayEffect {
        .trace(
            .cardReaction,
            [
                "card": .string(card.id), "moment": .string(card.kind.rawValue),
                "action": .string(action),
                "secondsToReact": .double(snapshot.now.timeIntervalSince(card.createdAt)),
            ])
    }
}
