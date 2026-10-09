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
        var context = context
        let body: String
        switch kind {
        case .morningPlan:
            if text == nil { context.eventIDs = MomentPrompts.leavingEvents(facts).map(\.id) }
            body = text ?? MomentPrompts.morningPlan(facts: facts)
        case .eveningWrapUp:
            body = text ?? MomentPrompts.eveningWrapUp(facts: facts, leftovers: leftovers(facts))
        case .breakpoint, .triage, .nightReflection:
            guard let text else { return [] }
            body = text
        }
        state.running = kind
        let request = MomentRequest(
            kind: kind, trigger: trigger, text: body, attempt: attempt, context: context,
            day: state.day)
        state.runningRequest = request
        state.runningSince = snapshot.now
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

    /// The Morning Plan: a card built by code goes up at once, and Jarvis's
    /// version replaces it in place when he has thought the day through —
    /// the owner never waits on the model for the essentials.
    static func morningPlan(
        trigger: MomentTrigger, snapshot: DaySnapshot, state: inout DayState
    ) -> [DayEffect] {
        guard state.running == nil, !snapshot.chatBusy else { return [] }
        // Whatever starts it (Today opened, Plan my day), a plan that waited
        // for the model is no longer waiting.
        state.morningPlanWaiting = false
        let facts = snapshot.facts(state: state)
        // The day's first sit-down is the owner starting their day: the plan
        // meets them on the panel, even before quiet hours end.
        let rungs =
            trigger == .firstPresence
            ? DeliveryLadder.rungs(for: .normal, snapshot: snapshot, sittingDown: true) : nil
        var effects = accept(
            .morningPlan(FallbackCards.morningPlan(facts: facts)), kind: .morningPlan,
            fallback: false, snapshot: snapshot, state: &state, refining: true, rungs: rungs)
        let cardID = state.cards.last?.id
        effects += run(
            .morningPlan, trigger: trigger, snapshot: snapshot, state: &state,
            context: MomentContext(cardID: cardID))
        return effects
    }

    /// A Morning Plan the app quit in the middle of runs again, once: its
    /// time was set when the code card went up, so nothing else would ever
    /// finish it, and the day would go by with no plan. Not once the owner
    /// closed the card, nor in the evening. The other moments run again on
    /// their own triggers.
    static func resumeInterrupted(snapshot: DaySnapshot, state: inout DayState) -> [DayEffect] {
        guard let kind = state.interrupted else { return [] }
        state.interrupted = nil
        guard kind == .morningPlan, !state.morningPlanResumed, !isEvening(snapshot),
            let index = state.cards.lastIndex(where: { $0.kind == .morningPlan }),
            !state.cards[index].dismissed
        else { return [] }
        state.morningPlanResumed = true
        state.cards[index].isRefining = true
        let effects = run(
            .morningPlan, trigger: .resumed, snapshot: snapshot, state: &state,
            context: MomentContext(cardID: state.cards[index].id))
        if state.running == nil { state.cards[index].isRefining = false }
        return effects
    }

    /// Plan the day before the owner sits down, when the Mac is awake and on
    /// power in the morning window and they have been away all night. The
    /// card waits in Today and comes forward at the first sit-down.
    static func prepareMorningPlanIfDue(snapshot: DaySnapshot, state: inout DayState)
        -> [DayEffect]
    {
        let hour = snapshot.minuteOfDay / 60
        let away = snapshot.now.timeIntervalSince(state.lastPresentAt ?? .distantPast)
        guard state.morningPlanAt == nil, !snapshot.ownerPresent,
            hour >= snapshot.settings.morningStartHour, hour < snapshot.settings.morningEndHour,
            away >= DaySettings.overnightGap, snapshot.power.onACPower,
            snapshot.power.thermal <= .fair
        else { return [] }
        return morningPlan(trigger: .prepared, snapshot: snapshot, state: &state)
    }

    /// A plan made while the owner was away comes forward when they sit down.
    static func presentPreparedMorningPlan(
        awayFrom: Date, snapshot: DaySnapshot, state: DayState
    ) -> [DayEffect] {
        guard
            let card = state.cards.last(where: {
                $0.kind == .morningPlan && !$0.dismissed && $0.createdAt >= awayFrom
            })
        else { return [] }
        let rungs = DeliveryLadder.rungs(for: .normal, snapshot: snapshot, sittingDown: true)
            .filter { $0 == .panel || $0 == .today }
        return rungs.map { .presentCard(card, $0) } + [
            .trace(
                .cardPresented,
                [
                    "card": .string(card.id), "moment": "morningPlan",
                    "rungs": .string(rungs.map(\.rawValue).joined(separator: ",")),
                    "prepared": true, "refining": .bool(card.isRefining),
                ])
        ]
    }

    /// What the Evening Wrap-up asks about: today's open tasks, planned or
    /// due today. On the week's last day also what has waited longest — up
    /// to five overdue tasks, oldest first — so the week ends on a clean
    /// slate, not a pile carried from day to day.
    static func leftovers(_ facts: DayFacts) -> [AgendaReminder] {
        let planned = Set(facts.plan.map(\.reminderID))
        let today = facts.openTasks.filter { reminder in
            if planned.contains(reminder.id) { return true }
            guard let due = reminder.due else { return false }
            return due >= facts.startOfToday && due < facts.endOfToday
        }
        guard facts.isWeekReview else { return today }
        let asked = Set(today.map(\.id))
        let waiting = facts.dueOrOverdue
            .filter { facts.isWaiting($0) && !asked.contains($0.id) }
            .sorted { ($0.due ?? .distantPast) < ($1.due ?? .distantPast) }
        return today + waiting.prefix(waitingAsked)
    }

    /// The most overdue tasks the week's look-back asks about.
    static let waitingAsked = 5

    // MARK: - Outcomes

    /// How long a moment may stay quiet, the Mac awake: the slowest run
    /// (a cold Night Reflection, waiting on the owner's own chat) takes a
    /// few minutes.
    static let momentPatience: TimeInterval = 15 * 60

    /// A moment quiet past `momentPatience` is given up on: its generation
    /// is cancelled, and it fails as any failed call does — one retry, then
    /// the card code built. A stalled run held every later moment back (no
    /// Triage, no Breakpoint, no Evening Wrap-up) until the app restarted.
    /// Time the Mac slept doesn't count: the run slept with it.
    static func giveUpOnQuietMoment(snapshot: DaySnapshot, state: inout DayState)
        -> [DayEffect]
    {
        guard let request = state.runningRequest, let since = state.runningSince else {
            return []
        }
        if let last = state.lastTickAt, snapshot.now.timeIntervalSince(last) > 3 * 60 {
            state.runningSince = since.addingTimeInterval(snapshot.now.timeIntervalSince(last))
            return []
        }
        guard snapshot.now.timeIntervalSince(since) >= momentPatience else { return [] }
        let minutes = Int(momentPatience / 60)
        return [.cancelMoment]
            + momentFinished(
                request, .failed("no answer in \(minutes) min", nil), snapshot: snapshot,
                state: &state)
    }

    static func momentFinished(
        _ request: MomentRequest, _ outcome: MomentOutcome, snapshot: DaySnapshot,
        state: inout DayState
    ) -> [DayEffect] {
        if let day = request.day, day != state.day {
            return lateMomentFinished(request, outcome, snapshot: snapshot, state: &state)
        }
        // A reply to a moment given up on: the one running now isn't it.
        if let running = state.runningRequest, running != request { return [] }
        state.running = nil
        state.runningRequest = nil
        state.runningSince = nil
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
            guard
                case .card(let body) = CardParser.morningPlan(
                    reply, facts: facts, eventIDs: request.context.eventIDs)
            else { return nil }
            return accept(
                body, kind: .morningPlan, fallback: false, snapshot: snapshot, state: &state,
                cardID: request.context.cardID)
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
            guard
                case .card(let body) = CardParser.nightReflection(
                    reply, facts: facts, open: snapshot.agenda.open)
            else { return nil }
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
            // The code-built card is already up: it stands, as the facts.
            if let cardID = request.context.cardID,
                let index = state.cards.firstIndex(where: { $0.id == cardID })
            {
                state.cards[index].isFallback = true
                state.cards[index].isRefining = false
                let card = state.cards[index]
                guard !card.dismissed else { return [] }
                // Taken in already: it stays in Today, never on the panel again.
                guard !card.kept else { return [.presentCard(card, .today)] }
                return DeliveryLadder.rungs(for: .normal, snapshot: snapshot)
                    .filter { $0 == .panel || $0 == .today }
                    .map { .presentCard(card, $0) }
            }
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

    /// Put a new card on the day and deliver it. With `cardID`, the model's
    /// version replaces the card code put up first; `refining` marks a code
    /// card the model is still working on; `rungs` overrides where a new card
    /// goes.
    static func accept(
        _ body: DayCard.Body, kind: MomentKind, fallback: Bool, snapshot: DaySnapshot,
        state: inout DayState, importance: Importance = .normal, cardID: String? = nil,
        refining: Bool = false, rungs: [DeliveryRung]? = nil
    ) -> [DayEffect] {
        switch body {
        case .morningPlan(let card):
            state.morningPlanAt = snapshot.now
            if let mustDo = card.mustDoID { setMustDo(mustDo, state: &state) }
            if !card.placements.isEmpty {
                // A step the owner started and is still in keeps its slot.
                let running = state.plan.filter { slot in
                    state.startedSteps.contains(StepCue.key(slot)) && slot.start <= snapshot.now
                        && snapshot.now
                            < slot.start.addingTimeInterval(TimeInterval(slot.minutes * 60))
                }
                state.plan =
                    (card.placements.filter { placement in
                        !running.contains { $0.reminderID == placement.reminderID }
                    } + running).sorted { $0.start < $1.start }
            }
            // Jarvis's own plan sets the day's departures, none included (a
            // class that went online); the code card leaves them be.
            if !fallback, !refining { state.departures = card.departures }
        case .eveningWrapUp(let card):
            state.eveningWrapUpAt = snapshot.now
            // The week's look-back names next week's focus: it rides each
            // day's opening and the plan until the next one.
            if let focus = card.focus {
                state.weekFocus = focus
                state.weekFocusSetAt = snapshot.now
            }
        case .reflection(let card):
            state.nightReflectionAt = snapshot.now
            state.carryOverForNextDay = card.carryOver
            state.taskProposals = card.tasks
            state.draftForNextDay = card.tomorrow
        case .breakpoint, .triage:
            break
        }
        // Refining a card in place (the model's version of a Breakpoint or
        // a Morning Plan code put up first).
        if let cardID, let index = state.cards.firstIndex(where: { $0.id == cardID }) {
            state.cards[index].isRefining = false
            guard !state.cards[index].dismissed else { return [] }
            // A card code kept in Today stays there unless the model found
            // something for the owner; one already on the panel updates there.
            let wasQuiet =
                deliveryRungs(state.cards[index].body, importance: importance, snapshot: snapshot)
                == [.today]
            state.cards[index].body = body
            state.cards[index].isFallback = fallback
            let card = state.cards[index]
            // Taken in already: it updates in Today, never on the panel again.
            let rungs =
                card.kept
                ? [.today]
                : wasQuiet
                    ? deliveryRungs(body, importance: importance, snapshot: snapshot)
                    : DeliveryLadder.rungs(for: importance, snapshot: snapshot)
            return rungs.filter { $0 == .panel || $0 == .today }.map { .presentCard(card, $0) }
        }
        // Triage cards gather: what still waits on an open one stays, the
        // new items after it — a second raise mustn't hide the first.
        var body = body
        if case .triage(var incoming) = body {
            let fresh = Set(incoming.raise.map(\.id))
            let waiting = state.cards.filter { $0.kind == .triage && !$0.dismissed }
                .flatMap { card -> [WaitingItem] in
                    if case .triage(let old) = card.body { old.raise } else { [] }
                }
                .filter {
                    !fresh.contains($0.id) && stillWaits($0, snapshot: snapshot, state: state)
                }
            incoming.raise = waiting + incoming.raise
            body = .triage(incoming)
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
            kind: kind, createdAt: snapshot.now, isFallback: fallback, body: body,
            isRefining: refining)
        state.cards.append(card)
        let rungs = rungs ?? deliveryRungs(body, importance: importance, snapshot: snapshot)
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
                    "refining": .bool(refining),
                ]))
        if case .reflection(let reflection) = body, !reflection.proposals.isEmpty {
            effects.append(.proposeFacts(reflection.proposals))
        }
        if case .reflection(let reflection) = body, !reflection.tasks.isEmpty {
            effects.append(.trace(.taskProposed, ["count": .int(reflection.tasks.count)]))
        }
        return effects
    }

    /// Where a card goes. The night's reflection waits in Today, and so does
    /// a Breakpoint with nothing that needs the owner — "nothing needs you"
    /// never pops up over their work. Every other card takes the ladder (the
    /// panel when the owner is at the Mac).
    static func deliveryRungs(_ body: DayCard.Body, importance: Importance, snapshot: DaySnapshot)
        -> [DeliveryRung]
    {
        switch body {
        case .reflection:
            return [.today]
        case .breakpoint(let card) where card.needsYou.isEmpty:
            return [.today]
        case .morningPlan, .eveningWrapUp, .breakpoint, .triage:
            return DeliveryLadder.rungs(for: importance, snapshot: snapshot)
        }
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
            fields["waitSeconds"] = .double(measure.waitSeconds)
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
            setMustDo(reminderID, state: &state)
            return [.trace(.cardReaction, ["action": "mustDo", "set": .bool(reminderID != nil)])]

        case .removeFromPlan(let reminderID):
            state.plan.removeAll { $0.reminderID == reminderID }
            return [.trace(.cardReaction, ["action": "unplanned"])]

        case .place(let reminderID, let start, let minutes):
            let slot = Placement(reminderID: reminderID, start: start, minutes: minutes)
            place(slot, snapshot: snapshot, state: &state)
            return [.trace(.cardReaction, ["action": "placed", "minutes": .int(minutes)])]

        case .placeAll(let slots):
            for slot in slots { place(slot, snapshot: snapshot, state: &state) }
            return [.trace(.cardReaction, ["action": "fitted", "count": .int(slots.count)])]

        case .leftover(let cardID, let reminderID, let suggestion):
            guard let index = state.cards.firstIndex(where: { $0.id == cardID }),
                case .eveningWrapUp(var card) = state.cards[index].body,
                card.leftovers.contains(where: { $0.reminderID == reminderID })
            else { return [] }
            card.leftovers.removeAll { $0.reminderID == reminderID }
            state.cards[index].body = .eveningWrapUp(card)
            state.plan.removeAll { $0.reminderID == reminderID }
            return [
                .mutateAgenda(
                    mutation(for: suggestion, reminderID: reminderID, snapshot: snapshot)),
                reaction(
                    "leftover.\(suggestion.rawValue)", card: state.cards[index], snapshot: snapshot),
            ]

        case .allLeftovers(let cardID):
            guard let index = state.cards.firstIndex(where: { $0.id == cardID }),
                case .eveningWrapUp(var card) = state.cards[index].body
            else { return [] }
            let mutations = card.leftovers.map {
                DayEffect.mutateAgenda(
                    mutation(for: $0.suggestion, reminderID: $0.reminderID, snapshot: snapshot))
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
            return morningPlan(trigger: .ownerAsked, snapshot: snapshot, state: &state)

        case .wrapUpNow:
            return run(.eveningWrapUp, trigger: .ownerAsked, snapshot: snapshot, state: &state)

        case .step(let reminderID, let choice):
            return stepChosen(reminderID, choice, snapshot: snapshot, state: &state)

        case .breakCue(let choice):
            return breakChosen(choice, snapshot: snapshot, state: &state)

        case .taskProposal(let id, let add):
            guard let proposal = state.taskProposals.first(where: { $0.id == id }) else {
                return []
            }
            state.taskProposals.removeAll { $0.id == id }
            // Already in Reminders (the owner wrote it down meanwhile): no twin.
            let exists = snapshot.agenda.open.contains {
                $0.title.lowercased() == proposal.title.lowercased()
            }
            let decided = DayEffect.trace(
                .taskDecided,
                [
                    "added": .bool(add && !exists), "dated": .bool(proposal.due != nil),
                    "existed": .bool(exists),
                ])
            return add && !exists
                ? [.mutateAgenda(.add(title: proposal.title, due: proposal.due)), decided]
                : [decided]

        case .keep(let cardID):
            // Taken in: off the panel (the panel closes itself), still in Today.
            guard let index = state.cards.firstIndex(where: { $0.id == cardID }) else { return [] }
            state.cards[index].kept = true
            return [reaction("kept", card: state.cards[index], snapshot: snapshot)]
        }
    }

    /// An item a card raised that still waits on the owner: a banner not
    /// seen since (read in its own app), an agent still waiting (not answered
    /// in the terminal), a reminder still open.
    private static func stillWaits(_ item: WaitingItem, snapshot: DaySnapshot, state: DayState)
        -> Bool
    {
        switch item.kind {
        case .notification:
            return state.ledger.entry(item.id).map { $0.seenAt == nil } ?? false
        case .agent:
            let id = String(item.id.dropFirst("agent:".count))
            return state.agentsWaiting(now: snapshot.now).contains { $0.id == id }
        case .reminder:
            let id = String(item.id.dropFirst("reminder:".count))
            return snapshot.agenda.open.contains { $0.id == id }
        }
    }

    /// A task's slot in today's plan, in place of any it had; one from now
    /// is started, and ends before the next meeting.
    private static func place(_ slot: Placement, snapshot: DaySnapshot, state: inout DayState) {
        state.plan.removeAll { $0.reminderID == slot.reminderID }
        state.plan.append(slot)
        state.plan.sort { $0.start < $1.start }
        guard slot.start <= snapshot.now.addingTimeInterval(60) else { return }
        keepClear(slot.reminderID, snapshot: snapshot, state: &state)
        markStartedIfNow(slot, snapshot: snapshot, state: &state)
    }

    /// A reply that landed after the 04:00 rollover (the lid closed on the
    /// Night Reflection, the Mac woke past four) belongs to the day it ran
    /// for. A night's reflection opens this morning instead — its note, its
    /// draft and its proposed tasks, where the morning has none — without
    /// taking tonight's; anything else is dropped (a wrap-up's leftovers were
    /// that day's). The new day's own moment, if one runs, keeps running.
    private static func lateMomentFinished(
        _ request: MomentRequest, _ outcome: MomentOutcome, snapshot: DaySnapshot,
        state: inout DayState
    ) -> [DayEffect] {
        var fields = traceFields(request, outcome: outcome, power: snapshot.power)
        fields["late"] = true
        guard request.kind == .nightReflection, case .reply(let text, _) = outcome,
            request.day?.next(calendar: snapshot.calendar) == state.day,
            case .card(.reflection(let card)) = CardParser.nightReflection(
                text, facts: snapshot.facts(state: state), open: snapshot.agenda.open)
        else {
            fields["reason"] = "after the rollover"
            return [.trace(.momentFailed, fields)]
        }
        state.carryOver = state.carryOver ?? card.carryOver
        if state.draft.isEmpty { state.draft = card.tomorrow }
        // Its "tomorrow" is this day: read with this day's facts, a task's
        // due date would be the day after.
        let today = state.day.date(calendar: snapshot.calendar)
        if state.taskProposals.isEmpty {
            state.taskProposals = card.tasks.map { task in
                var task = task
                if task.due != nil { task.due = today }
                return task
            }
        }
        var effects: [DayEffect] = [.trace(.momentFinished, fields)]
        if !card.proposals.isEmpty { effects.append(.proposeFacts(card.proposals)) }
        return effects
    }

    /// The day's must-do. A different one starts undone: the done mark was
    /// the old one's.
    static func setMustDo(_ reminderID: String?, state: inout DayState) {
        if state.mustDoID != reminderID { state.mustDoDoneAt = nil }
        state.mustDoID = reminderID
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

    /// A repeating reminder is one item for its whole series: Later can't
    /// take its date (EventKit refuses) and Let go would delete every
    /// occurrence, so for it both skip to tomorrow and the series goes on.
    private static func mutation(
        for suggestion: Leftover.Suggestion, reminderID: String, snapshot: DaySnapshot
    ) -> AgendaMutation {
        let repeats = snapshot.agenda.open.first { $0.id == reminderID }?.repeats == true
        switch suggestion {
        case .tomorrow: return .dueTomorrow(reminderID: reminderID)
        case .later where !repeats: return .clearDue(reminderID: reminderID)
        case .drop where !repeats: return .delete(reminderID: reminderID)
        case .later, .drop: return .dueTomorrow(reminderID: reminderID)
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
