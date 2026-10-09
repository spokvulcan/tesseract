//
//  DayEngine+Breakpoints.swift
//  tesseract
//
//  Coming back, and the world that kept moving: Breakpoints, Triage, other
//  apps' notifications and what the owner has seen, coding agents, and the
//  governor's deferred moments.
//
//  A Breakpoint shows a code-built card at once and lets the model refine it
//  only when there are notifications to judge; with nothing waiting there is
//  no card at all. Triage runs while the owner works, at most every ten
//  minutes, only on new unresolved notifications, and raises only what can't
//  wait. A waiting coding agent is spoken about once, when the owner has
//  been out of the terminal for two minutes — a rule, not a model call.
//

import Foundation

nonisolated extension DayEngine {

    // MARK: - Breakpoint

    static func breakpoint(
        awayFrom: Date, trigger: MomentTrigger, snapshot: DaySnapshot, state: inout DayState
    ) -> [DayEffect] {
        let inputs = breakpointInputs(awayFrom: awayFrom, snapshot: snapshot, state: state)
        guard !inputs.isEmpty else {
            return [
                .trace(
                    .cardPresented,
                    ["moment": "breakpoint", "rungs": "none", "reason": "nothing waiting"])
            ]
        }
        // The card goes up at once, built by code.
        let card = BreakpointMoment.card(inputs, line: nil, important: nil)
        var effects = accept(
            .breakpoint(card), kind: .breakpoint, fallback: false, snapshot: snapshot, state: &state
        )
        let cardID = state.cards.last?.id
        let offered = inputs.toJudge.map(\.id)
        let shown = (inputs.alreadyWaiting + inputs.raisedByRule).map(\.id)
        // Everything the card shows is now in front of the owner, except what
        // the model is about to judge.
        state.ledger.markPresented(shown, at: snapshot.now)
        guard !inputs.toJudge.isEmpty else {
            return effects
        }
        effects += run(
            .breakpoint, trigger: trigger, snapshot: snapshot, state: &state,
            text: BreakpointMoment.request(inputs, calendar: snapshot.calendar),
            context: MomentContext(
                awayFrom: awayFrom, awayUntil: snapshot.now, notificationIDs: offered,
                cardID: cardID, shownIDs: shown))
        if state.running != .breakpoint {
            // It couldn't run (the owner is chatting): the code-built card stands.
            state.ledger.markPresented(offered, at: snapshot.now)
        }
        return effects
    }

    static func breakpointInputs(awayFrom: Date, snapshot: DaySnapshot, state: DayState)
        -> BreakpointInputs
    {
        let now = snapshot.now
        let unresolved = state.ledger.unresolved(now: now)
        let raised = unresolved.filter { $0.rule == .raise }
        // Held banners (an app's own news, or an owner's "hold" rule) wait
        // without a model, like the ones a Triage already judged; only the
        // rest — people — are judged.
        let waiting = unresolved.filter {
            $0.rule == .hold || ($0.rule == nil && $0.triagedAt != nil)
        }
        let judge = unresolved.filter { $0.rule == nil && $0.triagedAt == nil }
        let dueWhileAway = snapshot.agenda.open.filter {
            guard $0.dueHasTime, let due = $0.due else { return false }
            return due >= awayFrom && due <= now
        }
        let next = snapshot.agenda.events
            .filter {
                !$0.isAllDay && $0.start > now
                    && snapshot.calendar.isDate($0.start, inSameDayAs: now)
            }
            .min { $0.start < $1.start }
        var fits: [AgendaReminder] = []
        if next.map({ $0.start.timeIntervalSince(now) >= 15 * 60 }) ?? true {
            let facts = snapshot.facts(state: state)
            fits =
                facts.dueOrOverdue.filter { !$0.dueHasTime }
                + facts.undated.filter { reminder in
                    state.plan.contains { $0.reminderID == reminder.id }
                }
        }
        return BreakpointInputs(
            awayFrom: awayFrom, now: now, toJudge: judge, alreadyWaiting: waiting,
            raisedByRule: raised, agents: state.agentsWaiting(now: now), dueWhileAway: dueWhileAway,
            nextEvent: next, fitsBefore: fits, whereYouWere: state.whereYouWere)
    }

    static func breakpointReplied(
        _ request: MomentRequest, reply: String, snapshot: DaySnapshot, state: inout DayState
    ) -> [DayEffect]? {
        // A newer Breakpoint replaced this one's card while the model ran:
        // it judges the same banners, so this reply marks nothing.
        if let cardID = request.context.cardID,
            let card = state.cards.first(where: { $0.id == cardID }), card.dismissed,
            state.cards.contains(where: {
                $0.kind == .breakpoint && !$0.dismissed && $0.createdAt >= card.createdAt
                    && $0.id != cardID
            })
        {
            return []
        }
        // By the request's numbering: one gone from the ledger meanwhile
        // leaves a gap, not a shift onto the next banner.
        let numbered = request.context.notificationIDs.map { state.ledger.entry($0) }
        let offered = numbered.compactMap { $0 }
        guard
            let choice = BreakpointMoment.choose(reply, from: numbered, field: .needsYou),
            choice.line != nil
        else { return nil }
        state.ledger.markPresented(request.context.notificationIDs, at: snapshot.now)
        state.ledger.markTriaged(request.context.notificationIDs, at: snapshot.now)
        // Rebuild from what is true now: items handled meanwhile stay gone.
        var inputs = breakpointInputs(
            awayFrom: request.context.awayFrom ?? snapshot.now, snapshot: snapshot, state: state)
        inputs.now = request.context.awayUntil ?? snapshot.now
        let stillOpen = Set(offered.filter { $0.seenAt == nil }.map(\.id))
        inputs.toJudge = offered.filter { stillOpen.contains($0.id) }
        // What the code card showed besides them still waits: marked
        // presented, it is no longer unresolved, so the rebuild alone would
        // drop it.
        let shown = request.context.shownIDs.compactMap { state.ledger.entry($0) }
            .filter { $0.seenAt == nil }
        let fresh = Set((inputs.raisedByRule + inputs.alreadyWaiting).map(\.id))
        inputs.raisedByRule += shown.filter { $0.rule == .raise && !fresh.contains($0.id) }
        inputs.alreadyWaiting += shown.filter { $0.rule != .raise && !fresh.contains($0.id) }
        let card = BreakpointMoment.card(
            inputs, line: choice.line,
            important: choice.entries.filter { stillOpen.contains($0.id) })
        return accept(
            .breakpoint(card), kind: .breakpoint, fallback: false, snapshot: snapshot,
            state: &state,
            cardID: request.context.cardID)
    }

    // MARK: - Triage

    static func triageIfDue(snapshot: DaySnapshot, state: inout DayState) -> [DayEffect] {
        // Never while the owner plays: what arrived waits for the next
        // Breakpoint, and the GPU stays the game's.
        guard state.running == nil, !snapshot.chatBusy, !snapshot.frontmostIsGame else {
            return []
        }
        if let last = state.lastTriageAt,
            snapshot.now.timeIntervalSince(last) < DaySettings.triageInterval
        {
            return []
        }
        let entries = state.ledger.untriaged(now: snapshot.now)
        guard !entries.isEmpty else { return [] }
        let effects = run(
            .triage, trigger: .notifications, snapshot: snapshot, state: &state,
            text: BreakpointMoment.triageRequest(entries),
            context: MomentContext(notificationIDs: entries.map(\.id)))
        if state.running == .triage { state.lastTriageAt = snapshot.now }
        return effects
    }

    static func triageReplied(
        _ request: MomentRequest, reply: String, snapshot: DaySnapshot, state: inout DayState
    ) -> [DayEffect]? {
        // By the request's numbering: one gone from the ledger meanwhile
        // leaves a gap, not a shift onto the next banner.
        let numbered = request.context.notificationIDs.map { state.ledger.entry($0) }
        let offered = numbered.compactMap { $0 }
        guard let choice = BreakpointMoment.choose(reply, from: numbered, field: .raise) else {
            return nil
        }
        let raised = choice.entries.filter { $0.seenAt == nil }
        let raisedIDs = raised.map(\.id)
        state.ledger.markTriaged(request.context.notificationIDs, at: snapshot.now)
        state.ledger.markPresented(raisedIDs, at: snapshot.now)
        var effects: [DayEffect] = [
            .trace(
                .notificationTriaged,
                ["offered": .int(offered.count), "raised": .int(raised.count)])
        ]
        guard !raised.isEmpty else { return effects }
        let line = choice.line ?? "\(raised.count == 1 ? "This" : "These") can't wait."
        effects += accept(
            .triage(TriageCard(line: line, raise: raised.map(BreakpointMoment.item(for:)))),
            kind: .triage, fallback: false, snapshot: snapshot, state: &state, importance: .urgent)
        return effects
    }

    // MARK: - Notifications

    static func notificationArrived(
        _ notification: ObservedNotification, snapshot: DaySnapshot, state: inout DayState
    ) -> [DayEffect] {
        guard
            state.ledger.arrived(
                notification, present: snapshot.ownerPresent, rules: snapshot.settings.rules)
        else { return [] }
        let rule = state.ledger.entry(notification.id)?.rule
        var effects: [DayEffect] = [
            .trace(
                .notificationArrived,
                [
                    "id": .string(notification.id), "app": .string(notification.app),
                    "rule": .string(rule?.rawValue ?? "none"),
                    "source": .string(notification.source?.rawValue ?? "unknown"),
                ])
        ]
        // An owner rule that raises: straight to the owner, no model.
        if rule == .raise, let entry = state.ledger.entry(notification.id) {
            state.ledger.markPresented([notification.id], at: snapshot.now)
            effects += accept(
                .triage(
                    TriageCard(line: notification.line, raise: [BreakpointMoment.item(for: entry)])),
                kind: .triage, fallback: false, snapshot: snapshot, state: &state,
                importance: .urgent)
        }
        return effects
    }

    static func appActivated(
        name: String, bundleID: String?, snapshot: DaySnapshot, state: inout DayState
    ) -> [DayEffect] {
        var effects: [DayEffect] = state.ledger.appActivated(name, at: snapshot.now).map {
            .trace(.notificationSeen, ["id": .string($0), "via": "app"])
        }
        // The terminal in front: its agents are in view, so they no longer wait.
        if TerminalApps.contains(bundleID), !state.agents.isEmpty {
            state.agents.removeAll { $0.at <= snapshot.now }
            for index in state.cards.indices {
                if case .breakpoint(var card) = state.cards[index].body {
                    card.needsYou.removeAll { $0.kind == .agent }
                    state.cards[index].body = .breakpoint(card)
                }
            }
            effects.append(.trace(.cardReaction, ["action": "agentsInView"]))
        }
        return effects
    }

    // MARK: - Coding agents

    static func agentSignal(_ agent: AgentSignal, snapshot: DaySnapshot, state: inout DayState)
        -> [DayEffect]
    {
        state.agents.removeAll { $0.id == agent.id }
        state.agents.append(agent)
        var effects: [DayEffect] = [
            .trace(
                .agentSignal,
                [
                    "kind": .string(agent.kind.rawValue), "agent": .string(agent.agent),
                    "project": .string(agent.project),
                ])
        ]
        if snapshot.ownerPresent {
            effects += speakForWaitingAgents(snapshot: snapshot, state: &state)
        }
        return effects
    }

    /// The coding-agent rule: an agent waits, the owner is at the Mac, and the
    /// terminal has been out of sight for two minutes → one spoken line, once.
    static func speakForWaitingAgents(snapshot: DaySnapshot, state: inout DayState) -> [DayEffect] {
        let terminalAway = snapshot.now.timeIntervalSince(
            snapshot.lastTerminalFrontAt ?? .distantPast)
        guard terminalAway >= DaySettings.agentSpeakAfter,
            !TerminalApps.contains(snapshot.frontmostBundleID)
        else { return [] }
        var effects: [DayEffect] = []
        for agent in state.agentsWaiting(now: snapshot.now) where agent.kind == .waiting {
            if let spoken = state.agentSpokenAt[agent.id], spoken >= agent.at { continue }
            state.agentSpokenAt[agent.id] = snapshot.now
            let line = "\(agent.title) is waiting for you."
            let rungs = DeliveryLadder.rungs(for: .urgent, snapshot: snapshot)
            if rungs.contains(.voice) {
                effects.append(.speak(line))
            } else if rungs.contains(.panel) || rungs.contains(.banner) {
                effects += accept(
                    .triage(TriageCard(line: line, raise: [BreakpointMoment.item(for: agent)])),
                    kind: .triage, fallback: false, snapshot: snapshot, state: &state,
                    importance: .urgent)
            }
            effects.append(
                .trace(
                    .cardPresented,
                    [
                        "moment": "agent",
                        "rungs": .string(rungs.map(\.rawValue).joined(separator: ",")),
                    ]))
        }
        return effects
    }

    // MARK: - Governor

    /// Moments the governor held back run once the Mac allows.
    static func runDeferred(snapshot: DaySnapshot, state: inout DayState) -> [DayEffect] {
        guard !state.deferred.isEmpty, state.running == nil else { return [] }
        if state.deferred.contains(.triage),
            Governor.deferral(for: .triage, power: snapshot.power) == nil
        {
            state.deferred.remove(.triage)
            state.lastTriageAt = nil
            return snapshot.ownerPresent ? triageIfDue(snapshot: snapshot, state: &state) : []
        }
        if state.deferred.contains(.nightReflection),
            Governor.deferral(for: .nightReflection, power: snapshot.power) == nil
        {
            state.deferred.remove(.nightReflection)
            return nightReflectionIfDue(snapshot: snapshot, state: &state)
        }
        return []
    }
}
