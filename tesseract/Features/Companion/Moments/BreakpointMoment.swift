//
//  BreakpointMoment.swift
//  tesseract
//
//  The Breakpoint and Triage moments. The model judges only the owner's
//  unresolved notifications (which need them, which can wait) and writes one
//  warm line; code builds everything else on the card: waiting agents,
//  reminders that came due while away, the next event and what fits before
//  it, where the owner was. With nothing for the model to judge, the card is
//  built by code alone — fast, and no GPU.
//

import Foundation

/// What a Breakpoint or Triage shows and asks about.
nonisolated struct BreakpointInputs: Sendable, Equatable {
    var awayFrom: Date
    var now: Date
    /// Unresolved notifications for the model to judge, in arrival order.
    var toJudge: [SeenLedger.Entry]
    /// Unresolved notifications already judged "can wait" by a Triage.
    var alreadyWaiting: [SeenLedger.Entry]
    /// Notifications an owner rule raises.
    var raisedByRule: [SeenLedger.Entry]
    var agents: [AgentSignal]
    var dueWhileAway: [AgendaReminder]
    var nextEvent: AgendaEvent?
    var fitsBefore: [AgendaReminder]
    var whereYouWere: String?

    /// Nothing needs the owner and nothing is waiting: no card at all.
    var isEmpty: Bool {
        toJudge.isEmpty && alreadyWaiting.isEmpty && raisedByRule.isEmpty && agents.isEmpty
            && dueWhileAway.isEmpty
    }

    /// The short ids the model sees ("n1"…), mapped back on parse.
    func shortID(of index: Int) -> String { "n\(index + 1)" }
}

nonisolated enum BreakpointMoment {

    // MARK: Request

    static func request(_ inputs: BreakpointInputs, calendar: Calendar = .current) -> String {
        let away = Int(inputs.now.timeIntervalSince(inputs.awayFrom) / 60)
        var lines = [
            "[Welcome back — away \(MomentPrompts.minutesText(away)), \(AgendaTime.clock(inputs.awayFrom, calendar: calendar))–\(AgendaTime.clock(inputs.now, calendar: calendar))]"
        ]
        if !inputs.toJudge.isEmpty {
            lines.append("Notifications they haven't seen (id · app — text):")
            for (index, entry) in inputs.toJudge.enumerated() {
                lines.append("- \(inputs.shortID(of: index)) · \(entry.notification.line)")
            }
        }
        if !inputs.agents.isEmpty {
            lines.append("Coding agents:")
            lines += inputs.agents.map {
                "- \($0.title) — \($0.kind == .waiting ? "waiting" : "finished"): \($0.message)"
            }
        }
        if !inputs.dueWhileAway.isEmpty {
            lines.append(
                "Came due while away: " + inputs.dueWhileAway.map(\.title).joined(separator: "; "))
        }
        if let next = inputs.nextEvent {
            lines.append("Next: \(AgendaTime.clock(next.start, calendar: calendar)) \(next.title)")
        }
        if let place = inputs.whereYouWere { lines.append("Where they were: \(place)") }
        lines.append("")
        lines.append("Reply with only this JSON, nothing before or after it:")
        lines.append(
            #"{"line": "<one warm sentence to welcome them back>", "needs_you": ["<notification ids that need them now, most important first>"]}"#
        )
        lines.append(
            "Only people waiting on them or things with a deadline need them; automated updates, promotions and FYIs can wait. Never repeat what they already saw."
        )
        return lines.joined(separator: "\n")
    }

    static func triageRequest(_ entries: [SeenLedger.Entry]) -> String {
        var lines = ["[Triage] New notifications while the owner works (id · app — text):"]
        for (index, entry) in entries.enumerated() {
            lines.append("- n\(index + 1) · \(entry.notification.line)")
        }
        lines.append("")
        lines.append("Reply with only this JSON, nothing before or after it:")
        lines.append(
            #"{"line": "<one short line if something must reach them now, else empty>", "raise": ["<ids that can't wait for their next break>"]}"#
        )
        lines.append(
            "Raise only what is urgent: a person waiting on them right now, or something time-critical. Everything else waits for the next break; an empty list is the usual answer."
        )
        return lines.joined(separator: "\n")
    }

    // MARK: Parse

    private struct Reply: Decodable {
        let line: String?
        let needsYou: [String]?
        let raise: [String]?

        enum CodingKeys: String, CodingKey {
            case line, raise
            case needsYou = "needs_you"
        }
    }

    enum ChoiceField: Sendable {
        case needsYou
        case raise
    }

    /// The notification entries a reply names, by short id, in the reply's
    /// order; nil when there is no readable reply.
    static func choose(_ reply: String, from entries: [SeenLedger.Entry], field: ChoiceField)
        -> (line: String?, entries: [SeenLedger.Entry])?
    {
        guard let data = CardParser.jsonObject(in: reply),
            let decoded = try? JSONDecoder().decode(Reply.self, from: data)
        else { return nil }
        let ids = (field == .needsYou ? decoded.needsYou : decoded.raise) ?? []
        var seen = Set<Int>()
        let picked = ids.compactMap { id -> SeenLedger.Entry? in
            guard id.hasPrefix("n"), let number = Int(id.dropFirst()), number >= 1,
                number <= entries.count, !seen.contains(number)
            else { return nil }
            seen.insert(number)
            return entries[number - 1]
        }
        return (CardParser.cleanLine(decoded.line), picked)
    }

    // MARK: Cards

    /// The Breakpoint card. `important` is the model's pick among `toJudge`
    /// (nil: the fallback — only rule-raised notifications need the owner).
    static func card(_ inputs: BreakpointInputs, line: String?, important: [SeenLedger.Entry]?)
        -> BreakpointCard
    {
        let importantIDs = Set((important ?? []).map(\.id))
        var needsYou: [WaitingItem] = inputs.agents.map(item(for:))
        needsYou += (inputs.raisedByRule + (important ?? [])).map(item(for:))
        needsYou += inputs.dueWhileAway.map {
            WaitingItem(
                id: "reminder:\($0.id)", kind: .reminder, title: $0.title, detail: "Came due",
                app: "Reminders")
        }
        let waiting =
            inputs.alreadyWaiting + inputs.toJudge.filter { !importantIDs.contains($0.id) }
        let groups = Dictionary(grouping: waiting, by: \.notification.app)
            .map { QuietGroup(app: $0.key, lines: $0.value.map(\.notification.line)) }
            .sorted { ($0.lines.count, $1.app) > ($1.lines.count, $0.app) }
        var next: [NextItem] = []
        if let event = inputs.nextEvent {
            next.append(
                NextItem(
                    id: "event:\(event.id)", kind: .event, title: event.title, at: event.start,
                    minutes: Int(event.duration / 60)))
        }
        next += inputs.fitsBefore.prefix(3).map {
            NextItem(id: "task:\($0.id)", kind: .task, title: $0.title, at: nil, minutes: nil)
        }
        return BreakpointCard(
            awayFrom: inputs.awayFrom, awayUntil: inputs.now,
            line: line ?? fallbackLine(inputs, needs: needsYou.count),
            needsYou: needsYou, next: next, whereYouWere: inputs.whereYouWere, canWait: groups)
    }

    static func item(for entry: SeenLedger.Entry) -> WaitingItem {
        let notification = entry.notification
        let detail = [notification.subtitle, notification.body].filter { !$0.isEmpty }.joined(
            separator: " — ")
        return WaitingItem(
            id: entry.id, kind: .notification,
            title: notification.title.isEmpty ? notification.app : notification.title,
            detail: detail.isEmpty ? notification.app : detail, app: notification.app)
    }

    static func item(for agent: AgentSignal) -> WaitingItem {
        WaitingItem(
            id: "agent:\(agent.id)", kind: .agent, title: agent.title,
            detail: agent.kind == .waiting ? agent.message : "Finished: \(agent.message)",
            app: nil)
    }

    static func fallbackLine(_ inputs: BreakpointInputs, needs: Int) -> String {
        switch needs {
        case 0: "Welcome back. Nothing needs you."
        case 1: "Welcome back. One thing needs you."
        default: "Welcome back. \(needs) things need you."
        }
    }
}
