//
//  CardParser.swift
//  tesseract
//
//  A moment's reply becomes a card here, or it doesn't. The reply must hold
//  one JSON object; ids it names must be ones the moment was shown; times
//  must fall in the day. Stray entries are dropped rather than failing the
//  whole card, but a reply with no readable card is invalid, and the engine
//  retries once, then falls back.
//

import Foundation

nonisolated enum CardParse: Sendable, Equatable {
    case card(DayCard.Body)
    case invalid(String)
}

nonisolated enum CardParser {

    /// The JSON object inside a reply: code fences and prose around it are
    /// tolerated; the outermost braces win.
    static func jsonObject(in reply: String) -> Data? {
        var text = reply
        if let fence = text.range(of: "```") {
            text = String(text[fence.upperBound...])
            if text.hasPrefix("json") { text = String(text.dropFirst(4)) }
            if let close = text.range(of: "```") { text = String(text[..<close.lowerBound]) }
        }
        guard let open = text.firstIndex(of: "{"), let close = text.lastIndex(of: "}"),
            open < close
        else { return nil }
        return Data(text[open...close].utf8)
    }

    // MARK: Morning Plan

    private struct MorningPlanReply: Decodable {
        struct Entry: Decodable {
            let id: String
            let at: String
            let minutes: Int?
        }
        struct Leave: Decodable {
            let event: String
            let at: String
        }
        let line: String?
        let mustDo: String?
        let plan: [Entry]?
        let suggestions: [String]?
        let leave: [Leave]?

        enum CodingKeys: String, CodingKey {
            case line, plan, suggestions, leave
            case mustDo = "must_do"
        }
    }

    /// The longest a departure may come before its event.
    static let longestTravel: TimeInterval = 3 * 3600

    /// - Parameter eventIDs: the events the request listed for leaving, in
    ///   the order their short ids ("e1"…) number them.
    static func morningPlan(_ reply: String, facts: DayFacts, eventIDs: [String] = [])
        -> CardParse
    {
        guard let data = jsonObject(in: reply) else { return .invalid("no JSON object") }
        guard let decoded = try? JSONDecoder().decode(MorningPlanReply.self, from: data) else {
            return .invalid("the JSON is not a Morning Plan")
        }
        guard let line = cleanLine(decoded.line) else { return .invalid("no line") }
        let mustDo = decoded.mustDo.flatMap { facts.task($0) != nil ? $0 : nil }
        let busy = facts.eventsToday.map { DateInterval(start: $0.start, end: $0.end) }
        var seen = Set<String>()
        let placements: [Placement] = (decoded.plan ?? []).compactMap { entry in
            guard facts.task(entry.id) != nil, !seen.contains(entry.id),
                let start = AgendaTime.parse(entry.at, now: facts.now, calendar: facts.calendar),
                start.hasTime, facts.calendar.isDate(start.date, inSameDayAs: facts.now),
                start.date >= facts.now.addingTimeInterval(-15 * 60)
            else { return nil }
            let minutes = min(max(entry.minutes ?? 15, 5), 240)
            let slot = DateInterval(start: start.date, duration: TimeInterval(minutes * 60))
            // Never over an event.
            guard
                !busy.contains(where: {
                    $0.intersects(slot) && $0.end != slot.start && slot.end != $0.start
                })
            else { return nil }
            seen.insert(entry.id)
            return Placement(reminderID: entry.id, start: start.date, minutes: minutes)
        }
        let suggestions = (decoded.suggestions ?? []).compactMap(cleanLine).prefix(3)
        // A time to leave: for an event the request listed, before it starts
        // and not hours ahead, and still to come.
        var leaving = Set<String>()
        let departures: [Departure] = (decoded.leave ?? []).compactMap { entry in
            guard entry.event.hasPrefix("e"), let number = Int(entry.event.dropFirst()),
                number >= 1, number <= eventIDs.count,
                let event = facts.events.first(where: { $0.id == eventIDs[number - 1] }),
                !leaving.contains(event.id),
                let at = AgendaTime.parse(entry.at, now: facts.now, calendar: facts.calendar),
                at.hasTime, at.date < event.start, at.date > facts.now,
                event.start.timeIntervalSince(at.date) <= longestTravel
            else { return nil }
            leaving.insert(event.id)
            return Departure(
                eventID: event.id, title: event.title, at: at.date, eventStart: event.start,
                location: event.location)
        }
        return .card(
            .morningPlan(
                MorningPlanCard(
                    line: line, mustDoID: mustDo,
                    placements: placements.sorted { $0.start < $1.start },
                    suggestions: Array(suggestions), departures: departures)))
    }

    // MARK: Evening Wrap-up

    private struct WrapUpReply: Decodable {
        struct Entry: Decodable {
            let id: String
            let suggest: String?
        }
        let line: String?
        let leftovers: [Entry]?
    }

    static func eveningWrapUp(
        _ reply: String, facts: DayFacts, leftovers: [AgendaReminder]
    ) -> CardParse {
        guard let data = jsonObject(in: reply) else { return .invalid("no JSON object") }
        guard let decoded = try? JSONDecoder().decode(WrapUpReply.self, from: data) else {
            return .invalid("the JSON is not an Evening Wrap-up")
        }
        guard let line = cleanLine(decoded.line) else { return .invalid("no line") }
        let suggested = Dictionary(
            (decoded.leftovers ?? []).map { ($0.id, $0.suggest ?? "") },
            uniquingKeysWith: { first, _ in first })
        var card = FallbackCards.eveningWrapUp(facts: facts, leftovers: leftovers)
        card.line = line
        card.leftovers = card.leftovers.map { leftover in
            var leftover = leftover
            if let raw = suggested[leftover.reminderID],
                let suggestion = Leftover.Suggestion(rawValue: raw.lowercased())
            {
                leftover.suggestion = suggestion
            }
            return leftover
        }
        return .card(.eveningWrapUp(card))
    }

    // MARK: Night Reflection

    private struct ReflectionReply: Decodable {
        struct Proposal: Decodable {
            let text: String?
            let reason: String?
        }
        let carryOver: String?
        let tomorrow: [String]?
        let proposals: [Proposal]?

        enum CodingKeys: String, CodingKey {
            case tomorrow, proposals
            case carryOver = "carry_over"
        }
    }

    static func nightReflection(_ reply: String) -> CardParse {
        guard let data = jsonObject(in: reply) else { return .invalid("no JSON object") }
        guard let decoded = try? JSONDecoder().decode(ReflectionReply.self, from: data),
            let note = decoded.carryOver?.trimmingCharacters(in: .whitespacesAndNewlines),
            !note.isEmpty
        else { return .invalid("no carry-over note") }
        let proposals = (decoded.proposals ?? []).compactMap { proposal -> ProposalDraft? in
            guard let text = cleanLine(proposal.text) else { return nil }
            return ProposalDraft(text: text, reason: cleanLine(proposal.reason) ?? "")
        }
        return .card(
            .reflection(
                ReflectionCard(
                    carryOver: String(note.prefix(1200)),
                    tomorrow: (decoded.tomorrow ?? []).compactMap(cleanLine).prefix(5).map { $0 },
                    proposals: Array(proposals.prefix(3)))))
    }

    // MARK: Shared

    /// A usable one-liner: trimmed, non-empty, a sentence rather than an essay.
    static func cleanLine(_ raw: String?) -> String? {
        guard let raw else { return nil }
        let line = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !line.isEmpty else { return nil }
        return line.count > 320 ? String(line.prefix(317)) + "…" : line
    }
}

// MARK: - Fallbacks

/// Cards built by code alone, for when the model call fails twice. Warm,
/// plain, and never empty of the essential facts.
nonisolated enum FallbackCards {

    /// The plan code can make alone, shown at once while Jarvis thinks and
    /// kept if he can't: the day's shape and where it starts.
    static func morningPlan(facts: DayFacts) -> MorningPlanCard {
        let remaining = facts.remainingEventsToday
        let line: String
        if let first = remaining.first {
            let count = remaining.count
            let clock = AgendaTime.clock(first.start, calendar: facts.calendar)
            let lead =
                first.start <= facts.now ? "now \(first.title)" : "first \(first.title) at \(clock)"
            line = "Here's your day: \(count) event\(count == 1 ? "" : "s") ahead — \(lead)."
        } else {
            line = "Here's your day. Nothing on the calendar — it's yours to shape."
        }
        return MorningPlanCard(
            line: line, mustDoID: facts.mustDoID, placements: [], suggestions: [])
    }

    static func nightReflection(facts: DayFacts) -> ReflectionCard {
        var parts: [String] = []
        if !facts.doneToday.isEmpty {
            parts.append(
                "Yesterday you finished " + facts.doneToday.map(\.title).joined(separator: ", ")
                    + ".")
        }
        if let first = facts.tomorrowEvents.first {
            parts.append(
                "Today starts with \(first.title) at \(AgendaTime.clock(first.start, calendar: facts.calendar))."
            )
        }
        return ReflectionCard(
            carryOver: parts.isEmpty ? "A new day." : parts.joined(separator: " "), tomorrow: [],
            proposals: [])
    }

    static func eveningWrapUp(facts: DayFacts, leftovers: [AgendaReminder]) -> EveningWrapUpCard {
        let done = facts.doneToday.map(\.title)
        let line =
            done.isEmpty
            ? "The day's done. Tomorrow's a fresh start."
            : "You got \(done.count) thing\(done.count == 1 ? "" : "s") done today."
        let first = facts.tomorrowEvents.first.map {
            "\(AgendaTime.clock($0.start, calendar: facts.calendar)) \($0.title)"
        }
        return EveningWrapUpCard(
            line: line, done: done,
            leftovers: leftovers.map {
                Leftover(reminderID: $0.id, title: $0.title, suggestion: .tomorrow)
            },
            tomorrowFirst: first)
    }
}
