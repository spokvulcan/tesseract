//
//  Cards.swift
//  tesseract
//
//  What a moment shows the owner. The model decides the content and the
//  importance of each item; code decides the delivery rung and builds every
//  fact it can know without the model (the next event, what is done, how
//  many things can wait), so a fallback card is never empty.
//

import Foundation

/// One card, whichever moment made it.
nonisolated struct DayCard: Sendable, Equatable, Codable, Identifiable {
    let id: String
    var kind: MomentKind
    var createdAt: Date
    /// Built by code alone because the model call failed.
    var isFallback: Bool
    var body: Body
    /// The owner dismissed it (the close button, or acted on everything).
    var dismissed: Bool = false
    /// Built by code and shown at once; Jarvis is still thinking it through,
    /// and his version will replace it in place.
    var isRefining: Bool = false

    init(
        id: String, kind: MomentKind, createdAt: Date, isFallback: Bool, body: Body,
        dismissed: Bool = false, isRefining: Bool = false
    ) {
        self.id = id
        self.kind = kind
        self.createdAt = createdAt
        self.isFallback = isFallback
        self.body = body
        self.dismissed = dismissed
        self.isRefining = isRefining
    }

    private enum CodingKeys: String, CodingKey {
        case id, kind, createdAt, isFallback, body, dismissed, isRefining
    }

    /// A card saved by an earlier build (no `isRefining`) still loads.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        kind = try c.decode(MomentKind.self, forKey: .kind)
        createdAt = try c.decode(Date.self, forKey: .createdAt)
        isFallback = try c.decode(Bool.self, forKey: .isFallback)
        body = try c.decode(Body.self, forKey: .body)
        dismissed = try c.decodeIfPresent(Bool.self, forKey: .dismissed) ?? false
        isRefining = try c.decodeIfPresent(Bool.self, forKey: .isRefining) ?? false
    }

    enum Body: Sendable, Equatable, Codable {
        case morningPlan(MorningPlanCard)
        case eveningWrapUp(EveningWrapUpCard)
        case breakpoint(BreakpointCard)
        case triage(TriageCard)
        case reflection(ReflectionCard)
    }

    /// The card's one warm sentence.
    var line: String {
        switch body {
        case .morningPlan(let card): card.line
        case .eveningWrapUp(let card): card.line
        case .breakpoint(let card): card.line
        case .triage(let card): card.line
        case .reflection(let card): card.carryOver
        }
    }
}

// MARK: - Morning Plan

nonisolated struct MorningPlanCard: Sendable, Equatable, Codable {
    var line: String
    /// The day's one optional must-do, a reminder id.
    var mustDoID: String?
    /// Tasks placed into the day's free time.
    var placements: [Placement]
    /// At most three small suggestions ("start with the two quick replies").
    var suggestions: [String]
    /// When to leave for the day's events in person.
    var departures: [Departure] = []

    private enum CodingKeys: String, CodingKey {
        case line, mustDoID, placements, suggestions, departures
    }

    init(
        line: String, mustDoID: String?, placements: [Placement], suggestions: [String],
        departures: [Departure] = []
    ) {
        self.line = line
        self.mustDoID = mustDoID
        self.placements = placements
        self.suggestions = suggestions
        self.departures = departures
    }

    /// A card saved before departures existed still loads.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        line = try c.decode(String.self, forKey: .line)
        mustDoID = try c.decodeIfPresent(String.self, forKey: .mustDoID)
        placements = try c.decode([Placement].self, forKey: .placements)
        suggestions = try c.decode([String].self, forKey: .suggestions)
        departures = (try? c.decodeIfPresent([Departure].self, forKey: .departures)) ?? []
    }
}

/// When the owner has to leave for an event in person, from the Morning Plan
/// (Jarvis knows the bus takes half an hour; the calendar doesn't). The OS
/// nudges at that time, and the Now Card says it as the event comes up.
nonisolated struct Departure: Sendable, Equatable, Hashable, Codable {
    var eventID: String
    var title: String
    var at: Date
    var eventStart: Date
    var location: String?
}

/// A reminder given a slot in today's plan. Tesseract-side only: the
/// reminder itself keeps its own date.
nonisolated struct Placement: Sendable, Equatable, Hashable, Codable {
    var reminderID: String
    var start: Date
    var minutes: Int
}

// MARK: - Evening Wrap-up

nonisolated struct EveningWrapUpCard: Sendable, Equatable, Codable {
    var line: String
    /// What got done today, titles in order.
    var done: [String]
    var leftovers: [Leftover]
    /// Tomorrow's first commitment ("09:30 Standup"), when there is one.
    var tomorrowFirst: String?
    /// On the week's last day: a line on the week…
    var week: String?
    /// …and next week's one focus.
    var focus: String?

    private enum CodingKeys: String, CodingKey {
        case line, done, leftovers, tomorrowFirst, week, focus
    }

    init(
        line: String, done: [String], leftovers: [Leftover], tomorrowFirst: String?,
        week: String? = nil, focus: String? = nil
    ) {
        self.line = line
        self.done = done
        self.leftovers = leftovers
        self.tomorrowFirst = tomorrowFirst
        self.week = week
        self.focus = focus
    }

    /// A card saved before the week's look-back still loads.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        line = try c.decode(String.self, forKey: .line)
        done = try c.decode([String].self, forKey: .done)
        leftovers = try c.decode([Leftover].self, forKey: .leftovers)
        tomorrowFirst = try c.decodeIfPresent(String.self, forKey: .tomorrowFirst)
        week = try c.decodeIfPresent(String.self, forKey: .week)
        focus = try c.decodeIfPresent(String.self, forKey: .focus)
    }
}

nonisolated struct Leftover: Sendable, Equatable, Hashable, Codable, Identifiable {
    enum Suggestion: String, Sendable, Equatable, Codable {
        /// Roll to tomorrow.
        case tomorrow
        /// Keep it, undated, for another day.
        case later
        /// Let it go.
        case drop
    }

    var id: String { reminderID }
    var reminderID: String
    var title: String
    var suggestion: Suggestion
}

// MARK: - Breakpoint

/// The card shown on coming back, validated in the prototype.
nonisolated struct BreakpointCard: Sendable, Equatable, Codable {
    var awayFrom: Date
    var awayUntil: Date
    var line: String
    /// People, agents and missed notifications that need the owner; one
    /// action each.
    var needsYou: [WaitingItem]
    /// The next event, and what fits before it.
    var next: [NextItem]
    /// The task or app open when the owner left.
    var whereYouWere: String?
    /// How many other things can wait, grouped by app.
    var canWait: [QuietGroup]

    var canWaitCount: Int { canWait.reduce(0) { $0 + $1.lines.count } }
}

nonisolated struct WaitingItem: Sendable, Equatable, Hashable, Codable, Identifiable {
    enum Kind: String, Sendable, Codable {
        case notification
        case agent
        case reminder
    }

    let id: String
    var kind: Kind
    var title: String
    var detail: String
    /// The app to bring forward for "Open".
    var app: String?
}

nonisolated struct NextItem: Sendable, Equatable, Hashable, Codable, Identifiable {
    enum Kind: String, Sendable, Codable {
        case event
        case task
    }

    let id: String
    var kind: Kind
    var title: String
    var at: Date?
    var minutes: Int?
}

nonisolated struct QuietGroup: Sendable, Equatable, Hashable, Codable, Identifiable {
    var id: String { app }
    var app: String
    var lines: [String]
}

// MARK: - Triage

nonisolated struct TriageCard: Sendable, Equatable, Codable {
    var line: String
    /// What should reach the owner now, not at the next Breakpoint.
    var raise: [WaitingItem]
}

// MARK: - Night Reflection

nonisolated struct ReflectionCard: Sendable, Equatable, Codable {
    /// The note tomorrow's Day Thread opens with.
    var carryOver: String
    /// A first draft of tomorrow, one line each.
    var tomorrow: [String]
    /// Zero to three "Should I remember this?" proposals.
    var proposals: [ProposalDraft]
}

nonisolated struct ProposalDraft: Sendable, Equatable, Hashable, Codable {
    var text: String
    var reason: String
}
