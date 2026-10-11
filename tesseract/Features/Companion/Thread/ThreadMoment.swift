//
//  ThreadMoment.swift
//  tesseract
//
//  A moment's turn in the Day Thread, read back for Today's Chat: which
//  moment it was, when, and what Jarvis said. The request's header line
//  names the moment ("[Evening Wrap-up]", "[Welcome back — away 45 min,
//  13:20–14:05]"); the card in his reply holds his line. The card the
//  moment made lives on in the day's state, and `card(in:turns:)` finds it,
//  so the chat can show the card itself. The Day Opening reads back into
//  what Jarvis started the day with (`DayOpeningDigest`). Pure: messages in,
//  values out.
//

import Foundation

nonisolated struct ThreadMoment: Sendable, Equatable, Identifiable {

    enum Kind: Sendable, Equatable {
        /// The thread's first message: what Jarvis started the day with.
        case dayOpening
        case moment(MomentKind)
        /// "That wasn't the card": the retry after a reply with no card in
        /// it. The moment's own turn shows the card it ends with.
        case retry
        /// A header this build doesn't name.
        case other(String)
    }

    /// The request's id, which the transcript row keeps.
    let id: UUID
    var kind: Kind
    /// When the request joined the thread.
    var at: Date
    /// What the header says after the moment's name: an opening's day
    /// ("Sunday 4 October"), a welcome back's time away.
    var detail: String?
    /// Jarvis's line, from the card in his reply.
    var line: String?

    init(request: UserMessage, reply: AssistantMessage?) {
        id = request.id
        at = request.timestamp
        (kind, detail) = Self.header(request.content)
        line = reply.flatMap { Self.line(in: $0.text) }
    }

    /// The moment a request's header names, and what follows the name.
    /// Only a retry's request has no header.
    static func header(_ request: String) -> (kind: Kind, detail: String?) {
        guard request.hasPrefix("["), let close = request.firstIndex(of: "]") else {
            return (.retry, nil)
        }
        let inside = request[request.index(after: request.startIndex)..<close]
        let name: String
        var detail: String?
        if let dash = inside.range(of: " — ") {
            name = String(inside[..<dash.lowerBound])
            detail = String(inside[dash.upperBound...])
        } else {
            name = String(inside)
        }
        if name == Self.dayOpeningName { return (.dayOpening, detail) }
        if let kind = MomentKind.allCases.first(where: { $0.requestName == name }) {
            return (.moment(kind), detail)
        }
        return (.other(name), detail)
    }

    static let dayOpeningName = "Day Opening"

    /// The card's one line, or a reflection's carry-over note.
    static func line(in reply: String) -> String? {
        guard let data = CardParser.jsonObject(in: reply),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let raw = object["line"] as? String ?? object["carry_over"] as? String
        else { return nil }
        let line = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        return line.isEmpty ? nil : line
    }

    // MARK: - Its card

    /// How long before its request a moment's card may go up: the code
    /// card of a Morning Plan or a Breakpoint goes up a beat before the
    /// request joins the thread.
    static let cardLead: TimeInterval = 2 * 60

    /// How long after its request a moment's card may come: two tries, each
    /// given up on after a quarter of an hour, then the card code builds.
    static let cardWindow: TimeInterval = 40 * 60

    /// The card this turn made, from the day's cards: the first of its kind
    /// from its request on, before the next request of that kind. Nil for an
    /// opening, a retry, or a moment that made no card (a Triage that raised
    /// nothing).
    func card(in cards: [DayCard], turns: [ThreadMoment]) -> DayCard? {
        guard case .moment(let kind) = kind else { return nil }
        let next = turns.filter { $0.kind == self.kind && $0.at > at }.map(\.at).min()
        let until = min(
            at.addingTimeInterval(Self.cardWindow),
            next.map { $0.addingTimeInterval(-Self.cardLead) } ?? .distantFuture)
        let from = at.addingTimeInterval(-Self.cardLead)
        return cards.first { $0.kind == kind && $0.createdAt >= from && $0.createdAt < until }
    }
}

extension MomentKind {
    /// The name its request's header line opens with, read back by the
    /// Today chat (`ThreadMoment.header`).
    nonisolated var requestName: String {
        switch self {
        case .morningPlan: "Morning Plan"
        case .breakpoint: "Welcome back"
        case .triage: "Triage"
        case .eveningWrapUp: "Evening Wrap-up"
        case .nightReflection: "Night Reflection"
        }
    }
}

// MARK: - Day Opening

/// What the Day Opening gave Jarvis, read back for the owner: last night's
/// note and first draft of the day, the week's focus, the day's calendar
/// and tasks as it began, and the Profile and Areas. Reads the text
/// `MomentPrompts.dayOpening` writes; a line it doesn't know is skipped.
nonisolated struct DayOpeningDigest: Sendable, Equatable {
    /// "Sunday 4 October".
    var day: String?
    var carryOver: String?
    var draft: [String] = []
    var profile: [String] = []
    var areas: [String] = []
    var weekFocus: String?
    /// "09:30–10:00 Standup".
    var calendar: [String] = []
    var allDay: [String] = []
    /// Titles of the tasks due that day or earlier.
    var due: [String] = []
    var doneToday: [String] = []

    private enum Section {
        case profile, carryOver, draft, calendar, due, undated
    }

    init(_ opening: String) {
        var section: Section?
        for line in opening.components(separatedBy: "\n") {
            if line.hasPrefix("[") {
                let header = ThreadMoment.header(line)
                if header.kind == .dayOpening { day = header.detail }
                continue
            }
            if line.trimmingCharacters(in: .whitespaces).isEmpty {
                section = nil
                continue
            }
            if line.hasPrefix("- ") {
                let item = String(line.dropFirst(2))
                switch section {
                case .profile: profile.append(item)
                case .draft: draft.append(item)
                case .calendar: calendar.append(item)
                case .due: due.append(Self.taskTitle(item))
                case .carryOver, .undated, nil: break
                }
                continue
            }
            if let rest = line.afterPrefix("Carried over from last night: ") {
                carryOver = rest
                section = .carryOver
            } else if section == .carryOver, let note = carryOver {
                // A note of several paragraphs runs on to the blank line.
                carryOver = note + "\n" + line
            } else if line.hasPrefix("What the owner has told you about themselves") {
                section = .profile
            } else if line.hasPrefix("Last night's first draft of today") {
                section = .draft
            } else if let rest = line.afterPrefix("Areas: ") {
                areas = rest.components(separatedBy: ", ")
                section = nil
            } else if let rest = line.afterPrefix("This week's focus: ") {
                weekFocus = rest
                section = nil
            } else if line.hasPrefix("Today's calendar") {
                section = .calendar
            } else if let rest = line.afterPrefix("All day: ") {
                allDay = rest.components(separatedBy: "; ")
            } else if line.hasPrefix("Tasks due today or earlier") {
                section = .due
            } else if line.hasPrefix("Undated tasks") {
                section = .undated
            } else if let rest = line.afterPrefix("Done today: ") {
                doneToday = rest.components(separatedBy: "; ")
                section = nil
            }
        }
    }

    /// "id · Area · today — title" (or without the when): the title.
    private static func taskTitle(_ line: String) -> String {
        guard let dash = line.range(of: " — ") else { return line }
        return String(line[dash.upperBound...])
    }
}

extension String {
    /// The rest after `prefix`, or nil when it doesn't start so.
    fileprivate nonisolated func afterPrefix(_ prefix: String) -> String? {
        hasPrefix(prefix) ? String(dropFirst(prefix.count)) : nil
    }
}
