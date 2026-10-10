//
//  AgendaTypes.swift
//  tesseract
//
//  The Agenda's value types: Apple Reminders and Calendar as Tesseract sees
//  them. Reminders and Calendar are the single source of truth for the owner's
//  tasks and plans; these values are snapshots read through the Agenda port,
//  never a second store.
//

import Foundation

// MARK: - Access

/// Whether Tesseract may read and write each half of the agenda.
nonisolated struct AgendaAccess: Sendable, Equatable {
    enum Level: String, Sendable, Equatable {
        case notDetermined
        case granted
        case denied
    }

    var calendar: Level
    var reminders: Level

    var canUseCalendar: Bool { calendar == .granted }
    var canUseReminders: Bool { reminders == .granted }
    var isFull: Bool { canUseCalendar && canUseReminders }
    var needsRequest: Bool { calendar == .notDetermined || reminders == .notDetermined }

    static let full = AgendaAccess(calendar: .granted, reminders: .granted)
    static let none = AgendaAccess(calendar: .denied, reminders: .denied)
    static let undetermined = AgendaAccess(calendar: .notDetermined, reminders: .notDetermined)
}

// MARK: - Items

/// One occurrence of a calendar event. The id names the occurrence, so a
/// move touches only this one, never the whole series.
nonisolated struct AgendaEvent: Sendable, Equatable, Hashable, Identifiable, Codable {
    let id: String
    var title: String
    var start: Date
    var end: Date
    var isAllDay: Bool
    var calendarID: String
    var calendarTitle: String
    var colorHex: String?
    var location: String?
    var notes: String?
    /// Whether anyone besides the owner is invited: a meeting, not a block.
    var hasOtherAttendees: Bool
    var isEditable: Bool
    /// The event's own link (a calendar invite's conference link may be here).
    var url: URL?

    init(
        id: String, title: String, start: Date, end: Date, isAllDay: Bool = false,
        calendarID: String, calendarTitle: String, colorHex: String? = nil,
        location: String? = nil, notes: String? = nil, hasOtherAttendees: Bool = false,
        isEditable: Bool = true, url: URL? = nil
    ) {
        self.id = id
        self.title = title
        self.start = start
        self.end = end
        self.isAllDay = isAllDay
        self.calendarID = calendarID
        self.calendarTitle = calendarTitle
        self.colorHex = colorHex
        self.location = location
        self.notes = notes
        self.hasOtherAttendees = hasOtherAttendees
        self.isEditable = isEditable
        self.url = url
    }

    var duration: TimeInterval { end.timeIntervalSince(start) }

    /// Where it happens, as the owner reads it: a meeting link becomes its
    /// service ("Zoom"), so "online" is plain and a password never shows; a
    /// place stays as written.
    var place: String? { AgendaPlace.label(location) }

    /// The call to join: the first meeting-service link in its place, its own
    /// link or its notes (where an invite writes "Join with Google Meet").
    var meetingLink: URL? { AgendaPlace.meetingLink(in: [location, url?.absoluteString, notes]) }

    /// Its calendar as Today names it beside the event. A calendar named by
    /// its account's address, as a Google account's own calendar is, reads
    /// as the account's domain ("acme.io", "gmail.com"): the owner's
    /// address is on all of them, the domain tells the accounts apart.
    var calendarLabel: String {
        let title = calendarTitle.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let match = title.wholeMatch(of: /[^@\s]+@([^@\s]+\.[^@\s.]+)/) else {
            return calendarTitle
        }
        return match.output.1.lowercased()
    }
}

/// A calendar location as people read it.
nonisolated enum AgendaPlace {

    /// Meeting services by host.
    static let services: [(host: String, name: String)] = [
        ("zoom.us", "Zoom"), ("meet.google.com", "Google Meet"),
        ("teams.microsoft.com", "Microsoft Teams"), ("teams.live.com", "Microsoft Teams"),
        ("webex.com", "Webex"), ("whereby.com", "Whereby"), ("meet.jit.si", "Jitsi"),
        ("facetime.apple.com", "FaceTime"), ("app.slack.com", "Slack"),
        ("discord.com", "Discord"), ("discord.gg", "Discord"),
    ]

    /// The first link to a meeting service in `texts`, in order: a call's to
    /// join. Other links (a doc, a map) aren't one.
    static func meetingLink(in texts: [String?]) -> URL? {
        for text in texts.compactMap(\.self) {
            for match in text.matches(of: /(?i)https?:\/\/[^\s,;<>"]+/) {
                guard let url = URL(string: String(match.output)),
                    let host = url.host?.lowercased(),
                    services.contains(where: { host == $0.host || host.hasSuffix("." + $0.host) })
                else { continue }
                return url
            }
        }
        return nil
    }

    /// - Parameter withLink: keep the meeting link, its query (a password)
    ///   dropped, after the service ("Zoom — us04web.zoom.us/j/123").
    static func label(_ location: String?, withLink: Bool = false) -> String? {
        guard let location = location?.trimmingCharacters(in: .whitespacesAndNewlines),
            !location.isEmpty
        else { return nil }
        // Every link goes, the first names the service; a passcode written
        // beside it goes too.
        var rest = location
        var service: (name: String, link: String)?
        while let range = rest.range(
            of: #"(?i)https?://[^\s,;]+"#, options: .regularExpression)
        {
            if service == nil, let url = URL(string: String(rest[range])),
                let host = url.host?.lowercased(), !host.isEmpty
            {
                let name =
                    services.first { host == $0.host || host.hasSuffix("." + $0.host) }?.name
                    ?? (host.hasPrefix("www.") ? String(host.dropFirst(4)) : host)
                service = (name, "\(host)\(url.path)")
            }
            rest.replaceSubrange(range, with: " ")
        }
        // "Passcode: 1234", "pwd=abc", "PIN 4455" — not "Pin Oak Park".
        rest = rest.replacingOccurrences(
            of: #"(?i)\b(pass ?code|password|pwd|pin)\b(\s*[:=#]\s*|\s+(?=\d))\S+"#,
            with: " ", options: .regularExpression)
        rest = rest.replacingOccurrences(of: #"\s{2,}"#, with: " ", options: .regularExpression)
            .trimmingCharacters(
                in: CharacterSet(charactersIn: " -–—/|,;:").union(.whitespacesAndNewlines))
        guard let service else { return rest.isEmpty ? nil : rest }
        let named = withLink ? "\(service.name) — \(service.link)" : service.name
        if rest.isEmpty { return named }
        // "Zoom Meeting" already says Zoom.
        if !withLink, rest.lowercased().contains(service.name.lowercased()) { return rest }
        return "\(rest) · \(named)"
    }
}

/// One reminder. `due` is a whole day when `dueHasTime` is false.
nonisolated struct AgendaReminder: Sendable, Equatable, Hashable, Identifiable, Codable {
    let id: String
    var title: String
    var notes: String?
    var listID: String
    var listTitle: String
    var colorHex: String?
    var due: Date?
    var dueHasTime: Bool
    var isCompleted: Bool
    var completedAt: Date?
    var createdAt: Date?
    /// A repeating reminder: one item for the whole series, so deleting it
    /// deletes them all, and it can't lose its date.
    var repeats: Bool

    init(
        id: String, title: String, notes: String? = nil, listID: String, listTitle: String,
        colorHex: String? = nil, due: Date? = nil, dueHasTime: Bool = false,
        isCompleted: Bool = false, completedAt: Date? = nil, createdAt: Date? = nil,
        repeats: Bool = false
    ) {
        self.id = id
        self.title = title
        self.notes = notes
        self.listID = listID
        self.listTitle = listTitle
        self.colorHex = colorHex
        self.due = due
        self.dueHasTime = dueHasTime
        self.isCompleted = isCompleted
        self.completedAt = completedAt
        self.createdAt = createdAt
        self.repeats = repeats
    }
}

/// A Reminders list. Lists are how the owner's Areas are stored.
nonisolated struct AgendaList: Sendable, Equatable, Hashable, Identifiable, Codable {
    let id: String
    var title: String
    var colorHex: String?
    var isDefault: Bool
}

/// A calendar events can be written to.
nonisolated struct AgendaCalendar: Sendable, Equatable, Hashable, Identifiable, Codable {
    let id: String
    var title: String
    var colorHex: String?
    var isWritable: Bool
    var isDefault: Bool
}

// MARK: - Writes

nonisolated struct ReminderDraft: Sendable, Equatable {
    var title: String
    /// nil files it in the default list (the Inbox).
    var listID: String?
    var due: Date?
    var dueHasTime: Bool
    var notes: String?

    init(
        title: String, listID: String? = nil, due: Date? = nil, dueHasTime: Bool = false,
        notes: String? = nil
    ) {
        self.title = title
        self.listID = listID
        self.due = due
        self.dueHasTime = dueHasTime
        self.notes = notes
    }
}

nonisolated struct ReminderChange: Sendable, Equatable {
    enum Due: Sendable, Equatable {
        case set(Date, hasTime: Bool)
        case clear
    }

    var title: String?
    var listID: String?
    var due: Due?
    var notes: String?
    var completed: Bool?

    init(
        title: String? = nil, listID: String? = nil, due: Due? = nil, notes: String? = nil,
        completed: Bool? = nil
    ) {
        self.title = title
        self.listID = listID
        self.due = due
        self.notes = notes
        self.completed = completed
    }

    var isEmpty: Bool {
        title == nil && listID == nil && due == nil && notes == nil && completed == nil
    }
}

nonisolated struct EventDraft: Sendable, Equatable {
    var title: String
    var start: Date
    var end: Date
    var isAllDay: Bool
    /// nil writes to the default calendar.
    var calendarID: String?
    var location: String?
    var notes: String?

    init(
        title: String, start: Date, end: Date, isAllDay: Bool = false, calendarID: String? = nil,
        location: String? = nil, notes: String? = nil
    ) {
        self.title = title
        self.start = start
        self.end = end
        self.isAllDay = isAllDay
        self.calendarID = calendarID
        self.location = location
        self.notes = notes
    }
}

nonisolated struct EventChange: Sendable, Equatable {
    var title: String?
    var start: Date?
    var end: Date?

    init(title: String? = nil, start: Date? = nil, end: Date? = nil) {
        self.title = title
        self.start = start
        self.end = end
    }
}

// MARK: - Errors

nonisolated enum AgendaError: LocalizedError, Equatable, Sendable {
    case noAccess(String)
    case notFound(String)
    case readOnly(String)
    case invalid(String)
    case failed(String)

    var errorDescription: String? {
        switch self {
        case .noAccess(let what):
            "Tesseract has no access to \(what). The owner can allow it in System Settings → Privacy & Security → \(what)."
        case .notFound(let what): "Not found: \(what)."
        case .readOnly(let what): "\(what) can't be changed."
        case .invalid(let why): why
        case .failed(let why): "The change didn't save: \(why)"
        }
    }
}
