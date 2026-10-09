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

    init(
        id: String, title: String, start: Date, end: Date, isAllDay: Bool = false,
        calendarID: String, calendarTitle: String, colorHex: String? = nil,
        location: String? = nil, notes: String? = nil, hasOtherAttendees: Bool = false,
        isEditable: Bool = true
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
    }

    var duration: TimeInterval { end.timeIntervalSince(start) }

    /// Where it happens, as the owner reads it: a meeting link becomes its
    /// service ("Zoom"), so "online" is plain and a password never shows; a
    /// place stays as written.
    var place: String? { AgendaPlace.label(location) }
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

    /// - Parameter withLink: keep the meeting link, its query (a password)
    ///   dropped, after the service ("Zoom — us04web.zoom.us/j/123").
    static func label(_ location: String?, withLink: Bool = false) -> String? {
        guard let location = location?.trimmingCharacters(in: .whitespacesAndNewlines),
            !location.isEmpty
        else { return nil }
        guard let range = location.range(of: #"https?://[^\s,;]+"#, options: .regularExpression)
        else { return location }
        let url = URL(string: String(location[range]))
        let host = (url?.host ?? "").lowercased()
        var service =
            services.first { host == $0.host || host.hasSuffix("." + $0.host) }?.name
            ?? (host.hasPrefix("www.") ? String(host.dropFirst(4)) : host)
        if withLink, let url {
            service += " — \(url.host ?? "")\(url.path)"
        }
        let rest = location.replacingCharacters(in: range, with: "")
            .trimmingCharacters(
                in: CharacterSet(charactersIn: " -–—/|,;:").union(.whitespacesAndNewlines))
        return rest.isEmpty ? service : "\(rest) · \(service)"
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

    init(
        id: String, title: String, notes: String? = nil, listID: String, listTitle: String,
        colorHex: String? = nil, due: Date? = nil, dueHasTime: Bool = false,
        isCompleted: Bool = false, completedAt: Date? = nil, createdAt: Date? = nil
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
