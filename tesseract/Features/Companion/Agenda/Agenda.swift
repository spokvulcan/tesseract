//
//  Agenda.swift
//  tesseract
//
//  The Agenda facade: the one door every surface uses to read and change the
//  owner's Reminders and Calendar — the agent's tools, the capture hotkey,
//  the Today page and the Day Engine's effects. It keeps a snapshot of today
//  for the views, turns every write into a one-line confirmation with an
//  undo, and records every change on the Companion Trace.
//

import Foundation
import Observation

/// What the Agenda knows right now, for the Today page and the Day Engine.
nonisolated struct AgendaSnapshot: Sendable, Equatable {
    var takenAt: Date
    var access: AgendaAccess
    /// Events from the start of today to the end of tomorrow.
    var events: [AgendaEvent]
    /// Every incomplete reminder.
    var open: [AgendaReminder]
    /// Reminders completed today.
    var doneToday: [AgendaReminder]
    var lists: [AgendaList]
    var calendars: [AgendaCalendar]
    /// Reminders completed in the last seven days, today included: the
    /// week's look-back counts them.
    var doneThisWeek: [AgendaReminder] = []

    static let empty = AgendaSnapshot(
        takenAt: .distantPast, access: .undetermined, events: [], open: [], doneToday: [],
        lists: [], calendars: [])
}

/// How to take a change back.
nonisolated enum AgendaUndo: Sendable, Equatable {
    case deleteReminder(id: String)
    case changeReminder(id: String, ReminderChange)
    case restoreReminder(ReminderDraft)
    case deleteEvent(id: String)
    case changeEvent(id: String, EventChange)
    case restoreEvent(EventDraft)
}

/// One change the owner can see and undo.
nonisolated struct AgendaChange: Sendable, Equatable, Identifiable {
    let id: UUID
    let at: Date
    /// The one-line confirmation: "Added “Call the dentist” — tomorrow 10:00, Health."
    let line: String
    let undo: AgendaUndo

    init(id: UUID = UUID(), at: Date, line: String, undo: AgendaUndo) {
        self.id = id
        self.at = at
        self.line = line
        self.undo = undo
    }
}

@Observable @MainActor
final class Agenda {

    @ObservationIgnored let store: any AgendaStore
    @ObservationIgnored private let areaMapJSON: @MainActor () -> String
    @ObservationIgnored private let defaultCalendarID: @MainActor () -> String?
    @ObservationIgnored private let trace: CompanionTrace?
    @ObservationIgnored private let now: @MainActor () -> Date
    @ObservationIgnored private var calendar: Calendar

    private(set) var snapshot: AgendaSnapshot = .empty
    /// The latest change, for the "Added … · Undo" line.
    private(set) var lastChange: AgendaChange?
    /// Listeners for any change, ours or from another device (the runtime
    /// re-plans nudges and refreshes the day through this).
    @ObservationIgnored private var listeners: [@MainActor () -> Void] = []
    @ObservationIgnored private var refreshTask: Task<Void, Never>?

    init(
        store: any AgendaStore,
        areaMapJSON: @escaping @MainActor () -> String = { "{}" },
        defaultCalendarID: @escaping @MainActor () -> String? = { nil },
        trace: CompanionTrace? = nil,
        calendar: Calendar = .current,
        now: @escaping @MainActor () -> Date = Date.init
    ) {
        self.store = store
        self.areaMapJSON = areaMapJSON
        self.defaultCalendarID = defaultCalendarID
        self.trace = trace
        self.calendar = calendar
        self.now = now
        store.onChange = { [weak self] in self?.storeChanged() }
    }

    var access: AgendaAccess { store.access }

    func addListener(_ listener: @escaping @MainActor () -> Void) {
        listeners.append(listener)
    }

    /// Ask for access once, then load. Returns the resulting access.
    @discardableResult
    func requestAccessIfNeeded() async -> AgendaAccess {
        let access = store.access.needsRequest ? await store.requestAccess() : store.access
        await refresh()
        return access
    }

    /// Ask for access if it was never asked, so the first "remind me" can
    /// prompt instead of failing.
    func ensureAccess() async {
        if store.access.needsRequest { await requestAccessIfNeeded() }
    }

    /// Reload the snapshot from the store.
    func refresh() async {
        let now = now()
        // The owner's day rolls over at 04:00 (DayKey): after midnight, today
        // is still the day that is ending, and what is done in the small
        // hours counts for it.
        let day = DayKey(for: now, calendar: calendar)
        let startOfToday = day.date(calendar: calendar) ?? calendar.startOfDay(for: now)
        let endOfTomorrow =
            calendar.date(byAdding: .day, value: 2, to: startOfToday)
            ?? now.addingTimeInterval(172_800)
        let endOfToday = calendar.date(byAdding: .day, value: 1, to: startOfToday) ?? now
        let events = store.events(from: startOfToday, to: endOfTomorrow)
        let open = await store.openReminders()
        // One read for the week; today's are the ones since its start.
        let weekStart = calendar.date(byAdding: .day, value: -6, to: startOfToday) ?? startOfToday
        let doneThisWeek = await store.completedReminders(
            from: weekStart, to: day.end(calendar: calendar) ?? endOfToday)
        let done = doneThisWeek.filter { ($0.completedAt ?? .distantPast) >= startOfToday }
        snapshot = AgendaSnapshot(
            takenAt: now, access: store.access, events: events, open: open, doneToday: done,
            lists: store.reminderLists(), calendars: store.eventCalendars(),
            doneThisWeek: doneThisWeek)
    }

    // MARK: Areas

    var areaMap: AreaMap { AreaMap(json: areaMapJSON()) }

    var areas: [Area] { areaMap.areas(in: store.reminderLists()) }

    var inbox: AgendaList? { areaMap.inbox(in: store.reminderLists()) }

    /// The Area a reminder belongs to, if its list is an Area.
    func area(of reminder: AgendaReminder) -> Area? {
        areaMap.area(forListID: reminder.listID, in: snapshot.lists)
    }

    // MARK: Reading

    func events(from: Date, to: Date) -> [AgendaEvent] { store.events(from: from, to: to) }

    // MARK: Writing

    /// Add a reminder. An unknown Area name is an error that names the real
    /// ones; no Area files it in the Inbox.
    @discardableResult
    func addReminder(
        title: String, areaName: String? = nil, due: Date? = nil, dueHasTime: Bool = false,
        notes: String? = nil, source: String
    ) throws -> (AgendaReminder, AgendaChange) {
        let title = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !title.isEmpty else { throw AgendaError.invalid("A reminder needs a title.") }
        let lists = store.reminderLists()
        var listID = areaMap.inbox(in: lists)?.id
        if let areaName, !areaName.trimmingCharacters(in: .whitespaces).isEmpty {
            guard let area = areaMap.area(named: areaName, in: lists) else {
                throw AgendaError.invalid(
                    "No Area called “\(areaName)”. Areas: \(areaMap.areas(in: lists).map(\.name).joined(separator: ", "))."
                )
            }
            listID = area.id
        }
        let reminder = try store.addReminder(
            ReminderDraft(
                title: title, listID: listID, due: due, dueHasTime: dueHasTime, notes: notes))
        var line = "Added “\(reminder.title)”"
        if let due = reminder.due {
            line +=
                " — \(AgendaTime.describe(due, hasTime: reminder.dueHasTime, now: now(), calendar: calendar))"
        }
        line += ", \(placeName(ofListID: reminder.listID, lists: lists))."
        let change = AgendaChange(at: now(), line: line, undo: .deleteReminder(id: reminder.id))
        record(change, kind: "reminder.added", source: source, id: reminder.id)
        return (reminder, change)
    }

    /// Change a reminder: complete or reopen, rename, re-time, re-file.
    @discardableResult
    func updateReminder(
        id: String, title: String? = nil, areaName: String? = nil, due: ReminderChange.Due? = nil,
        notes: String? = nil, completed: Bool? = nil, source: String
    ) async throws -> (AgendaReminder, AgendaChange) {
        guard let before = await store.reminder(id: id, now: now()) else {
            throw AgendaError.notFound("reminder \(id)")
        }
        let lists = store.reminderLists()
        var listID: String?
        if let areaName {
            guard let area = areaMap.area(named: areaName, in: lists) else {
                throw AgendaError.invalid(
                    "No Area called “\(areaName)”. Areas: \(areaMap.areas(in: lists).map(\.name).joined(separator: ", "))."
                )
            }
            listID = area.id
        }
        let change = ReminderChange(
            title: title, listID: listID, due: due, notes: notes, completed: completed)
        guard !change.isEmpty else { throw AgendaError.invalid("Nothing to change.") }
        let after = try store.updateReminder(id: id, change)

        var inverse = ReminderChange()
        if title != nil { inverse.title = before.title }
        if listID != nil { inverse.listID = before.listID }
        if due != nil {
            inverse.due = before.due.map { .set($0, hasTime: before.dueHasTime) } ?? .clear
        }
        if notes != nil { inverse.notes = before.notes ?? "" }
        if completed != nil { inverse.completed = before.isCompleted }

        let line: String
        if completed == true {
            line = "Done: “\(after.title)”."
        } else if completed == false {
            line = "Reopened “\(after.title)”."
        } else if let due = after.due, due != before.due || after.dueHasTime != before.dueHasTime {
            line =
                "Moved “\(after.title)” to \(AgendaTime.describe(due, hasTime: after.dueHasTime, now: now(), calendar: calendar))."
        } else if case .clear = due {
            line = "“\(after.title)” has no date now."
        } else if listID != nil {
            line = "Moved “\(after.title)” to \(placeName(ofListID: after.listID, lists: lists))."
        } else {
            line = "Updated “\(after.title)”."
        }
        let record = AgendaChange(at: now(), line: line, undo: .changeReminder(id: id, inverse))
        self.record(
            record, kind: completed == true ? "reminder.completed" : "reminder.updated",
            source: source, id: id)
        return (after, record)
    }

    /// Let a reminder go. The undo adds it back as it was.
    @discardableResult
    func deleteReminder(id: String, source: String) async throws -> AgendaChange {
        guard let before = await store.reminder(id: id, now: now()) else {
            throw AgendaError.notFound("reminder \(id)")
        }
        try store.deleteReminder(id: id)
        let change = AgendaChange(
            at: now(), line: "Let go of “\(before.title)”.",
            undo: .restoreReminder(
                ReminderDraft(
                    title: before.title, listID: before.listID, due: before.due,
                    dueHasTime: before.dueHasTime, notes: before.notes)))
        record(change, kind: "reminder.deleted", source: source, id: id)
        return change
    }

    /// Add a calendar event to the default calendar (or the named one).
    @discardableResult
    func addEvent(
        title: String, start: Date, end: Date, isAllDay: Bool = false,
        calendarName: String? = nil, location: String? = nil, notes: String? = nil,
        source: String
    ) throws -> (AgendaEvent, AgendaChange) {
        let title = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !title.isEmpty else { throw AgendaError.invalid("An event needs a title.") }
        var calendarID = defaultCalendarID()
        if let calendarName, !calendarName.isEmpty {
            let wanted = calendarName.lowercased()
            guard
                let match = store.eventCalendars().first(where: {
                    $0.isWritable && $0.title.lowercased() == wanted
                })
            else {
                throw AgendaError.invalid(
                    "No writable calendar called “\(calendarName)”. Calendars: \(store.eventCalendars().filter(\.isWritable).map(\.title).joined(separator: ", "))."
                )
            }
            calendarID = match.id
        }
        let event = try store.addEvent(
            EventDraft(
                title: title, start: start, end: end, isAllDay: isAllDay, calendarID: calendarID,
                location: location, notes: notes))
        let when =
            isAllDay
            ? AgendaTime.describe(start, hasTime: false, now: now(), calendar: calendar)
            : "\(AgendaTime.describe(start, hasTime: true, now: now(), calendar: calendar))–\(AgendaTime.clock(end, calendar: calendar))"
        let change = AgendaChange(
            at: now(), line: "Added “\(event.title)” — \(when), \(event.calendarTitle).",
            undo: .deleteEvent(id: event.id))
        record(change, kind: "event.added", source: source, id: event.id)
        return (event, change)
    }

    /// Move one occurrence of an event, keeping its length unless an end is given.
    @discardableResult
    func moveEvent(
        id: String, start: Date, end: Date? = nil, source: String
    ) throws -> (AgendaEvent, AgendaChange) {
        let window = store.events(
            from: now().addingTimeInterval(-30 * 86_400), to: now().addingTimeInterval(365 * 86_400)
        )
        guard let before = window.first(where: { $0.id == id }) else {
            throw AgendaError.notFound("event \(id)")
        }
        let newEnd = end ?? start.addingTimeInterval(before.duration)
        let after = try store.updateEvent(id: id, EventChange(start: start, end: newEnd))
        let change = AgendaChange(
            at: now(),
            line:
                "Moved “\(after.title)” to \(AgendaTime.describe(start, hasTime: true, now: now(), calendar: calendar))–\(AgendaTime.clock(newEnd, calendar: calendar)).",
            undo: .changeEvent(id: after.id, EventChange(start: before.start, end: before.end)))
        record(change, kind: "event.moved", source: source, id: after.id)
        return (after, change)
    }

    /// Delete one occurrence of an event the owner keeps for themselves — a
    /// block, a slot, an appointment. A meeting with other people invited is
    /// refused: declining or cancelling it belongs in Calendar, where they
    /// are told. The undo puts it back as it was.
    @discardableResult
    func deleteEvent(id: String, source: String) throws -> AgendaChange {
        let window = store.events(
            from: now().addingTimeInterval(-30 * 86_400), to: now().addingTimeInterval(365 * 86_400)
        )
        guard let before = window.first(where: { $0.id == id }) else {
            throw AgendaError.notFound("event \(id)")
        }
        guard !before.hasOtherAttendees else {
            throw AgendaError.invalid(
                "“\(before.title)” has other people invited. Decline or cancel it in Calendar so they're told."
            )
        }
        guard before.isEditable else { throw AgendaError.readOnly("“\(before.title)”") }
        try store.deleteEvent(id: id)
        let when =
            before.isAllDay
            ? AgendaTime.describe(before.start, hasTime: false, now: now(), calendar: calendar)
            : "\(AgendaTime.describe(before.start, hasTime: true, now: now(), calendar: calendar))–\(AgendaTime.clock(before.end, calendar: calendar))"
        let change = AgendaChange(
            at: now(), line: "Deleted “\(before.title)” — \(when), \(before.calendarTitle).",
            undo: .restoreEvent(
                EventDraft(
                    title: before.title, start: before.start, end: before.end,
                    isAllDay: before.isAllDay, calendarID: before.calendarID,
                    location: before.location, notes: before.notes)))
        record(change, kind: "event.deleted", source: source, id: id)
        return change
    }

    /// Take a change back.
    func undo(_ change: AgendaChange) async throws {
        switch change.undo {
        case .deleteReminder(let id): try store.deleteReminder(id: id)
        case .changeReminder(let id, let inverse): _ = try store.updateReminder(id: id, inverse)
        case .restoreReminder(let draft): _ = try store.addReminder(draft)
        case .deleteEvent(let id): try store.deleteEvent(id: id)
        case .changeEvent(let id, let inverse): _ = try store.updateEvent(id: id, inverse)
        case .restoreEvent(let draft): _ = try store.addEvent(draft)
        }
        if lastChange?.id == change.id { lastChange = nil }
        trace?.record(.agendaChanged, fields: ["kind": "undo", "line": .string(change.line)])
        await refresh()
    }

    /// Forget the confirmation line (the owner dismissed it).
    func clearLastChange() { lastChange = nil }

    // MARK: Private

    private func placeName(ofListID listID: String, lists: [AgendaList]) -> String {
        if let area = areaMap.area(forListID: listID, in: lists) {
            return area.id == areaMap.inbox(in: lists)?.id ? "Inbox" : area.name
        }
        return lists.first { $0.id == listID }?.title ?? "Reminders"
    }

    private func record(_ change: AgendaChange, kind: String, source: String, id: String) {
        lastChange = change
        trace?.record(
            .agendaChanged,
            fields: ["kind": .string(kind), "source": .string(source), "id": .string(id)])
    }

    private func storeChanged() {
        refreshTask?.cancel()
        refreshTask = Task { [weak self] in
            guard let self else { return }
            await self.refresh()
            guard !Task.isCancelled else { return }
            for listener in self.listeners { listener() }
        }
    }
}
