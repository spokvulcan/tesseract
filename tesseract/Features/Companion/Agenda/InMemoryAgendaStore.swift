//
//  InMemoryAgendaStore.swift
//  tesseract
//
//  A hermetic Agenda: dictionaries, not a mock, and a peer implementation of
//  the EventKit store. The test host's container uses it (ADR-0073), so a
//  test run never asks for or touches the owner's Reminders or Calendar; the
//  tests seed it with fixture days.
//

import Foundation

@MainActor
final class InMemoryAgendaStore: AgendaStore {

    var access: AgendaAccess
    var onChange: (@MainActor () -> Void)?

    private(set) var lists: [AgendaList]
    private(set) var calendars: [AgendaCalendar]
    private(set) var reminders: [String: AgendaReminder] = [:]
    private(set) var events: [String: AgendaEvent] = [:]
    /// Every write in order, for tests that assert what reached the store.
    private(set) var writeLog: [String] = []
    /// The clock completion stamps come from.
    var now: () -> Date

    init(
        access: AgendaAccess = .full,
        lists: [AgendaList] = [AgendaList(id: "inbox", title: "Reminders", isDefault: true)],
        calendars: [AgendaCalendar] = [
            AgendaCalendar(id: "home", title: "Home", isWritable: true, isDefault: true)
        ],
        reminders: [AgendaReminder] = [],
        events: [AgendaEvent] = [],
        now: @escaping () -> Date = Date.init
    ) {
        self.access = access
        self.lists = lists
        self.calendars = calendars
        self.now = now
        for reminder in reminders { self.reminders[reminder.id] = reminder }
        for event in events { self.events[event.id] = event }
    }

    func requestAccess() async -> AgendaAccess { access }

    // MARK: Seeding (tests)

    func seed(reminder: AgendaReminder) {
        reminders[reminder.id] = reminder
        onChange?()
    }

    func seed(event: AgendaEvent) {
        events[event.id] = event
        onChange?()
    }

    // MARK: Reading

    func events(from: Date, to: Date) -> [AgendaEvent] {
        guard access.canUseCalendar else { return [] }
        return events.values
            .filter { $0.start < to && $0.end > from }
            .sorted { ($0.start, $0.title) < ($1.start, $1.title) }
    }

    func openReminders() async -> [AgendaReminder] {
        guard access.canUseReminders else { return [] }
        return reminders.values.filter { !$0.isCompleted }.sorted(by: Self.order)
    }

    func completedReminders(from: Date, to: Date) async -> [AgendaReminder] {
        guard access.canUseReminders else { return [] }
        return reminders.values
            .filter { reminder in
                guard reminder.isCompleted, let done = reminder.completedAt else { return false }
                return done >= from && done < to
            }
            .sorted(by: Self.order)
    }

    func reminderLists() -> [AgendaList] { access.canUseReminders ? lists : [] }

    func eventCalendars() -> [AgendaCalendar] { access.canUseCalendar ? calendars : [] }

    // MARK: Writing

    func addReminder(_ draft: ReminderDraft) throws -> AgendaReminder {
        guard access.canUseReminders else { throw AgendaError.noAccess("Reminders") }
        let list = try list(for: draft.listID)
        let reminder = AgendaReminder(
            id: "R\(reminders.count + 1)-\(UUID().uuidString.prefix(4))",
            title: draft.title, notes: draft.notes, listID: list.id, listTitle: list.title,
            colorHex: list.colorHex, due: draft.due,
            dueHasTime: draft.due != nil && draft.dueHasTime,
            createdAt: now())
        reminders[reminder.id] = reminder
        record("addReminder \(reminder.title)")
        return reminder
    }

    func updateReminder(id: String, _ change: ReminderChange) throws -> AgendaReminder {
        guard access.canUseReminders else { throw AgendaError.noAccess("Reminders") }
        guard var reminder = reminders[id] else { throw AgendaError.notFound("reminder \(id)") }
        if let title = change.title { reminder.title = title }
        if let notes = change.notes { reminder.notes = notes.isEmpty ? nil : notes }
        if let listID = change.listID {
            let list = try list(for: listID)
            reminder.listID = list.id
            reminder.listTitle = list.title
            reminder.colorHex = list.colorHex
        }
        switch change.due {
        case .set(let date, let hasTime):
            reminder.due = date
            reminder.dueHasTime = hasTime
        case .clear:
            reminder.due = nil
            reminder.dueHasTime = false
        case nil:
            break
        }
        if let completed = change.completed {
            reminder.isCompleted = completed
            reminder.completedAt = completed ? now() : nil
        }
        reminders[id] = reminder
        record("updateReminder \(reminder.title)")
        return reminder
    }

    func deleteReminder(id: String) throws {
        guard access.canUseReminders else { throw AgendaError.noAccess("Reminders") }
        guard let removed = reminders.removeValue(forKey: id) else {
            throw AgendaError.notFound("reminder \(id)")
        }
        record("deleteReminder \(removed.title)")
    }

    func addEvent(_ draft: EventDraft) throws -> AgendaEvent {
        guard access.canUseCalendar else { throw AgendaError.noAccess("Calendar") }
        guard draft.end > draft.start else {
            throw AgendaError.invalid("An event must end after it starts.")
        }
        let calendar = try calendar(for: draft.calendarID)
        let event = AgendaEvent(
            id: "E\(events.count + 1)-\(UUID().uuidString.prefix(4))", title: draft.title,
            start: draft.start, end: draft.end, isAllDay: draft.isAllDay, calendarID: calendar.id,
            calendarTitle: calendar.title, colorHex: calendar.colorHex, location: draft.location,
            notes: draft.notes)
        events[event.id] = event
        record("addEvent \(event.title)")
        return event
    }

    func updateEvent(id: String, _ change: EventChange) throws -> AgendaEvent {
        guard access.canUseCalendar else { throw AgendaError.noAccess("Calendar") }
        guard var event = events[id] else { throw AgendaError.notFound("event \(id)") }
        guard event.isEditable else { throw AgendaError.readOnly("“\(event.title)”") }
        if let title = change.title { event.title = title }
        if let start = change.start { event.start = start }
        if let end = change.end { event.end = end }
        guard event.end > event.start else {
            throw AgendaError.invalid("An event must end after it starts.")
        }
        events[id] = event
        record("updateEvent \(event.title)")
        return event
    }

    func deleteEvent(id: String) throws {
        guard access.canUseCalendar else { throw AgendaError.noAccess("Calendar") }
        guard let removed = events.removeValue(forKey: id) else {
            throw AgendaError.notFound("event \(id)")
        }
        record("deleteEvent \(removed.title)")
    }

    // MARK: Private

    private func record(_ entry: String) {
        writeLog.append(entry)
        onChange?()
    }

    private func list(for id: String?) throws -> AgendaList {
        if let id {
            guard let list = lists.first(where: { $0.id == id }) else {
                throw AgendaError.notFound("list \(id)")
            }
            return list
        }
        guard let list = lists.first(where: \.isDefault) ?? lists.first else {
            throw AgendaError.notFound("a Reminders list")
        }
        return list
    }

    private func calendar(for id: String?) throws -> AgendaCalendar {
        if let id {
            guard let calendar = calendars.first(where: { $0.id == id && $0.isWritable }) else {
                throw AgendaError.notFound("calendar \(id)")
            }
            return calendar
        }
        guard
            let calendar = calendars.first(where: { $0.isDefault && $0.isWritable })
                ?? calendars.first(where: \.isWritable)
        else { throw AgendaError.notFound("a writable calendar") }
        return calendar
    }

    private static func order(_ a: AgendaReminder, _ b: AgendaReminder) -> Bool {
        switch (a.due, b.due) {
        case (let x?, let y?): x == y ? a.title < b.title : x < y
        case (.some, .none): true
        case (.none, .some): false
        case (.none, .none):
            (a.createdAt ?? .distantPast, a.title) < (b.createdAt ?? .distantPast, b.title)
        }
    }
}
