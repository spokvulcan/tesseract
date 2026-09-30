//
//  EventKitAgendaStore.swift
//  tesseract
//
//  The production Agenda: Apple Reminders and Calendar through EventKit.
//  Deliberately thin — reading converts EventKit objects to Agenda values at
//  the boundary, writing is one EventKit save per call — so the logic above
//  it is tested against the in-memory store and this adapter is verified by
//  hand (ADR-0073: no automated test touches the real store).
//
//  A reminder with a due time gets an alarm at that time, so Reminders itself
//  delivers it on the Mac, the phone and the watch, whether or not Tesseract
//  is running.
//

import AppKit
import EventKit
import Foundation

@MainActor
final class EventKitAgendaStore: AgendaStore {

    var onChange: (@MainActor () -> Void)?

    private let store = EKEventStore()
    private var changeObserver: NSObjectProtocol?
    private var changeDebounce: Task<Void, Never>?

    init() {
        changeObserver = NotificationCenter.default.addObserver(
            forName: .EKEventStoreChanged, object: store, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated { self?.storeChanged() }
        }
    }

    // MARK: Access

    var access: AgendaAccess {
        AgendaAccess(
            calendar: Self.level(EKEventStore.authorizationStatus(for: .event)),
            reminders: Self.level(EKEventStore.authorizationStatus(for: .reminder)))
    }

    func requestAccess() async -> AgendaAccess {
        if EKEventStore.authorizationStatus(for: .reminder) == .notDetermined {
            let granted = (try? await store.requestFullAccessToReminders()) ?? false
            Log.companion.info("Reminders access asked: \(granted ? "granted" : "denied")")
        }
        if EKEventStore.authorizationStatus(for: .event) == .notDetermined {
            let granted = (try? await store.requestFullAccessToEvents()) ?? false
            Log.companion.info("Calendar access asked: \(granted ? "granted" : "denied")")
        }
        store.reset()
        return access
    }

    private static func level(_ status: EKAuthorizationStatus) -> AgendaAccess.Level {
        switch status {
        case .fullAccess: .granted
        case .notDetermined: .notDetermined
        default: .denied
        }
    }

    // MARK: Reading

    func events(from: Date, to: Date) -> [AgendaEvent] {
        guard access.canUseCalendar else { return [] }
        let predicate = store.predicateForEvents(withStart: from, end: to, calendars: nil)
        return store.events(matching: predicate)
            .map(Self.value(of:))
            .sorted { ($0.start, $0.title) < ($1.start, $1.title) }
    }

    func openReminders() async -> [AgendaReminder] {
        guard access.canUseReminders else { return [] }
        let predicate = store.predicateForIncompleteReminders(
            withDueDateStarting: nil, ending: nil, calendars: nil)
        return await fetch(predicate)
    }

    func completedReminders(from: Date, to: Date) async -> [AgendaReminder] {
        guard access.canUseReminders else { return [] }
        let predicate = store.predicateForCompletedReminders(
            withCompletionDateStarting: from, ending: to, calendars: nil)
        return await fetch(predicate)
    }

    func reminderLists() -> [AgendaList] {
        guard access.canUseReminders else { return [] }
        let defaultID = store.defaultCalendarForNewReminders()?.calendarIdentifier
        return store.calendars(for: .reminder).map {
            AgendaList(
                id: $0.calendarIdentifier, title: $0.title, colorHex: Self.hex($0.cgColor),
                isDefault: $0.calendarIdentifier == defaultID)
        }
    }

    func eventCalendars() -> [AgendaCalendar] {
        guard access.canUseCalendar else { return [] }
        let defaultID = store.defaultCalendarForNewEvents?.calendarIdentifier
        return store.calendars(for: .event).map {
            AgendaCalendar(
                id: $0.calendarIdentifier, title: $0.title, colorHex: Self.hex($0.cgColor),
                isWritable: $0.allowsContentModifications,
                isDefault: $0.calendarIdentifier == defaultID)
        }
    }

    private func fetch(_ predicate: NSPredicate) async -> [AgendaReminder] {
        await withCheckedContinuation { continuation in
            store.fetchReminders(matching: predicate) { reminders in
                // Converted here, on EventKit's queue: only values cross over.
                let values = (reminders ?? []).map(Self.value(of:))
                continuation.resume(returning: values)
            }
        }
    }

    // MARK: Writing

    func addReminder(_ draft: ReminderDraft) throws -> AgendaReminder {
        guard access.canUseReminders else { throw AgendaError.noAccess("Reminders") }
        let reminder = EKReminder(eventStore: store)
        reminder.title = draft.title
        reminder.notes = draft.notes
        reminder.calendar = try reminderCalendar(draft.listID)
        setDue(of: reminder, to: draft.due, hasTime: draft.dueHasTime)
        try save(reminder)
        return Self.value(of: reminder)
    }

    func updateReminder(id: String, _ change: ReminderChange) throws -> AgendaReminder {
        guard access.canUseReminders else { throw AgendaError.noAccess("Reminders") }
        guard let reminder = store.calendarItem(withIdentifier: id) as? EKReminder else {
            throw AgendaError.notFound("reminder \(id)")
        }
        if let title = change.title { reminder.title = title }
        if let notes = change.notes { reminder.notes = notes.isEmpty ? nil : notes }
        if let listID = change.listID { reminder.calendar = try reminderCalendar(listID) }
        switch change.due {
        case .set(let date, let hasTime): setDue(of: reminder, to: date, hasTime: hasTime)
        case .clear: setDue(of: reminder, to: nil, hasTime: false)
        case nil: break
        }
        if let completed = change.completed { reminder.isCompleted = completed }
        try save(reminder)
        return Self.value(of: reminder)
    }

    func deleteReminder(id: String) throws {
        guard access.canUseReminders else { throw AgendaError.noAccess("Reminders") }
        guard let reminder = store.calendarItem(withIdentifier: id) as? EKReminder else {
            throw AgendaError.notFound("reminder \(id)")
        }
        do { try store.remove(reminder, commit: true) } catch {
            throw AgendaError.failed(error.localizedDescription)
        }
    }

    func addEvent(_ draft: EventDraft) throws -> AgendaEvent {
        guard access.canUseCalendar else { throw AgendaError.noAccess("Calendar") }
        guard draft.end > draft.start else {
            throw AgendaError.invalid("An event must end after it starts.")
        }
        let event = EKEvent(eventStore: store)
        event.title = draft.title
        event.startDate = draft.start
        event.endDate = draft.end
        event.isAllDay = draft.isAllDay
        event.location = draft.location
        event.notes = draft.notes
        event.calendar = try eventCalendar(draft.calendarID)
        do { try store.save(event, span: .thisEvent, commit: true) } catch {
            throw AgendaError.failed(error.localizedDescription)
        }
        return Self.value(of: event)
    }

    func updateEvent(id: String, _ change: EventChange) throws -> AgendaEvent {
        guard access.canUseCalendar else { throw AgendaError.noAccess("Calendar") }
        let event = try occurrence(id)
        guard event.calendar?.allowsContentModifications == true else {
            throw AgendaError.readOnly("“\(event.title ?? "This event")”")
        }
        if let title = change.title { event.title = title }
        if let start = change.start { event.startDate = start }
        if let end = change.end { event.endDate = end }
        guard event.endDate > event.startDate else {
            throw AgendaError.invalid("An event must end after it starts.")
        }
        do { try store.save(event, span: .thisEvent, commit: true) } catch {
            throw AgendaError.failed(error.localizedDescription)
        }
        return Self.value(of: event)
    }

    func deleteEvent(id: String) throws {
        guard access.canUseCalendar else { throw AgendaError.noAccess("Calendar") }
        let event = try occurrence(id)
        do { try store.remove(event, span: .thisEvent, commit: true) } catch {
            throw AgendaError.failed(error.localizedDescription)
        }
    }

    // MARK: Private

    private func storeChanged() {
        // Sync bursts arrive as a volley; one refresh is enough.
        changeDebounce?.cancel()
        changeDebounce = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(300))
            guard !Task.isCancelled else { return }
            self?.onChange?()
        }
    }

    private func save(_ reminder: EKReminder) throws {
        do { try store.save(reminder, commit: true) } catch {
            throw AgendaError.failed(error.localizedDescription)
        }
    }

    private func reminderCalendar(_ id: String?) throws -> EKCalendar {
        if let id {
            guard let calendar = store.calendar(withIdentifier: id) else {
                throw AgendaError.notFound("list \(id)")
            }
            return calendar
        }
        guard let calendar = store.defaultCalendarForNewReminders() else {
            throw AgendaError.notFound("a Reminders list")
        }
        return calendar
    }

    private func eventCalendar(_ id: String?) throws -> EKCalendar {
        if let id {
            guard let calendar = store.calendar(withIdentifier: id),
                calendar.allowsContentModifications
            else { throw AgendaError.notFound("calendar \(id)") }
            return calendar
        }
        guard let calendar = store.defaultCalendarForNewEvents else {
            throw AgendaError.notFound("a writable calendar")
        }
        return calendar
    }

    /// A due time gets an alarm at that time, so Reminders delivers it on every
    /// device; a whole-day due date gets none.
    private func setDue(of reminder: EKReminder, to date: Date?, hasTime: Bool) {
        for alarm in reminder.alarms ?? [] where alarm.absoluteDate != nil {
            reminder.removeAlarm(alarm)
        }
        guard let date else {
            reminder.dueDateComponents = nil
            return
        }
        let calendar = Calendar.current
        var components = calendar.dateComponents(
            hasTime ? [.year, .month, .day, .hour, .minute] : [.year, .month, .day], from: date)
        components.calendar = calendar
        components.timeZone = hasTime ? calendar.timeZone : nil
        reminder.dueDateComponents = components
        if hasTime { reminder.addAlarm(EKAlarm(absoluteDate: date)) }
    }

    /// Finds one occurrence from an `AgendaEvent.id` (`identifier|start`).
    private func occurrence(_ id: String) throws -> EKEvent {
        let parts = id.split(separator: "|", maxSplits: 1).map(String.init)
        guard parts.count == 2, let epoch = TimeInterval(parts[1]) else {
            throw AgendaError.notFound("event \(id)")
        }
        let start = Date(timeIntervalSince1970: epoch)
        let predicate = store.predicateForEvents(
            withStart: start.addingTimeInterval(-60), end: start.addingTimeInterval(86_400),
            calendars: nil)
        guard
            let event = store.events(matching: predicate).first(where: {
                ($0.eventIdentifier ?? $0.calendarItemIdentifier) == parts[0]
                    && abs($0.startDate.timeIntervalSince(start)) < 1
            })
        else { throw AgendaError.notFound("event \(id)") }
        return event
    }

    // MARK: Conversion

    nonisolated private static func value(of event: EKEvent) -> AgendaEvent {
        let identifier = event.eventIdentifier ?? event.calendarItemIdentifier
        let start = event.startDate ?? Date()
        let others = (event.attendees ?? []).contains { !$0.isCurrentUser }
        return AgendaEvent(
            id: "\(identifier)|\(Int(start.timeIntervalSince1970))",
            title: event.title ?? "(untitled)",
            start: start,
            end: event.endDate ?? start,
            isAllDay: event.isAllDay,
            calendarID: event.calendar?.calendarIdentifier ?? "",
            calendarTitle: event.calendar?.title ?? "",
            colorHex: hex(event.calendar?.cgColor),
            location: event.location,
            notes: event.notes,
            hasOtherAttendees: others,
            isEditable: event.calendar?.allowsContentModifications ?? false)
    }

    nonisolated private static func value(of reminder: EKReminder) -> AgendaReminder {
        let components = reminder.dueDateComponents
        var due: Date?
        if let components {
            let calendar = components.calendar ?? Calendar.current
            due = calendar.date(from: components)
        }
        return AgendaReminder(
            id: reminder.calendarItemIdentifier,
            title: reminder.title ?? "(untitled)",
            notes: reminder.notes,
            listID: reminder.calendar?.calendarIdentifier ?? "",
            listTitle: reminder.calendar?.title ?? "",
            colorHex: hex(reminder.calendar?.cgColor),
            due: due,
            dueHasTime: components?.hour != nil,
            isCompleted: reminder.isCompleted,
            completedAt: reminder.completionDate,
            createdAt: reminder.creationDate)
    }

    nonisolated private static func hex(_ color: CGColor?) -> String? {
        guard let color, let srgb = NSColor(cgColor: color)?.usingColorSpace(.sRGB) else {
            return nil
        }
        return String(
            format: "#%02X%02X%02X", Int(srgb.redComponent * 255), Int(srgb.greenComponent * 255),
            Int(srgb.blueComponent * 255))
    }
}
