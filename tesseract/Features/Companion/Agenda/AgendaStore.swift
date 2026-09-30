//
//  AgendaStore.swift
//  tesseract
//
//  The Agenda port: one protocol over Apple Reminders and Calendar. The
//  EventKit adapter is the production store; the in-memory store is the test
//  host's and every test's (ADR-0073: no automated test touches the owner's
//  real Reminders or Calendar, not even a scratch list).
//

import Foundation

@MainActor
protocol AgendaStore: AnyObject {

    /// What Tesseract may use right now.
    var access: AgendaAccess { get }

    /// Ask the OS for whichever half is still undetermined. Denied halves stay
    /// denied; the app keeps working in a reduced mode without them.
    func requestAccess() async -> AgendaAccess

    /// Called on the main actor whenever the store changes — a write here, an
    /// edit in Reminders or Calendar, or a sync from another device.
    var onChange: (@MainActor () -> Void)? { get set }

    // MARK: Reading

    /// Event occurrences overlapping `[from, to)`, soonest first.
    func events(from: Date, to: Date) -> [AgendaEvent]

    /// Every incomplete reminder, dated or not.
    func openReminders() async -> [AgendaReminder]

    /// Reminders completed within `[from, to)`.
    func completedReminders(from: Date, to: Date) async -> [AgendaReminder]

    func reminderLists() -> [AgendaList]

    func eventCalendars() -> [AgendaCalendar]

    // MARK: Writing

    func addReminder(_ draft: ReminderDraft) throws -> AgendaReminder

    func updateReminder(id: String, _ change: ReminderChange) throws -> AgendaReminder

    func deleteReminder(id: String) throws

    func addEvent(_ draft: EventDraft) throws -> AgendaEvent

    func updateEvent(id: String, _ change: EventChange) throws -> AgendaEvent

    func deleteEvent(id: String) throws
}

extension AgendaStore {
    /// The reminder with `id` among the open ones, or among those completed in
    /// the last week (enough for undo and for today's done list).
    func reminder(id: String, now: Date = Date()) async -> AgendaReminder? {
        if let open = await openReminders().first(where: { $0.id == id }) { return open }
        return await completedReminders(from: now.addingTimeInterval(-7 * 86_400), to: now)
            .first { $0.id == id }
    }
}
