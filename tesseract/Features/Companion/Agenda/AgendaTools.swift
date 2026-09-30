//
//  AgendaTools.swift
//  tesseract
//
//  The agenda tools, registered in every conversation: read the agenda, add
//  a reminder, update or complete one, add an event, move an event. Apple
//  Reminders and Calendar are the only place the owner's tasks and plans
//  live, so "remind me" always becomes a real reminder that reaches the phone
//  and watch — with or without the Companion switched on. Every write answers
//  with a one-line confirmation, and Today offers the undo.
//

import Foundation

nonisolated enum AgendaToolNames {
    static let agenda = "agenda"
    static let addReminder = "add_reminder"
    static let updateReminder = "update_reminder"
    static let addEvent = "add_event"
    static let moveEvent = "move_event"
    static let all: [String] = [agenda, addReminder, updateReminder, addEvent, moveEvent]
}

@MainActor
func createAgendaTools(agenda: Agenda, now: @escaping @MainActor () -> Date = Date.init)
    -> [AgentToolDefinition]
{
    [
        agendaReadTool(agenda: agenda, now: now),
        addReminderTool(agenda: agenda, now: now),
        updateReminderTool(agenda: agenda, now: now),
        addEventTool(agenda: agenda, now: now),
        moveEventTool(agenda: agenda, now: now),
    ]
}

// MARK: - agenda

@MainActor
private func agendaReadTool(agenda: Agenda, now: @escaping @MainActor () -> Date)
    -> AgentToolDefinition
{
    AgentToolDefinition(
        name: AgendaToolNames.agenda,
        label: "agenda",
        description: """
            Read the owner's calendar events and reminders (Apple Calendar and Reminders) for \
            a range of days, with the ids the other agenda tools take. With today in range it \
            also lists overdue reminders, the Inbox (reminders with no date) and what is done \
            today.
            """,
        parameterSchema: JSONSchema(
            type: "object",
            properties: [
                "from": PropertySchema(
                    type: "string",
                    description: "First day: YYYY-MM-DD, \"today\" or \"tomorrow\". Default: today."
                ),
                "days": PropertySchema(
                    type: "integer", description: "How many days, 1–14. Default: 1."),
            ],
            required: []),
        execute: { _, args, _, _ in
            await agenda.ensureAccess()
            let fromArg = ToolArgExtractor.string(args, key: "from")
            let days = min(max(ToolArgExtractor.int(args, key: "days") ?? 1, 1), 14)
            let (startDay, current): (Date, Date) = try await MainActor.run {
                let current = now()
                guard let fromArg else {
                    return (Calendar.current.startOfDay(for: current), current)
                }
                guard let parsed = AgendaTime.parse(fromArg, now: current) else {
                    throw AgendaError.invalid("Can't read the date “\(fromArg)”; use YYYY-MM-DD.")
                }
                return (Calendar.current.startOfDay(for: parsed.date), current)
            }
            let listing = await AgendaListing.render(
                agenda: agenda, from: startDay, days: days, now: current)
            return .text(listing)
        })
}

// MARK: - add_reminder

@MainActor
private func addReminderTool(agenda: Agenda, now: @escaping @MainActor () -> Date)
    -> AgentToolDefinition
{
    AgentToolDefinition(
        name: AgendaToolNames.addReminder,
        label: "add_reminder",
        description: """
            Add a reminder to Apple Reminders — for anything the owner needs to do or asks to \
            be reminded of. Give it a due time only when the owner gave one or one follows \
            from their words; a reminder with a time alerts on their Mac, phone and watch. \
            Without a due time it lands in the Inbox or in its Area undated.
            """,
        parameterSchema: JSONSchema(
            type: "object",
            properties: [
                "title": PropertySchema(
                    type: "string", description: "What to do, short: \"Call the dentist\"."),
                "area": PropertySchema(
                    type: "string",
                    description:
                        "The owner's Area (a Reminders list), e.g. Work or Health. Omit for the Inbox."
                ),
                "due": PropertySchema(
                    type: "string",
                    description: "Local time YYYY-MM-DDTHH:MM, or YYYY-MM-DD for a whole day."),
                "after_event": PropertySchema(
                    type: "string",
                    description:
                        "Instead of due: the calendar event it follows (\"the 1:1\", or an event id); due becomes that event's end."
                ),
                "notes": PropertySchema(type: "string", description: "Optional details."),
            ],
            required: ["title"]),
        execute: { _, args, _, _ in
            guard let title = ToolArgExtractor.string(args, key: "title") else {
                throw AgendaError.invalid("add_reminder needs a title.")
            }
            let area = ToolArgExtractor.string(args, key: "area")
            let dueArg = ToolArgExtractor.string(args, key: "due")
            let anchor = ToolArgExtractor.string(args, key: "after_event")
            let notes = ToolArgExtractor.string(args, key: "notes")
            await agenda.ensureAccess()
            return try await MainActor.run {
                let due = try resolveDue(dueArg: dueArg, anchor: anchor, agenda: agenda, now: now())
                let (reminder, change) = try agenda.addReminder(
                    title: title, areaName: area, due: due?.date, dueHasTime: due?.hasTime ?? false,
                    notes: notes, source: "tool")
                return .text("\(change.line) (id: \(reminder.id))")
            }
        })
}

// MARK: - update_reminder

@MainActor
private func updateReminderTool(agenda: Agenda, now: @escaping @MainActor () -> Date)
    -> AgentToolDefinition
{
    AgentToolDefinition(
        name: AgendaToolNames.updateReminder,
        label: "update_reminder",
        description: """
            Change one reminder by id (from the agenda tool): complete it or reopen it, rename \
            it, move it to another time or Area, or clear its date.
            """,
        parameterSchema: JSONSchema(
            type: "object",
            properties: [
                "id": PropertySchema(type: "string", description: "The reminder's id."),
                "completed": PropertySchema(
                    type: "boolean", description: "true completes it, false reopens it."),
                "title": PropertySchema(type: "string", description: "A new title."),
                "area": PropertySchema(type: "string", description: "Move it to this Area."),
                "due": PropertySchema(
                    type: "string",
                    description:
                        "New local time YYYY-MM-DDTHH:MM, a whole day YYYY-MM-DD, or \"none\" to clear the date."
                ),
                "after_event": PropertySchema(
                    type: "string",
                    description: "Instead of due: re-time it to the end of this calendar event."),
                "notes": PropertySchema(type: "string", description: "Replace its notes."),
            ],
            required: ["id"]),
        execute: { _, args, _, _ in
            guard let id = ToolArgExtractor.string(args, key: "id") else {
                throw AgendaError.invalid("update_reminder needs the reminder's id.")
            }
            let completed = try ToolArgExtractor.strictBool(args, key: "completed")
            let title = ToolArgExtractor.string(args, key: "title")
            let area = ToolArgExtractor.string(args, key: "area")
            let dueArg = ToolArgExtractor.string(args, key: "due")
            let anchor = ToolArgExtractor.string(args, key: "after_event")
            let notes = ToolArgExtractor.string(args, key: "notes")
            await agenda.ensureAccess()
            let due: ReminderChange.Due? = try await MainActor.run {
                if dueArg?.lowercased() == "none" { return .clear }
                return try resolveDue(dueArg: dueArg, anchor: anchor, agenda: agenda, now: now())
                    .map { .set($0.date, hasTime: $0.hasTime) }
            }
            let (_, change) = try await agenda.updateReminder(
                id: id, title: title, areaName: area, due: due, notes: notes,
                completed: completed, source: "tool")
            return .text(change.line)
        })
}

// MARK: - add_event

@MainActor
private func addEventTool(agenda: Agenda, now: @escaping @MainActor () -> Date)
    -> AgentToolDefinition
{
    AgentToolDefinition(
        name: AgendaToolNames.addEvent,
        label: "add_event",
        description: """
            Add an event to the owner's calendar (Apple Calendar), in the default calendar \
            unless one is named. Use it for time the owner wants to block or a meeting they \
            describe.
            """,
        parameterSchema: JSONSchema(
            type: "object",
            properties: [
                "title": PropertySchema(type: "string", description: "The event's title."),
                "start": PropertySchema(
                    type: "string",
                    description: "Local start YYYY-MM-DDTHH:MM, or YYYY-MM-DD for an all-day event."
                ),
                "end": PropertySchema(type: "string", description: "Local end YYYY-MM-DDTHH:MM."),
                "duration_minutes": PropertySchema(
                    type: "integer", description: "Instead of end. Default: 30."),
                "calendar": PropertySchema(type: "string", description: "A calendar's name."),
                "location": PropertySchema(type: "string", description: "Where."),
                "notes": PropertySchema(type: "string", description: "Details."),
            ],
            required: ["title", "start"]),
        execute: { _, args, _, _ in
            guard let title = ToolArgExtractor.string(args, key: "title"),
                let startArg = ToolArgExtractor.string(args, key: "start")
            else { throw AgendaError.invalid("add_event needs a title and a start.") }
            let endArg = ToolArgExtractor.string(args, key: "end")
            let minutes = ToolArgExtractor.int(args, key: "duration_minutes")
            let calendarName = ToolArgExtractor.string(args, key: "calendar")
            let location = ToolArgExtractor.string(args, key: "location")
            let notes = ToolArgExtractor.string(args, key: "notes")
            await agenda.ensureAccess()
            return try await MainActor.run {
                let current = now()
                guard let start = AgendaTime.parse(startArg, now: current) else {
                    throw AgendaError.invalid(
                        "Can't read the start “\(startArg)”; use YYYY-MM-DDTHH:MM.")
                }
                let calendar = Calendar.current
                let end: Date
                if !start.hasTime {
                    end = calendar.date(byAdding: .day, value: 1, to: start.date) ?? start.date
                } else if let endArg {
                    guard let parsed = AgendaTime.parse(endArg, now: current), parsed.hasTime else {
                        throw AgendaError.invalid(
                            "Can't read the end “\(endArg)”; use YYYY-MM-DDTHH:MM.")
                    }
                    end = parsed.date
                } else {
                    end = start.date.addingTimeInterval(TimeInterval((minutes ?? 30) * 60))
                }
                let (event, change) = try agenda.addEvent(
                    title: title, start: start.date, end: end, isAllDay: !start.hasTime,
                    calendarName: calendarName, location: location, notes: notes, source: "tool")
                return .text("\(change.line) (id: \(event.id))")
            }
        })
}

// MARK: - move_event

@MainActor
private func moveEventTool(agenda: Agenda, now: @escaping @MainActor () -> Date)
    -> AgentToolDefinition
{
    AgentToolDefinition(
        name: AgendaToolNames.moveEvent,
        label: "move_event",
        description: """
            Move one calendar event (by id, from the agenda tool) to a new time. It keeps its \
            length unless an end is given. Only this occurrence of a repeating event moves.
            """,
        parameterSchema: JSONSchema(
            type: "object",
            properties: [
                "id": PropertySchema(type: "string", description: "The event's id."),
                "start": PropertySchema(
                    type: "string", description: "New local start YYYY-MM-DDTHH:MM."),
                "end": PropertySchema(
                    type: "string", description: "New local end YYYY-MM-DDTHH:MM."),
            ],
            required: ["id", "start"]),
        execute: { _, args, _, _ in
            guard let id = ToolArgExtractor.string(args, key: "id"),
                let startArg = ToolArgExtractor.string(args, key: "start")
            else { throw AgendaError.invalid("move_event needs the event's id and a start.") }
            let endArg = ToolArgExtractor.string(args, key: "end")
            await agenda.ensureAccess()
            return try await MainActor.run {
                let current = now()
                guard let start = AgendaTime.parse(startArg, now: current), start.hasTime else {
                    throw AgendaError.invalid(
                        "Can't read the start “\(startArg)”; use YYYY-MM-DDTHH:MM.")
                }
                var end: Date?
                if let endArg {
                    guard let parsed = AgendaTime.parse(endArg, now: current), parsed.hasTime else {
                        throw AgendaError.invalid(
                            "Can't read the end “\(endArg)”; use YYYY-MM-DDTHH:MM.")
                    }
                    end = parsed.date
                }
                let (event, change) = try agenda.moveEvent(
                    id: id, start: start.date, end: end, source: "tool")
                return .text("\(change.line) (id: \(event.id))")
            }
        })
}

// MARK: - Shared

/// A due time from either an explicit time or an event anchor.
@MainActor
private func resolveDue(dueArg: String?, anchor: String?, agenda: Agenda, now: Date) throws
    -> AgendaTime.Parsed?
{
    if let anchor, !anchor.isEmpty {
        let events = agenda.events(
            from: now.addingTimeInterval(-12 * 3600), to: now.addingTimeInterval(7 * 86_400))
        guard let event = AgendaTime.anchorEvent(anchor, in: events, now: now) else {
            throw AgendaError.invalid("No calendar event matches “\(anchor)”.")
        }
        return AgendaTime.Parsed(date: event.end, hasTime: true)
    }
    guard let dueArg, !dueArg.isEmpty else { return nil }
    guard let parsed = AgendaTime.parse(dueArg, now: now) else {
        throw AgendaError.invalid(
            "Can't read the time “\(dueArg)”; use YYYY-MM-DDTHH:MM or YYYY-MM-DD.")
    }
    return parsed
}
