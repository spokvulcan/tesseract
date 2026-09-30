//
//  AgendaToolsTests.swift
//  tesseractTests
//
//  The agenda tools end to end, below the model: registered tool in, the
//  in-memory Agenda's state and the one-line confirmation out. Never the real
//  EventKit store (ADR-0073).
//

import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

@MainActor
struct AgendaToolsTests {

    /// Wednesday 30 September 2026, 09:00 local.
    static let now = local(30, 9, 0)

    static func local(_ day: Int, _ hour: Int, _ minute: Int) -> Date {
        Calendar.current.date(
            from: DateComponents(year: 2026, month: 9, day: day, hour: hour, minute: minute))!
    }

    private struct Fixture {
        let store: InMemoryAgendaStore
        let agenda: Agenda
        let tools: [String: AgentToolDefinition]

        func run(_ name: String, _ args: [String: JSONValue]) async throws -> String {
            let tool = try #require(tools[name])
            return try await tool.execute("call-1", args, nil, nil).content.textContent
        }
    }

    private func fixture(
        access: AgendaAccess = .full, areasJSON: String = "{}",
        reminders: [AgendaReminder] = [], events: [AgendaEvent] = []
    ) -> Fixture {
        let store = InMemoryAgendaStore(
            access: access,
            lists: [
                AgendaList(id: "inbox", title: "Reminders", isDefault: true),
                AgendaList(id: "work", title: "Work", isDefault: false),
                AgendaList(id: "health", title: "Health", isDefault: false),
            ],
            reminders: reminders, events: events, now: { Self.now })
        let agenda = Agenda(
            store: store, areaMapJSON: { areasJSON }, trace: scratchTrace(), now: { Self.now })
        let tools = Dictionary(
            uniqueKeysWithValues: createAgendaTools(agenda: agenda, now: { Self.now }).map {
                ($0.name, $0)
            })
        return Fixture(store: store, agenda: agenda, tools: tools)
    }

    private var oneOnOne: AgendaEvent {
        AgendaEvent(
            id: "one-on-one|1", title: "1:1 with Anna", start: Self.local(30, 13, 0),
            end: Self.local(30, 13, 45), calendarID: "home", calendarTitle: "Home",
            hasOtherAttendees: true)
    }

    @Test func thereAreFiveAgendaTools() {
        #expect(Set(fixture().tools.keys) == Set(AgendaToolNames.all))
        #expect(AgendaToolNames.all.count <= 5)
    }

    @Test func remindMeWithATimeBecomesATimedReminder() async throws {
        let f = fixture()
        let line = try await f.run(
            "add_reminder",
            [
                "title": .string("Call the dentist"), "due": .string("2026-10-01T10:00"),
                "area": .string("health"),
            ])
        let reminder = try #require(await f.store.openReminders().first)
        #expect(reminder.title == "Call the dentist")
        #expect(reminder.listID == "health")
        #expect(
            reminder.due
                == Calendar.current.date(
                    from: DateComponents(year: 2026, month: 10, day: 1, hour: 10)))
        #expect(reminder.dueHasTime)
        #expect(line.hasPrefix("Added “Call the dentist” — tomorrow 10:00, Health."))
    }

    @Test func noTimeLandsUndatedInTheInbox() async throws {
        let f = fixture()
        let line = try await f.run("add_reminder", ["title": .string("Buy stamps")])
        let reminder = try #require(await f.store.openReminders().first)
        #expect(reminder.listID == "inbox")
        #expect(reminder.due == nil)
        #expect(line.hasPrefix("Added “Buy stamps”, Inbox."))
    }

    @Test func afterAnEventResolvesToItsEnd() async throws {
        let f = fixture(events: [oneOnOne])
        _ = try await f.run(
            "add_reminder", ["title": .string("Send notes"), "after_event": .string("the 1:1")])
        let reminder = try #require(await f.store.openReminders().first)
        #expect(reminder.due == Self.local(30, 13, 45))
        #expect(reminder.dueHasTime)
    }

    @Test func anUnknownAreaNamesTheRealOnes() async throws {
        let f = fixture()
        await #expect(throws: AgendaError.self) {
            _ = try await f.run("add_reminder", ["title": .string("x"), "area": .string("Garden")])
        }
        #expect(await f.store.openReminders().isEmpty)
    }

    @Test func completingAndUndoingRoundTrips() async throws {
        let f = fixture(reminders: [
            AgendaReminder(id: "r1", title: "Pay rent", listID: "inbox", listTitle: "Reminders")
        ])
        let line = try await f.run(
            "update_reminder", ["id": .string("r1"), "completed": .bool(true)])
        #expect(line == "Done: “Pay rent”.")
        #expect(await f.store.openReminders().isEmpty)

        let change = try #require(f.agenda.lastChange)
        try await f.agenda.undo(change)
        #expect(await f.store.openReminders().map(\.id) == ["r1"])
    }

    @Test func reTimingAReminderSaysWhere() async throws {
        let f = fixture(reminders: [
            AgendaReminder(id: "gym", title: "Gym", listID: "health", listTitle: "Health")
        ])
        let line = try await f.run(
            "update_reminder", ["id": .string("gym"), "due": .string("today 20:00")])
        #expect(line == "Moved “Gym” to today 20:00.")
        #expect(await f.store.openReminders().first?.due == Self.local(30, 20, 0))
    }

    @Test func addEventDefaultsToHalfAnHourAndMoveKeepsLength() async throws {
        let f = fixture()
        let added = try await f.run(
            "add_event", ["title": .string("Focus block"), "start": .string("2026-09-30T15:00")])
        #expect(added.hasPrefix("Added “Focus block” — today 15:00–15:30, Home."))
        let event = try #require(
            f.store.events(from: Self.local(30, 0, 0), to: Self.local(31, 0, 0)).first)

        let moved = try await f.run(
            "move_event", ["id": .string(event.id), "start": .string("2026-09-30T17:00")])
        #expect(moved.hasPrefix("Moved “Focus block” to today 17:00–17:30."))
        let after = try #require(
            f.store.events(from: Self.local(30, 0, 0), to: Self.local(31, 0, 0)).first)
        #expect(after.start == Self.local(30, 17, 0))
        #expect(after.end == Self.local(30, 17, 30))
    }

    @Test func theListingShowsTheDayWithIDs() async throws {
        let f = fixture(
            reminders: [
                AgendaReminder(
                    id: "dentist", title: "Call the dentist", listID: "health", listTitle: "Health",
                    due: Self.local(30, 10, 0), dueHasTime: true),
                AgendaReminder(
                    id: "passport", title: "Renew passport", listID: "inbox",
                    listTitle: "Reminders",
                    due: Self.local(28, 0, 0)),
                AgendaReminder(
                    id: "stamps", title: "Buy stamps", listID: "inbox", listTitle: "Reminders"),
            ],
            events: [oneOnOne])
        let listing = try await f.run("agenda", [:])
        #expect(listing.contains("Wednesday 30 September 2026 (today)"))
        #expect(listing.contains("- 13:00–13:45 1:1 with Anna [Home; id one-on-one|1]"))
        #expect(listing.contains("- 10:00 Call the dentist · Health [id dentist]"))
        #expect(listing.contains("Overdue:"))
        #expect(listing.contains("Renew passport · Inbox [id passport]"))
        #expect(listing.contains("No date:"))
        #expect(listing.contains("- Buy stamps · Inbox [id stamps]"))
        #expect(listing.contains("Areas: Reminders, Work, Health · Inbox: Reminders"))
    }

    @Test func withoutAccessTheToolSaysHowToAllowIt() async throws {
        let f = fixture(access: .none)
        do {
            _ = try await f.run("add_reminder", ["title": .string("x")])
            Issue.record("a write without access must fail")
        } catch {
            #expect(error.localizedDescription.contains("System Settings"))
        }
    }

    /// Areas mapped in Settings rename lists and pick the Inbox.
    @Test func mappedAreasAndInbox() async throws {
        let map = AreaMap(
            entries: [.init(listID: "work", name: "Job"), .init(listID: "health", name: "Body")],
            inboxListID: "inbox")
        let f = fixture(areasJSON: map.json)
        let line = try await f.run(
            "add_reminder", ["title": .string("Stretch"), "area": .string("bo")])
        #expect(line.hasPrefix("Added “Stretch”, Body."))
        #expect(f.agenda.areas.map(\.name) == ["Job", "Body"])
    }
}
