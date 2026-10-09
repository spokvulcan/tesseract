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

    @Test func thereAreSixAgendaTools() {
        #expect(Set(fixture().tools.keys) == Set(AgendaToolNames.all))
        #expect(AgendaToolNames.all.count <= 6)
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

    @Test func deleteEventRemovesTheOwnersBlockAndUndoBringsItBack() async throws {
        let f = fixture()
        _ = try await f.run(
            "add_event", ["title": .string("Errand block"), "start": .string("2026-09-30T13:00")])
        let event = try #require(
            f.store.events(from: Self.local(30, 0, 0), to: Self.local(31, 0, 0)).first)

        let deleted = try await f.run("delete_event", ["id": .string(event.id)])
        #expect(deleted.hasPrefix("Deleted “Errand block” — today 13:00–13:30, Home."))
        #expect(f.store.events(from: Self.local(30, 0, 0), to: Self.local(31, 0, 0)).isEmpty)

        try await f.agenda.undo(try #require(f.agenda.lastChange))
        let restored = try #require(
            f.store.events(from: Self.local(30, 0, 0), to: Self.local(31, 0, 0)).first)
        #expect(restored.title == "Errand block")
        #expect(restored.start == Self.local(30, 13, 0))
    }

    @Test func aMeetingWithOthersIsNotDeletedHere() async throws {
        let f = fixture(events: [oneOnOne])
        await #expect(throws: AgendaError.self) {
            _ = try await f.run("delete_event", ["id": .string(oneOnOne.id)])
        }
        #expect(
            f.store.events(from: Self.local(30, 0, 0), to: Self.local(31, 0, 0)).map(\.id)
                == [oneOnOne.id])
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

    @Test func pastMidnightTheSnapshotStillHoldsTheDayThatIsEnding() async {
        // 00:40 on 1 October: until 04:00 the owner's day is still 30 September.
        let night = Self.local(31, 0, 40)
        func event(_ id: String, _ start: Date) -> AgendaEvent {
            AgendaEvent(
                id: id, title: id, start: start, end: start.addingTimeInterval(3600),
                calendarID: "home", calendarTitle: "Home")
        }
        func done(_ id: String, at: Date) -> AgendaReminder {
            AgendaReminder(
                id: id, title: id, listID: "inbox", listTitle: "Reminders", isCompleted: true,
                completedAt: at)
        }
        let store = InMemoryAgendaStore(
            lists: [AgendaList(id: "inbox", title: "Reminders", isDefault: true)],
            reminders: [done("evening", at: Self.local(30, 23, 30)), done("late", at: night)],
            events: [
                event("dinner", Self.local(30, 20, 0)), event("work", Self.local(31, 9, 0)),
                event("friday", Self.local(32, 9, 0)),
            ],
            now: { night })
        let agenda = Agenda(store: store, trace: scratchTrace(), now: { night })
        await agenda.refresh()
        #expect(agenda.snapshot.events.map(\.id) == ["dinner", "work"])
        #expect(Set(agenda.snapshot.doneToday.map(\.id)) == ["evening", "late"])
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

/// A calendar location as people read it: a meeting link is its service, a
/// password never shows, and a place stays as written.
struct AgendaPlaceTests {

    @Test(
        arguments: [
            ("https://us04web.zoom.us/j/78521739484?pwd=abc.1", "Zoom"),
            ("https://meet.google.com/abc-defg-hij", "Google Meet"),
            ("Room 4 / https://teams.microsoft.com/l/meetup-join/xyz", "Room 4 · Microsoft Teams"),
            ("https://www.example.org/call", "example.org"),
            ("Efstaleiti 1", "Efstaleiti 1"),
            ("  ", nil),
        ] as [(String, String?)])
    func aLocationReadsAsAPlace(_ location: String, _ label: String?) {
        #expect(AgendaPlace.label(location) == label)
    }

    @Test func theAgentKeepsTheLinkButNotItsPassword() {
        #expect(
            AgendaPlace.label("https://us04web.zoom.us/j/78521739484?pwd=abc.1", withLink: true)
                == "Zoom — us04web.zoom.us/j/78521739484")
    }
}
