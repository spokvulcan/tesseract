//
//  TodayGalleryTests.swift
//  tesseractTests
//
//  Today, rendered with the app's wiring over fixture days (an in-memory
//  Agenda and a saved day state, ADR-0073): the morning after the plan, the
//  same day at 14:20 deep in the must-do's slot (started: its time left
//  drains on the Now Card; not started: one click from starting), a busy
//  midday with two slid steps and things waiting, an evening whose wrap-up
//  was closed on the panel (its leftovers wait in Today), an evening
//  with the day done, and the same night past midnight, still that day until
//  04:00. Each renders at a wide, a regular and a phone width, so every
//  layout's body runs; the morning also renders with two pictures waiting in
//  the composer, for Jarvis. With TODAY_GALLERY_DIR set (TEST_RUNNER_TODAY_GALLERY_DIR
//  through xcodebuild), each render is also written there as a PNG, in dark
//  and light, for judging the page's layout by eye; glass colors don't survive
//  the offscreen render (docs/testing.md).
//

import AppKit
import Foundation
import SwiftUI
import Testing

@testable import Tesseract_Agent

@MainActor
@Suite(.serialized)
struct TodayGalleryTests {

    static let directory = ProcessInfo.processInfo.environment["TODAY_GALLERY_DIR"].map {
        URL(fileURLWithPath: $0, isDirectory: true)
    }

    /// Wide (the side column beside the table), regular (below it), phone.
    static let widths: [(name: String, width: CGFloat, height: CGFloat)] = [
        ("wide", 1280, 1100), ("regular", 860, 1240), ("phone", 390, 1700),
    ]

    /// Lets the hosted view's deferred updates run before the next layout. A
    /// gallery written out for judging by eye waits `duration`, for the page to
    /// settle; the test itself needs only the run loop to come round, and the
    /// longer waits were most of the suite's time.
    static func settle(_ duration: Duration) async throws {
        try await Task.sleep(for: directory == nil ? .milliseconds(20) : duration)
    }

    @Test(arguments: TodayFixture.allCases)
    func todayRendersAtEveryWidth(_ fixture: TodayFixture) async throws {
        for size in Self.widths {
            for dark in Self.directory == nil ? [true] : [true, false] {
                let image = try await render(
                    fixture, width: size.width, height: size.height, dark: dark)
                #expect(image.pixelsWide > 0)
                if let directory = Self.directory {
                    try FileManager.default.createDirectory(
                        at: directory, withIntermediateDirectories: true)
                    let name = "\(fixture.rawValue)-\(size.name)-\(dark ? "dark" : "light").png"
                    try #require(image.representation(using: .png, properties: [:]))
                        .write(to: directory.appendingPathComponent(name))
                }
            }
        }
    }

    /// The morning with two pictures waiting in the composer, for Jarvis:
    /// their strip, the picture button, and the question they ask as the
    /// field's placeholder.
    @Test func todayComposerShowsWaitingPictures() async throws {
        for size in Self.widths where size.name != "regular" {
            for dark in Self.directory == nil ? [true] : [true, false] {
                let image = try await render(
                    .morning, width: size.width, height: size.height, dark: dark,
                    prepare: { container in
                        container.todayVisionAvailability.refresh()
                        container.todayDraft.attachImages([
                            ImageAttachment(
                                data: ImageTestFixtures.flyerPNG(
                                    title: "Parents' evening", line: "Wed 14 Oct · 17:00",
                                    hue: 0.58),
                                mimeType: "image/png"),
                            ImageAttachment(
                                data: ImageTestFixtures.flyerPNG(
                                    title: "Market day", line: "Sat 17 Oct · 09:00", hue: 0.08),
                                mimeType: "image/png"),
                        ])
                    })
                #expect(image.pixelsWide > 0)
                if let directory = Self.directory {
                    try FileManager.default.createDirectory(
                        at: directory, withIntermediateDirectories: true)
                    let name = "morning-pictures-\(size.name)-\(dark ? "dark" : "light").png"
                    try #require(image.representation(using: .png, properties: [:]))
                        .write(to: directory.appendingPathComponent(name))
                }
            }
        }
    }

    private func render(
        _ fixture: TodayFixture, width: CGFloat, height: CGFloat, dark: Bool,
        prepare: (DependencyContainer) -> Void = { _ in }
    ) async throws -> NSBitmapImageRep {
        let container = try await fixture.container()
        prepare(container)
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: width, height: height),
            styleMask: [.borderless], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
        defer { window.close() }
        let host = NSHostingView(
            rootView: TodayView(fixedNow: fixture.now)
                .injectCompanionDependencies(from: container)
                .frame(width: width, height: height)
                .background(Color(nsColor: .windowBackgroundColor)))
        window.contentView = host
        window.layoutIfNeeded()
        try await Self.settle(.milliseconds(300))
        window.layoutIfNeeded()
        let rep = try #require(host.bitmapImageRepForCachingDisplay(in: host.bounds))
        host.cacheDisplay(in: host.bounds, to: rep)
        return rep
    }
}

// MARK: - Fixture days

/// A day Today can be drawn over, on Sunday 4 October 2026.
enum TodayFixture: String, CaseIterable, CustomTestStringConvertible {
    case morning
    case focus
    case unstarted
    case inCall
    case midday
    case wrapUpClosed
    case evening
    case night

    var testDescription: String { rawValue }

    static func at(_ hour: Int, _ minute: Int = 0, day: Int = 4) -> Date {
        Calendar.current.date(
            from: DateComponents(year: 2026, month: 10, day: day, hour: hour, minute: minute))!
    }

    var now: Date {
        switch self {
        case .morning: Self.at(8, 20)
        case .focus, .unstarted: Self.at(14, 20)
        case .inCall: Self.at(15, 10)
        case .midday: Self.at(14, 10)
        case .wrapUpClosed: Self.at(21, 10)
        case .evening: Self.at(22, 1)
        case .night: Self.at(0, 40, day: 5)
        }
    }

    // Reminders lists (every list is an Area until the owner maps them).
    static let lists = [
        AgendaList(id: "inbox", title: "Reminders", colorHex: "#8E8E93", isDefault: true),
        AgendaList(id: "work", title: "Work", colorHex: "#FF9F0A", isDefault: false),
        AgendaList(id: "life", title: "Life", colorHex: "#30B0C7", isDefault: false),
        AgendaList(id: "health", title: "Health", colorHex: "#FF375F", isDefault: false),
        AgendaList(id: "daily", title: "Daily", colorHex: "#AF52DE", isDefault: false),
        AgendaList(id: "duolingo", title: "Duolingo", colorHex: "#34C759", isDefault: false),
        AgendaList(id: "buy", title: "Buy", colorHex: "#0A84FF", isDefault: false),
    ]
    static let calendars = [
        AgendaCalendar(
            id: "work", title: "Work", colorHex: "#0A84FF", isWritable: true, isDefault: true),
        AgendaCalendar(
            id: "personal", title: "Personal", colorHex: "#30D158", isWritable: true,
            isDefault: false),
    ]

    static func reminder(
        _ id: String, _ title: String, list: String, due: Date? = nil, timed: Bool = false,
        doneAt: Date? = nil
    ) -> AgendaReminder {
        let listTitle = lists.first { $0.id == list }?.title ?? list
        let color = lists.first { $0.id == list }?.colorHex
        return AgendaReminder(
            id: id, title: title, listID: list, listTitle: listTitle, colorHex: color, due: due,
            dueHasTime: timed, isCompleted: doneAt != nil, completedAt: doneAt)
    }

    static func event(
        _ id: String, _ title: String, _ start: Date, _ end: Date, calendar: String = "work",
        location: String? = nil, meeting: Bool = false
    ) -> AgendaEvent {
        let found = calendars.first { $0.id == calendar }
        return AgendaEvent(
            id: id, title: title, start: start, end: end, calendarID: calendar,
            calendarTitle: found?.title ?? calendar, colorHex: found?.colorHex,
            location: location, hasOtherAttendees: meeting)
    }

    var reminders: [AgendaReminder] {
        let today = Self.at(0)
        switch self {
        case .morning, .focus, .unstarted, .inCall:
            let focus = self != .morning
            return [
                Self.reminder(
                    "mail", "Answer mail", list: "work", due: today,
                    doneAt: focus ? Self.at(8, 45) : nil),
                Self.reminder(
                    "pr", "Review PR #612", list: "work", due: today,
                    doneAt: focus ? Self.at(9, 20) : nil),
                Self.reminder("adr", "Write the cache ADR", list: "work", due: today),
                Self.reminder("rent", "Pay rent", list: "life", due: today),
                Self.reminder(
                    "duolingo", "Пройти урок у Duolingo 💚", list: "duolingo",
                    due: Self.at(21, 30), timed: true),
            ]
        case .midday:
            return [
                Self.reminder(
                    "mail", "Answer mail", list: "work", due: today, doneAt: Self.at(8, 40)),
                Self.reminder(
                    "pr", "Review PR #612", list: "work", due: today, doneAt: Self.at(11, 20)),
                Self.reminder("anna", "Reply to Anna about the deck", list: "work"),
                Self.reminder("adr", "Write the cache ADR", list: "work", due: today),
                Self.reminder("rent", "Pay rent", list: "life", due: today),
                Self.reminder(
                    "dentist", "Call the dentist", list: "health", due: Self.at(16, 30),
                    timed: true),
                Self.reminder("passport", "Renew passport", list: "life", due: Self.at(0, day: 2)),
                Self.reminder("izaro", "Izaro voice lines (PoE)", list: "inbox"),
                Self.reminder("chain", "Order a new bike chain", list: "inbox"),
            ]
        case .evening, .night, .wrapUpClosed:
            let leftovers =
                self == .wrapUpClosed
                ? [
                    Self.reminder("adr", "Write the cache ADR", list: "work", due: today),
                    Self.reminder("rent", "Pay rent", list: "life", due: today),
                ] : []
            return leftovers + [
                Self.reminder(
                    "duolingo", "Пройти урок у Duolingo 💚", list: "duolingo",
                    due: Self.at(21, 30), timed: true, doneAt: Self.at(21, 44)),
                Self.reminder(
                    "companion", "Implement the Companion with a cloud model, to make it useful",
                    list: "daily", due: today, doneAt: Self.at(19, 10)),
                Self.reminder(
                    "clippers", "Ножнички для нігтів", list: "buy", due: today,
                    doneAt: Self.at(13)),
                Self.reminder("izaro", "Izaro voice lines (PoE)", list: "inbox"),
                Self.reminder("moto", "Розпечатати це мото", list: "inbox"),
                Self.reminder(
                    "bins", "Put the bins out", list: "life", due: Self.at(7, 30, day: 5),
                    timed: true),
                Self.reminder("invoice", "Send the invoice", list: "work", due: Self.at(0, day: 5)),
            ]
        }
    }

    var events: [AgendaEvent] {
        let tomorrow = [
            Self.event("w", "Work", Self.at(9, day: 5), Self.at(13, day: 5)),
            Self.event(
                "id", "ID check", Self.at(14, day: 5), Self.at(14, 30, day: 5),
                calendar: "personal"),
            Self.event(
                "class", "Class", Self.at(16, day: 5), Self.at(17, 30, day: 5),
                calendar: "personal"),
            AgendaEvent(
                id: "birthday", title: "Mom's birthday", start: Self.at(0, day: 5),
                end: Self.at(0, day: 6), isAllDay: true, calendarID: "personal",
                calendarTitle: "Personal", colorHex: "#30D158"),
        ]
        switch self {
        case .morning, .focus, .unstarted, .inCall:
            var review = Self.event(
                "d", "Design review", Self.at(15), Self.at(16), location: "Room 4",
                meeting: true)
            review.notes = "Join with Google Meet: https://meet.google.com/abc-defg-hij"
            return [
                Self.event("s", "Standup", Self.at(9, 30), Self.at(10), meeting: true),
                Self.event(
                    "l", "Lunch with Sam", Self.at(12, 30), Self.at(13, 30), calendar: "personal",
                    location: "Café Nord"),
                review,
            ] + tomorrow
        case .midday:
            return [
                Self.event("s", "Standup", Self.at(9, 30), Self.at(10), meeting: true),
                Self.event(
                    "d", "Design review", Self.at(15), Self.at(16), location: "Room 4",
                    meeting: true),
                Self.event(
                    "c", "Climbing", Self.at(18, 30), Self.at(20), calendar: "personal",
                    location: "Boulderhalle"),
            ] + tomorrow
        case .evening, .night, .wrapUpClosed:
            return tomorrow
        }
    }

    var state: DayState {
        var state = DayState(day: DayKey(for: now))
        switch self {
        case .morning, .focus, .unstarted, .inCall:
            state.morningPlanAt = Self.at(8, 18)
            state.mustDoID = "adr"
            state.plan = [
                Placement(reminderID: "mail", start: Self.at(8, 30), minutes: 20),
                Placement(reminderID: "pr", start: Self.at(8, 55), minutes: 30),
                Placement(reminderID: "adr", start: Self.at(13, 45), minutes: 75),
            ]
            // At 14:20 the must-do is under way; unstarted, its cue was closed.
            if self == .focus || self == .inCall {
                state.startedSteps = [StepCue.key(state.plan[2])]
            }
            state.cards = [
                DayCard(
                    id: "morningPlan-0", kind: .morningPlan, createdAt: Self.at(8, 18),
                    isFallback: false,
                    body: .morningPlan(
                        MorningPlanCard(
                            line:
                                "A light morning: two quick things before standup, and the ADR gets the quiet stretch after lunch.",
                            mustDoID: "adr", placements: state.plan,
                            suggestions: [
                                "Start with mail while the coffee's hot.",
                                "Rent takes two minutes; leave it for the evening.",
                            ])))
            ]
        case .midday:
            state.morningPlanAt = Self.at(8, 15)
            state.mustDoID = "adr"
            state.plan = [
                Placement(reminderID: "anna", start: Self.at(13), minutes: 30),
                Placement(reminderID: "rent", start: Self.at(13, 30), minutes: 15),
            ]
            state.departures = [
                Departure(
                    eventID: "c", title: "Climbing", at: Self.at(18), eventStart: Self.at(18, 30),
                    location: "Boulderhalle")
            ]
            state.agents = [
                AgentSignal(
                    id: "session-1", kind: .waiting, agent: "Claude Code", project: "tesseract",
                    directory: "/Users/owl/projects/tesseract",
                    message: "Needs approval to run the test suite", at: Self.at(13, 50))
            ]
            state.cards = [
                DayCard(
                    id: "morningPlan-0", kind: .morningPlan, createdAt: Self.at(8, 15),
                    isFallback: false,
                    body: .morningPlan(
                        MorningPlanCard(
                            line: "A full day.", mustDoID: "adr", placements: state.plan,
                            suggestions: [])),
                    dismissed: true),
                DayCard(
                    id: "breakpoint-1", kind: .breakpoint, createdAt: Self.at(14, 5),
                    isFallback: false,
                    body: .breakpoint(
                        BreakpointCard(
                            awayFrom: Self.at(13, 20), awayUntil: Self.at(14, 5),
                            line:
                                "Two things need you: Anna is waiting on the deck, and Claude Code wants a yes.",
                            needsYou: [
                                WaitingItem(
                                    id: "n1", kind: .notification, title: "Anna · Telegram",
                                    detail: "Can you send me the deck before 4?", app: "Telegram"),
                                WaitingItem(
                                    id: "agent:session-1", kind: .agent,
                                    title: "Claude Code in tesseract",
                                    detail: "Needs approval to run the test suite", app: nil),
                            ],
                            next: [], whereYouWere: "Xcode",
                            canWait: [
                                QuietGroup(
                                    app: "Mail", lines: ["Your order shipped", "Weekly digest"])
                            ]))),
            ]
        case .evening, .night, .wrapUpClosed:
            state.morningPlanAt = Self.at(9)
            state.eveningWrapUpAt = Self.at(21)
            state.weekFocus = "Ship the Companion"
            state.weekFocusSetAt = Self.at(21)
            if self == .wrapUpClosed {
                // Closed on the panel at 21:01: taken in, its leftovers wait here.
                state.cards = [
                    DayCard(
                        id: "eveningWrapUp-0", kind: .eveningWrapUp, createdAt: Self.at(21),
                        isFallback: false,
                        body: .eveningWrapUp(
                            EveningWrapUpCard(
                                line: "The Companion shipped and the streak held.",
                                done: ["Implement the Companion", "Ножнички для нігтів"],
                                leftovers: [
                                    Leftover(
                                        reminderID: "adr", title: "Write the cache ADR",
                                        suggestion: .tomorrow),
                                    Leftover(
                                        reminderID: "rent", title: "Pay rent",
                                        suggestion: .later),
                                ],
                                tomorrowFirst: "09:00 Work")),
                        kept: true)
                ]
                return state
            }
            state.taskProposals = [
                TaskProposal(
                    id: "task-1", title: "Send the case worker the bus receipts",
                    due: Self.at(0, day: 5))
            ]
            state.nightReflectionAt = Self.at(21, 50)
            state.cards = [
                DayCard(
                    id: "nightReflection-0", kind: .nightReflection, createdAt: Self.at(21, 50),
                    isFallback: false,
                    body: .reflection(
                        ReflectionCard(
                            carryOver:
                                "A full Sunday: the Companion shipped and works, the Duolingo streak held, and a long walk in the park lifted the whole evening. You also set up your places and gave me your home base. Tomorrow is a real Monday — work in the morning, the ID check at two, class at four — but the hardest part of the week is behind you.",
                            tomorrow: [], proposals: [])))
            ]
        }
        return state
    }

    /// The app's container over this day: the in-memory Agenda seeded with
    /// it, and a runtime that loads its state.
    @MainActor
    func container() async throws -> DependencyContainer {
        let container = DependencyContainer()
        let now = self.now
        let store = InMemoryAgendaStore(
            lists: Self.lists, calendars: Self.calendars, reminders: reminders, events: events,
            now: { now })
        container.agendaStore = store
        container.agenda = Agenda(store: store, now: { now })
        await container.agenda.refresh()
        container.settingsManager.companionHeartbeatEnabled = true

        let stateURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("today-gallery-\(UUID().uuidString).json")
        DayStateStore(url: stateURL).save(state)
        defer { try? FileManager.default.removeItem(at: stateURL) }
        container.companionRuntime = CompanionRuntime(
            settings: container.settingsManager, agenda: container.agenda,
            notifier: container.companionNotifier, trace: container.companionTrace,
            idleMonitor: container.idleMonitor, presence: container.companionPresence,
            thread: container.dayThread, stateStore: DayStateStore(url: stateURL),
            frontmost: container.frontmostApp, power: container.powerMonitor,
            delivery: CompanionDelivery(), profile: container.profileStore, now: { now })
        if self == .evening {
            container.profileStore.propose(
                [
                    ProposalDraft(
                        text: "You walk in the park on Sundays when the weather is good.",
                        reason: "Two Sundays in a row, a long walk lifted your mood.")
                ], source: "night-reflection")
        }
        return container
    }
}
