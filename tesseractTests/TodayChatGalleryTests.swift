//
//  TodayChatGalleryTests.swift
//  tesseractTests
//
//  Today's Chat, rendered with the app's wiring over the full fixture day's
//  Day Thread: the Day Opening at the head, the Morning Plan, a word with
//  Jarvis, a Welcome back with Anna still waiting, a Triage that raised
//  nothing (no row), the Evening Wrap-up with two leftovers to settle, the
//  Night Reflection with a task and a fact to decide, and a last message
//  with two pictures — at a regular and a phone width. Then the same day
//  once everything is settled: the cards close and lose their accent edge.
//  With TODAY_GALLERY_DIR set (TEST_RUNNER_TODAY_GALLERY_DIR through
//  xcodebuild), each render is also written there as a PNG, in dark and
//  light, for judging the chat by eye.
//

import AppKit
import Foundation
import SwiftUI
import Testing

@testable import Tesseract_Agent

@MainActor
@Suite(.serialized)
struct TodayChatGalleryTests {

    /// About as tall as the thread, which sits at the bottom: a page much
    /// taller than its content comes out tiled in the offscreen render.
    static let widths: [(name: String, width: CGFloat, height: CGFloat)] = [
        ("regular", 900, 2200), ("phone", 390, 2510),
    ]

    @Test func theDayThreadRendersAtEveryWidth() async throws {
        for size in Self.widths {
            for dark in TodayGalleryTests.directory == nil ? [true] : [true, false] {
                let image = try await render(width: size.width, height: size.height, dark: dark)
                #expect(image.pixelsWide > 0)
                try write(image, "chat-\(size.name)-\(dark ? "dark" : "light").png")
            }
        }
    }

    /// Everything settled: the leftovers moved, Anna answered, the task and
    /// the fact decided. The cards close on their own.
    @Test func aSettledDayReadsQuiet() async throws {
        for dark in TodayGalleryTests.directory == nil ? [true] : [true, false] {
            let image = try await render(
                width: 900, height: 1340, dark: dark,
                settle: { container in
                    let runtime = container.companionRuntime
                    runtime.act(.allLeftovers(cardID: Self.wrapUpID))
                    runtime.act(.itemDone(cardID: Self.breakpointID, itemID: "n1"))
                    runtime.act(.taskProposal(id: "task-1", add: false))
                    for fact in container.profileStore.openProposals {
                        container.profileStore.decide(fact.id, keep: true)
                    }
                })
            #expect(image.pixelsWide > 0)
            try write(image, "chat-settled-\(dark ? "dark" : "light").png")
        }
    }

    private func write(_ image: NSBitmapImageRep, _ name: String) throws {
        guard let directory = TodayGalleryTests.directory else { return }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try #require(image.representation(using: .png, properties: [:]))
            .write(to: directory.appendingPathComponent(name))
    }

    private func render(
        width: CGFloat, height: CGFloat, dark: Bool,
        settle: (DependencyContainer) async throws -> Void = { _ in }
    ) async throws -> NSBitmapImageRep {
        let fixture = TodayFixture.fullDay
        let container = try await fixture.container()
        let thread = container.dayThread
        thread.show(day: DayKey(for: fixture.now))
        #expect(thread.chat.appendCommitted(Self.turns(agenda: container.agenda)))
        try await settle(container)
        // The runtime decides its signals in turn: let the actions land.
        try await Task.sleep(for: .milliseconds(200))

        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: width, height: height),
            styleMask: [.borderless], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
        defer { window.close() }
        let host = NSHostingView(
            rootView: TodayView(mode: .chat)
                .injectCompanionDependencies(from: container)
                .frame(width: width, height: height)
                .background(Color(nsColor: .windowBackgroundColor)))
        window.contentView = host
        window.layoutIfNeeded()
        try await TodayGalleryTests.settle(.milliseconds(400))
        window.layoutIfNeeded()
        let rep = try #require(host.bitmapImageRepForCachingDisplay(in: host.bounds))
        host.cacheDisplay(in: host.bounds, to: rep)
        return rep
    }

    // MARK: - The day's thread

    static let breakpointID = "breakpoint-2026-10-04-1"
    static let wrapUpID = "eveningWrapUp-2026-10-04-2"

    static func at(_ hour: Int, _ minute: Int = 0) -> Date { TodayFixture.at(hour, minute) }

    /// The thread's turns, as the day appended them.
    static func turns(agenda: Agenda) -> [any AgentMessageProtocol & Sendable] {
        let snapshot = agenda.snapshot
        let open =
            snapshot.open
            + snapshot.doneToday.map { done in
                var reminder = done
                reminder.isCompleted = false
                return reminder
            }
        let morning = DayFacts(
            now: at(7, 42), events: snapshot.events,
            dueOrOverdue: open.filter { ($0.due ?? .distantFuture) < TodayFixture.at(0, day: 5) },
            undated: open.filter { $0.due == nil }, areas: agenda.areas,
            inboxListID: agenda.inbox?.id, weekFocus: "Ship the Companion")
        let opening = MomentPrompts.dayOpening(
            facts: morning,
            profile: [
                "Lives in Reykjavík and walks to work when it isn't raining.",
                "Learns Ukrainian on Duolingo every evening.",
            ],
            carryOver:
                "Saturday was a rest day, and that's fine. Today the Companion ships: the cache ADR is the one thing that has to move, and the evening is yours once it does.",
            draft: [
                "Mail and the PR review before standup",
                "The ADR in the quiet stretch after lunch",
                "Duolingo at 21:30",
            ])

        func moment(_ text: String, _ at: Date, reply: String, thinking: String)
            -> [any AgentMessageProtocol & Sendable]
        {
            [
                UserMessage(content: text, timestamp: at, turnOrigin: .moment),
                AssistantMessage(
                    content: reply, thinking: thinking, timestamp: at.addingTimeInterval(40)),
            ]
        }

        var turns: [any AgentMessageProtocol & Sendable] = [
            UserMessage(content: opening, timestamp: at(7, 42), turnOrigin: .moment)
        ]
        turns += moment(
            MomentPrompts.morningPlan(facts: morning), at(8, 18),
            reply: """
                {"line": "A light morning: two quick things before standup, and the ADR gets the quiet stretch after lunch.", "must_do": "adr", "plan": [{"id": "mail", "at": "08:30", "minutes": 20}, {"id": "pr", "at": "08:55", "minutes": 30}, {"id": "adr", "at": "13:45", "minutes": 75}], "suggestions": ["Start with mail while the coffee's hot.", "Rent takes two minutes; leave it for the evening."]}
                """,
            thinking: "Standup at 9:30, lunch with Sam at 12:30, design review at 15:00.")
        turns += [
            UserMessage(
                content: "Is there room for a walk at lunch, or is that pushing it?",
                timestamp: at(9, 5)),
            AssistantMessage(
                content:
                    "There's room. Lunch with Sam runs 12:30–13:30 at Café Nord, and the ADR doesn't start until 13:45 — walk there and back and you've had your air without touching the plan.",
                thinking: "Lunch is at Café Nord. The ADR slot starts 13:45.",
                timestamp: at(9, 5)),
        ]
        turns += moment(
            """
            [Welcome back — away 45 min, 13:20–14:05]
            Notifications they haven't seen (id · app — text):
            - n1 · Telegram — Anna: Can you send me the deck before 4?
            - n2 · Mail — Your order shipped
            - n3 · Mail — Weekly digest
            Next: 15:00 Design review
            Where they were: Xcode
            """, at(14, 5),
            reply:
                #"{"line": "Welcome back — Anna is waiting on the deck before four; the rest can wait.", "needs_you": ["n1"]}"#,
            thinking: "Anna's message has a deadline. The mail can wait.")
        turns += moment(
            """
            [Triage] New notifications while the owner works (id · app — text):
            - n1 · Mail — Your package is out for delivery
            """, at(15, 40),
            reply: #"{"line": "", "raise": []}"#, thinking: "A delivery notice can wait.")
        turns += moment(
            """
            [Evening Wrap-up]
            Done today: Answer mail; Review PR #612; Implement the Companion with a cloud model, to make it useful
            Still open from today (id — title):
            - adr — Write the cache ADR
            - rent — Pay rent
            Tomorrow starts with: 09:00 Work
            """, at(21),
            reply:
                #"{"line": "The Companion shipped and the streak held — a real day. The ADR gets a clean start tomorrow, first thing.", "leftovers": [{"id": "adr", "suggest": "tomorrow"}, {"id": "rent", "suggest": "later"}]}"#,
            thinking: "Three done, the must-do slid. Suggest tomorrow for the ADR.")
        turns += moment(
            """
            [Night Reflection]
            Look back over today's thread.
            """, at(21, 50),
            reply: """
                {"carry_over": "\(TodayFixture.fullDayReflection.carryOver)", "tomorrow": ["Work from nine; the ADR first, while it's quiet", "ID check at two — bring the passport", "Class at four"], "proposals": [{"text": "You walk in the park on Sundays when the weather is good.", "reason": "Two Sundays in a row, a long walk lifted your mood."}], "tasks": [{"title": "Send the case worker the bus receipts", "when": "tomorrow"}]}
                """,
            thinking: "The day went well. One thing they promised: the receipts.")
        turns += [
            UserMessage(
                content:
                    "That's what I did today. Can you add this to my memories somewhere in the calendar? What do you think about my hike?",
                images: [
                    ImageAttachment(
                        data: ImageTestFixtures.flyerPNG(
                            title: "Outdoor Walk", line: "11,52 km · 4:07:33", hue: 0.42),
                        mimeType: "image/png"),
                    ImageAttachment(
                        data: ImageTestFixtures.flyerPNG(
                            title: "Esja ridge", line: "Sun 4 Oct · 13:49", hue: 0.6),
                        mimeType: "image/png"),
                ], timestamp: at(22, 10)),
            AssistantMessage(
                content:
                    "Saved it to your calendar — a 13:49–17:57 record of the day's hike, so it lives where your future self will find it.\n\nAnd honestly? 11.5 km and 660 m of climbing in four hours is a real day out, not a walk.",
                thinking: "A 4-hour hike, 11.52 km, 660 m up.",
                timestamp: at(22, 11)),
        ]
        return turns
    }
}

// MARK: - The full day's state

extension TodayFixture {

    static let fullDayReflection = ReflectionCard(
        carryOver:
            "A full Sunday: the Companion shipped and works, the Duolingo streak held, and a long walk lifted the whole evening. Tomorrow is a real Monday — work in the morning, the ID check at two, class at four — but the hardest part of the week is behind you.",
        tomorrow: [
            "Work from nine; the ADR first, while it's quiet",
            "ID check at two — bring the passport",
            "Class at four",
        ],
        proposals: [
            ProposalDraft(
                text: "You walk in the park on Sundays when the weather is good.",
                reason: "Two Sundays in a row, a long walk lifted your mood.")
        ],
        tasks: [
            TaskProposal(
                id: "task-1", title: "Send the case worker the bus receipts",
                due: Self.at(0, day: 5))
        ])

    /// Every moment's card, as the engine left them by 22:20.
    static func fullDayState(_ empty: DayState) -> DayState {
        var state = empty
        state.morningPlanAt = at(8, 18)
        state.eveningWrapUpAt = at(21)
        state.nightReflectionAt = at(21, 51)
        state.mustDoID = "adr"
        state.weekFocus = "Ship the Companion"
        state.weekFocusSetAt = at(9, day: 3)
        state.plan = [
            Step("mail", at: at(8, 30), minutes: 20),
            Step("pr", at: at(8, 55), minutes: 30),
            Step("adr", at: at(13, 45), minutes: 75),
        ]
        state.taskProposals = fullDayReflection.tasks
        let day = state.day.rawValue
        state.cards = [
            DayCard(
                id: "morningPlan-\(day)-0", kind: .morningPlan, createdAt: at(8, 18),
                isFallback: false,
                body: .morningPlan(
                    MorningPlanCard(
                        line:
                            "A light morning: two quick things before standup, and the ADR gets the quiet stretch after lunch.",
                        mustDoID: "adr", placements: state.plan.map(\.slot),
                        suggestions: [
                            "Start with mail while the coffee's hot.",
                            "Rent takes two minutes; leave it for the evening.",
                        ])),
                kept: true),
            DayCard(
                id: "breakpoint-\(day)-1", kind: .breakpoint, createdAt: at(14, 5),
                isFallback: false,
                body: .breakpoint(
                    BreakpointCard(
                        awayFrom: at(13, 20), awayUntil: at(14, 5),
                        line:
                            "Welcome back — Anna is waiting on the deck before four; the rest can wait.",
                        needsYou: [
                            WaitingItem(
                                id: "n1", kind: .notification, title: "Anna · Telegram",
                                detail: "Can you send me the deck before 4?", app: "Telegram")
                        ],
                        next: [
                            NextItem(
                                id: "event:d", kind: .event, title: "Design review",
                                at: at(15), minutes: 60)
                        ],
                        whereYouWere: "Xcode",
                        canWait: [
                            QuietGroup(app: "Mail", lines: ["Your order shipped", "Weekly digest"])
                        ])),
                kept: true),
            DayCard(
                id: "eveningWrapUp-\(day)-2", kind: .eveningWrapUp,
                createdAt: at(21).addingTimeInterval(40), isFallback: false,
                body: .eveningWrapUp(
                    EveningWrapUpCard(
                        line:
                            "The Companion shipped and the streak held — a real day. The ADR gets a clean start tomorrow, first thing.",
                        done: [
                            "Answer mail", "Review PR #612",
                            "Implement the Companion with a cloud model, to make it useful",
                            "Пройти урок у Duolingo 💚",
                        ],
                        leftovers: [
                            Leftover(
                                reminderID: "adr", title: "Write the cache ADR",
                                suggestion: .tomorrow),
                            Leftover(reminderID: "rent", title: "Pay rent", suggestion: .later),
                        ],
                        tomorrowFirst: "09:00 Work")),
                kept: true),
            DayCard(
                id: "nightReflection-\(day)-3", kind: .nightReflection,
                createdAt: at(21, 51), isFallback: false, body: .reflection(fullDayReflection)),
        ]
        return state
    }
}
