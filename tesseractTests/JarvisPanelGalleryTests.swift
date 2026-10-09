//
//  JarvisPanelGalleryTests.swift
//  tesseractTests
//
//  The Jarvis Panel's content, rendered with the app's view over fixture
//  cards — a Step Cue (one for the must-do, with a long title, one at the
//  end of a started step, both shown late, the owner back from away, the
//  word a step marked done gets, and the five-minute start offered once a
//  step was put off twice, with its check-in), the Morning Plan with its
//  steps over the
//  Today gallery's morning, the Evening Wrap-up with its leftovers (and on
//  the week's last day, with one that has waited since last week), and a
//  Breakpoint card — each at the height the panel fits to it. With
//  PANEL_GALLERY_DIR set (TEST_RUNNER_PANEL_GALLERY_DIR through xcodebuild),
//  each render is also written there as a PNG, in dark and light, for judging
//  the panel by eye. The glass itself is the window's; a plain background
//  stands in for it here.
//

import AppKit
import Foundation
import SwiftUI
import Testing

@testable import Tesseract_Agent

@MainActor
@Suite(.serialized)
struct JarvisPanelGalleryTests {

    static let directory = ProcessInfo.processInfo.environment["PANEL_GALLERY_DIR"].map {
        URL(fileURLWithPath: $0, isDirectory: true)
    }

    enum Shown: String, CaseIterable, CustomTestStringConvertible {
        case stepCue
        case stepCueMustDo
        case stepCheckIn
        case stepCueLate
        case stepCheckInLate
        case stepDone
        case stepCueSmall
        case stepCheckInSmall
        case morningPlan
        case eveningWrapUp
        case eveningWrapUpWeek
        case breakpoint

        var testDescription: String { rawValue }

        static func at(_ hour: Int, _ minute: Int = 0) -> Date {
            Calendar.current.date(
                from: DateComponents(year: 2026, month: 10, day: 9, hour: hour, minute: minute))!
        }

        /// The morning plan reads its steps from the Today gallery's morning.
        @MainActor func container() async throws -> DependencyContainer {
            if self == .morningPlan { return try await TodayFixture.morning.container() }
            return DependencyContainer()
        }

        var now: Date { self == .morningPlan ? TodayFixture.morning.now : Self.at(12) }

        @MainActor func fill(_ model: JarvisPanelModel) {
            switch self {
            case .stepCue:
                model.cue = StepCue(
                    reminderID: "letter", title: "Write to the case worker about the bus ticket",
                    start: Self.at(11, 10), minutes: 20, areaName: "Inbox", isMustDo: false,
                    next: "Companion work at 11:35")
            case .stepCueMustDo:
                model.cue = StepCue(
                    reminderID: "companion",
                    title:
                        "Implement the Companion with a cloud model, so it is actually useful every day",
                    start: Self.at(14), minutes: 90, areaName: "Daily", isMustDo: true,
                    next: nil)
            case .stepCheckIn:
                model.cue = StepCue(
                    reminderID: "letter", title: "Write to the case worker about the bus ticket",
                    start: Self.at(11, 12), minutes: 20, areaName: "Inbox", isMustDo: false,
                    next: "Companion work at 11:35", phase: .end)
            case .stepCueLate:
                model.cue = StepCue(
                    reminderID: "letter", title: "Write to the case worker about the bus ticket",
                    start: Self.at(11, 10), minutes: 40, areaName: "Inbox", isMustDo: false,
                    next: "Companion work at 11:55", late: true)
            case .stepCheckInLate:
                model.cue = StepCue(
                    reminderID: "letter", title: "Write to the case worker about the bus ticket",
                    start: Self.at(11, 12), minutes: 20, areaName: "Inbox", isMustDo: false,
                    next: nil, phase: .end, late: true)
            case .stepDone:
                model.done = StepCue(
                    reminderID: "adr", title: "Write the cache ADR", start: Self.at(13, 45),
                    minutes: 75, areaName: "Work", isMustDo: true,
                    next: "Design review at 15:00", phase: .end)
            case .stepCueSmall:
                model.cue = StepCue(
                    reminderID: "letter", title: "Write to the case worker about the bus ticket",
                    start: Self.at(11, 40), minutes: 20, areaName: "Inbox", isMustDo: false,
                    next: nil, putOff: 2)
            case .stepCheckInSmall:
                model.cue = StepCue(
                    reminderID: "letter", title: "Write to the case worker about the bus ticket",
                    start: Self.at(11, 41), minutes: 5, areaName: "Inbox", isMustDo: false,
                    next: nil, phase: .end, small: true)
            case .morningPlan:
                model.card = TodayFixture.morning.state.cards.first
            case .eveningWrapUp:
                model.card = DayCard(
                    id: "eveningWrapUp-1", kind: .eveningWrapUp, createdAt: Self.at(21),
                    isFallback: false,
                    body: .eveningWrapUp(
                        EveningWrapUpCard(
                            line:
                                "The letter is out, a school is chosen and the streak held — a tidy day.",
                            done: [
                                "Write to the case worker", "Choose the language school",
                                "Duolingo lesson", "Answer mail",
                            ],
                            leftovers: [
                                Leftover(
                                    reminderID: "companion", title: "Companion work, cloud model",
                                    suggestion: .tomorrow),
                                Leftover(
                                    reminderID: "chair", title: "Find a used work chair",
                                    suggestion: .later),
                            ],
                            tomorrowFirst: "07:30 All Hands")))
            case .eveningWrapUpWeek:
                let lastWeek = Calendar.current.date(
                    from: DateComponents(year: 2026, month: 10, day: 1))!
                model.card = DayCard(
                    id: "eveningWrapUp-2", kind: .eveningWrapUp, createdAt: Self.at(21),
                    isFallback: false,
                    body: .eveningWrapUp(
                        EveningWrapUpCard(
                            line: "A full week: the Companion shipped and the streak held.",
                            done: ["Ship the Companion", "Duolingo lesson"],
                            leftovers: [
                                Leftover(
                                    reminderID: "invoice", title: "Send the invoice",
                                    suggestion: .tomorrow),
                                Leftover(
                                    reminderID: "passport", title: "Renew passport",
                                    suggestion: .later, since: lastWeek),
                            ],
                            tomorrowFirst: "07:30 Put the bins out",
                            week: "Twelve things done, most of them for work.",
                            focus: "The job search")))
            case .breakpoint:
                model.card = DayCard(
                    id: "breakpoint-1", kind: .breakpoint, createdAt: Self.at(15, 44),
                    isFallback: false,
                    body: .breakpoint(
                        BreakpointCard(
                            awayFrom: Self.at(15, 6), awayUntil: Self.at(15, 44),
                            line: "Welcome back — Anna is still waiting on the deck.",
                            needsYou: [
                                WaitingItem(
                                    id: "n1", kind: .notification, title: "Anna · Slack",
                                    detail: "Can you send me the deck before 4?", app: "Slack")
                            ],
                            next: [
                                NextItem(
                                    id: "event:review", kind: .event, title: "Design review",
                                    at: Self.at(16), minutes: 60)
                            ],
                            whereYouWere: "Xcode",
                            canWait: [QuietGroup(app: "GitHub", lines: ["CI passed on main"])])))
            }
        }
    }

    @Test(arguments: Shown.allCases)
    func panelRenders(_ shown: Shown) async throws {
        for dark in Self.directory == nil ? [true] : [true, false] {
            let image = try await render(shown, dark: dark)
            #expect(image.pixelsWide > 0)
            if let directory = Self.directory {
                try FileManager.default.createDirectory(
                    at: directory, withIntermediateDirectories: true)
                let name = "\(shown.rawValue)-\(dark ? "dark" : "light").png"
                try #require(image.representation(using: .png, properties: [:]))
                    .write(to: directory.appendingPathComponent(name))
            }
        }
    }

    @Test func aStepMarkedDoneIsCreditedAndPointsOn() {
        var cue = StepCue(
            reminderID: "letter", title: "Write to the case worker", start: Shown.at(11, 10),
            minutes: 20, areaName: "Inbox", isMustDo: false, next: "Design review at 15:00")
        #expect(cue.doneLine == "Next: Design review at 15:00.")
        cue.isMustDo = true
        #expect(
            cue.doneLine
                == "That's the must-do — the rest is a bonus.\nNext: Design review at 15:00.")
        cue.next = nil
        #expect(cue.doneLine == "That's the must-do — the rest is a bonus.")
        cue.isMustDo = false
        #expect(cue.doneLine == nil)
    }

    @Test func thePanelFitsWhatItSays() {
        #expect(JarvisPanelController.height(forContent: 40) == JarvisPanelController.minimumHeight)
        #expect(JarvisPanelController.height(forContent: 200) == 324)
        #expect(
            JarvisPanelController.height(forContent: 2000) == JarvisPanelController.size.height)
    }

    @Test func thePlansStepsAreTheTasksItPlacedAmongTheEvents() async throws {
        let container = try await TodayFixture.morning.container()
        let card = try #require(TodayFixture.morning.state.cards.first)
        guard case .morningPlan(let plan) = card.body else {
            Issue.record("expected a Morning Plan card")
            return
        }
        let steps = PlanStep.ahead(
            plan: plan, agenda: container.agenda, now: TodayFixture.morning.now)
        #expect(
            steps.map(\.title) == [
                "Answer mail", "Review PR #612", "Standup", "Lunch with Sam", "Write the cache ADR",
            ])
        #expect(steps.last?.kind == .task(isMustDo: true))
    }

    /// Lay the panel out once to learn its height, as the panel does, then
    /// draw it at that height.
    private func render(_ shown: Shown, dark: Bool) async throws -> NSBitmapImageRep {
        let container = try await shown.container()
        let model = JarvisPanelModel()
        shown.fill(model)
        var content: CGFloat = 0
        let window = NSWindow(
            contentRect: NSRect(origin: .zero, size: JarvisPanelController.size),
            styleMask: [.borderless], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
        defer { window.close() }
        func host(height: CGFloat) -> NSHostingView<some View> {
            NSHostingView(
                rootView: JarvisPanelView(
                    model: model, thread: container.dayThread, agenda: container.agenda,
                    liveCard: { _ in nil }, close: {}, expand: {}, act: { _ in }, keep: {},
                    choose: { _ in }, send: {}, capture: {}, mic: {},
                    onContentHeight: { content = $0 }, now: { shown.now }
                )
                .frame(width: JarvisPanelController.size.width, height: height)
                .background(
                    Color(nsColor: .windowBackgroundColor),
                    in: RoundedRectangle(cornerRadius: 28, style: .continuous)))
        }
        window.contentView = host(height: JarvisPanelController.size.height)
        window.layoutIfNeeded()
        try await Task.sleep(for: .milliseconds(200))
        let height = JarvisPanelController.height(forContent: content)
        window.setContentSize(NSSize(width: JarvisPanelController.size.width, height: height))
        let fitted = host(height: height)
        window.contentView = fitted
        window.layoutIfNeeded()
        try await Task.sleep(for: .milliseconds(200))
        window.layoutIfNeeded()
        let rep = try #require(fitted.bitmapImageRepForCachingDisplay(in: fitted.bounds))
        fitted.cacheDisplay(in: fitted.bounds, to: rep)
        return rep
    }
}
