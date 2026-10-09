//
//  JarvisPanelGalleryTests.swift
//  tesseractTests
//
//  The Jarvis Panel's content, rendered with the app's view over fixture
//  cards: a Step Cue (and one for the must-do, with a long title) at its own
//  shorter height, and a Breakpoint card at the full one. With
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
        case breakpoint

        var testDescription: String { rawValue }

        static func at(_ hour: Int, _ minute: Int = 0) -> Date {
            Calendar.current.date(
                from: DateComponents(year: 2026, month: 10, day: 9, hour: hour, minute: minute))!
        }

        @MainActor var height: CGFloat {
            switch self {
            case .stepCue, .stepCueMustDo: JarvisPanelController.cueHeight
            case .breakpoint: JarvisPanelController.size.height
            }
        }

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

    private func render(_ shown: Shown, dark: Bool) async throws -> NSBitmapImageRep {
        let container = DependencyContainer()
        let model = JarvisPanelModel()
        shown.fill(model)
        let size = NSSize(width: JarvisPanelController.size.width, height: shown.height)
        let window = NSWindow(
            contentRect: NSRect(origin: .zero, size: size), styleMask: [.borderless],
            backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
        defer { window.close() }
        let host = NSHostingView(
            rootView: JarvisPanelView(
                model: model, thread: container.dayThread, close: {}, expand: {}, act: { _ in },
                choose: { _ in }, send: {}, capture: {}, mic: {}
            )
            .frame(width: size.width, height: size.height)
            .background(
                Color(nsColor: .windowBackgroundColor),
                in: RoundedRectangle(cornerRadius: 28, style: .continuous)))
        window.contentView = host
        window.layoutIfNeeded()
        try await Task.sleep(for: .milliseconds(300))
        window.layoutIfNeeded()
        let rep = try #require(host.bitmapImageRepForCachingDisplay(in: host.bounds))
        host.cacheDisplay(in: host.bounds, to: rep)
        return rep
    }
}
