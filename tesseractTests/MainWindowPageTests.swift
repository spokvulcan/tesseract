//
//  MainWindowPageTests.swift
//  tesseractTests
//
//  Opens every page of the main window the way the app does: a real
//  `DependencyContainer`, `ContentView` with the page selected, laid out in a
//  window, so each page's views run against exactly the dependencies the app
//  injects for that page. The Speech page once shipped reading
//  `ModelDownloadManager` from the environment when nothing injected it for
//  that page, and crashed on open. This suite is the guard for that mistake on
//  every page. The Settings window's root is here too; the Welcome window is
//  not, because its first chapter starts downloading models.
//
//  How it fails: a missing `@EnvironmentObject` or `@Environment(Type.self)` is
//  a fatal trap inside SwiftUI, not a thrown error, so a regression never
//  fails an expectation here. It crashes the test host. xcodebuild reports the
//  crash against the page's test case ("Crash: Tesseract Agent at ..."),
//  restarts the host for the tests that are left, and skips the pages after
//  the crashed one in that run. The crash report's stack names the page's view
//  and the environment lookup that failed.
//
//  Building a container in a test is safe because a test run never reaches
//  the owner's data (ADR-0073).
//

import AppKit
import SwiftUI
import Testing

@testable import Tesseract_Agent

@MainActor
@Suite(.serialized)
struct MainWindowPageTests {

    /// Each page's navigation title. The window showing it proves the page's
    /// own body ran, not just the split view around it. A new page fails to
    /// compile here until the suite covers it.
    private static func title(of page: NavigationItem) -> String {
        switch page {
        case .today: "Today"
        case .dictation: "Dictation"
        case .speech: "Speech"
        case .agent: "Agent"
        case .serverActivity: "Activity"
        case .serverCache: "Cache"
        case .model: "Models"
        }
    }

    /// Host `view` in a real window, lay it out, and let the run loop deliver
    /// `onAppear`, so every body and appear hook on screen runs.
    private func render(_ view: some View) async throws -> NSWindow {
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1200, height: 800),
            styleMask: [.titled, .resizable],
            backing: .buffered,
            defer: false
        )
        window.isReleasedWhenClosed = false
        window.contentView = NSHostingView(rootView: view)
        window.layoutIfNeeded()
        try await Task.sleep(for: .milliseconds(50))
        window.layoutIfNeeded()
        return window
    }

    @Test(arguments: NavigationItem.allCases)
    func pageOpensWithTheAppsWiring(_ page: NavigationItem) async throws {
        let container = DependencyContainer()
        let window = try await render(
            ContentView(container: container, selectedNavigation: .constant(page)))
        defer { window.close() }
        #expect(window.title == Self.title(of: page))
    }

    @Test func settingsWindowOpensWithTheAppsWiring() async throws {
        let container = DependencyContainer()
        // As the app's Settings scene builds it. Only the first pane renders,
        // because the tab view has no selection to drive.
        let window = try await render(
            SettingsWindowView().injectDependencies(from: container))
        defer { window.close() }
    }
}
