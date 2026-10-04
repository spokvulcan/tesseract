//
//  DictationPageTests.swift
//  tesseractTests
//
//  Crash and wiring guards for the Dictation page (PRD #612): the catch
//  record and the history sheet behind its toolbar, each hosted in a window
//  the way `MainWindowPageTests` hosts a page, with the app's own wiring
//  from a real `DependencyContainer` (core and dictation scopes, as
//  `ContentView` injects them). The Learned Words, Correction Pairs and
//  history are seeded in a temporary directory and put closer to the view
//  than the container's, so they are the stores the page reads.
//
//  They check that each state renders, not what it looks like: an empty
//  page, a page with Learned Words (one forgotten, one left alone in an
//  app, one with its before-and-after example), fixes and today's takes
//  with their catches, and the history sheet with and without takes. A
//  missing environment dependency is a fatal trap inside SwiftUI, so a
//  regression crashes the test host on that case rather than failing an
//  expectation (see `MainWindowPageTests`).
//

import AppKit
import SwiftUI
import Testing

@testable import Tesseract_Agent

@MainActor
@Suite(.serialized)
struct DictationPageTests {

    private struct Stores {
        let directory: URL
        let words: LearnedWordStore
        let pairs: CorrectionPairStore
        let history: TranscriptionHistory
    }

    private static let notes = TranscriptionEntry.App(bundleID: "com.apple.Notes", name: "Notes")
    private static let terminal = TranscriptionEntry.App(
        bundleID: "com.apple.Terminal", name: "Terminal")

    private func makeStores() -> Stores {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("dictation-page-tests-\(UUID().uuidString)", isDirectory: true)
        return Stores(
            directory: directory, words: LearnedWordStore(directory: directory),
            pairs: CorrectionPairStore(directory: directory),
            history: TranscriptionHistory(directory: directory))
    }

    private func pair(
        _ raw: String, fixes: [(heard: String, meant: String)], how: CorrectionPair.Fix.How
    ) -> CorrectionPair {
        CorrectionPair(
            rawASR: raw, cleaned: raw, verdict: .skipped, committed: raw,
            fixes: fixes.map {
                CorrectionPair.Fix(heard: $0.heard, meant: $0.meant, how: how, app: nil, at: Date())
            },
            conditions: .init(duration: 3, language: "en", asrModel: "test"))
    }

    /// Three Learned Words taught by fixes in two takes: "Claude Code" with
    /// its before-and-after example, "Tesseract" left alone in Terminal, and
    /// "worktree" forgotten. Today's two takes carry what the words caught,
    /// and an older take sits in the history behind them.
    private func seed(_ stores: Stores) throws {
        stores.history.add(
            TranscriptionEntry(
                text: "An older take.", timestamp: Date().addingTimeInterval(-3 * 86_400),
                duration: 2, model: "test", app: Self.notes))

        let first = pair(
            "Ask cloud code to open the sract repo.",
            fixes: [("cloud code", "Claude Code"), ("sract", "Tesseract")], how: .afterPaste)
        let second = pair(
            "Check the work tree first.", fixes: [("work tree", "worktree")], how: .page)
        stores.pairs.record(first)
        stores.pairs.record(second)

        let claudeCode = stores.words.learn(
            heard: "cloud code", meant: "Claude Code", pairID: first.id,
            example: LearnedWord.Example(
                before: "Ask cloud code to open …", after: "Ask Claude Code to open …"))
        let tesseract = stores.words.learn(heard: "sract", meant: "Tesseract", pairID: first.id)
        let worktree = stores.words.learn(
            heard: "work tree", meant: "worktree", pairID: second.id)
        try #require(claudeCode != nil)
        let tesseractID = try #require(tesseract?.wordID)
        let worktreeID = try #require(worktree?.wordID)

        for (raw, pairID, app) in [
            ("Ask cloud code to open the sract repo.", first.id, Self.notes),
            ("Check the work tree first.", second.id, Self.terminal),
        ] {
            let taken = stores.words.apply(to: raw, appBundleID: app.bundleID)
            stores.words.recordCatches(taken.catches)
            stores.history.add(
                text: taken.text, duration: 3, model: "test", pairID: pairID,
                catches: taken.catches, app: app)
        }

        let leftAlone = stores.words.leaveAlone(
            tesseractID, in: LearnedWord.App(bundleID: "com.apple.Terminal", name: "Terminal"))
        let forgotten = stores.words.forget(worktreeID)
        try #require(leftAlone != nil)
        try #require(forgotten != nil)
    }

    /// Host `view` in a real window, lay it out, and let the run loop deliver
    /// `onAppear`, as `MainWindowPageTests.render` does.
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

    /// The page as `ContentView` hosts it, reading the seeded stores.
    private func page(_ stores: Stores, container: DependencyContainer) -> some View {
        DictationContentView()
            .environment(stores.history)
            .environment(stores.pairs)
            .environment(stores.words)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .injectDictationDependencies(from: container)
            .injectCoreDependencies(from: container)
    }

    /// The history sheet as the page presents it, reading the seeded stores.
    private func historySheet(_ stores: Stores, container: DependencyContainer) -> some View {
        TranscriptionHistorySheet()
            .environment(stores.history)
            .environment(stores.pairs)
            .environment(stores.words)
            .injectDictationDependencies(from: container)
            .injectCoreDependencies(from: container)
    }

    // MARK: - The page

    @Test func theEmptyPageOpens() async throws {
        let stores = makeStores()
        defer { try? FileManager.default.removeItem(at: stores.directory) }
        let container = DependencyContainer()

        let window = try await render(page(stores, container: container))
        defer { window.close() }

        #expect(window.title == "Dictation")
    }

    @Test func thePageOpensWithLearnedWordsFixesAndTodaysTakes() async throws {
        let stores = makeStores()
        defer { try? FileManager.default.removeItem(at: stores.directory) }
        try seed(stores)
        // The seed reaches every part of the record: tiles (the forgotten
        // word left out), the week's catches and fixes, and today's takes.
        let record = CatchRecord(
            words: stores.words.words, pairs: stores.pairs.pairs,
            entries: stores.history.entries, now: Date())
        #expect(record.tiles.map(\.meant) == ["Tesseract", "Claude Code"])
        #expect(record.tiles.first?.leftAloneIn == ["Terminal"])
        #expect(record.tiles.last?.example != nil)
        #expect(record.caughtThisWeek == 2)
        #expect(record.fixesThisWeek == 3)
        #expect(record.today.map(\.catches.count).sorted() == [1, 2])
        let container = DependencyContainer()

        let window = try await render(page(stores, container: container))
        defer { window.close() }

        #expect(window.title == "Dictation")
    }

    // MARK: - The history sheet

    @Test func theHistorySheetOpensWithTakes() async throws {
        let stores = makeStores()
        defer { try? FileManager.default.removeItem(at: stores.directory) }
        try seed(stores)
        #expect(stores.history.entries.count == 3)
        let container = DependencyContainer()

        let window = try await render(historySheet(stores, container: container))
        window.close()
    }

    @Test func theEmptyHistorySheetOpens() async throws {
        let stores = makeStores()
        defer { try? FileManager.default.removeItem(at: stores.directory) }
        let container = DependencyContainer()

        let window = try await render(historySheet(stores, container: container))
        window.close()
    }
}
