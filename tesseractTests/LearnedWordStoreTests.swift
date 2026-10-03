//
//  LearnedWordStoreTests.swift
//  tesseractTests
//
//  The **Learned Word** store (PRD #612) against a temporary directory and
//  an injected clock: learning (new word, new form, a form moving to the
//  latest fix, a forgotten word revived), exceptions per app, Forget and
//  Undo restoring exactly, catches counted per local day, persistence
//  across a reload, and a corrupt file loading empty (and kept aside).
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct LearnedWordStoreTests {

    @MainActor
    private final class Clock {
        var now: Date
        init(_ now: Date) { self.now = now }
    }

    private static let calendar: Calendar = {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "Europe/Berlin")!
        return calendar
    }()

    /// A local time in Berlin, whole seconds so it survives the JSON round
    /// trip unchanged.
    private static func berlin(_ day: Int, _ hour: Int, _ minute: Int = 0) -> Date {
        calendar.date(
            from: DateComponents(year: 2026, month: 10, day: day, hour: hour, minute: minute))!
    }

    private static let notes = LearnedWord.App(bundleID: "com.apple.Notes", name: "Notes")
    private static let terminal = "com.apple.Terminal"

    private func makeTempDirectory() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("learned-words-tests-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func makeStore(in directory: URL, clock: Clock) -> LearnedWordStore {
        LearnedWordStore(directory: directory, now: { clock.now }, calendar: Self.calendar)
    }

    // MARK: - Learning

    @Test func learnCreatesAWordWithTheNormalizedHeardForm() throws {
        let directory = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let clock = Clock(Self.berlin(3, 10))
        let store = makeStore(in: directory, clock: clock)
        let pairID = UUID()
        let example = LearnedWord.Example(
            before: "open the SRACT repo", after: "open the Tesseract repo")

        let receipt = try #require(
            store.learn(heard: " SRACT, ", meant: " Tesseract ", pairID: pairID, example: example))

        let word = try #require(store.words.first)
        #expect(store.words.count == 1)
        #expect(receipt.wordID == word.id)
        #expect(word.meant == "Tesseract")
        #expect(word.heard == ["sract"])
        #expect(word.fixes == 1)
        #expect(word.learnedAt == clock.now)
        #expect(word.sourcePairID == pairID)
        #expect(word.example == example)
        #expect(!word.isForgotten)
        #expect(word.catchesByDay.isEmpty)
    }

    @Test func learningMultiWordFormsKeepsOneSpaceBetweenWords() throws {
        let directory = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = makeStore(in: directory, clock: Clock(Self.berlin(3, 10)))

        store.learn(heard: "D  flash two,", meant: "DFlash2")

        #expect(store.words.first?.heard == ["d flash two"])
        #expect(store.apply(to: "Use D flash two.", appBundleID: nil).text == "Use DFlash2.")
    }

    @Test func learnReturnsNilWhenThereIsNothingToLearn() {
        let directory = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = makeStore(in: directory, clock: Clock(Self.berlin(3, 10)))

        #expect(store.learn(heard: "", meant: "Claude") == nil)
        #expect(store.learn(heard: "...", meant: "Claude") == nil)
        #expect(store.learn(heard: "cloud", meant: "   ") == nil)
        #expect(store.learn(heard: "worktree", meant: "worktree") == nil)
        #expect(store.words.isEmpty)
    }

    @Test func learningTheSameMeantSpellingAddsAFormAndCountsTheFix() {
        let directory = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = makeStore(in: directory, clock: Clock(Self.berlin(3, 10)))

        store.learn(heard: "SRACT", meant: "Tesseract")
        store.learn(heard: "SRAX", meant: "Tesseract")
        #expect(store.words.count == 1)
        #expect(store.words.first?.heard == ["sract", "srax"])
        #expect(store.words.first?.fixes == 2)

        store.learn(heard: "srax", meant: "Tesseract")
        #expect(store.words.first?.heard == ["sract", "srax"])
        #expect(store.words.first?.fixes == 3)
    }

    @Test func aNewFixMovesItsWordToTheFront() {
        let directory = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = makeStore(in: directory, clock: Clock(Self.berlin(3, 10)))

        store.learn(heard: "SRACT", meant: "Tesseract")
        store.learn(heard: "cloud", meant: "Claude")
        #expect(store.words.map(\.meant) == ["Claude", "Tesseract"])

        store.learn(heard: "TSRAC", meant: "Tesseract")
        #expect(store.words.map(\.meant) == ["Tesseract", "Claude"])
    }

    @Test func aHeardFormMovesToTheLatestFixAndAWordLeftWithNoFormsIsForgotten() throws {
        let directory = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let clock = Clock(Self.berlin(3, 10))
        let store = makeStore(in: directory, clock: clock)
        store.learn(heard: "cloud", meant: "Claude")
        store.learn(heard: "clod", meant: "Claude")
        let claude = try #require(store.words.first)

        store.learn(heard: "cloud", meant: "Cloudflare")
        #expect(store.word(withID: claude.id)?.heard == ["clod"])
        #expect(store.word(withID: claude.id)?.isForgotten == false)
        #expect(store.word(heard: "cloud")?.meant == "Cloudflare")

        clock.now = Self.berlin(3, 11)
        store.learn(heard: "clod", meant: "Cloudflare")
        #expect(store.word(withID: claude.id)?.heard == [])
        #expect(store.word(withID: claude.id)?.forgottenAt == clock.now)
        #expect(store.activeWords.map(\.meant) == ["Cloudflare"])
        #expect(store.word(heard: "clod")?.meant == "Cloudflare")
    }

    @Test func aForgottenWordIsRevivedByANewFixKeepingItsExceptions() throws {
        let directory = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = makeStore(in: directory, clock: Clock(Self.berlin(3, 10)))
        let id = try #require(store.learn(heard: "cloud", meant: "Claude")?.wordID)
        store.leaveAlone(id, in: Self.notes)
        store.forget(id)
        #expect(store.activeWords.isEmpty)
        #expect(store.apply(to: "ask cloud", appBundleID: Self.terminal).text == "ask cloud")

        store.learn(heard: "Cloud", meant: "Claude")

        let word = try #require(store.word(withID: id))
        #expect(!word.isForgotten)
        #expect(word.fixes == 2)
        #expect(word.leftAloneIn == [Self.notes])
        #expect(store.apply(to: "ask cloud", appBundleID: Self.terminal).text == "ask Claude")
        #expect(store.apply(to: "ask cloud", appBundleID: Self.notes.bundleID).text == "ask cloud")
    }

    @Test func lookupsFindWordsByIDAndByNormalizedHeardForm() throws {
        let directory = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = makeStore(in: directory, clock: Clock(Self.berlin(3, 10)))
        let id = try #require(store.learn(heard: "work tree", meant: "worktree")?.wordID)

        #expect(store.word(withID: id)?.meant == "worktree")
        #expect(store.word(withID: UUID()) == nil)
        #expect(store.word(heard: "Work  Tree,")?.id == id)
        #expect(store.word(heard: "work") == nil)
    }

    // MARK: - Exceptions

    @Test func leaveAloneAddsTheAppOnceAndTheWordStopsApplyingThere() throws {
        let directory = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = makeStore(in: directory, clock: Clock(Self.berlin(3, 10)))
        let id = try #require(store.learn(heard: "cloud", meant: "Claude")?.wordID)

        let receipt = store.leaveAlone(id, in: Self.notes)
        store.leaveAlone(id, in: LearnedWord.App(bundleID: "com.apple.Notes", name: "Notes 2"))

        #expect(receipt?.wordID == id)
        #expect(store.word(withID: id)?.leftAloneIn == [Self.notes])
        #expect(
            store.apply(to: "a grey cloud", appBundleID: Self.notes.bundleID)
                == .unchanged("a grey cloud"))
        #expect(store.apply(to: "a grey cloud", appBundleID: Self.terminal).text == "a grey Claude")
        #expect(store.apply(to: "a grey cloud", appBundleID: nil).text == "a grey Claude")
        #expect(store.leaveAlone(UUID(), in: Self.notes) == nil)
    }

    @Test func applyAgainLetsTheWordApplyInTheAppAgain() throws {
        let directory = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = makeStore(in: directory, clock: Clock(Self.berlin(3, 10)))
        let id = try #require(store.learn(heard: "cloud", meant: "Claude")?.wordID)
        store.leaveAlone(id, in: Self.notes)

        store.applyAgain(id, in: Self.notes.bundleID)

        #expect(store.word(withID: id)?.leftAloneIn.isEmpty == true)
        #expect(
            store.apply(to: "a grey cloud", appBundleID: Self.notes.bundleID).text
                == "a grey Claude")
    }

    // MARK: - Forget and Undo

    @Test func forgetStopsTheWordAndUndoRestoresItExactly() throws {
        let directory = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = makeStore(in: directory, clock: Clock(Self.berlin(3, 10)))
        store.learn(heard: "SRACT", meant: "Tesseract")
        let id = try #require(store.learn(heard: "cloud", meant: "Claude")?.wordID)
        let before = store.words

        let receipt = try #require(store.forget(id))

        #expect(store.word(withID: id)?.isForgotten == true)
        #expect(store.activeWords.map(\.meant) == ["Tesseract"])
        #expect(store.word(heard: "cloud") == nil)
        #expect(store.apply(to: "ask cloud", appBundleID: nil).text == "ask cloud")
        #expect(store.forget(id) == nil)

        store.undo(receipt)

        #expect(store.words == before)
        #expect(store.apply(to: "ask cloud", appBundleID: nil).text == "ask Claude")
    }

    @Test func undoOfALearnRemovesTheNewWord() throws {
        let directory = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let clock = Clock(Self.berlin(3, 10))
        let store = makeStore(in: directory, clock: clock)
        store.learn(heard: "SRACT", meant: "Tesseract")
        let before = store.words

        let receipt = try #require(store.learn(heard: "cloud", meant: "Claude"))
        store.undo(receipt)

        #expect(store.words == before)
        #expect(makeStore(in: directory, clock: clock).words == before)
    }

    @Test func undoOfALearnRestoresAChangedWordWhereItStood() throws {
        let directory = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = makeStore(in: directory, clock: Clock(Self.berlin(3, 10)))
        store.learn(heard: "SRACT", meant: "Tesseract")
        store.learn(heard: "cloud", meant: "Claude")
        store.learn(heard: "work tree", meant: "worktree")
        let before = store.words

        let receipt = try #require(store.learn(heard: "SRAX", meant: "Tesseract"))
        #expect(store.words.first?.meant == "Tesseract")
        store.undo(receipt)

        #expect(store.words == before)
    }

    @Test func undoOfAMovedFormRestoresBothWords() throws {
        let directory = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = makeStore(in: directory, clock: Clock(Self.berlin(3, 10)))
        store.learn(heard: "cloud", meant: "Claude")
        store.learn(heard: "SRACT", meant: "Tesseract")
        let before = store.words

        let receipt = try #require(store.learn(heard: "cloud", meant: "Cloudflare"))
        #expect(store.words.contains { $0.meant == "Claude" && $0.isForgotten })
        store.undo(receipt)

        #expect(store.words == before)
        #expect(store.word(heard: "cloud")?.meant == "Claude")
    }

    @Test func undoOfLeaveAloneLetsTheWordApplyInTheAppAgain() throws {
        let directory = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = makeStore(in: directory, clock: Clock(Self.berlin(3, 10)))
        let id = try #require(store.learn(heard: "cloud", meant: "Claude")?.wordID)
        let before = store.words

        let receipt = try #require(store.leaveAlone(id, in: Self.notes))
        store.undo(receipt)

        #expect(store.words == before)
        #expect(
            store.apply(to: "a grey cloud", appBundleID: Self.notes.bundleID).text
                == "a grey Claude")
    }

    @Test func deleteRemovesTheWordForGood() throws {
        let directory = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let clock = Clock(Self.berlin(3, 10))
        let store = makeStore(in: directory, clock: clock)
        let id = try #require(store.learn(heard: "cloud", meant: "Claude")?.wordID)

        store.delete(id)

        #expect(store.words.isEmpty)
        #expect(makeStore(in: directory, clock: clock).words.isEmpty)
    }

    // MARK: - Catches

    @Test func recordCatchesCountsPerLocalDayAndStampsTheLastCatch() throws {
        let directory = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let clock = Clock(Self.berlin(3, 23, 30))
        let store = makeStore(in: directory, clock: clock)
        let id = try #require(store.learn(heard: "cloud", meant: "Claude")?.wordID)

        let first = store.apply(to: "ask cloud, then cloud again", appBundleID: nil)
        #expect(first.catches.count == 2)
        store.recordCatches(first.catches)

        // 00:30 in Berlin is still October 3 in UTC: the count follows the
        // local day.
        clock.now = Self.berlin(4, 0, 30)
        store.recordCatches(store.apply(to: "cloud", appBundleID: nil).catches)

        let word = try #require(store.word(withID: id))
        #expect(word.catchesByDay == ["2026-10-03": 2, "2026-10-04": 1])
        #expect(word.totalCatches == 3)
        #expect(word.lastCaughtAt == Self.berlin(4, 0, 30))
    }

    @Test func recordCatchesSkipsUnknownWordsAndEmptyLists() throws {
        let directory = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = makeStore(in: directory, clock: Clock(Self.berlin(3, 10)))
        let id = try #require(store.learn(heard: "cloud", meant: "Claude")?.wordID)

        store.recordCatches([])
        store.recordCatches([
            LearnedWordCatch(
                wordID: UUID(), heard: "srax", meant: "Tesseract", tokenStart: 0, tokenCount: 1)
        ])

        #expect(store.word(withID: id)?.catchesByDay.isEmpty == true)
        #expect(store.word(withID: id)?.lastCaughtAt == nil)
    }

    @Test func uncountTakesBackOneCatchOnTheDayItWasCaught() throws {
        let directory = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let clock = Clock(Self.berlin(3, 9))
        let store = makeStore(in: directory, clock: clock)
        let id = try #require(store.learn(heard: "cloud", meant: "Claude")?.wordID)
        let catches = store.apply(to: "cloud and cloud", appBundleID: nil).catches
        store.recordCatches(catches)
        let caughtAt = clock.now
        clock.now = Self.berlin(4, 9)

        store.uncount(catches[0], caughtAt: caughtAt)
        #expect(store.word(withID: id)?.catchesByDay == ["2026-10-03": 1])

        store.uncount(catches[1], caughtAt: caughtAt)
        #expect(store.word(withID: id)?.catchesByDay == [:])

        store.uncount(catches[1], caughtAt: caughtAt)
        #expect(store.word(withID: id)?.catchesByDay == [:])
    }

    // MARK: - Persistence

    @Test func everythingSurvivesAReloadFromTheSameDirectory() throws {
        let directory = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let clock = Clock(Self.berlin(3, 10))
        let store = makeStore(in: directory, clock: clock)
        let claude = try #require(
            store.learn(
                heard: "cloud", meant: "Claude", pairID: UUID(),
                example: LearnedWord.Example(before: "ask cloud", after: "ask Claude"))?.wordID)
        store.learn(heard: "clod", meant: "Claude")
        store.leaveAlone(claude, in: Self.notes)
        clock.now = Self.berlin(3, 12)
        store.recordCatches(store.apply(to: "ask cloud", appBundleID: nil).catches)
        let tesseract = try #require(store.learn(heard: "SRACT", meant: "Tesseract")?.wordID)
        store.forget(tesseract)

        let reloaded = makeStore(in: directory, clock: clock)

        #expect(reloaded.words == store.words)
        #expect(reloaded.words.map(\.meant) == ["Tesseract", "Claude"])
        #expect(reloaded.apply(to: "ask cloud", appBundleID: nil).text == "ask Claude")
        #expect(
            reloaded.apply(to: "ask cloud", appBundleID: Self.notes.bundleID).text == "ask cloud")
    }

    @Test func aCorruptFileLoadsEmptyAndTheStoreStillLearns() throws {
        let directory = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data("{ not json".utf8).write(
            to: directory.appendingPathComponent("learned_words.json"))

        let store = makeStore(in: directory, clock: Clock(Self.berlin(3, 10)))

        #expect(store.words.isEmpty)
        #expect(store.apply(to: "ask cloud", appBundleID: nil) == .unchanged("ask cloud"))
        // The unreadable file is kept aside, not overwritten by the next save.
        let files = try FileManager.default.contentsOfDirectory(atPath: directory.path)
        #expect(files.contains { $0.hasPrefix("learned_words.unreadable-") })
        store.learn(heard: "cloud", meant: "Claude")
        #expect(store.words.count == 1)
    }

    @Test func aMissingDirectoryIsCreated() {
        let parent = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: parent) }
        let directory = parent.appendingPathComponent("nested", isDirectory: true)

        let store = makeStore(in: directory, clock: Clock(Self.berlin(3, 10)))
        store.learn(heard: "cloud", meant: "Claude")

        #expect(
            FileManager.default.fileExists(
                atPath: directory.appendingPathComponent("learned_words.json").path))
    }
}
