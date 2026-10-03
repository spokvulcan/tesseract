//
//  CatchRecordTests.swift
//  tesseractTests
//
//  The **catch record** (PRD #612), the Dictation page's model, built from
//  plain values with a fixed clock and a UTC calendar: the seven days of the
//  week chart (catches by the Learned Words still known, fixes made in the
//  Lens), one tile per Learned Word, today's takes as the Lens opens them,
//  the page's one sentence and the line under it, and the marked runs that
//  show a take's catches and a word's before-and-after strip.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct CatchRecordTests {

    private static let calendar: Calendar = {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "UTC")!
        return calendar
    }()

    /// A UTC time in 2026.
    private static func utc(_ month: Int, _ day: Int, _ hour: Int = 0, _ minute: Int = 0) -> Date {
        calendar.date(
            from: DateComponents(year: 2026, month: month, day: day, hour: hour, minute: minute))!
    }

    /// 3 October 2026, 14:00 UTC: the week runs from 27 September to today.
    private static let now = utc(10, 3, 14)

    private static let terminal = LearnedWord.App(bundleID: "com.apple.Terminal", name: "Terminal")

    /// Caught twice today, once on the week's first day, and five times the
    /// day before the week.
    private static let claudeCode = LearnedWord(
        meant: "Claude Code", heard: ["cloud code"],
        catchesByDay: ["2026-10-03": 2, "2026-09-27": 1, "2026-09-26": 5], fixes: 2,
        learnedAt: utc(9, 20, 9), lastCaughtAt: utc(10, 3, 13),
        example: LearnedWord.Example(before: "… ask cloud code to", after: "… ask Claude Code to"))

    /// Left alone in Terminal; caught three times on 1 October.
    private static let tesseract = LearnedWord(
        meant: "Tesseract", heard: ["sract", "srax"], leftAloneIn: [terminal],
        catchesByDay: ["2026-10-01": 3], fixes: 1, learnedAt: utc(9, 25, 9))

    /// Forgotten yesterday: its catches and fixes leave the record with it.
    private static let worktree = LearnedWord(
        meant: "worktree", heard: ["work tree"],
        catchesByDay: ["2026-10-03": 7, "2026-10-02": 1], fixes: 4, learnedAt: utc(9, 10, 9),
        forgottenAt: utc(10, 2, 18))

    private func makeRecord(
        words: [LearnedWord] = [], pairs: [CorrectionPair] = [],
        entries: [TranscriptionEntry] = []
    ) -> CatchRecord {
        CatchRecord(
            words: words, pairs: pairs, entries: entries, now: Self.now, calendar: Self.calendar)
    }

    private func pair(fixesAt dates: [Date]) -> CorrectionPair {
        CorrectionPair(
            rawASR: "ask cloud code to", cleaned: "ask cloud code to", verdict: .skipped,
            committed: "ask cloud code to",
            fixes: dates.map {
                CorrectionPair.Fix(
                    heard: "cloud code", meant: "Claude Code", how: .page, app: nil, at: $0)
            },
            conditions: .init(duration: 3, language: "en", asrModel: "test"))
    }

    private func entry(
        _ text: String, at date: Date, pairID: UUID? = nil, catches: [LearnedWordCatch] = [],
        app: TranscriptionEntry.App? = nil
    ) -> TranscriptionEntry {
        TranscriptionEntry(
            text: text, timestamp: date, duration: 3, model: "test", pairID: pairID,
            catches: catches, app: app)
    }

    private func plain(_ text: String) -> CatchRecord.Run {
        CatchRecord.Run(text: text, marked: false)
    }

    private func mark(_ text: String) -> CatchRecord.Run {
        CatchRecord.Run(text: text, marked: true)
    }

    // MARK: - The week

    @Test func theWeekIsSevenDaysOldestFirstEndingToday() {
        let record = makeRecord()
        #expect(
            record.days.map(\.date) == [
                Self.utc(9, 27), Self.utc(9, 28), Self.utc(9, 29), Self.utc(9, 30),
                Self.utc(10, 1), Self.utc(10, 2), Self.utc(10, 3),
            ])
        #expect(record.days.last?.date == Self.calendar.startOfDay(for: Self.now))
        #expect(record.days.allSatisfy { $0.caught == 0 && $0.fixed == 0 })
        #expect(record.caughtThisWeek == 0)
        #expect(record.fixesThisWeek == 0)
    }

    @Test func caughtIsTheKnownWordsCatchesPerDay() {
        let record = makeRecord(words: [Self.claudeCode, Self.worktree, Self.tesseract])
        // The day before the week and the forgotten word's catches stay out.
        #expect(record.days.map(\.caught) == [1, 0, 0, 0, 3, 0, 2])
        #expect(record.caughtThisWeek == 6)
    }

    @Test func aForgottenWordTakesItsCatchesWithIt() {
        let record = makeRecord(words: [Self.worktree])
        #expect(record.caughtThisWeek == 0)
        #expect(record.tiles.isEmpty)
        #expect(record.teachingFixes == 0)
    }

    @Test func fixedIsTheLensFixesPerDayFromThePairs() {
        let record = makeRecord(pairs: [
            pair(fixesAt: [
                Self.utc(9, 26, 23, 59), Self.utc(9, 27), Self.utc(10, 3, 10),
                Self.utc(10, 3, 11),
            ]),
            pair(fixesAt: [Self.utc(10, 1, 9)]),
            pair(fixesAt: []),
        ])
        // The fix a minute before the week stays out; the one at its first
        // midnight counts.
        #expect(record.days.map(\.fixed) == [1, 0, 0, 0, 1, 0, 2])
        #expect(record.fixesThisWeek == 4)
    }

    @Test func teachingFixesAreTheKnownWordsFixes() {
        let record = makeRecord(words: [Self.claudeCode, Self.worktree, Self.tesseract])
        #expect(record.teachingFixes == 3)
    }

    // MARK: - Tiles

    @Test func tilesAreTheKnownWordsInTheStoresOrder() {
        let record = makeRecord(words: [Self.claudeCode, Self.worktree, Self.tesseract])
        #expect(record.tiles.map(\.id) == [Self.claudeCode.id, Self.tesseract.id])
        #expect(record.tiles.map(\.meant) == ["Claude Code", "Tesseract"])
    }

    @Test func aTileCarriesItsWordsRecord() throws {
        let record = makeRecord(words: [Self.claudeCode, Self.tesseract])

        let claude = try #require(record.tiles.first)
        #expect(claude.heard == ["cloud code"])
        #expect(claude.learnedAt == Self.utc(9, 20, 9))
        #expect(claude.fixes == 2)
        // Caught counts every day; this week only the window's.
        #expect(claude.caught == 8)
        #expect(claude.caughtThisWeek == 3)
        #expect(claude.lastCaughtAt == Self.utc(10, 3, 13))
        #expect(claude.leftAloneIn.isEmpty)
        #expect(claude.example?.before == "… ask cloud code to")
        #expect(claude.example?.after == "… ask Claude Code to")

        let tesseract = try #require(record.tiles.last)
        #expect(tesseract.heard == ["sract", "srax"])
        #expect(tesseract.leftAloneIn == ["Terminal"])
        #expect(tesseract.caught == 3)
        #expect(tesseract.caughtThisWeek == 3)
        #expect(tesseract.lastCaughtAt == nil)
        #expect(tesseract.example == nil)
    }

    // MARK: - Today

    @Test func todayHasOnlyTodaysTakesNewestFirst() {
        let entries = [
            entry("Earlier today.", at: Self.utc(10, 3, 9)),
            entry("Late last night.", at: Self.utc(10, 2, 23, 59)),
            entry("Just now.", at: Self.utc(10, 3, 13, 30)),
            entry("Right after midnight.", at: Self.utc(10, 3)),
            entry("Last week.", at: Self.utc(9, 28, 12)),
        ]
        let record = makeRecord(entries: entries)
        #expect(
            record.today.map(\.text) == ["Just now.", "Earlier today.", "Right after midnight."])
        #expect(record.today.map(\.id) == [entries[2].id, entries[0].id, entries[3].id])
        #expect(
            record.today.map(\.at) == [
                Self.utc(10, 3, 13, 30), Self.utc(10, 3, 9), Self.utc(10, 3),
            ])
    }

    @Test func aTakesFixesAreCountedFromItsPair() {
        let fixed = pair(fixesAt: [Self.utc(10, 3, 10), Self.utc(10, 3, 10, 1)])
        let gone = UUID()
        let record = makeRecord(
            pairs: [fixed, pair(fixesAt: [Self.utc(10, 3, 9)])],
            entries: [
                entry("Ask Claude Code to.", at: Self.utc(10, 3, 12), pairID: fixed.id),
                entry("No pair.", at: Self.utc(10, 3, 11)),
                entry("A pair since gone.", at: Self.utc(10, 3, 10), pairID: gone),
            ])
        #expect(record.today.map(\.fixes) == [2, 0, 0])
        #expect(record.today.map(\.pairID) == [fixed.id, nil, gone])
    }

    @Test func aTakeOpensInTheLensAsAPageTake() throws {
        let catches = [
            LearnedWordCatch(
                wordID: Self.claudeCode.id, heard: "cloud code", meant: "Claude Code",
                tokenStart: 1, tokenCount: 2)
        ]
        let pairID = UUID()
        let at = Self.utc(10, 3, 12)
        let app = TranscriptionEntry.App(bundleID: "com.apple.Terminal", name: "Terminal")
        let record = makeRecord(entries: [
            entry("Ask Claude Code why.", at: at, pairID: pairID, catches: catches, app: app)
        ])

        let take = try #require(record.today.first)
        #expect(take.catches == catches)
        #expect(take.app == app)

        let lensTake = take.lensTake
        #expect(
            lensTake
                == DictatedTake(
                    pairID: pairID, text: "Ask Claude Code why.", catches: catches,
                    app: TargetApp(bundleID: "com.apple.Terminal", name: "Terminal", pid: 0),
                    pasted: false, at: at))
        // Nothing is pasted back from the page; the app only names the take.
        #expect(!lensTake.pasted)
        #expect(!lensTake.held)
        #expect(lensTake.pastedInto == nil)
        #expect(lensTake.app?.pid == 0)
    }

    @Test func aTakeWithNoAppOpensWithNone() throws {
        let record = makeRecord(entries: [entry("Ship it.", at: Self.utc(10, 3, 12))])
        let take = try #require(record.today.first)
        #expect(take.app == nil)
        #expect(take.lensTake.app == nil)
        #expect(take.lensTake.pairID == nil)
        #expect(take.lensTake.catches.isEmpty)
    }

    /// The history opens its entries the same way the page opens a take.
    @MainActor
    @Test func aHistoryEntryOpensAsTheSameTake() throws {
        let item = entry(
            "Ask Claude Code why.", at: Self.utc(10, 3, 12), pairID: UUID(),
            app: TranscriptionEntry.App(bundleID: nil, name: "Notes"))
        let take = try #require(makeRecord(entries: [item]).today.first)
        #expect(item.lensTake == take.lensTake)
        #expect(item.lensTake.app == TargetApp(bundleID: nil, name: "Notes", pid: 0))
    }

    // MARK: - Words

    @Test func withNothingLearnedThePageSaysHowToStart() {
        let record = makeRecord()
        #expect(record.headline == "Nothing learned yet.")
        #expect(
            record.detail(fixHotkey: "⌃⌥Space")
                == "Fix a word in the Lens (⌃⌥Space after a paste) and I learn it "
                + "for every take after.")
    }

    @Test func theHowToStartLineNamesTheOwnersFixHotkey() {
        let detail = makeRecord().detail(fixHotkey: "⌘⇧F")
        #expect(detail.contains("(⌘⇧F after a paste)"))
        #expect(!detail.contains("⌃⌥Space"))
    }

    @Test func aWordThatCaughtNothingThisWeekSaysSo() {
        let quiet = LearnedWord(
            meant: "Tesseract", heard: ["sract"], catchesByDay: ["2026-09-01": 4])
        let record = makeRecord(words: [quiet])
        #expect(record.headline == "Nothing to catch yet this week.")
        #expect(record.detail(fixHotkey: "⌃⌥Space") == "You taught me 1 word with 1 fix.")
    }

    @Test func oneCatchIsOneMistake() {
        let word = LearnedWord(
            meant: "Tesseract", heard: ["sract"], catchesByDay: ["2026-10-03": 1], fixes: 2)
        let record = makeRecord(words: [word])
        #expect(record.headline == "This week I caught 1 mistake before it reached an app.")
        #expect(record.detail(fixHotkey: "⌃⌥Space") == "You taught me 1 word with 2 fixes.")
    }

    @Test func theSentenceCountsThisWeeksFixesBesideTheCatches() {
        let word = LearnedWord(
            meant: "Tesseract", heard: ["sract"], catchesByDay: ["2026-10-02": 3])
        let thisWeek = pair(fixesAt: [Self.utc(10, 1, 9), Self.utc(10, 3, 9)])
        let older = pair(fixesAt: [Self.utc(9, 20, 9)])
        #expect(
            makeRecord(words: [word], pairs: [thisWeek, older]).headline
                == "This week I caught 3 mistakes before they reached an app, "
                + "and you fixed 2 words.")
        #expect(
            makeRecord(pairs: [pair(fixesAt: [Self.utc(10, 2, 9)])]).headline
                == "This week you fixed 1 word; nothing caught yet.")
    }

    @Test func manyCatchesAreMistakes() {
        let record = makeRecord(words: [Self.claudeCode, Self.worktree, Self.tesseract])
        #expect(record.headline == "This week I caught 6 mistakes before they reached an app.")
        #expect(record.detail(fixHotkey: "⌃⌥Space") == "You taught me 2 words with 3 fixes.")
    }

    // MARK: - Marking a take's catches

    @Test func aTakesCatchesAreMarkedByTheirTokens() {
        let runs = CatchRecord.runs(
            "Ask Claude Code about Tesseract, please.", marking: [1..<3, 4..<5])
        #expect(
            runs == [
                plain("Ask "), mark("Claude Code"), plain(" about "), mark("Tesseract"),
                plain(", please."),
            ])
    }

    @Test func punctuationAroundACatchStaysOutsideTheMark() {
        let runs = CatchRecord.runs("Open \"Tesseract\".", marking: [1..<2])
        #expect(runs == [plain("Open \""), mark("Tesseract"), plain("\".")])
    }

    @Test func rangesOutsideTheTakeOrEmptyAreIgnored() {
        let text = "Ask Claude why."
        let ignored = CatchRecord.runs(text, marking: [3..<4, 2..<5, -1..<1, 1..<1])
        #expect(ignored == [plain(text)])

        let mixed = CatchRecord.runs(text, marking: [5..<6, 1..<1, 1..<2])
        #expect(mixed == [plain("Ask "), mark("Claude"), plain(" why.")])
    }

    @Test func nothingToMarkIsOneRun() {
        let none: [Range<Int>] = []
        #expect(CatchRecord.runs("Ask Claude why.", marking: none) == [plain("Ask Claude why.")])
        #expect(CatchRecord.runs("", marking: none).isEmpty)
    }

    // MARK: - Marking a word's forms

    @Test func aMultiWordFormIsMarkedAsOne() {
        let runs = CatchRecord.runs("… ask cloud code to fix it", marking: ["cloud code"])
        #expect(runs == [plain("… ask "), mark("cloud code"), plain(" to fix it")])
    }

    @Test func theMeantSpellingIsMarkedInTheAfterHalf() {
        let runs = CatchRecord.runs("… ask Claude Code to fix it", marking: ["Claude Code"])
        #expect(runs == [plain("… ask "), mark("Claude Code"), plain(" to fix it")])
    }

    @Test func formsMatchIgnoringCase() {
        let runs = CatchRecord.runs(
            "Cloud Code works. So does CLOUD CODE.", marking: ["cloud code"])
        #expect(
            runs == [
                mark("Cloud Code"), plain(" works. So does "), mark("CLOUD CODE"), plain("."),
            ])
    }

    @Test func theLongestFormIsMarkedFirst() {
        let runs = CatchRecord.runs(
            "Ask cloud code, then cloud.", marking: ["cloud", "cloud code"])
        #expect(
            runs == [
                plain("Ask "), mark("cloud code"), plain(", then "), mark("cloud"), plain("."),
            ])
    }

    @Test func onlyWholeWordsAreMarked() {
        let text = "The clouds cleared over Cloudflare."
        #expect(CatchRecord.runs(text, marking: ["cloud"]) == [plain(text)])

        let runs = CatchRecord.runs("A cloud, not clouds.", marking: ["cloud"])
        #expect(runs == [plain("A "), mark("cloud"), plain(", not clouds.")])
    }

    @Test func emptyFormsMarkNothing() {
        let text = "Ask cloud why."
        #expect(CatchRecord.runs(text, marking: ["", "  ", "…"]) == [plain(text)])
    }
}
