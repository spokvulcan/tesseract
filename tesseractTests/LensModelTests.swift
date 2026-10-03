//
//  LensModelTests.swift
//  tesseractTests
//
//  Pins the **Lens** fix flow (PRD #612) over real stores in a temp
//  directory: typing the word you meant picks the misheard word, a commit
//  teaches a **Learned Word** and makes the take's **Correction Pair** gold,
//  fixing a Learned Word back leaves it alone in the app, ordinary words
//  are fixed in the take only, and Undo takes back what a fix taught.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct LensModelTests {

    private struct Fixture {
        let model: LensModel
        let words: LearnedWordStore
        let pairs: CorrectionPairStore
        let history: FakeTranscriptionStore
        let pairID: UUID
    }

    private static let notes = TargetApp(bundleID: "com.apple.Notes", name: "Notes", pid: 42)

    private func makeFixture(
        text: String, catches: [LearnedWordCatch] = [], seed: (LearnedWordStore) -> Void = { _ in }
    ) -> (Fixture, DictatedTake) {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("lens-model-tests-\(UUID().uuidString)", isDirectory: true)
        let words = LearnedWordStore(directory: directory)
        seed(words)
        let pairs = CorrectionPairStore(directory: directory)
        let pair = CorrectionPair(
            rawASR: text, cleaned: text, verdict: .skipped, committed: text,
            conditions: .init(duration: 3, language: "en", asrModel: "test"))
        pairs.record(pair)
        let history = FakeTranscriptionStore()
        history.add(
            text: text, duration: 3, model: "test", pairID: pair.id, catches: catches, app: nil)
        let model = LensModel(learnedWords: words, pairs: pairs, history: history)
        let take = DictatedTake(
            pairID: pair.id, text: text, catches: catches, app: Self.notes, pasted: true)
        return (
            Fixture(model: model, words: words, pairs: pairs, history: history, pairID: pair.id),
            take
        )
    }

    private func targetText(_ model: LensModel) -> String? {
        model.target.map { model.tokens[$0.range].map(\.coreText).joined(separator: " ") }
    }

    // MARK: - Typing picks the word

    @Test func typingTheWordYouMeantPicksTheOneThatSoundsLikeIt() {
        let (f, take) = makeFixture(text: "Ask cloud why the server drops the first request.")
        f.model.open(take, mode: .afterPaste, vocabulary: ["Claude"])
        #expect(f.model.phase == .fixing)

        f.model.typed = "claude"
        #expect(f.model.completion == "Claude")
        #expect(targetText(f.model) == "cloud")
    }

    @Test func aCompletionFromTheOwnersWordsTargetsTheMishearing() {
        let (f, take) = makeFixture(text: "Run the SRAX tests again.")
        f.model.open(take, mode: .afterPaste, vocabulary: ["Tesseract", "TestFlight"])

        f.model.typed = "tess"
        #expect(f.model.completion == "Tesseract")
        #expect(f.model.completionSuffix == "eract")
        #expect(targetText(f.model) == "SRAX")
    }

    @Test func aSplitWordIsTargetedAsOne() {
        let (f, take) = makeFixture(text: "Open the work tree for the fix.")
        f.model.open(take, mode: .afterPaste, vocabulary: [])

        f.model.typed = "worktree"
        #expect(targetText(f.model) == "work tree")
    }

    @Test func nothingThatSoundsLikeItLeavesNoTarget() {
        let (f, take) = makeFixture(text: "Open the door.")
        f.model.open(take, mode: .afterPaste, vocabulary: [])

        f.model.typed = "Tesseract"
        #expect(f.model.target == nil)
        #expect(f.model.commit() == false)
    }

    // MARK: - A fix teaches

    @Test func aFixLearnsTheWordAndMakesThePairGold() throws {
        let (f, take) = makeFixture(text: "Ask cloud why it failed.")
        f.model.open(take, mode: .afterPaste, vocabulary: ["Claude"])
        f.model.typed = "claude"

        #expect(f.model.commit())
        #expect(f.model.text == "Ask Claude why it failed.")
        #expect(f.model.note == .learned(heard: "cloud", meant: "Claude"))
        #expect(f.model.learnedSummary == "Learned cloud → Claude")
        #expect(f.model.typed.isEmpty)
        #expect(f.model.target == nil)

        let word = try #require(f.words.word(heard: "cloud"))
        #expect(word.meant == "Claude")
        #expect(word.sourcePairID == f.pairID)
        #expect(word.example?.after.contains("Claude") == true)

        let pair = try #require(f.pairs.pair(withID: f.pairID))
        #expect(pair.isGold)
        #expect(pair.correction == "Ask Claude why it failed.")
        #expect(pair.fixes.count == 1)
        #expect(pair.fixes.first?.heard == "cloud")
        #expect(pair.fixes.first?.meant == "Claude")
        #expect(pair.fixes.first?.how == .afterPaste)
        #expect(pair.fixes.first?.app == "com.apple.Notes")

        #expect(f.history.entries.first?.text == "Ask Claude why it failed.")
    }

    @Test func theSameMishearingElsewhereInTheTakeIsFixedToo() {
        let (f, take) = makeFixture(text: "Tell cloud that cloud should rerun it.")
        f.model.open(take, mode: .afterPaste, vocabulary: ["Claude"])
        f.model.typed = "claude"

        #expect(f.model.commit())
        #expect(f.model.text == "Tell Claude that Claude should rerun it.")
        #expect(f.model.fixedTokens == [1, 3])
    }

    @Test func severalFixesInOneTakeEachLand() throws {
        let (f, take) = makeFixture(text: "Ask cloud to open APR for the SRAX tests.")
        f.model.open(take, mode: .afterPaste, vocabulary: ["Claude", "Tesseract", "a PR"])

        f.model.typed = "claude"
        #expect(f.model.commit())
        f.model.typed = "a PR"
        #expect(targetText(f.model) == "APR")
        #expect(f.model.commit())
        f.model.typed = "tess"
        #expect(f.model.commit())

        #expect(f.model.text == "Ask Claude to open a PR for the Tesseract tests.")
        #expect(f.model.fixCount == 3)
        #expect(f.model.receipts.count == 3)
        #expect(try #require(f.pairs.pair(withID: f.pairID)).fixes.count == 3)
        #expect(f.words.word(heard: "srax")?.meant == "Tesseract")
        #expect(f.words.word(heard: "apr")?.meant == "a PR")
    }

    @Test func anOrdinaryWordIsFixedInThisTakeOnly() throws {
        let (f, take) = makeFixture(text: "Put this fix in the release.")
        f.model.open(take, mode: .afterPaste, vocabulary: [])
        f.model.pick(1)
        f.model.typed = "the"

        #expect(f.model.commit())
        #expect(f.model.text == "Put the fix in the release.")
        #expect(f.model.note == .thisTakeOnly)
        #expect(f.words.words.isEmpty)
        #expect(try #require(f.pairs.pair(withID: f.pairID)).isGold)
    }

    // MARK: - Fixing a Learned Word back

    @Test func fixingALearnedWordBackLeavesItAloneInTheApp() throws {
        var wordID: UUID?
        let text = "If the Claude lifts by noon we fly."
        let (f, take) = makeFixture(
            text: text,
            catches: [],
            seed: { store in
                wordID = store.learn(heard: "cloud", meant: "Claude")?.wordID
            })
        let id = try #require(wordID)
        let caught = LearnedWordCatch(
            wordID: id, heard: "cloud", meant: "Claude", tokenStart: 2, tokenCount: 1)
        f.words.recordCatches([caught])
        let reopened = DictatedTake(
            pairID: take.pairID, text: text, catches: [caught], app: take.app, pasted: true,
            at: take.at)
        f.model.open(reopened, mode: .afterPaste, vocabulary: ["Claude"])

        f.model.typed = "cloud"
        #expect(targetText(f.model) == "Claude")
        #expect(f.model.commit())
        #expect(f.model.text == "If the cloud lifts by noon we fly.")
        #expect(f.model.note == .leftAlone(word: "Claude", app: "Notes"))

        let word = try #require(f.words.word(withID: id))
        #expect(word.isLeftAlone(in: "com.apple.Notes"))
        #expect(word.totalCatches == 0)
        #expect(f.words.apply(to: "the cloud", appBundleID: "com.apple.Notes").catches.isEmpty)
        #expect(!f.words.apply(to: "the cloud", appBundleID: "com.apple.Terminal").catches.isEmpty)
    }

    @Test func fixingAnUntrackedLearnedSpellingBackLeavesTheWordAlone() throws {
        // Whisper itself wrote "Claude" where "cloud" was meant: no catch
        // marks it, but the fix still means the Learned Word was wrong here,
        // not that "claude" should become "cloud" everywhere.
        var wordID: UUID?
        let (f, take) = makeFixture(
            text: "If the Claude lifts by noon we fly.",
            seed: { wordID = $0.learn(heard: "cloud", meant: "Claude")?.wordID })
        let id = try #require(wordID)
        f.model.open(take, mode: .afterPaste, vocabulary: [])
        f.model.typed = "cloud"

        #expect(f.model.commit())
        #expect(f.model.text == "If the cloud lifts by noon we fly.")
        #expect(f.model.note == .leftAlone(word: "Claude", app: "Notes"))
        #expect(f.words.word(heard: "claude") == nil)
        #expect(f.words.word(withID: id)?.isLeftAlone(in: "com.apple.Notes") == true)
    }

    // MARK: - Picking by hand

    @Test func arrowsPickAWordAndTypingKeepsIt() {
        let (f, take) = makeFixture(text: "Ship the build tonight.")
        f.model.open(take, mode: .afterPaste, vocabulary: [])

        f.model.moveTarget(by: -1)
        #expect(targetText(f.model) == "tonight")
        f.model.moveTarget(by: -1)
        #expect(targetText(f.model) == "build")
        #expect(f.model.isManualTarget)

        f.model.typed = "release"
        #expect(targetText(f.model) == "build")
        #expect(f.model.commit())
        #expect(f.model.text == "Ship the release tonight.")
    }

    @Test func escClearsTypingBeforeAnythingElse() {
        let (f, take) = makeFixture(text: "Ask cloud.")
        f.model.open(take, mode: .afterPaste, vocabulary: [])
        f.model.typed = "claude"

        #expect(f.model.clearTyping())
        #expect(f.model.typed.isEmpty)
        #expect(f.model.target == nil)
        #expect(!f.model.clearTyping())
    }

    // MARK: - Undo

    @Test func undoTakesBackWhatTheFixTaughtButKeepsTheText() {
        let (f, take) = makeFixture(text: "Ask cloud why.")
        f.model.open(take, mode: .afterPaste, vocabulary: ["Claude"])
        f.model.typed = "claude"
        #expect(f.model.commit())
        #expect(f.words.word(heard: "cloud") != nil)

        f.model.undoLearning()
        #expect(f.words.word(heard: "cloud") == nil)
        #expect(f.model.note == .undone)
        #expect(f.model.receipts.isEmpty)
        #expect(f.model.text == "Ask Claude why.")
    }

    // MARK: - Relocating catches

    @Test func catchesFollowTheirWordsIntoARewrittenText() {
        let id = UUID()
        let catches = [
            LearnedWordCatch(
                wordID: id, heard: "SRAX", meant: "Tesseract", tokenStart: 2, tokenCount: 1)
        ]
        let moved = LearnedWordMatcher.relocate(catches, in: "Please run the Tesseract tests.")
        #expect(moved.map(\.tokenStart) == [3])
        #expect(LearnedWordMatcher.relocate(catches, in: "Please run the tests.").isEmpty)
    }
}
