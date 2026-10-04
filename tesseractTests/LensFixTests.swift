//
//  LensFixTests.swift
//  tesseractTests
//
//  The fix (PRD #612): type the word you meant. Completion from the owner's
//  words, the target picked by sound on the owner's real mishearings, what
//  a fix teaches (a **Learned Word**, a word left alone in an app, or this
//  take only), and the edit itself with the take's catches kept in place.
//  The takes here are written for the tests.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct LensFixTests {

    private let vocabulary = [
        "Tesseract", "Claude", "CLAUDE.md", "DFlash2", "KV cache", "worktree", "TestFlight",
        "a PR",
    ]

    /// What typing `typed` on `take` would write, where, and what it teaches.
    private func fix(
        _ typed: String, on take: String, catches: [LearnedWordCatch] = [],
        skipping fixed: Set<Int> = []
    ) -> (word: String, span: LensFix.Span?, decision: LensFix.Decision?) {
        let word = LensFix.word(for: typed, vocabulary: vocabulary)
        let tokens = TakeText.tokens(take)
        let span = LensFix.target(for: word, typed: typed, in: tokens, skipping: fixed)
        let decision = span.map {
            LensFix.decide($0, word: word, tokens: tokens, catches: catches)
        }
        return (word, span, decision)
    }

    private func span(_ start: Int, _ count: Int) -> LensFix.Span {
        LensFix.Span(start: start, count: count)
    }

    private func words(_ text: String, _ range: Range<Int>) -> String {
        TakeText.tokens(text)[range].map(\.coreText).joined(separator: " ")
    }

    // MARK: - Completion

    @Test func aCaseInsensitiveMatchCompletesToTheOwnersSpelling() {
        #expect(LensFix.completion(for: "claude", vocabulary: vocabulary) == "Claude")
        #expect(LensFix.completion(for: "claude.md", vocabulary: vocabulary) == "CLAUDE.md")
        #expect(LensFix.completion(for: "TESSERACT", vocabulary: vocabulary) == "Tesseract")
    }

    @Test func typingTheExactSpellingNeedsNoCompletion() {
        #expect(LensFix.completion(for: "Claude", vocabulary: vocabulary) == nil)
        #expect(LensFix.word(for: "Claude", vocabulary: vocabulary) == "Claude")
    }

    @Test func aPrefixCompletesToTheWordItStarts() {
        #expect(LensFix.completion(for: "tess", vocabulary: vocabulary) == "Tesseract")
        #expect(LensFix.completion(for: "wor", vocabulary: vocabulary) == "worktree")
        #expect(LensFix.completion(for: "kv", vocabulary: vocabulary) == "KV cache")
        #expect(LensFix.completion(for: "a P", vocabulary: vocabulary) == "a PR")
        #expect(LensFix.word(for: "dfl", vocabulary: vocabulary) == "DFlash2")
    }

    @Test func aPrefixThatMatchesCaseWinsOverAnEarlierOneThatDoesNot() {
        let words = ["testbed", "Tesseract"]
        #expect(LensFix.completion(for: "Tes", vocabulary: words) == "Tesseract")
        #expect(LensFix.completion(for: "tes", vocabulary: words) == "testbed")
    }

    @Test func aTrailingSpaceKeepsWhatWasTyped() {
        #expect(LensFix.completion(for: "tess ", vocabulary: vocabulary) == nil)
        #expect(LensFix.word(for: "tess ", vocabulary: vocabulary) == "tess")
    }

    @Test func ordinaryWordsAndOneLetterNeverComplete() {
        let words = ["Theseus", "Anthropic", "Tesseract"]
        #expect(LensFix.completion(for: "the", vocabulary: words) == nil)
        #expect(LensFix.completion(for: "an", vocabulary: words) == nil)
        #expect(LensFix.completion(for: "t", vocabulary: words) == nil)
        #expect(LensFix.completion(for: "a", vocabulary: vocabulary) == nil)
        #expect(LensFix.completion(for: "th", vocabulary: words) == "Theseus")
    }

    @Test func aWordOutsideTheVocabularyIsWrittenAsTyped() {
        #expect(LensFix.completion(for: "Clyde", vocabulary: vocabulary) == nil)
        #expect(LensFix.word(for: " Clyde", vocabulary: vocabulary) == "Clyde")
        #expect(LensFix.completion(for: "tess", vocabulary: []) == nil)
    }

    // MARK: - Target on the owner's mishearings

    @Test func typingClaudeTargetsCloud() {
        let result = fix("claude", on: "I asked Cloud to review the diff.")
        #expect(result.word == "Claude")
        #expect(result.span == span(2, 1))
        #expect(result.decision == .learn(heard: "cloud", meant: "Claude"))
    }

    @Test func typingTessCompletesToTesseractAndTargetsSRAX() {
        let result = fix("tess", on: "Push the SRAX branch tonight.")
        #expect(result.word == "Tesseract")
        #expect(result.span == span(2, 1))
        #expect(result.decision == .learn(heard: "srax", meant: "Tesseract"))
    }

    @Test func typingWorktreeTargetsBothWordsOfWorkTree() {
        let result = fix("worktree", on: "Make a new work tree for the fix.")
        #expect(result.span == span(3, 2))
        #expect(result.decision == .learn(heard: "work tree", meant: "worktree"))
    }

    @Test func typingAPRTargetsAPR() {
        let result = fix("a PR", on: "Then open APR against main.")
        #expect(result.span == span(2, 1))
        #expect(result.decision == .learn(heard: "apr", meant: "a PR"))
    }

    @Test(arguments: [
        ("DFlash2", "Turn on D flash two for the draft model.", 2, 3, "d flash two"),
        ("KV cache", "The KiwiCache fills up fast.", 1, 1, "kiwicache"),
        ("TestFlight", "Ship the build to test flight today.", 4, 2, "test flight"),
        ("Tesseract", "Open TSRAC in Xcode.", 1, 1, "tsrac"),
        ("CLAUDE.md", "Read the cloud.md first.", 2, 1, "cloud.md"),
    ])
    func theOtherMishearingsAreTargetedAndLearned(
        _ typed: String, take: String, start: Int, count: Int, heard: String
    ) {
        let result = fix(typed, on: take)
        #expect(result.span == span(start, count))
        #expect(result.decision == .learn(heard: heard, meant: result.word))
    }

    @Test func aSpanAlreadyReadingTheWordIsSkipped() {
        #expect(fix("claude", on: "Claude asked Cloud for help.").span == span(2, 1))
        #expect(fix("claude", on: "Claude is here.").span == nil)
        // Nor is part of a several-word word: "PR" in "A PR" is not a
        // mishearing of "a PR".
        #expect(fix("a PR", on: "A PR is open.").span == nil)
        #expect(fix("KV cache", on: "The KV cache is full.").span == nil)
        #expect(fix("a PR", on: "Open a PR and APR.").span == span(4, 1))
    }

    @Test func aSpanMayNotStartOrEndOnAnOrdinaryWord() {
        #expect(fix("Tesseract", on: "So the SRACT is up.").span == span(2, 1))
        // "the SAURUS" keys exactly like "Thesaurus", but a mishearing does
        // not swallow the article in front of it.
        #expect(fix("Thesaurus", on: "Open the SAURUS app.").span == span(2, 1))
        // Nor the verb after it: "ATL is" keys exactly like "Atlas".
        #expect(fix("Atlas", on: "The ATL is down.").span == span(1, 1))
    }

    @Test func fixedTokensAreSkippedAndTheEarlierSpanWinsATie() {
        let take = "I asked Cloud about Cloud."
        #expect(fix("claude", on: take).span == span(2, 1))
        #expect(fix("claude", on: take, skipping: [2]).span == span(4, 1))
        #expect(fix("claude", on: take, skipping: [2, 4]).span == nil)
    }

    @Test func spellingBreaksTheTieBetweenSpansThatSoundAlike() {
        // "clod" and "Cloud" both sound like "Claude"; "Cloud" is spelled
        // closer.
        #expect(fix("Claude", on: "Ask the clod or the Cloud.").span == span(5, 1))
    }

    @Test func theCompletedWordPicksTheTargetNotTheTypedPrefix() {
        // "tess" alone sounds more like "this" than like "SRAX".
        #expect(fix("tess", on: "Open this SRAX repo.").span == span(2, 1))
    }

    @Test func aWordInCapitalsWinsATieWithAnOrdinaryWord() {
        // "test" and "SRAX" sound equally like "Tesseract"; Whisper writes a
        // word it does not know in capitals.
        #expect(fix("Tesseract", on: "Run the test on SRAX.").span == span(4, 1))
    }

    @Test func aWordCasedWrongIsTargetedWhenNothingElseSoundsLikeIt() {
        #expect(fix("Tesseract", on: "Open the tesseract repo.").span == span(2, 1))
        #expect(fix("Tesseract", on: "Tesseract is open.").span == nil)
    }

    @Test func punctuationBetweenWordsKeepsThemApart() {
        #expect(fix("worktree", on: "Make a work. Tree later.").span == span(2, 1))
    }

    @Test func nothingThatSoundsLikeTheWordIsNoTarget() {
        #expect(fix("Tesseract", on: "The build passed.").span == nil)
        #expect(fix("Tesseract", on: "").span == nil)
    }

    // MARK: - Decision

    @Test func typingWhatWasHeardOverACatchLeavesTheWordAlone() throws {
        let id = UUID()
        let take = LearnedWordMatcher.apply(
            [LearnedWordMatcher.Rule(wordID: id, heard: ["cloud"], meant: "Claude")],
            to: "The cloud is grey today.")
        #expect(take.text == "The Claude is grey today.")

        let result = fix("cloud", on: take.text, catches: take.catches)

        #expect(result.word == "cloud")
        #expect(result.span == span(1, 1))
        #expect(result.decision == .leaveAlone(wordID: id, catchIndex: 0))
    }

    @Test func aThirdWordOverACatchLeavesTheWordAloneAndFixesTheTake() {
        let id = UUID()
        let take = LearnedWordMatcher.apply(
            [LearnedWordMatcher.Rule(wordID: id, heard: ["cloud"], meant: "Claude")],
            to: "Ask cloud about the plan.")

        let result = fix("Clyde", on: take.text, catches: take.catches)

        #expect(result.span == span(1, 1))
        #expect(result.decision == .leaveAloneAndFix(wordID: id, catchIndex: 0))
    }

    @Test func theDecisionNamesTheCatchTheSpanOverlaps() {
        let first = UUID()
        let second = UUID()
        let take = LearnedWordMatcher.apply(
            [
                LearnedWordMatcher.Rule(wordID: first, heard: ["apr"], meant: "a PR"),
                LearnedWordMatcher.Rule(wordID: second, heard: ["cloud"], meant: "Claude"),
            ],
            to: "Open APR and ask cloud.")
        let tokens = TakeText.tokens(take.text)

        #expect(
            LensFix.decide(span(5, 1), word: "cloud", tokens: tokens, catches: take.catches)
                == .leaveAlone(wordID: second, catchIndex: 1))
        #expect(
            LensFix.decide(span(1, 2), word: "APR", tokens: tokens, catches: take.catches)
                == .leaveAlone(wordID: first, catchIndex: 0))
    }

    @Test func aFixToAnOrdinaryWordIsThisTakeOnly() {
        let result = fix("the", on: "Fix teh tests first.")
        #expect(result.word == "the")
        #expect(result.span == span(1, 1))
        #expect(result.decision == .thisTakeOnly)
        #expect(!LensFix.shouldLearn(heard: "teh", meant: "the"))
        #expect(!LensFix.shouldLearn(heard: "this", meant: "that"))
    }

    @Test func aChangeOfWordEndingIsThisTakeOnly() {
        let result = fix("drop", on: "It drops the frame.")
        #expect(result.span == span(1, 1))
        #expect(result.decision == .thisTakeOnly)
    }

    @Test(arguments: [
        ("drops", "drop"),
        ("drop", "dropped"),
        ("cats", "cat"),
        ("walked", "walk"),
        ("running", "run"),
        ("stop", "stopped"),
        ("making", "make"),
        ("use", "using"),
        ("quick", "quickly"),
    ])
    func inflectionsAreNotLearned(_ heard: String, _ meant: String) {
        #expect(LensFix.isInflection(heard, meant))
        #expect(!LensFix.shouldLearn(heard: heard, meant: meant))
    }

    @Test func differentWordsAreNotInflections() {
        #expect(!LensFix.isInflection("cloud", "claude"))
        #expect(!LensFix.isInflection("drop", "drop"))
        #expect(!LensFix.isInflection("is", "its"))
        #expect(!LensFix.isInflection("sract", "tesseract"))
    }

    @Test func casingFixesAreLearned() {
        let tokens = TakeText.tokens("Open the tesseract repo.")
        #expect(
            LensFix.decide(span(2, 1), word: "Tesseract", tokens: tokens, catches: [])
                == .learn(heard: "tesseract", meant: "Tesseract"))
        #expect(LensFix.shouldLearn(heard: "iphone", meant: "iPhone"))
    }

    @Test func takingCapitalsAwayIsThisTakeOnly() {
        // "Claude" → "claude" (the CLI) depends on the sentence.
        #expect(!LensFix.shouldLearn(heard: "claude", meant: "claude"))
        let tokens = TakeText.tokens("Run Claude in the terminal.")
        #expect(
            LensFix.decide(span(1, 1), word: "claude", tokens: tokens, catches: [])
                == .thisTakeOnly)
    }

    @Test func partOfTheOwnersWordIsNotLearnedAsTheWholeWord() {
        // "test, flight" keeps the words apart, so the fix lands on "flight":
        // learning it would turn every "flight" into TestFlight.
        #expect(!LensFix.shouldLearn(heard: "flight", meant: "TestFlight"))
        #expect(!LensFix.shouldLearn(heard: "tess", meant: "Tesseract"))
        #expect(LensFix.shouldLearn(heard: "test flight", meant: "TestFlight"))
    }

    @Test func extendingACaughtWordLearnsFromWhatWasHeard() {
        // "cloud code" was caught as "Claude code"; the owner wants "Claude
        // Code". The caught word stays on; the new word is learned from the
        // recognizer's own words.
        let id = UUID()
        let take = "Ask Claude code to fix it."
        let caught = LearnedWordCatch(
            wordID: id, heard: "cloud", meant: "Claude", tokenStart: 1, tokenCount: 1)
        let tokens = TakeText.tokens(take)
        #expect(LensFix.target(for: "Claude Code", typed: "Claude Code", in: tokens) == span(1, 2))
        #expect(
            LensFix.decide(span(1, 2), word: "Claude Code", tokens: tokens, catches: [caught])
                == .learn(heard: "cloud code", meant: "Claude Code"))
    }

    @Test func recasingACaughtWordIsThisTakeOnly() {
        let caught = LearnedWordCatch(
            wordID: UUID(), heard: "work tree", meant: "worktree", tokenStart: 0, tokenCount: 1)
        let tokens = TakeText.tokens("worktree is clean.")
        #expect(
            LensFix.decide(span(0, 1), word: "Worktree", tokens: tokens, catches: [caught])
                == .thisTakeOnly)
    }

    @Test func numbersAreNeverLearnedOnTheirOwn() {
        #expect(!LensFix.shouldLearn(heard: "two", meant: "2"))
        #expect(!LensFix.shouldLearn(heard: "2", meant: "two"))
        #expect(!LensFix.shouldLearn(heard: "one", meant: "1"))
        // Inside a term they are.
        #expect(LensFix.shouldLearn(heard: "d flash two", meant: "DFlash2"))
    }

    @Test func aSingleLetterIsNeverLearned() {
        #expect(!LensFix.shouldLearn(heard: "m", meant: "Em"))
        #expect(LensFix.shouldLearn(heard: "m dashes", meant: "em dashes"))
    }

    @Test func contractionsAreGrammar() {
        #expect(!LensFix.shouldLearn(heard: "well", meant: "we'll"))
        #expect(!LensFix.shouldLearn(heard: "its", meant: "it's"))
    }

    @Test func dictionaryHomophonesAndCapitalsAreThisTakeOnly() {
        let dictionary: (String) -> Bool = {
            ["weather", "whether", "go", "notes", "tesseract"].contains($0)
        }
        #expect(
            !LensFix.shouldLearn(heard: "weather", meant: "whether", isDictionaryWord: dictionary))
        #expect(!LensFix.shouldLearn(heard: "go", meant: "Go", isDictionaryWord: dictionary))
        #expect(!LensFix.shouldLearn(heard: "notes", meant: "Notes", isDictionaryWord: dictionary))
        // A word of the owner's with its own shape is still learned.
        #expect(
            LensFix.shouldLearn(
                heard: "testflight", meant: "TestFlight", isDictionaryWord: dictionary))
        #expect(LensFix.shouldLearn(heard: "cloud", meant: "Claude", isDictionaryWord: dictionary))
    }

    @Test func aRewriteIsThisTakeOnly() {
        let tokens = TakeText.tokens("Deploy the backend now.")
        #expect(
            LensFix.decide(span(2, 1), word: "Pi agent", tokens: tokens, catches: [])
                == .thisTakeOnly)
        #expect(!LensFix.shouldLearn(heard: "backend", meant: "Pi agent"))
        #expect(!LensFix.shouldLearn(heard: "one two three four", meant: "1234"))
    }

    @Test func aSpanThatAlreadyReadsTheWordIsUnchanged() {
        let tokens = TakeText.tokens("Ask Claude now. A PR is open. Worktree next.")
        #expect(
            LensFix.decide(span(1, 1), word: "Claude", tokens: tokens, catches: []) == .unchanged)
        // At the start of a sentence the word is read case fitted, as the
        // fix would write it.
        #expect(LensFix.decide(span(3, 2), word: "a PR", tokens: tokens, catches: []) == .unchanged)
        #expect(
            LensFix.decide(span(7, 1), word: "worktree", tokens: tokens, catches: []) == .unchanged)
        #expect(
            LensFix.decide(span(9, 1), word: "Claude", tokens: tokens, catches: []) == .unchanged)
        #expect(
            LensFix.decide(span(1, 0), word: "Claude", tokens: tokens, catches: []) == .unchanged)
    }

    @Test func ordinaryWordsHoldTheStopList() {
        for word in ["the", "a", "this", "and", "it", "um", "okay"] {
            #expect(LensFix.ordinaryWords.contains(word))
        }
        for word in ["claude", "tesseract", "worktree", "pr"] {
            #expect(!LensFix.ordinaryWords.contains(word))
        }
    }

    // MARK: - Edit

    @Test func replaceKeepsThePunctuationAroundTheWord() throws {
        let edit = try #require(
            LensFix.replace(
                span(2, 1), with: "Claude", in: "I asked (Cloud), then left.", catches: []))
        #expect(edit.text == "I asked (Claude), then left.")
        #expect(edit.replaced == "Cloud")
        #expect(edit.written == "Claude")
        #expect(edit.span == span(2, 1))
    }

    @Test func replaceFitsCaseAtTheStartOfASentence() throws {
        let edit = try #require(
            LensFix.replace(
                span(1, 2), with: "worktree", in: "Done. work tree is next", catches: []))
        #expect(edit.text == "Done. Worktree is next")
        #expect(edit.replaced == "work tree")
        #expect(edit.written == "Worktree")
        #expect(edit.span == span(1, 1))

        let split = try #require(
            LensFix.replace(span(0, 1), with: "a PR", in: "APR is open.", catches: []))
        #expect(split.text == "A PR is open.")
        #expect(split.span == span(0, 2))
    }

    @Test func replaceDropsTheCatchItOverlapsAndShiftsLaterOnes() throws {
        let pr = UUID()
        let tesseract = UUID()
        let claude = UUID()
        let take = LearnedWordMatcher.apply(
            [
                LearnedWordMatcher.Rule(wordID: pr, heard: ["apr"], meant: "a PR"),
                LearnedWordMatcher.Rule(wordID: tesseract, heard: ["sract"], meant: "Tesseract"),
                LearnedWordMatcher.Rule(wordID: claude, heard: ["cloud"], meant: "Claude"),
            ],
            to: "Open APR on the SRACT and ask cloud.")
        #expect(take.text == "Open a PR on the Tesseract and ask Claude.")
        #expect(take.catches.map(\.tokenStart) == [1, 5, 8])

        let edit = try #require(
            LensFix.replace(span(1, 2), with: "PRs", in: take.text, catches: take.catches))

        #expect(edit.text == "Open PRs on the Tesseract and ask Claude.")
        #expect(edit.catches.map(\.wordID) == [tesseract, claude])
        #expect(edit.catches.map(\.tokenStart) == [4, 7])
        #expect(edit.catches.map { words(edit.text, $0.tokenRange) } == ["Tesseract", "Claude"])
    }

    @Test func replaceKeepsEarlierCatchesAndShiftsLaterOnesWhenAFixGrows() throws {
        let pr = UUID()
        let claude = UUID()
        let take = LearnedWordMatcher.apply(
            [
                LearnedWordMatcher.Rule(wordID: pr, heard: ["apr"], meant: "a PR"),
                LearnedWordMatcher.Rule(wordID: claude, heard: ["cloud"], meant: "Claude"),
            ],
            to: "Open APR in the KiwiCache branch and ask cloud.")

        let edit = try #require(
            LensFix.replace(span(5, 1), with: "KV cache", in: take.text, catches: take.catches))

        #expect(edit.text == "Open a PR in the KV cache branch and ask Claude.")
        #expect(edit.span == span(5, 2))
        #expect(edit.catches.map(\.tokenStart) == [1, 10])
        #expect(edit.catches.map { words(edit.text, $0.tokenRange) } == ["a PR", "Claude"])
    }

    @Test func replaceRefusesAnEmptyOutOfRangeOrBlankFix() {
        let text = "Ask cloud now."
        #expect(LensFix.replace(span(1, 0), with: "Claude", in: text, catches: []) == nil)
        #expect(LensFix.replace(span(2, 2), with: "Claude", in: text, catches: []) == nil)
        #expect(LensFix.replace(span(1, 1), with: "  ", in: text, catches: []) == nil)
        #expect(LensFix.replace(span(0, 1), with: "Claude", in: "", catches: []) == nil)
    }

    @Test func shiftedDropsIndicesInTheReplacedSpanAndMovesLaterOnes() throws {
        let text = "Open the work tree for the SRAX repo now."
        let joined = try #require(
            LensFix.replace(span(2, 2), with: "worktree", in: text, catches: []))
        #expect(LensFix.shifted([0, 2, 3, 5, 6], by: joined, replacing: span(2, 2)) == [0, 4, 5])

        let split = try #require(
            LensFix.replace(span(0, 1), with: "Go and open", in: text, catches: []))
        #expect(LensFix.shifted([0, 1, 6], by: split, replacing: span(0, 1)) == [3, 8])

        let same = try #require(
            LensFix.replace(span(6, 1), with: "Tesseract", in: text, catches: []))
        #expect(LensFix.shifted([1, 6, 7], by: same, replacing: span(6, 1)) == [1, 7])
    }

    // MARK: - Several fixes

    @Test func severalFixesInOneTakeApplyInSequence() throws {
        let pr = UUID()
        let take = LearnedWordMatcher.apply(
            [LearnedWordMatcher.Rule(wordID: pr, heard: ["apr"], meant: "a PR")],
            to: "Open APR on the SRAX repo and push the work tree with the KiwiCache fix.")
        var text = take.text
        var catches = take.catches
        var fixed: Set<Int> = []
        var decisions: [LensFix.Decision] = []

        for typed in ["tess", "worktree", "kv"] {
            let word = LensFix.word(for: typed, vocabulary: vocabulary)
            let tokens = TakeText.tokens(text)
            let target = try #require(
                LensFix.target(for: word, typed: typed, in: tokens, skipping: fixed))
            decisions.append(LensFix.decide(target, word: word, tokens: tokens, catches: catches))
            let edit = try #require(LensFix.replace(target, with: word, in: text, catches: catches))
            fixed = LensFix.shifted(fixed, by: edit, replacing: target).union(edit.span.range)
            text = edit.text
            catches = edit.catches
        }

        #expect(
            text == "Open a PR on the Tesseract repo and push the worktree with the KV cache fix.")
        #expect(
            decisions == [
                .learn(heard: "srax", meant: "Tesseract"),
                .learn(heard: "work tree", meant: "worktree"),
                .learn(heard: "kiwicache", meant: "KV cache"),
            ])
        #expect(catches == take.catches)
        #expect(fixed == [5, 10, 13, 14])
        #expect(
            fixed.sorted().map { words(text, $0..<($0 + 1)) } == [
                "Tesseract", "worktree", "KV", "cache",
            ])
    }
}
