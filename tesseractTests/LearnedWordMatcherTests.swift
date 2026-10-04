//
//  LearnedWordMatcherTests.swift
//  tesseractTests
//
//  Applying **Learned Words** to a take (PRD #612): whole tokens, case
//  insensitive, multi-word heard forms, longest form first, punctuation
//  kept around a replacement and breaking a match between words, case
//  fitted at the start of a sentence, no **Catch** where the text already
//  reads the meant spelling, catch positions in the output, and the rules
//  a forgotten word or an app's exception leaves out.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct LearnedWordMatcherTests {

    private func word(
        _ meant: String, heard: [String], leftAloneIn apps: [String] = [],
        forgotten: Bool = false
    ) -> LearnedWord {
        LearnedWord(
            meant: meant, heard: heard,
            leftAloneIn: apps.map { LearnedWord.App(bundleID: $0, name: $0) },
            forgottenAt: forgotten ? Date(timeIntervalSince1970: 1_790_000_000) : nil)
    }

    private func apply(
        _ words: [LearnedWord], to text: String, in app: String? = nil
    ) -> LearnedWordsApplication {
        LearnedWordMatcher.apply(LearnedWordMatcher.rules(for: words, in: app), to: text)
    }

    /// The output words each catch points at, by its token positions.
    private func caughtWords(_ application: LearnedWordsApplication) -> [String] {
        let tokens = TakeText.tokens(application.text)
        return application.catches.map { item in
            tokens[item.tokenRange].map(\.coreText).joined(separator: " ")
        }
    }

    private let claude = LearnedWord(meant: "Claude", heard: ["cloud"])
    private let tesseract = LearnedWord(meant: "Tesseract", heard: ["sract", "srax", "tsrac"])
    private let dflash = LearnedWord(meant: "DFlash2", heard: ["d flash two", "d flash 2"])
    private let pullRequest = LearnedWord(meant: "a PR", heard: ["apr"])
    private let worktree = LearnedWord(meant: "worktree", heard: ["work tree"])
    private let kvCache = LearnedWord(meant: "KV cache", heard: ["kiwicache"])

    // MARK: - Matching

    @Test func aSingleWordHeardFormIsReplacedAndCaught() {
        let result = apply([claude], to: "Ask cloud about it.")
        #expect(result.text == "Ask Claude about it.")
        #expect(
            result.catches == [
                LearnedWordCatch(
                    wordID: claude.id, heard: "cloud", meant: "Claude", tokenStart: 1,
                    tokenCount: 1)
            ])
    }

    @Test func everyHeardFormOfAWordIsReplaced() {
        let result = apply([tesseract], to: "sract is up, srax is down. Then tsrac!")
        #expect(result.text == "Tesseract is up, Tesseract is down. Then Tesseract!")
        #expect(result.catches.map(\.heard) == ["sract", "srax", "tsrac"])
        #expect(result.catches.allSatisfy { $0.wordID == tesseract.id })
    }

    @Test func aMultiWordHeardFormIsReplacedAsOneWord() {
        let result = apply([dflash], to: "Turn on D flash two today.")
        #expect(result.text == "Turn on DFlash2 today.")
        #expect(result.catches.map(\.heard) == ["D flash two"])
        #expect(result.catches.map(\.tokenStart) == [2])
        #expect(result.catches.map(\.tokenCount) == [1])
    }

    @Test func aReplacementMayBeSeveralWords() {
        let result = apply([kvCache], to: "The KiwiCache is cold.")
        #expect(result.text == "The KV cache is cold.")
        #expect(caughtWords(result) == ["KV cache"])
    }

    @Test func matchingIsCaseInsensitive() {
        let result = apply([claude], to: "ask CLOUD, then Cloud, then cloud")
        #expect(result.text == "ask Claude, then Claude, then Claude")
        #expect(result.catches.map(\.heard) == ["CLOUD", "Cloud", "cloud"])
    }

    @Test func onlyWholeTokensMatch() {
        // "cloud.md" and "clouds" are other words, not "cloud".
        let result = apply([claude], to: "Open cloud.md and the clouds.")
        #expect(result == .unchanged("Open cloud.md and the clouds."))
    }

    @Test func theLongestHeardFormWinsOverItsPrefix() {
        let work = word("Workbench", heard: ["work"])
        let result = apply([work, worktree], to: "work tree and work")
        #expect(result.text == "Worktree and Workbench")
        #expect(result.catches.map(\.wordID) == [worktree.id, work.id])

        let dflashPrefix = word("Flash", heard: ["d flash"])
        let longer = apply([dflashPrefix, dflash], to: "use d flash two or d flash")
        #expect(longer.text == "use DFlash2 or Flash")
    }

    // MARK: - Punctuation

    @Test func punctuationAroundAReplacementIsKept() {
        let result = apply([claude, tesseract], to: "(cloud), \"SRACT\". [work tree]!")
        #expect(result.text == "(Claude), \"Tesseract\". [work tree]!")

        let multi = apply([worktree], to: "use the (work tree), please")
        #expect(multi.text == "use the (worktree), please")
    }

    @Test func punctuationBetweenWordsBreaksAMultiWordMatch() {
        #expect(apply([dflash], to: "Use D, flash two.") == .unchanged("Use D, flash two."))
        #expect(apply([worktree], to: "the work. Tree") == .unchanged("the work. Tree"))
        #expect(apply([worktree], to: "the (work) tree") == .unchanged("the (work) tree"))
    }

    // MARK: - Case

    @Test func aLowercaseMeantSpellingGetsACapitalAtTheStartOfASentence() {
        let result = apply([worktree, pullRequest], to: "work tree is done. apr is open.")
        #expect(result.text == "Worktree is done. A PR is open.")
        #expect(result.catches.map(\.meant) == ["Worktree", "A PR"])
    }

    @Test func aMeantSpellingIsWrittenAsGivenMidSentence() {
        let result = apply([worktree, pullRequest], to: "Make a work tree, then open APR")
        #expect(result.text == "Make a worktree, then open a PR")
    }

    @Test func textThatAlreadyReadsTheMeantSpellingIsNotACatch() {
        // A casing fix: the heard form is the meant spelling lowercased.
        let casing = word("Tesseract", heard: ["tesseract"])
        let result = apply([casing], to: "Tesseract is up, and tesseract too.")
        #expect(result.text == "Tesseract is up, and Tesseract too.")
        #expect(result.catches.map(\.tokenStart) == [4])

        let iPhone = word("iPhone", heard: ["iphone"])
        #expect(apply([iPhone], to: "iPhone is here.") == .unchanged("iPhone is here."))
    }

    // MARK: - Catch positions

    @Test func catchPositionsFollowReplacementsThatChangeTheWordCount() {
        let result = apply(
            [claude, dflash, pullRequest, worktree],
            to: "APR then work tree then cloud then D flash two end")
        #expect(result.text == "A PR then worktree then Claude then DFlash2 end")
        #expect(result.catches.map(\.tokenStart) == [0, 3, 5, 7])
        #expect(result.catches.map(\.tokenCount) == [2, 1, 1, 1])
        #expect(caughtWords(result) == ["A PR", "worktree", "Claude", "DFlash2"])
    }

    @Test func catchPositionsAfterASplitAndAJoinInOneTake() {
        let result = apply(
            [kvCache, worktree, tesseract], to: "The KiwiCache in the work tree of SRAX.")
        #expect(result.text == "The KV cache in the worktree of Tesseract.")
        #expect(caughtWords(result) == ["KV cache", "worktree", "Tesseract"])
    }

    // MARK: - Nothing to apply

    @Test func noRulesOrNoWordsLeaveTheTextUnchanged() {
        #expect(apply([], to: "Ask cloud.") == .unchanged("Ask cloud."))
        #expect(apply([claude], to: "") == .unchanged(""))
        #expect(apply([claude], to: " ... ") == .unchanged(" ... "))
    }

    // MARK: - Rules

    @Test func aForgottenWordDoesNotApply() {
        let forgotten = word("Claude", heard: ["cloud"], forgotten: true)
        #expect(LearnedWordMatcher.rules(for: [forgotten], in: nil).isEmpty)
        #expect(apply([forgotten], to: "Ask cloud.") == .unchanged("Ask cloud."))
    }

    @Test func aWordLeftAloneInAnAppDoesNotApplyThereButDoesElsewhere() {
        let notes = "com.apple.Notes"
        let leftAlone = word("Claude", heard: ["cloud"], leftAloneIn: [notes])
        #expect(apply([leftAlone], to: "a grey cloud", in: notes).text == "a grey cloud")
        #expect(
            apply([leftAlone], to: "a grey cloud", in: "com.apple.Terminal").text
                == "a grey Claude")
        #expect(apply([leftAlone], to: "a grey cloud", in: nil).text == "a grey Claude")
    }

    @Test func rulesListEveryHeardFormLongestFirst() {
        let rules = LearnedWordMatcher.rules(for: [claude, dflash, worktree], in: nil)
        #expect(
            rules.map { $0.heard.joined(separator: " ") } == [
                "d flash two", "d flash 2", "work tree", "cloud",
            ])
        #expect(rules.first?.meant == "DFlash2")
        #expect(rules.first?.wordID == dflash.id)
    }

    @Test func emptyHeardFormsMakeNoRule() {
        let empty = word("Claude", heard: ["", "cloud"])
        #expect(LearnedWordMatcher.rules(for: [empty], in: nil).count == 1)
    }
}
