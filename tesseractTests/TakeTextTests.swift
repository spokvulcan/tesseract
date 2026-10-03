//
//  TakeTextTests.swift
//  tesseractTests
//
//  The words of a take (PRD #612): whitespace tokens, their cores without
//  the punctuation around them, sentence ends and starts, the normalized
//  heard form, case fitting at the start of a sentence, and the core range
//  of a span.
//

import Testing

@testable import Tesseract_Agent

struct TakeTextTests {

    private func cores(_ text: String) -> [String] {
        TakeText.tokens(text).map(\.coreText)
    }

    // MARK: - Tokens

    @Test func splitsOnAnyWhitespace() {
        let tokens = TakeText.tokens("  open\tthe\nrepo  now ")
        #expect(tokens.map(\.text) == ["open", "the", "repo", "now"])
        #expect(TakeText.tokens("").isEmpty)
        #expect(TakeText.tokens(" \n\t ").isEmpty)
    }

    @Test func tokenRangesPointIntoTheText() {
        let text = "Ask (Cloud), then go."
        for token in TakeText.tokens(text) {
            #expect(String(text[token.range]) == token.text)
            #expect(String(text[token.core]) == token.coreText)
        }
    }

    @Test func coreDropsLeadingAndTrailingPunctuation() {
        #expect(cores("Hello, world.") == ["Hello", "world"])
        #expect(cores("(stop)! [now]; {ok}:") == ["stop", "now", "ok"])
        #expect(cores("¿qué? ¡sí!") == ["qué", "sí"])
    }

    @Test func coreDropsQuotes() {
        #expect(
            cores("\"Claude\" 'cloud' “Tesseract” ‘worktree’ «PR»") == [
                "Claude", "cloud", "Tesseract", "worktree", "PR",
            ])
    }

    @Test func innerPunctuationStaysInTheCore() {
        #expect(
            cores("Open \"cloud.md\", don't use D-flash.") == [
                "Open", "cloud.md", "don't", "use", "D-flash",
            ])
        #expect(cores("version 3.5 now") == ["version", "3.5", "now"])
    }

    @Test func aTokenOfPunctuationIsNotAWord() {
        let tokens = TakeText.tokens("wait \u{2014} then ... go")
        #expect(tokens.map(\.isWord) == [true, false, true, false, true])
        #expect(tokens[1].coreText == "\u{2014}")
        #expect(tokens[3].coreText == "")
    }

    @Test func bareIsTheCoreLowercased() {
        let tokens = TakeText.tokens("\"CLAUDE.md\", SRACT!")
        #expect(tokens.map(\.bare) == ["claude.md", "sract"])
    }

    // MARK: - Sentences

    @Test func periodExclamationAndQuestionEndASentence() {
        let tokens = TakeText.tokens("one. two! three? four?! five.\" six")
        #expect(tokens.map(\.endsSentence) == [true, true, true, true, true, false])
    }

    @Test func anEllipsisIsAPauseNotASentenceEnd() {
        #expect(TakeText.tokens("wait... then")[0].endsSentence == false)
        #expect(TakeText.tokens("wait… then")[0].endsSentence == false)
        #expect(TakeText.tokens("wait...\" then")[0].endsSentence == false)
    }

    @Test func otherPunctuationDoesNotEndASentence() {
        let tokens = TakeText.tokens("one, two; three: four) five")
        #expect(tokens.map(\.endsSentence) == [false, false, false, false, false])
    }

    @Test func aWordStartsASentenceFirstOrAfterASentenceEnd() {
        let tokens = TakeText.tokens("First one. Then this, and that... then! Last")
        let starts = tokens.indices.map { TakeText.startsSentence(tokens, at: $0) }
        #expect(starts == [true, false, true, false, false, false, false, true])
    }

    // MARK: - Normalizing

    @Test func normalizedLowercasesStripsPunctuationAndSingleSpaces() {
        #expect(TakeText.normalized("  D flash   two, ") == "d flash two")
        #expect(TakeText.normalized("\"Claude\"") == "claude")
        #expect(TakeText.normalized("SRACT!") == "sract")
        #expect(TakeText.normalized("cloud.md") == "cloud.md")
    }

    @Test func normalizedOfNothingIsEmpty() {
        #expect(TakeText.normalized("") == "")
        #expect(TakeText.normalized("   ") == "")
        #expect(TakeText.normalized("...") == "")
        #expect(TakeText.normalized("\u{2014}") == "")
    }

    @Test func normalizedKeepsOnlyTheWords() {
        #expect(TakeText.normalized("D ... flash \u{2014} two") == "d flash two")
    }

    @Test func bareJoinsASpansBareFormsWithSingleSpaces() {
        let tokens = TakeText.tokens("Use D flash two, now")
        #expect(TakeText.bare(tokens[1...3]) == "d flash two")
        #expect(TakeText.bare(tokens[0..<0]) == "")
    }

    // MARK: - Case fitting

    @Test func aLowercaseReplacementGetsACapitalAtTheStartOfASentence() {
        #expect(TakeText.fitCase("worktree", atSentenceStart: true) == "Worktree")
        #expect(TakeText.fitCase("a PR", atSentenceStart: true) == "A PR")
    }

    @Test func aReplacementIsWrittenAsGivenMidSentence() {
        #expect(TakeText.fitCase("worktree", atSentenceStart: false) == "worktree")
        #expect(TakeText.fitCase("a PR", atSentenceStart: false) == "a PR")
    }

    @Test func wordsWithTheirOwnCasingAreNeverRecased() {
        #expect(TakeText.fitCase("iPhone", atSentenceStart: true) == "iPhone")
        #expect(TakeText.fitCase("CLAUDE.md", atSentenceStart: true) == "CLAUDE.md")
        #expect(TakeText.fitCase("Tesseract", atSentenceStart: true) == "Tesseract")
        #expect(TakeText.fitCase("DFlash2", atSentenceStart: false) == "DFlash2")
    }

    @Test func fitCaseLeavesEmptyAndDigitFirstReplacementsAlone() {
        #expect(TakeText.fitCase("", atSentenceStart: true) == "")
        #expect(TakeText.fitCase("2fa code", atSentenceStart: true) == "2fa code")
    }

    // MARK: - Core range

    @Test func coreRangeSpansFirstCoreToLastCoreKeepingOuterPunctuation() throws {
        let text = "Open (work tree), now."
        let tokens = TakeText.tokens(text)
        let range = try #require(TakeText.coreRange(tokens[1...2]))
        #expect(text[range] == "work tree")

        let single = try #require(TakeText.coreRange(tokens[3...3]))
        #expect(text[single] == "now")
    }

    @Test func coreRangeOfAnEmptySpanIsNil() {
        let tokens = TakeText.tokens("one two")
        #expect(TakeText.coreRange(tokens[1..<1]) == nil)
    }

    @Test func aFileNameOrAVersionEndsASentenceButAnInitialismDoesNot() {
        let tokens = TakeText.tokens("I updated CLAUDE.md. Then e.g. this, and 2.0. Done")
        #expect(tokens[2].endsSentence)  // CLAUDE.md.
        #expect(!tokens[4].endsSentence)  // e.g.
        #expect(tokens[7].endsSentence)  // 2.0.
    }
}
