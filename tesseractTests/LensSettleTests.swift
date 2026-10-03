//
//  LensSettleTests.swift
//  tesseractTests
//
//  Which words settle when the final pass lands (PRD #612): the words of
//  the full pass the Live Preview did not show, by a word-level longest
//  common subsequence over bare words.
//

import Testing

@testable import Tesseract_Agent

@MainActor
struct LensSettleTests {

    @Test func aWordThePreviewHadWrongSettles() {
        #expect(
            LensSettle.settled(preview: "Ask Claude why this", final: "Ask Claude why the") == [3])
    }

    @Test func casingAndPunctuationAloneDoNotSettle() {
        #expect(LensSettle.settled(preview: "ask claude why", final: "Ask Claude why.").isEmpty)
    }

    @Test func wordsTheFinalAddsSettle() {
        #expect(
            LensSettle.settled(preview: "Run the tests", final: "Run the tests again today")
                == [3, 4])
    }

    @Test func wordsTheFinalDropsSettleNothing() {
        #expect(LensSettle.settled(preview: "Run the the tests", final: "Run the tests").isEmpty)
    }

    @Test func withNoPreviewNothingSettles() {
        #expect(LensSettle.settled(preview: "", final: "Ship it").isEmpty)
        #expect(LensSettle.settled(preview: "Ship it", final: "").isEmpty)
    }
}
