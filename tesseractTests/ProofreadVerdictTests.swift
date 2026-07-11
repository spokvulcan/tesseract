//
//  ProofreadVerdictTests.swift
//  tesseractTests
//
//  The pure value layer of the **Proofread Pass** (map #283, ADR-0034):
//  `WordDiff` (the narration diff), `ProofreadGuard` (the fail-open
//  acceptance guard), `ProofreadReply` (the plain-text reply contract), and
//  the defensive think-block strip. No model, no actor — plain values in,
//  plain values out.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct WordDiffTests {

    @Test func identicalTextsProduceNoEdits() {
        #expect(WordDiff.edits(from: "hello world", to: "hello world") == [])
    }

    @Test func singleMisheardWordPairsAsOneSwap() {
        #expect(
            WordDiff.edits(from: "peace of cake", to: "piece of cake") == [
                WordEdit(original: "peace", replacement: "piece")
            ])
    }

    @Test func insertionHasEmptyOriginal() {
        #expect(
            WordDiff.edits(from: "send the report", to: "send me the report") == [
                WordEdit(original: "", replacement: "me")
            ])
    }

    @Test func deletionHasEmptyReplacement() {
        #expect(
            WordDiff.edits(from: "the um meeting", to: "the meeting") == [
                WordEdit(original: "um", replacement: "")
            ])
    }

    /// Adjacent delete+insert runs pair positionally into swaps; the leftover
    /// becomes a pure insert/delete.
    @Test func adjacentRunsPairIntoSwapsWithLeftovers() {
        #expect(
            WordDiff.edits(from: "a b c", to: "a x y c") == [
                WordEdit(original: "b", replacement: "x"),
                WordEdit(original: "", replacement: "y"),
            ])
    }

    @Test func trailingChangeIsFlushed() {
        #expect(
            WordDiff.edits(from: "call mum", to: "call mom") == [
                WordEdit(original: "mum", replacement: "mom")
            ])
    }
}

struct ProofreadGuardTests {

    @Test func smallFixIsAcceptable() {
        #expect(ProofreadGuard.acceptable(raw: "peace of cake", corrected: "piece of cake"))
    }

    @Test func emptyCorrectionIsRejected() {
        #expect(!ProofreadGuard.acceptable(raw: "hello", corrected: ""))
    }

    /// The model answering instead of correcting: far longer than the raw.
    @Test func correctionBeyondTheLengthRatioIsRejected() {
        #expect(
            !ProofreadGuard.acceptable(
                raw: "what is two plus two",
                corrected:
                    "The answer to your question of two plus two is four, which is basic arithmetic"
            ))
    }

    /// A wholesale summary: far shorter than the raw.
    @Test func correctionFarShorterThanRawIsRejected() {
        #expect(
            !ProofreadGuard.acceptable(
                raw: "please remind me to pick up the dry cleaning tomorrow at nine",
                corrected: "dry cleaning"
            ))
    }

    /// Same length but most words replaced — the 0.8B rewrote, not corrected.
    @Test func rewritingMostWordsIsRejected() {
        #expect(
            !ProofreadGuard.acceptable(
                raw: "send the quarterly report now",
                corrected: "dispatch a yearly summary immediately"
            ))
    }
}

struct ProofreadReplyTests {

    @Test func rejectPrefixParsesToRejectedWithReason() {
        #expect(
            ProofreadReply.parse("REJECT: unintelligible mumbling", raw: "asdf ghjk")
                == .rejected(reason: "unintelligible mumbling"))
    }

    @Test func rejectWithoutReasonGetsTheDefaultReason() {
        #expect(
            ProofreadReply.parse("REJECT:", raw: "asdf")
                == .rejected(reason: "Unintelligible transcription"))
    }

    @Test func echoedRawParsesToUnchanged() {
        #expect(ProofreadReply.parse("hello world", raw: "hello world") == .unchanged)
    }

    @Test func emptyReplyParsesToUnchanged() {
        #expect(ProofreadReply.parse("  \n ", raw: "hello world") == .unchanged)
    }

    /// A reply that fails the acceptance guard fails *open*: the user's words win.
    @Test func wanderingReplyFailsOpenToUnchanged() {
        #expect(
            ProofreadReply.parse(
                "I think you meant to ask about the weather today, which is sunny",
                raw: "whether report"
            ) == .unchanged)
    }

    @Test func genuineFixParsesToCorrectedWithEdits() {
        let verdict = ProofreadReply.parse("piece of cake", raw: "peace of cake")
        #expect(
            verdict
                == .corrected(
                    text: "piece of cake",
                    edits: [WordEdit(original: "peace", replacement: "piece")]))
    }

    @Test func surroundingWhitespaceIsTrimmedBeforeParsing() {
        #expect(ProofreadReply.parse("\n hello world \n", raw: "hello world") == .unchanged)
    }
}

struct ThinkBlockStripTests {

    @Test func replyWithoutThinkBlockIsOnlyTrimmed() {
        #expect(ProofreadModel.strippingThinkBlock("  piece of cake \n") == "piece of cake")
    }

    @Test func leadingThinkBlockIsStripped() {
        #expect(
            ProofreadModel.strippingThinkBlock(
                "<think>the user said peace, likely piece</think>\npiece of cake")
                == "piece of cake")
    }

    @Test func unterminatedThinkBlockIsLeftAlone() {
        #expect(
            ProofreadModel.strippingThinkBlock("<think>never closed")
                == "<think>never closed")
    }
}
