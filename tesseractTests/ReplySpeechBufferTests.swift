//
//  ReplySpeechBufferTests.swift
//  tesseractTests
//
//  The Reply Speech Buffer (ADR-0097): a reply written token by token comes
//  out as whole sentences, so speech starts on the first one and never
//  renders a sentence before its end has arrived.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@Suite struct ReplySpeechBufferTests {

    /// Feed `text` one character at a time, collecting every piece taken.
    private func stream(_ text: String, into buffer: inout ReplySpeechBuffer) -> [String] {
        var pieces: [String] = []
        for character in text {
            buffer.append(String(character))
            if let piece = buffer.takeSentences() { pieces.append(piece) }
        }
        return pieces
    }

    @Test func aSentenceIsReadyOnceTheSpaceAfterItArrives() {
        var buffer = ReplySpeechBuffer()
        buffer.append("You have two meetings.")
        #expect(buffer.takeSentences() == nil)
        buffer.append(" The first")
        #expect(buffer.takeSentences() == "You have two meetings.")
        #expect(buffer.pending == " The first")
    }

    @Test func tokensComeOutSentenceBySentence() {
        var buffer = ReplySpeechBuffer()
        let pieces = stream("Sure! What time works? I can do three. ", into: &buffer)
        #expect(pieces == ["Sure!", "What time works?", "I can do three."])
        #expect(buffer.takeAll() == nil)
    }

    @Test func everythingCompleteAtOnceIsOnePiece() {
        var buffer = ReplySpeechBuffer()
        buffer.append("One. Two. Three is still")
        #expect(buffer.takeSentences() == "One. Two.")
        #expect(buffer.takeAll() == "Three is still")
    }

    /// The sentence ends a take's words have (`TakeText`).
    @Test func numbersAbbreviationsAndInitialismsEndNoSentence() {
        var buffer = ReplySpeechBuffer()
        let pieces = stream(
            "It is 3.5 km away, e.g. near Prof. Lee at the U.S. office at 3 p.m. today. Next",
            into: &buffer)
        #expect(
            pieces == [
                "It is 3.5 km away, e.g. near Prof. Lee at the U.S. office at 3 p.m. today."
            ])
        #expect(buffer.takeAll() == "Next")
    }

    @Test func aPauseIsNoSentenceEnd() {
        var buffer = ReplySpeechBuffer()
        buffer.append("Let me think… right. Then")
        #expect(buffer.takeSentences() == "Let me think… right.")
    }

    @Test func aLineBreakEndsAPieceAndListsSpeakItemByItem() {
        var buffer = ReplySpeechBuffer()
        let pieces = stream("Your afternoon:\n- Standup at 2\n- Dentist at", into: &buffer)
        #expect(pieces == ["Your afternoon:", "- Standup at 2"])
        #expect(buffer.takeAll() == "- Dentist at")
    }

    @Test func closingMarksStayWithTheirSentence() {
        var buffer = ReplySpeechBuffer()
        buffer.append("He said \"Done.\" Then (later.) we")
        #expect(buffer.takeSentences() == "He said \"Done.\" Then (later.)")
    }

    @Test func whitespaceAloneIsNothingToSay() {
        var buffer = ReplySpeechBuffer()
        buffer.append("  \n ")
        #expect(buffer.takeSentences() == nil)
        #expect(buffer.takeAll() == nil)
    }

    @Test func resetDropsWhatWasHeld() {
        var buffer = ReplySpeechBuffer()
        buffer.append("half a sent")
        buffer.reset()
        #expect(buffer.takeAll() == nil)
    }
}
