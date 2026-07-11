//
//  TranscriptionPostProcessorTests.swift
//  tesseractTests
//
//  Pins the post-processing behaviour with literal expectations. Until now the
//  coordinator suites used `process()` itself to build their expected values,
//  so a regression here would have shipped green — these tests are the direct
//  oracle for the blocklist, dedupe, punctuation, and capitalization rules.
//

import Testing

@testable import Tesseract_Agent

struct TranscriptionPostProcessorTests {

    private let processor = TranscriptionPostProcessor()

    // MARK: - Whitespace

    @Test
    func trimsSurroundingWhitespaceAndNewlines() {
        #expect(processor.process("  hello world \n") == "Hello world")
    }

    @Test
    func collapsesInternalWhitespaceRuns() {
        #expect(processor.process("hello   world\t again") == "Hello world again")
    }

    @Test
    func emptyAndWhitespaceOnlyInputYieldsEmpty() {
        #expect(processor.process("") == "")
        #expect(processor.process("   \n ") == "")
    }

    // MARK: - Duplicate-word collapse

    @Test
    func collapsesImmediateDuplicateWords() {
        #expect(processor.process("the the meeting starts now") == "The meeting starts now")
    }

    @Test
    func duplicateCollapseIsCaseInsensitive() {
        #expect(processor.process("The the meeting") == "The meeting")
    }

    @Test
    func keepsNonAdjacentRepeats() {
        #expect(processor.process("it is what is needed") == "It is what is needed")
    }

    // MARK: - Hallucination blocklist

    @Test
    func removesKnownWhisperHallucinations() {
        #expect(processor.process("Thank you for watching.") == "")
        #expect(processor.process("[Music]") == "")
        #expect(processor.process("(upbeat music)") == "")
    }

    @Test
    func removesHallucinationTrailingRealSpeech() {
        #expect(
            processor.process("send the report today. Thanks for watching.")
                == "Send the report today.")
    }

    // MARK: - Punctuation spacing

    @Test
    func removesSpaceBeforePunctuation() {
        #expect(processor.process("hello , world .") == "Hello, world.")
    }

    @Test
    func insertsSpaceAfterPunctuation() {
        #expect(processor.process("first.second") == "First. Second")
    }

    @Test
    func collapsesRepeatedTerminalPunctuation() {
        #expect(processor.process("really??") == "Really?")
        #expect(processor.process("stop!!!") == "Stop!")
    }

    // MARK: - Capitalization

    @Test
    func capitalizesFirstLetterAndSentenceStarts() {
        #expect(
            processor.process("hello. how are you? fine!")
                == "Hello. How are you? Fine!")
    }

    @Test
    func capitalizesStandaloneI() {
        #expect(processor.process("i think i can") == "I think I can")
    }

    @Test
    func leavesEmbeddedLowercaseIAlone() {
        #expect(processor.process("it is in the bin") == "It is in the bin")
    }

    // MARK: - Composition

    @Test
    func fullPipelineOnMessyDictation() {
        #expect(
            processor.process("  so so i said , wait.. we we ship it today !  [Music]")
                == "So I said, wait. We ship it today!")
    }
}
