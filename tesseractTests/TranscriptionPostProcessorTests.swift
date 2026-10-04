//
//  TranscriptionPostProcessorTests.swift
//  tesseractTests
//
//  Pins the regex cleanup with literal expectations: the direct oracle for the
//  blocklist, stutter collapse, punctuation and capitalization rules (the
//  coordinator suites build their expected values with `process()` itself).
//  The PRD #612 fixes are pinned here too: file names, domains, decimals and
//  abbreviations keep their dots, Whisper's "..." stays a pause, deliberate
//  repetition survives while stutters collapse, and a take that is only
//  punctuation commits nothing.
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

    // MARK: - Stutters

    @Test
    func collapsesImmediateDuplicateWords() {
        #expect(processor.process("the the meeting starts now") == "The meeting starts now")
    }

    @Test
    func duplicateCollapseIsCaseInsensitive() {
        #expect(processor.process("The the meeting") == "The meeting")
    }

    @Test
    func collapsesLongerStuttersToOneWord() {
        #expect(processor.process("I I I think so") == "I think so")
        #expect(processor.process("the the the") == "The")
    }

    @Test
    func collapsesStuttersInsideASentence() {
        #expect(processor.process("the plan is to to ship it") == "The plan is to ship it")
        #expect(processor.process("the current current state") == "The current state")
    }

    @Test
    func stutterBeforePunctuationKeepsThePunctuation() {
        #expect(processor.process("we ship it it.") == "We ship it.")
    }

    @Test
    func keepsNonAdjacentRepeats() {
        #expect(processor.process("it is what is needed") == "It is what is needed")
    }

    // MARK: - Deliberate repetition

    @Test
    func keepsWordsRepeatedForEmphasis() {
        #expect(processor.process("it is very very good") == "It is very very good")
        #expect(processor.process("much much better") == "Much much better")
        #expect(processor.process("so so good") == "So so good")
    }

    @Test
    func keepsThreeOrMoreRepeatsOfADeliberateWord() {
        #expect(processor.process("no no no") == "No no no")
        #expect(processor.process("really really really") == "Really really really")
    }

    @Test
    func keepsDoubledInterjections() {
        #expect(processor.process("bye bye") == "Bye bye")
        #expect(processor.process("knock knock") == "Knock knock")
    }

    @Test
    func keepsRepetitionWithPunctuationBetweenTheCopies() {
        #expect(processor.process("very, very good") == "Very, very good")
        #expect(processor.process("much, much, much better") == "Much, much, much better")
        #expect(processor.process("yes. yes.") == "Yes. Yes.")
    }

    @Test
    func keepsRepeatedNumbers() {
        #expect(processor.process("the code is 4 4 7") == "The code is 4 4 7")
        #expect(processor.process("dial one one two") == "Dial one one two")
    }

    @Test
    func keepsDoubledNames() {
        #expect(processor.process("a trip to Bora Bora") == "A trip to Bora Bora")
    }

    @Test
    func keepsGrammaticalDoubles() {
        #expect(processor.process("I had had enough") == "I had had enough")
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

    @Test
    func removingAHallucinationMidTextLeavesOneSpace() {
        #expect(processor.process("first part [Music] second part") == "First part second part")
    }

    @Test
    func keepsThankYouBecauseSilenceIsTheLevelChecksJob() {
        // A silent capture never reaches the cleanup (`CaptureLevel`), and a
        // spoken "Thank you." is real speech.
        #expect(processor.process("Thank you.") == "Thank you.")
    }

    // MARK: - File names, domains, numbers and abbreviations

    @Test
    func keepsFileNamesWhole() {
        #expect(processor.process("open cloud.md and edit it") == "Open cloud.md and edit it")
        #expect(processor.process("open CLAUDE.md and edit it") == "Open CLAUDE.md and edit it")
        #expect(
            processor.process("the list is in versions.txt now")
                == "The list is in versions.txt now")
    }

    @Test
    func keepsDomainsAndDottedNamesWhole() {
        #expect(processor.process("check example.com today") == "Check example.com today")
        #expect(
            processor.process("we use Node.js and Vue.js here") == "We use Node.js and Vue.js here")
    }

    @Test
    func keepsFileLocationsWhole() {
        #expect(
            processor.process("see main.swift:42 for details") == "See main.swift:42 for details")
    }

    @Test
    func leavesAddressesAndCodeAsWritten() {
        #expect(
            processor.process("open https://example.com/a?b=c,d now")
                == "Open https://example.com/a?b=c,d now")
        #expect(processor.process("write to a.b@example.com") == "Write to a.b@example.com")
        #expect(processor.process("use std::vector here") == "Use std::vector here")
    }

    @Test
    func keepsTheSpaceBeforeADotFile() {
        #expect(processor.process("the .env file") == "The .env file")
    }

    @Test
    func doesNotCapitalizeADottedWordAtTheStart() {
        #expect(processor.process("cloud.md is open") == "cloud.md is open")
        #expect(processor.process("first.second") == "first.second")
    }

    @Test
    func noCapitalAfterADecimalPoint() {
        #expect(processor.process("version 3.5 is out") == "Version 3.5 is out")
        #expect(processor.process("pi is 3.14") == "Pi is 3.14")
    }

    @Test
    func keepsTimesAndThousands() {
        #expect(processor.process("at 10:30 we meet") == "At 10:30 we meet")
        #expect(processor.process("it costs 1,000 dollars") == "It costs 1,000 dollars")
    }

    @Test
    func abbreviationsEndNoSentence() {
        #expect(processor.process("use a tool, e.g. this one") == "Use a tool, e.g. this one")
        #expect(processor.process("the short one, i.e. that one") == "The short one, i.e. that one")
        #expect(processor.process("at 9 a.m. tomorrow") == "At 9 a.m. tomorrow")
        #expect(processor.process("tabs vs. spaces") == "Tabs vs. spaces")
    }

    // MARK: - Pauses

    @Test
    func ellipsisIsAPauseNotASentenceBreak() {
        #expect(processor.process("I... don't know") == "I... don't know")
        #expect(processor.process("so... maybe later") == "So... maybe later")
        #expect(processor.process("wait… what") == "Wait… what")
    }

    @Test
    func ellipsisKeepsTheCaseWhisperWrote() {
        #expect(processor.process("I think... Then we left") == "I think... Then we left")
    }

    @Test
    func detachedEllipsisJoinsTheWordBefore() {
        #expect(processor.process("I ... don't") == "I... don't")
    }

    @Test
    func trailingEllipsisStaysAtTheEnd() {
        #expect(processor.process("also, please...") == "Also, please...")
    }

    @Test
    func longDotRunsBecomeOneEllipsis() {
        #expect(processor.process("well.... ok") == "Well... ok")
        #expect(processor.process("a long......pause here") == "A long... pause here")
    }

    @Test
    func twoDotsAreOneSentenceEnd() {
        #expect(processor.process("wait.. we ship") == "Wait. We ship")
    }

    @Test
    func takeOpeningWithAnEllipsisContinuesAThought() {
        #expect(processor.process("... and then") == "... and then")
    }

    // MARK: - Punctuation only

    @Test
    func punctuationOnlyTakeIsEmpty() {
        #expect(processor.process("...") == "")
        #expect(processor.process(" . ") == "")
        #expect(processor.process("…") == "")
        #expect(processor.process("?!") == "")
        #expect(processor.process("- ...") == "")
    }

    // MARK: - Punctuation spacing

    @Test
    func removesSpaceBeforePunctuation() {
        #expect(processor.process("hello , world .") == "Hello, world.")
    }

    @Test
    func insertsSpaceAfterPunctuationGluedToAWord() {
        #expect(processor.process("hello,world") == "Hello, world")
        #expect(processor.process("really?yes") == "Really? Yes")
        #expect(processor.process("note:this") == "Note: this")
    }

    @Test
    func splitsTwoSentencesGluedAtADot() {
        #expect(processor.process("the end.Then we left") == "The end. Then we left")
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
    func capitalizesAfterASentenceEndInsideQuotes() {
        #expect(
            processor.process("he said \"stop.\" then left") == "He said \"stop.\" Then left")
    }

    @Test
    func capitalizesStandaloneI() {
        #expect(processor.process("i think i can") == "I think I can")
        #expect(processor.process("i think i'm right") == "I think I'm right")
    }

    @Test
    func leavesEmbeddedLowercaseIAlone() {
        #expect(processor.process("it is in the bin") == "It is in the bin")
    }

    @Test
    func leavesWordsWithInnerCapitalsAlone() {
        #expect(processor.process("iPhone is great") == "iPhone is great")
        #expect(processor.process("done. macOS next") == "Done. macOS next")
    }

    // MARK: - Composition

    @Test
    func fullPipelineOnMessyDictation() {
        #expect(
            processor.process("  the the plan , wait.. we we ship it today !  [Music]")
                == "The plan, wait. We ship it today!")
    }

    @Test
    func fullPipelineKeepsWhatWhisperGotRight() {
        #expect(
            processor.process("so... update CLAUDE.md to 2.5, it is very very important")
                == "So... update CLAUDE.md to 2.5, it is very very important")
    }
}
