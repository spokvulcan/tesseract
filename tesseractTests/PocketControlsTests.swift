//
//  PocketControlsTests.swift
//  tesseractTests
//
//  Reading in the pocket (#515): an interruption stops the reading at the
//  heard sentence and starts it again there; losing the headphones stops
//  it; the lock screen's buttons play, pause and skip by sentence.
//

import Foundation
import Testing
import TesseractSpeech

@testable import Tesseract_Agent

@MainActor
struct PocketControlsTests {
    private static let threeSentences =
        "The river watched the harbor. A small boat returned home. The keeper climbed the steps."

    private func waitUntil(
        timeout: Duration = .seconds(60), _ condition: @MainActor () async -> Bool
    ) async -> Bool {
        let deadline = ContinuousClock.now + timeout
        while ContinuousClock.now < deadline {
            if await condition() { return true }
            try? await Task.sleep(for: .milliseconds(10))
        }
        return await condition()
    }

    /// A reading heard into its second sentence.
    private func readingInTheSecondSentence() async -> (ReaderHarness, PocketControls) {
        let harness = await ReaderHarness(text: Self.threeSentences)
        let controls = PocketControls { harness.reader }
        harness.reader.play()
        _ = await waitUntil { harness.playback.finishStreamingCount == 1 }
        harness.play(to: harness.playback.totalScheduledDuration * 0.5)
        _ = await waitUntil { harness.heardSentence?.hasPrefix("A small boat") == true }
        return (harness, controls)
    }

    @Test func aCallStopsTheReadingAndItGoesOnFromTheSameSentence() async {
        let (harness, controls) = await readingInTheSecondSentence()

        controls.handle(.interruptionBegan)
        #expect(!harness.reader.isReading)
        #expect(harness.reader.bookmark == harness.text.range(of: "A small").location)

        controls.handle(.interruptionEnded(shouldResume: true))
        #expect(await waitUntil { harness.reader.isReading })
        #expect(
            await waitUntil {
                await harness.synthesizer.requests.last?.text.hasPrefix("A small boat") == true
            })
        harness.tearDown()
    }

    @Test func anInterruptionTheSystemDoesntResumeStaysStopped() async {
        let (harness, controls) = await readingInTheSecondSentence()
        controls.handle(.interruptionBegan)
        controls.handle(.interruptionEnded(shouldResume: false))
        try? await Task.sleep(for: .milliseconds(100))
        #expect(!harness.reader.isReading)
        harness.tearDown()
    }

    @Test func anInterruptionWhilePausedDoesntStartReading() async {
        let (harness, controls) = await readingInTheSecondSentence()
        harness.reader.togglePause()
        controls.handle(.interruptionBegan)
        controls.handle(.interruptionEnded(shouldResume: true))
        try? await Task.sleep(for: .milliseconds(100))
        #expect(!harness.reader.isReading)
        harness.tearDown()
    }

    @Test func losingTheHeadphonesStopsTheReading() async {
        let (harness, controls) = await readingInTheSecondSentence()
        controls.handle(.outputLost)
        #expect(!harness.reader.isReading)
        #expect(harness.reader.bookmark == harness.text.range(of: "A small").location)
        harness.tearDown()
    }

    @Test func thePreviousAndNextButtonsSkipBySentence() async {
        let (harness, controls) = await readingInTheSecondSentence()
        controls.handle(.remote(.nextTrack))
        #expect(
            await waitUntil {
                await harness.synthesizer.requests.last?.text == "The keeper climbed the steps."
            })
        controls.handle(.remote(.previousTrack))
        #expect(
            await waitUntil {
                await harness.synthesizer.requests.last?.text.hasPrefix("A small boat") == true
            })
        harness.tearDown()
    }

    @Test func theLockScreensPauseAndPlayGoOnFromTheHeardSentence() async {
        let (harness, controls) = await readingInTheSecondSentence()
        controls.handle(.remote(.pause))
        #expect(!harness.reader.isReading)

        controls.handle(.remote(.togglePlayPause))
        #expect(await waitUntil { harness.reader.isReading })
        #expect(
            await waitUntil {
                await harness.synthesizer.requests.last?.text.hasPrefix("A small boat") == true
            })
        harness.tearDown()
    }

    /// A pause held in the app becomes a stop at the heard sentence once the
    /// app leaves the screen, so coming back never resumes a dead stream.
    @Test func aPauseInTheBackgroundBecomesAStopAtTheSentence() async {
        let (harness, controls) = await readingInTheSecondSentence()
        harness.reader.togglePause()
        controls.handle(.movedToBackground)
        #expect(!harness.reader.isReading)
        #expect(harness.reader.bookmark == harness.text.range(of: "A small").location)
        harness.tearDown()
    }

    @Test func aPlayingReadingCarriesOnInTheBackground() async {
        let (harness, controls) = await readingInTheSecondSentence()
        controls.handle(.movedToBackground)
        #expect(harness.reader.isReading)
        #expect(!harness.reader.isPaused)
        harness.tearDown()
    }
}

struct ThermalPolicyTests {
    @Test func eachThermalStateDecidesTheVoice() {
        #expect(ThermalPolicy.decision(for: .nominal) == .neuralVoice)
        #expect(ThermalPolicy.decision(for: .fair) == .neuralVoice)
        #expect(ThermalPolicy.decision(for: .serious) == .systemVoice)
        #expect(ThermalPolicy.decision(for: .critical) == .pause)
    }

    @Test func theReaderSaysWhyWheneverTheVoiceChanges() {
        #expect(ThermalPolicy.notice(for: .neuralVoice) == nil)
        #expect(ThermalPolicy.notice(for: .systemVoice) != nil)
        #expect(ThermalPolicy.notice(for: .pause) != nil)
    }
}
