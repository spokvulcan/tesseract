//
//  SpeechReaderTests.swift
//  tesseractTests
//
//  The Reader over the real coordinator, v2 engine and Read-Along: a scripted
//  synthesizer, an in-memory sink with a virtual playback clock, and a
//  scratch document store. The clock only moves when a test moves it, and
//  `tick()` samples it as the Read-Along's timer would.
//

import Foundation
import Testing
import TesseractSpeech

@testable import Tesseract_Agent

@MainActor
struct ReaderHarness {
    let coordinator: SpeechCoordinator
    let synthesizer: ScriptedSpeechSynthesizer
    let playback: InMemoryAudioPlayback
    let readAlong: SpeechReadAlong
    let settings: SettingsManager
    let store: ReaderDocumentStore
    let reader: SpeechReader
    let text: NSString

    init(text: String, bookmark: Int = 0, script: ScriptedSpeechSynthesizer.Script = .init()) async
    {
        self.text = text as NSString
        store = ReaderDocumentStore(directory: makeTempDir("reader"))
        store.save(text: text)
        store.save(bookmark: bookmark, length: (text as NSString).length)
        synthesizer = ScriptedSpeechSynthesizer()
        await synthesizer.configure(script)
        let engine = SpeechEngine(
            model: ModelDefinition.textToSpeechModelSpec, synthesizer: synthesizer)
        playback = InMemoryAudioPlayback()
        readAlong = SpeechReadAlong()
        settings = SettingsManager(store: InMemorySettingsStore())
        coordinator = SpeechCoordinator(
            textExtractor: InMemoryTextExtractor(),
            engine: SpeechEnginePresenter(engine: engine),
            voiceEngineStatus: { .downloaded(sizeOnDisk: 1) },
            playback: playback,
            settings: settings,
            notchOverlay: readAlong,
            pinnedVoices: PinnedVoiceStore(directory: makeTempDir("reader-voices")))
        reader = SpeechReader(
            coordinator: coordinator, readAlong: readAlong, settings: settings, store: store)
        reader.start()
    }

    /// Moves the playback head to `time` and samples it.
    func play(to time: TimeInterval) {
        playback.advance(by: time - playback.currentPlaybackTime())
        readAlong.tick()
    }

    /// The text of the word the page highlights.
    var heardWord: String? { reader.highlight.map { text.substring(with: $0.word) } }

    var heardSentence: String? { reader.highlight.map { text.substring(with: $0.sentence) } }

    func tearDown() {
        coordinator.stop()
        readAlong.dismiss()
    }
}

@MainActor
struct SpeechReaderTests {

    /// Poll until `condition` holds. The minute is a backstop, not a latency
    /// budget: every event of an utterance hops between the engine's actors
    /// and the main actor, and in the first seconds of a parallel run each hop
    /// can wait that long for a thread behind the other suites' work. A
    /// member, so it shadows the shared five-second `waitUntil`, which a
    /// file-level helper loses to for every condition that doesn't await.
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

    private static let threeSentences =
        "The river watched the harbor. A small boat returned home. The keeper climbed the steps."

    @Test func theHeardWordLightsUpInTheText() async {
        let harness = await ReaderHarness(text: Self.threeSentences)
        harness.reader.play()
        #expect(await waitUntil { harness.reader.highlight != nil })
        #expect(harness.reader.isReading)
        #expect(harness.heardWord == "The")
        #expect(harness.heardSentence?.hasPrefix("The river watched the harbor.") == true)

        // All generated: the segment's end is known, so halfway through the
        // audio is halfway through its characters.
        #expect(await waitUntil { harness.playback.finishStreamingCount == 1 })
        harness.play(to: harness.playback.totalScheduledDuration * 0.5)
        let spoken = harness.readAlong.segment.map { $0.words.words[harness.readAlong.word].text }
        #expect(await waitUntil { harness.heardWord == spoken })
        #expect(harness.heardWord == "returned", "word i of the segment is word i of the text")
        #expect(harness.heardSentence?.hasPrefix("A small boat") == true)
        harness.tearDown()
    }

    /// With the engine timing its words (ADR-0077), the page lights the
    /// word whose sound has started, wherever its characters sit.
    @Test func theEnginesWordTimingLightsTheHeardWord() async {
        // Six frames of audio: "The" at 0, "river" at 1, "watched" at 5.
        let harness = await ReaderHarness(
            text: Self.threeSentences,
            script: .init(wordStarts: [
                WordStart(word: 0, frame: 0), WordStart(word: 1, frame: 1),
                WordStart(word: 2, frame: 5),
            ]))
        harness.reader.play()
        #expect(await waitUntil { harness.playback.finishStreamingCount == 1 })
        #expect(harness.playback.totalScheduledDuration == 0.48)

        harness.play(to: 0.3)
        #expect(await waitUntil { harness.heardWord == "river" })
        // 0.41 s is 85% of the audio: evenly spread that would be "home."
        harness.play(to: 0.41)
        #expect(await waitUntil { harness.heardWord == "watched" })
        harness.tearDown()
    }

    @Test func playReadsTheSelectionWhenThereIsOne() async {
        let harness = await ReaderHarness(text: Self.threeSentences)
        let selection = harness.text.range(of: "A small boat returned home.")
        harness.reader.selection = selection
        harness.reader.play()

        #expect(await waitUntil { harness.reader.highlight != nil })
        #expect(await harness.synthesizer.requests.first?.text == "A small boat returned home.")
        #expect(harness.reader.highlight?.word.location == selection.location)
        harness.tearDown()
    }

    /// A jump starts a new utterance while the old one waits for playback
    /// demand, and the reading follows the new one. (The coordinator test
    /// `aSupersededRequestLeavesTheNewOneAlone` pins the race that once
    /// ended the reading here.)
    @Test func aJumpKeepsReadingFromTheNewPlace() async {
        let sentence = "The river watched the harbor all night long. "
        let harness = await ReaderHarness(text: String(repeating: sentence, count: 300))
        harness.reader.play()
        #expect(await waitUntil { harness.reader.highlight != nil })
        // Generation has run 8 s ahead of the head and parked.
        #expect(await waitUntil { harness.playback.totalScheduledDuration >= 8 })
        let first = harness.readAlong.utteranceID

        harness.reader.skip(sentences: 1)
        #expect(
            await waitUntil {
                harness.readAlong.utteranceID != first && harness.reader.highlight != nil
            })
        try? await Task.sleep(for: .milliseconds(200))

        #expect(harness.reader.isReading)
        #expect(harness.coordinator.state != .idle)
        #expect(harness.reader.highlight?.sentence.location == (sentence as NSString).length)
        harness.tearDown()
    }

    @Test func otherSpeechTakingOverEndsTheReading() async {
        let harness = await ReaderHarness(text: Self.threeSentences)
        harness.reader.play()
        #expect(await waitUntil { harness.reader.highlight != nil })

        // The assistant speaks: a different utterance that isn't this reading.
        harness.coordinator.speakText("Something else entirely.")
        #expect(await waitUntil { !harness.reader.isReading })
        #expect(harness.reader.highlight == nil)
        harness.tearDown()
    }

    @Test func stoppingKeepsTheBookmarkAndPlayResumesThere() async {
        let harness = await ReaderHarness(text: Self.threeSentences)
        let second = harness.text.range(of: "A small").location
        harness.reader.play()
        #expect(await waitUntil { harness.playback.finishStreamingCount == 1 })
        harness.play(to: harness.playback.totalScheduledDuration * 0.5)
        #expect(await waitUntil { harness.reader.highlight?.sentence.location == second })

        harness.reader.stop()
        #expect(!harness.reader.isReading)
        #expect(harness.reader.bookmark == second)

        harness.reader.play()
        #expect(await waitUntil { harness.reader.highlight != nil })
        #expect(await harness.synthesizer.requests.last?.text.hasPrefix("A small boat") == true)
        harness.tearDown()
    }

    @Test func readingToTheEndStartsOverNextTime() async {
        let harness = await ReaderHarness(text: Self.threeSentences, bookmark: 30)
        harness.reader.play()
        #expect(await waitUntil { harness.playback.finishStreamingCount == 1 })
        harness.play(to: harness.playback.totalScheduledDuration - 0.01)
        #expect(await waitUntil { harness.heardWord == "steps." })

        // The audio drains: the reading is complete.
        harness.playback.firePlaybackFinished()
        #expect(await waitUntil { !harness.reader.isReading })
        #expect(harness.reader.bookmark == 0)
        #expect(await waitUntil { harness.store.load().bookmark == 0 })
        harness.tearDown()
    }

    @Test func aTapAtRestMovesTheBookmarkToItsSentence() async {
        let harness = await ReaderHarness(text: Self.threeSentences)
        let boat = harness.text.range(of: "boat").location
        harness.reader.jump(to: boat)
        #expect(!harness.reader.isReading)
        #expect(harness.reader.bookmark == harness.text.range(of: "A small").location)
    }

    @Test func aTapWhileReadingJumpsTheReadingThere() async {
        let harness = await ReaderHarness(text: Self.threeSentences)
        harness.reader.play()
        #expect(await waitUntil { harness.reader.highlight != nil })

        harness.reader.jump(to: harness.text.range(of: "climbed").location)
        #expect(
            await waitUntil {
                await harness.synthesizer.requests.last?.text == "The keeper climbed the steps."
            })
        #expect(harness.reader.isReading)
        harness.tearDown()
    }

    /// A text whose language the Library knows reads in it, whatever the
    /// setting says.
    @Test func aReadingSpeaksInTheTextsLanguage() async {
        let harness = await ReaderHarness(text: Self.threeSentences)
        harness.reader.language = TTSLanguage.german.rawValue
        harness.reader.play()
        #expect(await waitUntil { await !harness.synthesizer.requests.isEmpty })
        #expect(await harness.synthesizer.requests.first?.language == "German")
        harness.tearDown()
    }

    @Test func editsBeforeTheBookmarkCarryItAlong() async {
        let text = Self.threeSentences
        let length = (text as NSString).length
        let harness = await ReaderHarness(text: text, bookmark: 30)

        // Ten characters typed before it.
        harness.reader.textDidChange(
            edited: NSRange(location: 4, length: 10), delta: 10, newLength: length + 10)
        #expect(harness.reader.bookmark == 40)

        // An edit after it leaves it.
        harness.reader.textDidChange(
            edited: NSRange(location: 60, length: 3), delta: 3, newLength: length + 13)
        #expect(harness.reader.bookmark == 40)

        // Pasting over everything puts it at the start.
        harness.reader.textDidChange(
            edited: NSRange(location: 0, length: 5), delta: 5 - (length + 13), newLength: 5)
        #expect(harness.reader.bookmark == 0)
        #expect(!harness.reader.isEmpty)
    }
}
