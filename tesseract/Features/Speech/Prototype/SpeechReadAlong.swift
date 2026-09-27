//
//  SpeechReadAlong.swift
//  tesseract
//
//  PROTOTYPE — THROWAWAY (Speech page redesign, 2026-09-27). Never merge.
//
//  One answer to "which word is being heard right now", for live speech and
//  for a replayed take alike, so the page, the overlay and the history all
//  highlight from the same clock.
//
//  Two differences from the shipping notch tracker, both deliberate:
//  - A segment switches when the playback head reaches its start, not when
//    its script arrives. Under lookahead pacing the next script arrives up to
//    8 s early, which is why today's notch jumps to the next paragraph before
//    the current one has finished playing.
//  - The clock publishes only when the word changes (a few times a second),
//    never per frame, and ticks without allocating a Task per frame.
//

import Foundation
import NaturalLanguage
import Observation

// MARK: - Pure mapping

nonisolated enum ReadAlongMap {
    nonisolated struct Segment: Equatable, Sendable {
        let timeline: WordTimeline
        /// Seconds into the take where this segment starts.
        let base: TimeInterval
        /// Where it ends, once its audio has all been generated.
        var end: TimeInterval?
        /// Words before this segment in the spoken text.
        let firstWord: Int

        var wordCount: Int { timeline.words.count }
    }

    /// The word being spoken at `time`: its segment and its index in the
    /// spoken text. A segment still generating ends at a chars-per-second
    /// estimate until its real end is known.
    static func position(
        at time: TimeInterval, in segments: [Segment], charsPerSecond: Double
    ) -> (segment: Int, word: Int)? {
        guard !segments.isEmpty else { return nil }
        var index = 0
        for (i, segment) in segments.enumerated() where segment.base <= time + 0.02 {
            index = i
        }
        let segment = segments[index]
        let chars = segment.timeline.totalCharCount
        guard chars > 0 else { return (index, segment.firstWord) }
        let estimate = Double(chars) / max(charsPerSecond, 1)
        let duration = max((segment.end ?? (segment.base + estimate)) - segment.base, 0.05)
        let progress = min(max((time - segment.base) / duration, 0), 1)
        let local = segment.timeline.activeWordIndex(
            highlightedCharCount: Int(progress * Double(chars)))
        return (index, segment.firstWord + local)
    }
}

// MARK: - The clock

@Observable @MainActor
final class SpeechReadAlong {
    enum Mode: Equatable {
        case idle, live, replay
    }

    private(set) var mode: Mode = .idle
    private(set) var takeID: UUID?
    private(set) var source: SpeechTake.Source?
    private(set) var segments: [ReadAlongMap.Segment] = []
    private(set) var segmentIndex = 0
    /// The word being heard, counted in the spoken text; −1 before the first.
    private(set) var wordIndex = -1
    /// Document words before the spoken text (speech that started mid-text).
    private(set) var wordOffset = 0
    /// What the page asked to speak, when the page asked.
    private(set) var requestText: String?
    private(set) var isGenerationComplete = false

    @ObservationIgnored private var clock: (() -> TimeInterval)?
    @ObservationIgnored private var timer: Timer?
    @ObservationIgnored private var charsPerSecond: Double = 14
    @ObservationIgnored private var finishTask: Task<Void, Never>?

    /// The word in the page's own text, when the page asked for this speech.
    var documentWordIndex: Int? { wordIndex >= 0 ? wordOffset + wordIndex : nil }

    var totalWords: Int { segments.last.map { $0.firstWord + $0.wordCount } ?? 0 }

    /// The playback head, sampled on demand (for progress bars).
    func now() -> TimeInterval { clock?() ?? 0 }

    /// Known once every segment has finished generating.
    var knownDuration: TimeInterval? {
        guard isGenerationComplete else { return nil }
        return segments.last?.end
    }

    /// The segment currently heard, as its words.
    var currentWords: [WordTimeline.Word] {
        guard segments.indices.contains(segmentIndex) else { return [] }
        return segments[segmentIndex].timeline.words
    }

    /// The heard word's index inside `currentWords`.
    var currentLocalWord: Int {
        guard segments.indices.contains(segmentIndex) else { return -1 }
        return wordIndex - segments[segmentIndex].firstWord
    }

    // MARK: Live

    func prepareLive(
        takeID: UUID?, wordOffset: Int, requestText: String?, source: SpeechTake.Source
    ) {
        stopTimer()
        finishTask?.cancel()
        mode = .idle
        clock = nil
        self.takeID = takeID
        self.wordOffset = wordOffset
        self.requestText = requestText
        self.source = source
        segments = []
        segmentIndex = 0
        wordIndex = -1
        isGenerationComplete = false
    }

    func startLiveClock(_ clock: @escaping () -> TimeInterval) {
        self.clock = clock
        mode = .live
        startTimer()
    }

    func appendSegment(_ segment: ReadAlongMap.Segment) {
        segments.append(segment)
    }

    func setScheduledEnd(_ cumulative: TimeInterval) {
        guard !segments.isEmpty else { return }
        let last = segments.count - 1
        segments[last].end = cumulative
        let span = cumulative - segments[last].base
        let chars = segments[last].timeline.totalCharCount
        if span > 0.5, chars > 0 {
            charsPerSecond = 0.7 * (Double(chars) / span) + 0.3 * charsPerSecond
        }
    }

    func markGenerationComplete() {
        isGenerationComplete = true
    }

    /// The audio drained: light the last word, then go idle.
    func playbackFinished() {
        guard mode == .live else { return }
        stopTimer()
        if totalWords > 0 { publish(segment: max(segments.count - 1, 0), word: totalWords - 1) }
        finishTask?.cancel()
        finishTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(600))
            guard !Task.isCancelled, let self, self.mode == .live else { return }
            self.mode = .idle
            self.clock = nil
        }
    }

    func endLive() {
        guard mode == .live else { return }
        stopTimer()
        mode = .idle
        clock = nil
    }

    // MARK: Replay

    func beginReplay(take: SpeechTake, clock: @escaping () -> TimeInterval) {
        stopTimer()
        finishTask?.cancel()
        takeID = take.id
        source = take.source
        segments = take.segments
        wordOffset = take.wordOffset
        requestText = take.text
        isGenerationComplete = true
        segmentIndex = 0
        wordIndex = -1
        self.clock = clock
        mode = .replay
        startTimer()
    }

    func endReplay() {
        guard mode == .replay else { return }
        stopTimer()
        mode = .idle
        clock = nil
        wordIndex = -1
    }

    // MARK: Ticking

    private func startTimer() {
        stopTimer()
        let timer = Timer(timeInterval: 1.0 / 30.0, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.tick() }
        }
        RunLoop.main.add(timer, forMode: .common)
        self.timer = timer
        tick()
    }

    private func stopTimer() {
        timer?.invalidate()
        timer = nil
    }

    private func tick() {
        guard let clock, !segments.isEmpty else { return }
        guard
            let position = ReadAlongMap.position(
                at: clock(), in: segments, charsPerSecond: charsPerSecond)
        else { return }
        publish(segment: position.segment, word: position.word)
    }

    /// Observation fires on every assignment, so only assign on change.
    private func publish(segment: Int, word: Int) {
        if segment != segmentIndex { segmentIndex = segment }
        if word != wordIndex { wordIndex = word }
    }
}

// MARK: - A page's text, as words, sentences and paragraphs

/// The page's text split once into what the read-along highlights. Built off
/// the typing path (on Speak, or when switching to a reading view).
nonisolated struct ReadingDocument: Equatable, Sendable {
    nonisolated struct Word: Equatable, Sendable {
        let range: Range<String.Index>
        let sentence: Int
        let paragraph: Int
    }

    nonisolated struct Sentence: Equatable, Sendable {
        let range: Range<String.Index>
        let firstWord: Int
        let wordCount: Int
        let paragraph: Int
    }

    nonisolated struct Paragraph: Equatable, Sendable {
        let range: Range<String.Index>
        let firstWord: Int
        let wordCount: Int
        let firstSentence: Int
        let sentenceCount: Int
    }

    let text: String
    let words: [Word]
    let sentences: [Sentence]
    let paragraphs: [Paragraph]

    init(text: String) {
        self.text = text

        // Paragraphs: runs of text separated by blank lines or line breaks.
        var paragraphRanges: [Range<String.Index>] = []
        text.enumerateSubstrings(in: text.startIndex..<text.endIndex, options: .byParagraphs) {
            substring, range, _, _ in
            if let substring, !substring.trimmingCharacters(in: .whitespaces).isEmpty {
                paragraphRanges.append(range)
            }
        }

        // Sentences within the whole text.
        var sentenceRanges: [Range<String.Index>] = []
        let tokenizer = NLTokenizer(unit: .sentence)
        tokenizer.string = text
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            sentenceRanges.append(range)
            return true
        }

        // Words: whitespace-separated, the same definition the engine and
        // the Word Timeline use, so indices line up with the read-along.
        var words: [Word] = []
        var sentenceIndex = 0
        var paragraphIndex = 0
        var index = text.startIndex
        while index < text.endIndex {
            while index < text.endIndex, text[index].isWhitespace || text[index].isNewline {
                index = text.index(after: index)
            }
            guard index < text.endIndex else { break }
            let start = index
            while index < text.endIndex, !(text[index].isWhitespace || text[index].isNewline) {
                index = text.index(after: index)
            }
            while sentenceIndex + 1 < sentenceRanges.count,
                sentenceRanges[sentenceIndex].upperBound <= start
            {
                sentenceIndex += 1
            }
            while paragraphIndex + 1 < paragraphRanges.count,
                paragraphRanges[paragraphIndex].upperBound <= start
            {
                paragraphIndex += 1
            }
            words.append(
                Word(range: start..<index, sentence: sentenceIndex, paragraph: paragraphIndex))
        }
        self.words = words

        var sentences: [Sentence] = []
        for (i, range) in sentenceRanges.enumerated() {
            let first = words.firstIndex { $0.sentence == i } ?? words.count
            let count = words[first...].prefix { $0.sentence == i }.count
            guard count > 0 else { continue }
            sentences.append(
                Sentence(
                    range: range, firstWord: first, wordCount: count,
                    paragraph: words[first].paragraph))
        }
        self.sentences = sentences

        var paragraphs: [Paragraph] = []
        for (i, range) in paragraphRanges.enumerated() {
            let first = words.firstIndex { $0.paragraph == i } ?? words.count
            let count = words[first...].prefix { $0.paragraph == i }.count
            let firstSentence = sentences.firstIndex { $0.paragraph == i } ?? 0
            let sentenceCount = sentences.filter { $0.paragraph == i }.count
            guard count > 0 else { continue }
            paragraphs.append(
                Paragraph(
                    range: range, firstWord: first, wordCount: count,
                    firstSentence: firstSentence, sentenceCount: sentenceCount))
        }
        self.paragraphs = paragraphs
    }

    func sentence(containingWord word: Int) -> Int? {
        guard words.indices.contains(word) else { return nil }
        return sentences.firstIndex { word >= $0.firstWord && word < $0.firstWord + $0.wordCount }
    }

    /// The text from the start of sentence `index` to the end.
    func text(fromSentence index: Int) -> (text: String, wordOffset: Int)? {
        guard sentences.indices.contains(index) else { return nil }
        let sentence = sentences[index]
        return (String(text[sentence.range.lowerBound...]), sentence.firstWord)
    }
}
