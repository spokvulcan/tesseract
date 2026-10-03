//
//  SpeechReader.swift
//  tesseract
//
//  The **Reader**: the Speech page's model (ADR-0076). It owns the page's
//  text (the text view edits it; this keeps it saved), the **Bookmark**
//  where reading resumes, and the reading in progress: it starts speech from
//  the bookmark or a selection, follows the Read-Along, and turns each heard
//  word into a range of the text for the page to highlight.
//
//  Nothing here walks the whole document. A reading maps each spoken
//  segment onto the text by counting its words forward from where the
//  previous one ended (the engine and the Word Timeline split words the same
//  way), and finds sentences only inside the segment at hand. A whole book
//  costs what a page costs.
//

import Foundation
import Observation
import os

@Observable @MainActor
final class SpeechReader {
    /// The word being heard and its sentence, as ranges of the text.
    struct Highlight: Equatable {
        let word: NSRange
        let sentence: NSRange
    }

    /// UTF-16 length of the text (the text view's units).
    private(set) var length: Int
    /// Changes only when the text becomes empty or stops being empty, so
    /// the page's empty state doesn't redraw on every keystroke.
    private(set) var isEmpty: Bool
    /// Where reading resumes: a UTF-16 offset into the text.
    private(set) var bookmark: Int
    /// A reading is in progress (playing, paused, or starting up).
    private(set) var isReading = false
    private(set) var highlight: Highlight?
    /// The Speech page is on screen in the key window. The page keeps it
    /// current; the overlay hides while it holds, in its automatic scope.
    var isInFront = false

    /// The text as last saved or synced. While a text view shows the
    /// document, `liveText` reads its current text instead.
    @ObservationIgnored private(set) var text: String
    /// The showing text view's current text; nil when none shows it.
    @ObservationIgnored var liveText: (() -> String)?
    /// The text view's selection: Play reads a selection when there is one.
    @ObservationIgnored var selection: NSRange?
    /// The text's language (a `TTSLanguage` raw value) when it is known, as
    /// the phone's Library knows each text's; nil reads in the setting's.
    @ObservationIgnored var language: String?

    @ObservationIgnored private let coordinator: SpeechCoordinator
    @ObservationIgnored private let readAlong: SpeechReadAlong
    @ObservationIgnored private let settings: any SpeechSettings
    @ObservationIgnored private let store: ReaderDocumentStore
    @ObservationIgnored private var session: ReadingSession?
    @ObservationIgnored private var saveTask: Task<Void, Never>?
    @ObservationIgnored private var observations: [Task<Void, Never>] = []
    @ObservationIgnored private var lastSavedBookmark: Int

    init(
        coordinator: SpeechCoordinator, readAlong: SpeechReadAlong, settings: any SpeechSettings,
        store: ReaderDocumentStore = ReaderDocumentStore()
    ) {
        self.coordinator = coordinator
        self.readAlong = readAlong
        self.settings = settings
        self.store = store
        let saved = store.load()
        self.text = saved.text
        self.length = (saved.text as NSString).length
        self.isEmpty = saved.text.isEmpty
        self.bookmark = saved.bookmark
        self.lastSavedBookmark = saved.bookmark
    }

    /// Starts following the Read-Along and the coordinator.
    func start() {
        guard observations.isEmpty else { return }
        observations.append(
            Task { [weak self] in
                guard let self else { return }
                for await _ in Observations({
                    (self.readAlong.utteranceID, self.readAlong.segment?.index, self.readAlong.word)
                }) {
                    self.followReadAlong()
                }
            })
        observations.append(
            Task { [weak self] in
                guard let self else { return }
                for await state in Observations({ self.coordinator.state }) {
                    self.follow(state)
                }
            })
    }

    // MARK: - Commands

    var isPaused: Bool {
        if case .paused = coordinator.state { return true }
        return false
    }

    /// Reads the selection if there is one, else from the bookmark to the
    /// end (from the top once the bookmark has reached the end).
    func play() {
        let text = currentText() as NSString
        if let selection, selection.length > 0, selection.upperBound <= text.length,
            ReaderText.hasWords(text, in: selection)
        {
            begin(text, range: selection)
            return
        }
        let resumable =
            bookmark < text.length
            && ReaderText.hasWords(
                text, in: NSRange(location: bookmark, length: text.length - bookmark))
        read(from: resumable ? bookmark : 0, in: text)
    }

    /// Reads from the sentence holding `offset` to the end: a click, the
    /// progress bar, "Read from Here".
    func read(from offset: Int) {
        read(from: offset, in: currentText() as NSString)
    }

    /// A tap on the text at `offset`: reading jumps to the sentence there,
    /// or, at rest, the bookmark moves to it.
    func jump(to offset: Int) {
        let text = (session?.text) ?? (currentText() as NSString)
        guard text.length > 0 else { return }
        let start = ReaderText.sentenceStart(
            containing: min(max(offset, 0), text.length - 1), in: text)
        if isReading { begin(text, from: start) } else { setBookmark(start) }
    }

    func togglePause() {
        if isPaused { coordinator.resume() } else { coordinator.pause() }
    }

    func stop() {
        coordinator.stop()
        end(completed: false)
    }

    /// One sentence back or forward. While reading, reading jumps there; at
    /// rest, the bookmark moves.
    func skip(sentences delta: Int) {
        let text = (session?.text) ?? (currentText() as NSString)
        let here = highlight?.sentence.location ?? bookmark
        let target: Int? =
            delta < 0
            ? ReaderText.previousSentenceStart(before: here, in: text)
            : ReaderText.nextSentenceStart(after: here, in: text)
        guard let target else { return }
        if isReading { begin(text, from: target) } else { setBookmark(target) }
    }

    /// Jumps to `fraction` of the text: reading continues there, or the
    /// bookmark moves there.
    func seek(toFraction fraction: Double) {
        let text = (session?.text) ?? (currentText() as NSString)
        let offset = Int(Double(text.length) * min(max(fraction, 0), 1))
        let start = ReaderText.sentenceStart(containing: offset, in: text)
        if isReading { begin(text, from: start) } else { setBookmark(start) }
    }

    // MARK: - Progress

    /// Where reading is (or resumes), as a fraction of the text.
    var progress: Double {
        guard length > 0 else { return 0 }
        return Double(highlight?.word.location ?? bookmark) / Double(length)
    }

    /// Time to the end of the text at the learned pace and the chosen speed.
    var timeLeft: TimeInterval {
        let position = highlight?.word.location ?? bookmark
        let rate = max(settings.ttsPlaybackRate, 0.5)
        return Double(max(length - position, 0)) / max(readAlong.charsPerSecond, 1) / rate
    }

    // MARK: - Edits from the text view

    /// The text changed by `delta` characters at `edited` (after the edit):
    /// shift the bookmark with the text before it, and save soon.
    func textDidChange(edited: NSRange, delta: Int, newLength: Int) {
        length = newLength
        if isEmpty != (newLength == 0) { isEmpty = newLength == 0 }
        // Where the replaced text ended before the edit.
        let replacedEnd = edited.location + edited.length - delta
        if bookmark >= replacedEnd {
            bookmark += delta
        } else if bookmark > edited.location {
            // Inside text that was replaced (a paste over everything).
            bookmark = edited.location
        }
        bookmark = min(max(bookmark, 0), newLength)
        scheduleSave()
    }

    /// The showing text view is going away: keep its text.
    func textViewWillClose() {
        guard let liveText else { return }
        text = liveText()
        self.liveText = nil
        saveNow()
    }

    // MARK: - Reading

    private func currentText() -> String {
        liveText?() ?? text
    }

    private func read(from offset: Int, in text: NSString) {
        guard text.length > 0 else { return }
        let start = ReaderText.sentenceStart(containing: offset, in: text)
        begin(text, from: start)
    }

    private func begin(_ text: NSString, from start: Int) {
        begin(text, range: NSRange(location: start, length: text.length - start))
    }

    /// Speaks `range` of a snapshot of the text. The snapshot stays with the
    /// reading, so offsets hold while the text view is read-only.
    private func begin(_ text: NSString, range: NSRange) {
        let snapshot = text.copy() as? NSString ?? text
        guard ReaderText.hasWords(snapshot, in: range) else { return }
        let start = ReaderText.firstReadable(in: snapshot, from: range.location)
        let spoken = NSRange(location: start, length: range.upperBound - start)
        session = ReadingSession(text: snapshot, range: spoken)
        isReading = true
        highlight = nil
        setBookmark(start)
        coordinator.speakText(
            snapshot.substring(with: spoken), userInitiated: true, language: language)
    }

    /// Maps the heard word onto the text, claiming the utterance first: it is
    /// this reading's if its opening words are the words where it started.
    private func followReadAlong() {
        guard var session else { return }
        guard readAlong.isActive, let segment = readAlong.segment else { return }
        if session.utteranceID == nil {
            guard segment.index == 0, session.opens(with: segment) else { return }
            session.utteranceID = readAlong.utteranceID
        } else if session.utteranceID != readAlong.utteranceID {
            // Other speech took over (the assistant, the hotkey).
            end(completed: false)
            return
        }
        guard let mapped = session.map(segment) else {
            self.session = session
            return
        }
        self.session = session
        let word = mapped.words[min(max(readAlong.word, 0), mapped.words.count - 1)]
        let sentence = mapped.sentences.first { NSLocationInRange(word.location, $0) } ?? word
        let next = Highlight(word: word, sentence: sentence)
        if next != highlight { highlight = next }
        if sentence.location != bookmark { setBookmark(sentence.location, persist: false) }
    }

    private func follow(_ state: SpeechState) {
        guard session != nil else { return }
        switch state {
        case .idle:
            // Finished, stopped, or failed before speaking.
            let completed = highlight.map { session?.isDone(after: $0.word) ?? false } ?? false
            end(completed: completed)
        case .error:
            end(completed: false)
        default:
            break
        }
    }

    private func end(completed: Bool) {
        guard let session else { return }
        let finishedText = completed && session.range.upperBound >= session.text.length
        self.session = nil
        isReading = false
        highlight = nil
        // Read to the end of the text: start over next time.
        if finishedText { bookmark = 0 }
        saveBookmark()
    }

    // MARK: - Saving

    private func setBookmark(_ offset: Int, persist: Bool = true) {
        bookmark = min(max(offset, 0), length)
        // While reading, save only as sentences pass every so often.
        if persist || abs(bookmark - lastSavedBookmark) > 2_000 { saveBookmark() }
    }

    private func saveBookmark() {
        lastSavedBookmark = bookmark
        let store = store
        let (offset, length) = (bookmark, length)
        Task.detached(priority: .utility) { store.save(bookmark: offset, length: length) }
    }

    private func scheduleSave() {
        saveTask?.cancel()
        saveTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(1))
            guard !Task.isCancelled else { return }
            self?.saveNow()
        }
    }

    private func saveNow() {
        saveTask?.cancel()
        saveTask = nil
        let snapshot = currentText()
        text = snapshot
        let store = store
        let (offset, length) = (bookmark, (snapshot as NSString).length)
        Task.detached(priority: .utility) {
            store.save(text: snapshot)
            store.save(bookmark: offset, length: length)
        }
    }
}

/// One reading: the text it speaks and where each of its segments sits.
private struct ReadingSession {
    let text: NSString
    /// The spoken part of the text.
    let range: NSRange
    /// Set once the Read-Along shows this reading's first segment.
    var utteranceID: UUID?
    private var mapped: [Int: MappedSegment] = [:]
    /// Where the next unmapped word starts, and how many words precede it.
    private var cursor: Int
    private var wordsBeforeCursor = 0

    struct MappedSegment {
        let words: [NSRange]
        let sentences: [NSRange]
    }

    init(text: NSString, range: NSRange) {
        self.text = text
        self.range = range
        self.cursor = range.location
    }

    /// Nothing is left to read after `word`.
    func isDone(after word: NSRange) -> Bool {
        let rest = NSRange(
            location: word.upperBound, length: max(range.upperBound - word.upperBound, 0))
        return !ReaderText.hasWords(text, in: rest)
    }

    /// The utterance opens with the words this reading starts with.
    func opens(with segment: ReadAlongTimeline.Segment) -> Bool {
        let expected = segment.words.words.prefix(3).map(\.text)
        let found = ReaderText.words(in: text, from: range.location, count: expected.count)
            .map { text.substring(with: $0) }
        return !expected.isEmpty && expected == found
    }

    /// Where `segment`'s words are in the text: counted forward from the
    /// last mapped word, with the segment's own word offset as the check.
    mutating func map(_ segment: ReadAlongTimeline.Segment) -> MappedSegment? {
        if let known = mapped[segment.index] { return known }
        // Skip words of segments passed unseen.
        if segment.firstWord > wordsBeforeCursor {
            let skipped = ReaderText.words(
                in: text, from: cursor, count: segment.firstWord - wordsBeforeCursor)
            cursor = skipped.last?.upperBound ?? cursor
            wordsBeforeCursor = segment.firstWord
        }
        let count = segment.words.words.count
        let words = ReaderText.words(in: text, from: cursor, count: count)
        guard !words.isEmpty, words.first!.location < range.upperBound else { return nil }
        if let first = segment.words.words.first?.text, text.substring(with: words[0]) != first {
            Log.speech.info("[Reader] segment \(segment.index) starts off its expected word")
        }
        let span = NSRange(
            location: words[0].location,
            length: words[words.count - 1].upperBound - words[0].location)
        let result = MappedSegment(
            words: words, sentences: ReaderText.sentences(in: text, range: span))
        mapped[segment.index] = result
        // Older segments are never asked for again.
        mapped = mapped.filter { $0.key >= segment.index - 1 }
        cursor = words[words.count - 1].upperBound
        wordsBeforeCursor = segment.firstWord + words.count
        return result
    }
}
