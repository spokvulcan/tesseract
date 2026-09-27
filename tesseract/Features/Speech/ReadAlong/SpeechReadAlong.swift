//
//  SpeechReadAlong.swift
//  tesseract
//
//  The **Read-Along** (ADR-0076): which word of the speech now playing is
//  being heard, on one clock that the Reader and the Speech Overlay both
//  follow. It is the production **Word Highlight Surface**: the coordinator
//  hands it each Segment Script as the script arrives, and the playback
//  head, not the arrival, decides when a segment starts. Under lookahead
//  pacing a script arrives up to 8 s before its audio plays; the old notch
//  switched text on arrival and ran ahead of the voice.
//
//  The clock samples the playback head 30 times a second but publishes only
//  when the heard word changes, a few times a second, so a view that follows
//  it redraws at word pace, not frame pace.
//

import Foundation
import Observation

// MARK: - Timing

/// The segments of one utterance placed on its audio timeline: where each
/// starts, where it ends once generated, and the word heard at a moment.
/// Pure; `SpeechReadAlong` drives it.
nonisolated struct ReadAlongTimeline: Equatable, Sendable {
    nonisolated struct Segment: Equatable, Sendable {
        /// The segment's position in the utterance, from 0.
        let index: Int
        let text: String
        let words: WordTimeline
        /// Seconds into the utterance where the segment's audio starts.
        let start: TimeInterval
        /// Where it ends, once all its audio has been generated.
        var end: TimeInterval?
        /// Words before this segment in the utterance.
        let firstWord: Int
        /// When the end became known while the segment was already playing
        /// on the estimated pace, and the characters heard by then.
        var handover: Handover?

        /// Characters heard by `time`: at the reading pace until the end is
        /// known, then evenly to the end, from wherever the pace had got to.
        /// Continuous, so the word never jumps back when a segment turns out
        /// longer than estimated.
        func heardChars(at time: TimeInterval, charsPerSecond: Double) -> Double {
            let chars = Double(words.totalCharCount)
            guard let end else { return (time - start) * charsPerSecond }
            let from = handover ?? Handover(time: start, chars: 0)
            guard end - from.time > 0.05 else { return chars }
            return from.chars + (time - from.time) / (end - from.time) * (chars - from.chars)
        }
    }

    nonisolated struct Handover: Equatable, Sendable {
        let time: TimeInterval
        let chars: Double
    }

    /// The latest segments, oldest first. A long reading keeps only a window
    /// (`prune`), so a book-length utterance holds a few segments, not all.
    private(set) var segments: [Segment] = []

    /// Reading pace, learned from finished segments; places the word inside
    /// a segment whose end is not yet known.
    private(set) var charsPerSecond: Double

    /// `charsPerSecond`: the pace learned from earlier readings, if any.
    init(charsPerSecond: Double = 14) {
        self.charsPerSecond = charsPerSecond
    }

    mutating func append(text: String, start: TimeInterval) {
        let index = segments.last.map { $0.index + 1 } ?? 0
        let firstWord = segments.last.map { $0.firstWord + $0.words.words.count } ?? 0
        segments.append(
            Segment(
                index: index, text: text, words: WordTimeline(text: text), start: start,
                end: nil, firstWord: firstWord))
    }

    /// `cumulative` is everything generated so far: the latest segment ends
    /// there. `now` is the playback head, in case the segment is already
    /// playing (the first one always is: playback starts with its first
    /// audio).
    mutating func setEnd(_ cumulative: TimeInterval, at now: TimeInterval) {
        guard let last = segments.indices.last, segments[last].end == nil else { return }
        var segment = segments[last]
        let chars = Double(segment.words.totalCharCount)
        if now > segment.start {
            let heard = min(
                max(segment.heardChars(at: now, charsPerSecond: charsPerSecond), 0), chars)
            segment.handover = Handover(time: now, chars: heard)
        }
        segment.end = cumulative
        segments[last] = segment
        let span = cumulative - segment.start
        if span > 0.5, chars > 0 {
            charsPerSecond = 0.7 * (chars / span) + 0.3 * charsPerSecond
        }
    }

    /// The segment (by position in `segments`) and the word within it heard
    /// at `time`. A segment starts when the playback head reaches its start,
    /// never earlier; inside it, words are placed in proportion to their
    /// characters over the segment's duration.
    func position(at time: TimeInterval) -> (segment: Int, word: Int)? {
        guard !segments.isEmpty else { return nil }
        // The last segment whose audio has started (binary search: starts ascend).
        var low = 0
        var high = segments.count - 1
        while low < high {
            let mid = (low + high + 1) / 2
            if segments[mid].start <= time + 0.02 { low = mid } else { high = mid - 1 }
        }
        let segment = segments[low]
        let chars = Double(segment.words.totalCharCount)
        guard chars > 0 else { return (low, 0) }
        let heard = min(
            max(segment.heardChars(at: time, charsPerSecond: max(charsPerSecond, 1)), 0), chars)
        return (low, segment.words.activeWordIndex(highlightedCharCount: Int(heard)))
    }

    /// Whether the head has passed the end of everything generated.
    func isFinished(at time: TimeInterval) -> Bool {
        guard let end = segments.last?.end else { return false }
        return time >= end - 0.05
    }

    /// Drops segments that ended before the one at `keeping`, keeping one
    /// behind it.
    mutating func prune(keeping position: Int) {
        let drop = position - 1
        guard drop > 0 else { return }
        segments.removeFirst(drop)
    }
}

// MARK: - The clock

@Observable @MainActor
final class SpeechReadAlong: WordHighlightSurface {
    /// New for every utterance shown, so a follower can tell readings apart.
    private(set) var utteranceID = UUID()
    /// Speech is being read and followed: from its first segment until its
    /// audio ends or it is stopped.
    private(set) var isActive = false
    /// The segment being heard.
    private(set) var segment: ReadAlongTimeline.Segment?
    /// The word being heard, counted within `segment`.
    private(set) var word = 0

    @ObservationIgnored private var timeline = ReadAlongTimeline()
    @ObservationIgnored private var clock: (() -> TimeInterval)?
    @ObservationIgnored private var timer: Timer?
    @ObservationIgnored private var isGenerationComplete = false
    @ObservationIgnored private var finishTask: Task<Void, Never>?

    /// The playback head, sampled on demand (a progress label, not per frame).
    func now() -> TimeInterval { clock?() ?? 0 }

    /// Characters of text spoken per second, as learned so far. It changes
    /// about once a segment, so a time-left label can follow it.
    private(set) var charsPerSecond = ReadAlongTimeline().charsPerSecond

    // MARK: Word Highlight Surface

    func show(text: String, playbackTimeProvider: @escaping () -> TimeInterval) {
        finishTask?.cancel()
        timeline = ReadAlongTimeline(charsPerSecond: timeline.charsPerSecond)
        timeline.append(text: text, start: 0)
        clock = playbackTimeProvider
        isGenerationComplete = false
        utteranceID = UUID()
        segment = timeline.segments.first
        word = 0
        isActive = true
        startTimer()
    }

    func switchText(_ text: String, segmentBase: TimeInterval) {
        timeline.append(text: text, start: segmentBase)
    }

    func updateTotalDuration(_ duration: TimeInterval) {
        timeline.setEnd(duration, at: now())
        if timeline.charsPerSecond != charsPerSecond { charsPerSecond = timeline.charsPerSecond }
    }

    func markSegmentComplete() {}

    func markGenerationComplete() {
        isGenerationComplete = true
    }

    func dismiss() {
        finishTask?.cancel()
        finish()
    }

    // MARK: Ticking

    private func startTimer() {
        timer?.invalidate()
        let timer = Timer(timeInterval: 1.0 / 30.0, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.tick() }
        }
        RunLoop.main.add(timer, forMode: .common)
        self.timer = timer
    }

    /// Samples the playback head. The timer calls it 30 times a second;
    /// tests call it after moving their own clock.
    func tick() {
        guard let clock, let position = timeline.position(at: clock()) else { return }
        let heard = timeline.segments[position.segment]
        // Observation fires on every assignment, so assign only on change.
        if heard.index != segment?.index {
            segment = heard
            word = position.word
            timeline.prune(keeping: position.segment)
        } else if position.word > word {
            // Within a segment the word only moves on: the head never runs
            // backwards in one utterance (a jump starts a new one).
            word = position.word
        }
        if isGenerationComplete, timeline.isFinished(at: clock()), finishTask == nil {
            // Leave the last word lit a moment, then let go.
            finishTask = Task { [weak self] in
                try? await Task.sleep(for: .milliseconds(500))
                guard !Task.isCancelled else { return }
                self?.finish()
            }
        }
    }

    private func finish() {
        timer?.invalidate()
        timer = nil
        finishTask = nil
        clock = nil
        isActive = false
    }
}
