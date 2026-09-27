//
//  WordHighlightSurface.swift
//  tesseract
//
//  The port (seam) the Segment Playback loop and `SpeechCoordinator`'s session-level
//  calls drive to render spoken-word highlighting — `show` a fresh segment,
//  `switchText` to the next segment at a crossed Segment Window, hand over the
//  words' starts as the engine times them (`timeWords`, ADR-0077), push the
//  running `updateTotalDuration`, `markSegmentComplete` / `markGenerationComplete`,
//  and `dismiss`. The methods are exactly the surface the real call sites use.
//
//  Same `@MainActor`-sibling shape as `AudioPlayback` (ADR-0003): class-bound,
//  main-actor-isolated, and called *synchronously* on the hot path — deliberately not
//  an actor, since the calls are already main-actor-bound. The production adapter is
//  `SpeechReadAlong`, the one clock the Reader and the Speech Overlay follow
//  (ADR-0076); the test peer `RecordingHighlightSurface` records the call sequence,
//  which is what makes the segment-boundary switch assertable (ADR-0004).
//

import Foundation

@MainActor
protocol WordHighlightSurface: AnyObject {
    /// Show a new utterance's first segment and begin tracking. `playbackTimeProvider`
    /// is the clock the surface samples to pace the highlight.
    func show(text: String, playbackTimeProvider: @escaping () -> TimeInterval)

    /// The next segment's text, arriving ahead of its audio: the surface switches to
    /// it once the playback head crosses its **Segment Window** (`segmentBase`: the
    /// cumulative scheduled duration before it).
    func switchText(_ text: String, segmentBase: TimeInterval)

    /// When words of segment `segment` (0 is the utterance's first) start,
    /// in the utterance's audio time, as the engine learns them: in word
    /// order, from the segment's first word, ahead of their audio playing.
    func timeWords(_ words: [TimedWord], segment: Int)

    /// Everything generated so far: the latest segment's audio ends here.
    func updateTotalDuration(_ duration: TimeInterval)

    /// One segment's generation finished and more remain.
    func markSegmentComplete()

    /// The whole generation finished: let the highlight run to the end and auto-dismiss.
    func markGenerationComplete()

    /// Tear the surface down.
    func dismiss()
}

/// A word of a segment (its place among the segment's words, from 0) and
/// when its sound starts in the utterance's audio (ADR-0077).
nonisolated struct TimedWord: Equatable, Sendable {
    let word: Int
    let start: TimeInterval
}
