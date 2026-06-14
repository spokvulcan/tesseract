//
//  WordHighlightSurface.swift
//  tesseract
//
//  The port (seam) the Segment Playback loop and `SpeechCoordinator`'s session-level
//  calls drive to render spoken-word highlighting — `show` a fresh segment,
//  `switchText` to the next segment at a crossed Segment Window, push the running
//  `updateTotalDuration`, `markSegmentComplete` / `markGenerationComplete`, and
//  `dismiss`. The methods are exactly the surface the real call sites use.
//
//  Same `@MainActor`-sibling shape as `AudioPlayback` (ADR-0003): class-bound,
//  main-actor-isolated, and called *synchronously* on the hot path — deliberately not
//  an actor, since the calls are already main-actor-bound. The production adapter is
//  `TTSNotchPanelController` (the `NSPanel` plus the `TTSWordTracker` it hosts); the
//  test peer `RecordingHighlightSurface` records the call sequence, which is what makes
//  the segment-boundary switch assertable (ADR-0004).
//

import Foundation

@MainActor
protocol WordHighlightSurface: AnyObject {
    /// Show a fresh segment and begin tracking. `playbackTimeProvider` is the clock
    /// the surface samples to pace the highlight.
    func show(
        text: String, tokenCharOffsets: [Int], playbackTimeProvider: @escaping () -> TimeInterval)

    /// Switch to the next segment's text once the playback head crosses the
    /// **Segment Window** (`segmentBase`: the previous segment's cumulative scheduled
    /// duration). Collapses the old `updateText(segmentTimeBase:segmentDurationBase:)`,
    /// which was always passed the same value twice, into one base.
    func switchText(_ text: String, tokenCharOffsets: [Int], segmentBase: TimeInterval)

    /// Push the running cumulative scheduled duration for the segment currently shown.
    func updateTotalDuration(_ duration: TimeInterval)

    /// One segment's generation finished (more remain): align its duration estimate so
    /// the highlight converges to 100% without triggering auto-dismiss.
    func markSegmentComplete()

    /// The whole generation finished: let the highlight run to the end and auto-dismiss.
    func markGenerationComplete()

    /// Tear the surface down.
    func dismiss()
}
