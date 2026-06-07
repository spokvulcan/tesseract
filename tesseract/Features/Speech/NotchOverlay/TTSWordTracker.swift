//
//  TTSWordTracker.swift
//  tesseract
//
//  The @Observable @MainActor stateful driver of the pure `WordTimeline`. It owns
//  the 60fps Timer, the injected `playbackTimeProvider` clock seam, the monotonic
//  `recognizedCharCount` and the other published view state the notch overlay reads,
//  and the per-segment carry-over (the smoothing `Pacing`, the static learned
//  chars/sec, the segment time window). It delegates the per-tick pacing *fold* to
//  `WordTimeline`, keeping only the cross-segment estimate model here (the duration
//  seed, the learned chars/sec, the smoothing carry) — the Chat Transcript / Chat
//  Transcript Controller shape, one layer over in the speech feature.
//

import Foundation
import os

@Observable
@MainActor
final class TTSWordTracker {
    // MARK: - Public state (drives the overlay view)

    private(set) var recognizedCharCount: Int = 0
    private(set) var isGenerationComplete: Bool = false
    var shouldDismiss: Bool = false {
        didSet {
            if shouldDismiss != oldValue {
                Log.speech.info("[WordTracker] shouldDismiss changed: \(oldValue) → \(self.shouldDismiss)")
            }
        }
    }
    private(set) var isActive: Bool = false

    /// The pure word model the view renders — word char-ranges, annotation flags, and
    /// the pacing fold. Reassigned per segment; the view reads it instead of
    /// re-deriving word boundaries.
    private(set) var timeline = WordTimeline(text: "")

    // MARK: - Internal state

    private var totalDuration: TimeInterval = 0
    private var playbackTimeProvider: (() -> TimeInterval)?
    private var timer: Timer?
    private var tickLogCounter: Int = 0

    // MARK: - Segment Window

    /// The single playback-time base this segment's pacing is measured against — the
    /// previous segment's cumulative scheduled duration (CONTEXT.md → Segment Window).
    /// Replaces the coupled time/duration bases that were always assigned the same value.
    ///
    /// Invariant: this is subtracted from BOTH the playback head time (`tick`) and the
    /// cumulative scheduled duration (`updateTotalDuration`). One base is correct for
    /// both only because they share an origin — playback start, where head time and
    /// scheduled duration coincide. If playback ever offsets the head from scheduled
    /// duration (pre-roll, gapless trim, a resume that shifts the head origin), this
    /// must split back into two bases.
    private var segmentBase: TimeInterval = 0

    // MARK: - Pacing carry-over

    /// The `WordTimeline` smoothing carry threaded through `advance` each tick.
    private var pacing = WordTimeline.Pacing(seed: 0)
    /// The text-length-based estimate computed at start, aligned to actual on
    /// `markSegmentComplete`. Fed into the pacing fold; never stored on the timeline.
    private var estimatedFinalDuration: TimeInterval = 0

    /// Adaptive chars/sec rate, learned from previous generations.
    /// Persisted across show() calls on the same controller instance.
    /// Default of 15 is a reasonable starting point for English TTS.
    private static var learnedCharsPerSec: Double = 15.0

    // MARK: - Public API

    func start(text: String, tokenCharOffsets: [Int], playbackTimeProvider: @escaping () -> TimeInterval) {
        Log.speech.info("[WordTracker] start() — text=\(text.prefix(40))…, tokenOffsets=\(tokenCharOffsets.count)")
        stop()

        timeline = WordTimeline(text: text, tokenCharOffsets: tokenCharOffsets)
        self.playbackTimeProvider = playbackTimeProvider
        recognizedCharCount = 0
        isGenerationComplete = false
        shouldDismiss = false
        isActive = true
        segmentBase = 0

        // Seed the pacing smoothing from the text-length estimate.
        estimatedFinalDuration = seededEstimate()
        pacing = WordTimeline.Pacing(seed: estimatedFinalDuration)
        Log.speech.info("[WordTracker] start() — \(tokenCharOffsets.isEmpty ? "proportional" : "token-aligned"), estDur=\(String(format: "%.1f", self.estimatedFinalDuration))s")

        tickLogCounter = 0
        startTimer()
    }

    func updateTotalDuration(_ cumulativeDuration: TimeInterval) {
        totalDuration = cumulativeDuration - segmentBase
    }

    func updateText(_ text: String, tokenCharOffsets: [Int], segmentBase: TimeInterval) {
        Log.speech.info("[WordTracker] updateText() — \(text.prefix(40))…, tokenOffsets=\(tokenCharOffsets.count), segmentBase=\(String(format: "%.2f", segmentBase))")

        timeline = WordTimeline(text: text, tokenCharOffsets: tokenCharOffsets)
        recognizedCharCount = 0
        isGenerationComplete = false
        totalDuration = 0
        self.segmentBase = segmentBase

        estimatedFinalDuration = seededEstimate()
        pacing = WordTimeline.Pacing(seed: estimatedFinalDuration)
    }

    /// The text-length duration estimate seeded into the pacing smoothing at each
    /// segment start: chars over the learned chars/sec. One home for `start` /
    /// `updateText` so the seed formula can't drift between them.
    private func seededEstimate() -> TimeInterval {
        Double(timeline.totalCharCount) / Self.learnedCharsPerSec
    }

    /// Called when a single segment's generation finishes (but more segments remain).
    /// Aligns the duration estimate so highlighting converges to 100% for this segment
    /// without setting isGenerationComplete (which would trigger auto-dismiss).
    func markSegmentComplete() {
        if totalDuration > 0 {
            estimatedFinalDuration = totalDuration
            Log.speech.info("[WordTracker] markSegmentComplete() — aligned estimate to actual \(String(format: "%.1f", self.totalDuration))s")
        }
    }

    func markGenerationComplete() {
        isGenerationComplete = true

        // Learn the actual chars/sec for future duration estimates
        if totalDuration > 0 && timeline.totalCharCount > 0 {
            let actualRate = Double(timeline.totalCharCount) / totalDuration
            Self.learnedCharsPerSec = actualRate * 0.7 + Self.learnedCharsPerSec * 0.3
            Log.speech.info("[WordTracker] markGenerationComplete() — actualRate=\(String(format: "%.1f", actualRate)) c/s, learnedRate=\(String(format: "%.1f", Self.learnedCharsPerSec)) c/s, totalDuration=\(String(format: "%.1f", self.totalDuration))s")
        }
    }

    func jumpTo(charOffset: Int) {
        recognizedCharCount = max(0, min(charOffset, timeline.totalCharCount))
    }

    func stop() {
        let wasActive = isActive
        stopTimer()
        isActive = false
        playbackTimeProvider = nil
        timeline = WordTimeline(text: "")
        totalDuration = 0
        recognizedCharCount = 0
        pacing = WordTimeline.Pacing(seed: 0)
        estimatedFinalDuration = 0
        segmentBase = 0
        if wasActive {
            Log.speech.info("[WordTracker] stop() — was active, now stopped")
        }
    }

    // MARK: - Timer

    private func startTimer() {
        timer = Timer.scheduledTimer(withTimeInterval: 1.0 / 60.0, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.tick()
            }
        }
    }

    private func stopTimer() {
        if timer != nil {
            Log.speech.info("[WordTracker] timer stopped")
        }
        timer?.invalidate()
        timer = nil
    }

    private func tick() {
        guard isActive,
              totalDuration > 0,
              timeline.totalCharCount > 0,
              let provider = playbackTimeProvider else { return }

        let elapsed = provider() - segmentBase

        // Fold the pure timeline; it threads the smoothing carry in and out.
        let (newCount, newPacing) = timeline.advance(
            elapsed: elapsed,
            totalDuration: totalDuration,
            estimatedFinalDuration: estimatedFinalDuration,
            isGenerationComplete: isGenerationComplete,
            pacing: pacing
        )
        pacing = newPacing

        // Monotonic: only advance forward to prevent highlight from jumping back.
        if newCount > recognizedCharCount {
            recognizedCharCount = newCount
        }

        // Log every ~1 second (60 ticks)
        tickLogCounter += 1
        if tickLogCounter % 60 == 1 {
            Log.speech.info("[WordTracker] tick #\(self.tickLogCounter): elapsed=\(String(format: "%.2f", elapsed)), charCount=\(self.recognizedCharCount)/\(self.timeline.totalCharCount)")
        }
    }
}
