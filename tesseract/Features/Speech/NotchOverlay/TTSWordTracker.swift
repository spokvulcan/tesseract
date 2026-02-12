//
//  TTSWordTracker.swift
//  tesseract
//
//  Token-aligned word tracker for TTS playback overlay.
//  Uses BPE token-to-character offsets for ~80ms resolution tracking.
//  Falls back to proportional estimation when token offsets are unavailable.
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

    // MARK: - Internal state

    private(set) var words: [String] = []
    private var wordEndOffsets: [Int] = []
    private(set) var totalCharCount: Int = 0
    private var totalDuration: TimeInterval = 0
    private var playbackTimeProvider: (() -> TimeInterval)?
    private var timer: Timer?
    private var tickLogCounter: Int = 0

    // MARK: - Segment time windowing

    /// Cumulative playback time at the start of the current segment
    private var segmentTimeBase: TimeInterval = 0
    /// Cumulative scheduled duration at the start of the current segment
    private var segmentDurationBase: TimeInterval = 0

    // MARK: - Token timeline (primary path)

    /// Per-token character offsets from BPE tokenization.
    /// Entry i = character offset where token i starts in the text.
    /// Stretched across the full audio duration for pacing.
    private var tokenCharOffsets: [Int] = []

    // MARK: - Proportional fallback state

    /// Smoothed effective duration — blends from estimate to actual when generation completes
    private var smoothedEffDuration: TimeInterval = 0
    /// The text-length-based estimate computed at start
    private var estimatedFinalDuration: TimeInterval = 0

    /// Adaptive chars/sec rate, learned from previous generations.
    /// Persisted across show() calls on the same controller instance.
    /// Default of 15 is a reasonable starting point for English TTS.
    private static var learnedCharsPerSec: Double = 15.0

    /// Whether we're using token-aligned tracking (true) or proportional fallback (false)
    private var useTokenTimeline: Bool = false

    // MARK: - Public API

    func start(text: String, tokenCharOffsets: [Int], playbackTimeProvider: @escaping () -> TimeInterval) {
        Log.speech.info("[WordTracker] start() — text=\(text.prefix(40))…, tokenOffsets=\(tokenCharOffsets.count)")
        stop()

        let normalized = text
            .replacingOccurrences(of: "\n", with: " ")
            .split(omittingEmptySubsequences: true, whereSeparator: { $0.isWhitespace })
            .map { String($0) }

        words = normalized
        totalCharCount = normalized.joined(separator: " ").count
        self.playbackTimeProvider = playbackTimeProvider
        recognizedCharCount = 0
        isGenerationComplete = false
        shouldDismiss = false
        isActive = true
        segmentTimeBase = 0
        segmentDurationBase = 0

        // Build cumulative character offsets for word boundaries
        var offset = 0
        wordEndOffsets = []
        for word in words {
            offset += word.count
            wordEndOffsets.append(offset)
            offset += 1 // space
        }

        // Both paths need a duration estimate for pacing
        estimatedFinalDuration = Double(totalCharCount) / Self.learnedCharsPerSec
        smoothedEffDuration = estimatedFinalDuration

        // Use token timeline if offsets are available, otherwise fall back
        if !tokenCharOffsets.isEmpty {
            self.tokenCharOffsets = tokenCharOffsets
            useTokenTimeline = true
            Log.speech.info("[WordTracker] start() — using token timeline (\(tokenCharOffsets.count) tokens, estDur=\(String(format: "%.1f", self.estimatedFinalDuration))s)")
        } else {
            self.tokenCharOffsets = []
            useTokenTimeline = false
            Log.speech.info("[WordTracker] start() — proportional fallback, estimatedDuration=\(String(format: "%.1f", self.estimatedFinalDuration))s")
        }

        tickLogCounter = 0
        startTimer()
    }

    func updateTotalDuration(_ cumulativeDuration: TimeInterval) {
        totalDuration = cumulativeDuration - segmentDurationBase
    }

    func updateText(_ text: String, tokenCharOffsets: [Int], segmentTimeBase: TimeInterval = 0, segmentDurationBase: TimeInterval = 0) {
        Log.speech.info("[WordTracker] updateText() — \(text.prefix(40))…, tokenOffsets=\(tokenCharOffsets.count), timeBase=\(String(format: "%.2f", segmentTimeBase)), durBase=\(String(format: "%.2f", segmentDurationBase))")
        let normalized = text
            .replacingOccurrences(of: "\n", with: " ")
            .split(omittingEmptySubsequences: true, whereSeparator: { $0.isWhitespace })
            .map { String($0) }

        words = normalized
        totalCharCount = normalized.joined(separator: " ").count
        recognizedCharCount = 0
        isGenerationComplete = false
        totalDuration = 0
        self.segmentTimeBase = segmentTimeBase
        self.segmentDurationBase = segmentDurationBase

        var offset = 0
        wordEndOffsets = []
        for word in words {
            offset += word.count
            wordEndOffsets.append(offset)
            offset += 1
        }

        estimatedFinalDuration = Double(totalCharCount) / Self.learnedCharsPerSec
        smoothedEffDuration = estimatedFinalDuration

        if !tokenCharOffsets.isEmpty {
            self.tokenCharOffsets = tokenCharOffsets
            useTokenTimeline = true
        } else {
            self.tokenCharOffsets = []
            useTokenTimeline = false
        }
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
        if totalDuration > 0 && totalCharCount > 0 {
            let actualRate = Double(totalCharCount) / totalDuration
            Self.learnedCharsPerSec = actualRate * 0.7 + Self.learnedCharsPerSec * 0.3
            Log.speech.info("[WordTracker] markGenerationComplete() — actualRate=\(String(format: "%.1f", actualRate)) c/s, learnedRate=\(String(format: "%.1f", Self.learnedCharsPerSec)) c/s, totalDuration=\(String(format: "%.1f", self.totalDuration))s")
        }
    }

    func jumpTo(charOffset: Int) {
        recognizedCharCount = max(0, min(charOffset, totalCharCount))
    }

    func stop() {
        let wasActive = isActive
        stopTimer()
        isActive = false
        playbackTimeProvider = nil
        words = []
        wordEndOffsets = []
        totalCharCount = 0
        totalDuration = 0
        recognizedCharCount = 0
        smoothedEffDuration = 0
        estimatedFinalDuration = 0
        tokenCharOffsets = []
        useTokenTimeline = false
        segmentTimeBase = 0
        segmentDurationBase = 0
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
              totalCharCount > 0,
              let provider = playbackTimeProvider else { return }

        let elapsed = provider() - segmentTimeBase

        let newCount: Int
        if useTokenTimeline {
            newCount = tickTokenTimeline(elapsed: elapsed)
        } else {
            newCount = tickProportional(elapsed: elapsed)
        }

        // Monotonic: only advance forward to prevent highlight from jumping back
        if newCount > recognizedCharCount {
            recognizedCharCount = newCount
        }

        // Log every ~1 second (60 ticks)
        tickLogCounter += 1
        if tickLogCounter % 60 == 1 {
            Log.speech.info("[WordTracker] tick #\(self.tickLogCounter): elapsed=\(String(format: "%.2f", elapsed)), charCount=\(self.recognizedCharCount)/\(self.totalCharCount), mode=\(self.useTokenTimeline ? "token" : "proportional")")
        }
    }

    // MARK: - Token timeline tick

    /// Map elapsed time → token index → character offset.
    /// Token offsets are stretched across the effective audio duration so that
    /// highlighting paces with actual speech, not raw token consumption rate.
    private func tickTokenTimeline(elapsed: TimeInterval) -> Int {
        guard !tokenCharOffsets.isEmpty else { return 0 }

        // Compute effective duration (same smoothing as proportional path)
        let targetEffDuration: TimeInterval
        if isGenerationComplete {
            targetEffDuration = totalDuration
        } else {
            targetEffDuration = max(totalDuration, estimatedFinalDuration)
        }
        smoothedEffDuration += (targetEffDuration - smoothedEffDuration) * 0.08

        let progress = min(max(elapsed / smoothedEffDuration, 0), 1.0)

        // Map time proportionally to characters (weighted by text length per token)
        // rather than uniform per-token, better matching speech pacing
        let targetChars = Int(progress * Double(totalCharCount))

        // Binary search tokenCharOffsets for the token covering targetChars
        var lo = 0, hi = tokenCharOffsets.count
        while lo < hi {
            let mid = (lo + hi) / 2
            if tokenCharOffsets[mid] <= targetChars {
                lo = mid + 1
            } else {
                hi = mid
            }
        }
        // lo is the first index where tokenCharOffsets[lo] > targetChars
        let charPos = lo > 0 ? tokenCharOffsets[lo - 1] : 0
        return min(max(charPos, targetChars), totalCharCount)
    }

    // MARK: - Proportional fallback tick

    private func tickProportional(elapsed: TimeInterval) -> Int {
        let targetEffDuration: TimeInterval
        if isGenerationComplete {
            targetEffDuration = totalDuration
        } else {
            targetEffDuration = max(totalDuration, estimatedFinalDuration)
        }

        smoothedEffDuration += (targetEffDuration - smoothedEffDuration) * 0.08

        let progress = min(max(elapsed / smoothedEffDuration, 0), 1.0)
        let targetCharPosition = Int(progress * Double(totalCharCount))
        return min(targetCharPosition, totalCharCount)
    }
}
