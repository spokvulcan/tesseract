//
//  WordTimeline.swift
//  tesseract
//
//  The pure projection of one segment's spoken text plus the current playback
//  position into the highlighted character count and the active word — the single
//  home for the token→char→word model and the elapsed→position pacing the TTS notch
//  overlay renders (CONTEXT.md → "Speech word timeline").
//
//  An immutable `nonisolated` value built once per segment from the text plus token
//  offsets. It owns the per-word character ranges (the `offset += word.count + 1`
//  model, computed once at construction) and three entry points: `advance` (the
//  single pacing fold), `activeWordIndex` (the char→word query), and `litFraction`
//  (the per-word value the view renders). It holds no timer, clock, `@Observable`
//  state, or UI: elapsed time, durations, and the smoothing carry-over pass in and out
//  via `Pacing`, never stored. The driver — `TTSWordTracker` — decides *when* to
//  re-fold and owns the monotonic clamp on the published count; this fold is not
//  monotonic.
//

import Foundation

nonisolated struct WordTimeline: Equatable, Sendable {

    /// One whitespace-separated word with its start offset into the highlighted
    /// character space, whether it is an annotation (`[...]` or punctuation-only), and
    /// the per-word lengths the render path needs — all computed once at construction
    /// so the 60fps highlight never re-walks the string.
    struct Word: Equatable, Sendable {
        let text: String
        /// Character offset where this word starts (cumulative `word.count + 1`).
        let charOffset: Int
        let isAnnotation: Bool
        /// `text.count`, precomputed — the highlight cap for this word.
        let charCount: Int
        /// Letters/numbers in the word, precomputed — the lit-fraction denominator.
        let letterCount: Int
    }

    let words: [Word]
    /// Length of the words joined by single spaces — the highlight char space.
    let totalCharCount: Int

    /// Per-token character offsets from BPE tokenization, stretched across the
    /// effective duration for pacing. Empty ⇒ uniform-proportional pacing.
    private let tokenCharOffsets: [Int]

    init(text: String, tokenCharOffsets: [Int] = []) {
        // One shared definition of "a word" (see StringWordSplitting) so the rendered
        // model and TextSegmenter's token estimate can't drift on what splits a word.
        let normalized = text.splitIntoWords().map(String.init)

        var built: [Word] = []
        built.reserveCapacity(normalized.count)
        var offset = 0
        for word in normalized {
            // A word is an annotation if it is bracketed (`[...]`) or carries no letters
            // or numbers (punctuation-only); letterCount is reused as the lit denominator.
            let letterCount = word.filter { $0.isLetter || $0.isNumber }.count
            let isAnnotation = (word.hasPrefix("[") && word.hasSuffix("]")) || letterCount == 0
            built.append(Word(
                text: word,
                charOffset: offset,
                isAnnotation: isAnnotation,
                charCount: word.count,
                letterCount: letterCount
            ))
            offset += word.count + 1   // word characters plus one separating space
        }

        self.words = built
        // The single-space-joined length the highlight runs across. The loop above
        // already accumulated Σ(word.count + 1), so this is `offset - 1` (0 when empty)
        // — no need to build and measure a joined copy.
        self.totalCharCount = max(0, offset - 1)
        self.tokenCharOffsets = tokenCharOffsets
    }

    // MARK: - char → word query

    /// The active word for a highlighted character count: the first word whose end
    /// offset reaches `highlightedCharCount`, clamping to the last word once the
    /// highlight runs past the text — the rule the view's scroll-centering uses. How
    /// far into that word the highlight has progressed is `litFraction`, computed only
    /// where the view needs it, never eagerly here.
    func activeWordIndex(highlightedCharCount: Int) -> Int {
        guard !words.isEmpty else { return 0 }
        for (i, word) in words.enumerated() where highlightedCharCount <= word.charOffset + word.charCount {
            return i
        }
        return words.count - 1
    }

    /// How lit one word is: lit characters over its letter/number count (the word's
    /// own length is the cap; `>= 1.0` means fully lit). Reads the per-word lengths
    /// cached on `Word` at construction rather than re-walking the string each call —
    /// this runs ~2×N per frame on the highlight path. Out-of-range indices read as `0`.
    func litFraction(wordIndex: Int, charCount: Int) -> Double {
        guard words.indices.contains(wordIndex) else { return 0 }
        let word = words[wordIndex]
        let charsIntoWord = charCount - word.charOffset
        let litCount = max(0, min(word.charCount, charsIntoWord))
        let letterCount = max(1, word.letterCount)
        return Double(litCount) / Double(letterCount)
    }

    // MARK: - the single pacing fold

    /// The smoothing carry-over threaded through `advance` — the timeline stores no
    /// time-varying state, so the driver holds this between ticks and passes it back.
    struct Pacing: Equatable, Sendable {
        /// Smoothed effective duration — blends from the text-length estimate toward
        /// the actual scheduled duration as generation progresses.
        var smoothedEffDuration: TimeInterval
        /// Seed at segment start (the text-length estimate).
        init(seed: TimeInterval) { self.smoothedEffDuration = seed }
    }

    private static let smoothingFactor = 0.08

    /// Map elapsed playback time to a highlighted character count for this segment.
    /// The effective duration blends estimate→actual (and snaps to the actual once
    /// generation completes); the proportional position is then optionally snapped to
    /// a token boundary. **Not monotonic** — the driver clamps the published count.
    /// Returns the candidate count plus the updated `Pacing` to carry to the next tick.
    func advance(
        elapsed: TimeInterval,
        totalDuration: TimeInterval,
        estimatedFinalDuration: TimeInterval,
        isGenerationComplete: Bool,
        pacing: Pacing
    ) -> (charCount: Int, pacing: Pacing) {
        guard totalCharCount > 0 else { return (0, pacing) }

        let target = isGenerationComplete ? totalDuration : max(totalDuration, estimatedFinalDuration)
        var next = pacing
        next.smoothedEffDuration += (target - next.smoothedEffDuration) * Self.smoothingFactor

        guard next.smoothedEffDuration > 0 else { return (0, next) }

        let progress = min(max(elapsed / next.smoothedEffDuration, 0), 1.0)
        let targetChars = Int(progress * Double(totalCharCount))
        return (snap(toCharCount: targetChars), next)
    }

    /// Snap a proportional target to a token boundary. Preserved exactly from the
    /// previous token path: the boundary at or below the target, then `max` with the
    /// target — which makes the snap currently coincide with the proportional target
    /// (a latent no-op). Kept so the token model has one home and the latent snap one
    /// place to live; *activating* it (e.g. snapping up to the next boundary) would
    /// change pacing and is out of scope for the behavior-preserving carve (#55).
    private func snap(toCharCount targetChars: Int) -> Int {
        guard !tokenCharOffsets.isEmpty else { return min(targetChars, totalCharCount) }
        var lo = 0, hi = tokenCharOffsets.count
        while lo < hi {
            let mid = (lo + hi) / 2
            if tokenCharOffsets[mid] <= targetChars { lo = mid + 1 } else { hi = mid }
        }
        let charPos = lo > 0 ? tokenCharOffsets[lo - 1] : 0
        return min(max(charPos, targetChars), totalCharCount)
    }
}
