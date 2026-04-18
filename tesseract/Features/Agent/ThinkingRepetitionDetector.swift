import Foundation

/// Online detector for degenerate repetition inside a model's `<think>...</think>`
/// block. Fed one streamed decoded-text chunk at a time; emits a decision to
/// intervene when the accumulated thinking content shows clear repetition.
///
/// Two complementary signals, because real failure modes alternate between them:
/// - **Line frequency** — a whitespace-normalized line repeated
///   ``Config/maxLineRepeats`` times catches the common "same sentence over and
///   over" loop (the repro we've seen with Qwen3.5-4B-PARO at `temperature=0`).
/// - **N-gram rolling hash** — a fixed-size substring (``Config/ngramSize`` chars)
///   repeated ``Config/maxNgramRepeats`` times inside a rolling window covers
///   paraphrase-dense loops that don't line up on newlines.
///
/// Detection is intentionally string-level: we only have the decoded text chunks
/// on the streaming path (`MLXLMCommon.generateTask` yields strings, not token
/// IDs). Pushing detection into the logit pipeline would require re-deriving
/// state from token IDs every step; the string path is orders of magnitude
/// cheaper and sufficient for the observed failure mode.
///
/// Detector is `nonisolated` and holds no concurrency state; construct one per
/// request.
nonisolated final class ThinkingRepetitionDetector {

    struct Config: Sendable, Codable, Equatable {
        /// Master switch. When `false`, `ingest` always returns `.continue`.
        var enabled: Bool = true
        /// Lines shorter than this are never considered for the line-frequency
        /// signal. Prevents false positives on natural short repeats ("OK.").
        var minLineLength: Int = 20
        /// A normalized line is flagged when its count reaches this value.
        var maxLineRepeats: Int = 6
        /// Number of leading characters of a normalized line used as its
        /// starter signature. Catches paraphrased loops that share an opening
        /// but vary identifiers later in the line (e.g. "Let me also check
        /// the remaining pages and the utility file…" vs "Now let me also
        /// read the remaining page files and the lib/utils.ts…").
        var starterPrefixChars: Int = 32
        /// A starter signature is flagged when its count reaches this value.
        /// `nil` disables the starter signal. Defaults to matching
        /// ``maxLineRepeats`` so the paraphrase signal is no more aggressive
        /// than the exact-line signal.
        var maxStarterRepeats: Int? = 6
        /// N-gram size in characters for the rolling-hash signal (~10 tokens at
        /// Qwen3.5's ~3.6 chars/token).
        var ngramSize: Int = 40
        /// Rolling-window hit count that flags the n-gram signal.
        var maxNgramRepeats: Int = 8
        /// Size of the rolling window, in characters, used by the n-gram signal.
        var windowChars: Int = 8_192
        /// Optional heuristic budget: flag when total thinking chars exceed this.
        /// NOT a token budget — chars only. `nil` disables.
        ///
        /// Default `16_384` (~4.5K tokens at Qwen3.5's ~3.6 chars/token) is a
        /// conservative backstop for loops whose structural variation delays
        /// the line/n-gram triggers (e.g. counting loops that vary each round).
        /// Raise or set to `nil` for tasks that legitimately need longer
        /// reasoning; lower for tighter worst-case latency.
        var maxThinkingChars: Int? = 16_384
        /// Minimum accumulated thinking-content chars before ANY repetition
        /// signal (line-repeat, n-gram, budget) is allowed to fire. Detector
        /// state still updates during the grace period — only the trigger
        /// decision is gated — so a pattern that was already repeating can
        /// fire on the first post-grace ingest.
        ///
        /// Default `8_192` (~2K tokens at Qwen3.5's ~3.6 chars/token).
        /// Structured reasoning on multi-field extraction tasks routinely
        /// produces 1–2K tokens of legitimate "Field: X / Wait, check Y"
        /// alternation; 2K tokens gives the heuristics enough evidence to
        /// distinguish a true loop from step-by-step analysis. Set to `0`
        /// to disable the grace period.
        var minCharsBeforeIntervention: Int = 8_192
        /// Text appended to the safe prefix when the intervention fires. Sent
        /// downstream as a `.thinking(String)` event just before `.thinkEnd`.
        var injectionMessage: String =
            "\n\nI have enough information. Responding now.\n"

        /// Text appended to ``injectionMessage`` when rebuilding the continuation
        /// prompt fed back to the model. Closes the `<think>` block so the
        /// decoder transitions into the answer phase. Tied to the Qwen3/3.5
        /// chat-template's think-tag protocol — override only for models with a
        /// different closing sequence.
        var thinkCloseSuffix: String = "\n</think>\n\n"

        /// Full hand-off text for the continuation prompt: injection message +
        /// the think-close suffix. Callers rebuilding the continuation prompt
        /// use this instead of string-concatenating the two fields manually.
        var continuationHandOff: String { injectionMessage + thinkCloseSuffix }

        init(
            enabled: Bool = true,
            minLineLength: Int = 20,
            maxLineRepeats: Int = 6,
            starterPrefixChars: Int = 32,
            maxStarterRepeats: Int? = 6,
            ngramSize: Int = 40,
            maxNgramRepeats: Int = 8,
            windowChars: Int = 8_192,
            maxThinkingChars: Int? = 16_384,
            minCharsBeforeIntervention: Int = 8_192,
            injectionMessage: String =
                "\n\nI have enough information. Responding now.\n",
            thinkCloseSuffix: String = "\n</think>\n\n"
        ) {
            self.enabled = enabled
            self.minLineLength = minLineLength
            self.maxLineRepeats = maxLineRepeats
            self.starterPrefixChars = starterPrefixChars
            self.maxStarterRepeats = maxStarterRepeats
            self.ngramSize = ngramSize
            self.maxNgramRepeats = maxNgramRepeats
            self.windowChars = windowChars
            self.maxThinkingChars = maxThinkingChars
            self.minCharsBeforeIntervention = minCharsBeforeIntervention
            self.injectionMessage = injectionMessage
            self.thinkCloseSuffix = thinkCloseSuffix
        }
    }

    enum Reason: String, Sendable, Codable {
        case duplicateLine
        case duplicateStarter
        case duplicateNgram
        case budgetExceeded
    }

    enum Decision: Sendable, Equatable {
        case `continue`
        case intervene(reason: Reason, safePrefix: String)
    }

    private let config: Config
    /// Everything the model has emitted inside `<think>` so far, *excluding* any
    /// characters of the current partially-formed line. Terminated at the last
    /// seen `\n`.
    private var completedLines: String = ""
    /// Characters since the last `\n`. Joined with `completedLines` for n-gram
    /// and budget checks.
    private var pendingLine: String = ""
    /// Counts of each normalized line seen so far.
    private var lineFreq: [String: Int] = [:]
    /// Snapshot of `completedLines` taken the first time a given normalized
    /// line is seen. On trigger we return the snapshot captured when the
    /// flagged line was first observed — i.e. the content *before* the
    /// repetition began.
    private var firstOccurrenceSafePrefix: [String: String] = [:]
    /// Counts of each starter signature (first `starterPrefixChars` of a
    /// normalized line) seen so far. Separate from `lineFreq` because two
    /// lines with the same starter but different tails are distinct line
    /// entries but share a starter entry — paraphrase detection lives here.
    private var starterFreq: [String: Int] = [:]
    /// First-occurrence snapshot keyed by starter signature, mirroring
    /// `firstOccurrenceSafePrefix` for the line-frequency signal.
    private var firstOccurrenceSafePrefixForStarter: [String: String] = [:]

    init(config: Config = .init()) {
        self.config = config
    }

    /// Feed the next decoded chunk of thinking content. Returns `.intervene`
    /// the first time any trigger fires. After intervening the caller should
    /// stop feeding this instance (or call ``reset()``).
    func ingest(chunk: String) -> Decision {
        guard config.enabled, !chunk.isEmpty else { return .continue }

        // Grace gate: state still updates during grace, but no trigger fires
        // until the accumulated buffer (including this chunk) clears the
        // threshold. Computed once at the top using post-chunk size so a
        // pattern already repeating during grace can fire on the same ingest
        // that crosses the boundary.
        let graceGate =
            completedLines.count + pendingLine.count + chunk.count
            >= config.minCharsBeforeIntervention

        for char in chunk {
            if char == "\n" {
                if let decision = closePendingLine(), graceGate { return decision }
            } else {
                pendingLine.append(char)
            }
        }

        guard graceGate else { return .continue }

        if let decision = processNgrams() { return decision }

        if let limit = config.maxThinkingChars {
            let total = completedLines.count + pendingLine.count
            if total >= limit {
                let full = completedLines + pendingLine
                return .intervene(
                    reason: .budgetExceeded,
                    safePrefix: Self.backtrackToLineBoundary(
                        String(full.prefix(limit)))
                )
            }
        }

        return .continue
    }

    func reset() {
        completedLines.removeAll(keepingCapacity: true)
        pendingLine.removeAll(keepingCapacity: true)
        lineFreq.removeAll(keepingCapacity: true)
        firstOccurrenceSafePrefix.removeAll(keepingCapacity: true)
        starterFreq.removeAll(keepingCapacity: true)
        firstOccurrenceSafePrefixForStarter.removeAll(keepingCapacity: true)
    }

    // MARK: - Line frequency

    /// A `\n` just arrived; classify and flush the pending line.
    private func closePendingLine() -> Decision? {
        defer {
            completedLines.append(pendingLine)
            completedLines.append("\n")
            pendingLine.removeAll(keepingCapacity: true)
        }

        let normalized = Self.normalize(pendingLine)
        guard normalized.count >= config.minLineLength else { return nil }

        // completedLines at this point is the buffer BEFORE the current
        // line — `defer` appends pendingLine on return. Safe-prefix
        // snapshots captured here are therefore the correct "content
        // before the repetition began."
        if let decision = bumpAndCheck(
            key: normalized,
            freq: &lineFreq,
            firstOccurrence: &firstOccurrenceSafePrefix,
            threshold: config.maxLineRepeats,
            reason: .duplicateLine
        ) {
            return decision
        }

        if let maxStarterRepeats = config.maxStarterRepeats,
           config.starterPrefixChars > 0,
           normalized.count >= config.starterPrefixChars {
            let starter = String(normalized.prefix(config.starterPrefixChars))
            if let decision = bumpAndCheck(
                key: starter,
                freq: &starterFreq,
                firstOccurrence: &firstOccurrenceSafePrefixForStarter,
                threshold: maxStarterRepeats,
                reason: .duplicateStarter
            ) {
                return decision
            }
        }

        return nil
    }

    /// Increment `freq[key]`, stash the first-occurrence safe-prefix snapshot,
    /// and return an intervention once the count reaches `threshold`.
    private func bumpAndCheck(
        key: String,
        freq: inout [String: Int],
        firstOccurrence: inout [String: String],
        threshold: Int,
        reason: Reason
    ) -> Decision? {
        let prior = freq[key] ?? 0
        if prior == 0 { firstOccurrence[key] = completedLines }
        let next = prior + 1
        freq[key] = next
        guard next >= threshold else { return nil }
        return .intervene(
            reason: reason,
            safePrefix: firstOccurrence[key] ?? completedLines
        )
    }

    /// Lowercase + collapse runs of whitespace to a single space + trim edges.
    /// Robust against leading-indent drift and tab/space noise without being so
    /// aggressive that distinct sentences collide.
    nonisolated static func normalize(_ line: String) -> String {
        var result = ""
        result.reserveCapacity(line.count)
        var inWhitespace = false
        for scalar in line.unicodeScalars {
            if CharacterSet.whitespaces.contains(scalar) {
                if !inWhitespace && !result.isEmpty { result.append(" ") }
                inWhitespace = true
            } else {
                result.append(Character(scalar))
                inWhitespace = false
            }
        }
        while result.last == " " { result.removeLast() }
        return result.lowercased()
    }

    /// Snap a safe-prefix candidate back to the last `\n` so the surfaced
    /// reasoning never ends mid-sentence or mid-word. Returns `""` when the
    /// prefix has no newline — "show nothing" is preferable to "show a torn
    /// line" for a user-visible reasoning surface.
    nonisolated static func backtrackToLineBoundary(_ prefix: String) -> String {
        if prefix.isEmpty || prefix.hasSuffix("\n") { return prefix }
        if let idx = prefix.lastIndex(of: "\n") {
            return String(prefix[...idx])
        }
        return ""
    }

    // MARK: - N-gram rolling window

    /// Rebuild n-gram counts over the last `windowChars` characters of the
    /// combined (completed + pending) buffer. We don't maintain a true O(1)
    /// incremental hash — String doesn't expose a fixed-stride byte view
    /// cheaply — so we rescan the window per chunk. At `windowChars=8192` and
    /// chunk-interval ≥ 1 ms this is negligible next to model inference time.
    private func processNgrams() -> Decision? {
        let n = config.ngramSize
        let totalLen = completedLines.count + pendingLine.count
        guard n > 1, totalLen >= n else { return nil }

        // Combined = everything in the thinking buffer so far. When it exceeds
        // `windowChars`, scan only the trailing window to bound work per chunk.
        let combined = completedLines + pendingLine
        let windowStartOffset = max(0, totalLen - config.windowChars)
        let windowStart = combined.index(combined.startIndex, offsetBy: windowStartOffset)
        let window = combined[windowStart..<combined.endIndex]

        var ngramCounts: [Substring: Int] = [:]
        ngramCounts.reserveCapacity(config.windowChars)
        var i = window.startIndex
        let lastStart = window.index(window.endIndex, offsetBy: -n)
        while i <= lastStart {
            let end = window.index(i, offsetBy: n)
            let slice = window[i..<end]
            let count = (ngramCounts[slice] ?? 0) + 1
            ngramCounts[slice] = count
            if count >= config.maxNgramRepeats {
                // Cut at the first occurrence of the repeating n-gram; fall
                // back to the window start if `range(of:)` can't locate it
                // (shouldn't happen in practice — slice came FROM `combined`).
                let needle = String(slice)
                let cut = combined.range(of: needle)?.lowerBound ?? windowStart
                return .intervene(
                    reason: .duplicateNgram,
                    safePrefix: Self.backtrackToLineBoundary(
                        String(combined[..<cut]))
                )
            }
            i = window.index(after: i)
        }
        return nil
    }
}
