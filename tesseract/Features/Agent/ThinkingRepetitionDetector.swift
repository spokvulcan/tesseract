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
            ngramSize: Int = 40,
            maxNgramRepeats: Int = 8,
            windowChars: Int = 8_192,
            maxThinkingChars: Int? = 16_384,
            injectionMessage: String =
                "\n\nI have enough information. Responding now.\n",
            thinkCloseSuffix: String = "\n</think>\n\n"
        ) {
            self.enabled = enabled
            self.minLineLength = minLineLength
            self.maxLineRepeats = maxLineRepeats
            self.ngramSize = ngramSize
            self.maxNgramRepeats = maxNgramRepeats
            self.windowChars = windowChars
            self.maxThinkingChars = maxThinkingChars
            self.injectionMessage = injectionMessage
            self.thinkCloseSuffix = thinkCloseSuffix
        }
    }

    enum Reason: String, Sendable, Codable {
        case duplicateLine
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

    init(config: Config = .init()) {
        self.config = config
    }

    /// Feed the next decoded chunk of thinking content. Returns `.intervene`
    /// the first time any trigger fires. After intervening the caller should
    /// stop feeding this instance (or call ``reset()``).
    func ingest(chunk: String) -> Decision {
        guard config.enabled else { return .continue }
        guard !chunk.isEmpty else { return .continue }

        // 1. Line-frequency signal. Consume characters one at a time; every
        //    time we hit \n, classify the completed line.
        for char in chunk {
            if char == "\n" {
                if let decision = closePendingLine() { return decision }
            } else {
                pendingLine.append(char)
            }
        }

        // 2. N-gram signal over the full current content.
        if let decision = processNgrams() { return decision }

        // 3. Optional hard char budget.
        if let limit = config.maxThinkingChars {
            let total = completedLines.count + pendingLine.count
            if total >= limit {
                let full = completedLines + pendingLine
                return .intervene(
                    reason: .budgetExceeded,
                    safePrefix: String(full.prefix(limit))
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

        let prior = lineFreq[normalized] ?? 0
        if prior == 0 {
            // Capture the buffer BEFORE the line itself. completedLines is exactly
            // that — we haven't appended the pending line yet (defer runs on
            // return).
            firstOccurrenceSafePrefix[normalized] = completedLines
        }
        let next = prior + 1
        lineFreq[normalized] = next

        if next >= config.maxLineRepeats {
            let safe = firstOccurrenceSafePrefix[normalized] ?? completedLines
            return .intervene(reason: .duplicateLine, safePrefix: safe)
        }
        return nil
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
                // safePrefix = everything up to the START of the first occurrence
                // of this n-gram in the full combined buffer.
                let needle = String(slice)
                if let firstRange = combined.range(of: needle) {
                    return .intervene(
                        reason: .duplicateNgram,
                        safePrefix: String(combined[..<firstRange.lowerBound])
                    )
                } else {
                    return .intervene(
                        reason: .duplicateNgram,
                        safePrefix: String(combined[..<windowStart])
                    )
                }
            }
            i = window.index(after: i)
        }
        return nil
    }
}
