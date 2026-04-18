import Testing

@testable import Tesseract_Agent

struct ThinkingRepetitionDetectorTests {

    // MARK: - Helpers

    private func feed(
        _ detector: ThinkingRepetitionDetector,
        _ chunks: [String]
    ) -> ThinkingRepetitionDetector.Decision {
        var last: ThinkingRepetitionDetector.Decision = .continue
        for chunk in chunks {
            last = detector.ingest(chunk: chunk)
            if case .intervene = last { return last }
        }
        return last
    }

    /// Zero-pad an Int to 7 digits without touching Foundation.
    private func padded(_ i: Int) -> String {
        let s = String(i)
        return String(repeating: "0", count: max(0, 7 - s.count)) + s
    }

    // MARK: - Line frequency signal

    @Test func detectsExactLineRepeat() {
        let config = ThinkingRepetitionDetector.Config(
            minLineLength: 20, maxLineRepeats: 6
        )
        let detector = ThinkingRepetitionDetector(config: config)

        // Repeat the same long-enough line 7 times; should trigger on the 6th.
        let line = "Wait, I need to check if the prompt implies `[EN]` is a word.\n"
        var decision: ThinkingRepetitionDetector.Decision = .continue
        for i in 1...7 {
            decision = detector.ingest(chunk: line)
            if case .intervene = decision {
                #expect(i == 6)
                break
            }
        }
        guard case .intervene(let reason, let safePrefix) = decision else {
            Issue.record("expected intervention, got .continue")
            return
        }
        #expect(reason == .duplicateLine)
        // safePrefix is the buffer captured BEFORE the first occurrence — empty
        // in this case because nothing preceded the repeated line.
        #expect(safePrefix == "")
    }

    @Test func safePrefixIsBufferBeforeFirstDuplicateLine() {
        let config = ThinkingRepetitionDetector.Config(
            minLineLength: 10, maxLineRepeats: 3
        )
        let detector = ThinkingRepetitionDetector(config: config)

        let prelude = "Reasoning about the user's constraints first.\n"
        let repeated = "Now I loop on the same thought forever.\n"

        // One legit line, then the loop line 3 times.
        _ = detector.ingest(chunk: prelude)
        _ = detector.ingest(chunk: repeated)  // count=1
        _ = detector.ingest(chunk: repeated)  // count=2
        let decision = detector.ingest(chunk: repeated)  // count=3 → trigger

        guard case .intervene(let reason, let safe) = decision else {
            Issue.record("expected intervention")
            return
        }
        #expect(reason == .duplicateLine)
        #expect(safe == prelude)
    }

    @Test func normalizesWhitespaceAndCaseForLineMatching() {
        let config = ThinkingRepetitionDetector.Config(
            minLineLength: 15, maxLineRepeats: 3
        )
        let detector = ThinkingRepetitionDetector(config: config)

        // Same line with different whitespace/case should count as one.
        _ = detector.ingest(chunk: "This is the LOOP sentence today.\n")
        _ = detector.ingest(chunk: "this is the loop sentence today.\n")
        let d = detector.ingest(chunk: "  This  is   the  LOOP sentence  today.\n")

        guard case .intervene(let reason, _) = d else {
            Issue.record("expected intervention on normalized duplicate")
            return
        }
        #expect(reason == .duplicateLine)
    }

    @Test func ignoresShortRepeatsBelowMinLineLength() {
        let config = ThinkingRepetitionDetector.Config(
            minLineLength: 20, maxLineRepeats: 3
        )
        let detector = ThinkingRepetitionDetector(config: config)

        // Short "OK." repeated 30 times should NOT trigger the line signal.
        for _ in 0..<30 {
            let d = detector.ingest(chunk: "OK.\n")
            if case .intervene(.duplicateLine, _) = d {
                Issue.record("short line should not trigger line signal")
                return
            }
        }
        // sanity — no trigger of any kind (ngram also shouldn't, content too repetitive
        // but short; line freq suppresses and ngram on "OK.\n" hits only 4 chars).
    }

    @Test func accumulatesAcrossChunkBoundariesMidLine() {
        let config = ThinkingRepetitionDetector.Config(
            minLineLength: 10, maxLineRepeats: 3
        )
        let detector = ThinkingRepetitionDetector(config: config)

        // Feed the same line in 3 pieces per repetition.
        let full = "The same thought I keep circling back to.\n"
        let a = String(full.prefix(15))
        let b = String(full[full.index(full.startIndex, offsetBy: 15)..<full.index(full.startIndex, offsetBy: 30)])
        let c = String(full[full.index(full.startIndex, offsetBy: 30)...])
        #expect(a + b + c == full)

        var decision: ThinkingRepetitionDetector.Decision = .continue
        for _ in 0..<3 {
            _ = detector.ingest(chunk: a)
            _ = detector.ingest(chunk: b)
            decision = detector.ingest(chunk: c)
        }
        guard case .intervene(.duplicateLine, _) = decision else {
            Issue.record("expected duplicateLine trigger across chunk boundaries")
            return
        }
    }

    // MARK: - N-gram signal

    @Test func detectsNgramRepeatWithoutNewlines() {
        let config = ThinkingRepetitionDetector.Config(
            // Line signal effectively off (no newlines in this test).
            minLineLength: 10,
            maxLineRepeats: 100,
            ngramSize: 20,
            maxNgramRepeats: 5,
            windowChars: 2_000
        )
        let detector = ThinkingRepetitionDetector(config: config)

        // Exactly 20-char pattern; repeating it 6× places the same 20-char
        // ngram at offsets 0, 20, 40, 60, 80, 100 → 6 matches, above the 5
        // threshold. No newlines → line-signal is dormant.
        let pattern = "abcdefghijklmnopqrst"
        #expect(pattern.count == 20)

        let big = String(repeating: pattern, count: 6)
        let decision = detector.ingest(chunk: big)

        guard case .intervene(.duplicateNgram, let safe) = decision else {
            Issue.record("expected duplicateNgram trigger, got \(decision)")
            return
        }
        // safePrefix is content before the pattern's first occurrence — which
        // is the very start, so empty.
        #expect(safe == "")
    }

    // MARK: - Budget signal

    @Test func charBudgetTriggersWhenConfigured() {
        let config = ThinkingRepetitionDetector.Config(
            enabled: true,
            minLineLength: 9999,   // line signal off
            maxLineRepeats: 999,
            ngramSize: 999,        // ngram signal off (n > content)
            maxNgramRepeats: 999,
            windowChars: 16,
            maxThinkingChars: 200
        )
        let detector = ThinkingRepetitionDetector(config: config)

        // Feed 250 chars of varied content — no line, no ngram match, but budget fires.
        let filler = String(repeating: "abcdefghij ", count: 25)  // 275 chars
        let decision = detector.ingest(chunk: filler)

        guard case .intervene(.budgetExceeded, let safe) = decision else {
            Issue.record("expected budgetExceeded, got \(decision)")
            return
        }
        #expect(safe.count == 200)
        #expect(safe == String(filler.prefix(200)))
    }

    @Test func charBudgetActiveByDefault() {
        // Default config sets maxThinkingChars = 16_384 as a heuristic
        // backstop for loops whose structural variation delays the line and
        // n-gram signals (e.g. list-counting loops). Feed >16K chars of short,
        // monotonically incrementing lines so neither the line signal
        // (length < minLineLength) nor the n-gram signal (every 40-char
        // slice unique) can fire — budget must be what catches this.
        let detector = ThinkingRepetitionDetector()
        var chunk = ""
        for i in 0..<3_000 {
            chunk += "tok\(padded(i))\n"  // 11 chars each → 33K chars
        }
        let decision = detector.ingest(chunk: chunk)
        guard case .intervene(.budgetExceeded, let safe) = decision else {
            Issue.record("expected budgetExceeded from default budget, got \(decision)")
            return
        }
        #expect(safe.count == 16_384)
    }

    @Test func charBudgetCanBeDisabledViaNil() {
        // Callers that legitimately need longer reasoning can opt out by
        // setting maxThinkingChars: nil. Same varied input should now pass
        // through without any trigger firing.
        let config = ThinkingRepetitionDetector.Config(maxThinkingChars: nil)
        let detector = ThinkingRepetitionDetector(config: config)
        var chunk = ""
        for i in 0..<3_000 {
            chunk += "tok\(padded(i))\n"
        }
        let decision = detector.ingest(chunk: chunk)
        #expect(decision == .continue)
    }

    // MARK: - Enablement and reset

    @Test func disabledConfigAlwaysContinues() {
        let config = ThinkingRepetitionDetector.Config(
            enabled: false, minLineLength: 10, maxLineRepeats: 2
        )
        let detector = ThinkingRepetitionDetector(config: config)
        for _ in 0..<10 {
            let d = detector.ingest(chunk: "The same sentence that would otherwise loop forever.\n")
            #expect(d == .continue)
        }
    }

    @Test func resetClearsStateAllowingSecondTrigger() {
        let config = ThinkingRepetitionDetector.Config(minLineLength: 15, maxLineRepeats: 3)
        let detector = ThinkingRepetitionDetector(config: config)
        let line = "A sentence that keeps repeating.\n"
        for _ in 0..<3 {
            _ = detector.ingest(chunk: line)
        }
        detector.reset()
        // After reset, need another 3 repeats before triggering again.
        let first = detector.ingest(chunk: line)
        let second = detector.ingest(chunk: line)
        let third = detector.ingest(chunk: line)
        #expect(first == .continue)
        #expect(second == .continue)
        if case .intervene(.duplicateLine, _) = third {
            // ok
        } else {
            Issue.record("expected trigger after reset + 3 repeats")
        }
    }

    @Test func emptyChunkIsNoop() {
        let detector = ThinkingRepetitionDetector()
        #expect(detector.ingest(chunk: "") == .continue)
    }

    // MARK: - No false positives on legit long reasoning

    @Test func naturalReasoningDoesNotTrigger() {
        let detector = ThinkingRepetitionDetector()
        // Deliberately varied reasoning — multiple distinct sentences, common
        // "let me verify" re-examinations but not literal duplicates.
        let legit = """
        Let me think about the user's constraints one by one.
        First, line 1 needs to start with [EN] and contain "cat".
        Second, line 2 needs to start with [FR] and contain "chat".
        Third, line 3 needs to start with [ES] and contain "gato".
        Also each line should end with a period and be 3 to 6 words.
        Let me draft candidates: "[EN] The cat sits quietly."
        Checking word count — that's 5 words including the tag. Good.
        Next, "[FR] Le chat dort paisiblement." — 4 words with tag. Good.
        Next, "[ES] El gato duerme tranquilamente." — 4 words with tag. Good.
        All three end with periods. All three contain the required word.
        Final check — constraints all satisfied, confidence high.
        """
        let decision = detector.ingest(chunk: legit)
        #expect(decision == .continue)
    }
}
