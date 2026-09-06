import Testing

@testable import Tesseract_Agent

/// The **Live Leaf Capture** decision, pinned without a model: the live token
/// path (prompt key path + the ids the loop fed) must be a prefix of the
/// canonical stored path for the live final cache to become the leaf; every
/// other outcome names a typed fallback reason carrying the numbers its skip
/// record prints (pinned in `ServerCompletionLeafSkipLogTests`). The rules
/// are what make the zero-prefill tail safe, so each one gets a test that
/// would fail if it were relaxed.
struct LiveLeafCaptureTests {

    private let prompt = [1, 2, 3, 4]
    private let generated = [10, 11, 12, 99]  // 99 plays the stop token
    /// The canonical re-render: prompt + emitted ids + template glue (`\n`).
    private var stored: [Int] { prompt + generated + [7] }

    private func decide(
        generated: [Int]? = nil,
        cacheOffset: Int? = nil,
        stored: [Int]? = nil,
        intervened: Bool = false,
        identity: Bool = true
    ) -> LiveLeafCapture.Decision {
        let gen = generated ?? self.generated
        return LiveLeafCapture.decide(
            promptKeyPath: prompt,
            generatedTokens: gen,
            cacheOffset: cacheOffset ?? prompt.count + gen.count,
            storedTokens: stored ?? self.stored,
            intervened: intervened,
            keySpaceIsIdentity: identity
        )
    }

    // MARK: live

    @Test func autoregressiveTurnCapturesAtTheFullLiveOffset() {
        // AR decode feeds every returned token, stop token included: the
        // cache offset equals prompt + generated.
        #expect(decide() == .live(offset: prompt.count + generated.count))
    }

    @Test func draftedTurnWithUnfedBonusCapturesAtTheCacheOffset() {
        // DFlash2's finalize can leave the last returned token unfed; the
        // leaf is taken at the cache's own offset, never past it.
        let offset = prompt.count + generated.count - 1
        #expect(decide(cacheOffset: offset) == .live(offset: offset))
    }

    @Test func storedPathIdenticalToLivePathIsLive() {
        // No glue at all (a template that renders nothing after EOS).
        #expect(decide(stored: prompt + generated) == .live(offset: prompt.count + generated.count))
    }

    // MARK: eligibility

    @Test func intervenedTurnFallsBackBeforeAnyComparison() {
        // Even a perfectly matching path is refused: the registered final
        // cache belongs to the cancelled phase.
        #expect(decide(intervened: true) == .boundary(.intervened))
    }

    @Test func nonIdentityKeySpaceFallsBack() {
        #expect(decide(identity: false) == .boundary(.nonIdentityKeySpace))
    }

    @Test func emptyGenerationFallsBack() {
        #expect(decide(generated: [], cacheOffset: prompt.count) == .boundary(.noGeneratedTokens))
    }

    // MARK: cache offset vs live path

    @Test func cacheOffsetAtThePromptEndIsOutsideTheLivePath() {
        // Nothing generated reached the cache — no leaf beyond the prompt.
        let decision = decide(cacheOffset: prompt.count)
        #expect(
            decision
                == .boundary(
                    .cacheOffsetOutsideLivePath(
                        cacheOffset: prompt.count, promptCount: prompt.count,
                        liveCount: prompt.count + generated.count)))
    }

    @Test func cacheOffsetPastTheLivePathIsOutsideTheLivePath() {
        let liveCount = prompt.count + generated.count
        let decision = decide(cacheOffset: liveCount + 1)
        #expect(
            decision
                == .boundary(
                    .cacheOffsetOutsideLivePath(
                        cacheOffset: liveCount + 1, promptCount: prompt.count,
                        liveCount: liveCount)))
    }

    // MARK: live vs stored

    @Test func livePathLongerThanStoredFallsBack() {
        // The render dropped an emitted token (whitespace normalization):
        // the live state past the stored end has no key.
        let shortStored = prompt + generated.dropLast()
        let decision = decide(stored: shortStored)
        #expect(
            decision
                == .boundary(
                    .liveLongerThanStored(
                        cacheOffset: prompt.count + generated.count,
                        storedLen: shortStored.count)))
    }

    @Test func divergenceAtTheFirstGeneratedTokenNamesTheOffset() {
        // A strip-by-default render replaces the emitted `<think>` opener.
        var reRendered = stored
        reRendered[prompt.count] = 500
        let decision = decide(stored: reRendered)
        guard
            case .boundary(
                .divergence(let offset, let live, let storedTok, let liveCtx, let storedCtx)) =
                decision
        else {
            Issue.record("expected a divergence, got \(decision)")
            return
        }
        #expect(offset == prompt.count)
        #expect(live == generated[0])
        #expect(storedTok == 500)
        // Four ids either side of the divergence, clipped to the paths.
        #expect(liveCtx == [1, 2, 3, 4, 10, 11, 12, 99])
        #expect(storedCtx == [1, 2, 3, 4, 500, 11, 12, 99])
    }

    @Test func divergenceInsideThePromptIsCaughtToo() {
        // The prompt prefix is the same render on both sides by
        // construction; the comparison still proves it.
        var reRendered = stored
        reRendered[1] = 42
        guard
            case .boundary(.divergence(let offset, let live, let storedTok, let liveCtx, _)) =
                decide(stored: reRendered)
        else {
            Issue.record("expected a divergence inside the prompt")
            return
        }
        #expect(offset == 1)
        #expect(live == prompt[1])
        #expect(storedTok == 42)
        #expect(liveCtx == [1, 2, 3, 4, 10, 11])
    }

    @Test func divergencePastTheCacheOffsetIsNotCompared() {
        // The unfed bonus token differs from the render — irrelevant, the
        // leaf ends at the cache offset.
        var reRendered = stored
        let unfedIndex = prompt.count + generated.count - 1
        reRendered[unfedIndex] = 777
        #expect(decide(cacheOffset: unfedIndex, stored: reRendered) == .live(offset: unfedIndex))
    }
}
