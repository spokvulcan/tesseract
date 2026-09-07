import Testing

@testable import Tesseract_Agent

/// The **Leaf Store** fast path's eligibility (ADR-0063 decisions 10 to
/// 12), pinned without a model: a finished turn stores its leaf straight
/// from the live decode cache, under the **Emitted Path**, whenever the
/// template renders the turn verbatim for the next request and the
/// structural guards hold. Nothing is compared against a re-render any
/// more — the per-token comparison of ADR-0062 is gone — so each rule here
/// is one that, if relaxed, would key a leaf on a path the model never fed
/// or send a verbatim-rendered turn to a needless re-prefill.
struct LeafStoreFastPathTests {

    private let prompt = [1, 2, 3, 4]
    private let generated = [10, 11, 12, 99]  // 99 plays the stop token

    /// Defaults to the shape every rule accepts — a tool-stretch turn under
    /// the preserve-thinking render — so each test names only what it varies.
    private func decide(
        mode: HTTPLeafStoreMode = .directToolLeaf,
        preservesThinking: Bool = true,
        generated: [Int]? = nil,
        cacheOffset: Int? = nil,
        identity: Bool = true
    ) -> LiveLeafCapture.Decision {
        let gen = generated ?? self.generated
        return LiveLeafCapture.decide(
            mode: mode,
            preservesThinking: preservesThinking,
            promptKeyPath: prompt,
            generatedTokens: gen,
            cacheOffset: cacheOffset ?? prompt.count + gen.count,
            keySpaceIsIdentity: identity
        )
    }

    // MARK: the fast path

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

    @Test func everyModeIsLiveUnderThePreserveThinkingRender() {
        let offset = prompt.count + generated.count
        for mode in [HTTPLeafStoreMode.directToolLeaf, .canonicalUserLeaf, .directLeaf] {
            #expect(decide(mode: mode, preservesThinking: true) == .live(offset: offset))
        }
    }

    @Test func toolStretchTurnsAreLiveUnderAThinkStrippingTemplate() {
        // The Qwen3-family templates render every tool-stretch turn
        // verbatim for the tool-result continuation.
        #expect(
            decide(mode: .directToolLeaf, preservesThinking: false)
                == .live(offset: prompt.count + generated.count))
    }

    @Test func nonThinkingTemplateStopTurnsAreLive() {
        #expect(
            decide(mode: .directLeaf, preservesThinking: false)
                == .live(offset: prompt.count + generated.count))
    }

    // MARK: the boundary path

    @Test func aThinkStrippingTemplateAtAUserBoundaryKeepsTheBoundaryPath() {
        // The next user message strips this turn's thinking, so the live
        // path is not what the next request renders: ADR-0009's answer
        // stays (decision 11).
        #expect(
            decide(mode: .canonicalUserLeaf, preservesThinking: false)
                == .boundary(.thinkStrippingUserBoundary))
    }

    @Test func nonIdentityKeySpaceFallsBack() {
        #expect(decide(preservesThinking: true, identity: false) == .boundary(.nonIdentityKeySpace))
    }

    @Test func emptyGenerationFallsBack() {
        #expect(
            decide(preservesThinking: true, generated: [], cacheOffset: prompt.count)
                == .boundary(.noGeneratedTokens))
    }

    @Test func cacheOffsetAtThePromptEndIsOutsideTheLivePath() {
        // Nothing generated reached the cache — no leaf beyond the prompt.
        #expect(
            decide(preservesThinking: true, cacheOffset: prompt.count)
                == .boundary(
                    .cacheOffsetOutsideLivePath(
                        cacheOffset: prompt.count, promptCount: prompt.count,
                        liveCount: prompt.count + generated.count)))
    }

    @Test func cacheOffsetPastTheLivePathIsOutsideTheLivePath() {
        let liveCount = prompt.count + generated.count
        #expect(
            decide(preservesThinking: true, cacheOffset: liveCount + 1)
                == .boundary(
                    .cacheOffsetOutsideLivePath(
                        cacheOffset: liveCount + 1, promptCount: prompt.count,
                        liveCount: liveCount)))
    }

    @Test func theStructuralGuardsAreNamedBeforeTheRenderRule() {
        #expect(
            decide(mode: .canonicalUserLeaf, preservesThinking: false, identity: false)
                == .boundary(.nonIdentityKeySpace))
    }

    // MARK: the path the leaf is admitted under

    @Test func theLivePathIsThePromptKeyPathPlusTheFedIdsUpToTheCacheOffset() {
        // An unfed bonus token past the cache offset is not in the path.
        #expect(
            LiveLeafCapture.livePath(
                promptKeyPath: prompt, generatedTokens: generated, offset: prompt.count + 3)
                == [1, 2, 3, 4, 10, 11, 12])
        #expect(
            LiveLeafCapture.livePath(
                promptKeyPath: prompt, generatedTokens: generated,
                offset: prompt.count + generated.count)
                == prompt + generated)
    }
}
