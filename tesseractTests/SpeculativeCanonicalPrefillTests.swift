import Foundation
import Testing

@testable import Tesseract_Agent

/// The pure planning arithmetic of the **Speculative Canonical Prefill**
/// (ADR-0009): the margin-trimmed admit path, the worth-it threshold, and
/// the capture-on-preempt threshold. The GPU executor and lifecycle are
/// pinned by `ServerCompletionDrainTests`; the probe itself by
/// `LeafAdmissionBuilderTests`.
struct SpeculativeCanonicalPrefillTests {

    private func path(_ count: Int) -> [Int] {
        Array(0..<count)
    }

    @Test func admitPathTrimsTheSafetyMarginOffTheProbeLCP() {
        let margin = SpeculativeCanonicalPrefill.futureBoundarySafetyMarginTokens
        let future = path(2_000)

        let admit = SpeculativeCanonicalPrefill.admitPath(
            futureSharedPrefix: future,
            canonicalLeafOffset: 100
        )

        #expect(admit?.count == 2_000 - margin)
        #expect(admit == Array(future[0..<(2_000 - margin)]))
    }

    @Test func admitPathSkipsSpansBelowTheResidualThreshold() {
        let margin = SpeculativeCanonicalPrefill.futureBoundarySafetyMarginTokens
        let threshold = SpeculativeCanonicalPrefill.minimumResidualTokens
        // Trimmed span one token short of the threshold.
        let future = path(200 + threshold + margin - 1)

        let admit = SpeculativeCanonicalPrefill.admitPath(
            futureSharedPrefix: future,
            canonicalLeafOffset: 200
        )

        #expect(admit == nil)
    }

    @Test func admitPathAcceptsSpansAtExactlyTheResidualThreshold() {
        let margin = SpeculativeCanonicalPrefill.futureBoundarySafetyMarginTokens
        let threshold = SpeculativeCanonicalPrefill.minimumResidualTokens
        let future = path(200 + threshold + margin)

        let admit = SpeculativeCanonicalPrefill.admitPath(
            futureSharedPrefix: future,
            canonicalLeafOffset: 200
        )

        #expect(admit?.count == 200 + threshold)
    }

    @Test func admitPathSkipsWhenTheLeafAlreadyCoversTheTrimmedTarget() {
        // Degenerate stop-finish turn with no rewind span: the canonical leaf
        // sits at (or past) the trimmed target — nothing to extend.
        let future = path(700)

        let admit = SpeculativeCanonicalPrefill.admitPath(
            futureSharedPrefix: future,
            canonicalLeafOffset: 698
        )

        #expect(admit == nil)
    }

    // MARK: - preemptCaptureOffset — capture-on-preempt (ADR-0009)

    @Test func preemptCaptureKeepsProgressAtTheThreshold() {
        let threshold = SpeculativeCanonicalPrefill.minimumPreemptCaptureTokens

        let offset = SpeculativeCanonicalPrefill.preemptCaptureOffset(
            boundaryOffset: 1_000,
            consumedTokens: threshold,
            canonicalLeafOffset: 1_000
        )

        #expect(offset == 1_000 + threshold)
    }

    @Test func preemptCaptureDropsProgressBelowTheThreshold() {
        let threshold = SpeculativeCanonicalPrefill.minimumPreemptCaptureTokens

        let offset = SpeculativeCanonicalPrefill.preemptCaptureOffset(
            boundaryOffset: 1_000,
            consumedTokens: threshold - 1,
            canonicalLeafOffset: 1_000
        )

        #expect(offset == nil)
    }

    @Test func preemptCaptureDropsProgressNotDeeperThanTheCanonicalLeaf() {
        // A boundary resolved shallower than the canonical leaf (the leaf was
        // evicted between admission and resolve): progress past the threshold
        // can still sit inside the canonical span — admitting it gains the
        // next request nothing.
        let threshold = SpeculativeCanonicalPrefill.minimumPreemptCaptureTokens

        let offset = SpeculativeCanonicalPrefill.preemptCaptureOffset(
            boundaryOffset: 100,
            consumedTokens: threshold,
            canonicalLeafOffset: 100 + threshold
        )

        #expect(offset == nil)
    }

    // MARK: - Stretch Abandonment trigger table (issue #100)

    @Test func stopFinishSeedsImmediatelyAndDurably() {
        let plan = LeafStorePhase.speculativeSeedPlan(
            boundaryMode: .canonical, renderContext: .canonical
        )
        #expect(plan?.idleDelay == .zero)
        #expect(plan?.ramOnlySpine == false)
    }

    @Test func toolStretchArmsTheIdleTimerAndStaysRamOnly() {
        let plan = LeafStorePhase.speculativeSeedPlan(
            boundaryMode: .directTool, renderContext: .canonical
        )
        #expect(plan?.idleDelay == SpeculativeCanonicalPrefill.stretchAbandonmentIdleWindow)
        #expect(plan?.ramOnlySpine == true)
    }

    @Test func preserveThinkingDisablesEveryTrigger() {
        let preserve = TemplateRenderContext(
            flags: [.preserveThinking]
        )
        #expect(
            LeafStorePhase.speculativeSeedPlan(
                boundaryMode: .canonical, renderContext: preserve
            ) == nil)
        #expect(
            LeafStorePhase.speculativeSeedPlan(
                boundaryMode: .directTool, renderContext: preserve
            ) == nil)
    }
}
