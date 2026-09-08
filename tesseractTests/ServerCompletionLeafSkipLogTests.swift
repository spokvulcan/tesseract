import Testing

@testable import Tesseract_Agent

/// Byte-for-byte wire format of the leaf-skip diagnostics — the stage / reason /
/// level / fields the dissolved `captureDirectToolLeaf` and
/// `captureCanonicalTemplateLeaf` helpers logged, and the **Live Leaf Capture**
/// fallback record beside them. `LeafStorePhase.leafSkipLog` and
/// `LeafStorePhase.liveFallbackLog` are the two pure mappings; pinning them here
/// means a renamed stage label or a flipped log level fails a test rather than
/// silently shifting dashboards and the diagnostics net. (Prior art:
/// `ssdDropReasonString`.)
struct ServerCompletionLeafSkipLogTests {

    private func fields(_ log: LeafStorePhase.LeafSkipLog) -> [[String]] {
        log.extraFields.map { [$0.0, $0.1] }
    }

    // MARK: stage prefix follows the boundary mode

    @Test func directToolStagePrefixMatchesTheDissolvedHelper() {
        let log = LeafStorePhase.leafSkipLog(for: .noTransientBoundary, mode: .directTool)
        #expect(log.stage == "directToolLeafStore")
        #expect(log.reason == "no-transient-boundary-snapshot")
        #expect(log.level == .info)
        #expect(log.extraFields.isEmpty)
    }

    @Test func canonicalStagePrefixMatchesTheDissolvedHelper() {
        let log = LeafStorePhase.leafSkipLog(
            for: .noResolvedBoundary(canonicalLen: 12), mode: .canonical)
        #expect(log.stage == "canonicalLeafStore")
        #expect(log.reason == "no-canonical-restore-boundary")
        #expect(log.level == .info)
        #expect(fields(log) == [["canonicalLen", "12"]])
    }

    // MARK: each reason's reason-string / level / fields

    @Test func tokenizationFailureIsPrefillThrewAtWarning() {
        let log = LeafStorePhase.leafSkipLog(
            for: .tokenizationFailed(error: "boom"), mode: .directTool)
        #expect(log.stage == "directToolLeafStore")
        #expect(log.reason == "prefill-threw")
        #expect(log.level == .warning)
        #expect(fields(log) == [["error", "boom"]])
    }

    @Test func probeDivergenceIsInfoWithNoFields() {
        let log = LeafStorePhase.leafSkipLog(for: .probeDivergence, mode: .canonical)
        #expect(log.stage == "canonicalLeafStore")
        #expect(log.reason == "probe-divergence-failed")
        #expect(log.level == .info)
        #expect(log.extraFields.isEmpty)
    }

    @Test func storedAtOrBeforeBoundaryCarriesStoredLenThenBoundaryOffset() {
        let log = LeafStorePhase.leafSkipLog(
            for: .storedAtOrBeforeBoundary(storedLen: 7, boundaryOffset: 7), mode: .directTool
        )
        #expect(log.stage == "directToolLeafStore")
        #expect(log.reason == "stored-at-or-before-boundary")
        #expect(log.level == .info)
        #expect(fields(log) == [["storedLen", "7"], ["boundaryOffset", "7"]])
    }

    @Test func canonicalLongerThanStoredIsWarningWithBothLengths() {
        let log = LeafStorePhase.leafSkipLog(
            for: .canonicalLongerThanStored(canonicalLen: 9, storedLen: 4), mode: .canonical
        )
        #expect(log.stage == "canonicalLeafStore")
        #expect(log.reason == "canonical-longer-than-stored")
        #expect(log.level == .warning)
        #expect(fields(log) == [["canonicalLen", "9"], ["storedLen", "4"]])
    }

    @Test func renderTranslationFailureIsWarningWithTheTypedFailure() {
        let log = LeafStorePhase.leafSkipLog(
            for: .renderTranslationFailed(
                failure: .placeholderOccurrencesExceedImages(occurrences: 2, images: 1)
            ),
            mode: .directTool
        )
        #expect(log.stage == "directToolLeafStore")
        #expect(log.reason == "render-translation-failed")
        #expect(log.level == .warning)
        #expect(log.extraFields.count == 1)
        #expect(log.extraFields[0].0 == "failure")
    }

    @Test func boundaryInsideImagePrefixIsInfoWithBothOffsets() {
        let log = LeafStorePhase.leafSkipLog(
            for: .boundaryInsideImagePrefix(boundaryOffset: 3, minimumWarmOffset: 9),
            mode: .directTool
        )
        #expect(log.stage == "directToolLeafStore")
        #expect(log.reason == "boundary-inside-image-prefix")
        #expect(log.level == .info)
        #expect(fields(log) == [["boundaryOffset", "3"], ["minimumWarmOffset", "9"]])
    }

    // MARK: the Live Leaf Capture fallback record

    @Test func liveFallbackEligibilityReasonsAreInfoWithTheModeFields() {
        let log = LeafStorePhase.liveFallbackLog(
            for: .noGeneratedTokens, mode: .directToolLeaf, preservesThinking: true)
        #expect(log.stage == "liveLeafCapture")
        #expect(log.reason == "no-generated-tokens")
        #expect(log.level == .info)
        #expect(fields(log) == [["mode", "directToolLeaf"], ["preservesThinking", "true"]])
        #expect(
            LeafStorePhase.liveFallbackLog(
                for: .nonIdentityKeySpace, mode: .canonicalUserLeaf, preservesThinking: false
            ).reason == "non-identity-key-space")
        #expect(
            LeafStorePhase.liveFallbackLog(
                for: .noGeneratedTokens, mode: .canonicalUserLeaf, preservesThinking: false
            ).level == .info)
    }

    @Test func liveFallbackUnderANonThinkingTemplateNamesTheDirectMode() {
        // An image-bearing request under a non-thinking template: the
        // pre-existing direct labels, so dashboards keep their vocabulary.
        let log = LeafStorePhase.liveFallbackLog(
            for: .nonIdentityKeySpace, mode: .directLeaf, preservesThinking: false)
        #expect(log.reason == "non-identity-key-space")
        #expect(fields(log) == [["mode", "leaf"], ["preservesThinking", "false"]])
    }

    @Test func liveFallbackThinkStrippingUserBoundaryIsInfo() {
        // The expected shape of a strip-by-default template at a stop
        // finish — ADR-0009's boundary path, not a disagreement.
        let log = LeafStorePhase.liveFallbackLog(
            for: .thinkStrippingUserBoundary, mode: .canonicalUserLeaf, preservesThinking: false)
        #expect(log.stage == "liveLeafCapture")
        #expect(log.reason == "think-stripping-user-boundary")
        #expect(log.level == .info)
        #expect(fields(log) == [["mode", "canonicalLeaf"], ["preservesThinking", "false"]])
    }

    @Test func liveFallbackCacheOffsetOutsideTheLivePathAlwaysWarns() {
        // The loop and the cache disagree about what was fed — never
        // expected, whatever the render.
        let log = LeafStorePhase.liveFallbackLog(
            for: .cacheOffsetOutsideLivePath(cacheOffset: 4, promptCount: 4, liveCount: 8),
            mode: .canonicalUserLeaf, preservesThinking: false)
        #expect(log.level == .warning)
        #expect(log.reason == "cache-offset-outside-live-path")
        #expect(
            fields(log) == [
                ["cacheOffset", "4"], ["promptCount", "4"], ["liveCount", "8"],
                ["mode", "canonicalLeaf"], ["preservesThinking", "false"],
            ])
    }

    // MARK: the leafStore event's source and boundary fields

    private func reportFields(_ report: LeafStorePhase.Report) -> [String: String] {
        Dictionary(report.fields, uniquingKeysWith: { first, _ in first })
    }

    @Test func aLiveLeafReportsItsSourceAndNoBoundary() {
        var report = LeafStorePhase.Report()
        report.mode = HTTPLeafStoreMode.directToolLeaf.rawValue
        report.absorb(LeafStorePhase.LeafCapture(leafOffset: 120), path: .live)
        let fields = reportFields(report)
        #expect(fields["path"] == "live")
        #expect(fields["source"] == "live")
        #expect(fields["boundary"] == nil)
        #expect(fields["leafOffset"] == "120")
        #expect(fields["residualTokens"] == "0")
    }

    @Test func aCopiedImageLeafReportsItsCopyReason() {
        var report = LeafStorePhase.Report()
        report.absorb(
            LeafStorePhase.LeafCapture(leafOffset: 120, copyReason: .imageKeySpace), path: .direct)
        let fields = reportFields(report)
        #expect(fields["source"] == "live")
        #expect(fields["copyReason"] == "imageKeySpace")
    }

    @Test func aBoundaryLeafReportsItsSourceAndTheBoundaryReason() {
        var report = LeafStorePhase.Report()
        report.boundaryReason = "think-stripping-user-boundary"
        report.absorb(
            LeafStorePhase.LeafCapture(leafOffset: 300, residualTokens: 42), path: .boundary)
        let fields = reportFields(report)
        #expect(fields["path"] == "boundary")
        #expect(fields["source"] == "boundary")
        #expect(fields["boundary"] == "think-stripping-user-boundary")
        #expect(fields["residualTokens"] == "42")
    }

    @Test func aSkippedStoreReportsNoSourceButKeepsTheBoundaryReason() {
        var report = LeafStorePhase.Report()
        report.boundaryReason = "intervened"
        report.absorb(LeafStorePhase.LeafCapture(skipReason: "prefill-threw"), path: .boundary)
        let fields = reportFields(report)
        #expect(fields["path"] == "skipped")
        #expect(fields["source"] == nil)
        #expect(fields["boundary"] == "intervened")
        #expect(fields["skip"] == "prefill-threw")
    }

    @Test func theDirectExecutorsLeafIsLiveSourced() {
        var report = LeafStorePhase.Report()
        report.boundaryReason = "non-identity-key-space"
        report.absorb(LeafStorePhase.LeafCapture(leafOffset: 9), path: .direct)
        let fields = reportFields(report)
        #expect(fields["path"] == "direct")
        #expect(fields["source"] == "live")
        #expect(fields["boundary"] == "non-identity-key-space")
    }
}
