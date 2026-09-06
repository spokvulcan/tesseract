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
            for: .intervened, mode: .directTool, preservesThinking: true)
        #expect(log.stage == "liveLeafCapture")
        #expect(log.reason == "intervened")
        #expect(log.level == .info)
        #expect(fields(log) == [["mode", "directToolLeaf"], ["preservesThinking", "true"]])
        #expect(
            LeafStorePhase.liveFallbackLog(
                for: .nonIdentityKeySpace, mode: .canonical, preservesThinking: false
            ).reason == "non-identity-key-space")
        #expect(
            LeafStorePhase.liveFallbackLog(
                for: .noGeneratedTokens, mode: .canonical, preservesThinking: false
            ).level == .info)
    }

    @Test func liveFallbackDisagreementWarnsOnAnAppendStableRender() {
        let divergence = LiveLeafCapture.FallbackReason.divergence(
            offset: 9, liveToken: 1, storedToken: 2, liveContext: [1], storedContext: [2])
        let preserved = LeafStorePhase.liveFallbackLog(
            for: divergence, mode: .canonical, preservesThinking: true)
        #expect(preserved.level == .warning)
        #expect(preserved.reason == "divergence")
        #expect(
            fields(preserved) == [
                ["offset", "9"], ["liveToken", "1"], ["storedToken", "2"],
                ["liveContext", "[1]"], ["storedContext", "[2]"],
                ["mode", "canonicalLeaf"], ["preservesThinking", "true"],
            ])
        // A tool stretch is append-stable under every template.
        let toolStretch = LeafStorePhase.liveFallbackLog(
            for: .liveLongerThanStored(cacheOffset: 12, storedLen: 10),
            mode: .directTool, preservesThinking: false)
        #expect(toolStretch.level == .warning)
        #expect(
            fields(toolStretch) == [
                ["cacheOffset", "12"], ["storedLen", "10"],
                ["mode", "directToolLeaf"], ["preservesThinking", "false"],
            ])
    }

    @Test func liveFallbackDisagreementIsInfoUnderTheStripByDefaultCanonicalRender() {
        // The strip-by-default render drops the emitted thinking, so the
        // canonical path ends before the live one by design.
        let log = LeafStorePhase.liveFallbackLog(
            for: .liveLongerThanStored(cacheOffset: 12, storedLen: 10),
            mode: .canonical, preservesThinking: false)
        #expect(log.level == .info)
        #expect(log.reason == "live-longer-than-stored")
    }

    @Test func liveFallbackCacheOffsetOutsideTheLivePathAlwaysWarns() {
        // The loop and the cache disagree about what was fed — never
        // expected, whatever the render.
        let log = LeafStorePhase.liveFallbackLog(
            for: .cacheOffsetOutsideLivePath(cacheOffset: 4, promptCount: 4, liveCount: 8),
            mode: .canonical, preservesThinking: false)
        #expect(log.level == .warning)
        #expect(log.reason == "cache-offset-outside-live-path")
        #expect(
            fields(log) == [
                ["cacheOffset", "4"], ["promptCount", "4"], ["liveCount", "8"],
                ["mode", "canonicalLeaf"], ["preservesThinking", "false"],
            ])
    }
}
