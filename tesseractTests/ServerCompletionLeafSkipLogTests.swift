import Testing

@testable import Tesseract_Agent

/// Byte-for-byte wire format of the leaf-skip diagnostics — the stage / reason /
/// level / fields the dissolved `captureDirectToolLeaf` and
/// `captureCanonicalTemplateLeaf` helpers logged. `ServerCompletion.leafSkipLog` is the
/// single pure mapping behind `logLeafSkip`; pinning it here means a renamed
/// stage label or a flipped log level fails a test rather than silently shifting
/// dashboards and the diagnostics net. (Prior art: `ssdDropReasonString`.)
struct ServerCompletionLeafSkipLogTests {

    private func fields(_ log: ServerCompletion.LeafSkipLog) -> [[String]] {
        log.extraFields.map { [$0.0, $0.1] }
    }

    // MARK: stage prefix follows the boundary mode

    @Test func directToolStagePrefixMatchesTheDissolvedHelper() {
        let log = ServerCompletion.leafSkipLog(for: .noTransientBoundary, mode: .directTool)
        #expect(log.stage == "directToolLeafStore")
        #expect(log.reason == "no-transient-boundary-snapshot")
        #expect(log.level == .info)
        #expect(log.extraFields.isEmpty)
    }

    @Test func canonicalStagePrefixMatchesTheDissolvedHelper() {
        let log = ServerCompletion.leafSkipLog(for: .noResolvedBoundary(canonicalLen: 12), mode: .canonical)
        #expect(log.stage == "canonicalLeafStore")
        #expect(log.reason == "no-canonical-restore-boundary")
        #expect(log.level == .info)
        #expect(fields(log) == [["canonicalLen", "12"]])
    }

    // MARK: each reason's reason-string / level / fields

    @Test func tokenizationFailureIsPrefillThrewAtWarning() {
        let log = ServerCompletion.leafSkipLog(for: .tokenizationFailed(error: "boom"), mode: .directTool)
        #expect(log.stage == "directToolLeafStore")
        #expect(log.reason == "prefill-threw")
        #expect(log.level == .warning)
        #expect(fields(log) == [["error", "boom"]])
    }

    @Test func probeDivergenceIsInfoWithNoFields() {
        let log = ServerCompletion.leafSkipLog(for: .probeDivergence, mode: .canonical)
        #expect(log.stage == "canonicalLeafStore")
        #expect(log.reason == "probe-divergence-failed")
        #expect(log.level == .info)
        #expect(log.extraFields.isEmpty)
    }

    @Test func storedAtOrBeforeBoundaryCarriesStoredLenThenBoundaryOffset() {
        let log = ServerCompletion.leafSkipLog(
            for: .storedAtOrBeforeBoundary(storedLen: 7, boundaryOffset: 7), mode: .directTool
        )
        #expect(log.stage == "directToolLeafStore")
        #expect(log.reason == "stored-at-or-before-boundary")
        #expect(log.level == .info)
        #expect(fields(log) == [["storedLen", "7"], ["boundaryOffset", "7"]])
    }

    @Test func canonicalLongerThanStoredIsWarningWithBothLengths() {
        let log = ServerCompletion.leafSkipLog(
            for: .canonicalLongerThanStored(canonicalLen: 9, storedLen: 4), mode: .canonical
        )
        #expect(log.stage == "canonicalLeafStore")
        #expect(log.reason == "canonical-longer-than-stored")
        #expect(log.level == .warning)
        #expect(fields(log) == [["canonicalLen", "9"], ["storedLen", "4"]])
    }

    @Test func renderTranslationFailureIsWarningWithTheTypedFailure() {
        let log = ServerCompletion.leafSkipLog(
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
        let log = ServerCompletion.leafSkipLog(
            for: .boundaryInsideImagePrefix(boundaryOffset: 3, minimumWarmOffset: 9),
            mode: .directTool
        )
        #expect(log.stage == "directToolLeafStore")
        #expect(log.reason == "boundary-inside-image-prefix")
        #expect(log.level == .info)
        #expect(fields(log) == [["boundaryOffset", "3"], ["minimumWarmOffset", "9"]])
    }
}
