import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

struct TriAttentionRuntimeSelectionTests {

    private func makeArtifact() -> TriAttentionCalibrationArtifact {
        let sampledHeads = [
            TriAttentionCalibrationHeadKey(layerIndex: 1, headIndex: 0),
            TriAttentionCalibrationHeadKey(layerIndex: 1, headIndex: 1),
        ]
        let stats = Dictionary(uniqueKeysWithValues: sampledHeads.map { head in
            (
                head,
                TriAttentionCalibrationHeadStats(
                    qMeanReal: [1, 2, 3, 4],
                    qMeanImag: [0.5, 1, 1.5, 2],
                    qAbsMean: [2, 3, 4, 5]
                )
            )
        })
        return TriAttentionCalibrationArtifact(
            metadata: TriAttentionCalibrationMetadata(
                sampledHeads: sampledHeads,
                headDim: 8,
                ropeStyle: "half",
                modelName: "tests/qwen35-paro"
            ),
            statsByHead: stats
        )
    }

    private func makeFakeModelDirectory(paro: Bool = false) throws -> URL {
        try TriAttentionTestFixtures.makeFakeModelDirectory(
            prefix: "triattention-selection-model",
            paro: paro
        )
    }

    private func makeScratchDir(prefix: String) throws -> URL {
        try TriAttentionTestFixtures.makeScratchDir(prefix: prefix)
    }

    @Test
    func resolveEnablesTriAttentionForEligibleParoTextArtifact() {
        let requested = TriAttentionConfiguration(enabled: true)
        let identity = TriAttentionCalibrationArtifactIdentity(rawValue: "artifact")
        let selection = TriAttentionRuntimeSelection.resolve(
            requestedConfiguration: requested,
            isTriAttentionEligible: true,
            visionMode: false,
            calibrationArtifactLookup: .loaded(
                artifact: makeArtifact(),
                identity: identity,
                relativeResourcePath: "TriAttention/v1/test.pt"
            )
        )

        #expect(selection.requestedConfiguration == requested)
        #expect(selection.effectiveConfiguration.enabled == true)
        #expect(selection.effectiveConfiguration.calibrationArtifactIdentity == identity)
        #expect(selection.effectiveConfiguration.prefixProtectionMode == requested.prefixProtectionMode)
        #expect(selection.fallbackReason == nil)
    }

    @Test
    func resolveFallsBackForVisionModeWithoutLookup() {
        let requested = TriAttentionConfiguration(enabled: true)
        let selection = TriAttentionRuntimeSelection.resolve(
            requestedConfiguration: requested,
            isTriAttentionEligible: true,
            visionMode: true,
            calibrationArtifactLookup: nil
        )

        #expect(selection.effectiveConfiguration.enabled == false)
        #expect(selection.effectiveConfiguration.prefixProtectionMode == requested.prefixProtectionMode)
        #expect(selection.fallbackReason == .visionMode)
        switch selection.calibrationArtifactLookup {
        case nil:
            break
        default:
            Issue.record("Vision-mode fallback should not perform a calibration lookup")
        }
    }

    @Test
    func resolveFallsBackForUnsupportedModelWithoutLookup() {
        let requested = TriAttentionConfiguration(enabled: true)
        let selection = TriAttentionRuntimeSelection.resolve(
            requestedConfiguration: requested,
            isTriAttentionEligible: false,
            visionMode: false,
            calibrationArtifactLookup: nil
        )

        #expect(selection.effectiveConfiguration.enabled == false)
        #expect(selection.effectiveConfiguration.prefixProtectionMode == requested.prefixProtectionMode)
        #expect(selection.fallbackReason == .unsupportedModel)
        switch selection.calibrationArtifactLookup {
        case nil:
            break
        default:
            Issue.record("Unsupported-model fallback should not perform a calibration lookup")
        }
    }

    @Test
    func resolveFallsBackForMissingArtifact() {
        let requested = TriAttentionConfiguration(enabled: true)
        let selection = TriAttentionRuntimeSelection.resolve(
            requestedConfiguration: requested,
            isTriAttentionEligible: true,
            visionMode: false,
            calibrationArtifactLookup: .missing(
                expectedModelFingerprint: "fingerprint",
                expectedURL: URL(fileURLWithPath: "/tmp/fingerprint.pt")
            )
        )

        #expect(selection.effectiveConfiguration.enabled == false)
        #expect(selection.effectiveConfiguration.prefixProtectionMode == requested.prefixProtectionMode)
        #expect(selection.fallbackReason == .missingCalibrationArtifact)
    }

    @Test
    func resolveFallsBackForFingerprintMismatch() {
        let requested = TriAttentionConfiguration(enabled: true)
        let selection = TriAttentionRuntimeSelection.resolve(
            requestedConfiguration: requested,
            isTriAttentionEligible: true,
            visionMode: false,
            calibrationArtifactLookup: .fingerprintMismatch(
                expectedModelFingerprint: "expected",
                actualModelFingerprint: "actual",
                url: URL(fileURLWithPath: "/tmp/actual.pt")
            )
        )

        #expect(selection.effectiveConfiguration.enabled == false)
        #expect(selection.effectiveConfiguration.prefixProtectionMode == requested.prefixProtectionMode)
        #expect(selection.fallbackReason == .mismatchedCalibrationArtifact)
    }

    @Test
    func llmActorLoadFallsBackForNonParoWithoutLookup() async throws {
        let modelDir = try makeFakeModelDirectory(paro: false)
        let root = try makeScratchDir(prefix: "triattention-selection-root")
        defer {
            try? FileManager.default.removeItem(at: modelDir)
            try? FileManager.default.removeItem(at: root)
        }

        let actor = LLMActor(
            triAttentionCalibrationArtifactLoader: TriAttentionCalibrationArtifactLoader(rootURL: root)
        )

        do {
            try await actor.loadModel(
                from: modelDir,
                visionMode: false,
                triAttention: TriAttentionConfiguration(enabled: true)
            )
            Issue.record("Expected fake model load to fail before container verification")
        } catch {
            // Expected: fingerprintable but not loadable.
        }

        let selection = await actor.currentTriAttentionRuntimeSelectionForTesting
        #expect(selection.fallbackReason == .unsupportedModel)
        #expect(selection.effectiveConfiguration.enabled == false)
        switch selection.calibrationArtifactLookup {
        case nil:
            break
        default:
            Issue.record("Non-PARO fallback should not perform a calibration lookup")
        }
    }

    @Test
    func llmActorLoadFallsBackForVisionModeWithoutLookup() async throws {
        let modelDir = try makeFakeModelDirectory(paro: true)
        let root = try makeScratchDir(prefix: "triattention-selection-root")
        defer {
            try? FileManager.default.removeItem(at: modelDir)
            try? FileManager.default.removeItem(at: root)
        }

        let actor = LLMActor(
            triAttentionCalibrationArtifactLoader: TriAttentionCalibrationArtifactLoader(rootURL: root)
        )

        do {
            try await actor.loadModel(
                from: modelDir,
                visionMode: true,
                triAttention: TriAttentionConfiguration(enabled: true)
            )
            Issue.record("Expected fake model load to fail before container verification")
        } catch {
            // Expected: fingerprintable but not loadable.
        }

        let selection = await actor.currentTriAttentionRuntimeSelectionForTesting
        #expect(selection.fallbackReason == .visionMode)
        #expect(selection.effectiveConfiguration.enabled == false)
        switch selection.calibrationArtifactLookup {
        case nil:
            break
        default:
            Issue.record("Vision-mode fallback should not perform a calibration lookup")
        }
    }
}
