import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

struct TriAttentionCalibrationArtifactLoaderTests {

    private func fixtureURL() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent(
                "Vendor/mlx-swift-lm/Tests/MLXLMTests/Resources/TriAttention/triattention-minimal-valid.pt",
                isDirectory: false
            )
    }

    private func makeScratchDir(prefix: String) throws -> URL {
        try TriAttentionTestFixtures.makeScratchDir(prefix: prefix)
    }

    private func copyFixture(to destinationURL: URL) throws {
        let data = try Data(contentsOf: fixtureURL())
        try data.write(to: destinationURL)
    }

    private func writeInvalidArtifact(to destinationURL: URL) throws {
        try Data("not a torch zip archive".utf8).write(to: destinationURL)
    }

    private func makeFakeModelDirectory(paro: Bool = false) throws -> URL {
        try TriAttentionTestFixtures.makeFakeModelDirectory(
            prefix: "triattention-fake-model",
            paro: paro
        )
    }

    private func makeFakeModelDirectory(kind: TriAttentionTestFixtures.Kind) throws -> URL {
        try TriAttentionTestFixtures.makeFakeModelDirectory(
            prefix: "triattention-fake-model",
            kind: kind
        )
    }

    @Test
    func lookupLoadsArtifactForMatchingFingerprint() throws {
        let modelDir = try makeFakeModelDirectory()
        let root = try makeScratchDir(prefix: "triattention-loader-root")
        defer {
            try? FileManager.default.removeItem(at: modelDir)
            try? FileManager.default.removeItem(at: root)
        }

        let fingerprint = try ModelFingerprint.computeFingerprint(modelDir: modelDir)
        let artifactURL = root.appendingPathComponent("\(fingerprint).pt", isDirectory: false)
        try copyFixture(to: artifactURL)

        let loader = TriAttentionCalibrationArtifactLoader(rootURL: root)
        let result = try loader.lookup(modelFingerprint: fingerprint)
        let expectedIdentity = TriAttentionCalibrationArtifactIdentity.sha256(
            of: try Data(contentsOf: artifactURL)
        )

        switch result {
        case .loaded(let artifact, let identity, let relativeResourcePath):
            #expect(identity == expectedIdentity)
            #expect(relativeResourcePath == TriAttentionCalibrationArtifactLoader.relativeResourcePath(modelFingerprint: fingerprint))
            #expect(artifact.metadata.headDim == 8)
            #expect(artifact.metadata.sampledHeads.count == 2)
        default:
            Issue.record("Expected loaded TriAttention artifact result")
        }
    }

    @Test
    func lookupReturnsMissingWhenArtifactAbsent() throws {
        let root = try makeScratchDir(prefix: "triattention-loader-missing")
        defer { try? FileManager.default.removeItem(at: root) }

        let loader = TriAttentionCalibrationArtifactLoader(rootURL: root)
        let fingerprint = "missing-fingerprint"
        let result = try loader.lookup(modelFingerprint: fingerprint)

        switch result {
        case .missing(let expectedModelFingerprint, let expectedURL):
            #expect(expectedModelFingerprint == fingerprint)
            #expect(expectedURL == root.appendingPathComponent("\(fingerprint).pt", isDirectory: false))
        default:
            Issue.record("Expected missing TriAttention artifact result")
        }
    }

    @Test
    func explicitLoadReturnsFingerprintMismatch() throws {
        let root = try makeScratchDir(prefix: "triattention-loader-mismatch")
        defer { try? FileManager.default.removeItem(at: root) }

        let actualFingerprint = "actual-fingerprint"
        let expectedFingerprint = "expected-fingerprint"
        let artifactURL = root.appendingPathComponent("\(actualFingerprint).pt", isDirectory: false)
        try copyFixture(to: artifactURL)

        let loader = TriAttentionCalibrationArtifactLoader(rootURL: root)
        let result = try loader.load(url: artifactURL, expectedModelFingerprint: expectedFingerprint)

        switch result {
        case .fingerprintMismatch(let expectedModelFingerprint, let observedFingerprint, let url):
            #expect(expectedModelFingerprint == expectedFingerprint)
            #expect(observedFingerprint == actualFingerprint)
            #expect(url == artifactURL)
        default:
            Issue.record("Expected fingerprint mismatch TriAttention artifact result")
        }
    }

    @Test
    func lookupThrowsForUnreadableArtifact() throws {
        let root = try makeScratchDir(prefix: "triattention-loader-unreadable")
        defer { try? FileManager.default.removeItem(at: root) }

        let fingerprint = "unreadable-fingerprint"
        let artifactURL = root.appendingPathComponent("\(fingerprint).pt", isDirectory: false)
        try writeInvalidArtifact(to: artifactURL)

        let loader = TriAttentionCalibrationArtifactLoader(rootURL: root)

        #expect(throws: TriAttentionCalibrationArtifactError.unreadableArchive(artifactURL)) {
            _ = try loader.lookup(modelFingerprint: fingerprint)
        }
    }

    // MARK: - Actor-integration: artifact lookup + runtime-selection wiring
    //
    // Exercised only for PARO (`z-lab/Qwen3.5-*-PARO`). MLX-native MoE
    // (`unsloth/Qwen3.6-35B-A3B-UD-MLX-*`) is gated out of TriAttention at
    // `LLMActor.isTriAttentionEligibleModel` and therefore never reaches
    // artifact lookup; its dense-gate behavior is covered below by
    // `assertMoEFallsBackToUnsupported`.

    private func assertLoadedArtifact(kind: TriAttentionTestFixtures.Kind) async throws {
        let modelDir = try makeFakeModelDirectory(kind: kind)
        let root = try makeScratchDir(prefix: "triattention-actor-loaded")
        defer {
            try? FileManager.default.removeItem(at: modelDir)
            try? FileManager.default.removeItem(at: root)
        }

        let fingerprint = try ModelFingerprint.computeFingerprint(modelDir: modelDir)
        try copyFixture(to: root.appendingPathComponent("\(fingerprint).pt", isDirectory: false))

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

        #expect(await actor.currentModelFingerprintForTesting == fingerprint)

        let selection = await actor.currentTriAttentionRuntimeSelectionForTesting
        let config = await actor.currentTriAttentionConfigForTesting
        switch selection.calibrationArtifactLookup {
        case .some(.loaded(_, let identity, let relativeResourcePath)):
            #expect(selection.requestedConfiguration.enabled == true)
            #expect(selection.fallbackReason == nil)
            #expect(config.calibrationArtifactIdentity == identity)
            #expect(config.enabled == true)
            #expect(relativeResourcePath == TriAttentionCalibrationArtifactLoader.relativeResourcePath(modelFingerprint: fingerprint))
        default:
            Issue.record("Expected actor to record loaded TriAttention artifact result (kind=\(kind))")
        }

        let generateParameters = await actor.makeGenerateParametersForTesting(
            from: AgentGenerateParameters(
                maxTokens: 256,
                temperature: 0.6,
                topP: 1.0,
                topK: 0,
                minP: 0.0,
                repetitionPenalty: nil,
                repetitionContextSize: 20,
                presencePenalty: nil,
                kvBits: 8,
                kvGroupSize: 64,
                prefillStepSize: 1024,
                triAttention: config
            )
        )
        #expect(generateParameters.triAttention == config)
        #expect(generateParameters.triAttentionCalibrationArtifact?.metadata.headDim == 8)
    }

    private func assertMissingArtifact(kind: TriAttentionTestFixtures.Kind) async throws {
        let modelDir = try makeFakeModelDirectory(kind: kind)
        let root = try makeScratchDir(prefix: "triattention-actor-missing")
        defer {
            try? FileManager.default.removeItem(at: modelDir)
            try? FileManager.default.removeItem(at: root)
        }

        let fingerprint = try ModelFingerprint.computeFingerprint(modelDir: modelDir)
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

        #expect(await actor.currentModelFingerprintForTesting == fingerprint)
        let selection = await actor.currentTriAttentionRuntimeSelectionForTesting
        #expect(selection.effectiveConfiguration.enabled == false)
        #expect(selection.fallbackReason == .missingCalibrationArtifact)
        #expect(await actor.currentTriAttentionConfigForTesting.calibrationArtifactIdentity == nil)

        switch selection.calibrationArtifactLookup {
        case .some(.missing(let expectedModelFingerprint, let expectedURL)):
            #expect(expectedModelFingerprint == fingerprint)
            #expect(expectedURL == root.appendingPathComponent("\(fingerprint).pt", isDirectory: false))
        default:
            Issue.record("Expected actor to record missing TriAttention artifact result (kind=\(kind))")
        }

        let generateParameters = await actor.makeGenerateParametersForTesting(
            from: AgentGenerateParameters(
                maxTokens: 256,
                temperature: 0.6,
                topP: 1.0,
                topK: 0,
                minP: 0.0,
                repetitionPenalty: nil,
                repetitionContextSize: 20,
                presencePenalty: nil,
                kvBits: 8,
                kvGroupSize: 64,
                prefillStepSize: 1024,
                triAttention: TriAttentionConfiguration(enabled: true)
            )
        )
        #expect(generateParameters.triAttention.enabled == false)
        #expect(generateParameters.triAttentionCalibrationArtifact == nil)
    }

    private func assertUnavailableArtifact(kind: TriAttentionTestFixtures.Kind) async throws {
        let modelDir = try makeFakeModelDirectory(kind: kind)
        let root = try makeScratchDir(prefix: "triattention-actor-unavailable")
        defer {
            try? FileManager.default.removeItem(at: modelDir)
            try? FileManager.default.removeItem(at: root)
        }

        let fingerprint = try ModelFingerprint.computeFingerprint(modelDir: modelDir)
        let artifactURL = root.appendingPathComponent("\(fingerprint).pt", isDirectory: false)
        try writeInvalidArtifact(to: artifactURL)

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

        #expect(await actor.currentModelFingerprintForTesting == fingerprint)
        let selection = await actor.currentTriAttentionRuntimeSelectionForTesting
        #expect(selection.effectiveConfiguration.enabled == false)
        #expect(selection.fallbackReason == .unavailableCalibrationArtifact)
        #expect(await actor.currentTriAttentionConfigForTesting.calibrationArtifactIdentity == nil)

        switch selection.calibrationArtifactLookup {
        case .some(.unavailable(let expectedModelFingerprint, let attemptedURL, let errorDescription)):
            #expect(expectedModelFingerprint == fingerprint)
            #expect(attemptedURL == artifactURL)
            #expect(errorDescription.contains("not a readable torch archive"))
        default:
            Issue.record("Expected actor to record unavailable TriAttention artifact result (kind=\(kind))")
        }
    }

    @Test func llmActorLoadRecordsLoadedArtifactForParo() async throws { try await assertLoadedArtifact(kind: .paro) }
    @Test func llmActorLoadRecordsMissingArtifactForParo() async throws { try await assertMissingArtifact(kind: .paro) }
    @Test func llmActorLoadRecordsUnavailableArtifactForParo() async throws { try await assertUnavailableArtifact(kind: .paro) }

    // MoE is gated out of TriAttention regardless of artifact state — the
    // sparse-KV runtime regresses decode by 15–25× on Qwen3.6-35B-A3B at
    // p > budget (see bench logs). These tests pin that gate.

    private func assertMoEFallsBackToUnsupported(artifactPresent: Bool) async throws {
        let modelDir = try makeFakeModelDirectory(kind: .qwen35MoeMlxNative)
        let root = try makeScratchDir(prefix: "triattention-actor-moe-gate")
        defer {
            try? FileManager.default.removeItem(at: modelDir)
            try? FileManager.default.removeItem(at: root)
        }

        let fingerprint = try ModelFingerprint.computeFingerprint(modelDir: modelDir)
        if artifactPresent {
            try copyFixture(to: root.appendingPathComponent("\(fingerprint).pt", isDirectory: false))
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
        let config = await actor.currentTriAttentionConfigForTesting
        #expect(selection.effectiveConfiguration.enabled == false)
        #expect(selection.fallbackReason == .unsupportedModel)
        #expect(config.enabled == false)
        #expect(config.calibrationArtifactIdentity == nil)
        switch selection.calibrationArtifactLookup {
        case nil:
            break
        default:
            Issue.record("MoE eligibility gate should skip calibration lookup entirely")
        }
    }

    @Test func llmActorFallsBackToUnsupportedForMoEWithoutArtifact() async throws {
        try await assertMoEFallsBackToUnsupported(artifactPresent: false)
    }
    @Test func llmActorFallsBackToUnsupportedForMoEWithArtifact() async throws {
        try await assertMoEFallsBackToUnsupported(artifactPresent: true)
    }

    @Test
    func shipped4BArtifactLoadsSuccessfully() throws {
        let shippedURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent(
                "TriAttention/v1/0b0d94803c53b186006c100dd12a26ad1f955399ce5b52100e5f607a5fcb00fe.pt",
                isDirectory: false
            )
        let artifact = try TriAttentionCalibrationArtifact.load(contentsOf: shippedURL)
        #expect(artifact.statsByHead.count == 128)
    }
}
