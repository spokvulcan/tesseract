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
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(prefix)-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func copyFixture(to destinationURL: URL) throws {
        let data = try Data(contentsOf: fixtureURL())
        try data.write(to: destinationURL)
    }

    private func writeInvalidArtifact(to destinationURL: URL) throws {
        try Data("not a torch zip archive".utf8).write(to: destinationURL)
    }

    private func makeFakeModelDirectory() throws -> URL {
        try makeScratchDir(prefix: "triattention-fake-model")
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

    @Test
    func llmActorLoadRecordsLoadedCalibrationArtifactAndIdentity() async throws {
        let modelDir = try makeFakeModelDirectory()
        let root = try makeScratchDir(prefix: "triattention-actor-loaded")
        defer {
            try? FileManager.default.removeItem(at: modelDir)
            try? FileManager.default.removeItem(at: root)
        }

        let fingerprint = try ModelFingerprint.computeFingerprint(modelDir: modelDir)
        let artifactURL = root.appendingPathComponent("\(fingerprint).pt", isDirectory: false)
        try copyFixture(to: artifactURL)

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
            // Expected: the fake directory is fingerprintable but not loadable.
        }

        #expect(await actor.currentModelFingerprintForTesting == fingerprint)

        let config = await actor.currentTriAttentionConfigForTesting
        let lookup = await actor.currentTriAttentionCalibrationArtifactLookupForTesting
        switch lookup {
        case .loaded(_, let identity, let relativeResourcePath):
            #expect(config.calibrationArtifactIdentity == identity)
            #expect(config.enabled == true)
            #expect(relativeResourcePath == TriAttentionCalibrationArtifactLoader.relativeResourcePath(modelFingerprint: fingerprint))
        default:
            Issue.record("Expected actor to record loaded TriAttention artifact result")
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

    @Test
    func llmActorLoadRecordsMissingCalibrationArtifactWithoutIdentity() async throws {
        let modelDir = try makeFakeModelDirectory()
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
            // Expected: the fake directory is fingerprintable but not loadable.
        }

        #expect(await actor.currentModelFingerprintForTesting == fingerprint)
        #expect(await actor.currentTriAttentionConfigForTesting.calibrationArtifactIdentity == nil)

        let lookup = await actor.currentTriAttentionCalibrationArtifactLookupForTesting
        switch lookup {
        case .missing(let expectedModelFingerprint, let expectedURL):
            #expect(expectedModelFingerprint == fingerprint)
            #expect(expectedURL == root.appendingPathComponent("\(fingerprint).pt", isDirectory: false))
        default:
            Issue.record("Expected actor to record missing TriAttention artifact result")
        }

        let triAttentionConfig = await actor.currentTriAttentionConfigForTesting
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
                triAttention: triAttentionConfig
            )
        )
        #expect(generateParameters.triAttentionCalibrationArtifact == nil)
    }

    @Test
    func llmActorLoadRecordsUnavailableCalibrationArtifactWithoutIdentity() async throws {
        let modelDir = try makeFakeModelDirectory()
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
            // Expected: the fake directory is fingerprintable but not loadable.
        }

        #expect(await actor.currentModelFingerprintForTesting == fingerprint)
        #expect(await actor.currentTriAttentionConfigForTesting.calibrationArtifactIdentity == nil)

        let lookup = await actor.currentTriAttentionCalibrationArtifactLookupForTesting
        switch lookup {
        case .unavailable(let expectedModelFingerprint, let attemptedURL, let errorDescription):
            #expect(expectedModelFingerprint == fingerprint)
            #expect(attemptedURL == artifactURL)
            #expect(errorDescription.contains("not a readable torch archive"))
        default:
            Issue.record("Expected actor to record unavailable TriAttention artifact result")
        }
    }
}
