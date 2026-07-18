import Foundation
import Testing

@testable import Tesseract_Agent

/// Behaviour of the **Integration snapshot** builder: which models it admits
/// (downloaded agent models only, mirroring `/v1/models`), how vision
/// capability is read from the model's own on-disk config, and how the
/// default model is chosen. Fixture directories exercise the real
/// file-reading path, like `ModelIdentityTests`.
struct IntegrationSnapshotBuilderTests {

    @Test func onlyDownloadedAgentModelsAreIncluded() throws {
        let definitions = [
            agentDefinition(id: "downloaded-a"),
            agentDefinition(id: "not-downloaded"),
            ttsDefinition(id: "downloaded-tts"),
        ]
        let statuses: [String: ModelStatus] = [
            "downloaded-a": .downloaded(sizeOnDisk: 1),
            "downloaded-tts": .downloaded(sizeOnDisk: 1),
        ]

        let snapshot = IntegrationSnapshotBuilder.build(
            definitions: definitions,
            statuses: statuses,
            selectedAgentModelID: "downloaded-a",
            port: 8321,
            modelDirectory: { _ in nil }
        )

        #expect(snapshot.models.map(\.id) == ["downloaded-a"])
        #expect(snapshot.port == 8321)
    }

    @Test func visionCapabilityIsReadFromTheModelDirectory() throws {
        let visionDir = try makeModelDir(
            config: #"{ "model_type": "qwen3_5", "vision_config": { "spatial_merge_size": 2 } }"#
        )
        let textDir = try makeModelDir(config: #"{ "model_type": "qwen3_5" }"#)
        defer {
            try? FileManager.default.removeItem(at: visionDir)
            try? FileManager.default.removeItem(at: textDir)
        }
        let directories = ["vision-model": visionDir, "text-model": textDir]

        let snapshot = IntegrationSnapshotBuilder.build(
            definitions: [
                agentDefinition(id: "vision-model"),
                agentDefinition(id: "text-model"),
            ],
            statuses: [
                "vision-model": .downloaded(sizeOnDisk: 1),
                "text-model": .downloaded(sizeOnDisk: 1),
            ],
            selectedAgentModelID: "vision-model",
            port: 8321,
            modelDirectory: { directories[$0] }
        )

        #expect(snapshot.models.first { $0.id == "vision-model" }?.visionCapable == true)
        #expect(snapshot.models.first { $0.id == "text-model" }?.visionCapable == false)
    }

    @Test func audioCapabilityIsReadFromTheModelDirectory() throws {
        let audioDir = try makeModelDir(
            config: #"""
                { "model_type": "gemma4_unified",
                  "vision_config": { "model_type": "gemma4_unified_vision" },
                  "audio_config": { "model_type": "gemma4_unified_audio" } }
                """#
        )
        let textDir = try makeModelDir(config: #"{ "model_type": "qwen3_5" }"#)
        defer {
            try? FileManager.default.removeItem(at: audioDir)
            try? FileManager.default.removeItem(at: textDir)
        }
        let directories = ["audio-model": audioDir, "text-model": textDir]

        let snapshot = IntegrationSnapshotBuilder.build(
            definitions: [
                agentDefinition(id: "audio-model"),
                agentDefinition(id: "text-model"),
            ],
            statuses: [
                "audio-model": .downloaded(sizeOnDisk: 1),
                "text-model": .downloaded(sizeOnDisk: 1),
            ],
            selectedAgentModelID: "audio-model",
            port: 8321,
            modelDirectory: { directories[$0] }
        )

        let audioModel = snapshot.models.first { $0.id == "audio-model" }
        #expect(audioModel?.audioCapable == true)
        #expect(audioModel?.visionCapable == true)
        #expect(snapshot.models.first { $0.id == "text-model" }?.audioCapable == false)
    }

    @Test func missingDirectoryMeansNoVision() throws {
        let snapshot = IntegrationSnapshotBuilder.build(
            definitions: [agentDefinition(id: "a")],
            statuses: ["a": .downloaded(sizeOnDisk: 1)],
            selectedAgentModelID: "a",
            port: 8321,
            modelDirectory: { _ in nil }
        )

        #expect(snapshot.models.first?.visionCapable == false)
    }

    @Test func contextLengthMirrorsTheModelsEndpoint() throws {
        let snapshot = IntegrationSnapshotBuilder.build(
            definitions: [agentDefinition(id: "a")],
            statuses: ["a": .downloaded(sizeOnDisk: 1)],
            selectedAgentModelID: "a",
            port: 8321,
            modelDirectory: { _ in nil }
        )

        #expect(snapshot.models.first?.contextLength == 262_144)
    }

    // MARK: - Default model

    @Test func defaultIsTheSelectedModelWhenDownloaded() throws {
        let snapshot = IntegrationSnapshotBuilder.build(
            definitions: [agentDefinition(id: "a"), agentDefinition(id: "b")],
            statuses: [
                "a": .downloaded(sizeOnDisk: 1),
                "b": .downloaded(sizeOnDisk: 1),
            ],
            selectedAgentModelID: "b",
            port: 8321,
            modelDirectory: { _ in nil }
        )

        #expect(snapshot.defaultModelID == "b")
    }

    @Test func defaultFallsBackToFirstDownloadedWhenSelectionIsNotOnDisk() throws {
        let snapshot = IntegrationSnapshotBuilder.build(
            definitions: [agentDefinition(id: "a"), agentDefinition(id: "b")],
            statuses: ["b": .downloaded(sizeOnDisk: 1)],
            selectedAgentModelID: "a",
            port: 8321,
            modelDirectory: { _ in nil }
        )

        #expect(snapshot.defaultModelID == "b")
    }

    @Test func defaultIsNilWhenNothingIsDownloaded() throws {
        let snapshot = IntegrationSnapshotBuilder.build(
            definitions: [agentDefinition(id: "a")],
            statuses: [:],
            selectedAgentModelID: "a",
            port: 8321,
            modelDirectory: { _ in nil }
        )

        #expect(snapshot.models.isEmpty)
        #expect(snapshot.defaultModelID == nil)
    }

    // MARK: - Fixtures

    private func agentDefinition(id: String) -> ModelDefinition {
        ModelDefinition(
            id: id,
            displayName: "Display \(id)",
            description: "",
            category: .agent,
            source: .huggingFace(repo: "fixture/\(id)", requiredExtension: "safetensors"),
            sizeDescription: "",
            dependencies: []
        )
    }

    private func ttsDefinition(id: String) -> ModelDefinition {
        ModelDefinition(
            id: id,
            displayName: "Display \(id)",
            description: "",
            category: .textToSpeech,
            source: .huggingFace(repo: "fixture/\(id)", requiredExtension: "safetensors"),
            sizeDescription: "",
            dependencies: []
        )
    }

    private func makeModelDir(config: String) throws -> URL {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("integrationsnapshot-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try config.write(
            to: dir.appendingPathComponent("config.json"),
            atomically: true, encoding: .utf8
        )
        return dir
    }
}
