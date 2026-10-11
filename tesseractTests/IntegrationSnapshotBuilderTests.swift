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
            isVisionCapable: { _ in false }
        )

        #expect(snapshot.models.map(\.id) == ["downloaded-a"])
        #expect(snapshot.port == 8321)
    }

    /// Each model carries the catalog's own vision verdict
    /// (`ModelDownloadManager.isVisionCapable`, probed and memoized there;
    /// `ModelCatalogTests` pin the probe).
    @Test func eachModelCarriesTheCatalogsVisionVerdict() throws {
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
            isVisionCapable: { $0 == "vision-model" }
        )

        #expect(snapshot.models.first { $0.id == "vision-model" }?.visionCapable == true)
        #expect(snapshot.models.first { $0.id == "text-model" }?.visionCapable == false)
    }

    @Test func contextLengthMirrorsTheModelsEndpoint() throws {
        let snapshot = IntegrationSnapshotBuilder.build(
            definitions: [agentDefinition(id: "a")],
            statuses: ["a": .downloaded(sizeOnDisk: 1)],
            selectedAgentModelID: "a",
            port: 8321,
            isVisionCapable: { _ in false }
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
            isVisionCapable: { _ in false }
        )

        #expect(snapshot.defaultModelID == "b")
    }

    @Test func defaultFallsBackToFirstDownloadedWhenSelectionIsNotOnDisk() throws {
        let snapshot = IntegrationSnapshotBuilder.build(
            definitions: [agentDefinition(id: "a"), agentDefinition(id: "b")],
            statuses: ["b": .downloaded(sizeOnDisk: 1)],
            selectedAgentModelID: "a",
            port: 8321,
            isVisionCapable: { _ in false }
        )

        #expect(snapshot.defaultModelID == "b")
    }

    @Test func defaultIsNilWhenNothingIsDownloaded() throws {
        let snapshot = IntegrationSnapshotBuilder.build(
            definitions: [agentDefinition(id: "a")],
            statuses: [:],
            selectedAgentModelID: "a",
            port: 8321,
            isVisionCapable: { _ in false }
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
}
