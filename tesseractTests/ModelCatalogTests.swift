//
//  ModelCatalogTests.swift
//  tesseractTests
//
//  Pins the **Model Catalog** pure projection (CONTEXT.md → Model catalog): the
//  join of the static `ModelDefinition` table with download `statuses` that every
//  caller used to re-derive inline, plus the single vision-detection rule. All
//  hermetic — canned definitions/statuses and fixture directories, no manager,
//  no disk beyond the config.json fixtures (the same pattern as
//  `IntegrationSnapshotBuilderTests`).
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct ModelCatalogTests {

    // MARK: - downloaded(in:)

    @Test func downloadedFiltersByCategoryAndDownloadedState() throws {
        let definitions = [
            agentDef(id: "agent-down"),
            agentDef(id: "agent-not"),
            speechDef(id: "speech-down"),
        ]
        let statuses: [String: ModelStatus] = [
            "agent-down": .downloaded(sizeOnDisk: 1),
            "speech-down": .downloaded(sizeOnDisk: 1),
            // "agent-not" deliberately absent
        ]

        let agents = ModelCatalog.downloaded(
            in: .agent, definitions: definitions, statuses: statuses)
        #expect(agents.map(\.id) == ["agent-down"])

        let speech = ModelCatalog.downloaded(
            in: .speechToText, definitions: definitions, statuses: statuses)
        #expect(speech.map(\.id) == ["speech-down"])
    }

    @Test func downloadedPreservesCatalogueOrder() throws {
        let definitions = [agentDef(id: "a"), agentDef(id: "b"), agentDef(id: "c")]
        let statuses: [String: ModelStatus] = [
            "a": .downloaded(sizeOnDisk: 1),
            "b": .downloaded(sizeOnDisk: 1),
            "c": .downloaded(sizeOnDisk: 1),
        ]

        let result = ModelCatalog.downloaded(
            in: .agent, definitions: definitions, statuses: statuses)
        #expect(result.map(\.id) == ["a", "b", "c"])
    }

    @Test func downloadedExcludesInProgressAndErrorStates() throws {
        let definitions = [
            agentDef(id: "downloading"),
            agentDef(id: "verifying"),
            agentDef(id: "errored"),
            agentDef(id: "done"),
        ]
        let statuses: [String: ModelStatus] = [
            "downloading": .downloading(progress: 0.5),
            "verifying": .verifying(progress: 0.9),
            "errored": .error("nope"),
            "done": .downloaded(sizeOnDisk: 1),
        ]

        let result = ModelCatalog.downloaded(
            in: .agent, definitions: definitions, statuses: statuses)
        #expect(result.map(\.id) == ["done"])
    }

    // MARK: - isDownloaded

    @Test func isDownloadedOnlyForTheDownloadedCase() throws {
        let statuses: [String: ModelStatus] = [
            "down": .downloaded(sizeOnDisk: 1),
            "dl": .downloading(progress: 0.3),
            "ver": .verifying(progress: 0.3),
            "err": .error("x"),
            "none": .notDownloaded,
        ]
        #expect(ModelCatalog.isDownloaded("down", statuses: statuses) == true)
        #expect(ModelCatalog.isDownloaded("dl", statuses: statuses) == false)
        #expect(ModelCatalog.isDownloaded("ver", statuses: statuses) == false)
        #expect(ModelCatalog.isDownloaded("err", statuses: statuses) == false)
        #expect(ModelCatalog.isDownloaded("none", statuses: statuses) == false)
        #expect(ModelCatalog.isDownloaded("absent", statuses: statuses) == false)
    }

    // MARK: - isVisionCapable(directory:)

    @Test func visionRuleReadsTheModelConfig() throws {
        let visionDir = try makeModelDir(
            config: #"{ "model_type": "qwen3_5", "vision_config": { "spatial_merge_size": 2 } }"#)
        let textDir = try makeModelDir(config: #"{ "model_type": "qwen3_5" }"#)
        defer {
            try? FileManager.default.removeItem(at: visionDir)
            try? FileManager.default.removeItem(at: textDir)
        }

        #expect(ModelCatalog.isVisionCapable(directory: visionDir) == true)
        #expect(ModelCatalog.isVisionCapable(directory: textDir) == false)
    }

    // MARK: - Fixtures

    private func agentDef(id: String) -> ModelDefinition {
        ModelDefinition(
            id: id, displayName: "Display \(id)", description: "",
            category: .agent,
            source: .huggingFace(repo: "fixture/\(id)", requiredExtension: "safetensors"),
            sizeDescription: "", dependencies: [])
    }

    private func speechDef(id: String) -> ModelDefinition {
        ModelDefinition(
            id: id, displayName: "Display \(id)", description: "",
            category: .speechToText,
            source: .huggingFace(repo: "fixture/\(id)", requiredExtension: "mlmodelc"),
            sizeDescription: "", dependencies: [])
    }

    private func makeModelDir(config: String) throws -> URL {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("modelcatalog-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try config.write(
            to: dir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        return dir
    }
}
