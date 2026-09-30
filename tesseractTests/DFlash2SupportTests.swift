import Foundation
import Testing

@testable import Tesseract_Agent

/// The DFlash2 draft's own facts: checkpoint refusal, target geometry,
/// folder detection and catalog wiring. When a request engages the draft is
/// the Speculation Plan's table (`SpeculationPlanTests`).
struct DFlash2SupportTests {

    // MARK: - Checkpoint-level refusal

    /// A Rotated Ternary Checkpoint runs the pairable class at the pairable
    /// depth, yet the draft (distilled for the full-precision target) decoded
    /// slower on it, so the checkpoint itself refuses the pairing.
    @Test func rotatedTernaryCheckpointRefusesTheDraft() {
        let rotated = ModelIdentity(
            configJSON: [
                "model_type": "prism_hadamard_qwen35",
                "base_model_type": "qwen3_5",
                "modules": [["path": "model.embed_tokens", "block": 1024, "embedding": true]],
            ],
            chatTemplate: nil)
        let plain = ModelIdentity(configJSON: ["model_type": "qwen3_5"], chatTemplate: nil)

        #expect(DFlash2Support.checkpointRefusesDraft(rotated))
        #expect(!DFlash2Support.checkpointRefusesDraft(plain))
    }

    // MARK: - Target geometry

    /// The release draft's shape: distilled against a 64-layer target,
    /// reading layers 5/19/33/47/61.
    private func geometryMatches(targetLayerCount: Int) -> Bool {
        DFlash2Support.geometryMatches(
            targetLayerCount: targetLayerCount,
            draftNumTargetLayers: 64,
            draftTargetLayerIds: [5, 19, 33, 47, 61])
    }

    @Test func geometryMatchesTheTargetItWasDistilledFor() {
        #expect(geometryMatches(targetLayerCount: 64))
    }

    @Test func geometryRefusesOtherDepthsOfThePairableClass() {
        // The class check admits every Qwen3.5 dense size; the draft's
        // captured layers only mean something at the depth it was trained on.
        #expect(!geometryMatches(targetLayerCount: 40))  // 9B-class stack
        #expect(!geometryMatches(targetLayerCount: 80))
    }

    @Test func geometryRefusesCapturedLayersPastTheTargetEnd() {
        #expect(
            !DFlash2Support.geometryMatches(
                targetLayerCount: 64, draftNumTargetLayers: 64,
                draftTargetLayerIds: [5, 19, 33, 47, 64]))
    }

    // MARK: - Draft folder detection

    private func makeStorageRoot() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("dflash2-detect-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    @Test func draftDirectoryRequiresConfigAndWeights() throws {
        let root = try makeStorageRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let draftDir = root.appendingPathComponent(DFlash2Support.draftCacheSubdirectory)

        // Missing entirely.
        #expect(
            DFlash2Support.draftDirectory(storageRoot: root) == nil,
            "empty storage root must yield nil, got \(String(describing: DFlash2Support.draftDirectory(storageRoot: root)))"
        )

        // Config only — no weights.
        try FileManager.default.createDirectory(at: draftDir, withIntermediateDirectories: true)
        try "{}".write(
            to: draftDir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        #expect(
            DFlash2Support.draftDirectory(storageRoot: root) == nil,
            "config without safetensors must yield nil, got \(String(describing: DFlash2Support.draftDirectory(storageRoot: root)))"
        )

        // Weights too — detected. (Compare standardized paths: URL equality
        // is literal about trailing slashes.)
        try Data([0, 1]).write(to: draftDir.appendingPathComponent("model.safetensors"))
        let detected = DFlash2Support.draftDirectory(storageRoot: root)
        #expect(
            detected?.standardizedFileURL.path == draftDir.standardizedFileURL.path,
            "config + safetensors must detect: got \(String(describing: detected)) vs \(draftDir)"
        )
    }

    // MARK: - Model definition wiring

    /// Both Qwen3.8-27B targets — the uniform quant and the PARO Checkpoint
    /// — pull the draft, and both carry the Text-Only Override: the draft
    /// pairs only with the MLXLLM text classes (`pairsWithTarget`), so a
    /// vision-mode load of either would never speculate (map #457 lifts it).
    @MainActor
    @Test func draftIsDownloadableDependencyOfBothQwen38Targets() {
        let draft = ModelDefinition.withID(DFlash2Support.draftModelID)
        #expect(draft != nil)
        #expect(draft?.category == .draft)
        #expect(draft?.repoID == "incoai/Qwen3.8-27B-DFlash2")
        for id in ["qwen3.8-27b", "qwen3.8-27b-paro"] {
            let target = ModelDefinition.withID(id)
            #expect(target != nil, "missing \(id)")
            #expect(target?.dependencies.contains(DFlash2Support.draftModelID) == true)
            #expect(target?.textOnlyOverride == true, "\(id) must load the text class")
        }
    }

    @MainActor
    @Test func draftCacheSubdirectoryMatchesDefinitionRule() {
        // `DFlash2Support.draftCacheSubdirectory` duplicates the definition's
        // `cacheSubdirectory` because the latter is MainActor-isolated; this
        // pins them together.
        let draft = ModelDefinition.withID(DFlash2Support.draftModelID)
        #expect(draft?.cacheSubdirectory == DFlash2Support.draftCacheSubdirectory)
    }
}
