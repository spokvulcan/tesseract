import Foundation
import MLXLLM
import MLXLMCommon
import MLXVLM
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

    // MARK: - Target pairing

    /// A four-layer Qwen3.5 checkpoint with a vision tower.
    private static let tinyVisionCheckpoint = """
        {
            "model_type": "qwen3_5", "image_token_id": 500, "video_token_id": 501,
            "vision_start_token_id": 502, "vision_end_token_id": 503, "vocab_size": 512,
            "text_config": {
                "model_type": "qwen3_5", "hidden_size": 64, "num_hidden_layers": 4,
                "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2,
                "head_dim": 32, "vocab_size": 512, "full_attention_interval": 2,
                "linear_num_value_heads": 4, "linear_num_key_heads": 2,
                "linear_key_head_dim": 32, "linear_value_head_dim": 32,
                "linear_conv_kernel_dim": 4
            },
            "vision_config": {
                "model_type": "qwen3_vl", "depth": 1, "hidden_size": 32, "intermediate_size": 64,
                "out_hidden_size": 64, "num_heads": 2, "patch_size": 16,
                "spatial_merge_size": 2, "temporal_patch_size": 2, "num_position_embeddings": 64
            }
        }
        """

    /// The vision class runs the text class's engine (ADR-0089), so a
    /// vision-mode load pairs with the draft at the same depth.
    @Test func bothQwen35ClassesPairAtTheirDepth() throws {
        let data = Data(Self.tinyVisionCheckpoint.utf8)
        let vision = MLXVLM.Qwen35(
            try JSONDecoder().decode(MLXVLM.Qwen35Configuration.self, from: data))
        let text = MLXLLM.Qwen35Model(
            try JSONDecoder().decode(MLXLLM.Qwen35Configuration.self, from: data))
        for model in [vision, text] as [any LanguageModel] {
            #expect(DFlash2Support.pairsWithTarget(model), "\(type(of: model))")
            #expect(DFlash2Support.targetLayerCount(model) == 4, "\(type(of: model))")
        }
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
    /// — pull the draft. Either class pairs with it, the vision one too
    /// (ADR-0089), so a vision-mode load speculates as a text one does.
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
