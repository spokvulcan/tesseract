import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

struct DFlash2SupportTests {

    // MARK: - Server keyed-path engagement policy

    /// `shouldEngage` with the engaging happy path as defaults, so each test
    /// names only the axis it varies.
    private func engages(
        hasDrafter: Bool = true,
        textOnlyIdentityKeySpace: Bool = true,
        kvBits: Int? = nil
    ) -> Bool {
        DFlash2Support.shouldEngage(
            hasDrafter: hasDrafter,
            textOnlyIdentityKeySpace: textOnlyIdentityKeySpace,
            kvBits: kvBits
        )
    }

    @Test func engagesOnTextOnlyIdentityRequests() {
        // No leaf-mode or cold-path axis anymore: the arm rides the keyed
        // path's own restore + checkpoint-capturing prefill, so thinking and
        // tool traffic (and warm restores) engage too.
        #expect(engages())
    }

    @Test func refusesWithoutDrafter() {
        #expect(!engages(hasDrafter: false))
    }

    @Test func refusesImageBearingRequests() {
        #expect(!engages(textOnlyIdentityKeySpace: false))
    }

    @Test func refusesQuantizedKVPartitions() {
        // Speculation rewinds verify rows in place through the plain
        // `KVCacheSimple` machinery; quantized-KV partitions keep the
        // ordinary decode path.
        #expect(!engages(kvBits: 8))
    }

    // MARK: - Agent raw-arm engagement policy

    @Test func rawArmEngagesOnTextOnlyInput() {
        let input = LMInput(text: .init(tokens: MLXArray([Int32(1), 2, 3])))
        #expect(DFlash2Support.shouldEngageRawArm(hasDrafter: true, input: input))
    }

    @Test func rawArmRefusesWithoutDrafter() {
        let input = LMInput(text: .init(tokens: MLXArray([Int32(1), 2, 3])))
        #expect(!DFlash2Support.shouldEngageRawArm(hasDrafter: false, input: input))
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
