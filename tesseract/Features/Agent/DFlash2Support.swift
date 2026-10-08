import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// App-side surface for the DFlash2 block-parallel speculative drafter
/// (`incoai/Qwen3.8-27B-DFlash2`) paired with Qwen3.8-27B.
///
/// Unlike the MTP head (which ships inside the target checkpoint), the DFlash2
/// draft is a *separate* model: 5 bidirectional sliding-window layers plus a
/// bigram selector, trained against the target's layer-5/19/33/47/61 hidden
/// states. It downloads as a dependency of the `qwen3.8-27b` model entry and
/// lives in its own folder under the models directory.
///
/// The draft's facts, mirroring `MTPDrafterSupport`: detection (is the draft
/// folder on disk and complete?), pairing and geometry (which loaded targets
/// it can speculate for), and loading (instantiate the draft and 4-bit
/// quantize it; the target's embedding and head are borrowed per proposal by
/// the vendor iterator). **Speculation** loads it through these and decides
/// when a request engages it (ADR-0079); the DFlash2 bench uses them too.
nonisolated enum DFlash2Support {

    /// The `ModelDefinition` id of the DFlash2 draft dependency.
    static let draftModelID = "qwen3.8-27b-dflash2-draft"

    /// The draft's folder name under the models directory
    /// (`incoai/Qwen3.8-27B-DFlash2` under the `/` → `_` `cacheSubdirectory`
    /// rule). Duplicated here because `ModelDefinition` is MainActor-isolated
    /// by the project's default isolation and this enum is nonisolated; the
    /// test target pins the two together.
    static let draftCacheSubdirectory = "incoai_Qwen3.8-27B-DFlash2"

    /// Tokens per verify pass (1 anchor + 7 drafts): the width the checkpoint
    /// was distilled at, and the fastest one on the bench (ADR-0058).
    static let blockSize = 8

    // MARK: - Detection

    /// The draft's local folder, when the download completed (config.json +
    /// at least one safetensors — mirrors `ModelDownloadManager`'s
    /// completeness rule of "every remote file present"). The storage root is
    /// caller-supplied because `ModelDownloadManager.modelStorageURL` is
    /// MainActor-isolated and this enum is not.
    static func draftDirectory(
        storageRoot: URL
    ) -> URL? {
        let directory = storageRoot.appendingPathComponent(draftCacheSubdirectory)
        let configPresent = FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("config.json").path)
        let hasWeights =
            (try? FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: nil))?
            .contains { $0.pathExtension == "safetensors" } ?? false
        return configPresent && hasWeights ? directory : nil
    }

    /// Which loaded targets the DFlash2 draft can speculate for: the vendor
    /// `DFlash2TargetModel` conformers, the MLXLLM Qwen3.5 text classes and
    /// the MLXVLM Qwen3.5 vision class that runs the same engine
    /// (ADR-0089), so a vision-mode load speculates as a text one does.
    static func pairsWithTarget(_ model: any LanguageModel) -> Bool {
        model is any DFlash2TargetModel
    }

    /// Whether the target's checkpoint declines the draft regardless of class
    /// and geometry. A **Rotated Ternary Checkpoint** (ADR-0067) runs the
    /// pairable class at the pairable depth, but the draft was distilled
    /// against the full-precision target's distribution: on Bonsai 2 27B it
    /// accepted 22% of its proposals and decoded a third slower than plain
    /// autoregression (2026-09-18, `docs/model-parameters.md`), so the pairing
    /// is refused at load rather than measured per request.
    static func checkpointRefusesDraft(_ identity: ModelIdentity) -> Bool {
        identity.isRotatedTernaryCheckpoint
    }

    /// Depth of the loaded target's layer stack when it pairs
    /// (``pairsWithTarget(_:)``), `nil` otherwise. Every pairable class
    /// reports its depth through `KVCacheDimensionProvider`.
    static func targetLayerCount(_ model: any LanguageModel) -> Int? {
        guard pairsWithTarget(model) else { return nil }
        return (model as? KVCacheDimensionProvider)?.kvHeads.count
    }

    /// The draft checkpoint's configuration — the geometry facts
    /// (`num_target_layers`, `target_layer_ids`) without the weights.
    static func draftConfiguration(directory: URL) throws -> DFlash2Configuration {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        return try JSONDecoder.json5().decode(DFlash2Configuration.self, from: data)
    }

    /// Whether a draft was distilled for a target of this depth. The class
    /// check admits every Qwen3.5 dense size and every PARO checkpoint of
    /// the family, but the draft reads hidden states at fixed
    /// `target_layer_ids` of a `num_target_layers`-deep stack: bound to a
    /// shallower target it would index past the end, to a deeper one it
    /// would read layers it never saw. Both are a silent-garbage or trap
    /// outcome, so the pairing is refused at load instead.
    static func geometryMatches(
        targetLayerCount: Int,
        draftNumTargetLayers: Int,
        draftTargetLayerIds: [Int]
    ) -> Bool {
        targetLayerCount == draftNumTargetLayers
            && draftTargetLayerIds.allSatisfy { $0 < targetLayerCount }
    }

    // MARK: - Loading

    /// Load + 4-bit quantize the draft (reference: `nn.quantize(draft,
    /// group_size: 64, bits: 4)`). The draft holds no target state, so the
    /// loaded value can be boxed and shared across sessions; the iterator
    /// checks the pairing (`DFlash2SpeculationError`) at construction.
    ///
    /// The checkpoint is bfloat16 (3.85 GB on the 27B pairing) and only its
    /// packed form stays resident, so the file's arrays are applied unread
    /// and each leaf is realized on its own: a leaf's rows are read, packed
    /// and dropped before the next leaf's are read. Reading the whole file
    /// first (`loadWeights`) held it beside the packed copy at the load's
    /// peak.
    static func loadDrafter(
        directory: URL
    ) throws -> any DFlash2DrafterModel {
        let config = try draftConfiguration(directory: directory)
        let draft = DFlash2DraftModel(config)
        try applyUnreadWeights(in: directory, to: draft)
        quantize(model: draft, groupSize: 64, bits: 4)
        for (_, leaf) in draft.leafModules().flattened() {
            eval(leaf)
        }
        eval(draft)
        return draft
    }

    /// The checkpoint's arrays applied to `draft` as MLX's lazy file loads,
    /// read only when a parameter is first evaluated.
    private static func applyUnreadWeights(in directory: URL, to draft: DFlash2DraftModel) throws {
        let files = try FileManager.default
            .contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        var weights = [String: MLXArray]()
        for url in files {
            weights.merge(try loadArrays(url: url)) { _, new in new }
        }
        weights = draft.sanitize(weights: weights)
        try draft.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
    }
}
