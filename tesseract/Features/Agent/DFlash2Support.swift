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
/// Three jobs, mirroring `MTPDrafterSupport`:
/// 1. **Detection** — is the draft folder on disk and complete?
/// 2. **Loading** — instantiate the draft and 4-bit quantize it (reference:
///    `nn.quantize(draft, group_size: 64, bits: 4)`); the target's
///    embedding/head are borrowed per proposal by the vendor iterator.
/// 3. **Engagement policy** — pure, unit-tested predicates for the server
///    cold path and the agent's raw arm.
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
    /// `DFlash2TargetModel` conformers (the MLXLLM Qwen3.5 text classes).
    /// The app force-loads the MLXLLM text target even from VLM-shaped
    /// checkpoints (see `MTPDrafterSupport`), so the MLXVLM container never
    /// reaches here in practice.
    static func pairsWithTarget(_ model: any LanguageModel) -> Bool {
        model is any DFlash2TargetModel
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
    static func loadDrafter(
        directory: URL
    ) throws -> any DFlash2DrafterModel {
        let config = try draftConfiguration(directory: directory)
        let draft = DFlash2DraftModel(config)
        try loadWeights(modelDirectory: directory, model: draft)
        quantize(model: draft, groupSize: 64, bits: 4)
        eval(draft)
        return draft
    }

    // MARK: - Engagement policy

    /// The server keyed-path engagement decision (pure, unit-tested).
    /// Speculate when a drafter is loaded, the request is text-only on the
    /// identity key space, and the partition's KV is unquantized.
    ///
    /// Unlike MTP there is no cold-path or leaf-mode gate: the app driver
    /// runs the same restore + checkpoint-capturing chunked prefill as the
    /// ordinary path up to the deepest planned capture, then hands the warm
    /// cache to the iterator (`prefilledPrefixTokens`), whose capture
    /// prefill covers only the tail. The transient boundary snapshots — what
    /// a thinking template's canonical leaf and a tool turn's direct-tool
    /// leaf are synthesized from (ADR-0056 amendment) — are therefore
    /// captured exactly as on the ordinary path, and warm restores speculate
    /// too: the cache stays first, speculation rides on top.
    ///
    /// The `kvBits == nil` gate exists because speculation rewinds verify
    /// rows in place: the round machinery trims and (on pipelined rounds)
    /// commits plain `KVCacheSimple` rows, so a quantized-KV partition keeps
    /// the ordinary decode path.
    ///
    /// No greedy gate (the iterator rejection-samples against the selector's
    /// candidate distribution, so sampling presets speculate identically)
    /// and no scratch gate (the prefill is chunked and the draft's context
    /// window is fixed at 2047 rows — prompt length does not change the
    /// memory shape), as before.
    static func shouldEngage(
        hasDrafter: Bool,
        textOnlyIdentityKeySpace: Bool,
        kvBits: Int?
    ) -> Bool {
        hasDrafter && textOnlyIdentityKeySpace && kvBits == nil
    }

    /// The **Raw Generation Start** engagement decision: text-only input with
    /// a loaded drafter (the drafter is only loaded when it pairs with the
    /// target — `pairsWithTarget` at load is the family gate).
    static func shouldEngageRawArm(
        hasDrafter: Bool,
        input: LMInput
    ) -> Bool {
        hasDrafter && input.image == nil && input.video == nil && input.audio == nil
    }

    /// The **Raw Generation Start**'s iterator parameters once the arm
    /// engages: `kvBits` cleared, because speculation rewinds verify rows in
    /// place and needs trimmable caches. The one home for the override the
    /// server's keyed gate spells as a refusal (`shouldEngage`) — reconciling
    /// the two is the Speculation Plan's job.
    static func rawArmParameters(_ parameters: GenerateParameters) -> GenerateParameters {
        var specParams = parameters
        specParams.kvBits = nil
        return specParams
    }

    /// Build the DFlash2 iterator with the app's penalty discipline (ADR-0053):
    /// penalties are stripped from the iterator parameters and re-attached as
    /// the app logit processor through `GenerationComponents`, so the vendor's
    /// parameter-built penalty processor never doubles them. `kvBits` is
    /// cleared by the caller (speculation needs trimmable caches).
    ///
    /// `prefilledPrefixTokens` is the number of leading `input` positions
    /// `cache` already holds (a warm prefix-cache restore plus the app
    /// driver's checkpoint-capturing prefill); the iterator capture-prefills
    /// only the remainder.
    static func makeIterator(
        input: LMInput,
        model: any LanguageModel,
        drafter: any DFlash2DrafterModel,
        cache: [any KVCache],
        prefilledPrefixTokens: Int = 0,
        parameters: GenerateParameters
    ) throws -> DFlash2SpeculativeTokenIterator {
        var iteratorParams = parameters
        iteratorParams.repetitionPenalty = nil
        iteratorParams.presencePenalty = nil
        iteratorParams.frequencyPenalty = nil
        var components = GenerationComponents()
        if let processor = GenerationLogitProcessor.resolve(
            for: parameters, pathQuantizesKVUpFront: true)
        {
            // One iterator = one generation, so handing the factory a single
            // resolved instance preserves the fresh-state contract; the box
            // routes the non-Sendable processor into the @Sendable factory.
            let box = UnsafeSendableBox(processor)
            components = components.appendingLogitProcessor { box.value }
        }
        return try DFlash2SpeculativeTokenIterator(
            input: input,
            mainModel: model,
            drafter: drafter,
            mainCache: cache,
            prefilledPrefixTokens: prefilledPrefixTokens,
            parameters: iteratorParams,
            blockSize: blockSize,
            components: components
        )
    }
}
