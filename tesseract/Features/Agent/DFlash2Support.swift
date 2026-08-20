import Foundation
import MLX
import MLXLLM
import MLXLMCommon

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
/// 2. **Loading** — instantiate the draft, 4-bit quantize it (reference:
///    `nn.quantize(draft, group_size: 64, bits: 4)`), bind the target's
///    embedding/head.
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

    /// Round-width cap (1 anchor + `blockSize - 1` drafts per verify pass at
    /// the widest). The checkpoint was distilled at block size 8 and the mma8
    /// verify kernel (ADR-0058) makes wide blocks near-flat in cost, so the
    /// iterator runs its adaptive-width policy under this cap: it narrows on
    /// acceptance (measured per-position acceptance decays with width on
    /// high-entropy content — 59% → 22% from bs3 to bs8 on the docs-summary
    /// bench — while width is free money on predictable content) instead of
    /// pinning one width for everything.
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

    /// Which loaded target classes the DFlash2 draft can bind to — exactly the
    /// classes `bindDFlashTarget` knows (anything else binds nothing and would
    /// trap on first use). The app force-loads the MLXLLM text target even
    /// from VLM-shaped checkpoints (see `MTPDrafterSupport`), so the MLXVLM
    /// container never reaches here in practice.
    static func pairsWithTarget(_ model: any LanguageModel) -> Bool {
        model is MLXLLM.Qwen35Model || model is Qwen35TextModel
    }

    // MARK: - Loading

    /// Load + 4-bit quantize the draft (reference: `nn.quantize(draft,
    /// group_size: 64, bits: 4)`). Target binding happens per generation in
    /// the iterator's `init` (`bindDFlashTarget`), so the loaded draft stays
    /// a value the actor can box without a `perform` hop. Callers must have
    /// checked ``pairsWithTarget(_:)`` — a wrong-family target traps in
    /// `bindDFlashTarget` by design.
    static func loadDrafter(
        directory: URL
    ) throws -> any DFlash2DrafterModel {
        try loadDFlash2Draft(
            from: directory, quantization: (groupSize: 64, bits: 4))
    }

    // MARK: - Engagement policy

    /// The server cold-path engagement decision (pure, unit-tested).
    /// Speculate when a drafter is loaded, the request is text-only on the
    /// identity key space, and the leaf store will run off `finalCache` alone
    /// (`.directLeaf` — the DFlash2 iterator runs its own chunked prefill, so
    /// the mid-prefill boundary snapshots every other mode synthesizes its
    /// leaf from never exist; same constraint as MTP, ADR-0056).
    ///
    /// Unlike MTP there is no greedy gate (the DFlash2 iterator rejection-
    /// samples against the selector's candidate distribution, so sampling
    /// presets speculate identically) and no scratch gate (the prefill is
    /// chunked and the draft's context window is fixed at 2047 rows — prompt
    /// length does not change the memory shape).
    static func shouldEngage(
        hasDrafter: Bool,
        textOnlyIdentityKeySpace: Bool,
        predictedLeafStoreMode: HTTPLeafStoreMode
    ) -> Bool {
        hasDrafter && textOnlyIdentityKeySpace && predictedLeafStoreMode == .directLeaf
    }

    /// The agent raw-arm engagement decision: text-only input on a pairing
    /// target with a loaded drafter.
    static func shouldEngageRawArm(
        hasDrafter: Bool,
        input: LMInput
    ) -> Bool {
        hasDrafter && input.image == nil && input.video == nil && input.audio == nil
    }

    /// Build the DFlash2 iterator with the app's penalty discipline (ADR-0053):
    /// penalties are stripped from the iterator parameters and re-attached as
    /// the app logit processor through `GenerationComponents`, so the vendor's
    /// parameter-built penalty processor never doubles them. `kvBits` is
    /// cleared by the caller (speculation needs trimmable caches).
    static func makeIterator(
        input: LMInput,
        model: any LanguageModel,
        drafter: any DFlash2DrafterModel,
        cache: [any KVCache],
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
            parameters: iteratorParams,
            blockSize: blockSize,
            components: components
        )
    }
}
