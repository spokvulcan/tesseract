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
    ///
    /// `DFLASH2_DRAFT_BITS` (probe lever, ledger R53): 8 loads an 8-bit
    /// draft, ≥16 keeps the checkpoint's bf16. Draft precision is
    /// identity-safe by construction — the target verifies every proposal,
    /// so it moves acceptance, never output — making it the one lever the
    /// R44 trajectory trap does not bind. Default 4 is the reference path.
    static func loadDrafter(
        directory: URL
    ) throws -> any DFlash2DrafterModel {
        let bits =
            ProcessInfo.processInfo.environment["DFLASH2_DRAFT_BITS"]
            .flatMap(Int.init) ?? 4
        return try loadDFlash2Draft(
            from: directory,
            quantization: bits >= 16 ? nil : (groupSize: 64, bits: bits))
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

    /// The agent raw-arm engagement decision: text-only input on a pairing
    /// target with a loaded drafter.
    static func shouldEngageRawArm(
        hasDrafter: Bool,
        input: LMInput
    ) -> Bool {
        hasDrafter && input.image == nil && input.video == nil && input.audio == nil
    }

    /// The raw arms' engage-and-build step, shared by `startRawGeneration`
    /// and the thinking-safeguard continuation so the engagement contract has
    /// one home: gate (loaded drafter, text-only input, pairing target),
    /// clear `kvBits` (speculation needs trimmable caches), build the fresh
    /// cache and the iterator. Returns `nil` when the arm doesn't engage —
    /// the caller falls back to the ordinary `PrefillStrategy` path.
    static func rawArmIterator(
        input: LMInput,
        model: any LanguageModel,
        drafter: (any DFlash2DrafterModel)?,
        parameters: GenerateParameters
    ) throws -> DFlash2SpeculativeTokenIterator? {
        guard let drafter,
            shouldEngageRawArm(hasDrafter: true, input: input),
            pairsWithTarget(model)
        else { return nil }
        var specParams = parameters
        specParams.kvBits = nil  // speculation needs trimmable caches
        let cache = try model.newCache(parameters: specParams)
        return try makeIterator(
            input: input, model: model, drafter: drafter,
            cache: cache, parameters: specParams)
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
