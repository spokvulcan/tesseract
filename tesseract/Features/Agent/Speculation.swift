import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// **Speculation** (CONTEXT.md → Speculative decoding; ADR-0079): the
/// speculative drafters resident beside one model load, and the one place
/// that decides what a request runs with them.
///
/// Two drafter families: the MTP head that ships inside a Qwen3.5-family
/// checkpoint (ADR-0056), and the separate DFlash2 draft beside Qwen3.8-27B
/// (ADR-0057, ADR-0059). A model load builds the value with
/// ``load(_:beside:checkpoint:identity:draftStorageRoot:)``, the **Model
/// Session** carries it, and each request asks it for a **Speculation
/// Plan** with ``plan(for:)``. The Server Completion's keyed path and the
/// Raw Generation Start ask the same question with the same facts, so
/// neither holds an engagement rule of its own.
///
/// Drafters hold no per-stream state (the vendor documents both families as
/// stateless), so one resident drafter serves every session. They are boxed
/// rather than container-wrapped; every use is generation work inside a
/// Model Session.
nonisolated struct Speculation: Sendable {

    /// No drafter resident: plain autoregressive decoding.
    static let none = Speculation()

    /// Ceiling on MTP's single-shot full-attention score matrix
    /// (`[heads, L, L]`). The vendor MTP prompt prefill is unchunked by
    /// design (the head needs one target hidden row per prompt token), so
    /// prompt length decides engagement: past this bound the request keeps
    /// the ordinary chunked path. 4 GiB ≈ a 9K-token prompt on the 27B's
    /// 24-head bf16 profile.
    static let mtpSingleShotScratchBudgetBytes: UInt64 = 4 << 30

    private var mtp: UnsafeSendableBox<any MTPDrafterModel>?
    private var dflash2: UnsafeSendableBox<any DFlash2DrafterModel>?
    /// The loaded target's full-attention scratch profile, which prices
    /// MTP's single-shot prompt prefill. `nil` prices nothing, so MTP never
    /// engages unpriced.
    private let scratchProfile: ModelIdentity.FullAttentionScratchProfile?

    /// The drafters a load attached. Production builds this through
    /// ``load(_:beside:checkpoint:identity:draftStorageRoot:)``; test
    /// sessions hand in scripted or presence-only drafters.
    init(
        mtpDrafter: (any MTPDrafterModel)? = nil,
        dflash2Drafter: (any DFlash2DrafterModel)? = nil,
        scratchProfile: ModelIdentity.FullAttentionScratchProfile? = nil
    ) {
        self.mtp = mtpDrafter.map(UnsafeSendableBox.init)
        self.dflash2 = dflash2Drafter.map(UnsafeSendableBox.init)
        self.scratchProfile = scratchProfile
    }

    /// Whether the drafter behind `arm` is resident beside the loaded model.
    func isResident(_ arm: SpeculativeArm) -> Bool {
        switch arm {
        case .mtp: mtp != nil
        case .dflash2: dflash2 != nil
        }
    }

    /// Residency facts for the request-memory telemetry: the DFlash2 draft's
    /// weight bytes and which drafters are loaded.
    var memoryFacts: [String: String] {
        let draftBytes =
            dflash2?.value.parameters().flattened().reduce(0) { $0 + $1.1.nbytes } ?? 0
        return [
            "dflash2WeightArrayBytes": "\(draftBytes)",
            "dflash2Loaded": "\(dflash2 != nil)",
            "mtpLoaded": "\(mtp != nil)",
        ]
    }

    // MARK: - The Speculation Plan

    /// The **Speculation Plan** for one request, or `nil` when it decodes
    /// without speculation. The engagement table, in one place:
    ///
    /// - Speculation needs KV a verify pass can write in place: full
    ///   precision, or a TurboQuant **KV Scheme** for DFlash2 (its iterator
    ///   converts after prefill and verifies over the compressed rows).
    ///   Affine `kvBits` never speculates, and MTP does not take a KV Scheme.
    /// - DFlash2, when resident, is preferred and engages on every such
    ///   request, warm or cold, whatever leaf it stores (ADR-0059), with
    ///   images or without (ADR-0089). The app prefills and captures up to
    ///   the split, and the iterator capture-prefills the text tail: a keyed
    ///   split sits past the last image and the tail rotates by the images'
    ///   rope delta; a whole prompt from zero has the target prefill its own
    ///   images first. Only the Qwen3.5 classes pair with the draft, and of
    ///   those only the vision class receives images. It rejection-samples
    ///   against its selector, so sampling presets speculate too.
    /// - MTP needs text-only input and engages only at temperature 0 (the
    ///   Qwen heads are greedy-only; the vendor iterator would pass through
    ///   after paying the drafter prefill), on a turn that restores nothing
    ///   and stores a direct leaf, when the single-shot prefill fits the
    ///   scratch budget. Its iterator prefills the whole prompt from zero,
    ///   which forfeits the boundary snapshots every other leaf mode is
    ///   synthesized from (ADR-0056 amendment). A turn that stores no leaf —
    ///   the Raw Generation Start — does not engage it.
    func plan(for request: SpeculationRequest) -> SpeculationPlan? {
        guard request.kvBits == nil else { return nil }
        if let dflash2 {
            return SpeculationPlan(drafter: .dflash2(dflash2))
        }
        if let mtp, request.isTextOnly, request.kvScheme == nil, request.temperature == 0,
            !request.restoresPrefix, request.storedLeaf == .directLeaf,
            mtpPrefillFits(promptTokens: request.promptTokens)
        {
            return SpeculationPlan(drafter: .mtp(mtp))
        }
        return nil
    }

    private func mtpPrefillFits(promptTokens: Int) -> Bool {
        guard let bytes = scratchProfile?.scoreMatrixBytes(sequenceLength: promptTokens)
        else { return false }
        return bytes <= Self.mtpSingleShotScratchBudgetBytes
    }
}

/// The facts a **Speculation Plan** is decided from, as both arms know
/// them before their prefill.
nonisolated struct SpeculationRequest: Sendable, Equatable {
    /// No image, video or audio reaches the model (on the keyed path, the
    /// identity key space). Only MTP reads it.
    var isTextOnly: Bool
    /// The request's affine KV quantization; `nil` is unquantized.
    var kvBits: Int?
    /// The request's **KV Scheme**; `nil` is full precision.
    var kvScheme: KVScheme? = nil
    var temperature: Float
    /// Tokens in the whole prompt.
    var promptTokens: Int
    /// Whether the turn restores a cached prefix before it prefills.
    var restoresPrefix: Bool
    /// The leaf the turn will store in the prefix cache (predicted: tool
    /// emission is unknowable up front, so defined tools predict a tool
    /// leaf), or `nil` when it stores none.
    var storedLeaf: HTTPLeafStoreMode?
}

/// **Speculation Plan** (CONTEXT.md → Speculative decoding; ADR-0079): what
/// one request runs speculatively, decided once by ``Speculation/plan(for:)``
/// and read by whichever arm serves the request.
nonisolated struct SpeculationPlan: Sendable {

    fileprivate enum Drafter: Sendable {
        case mtp(UnsafeSendableBox<any MTPDrafterModel>)
        case dflash2(UnsafeSendableBox<any DFlash2DrafterModel>)
    }

    fileprivate let drafter: Drafter

    /// The arm that decodes the request.
    var arm: SpeculativeArm {
        switch drafter {
        case .mtp: .mtp
        case .dflash2: .dflash2
        }
    }

    /// Tokens a round may write past the last emitted token: the block the
    /// iterator verifies. `CacheClaim.maximumAdvance` adds it to the turn's
    /// growth.
    var advanceAllowance: Int {
        switch drafter {
        case .mtp: MTPDrafterSupport.blockSize
        case .dflash2: DFlash2Support.blockSize
        }
    }

    /// Whether the iterator prefills the whole prompt itself, into an empty
    /// cache. MTP's head needs a target hidden row for every prompt token,
    /// so its prefill is one unchunked vendor pass and the app's prefill
    /// captures nothing; the plan only exists when nothing is restored.
    var prefillsWholePrompt: Bool {
        switch drafter {
        case .mtp: true
        case .dflash2: false
        }
    }

    /// Where the app's checkpoint-capturing prefill hands over to the
    /// iterator's own prefill, for a prompt whose cache already holds
    /// `executionBaseOffset` positions.
    ///
    /// DFlash2 splits past the deepest planned capture (every checkpoint and
    /// boundary snapshot lands), no earlier than its context window's start
    /// (a deep cold prompt keeps the pipelined driver for its bulk), and
    /// always leaves at least the final prompt token to the iterator, which
    /// samples the first token from it. MTP hands over at the base: its
    /// iterator takes every prompt row.
    func prefillSplit(
        checkpointOffsets: some Sequence<Int>,
        executionBaseOffset: Int,
        promptTokens: Int
    ) -> Int {
        switch drafter {
        case .mtp:
            return executionBaseOffset
        case .dflash2(let drafter):
            let lastCapture = checkpointOffsets.max() ?? executionBaseOffset
            let windowStart = promptTokens - 1 - drafter.value.contextWindow
            return min(max(executionBaseOffset, lastCapture, windowStart), promptTokens - 1)
        }
    }

    /// Build the plan's iterator over `cache`, which already holds the first
    /// `prefilledPrefixTokens` positions of `input`. The Model Session calls
    /// this with its model; nothing else constructs a speculative iterator.
    ///
    /// An `input` with images (a whole prompt from zero) has the target
    /// prefill through its last image first. A caller that prefilled the
    /// images itself passes the text alone, a split past the last image, and
    /// `positionDelta`, the rope delta the text after them rotates by.
    ///
    /// Penalties ride the app logit processor (ADR-0053), injected through
    /// `GenerationComponents`; they are stripped from the iterator's
    /// parameters so the vendor's parameter-built penalty processor never
    /// doubles them. The caller's parameters carry no `kvBits` (the plan
    /// refuses affine KV); a KV Scheme rides them to the DFlash2 iterator,
    /// which converts after its prefill, so read the cache back from
    /// ``SpeculativeDecodeIterator/cache``.
    func makeIterator(
        input: LMInput,
        model: any LanguageModel,
        cache: [any KVCache],
        prefilledPrefixTokens: Int,
        positionDelta: Int = 0,
        parameters: GenerateParameters
    ) throws -> SpeculativeDecodeIterator {
        let (iteratorParameters, components) = GenerationLogitProcessor.components(
            for: parameters)
        switch drafter {
        case .mtp(let drafter):
            precondition(
                prefilledPrefixTokens == 0 && positionDelta == 0,
                "the MTP iterator prefills the whole text-only prompt")
            return .mtp(
                try MTPSpeculativeTokenIterator(
                    input: input,
                    mainModel: model,
                    drafter: drafter.value,
                    mainCache: cache,
                    parameters: iteratorParameters,
                    blockSize: MTPDrafterSupport.blockSize,
                    components: components
                ))
        case .dflash2(let drafter):
            return .dflash2(
                try DFlash2SpeculativeTokenIterator(
                    input: input,
                    mainModel: model,
                    drafter: drafter.value,
                    mainCache: cache,
                    prefilledPrefixTokens: prefilledPrefixTokens,
                    positionDelta: positionDelta,
                    parameters: iteratorParameters,
                    blockSize: DFlash2Support.blockSize,
                    components: components
                ))
        }
    }
}

/// A **Speculation Plan**'s iterator, built inside a Model Session.
nonisolated enum SpeculativeDecodeIterator {
    case mtp(MTPSpeculativeTokenIterator)
    case dflash2(DFlash2SpeculativeTokenIterator)

    var arm: SpeculativeArm {
        switch self {
        case .mtp: .mtp
        case .dflash2: .dflash2
        }
    }

    /// The target cache the iterator decodes into, after its prefill: a
    /// KV Scheme replaced the attention entries of the array it was given.
    /// `nil` for MTP, which never takes a scheme.
    var cache: [any KVCache]? {
        switch self {
        case .mtp: nil
        case .dflash2(let iterator): iterator.cache
        }
    }

    /// Start the app-owned generation stream over the iterator: one
    /// `TokenGenerationLoop.start` per case, because the loop's entry is
    /// generic over the concrete iterator type.
    consuming func startGeneration(
        promptTokenCount: Int,
        modelConfiguration: ModelConfiguration,
        tokenizer: any MLXLMCommon.Tokenizer,
        tools: [ToolSpec]?,
        generatedTokens: GeneratedTokenRecorder? = nil
    ) -> (AsyncStream<RawGeneration>, Task<Void, Never>) {
        switch consume self {
        case .mtp(let iterator):
            TokenGenerationLoop.start(
                promptTokenCount: promptTokenCount,
                modelConfiguration: modelConfiguration,
                tokenizer: tokenizer,
                iterator: iterator,
                tools: tools,
                generatedTokens: generatedTokens,
                speculativeArm: .mtp
            )
        case .dflash2(let iterator):
            TokenGenerationLoop.start(
                promptTokenCount: promptTokenCount,
                modelConfiguration: modelConfiguration,
                tokenizer: tokenizer,
                iterator: iterator,
                tools: tools,
                generatedTokens: generatedTokens,
                speculativeArm: .dflash2
            )
        }
    }
}

// MARK: - Residency

nonisolated extension Speculation {

    /// Load every drafter `mode` allows that pairs with the model in
    /// `container`, the target loaded from `directory`. Never fails the model
    /// load: a drafter that is absent, refused or fails to load leaves its
    /// arm off, with a log line saying why. MTP loads first, then DFlash2.
    ///
    /// `draftStorageRoot` is the models directory the DFlash2 draft
    /// downloads into (`ModelDownloadManager.modelStorageURL`, which is
    /// MainActor-isolated, so the caller reads it).
    static func load(
        _ mode: SpeculationMode,
        beside container: ModelContainer,
        checkpoint directory: URL,
        identity: ModelIdentity,
        draftStorageRoot: URL
    ) async -> Speculation {
        let mtp: (any MTPDrafterModel)?
        if mode.allowsMTP {
            mtp = await loadMTPDrafter(directory: directory, container: container)
        } else {
            Log.agent.info("MTP drafter: disabled by setting — speculation off")
            mtp = nil
        }
        let dflash2 =
            mode.allowsDFlash2
            ? await loadDFlash2Drafter(
                container: container, identity: identity, storageRoot: draftStorageRoot)
            : nil
        return Speculation(
            mtpDrafter: mtp, dflash2Drafter: dflash2,
            scratchProfile: identity.fullAttentionScratchProfile)
    }

    /// Release the drafters, MTP first, recording a memory phase after each
    /// so an unload's allocation trace attributes their bytes.
    mutating func unload() {
        mtp = nil
        RequestMemoryTelemetry.recordAllocation(phase: "modelUnloadMTPReleased", facts: [:])
        dflash2 = nil
        RequestMemoryTelemetry.recordAllocation(phase: "modelUnloadDFlash2Released", facts: [:])
    }

    /// Weak references to the resident drafters, taken before an unload so
    /// it can report whether anything still retains them afterwards.
    var releaseProbe: ReleaseProbe {
        ReleaseProbe(mtp: mtp?.value as AnyObject?, dflash2: dflash2?.value as AnyObject?)
    }

    final class ReleaseProbe {
        private weak var mtp: AnyObject?
        private weak var dflash2: AnyObject?

        fileprivate init(mtp: AnyObject?, dflash2: AnyObject?) {
            self.mtp = mtp
            self.dflash2 = dflash2
        }

        /// The unload's retention facts.
        var facts: [String: String] {
            [
                "mtpDrafterRetained": "\(mtp != nil)",
                "dflash2DrafterRetained": "\(dflash2 != nil)",
            ]
        }
    }

    /// The MTP head, loaded from the target's own checkpoint when it ships
    /// `mtp.*` weights. The drafter family is derived from the class of the
    /// model instance in `container`, never from the vision intent: the
    /// generic loader falls back VLM → LLM when the VLM factory throws, so
    /// intent and outcome can diverge (see `MTPDrafterSupport.drafterPairing`).
    private static func loadMTPDrafter(
        directory: URL, container: ModelContainer
    ) async -> (any MTPDrafterModel)? {
        guard MTPDrafterSupport.checkpointShipsMTPHead(directory: directory) else {
            Log.agent.info("MTP drafter: checkpoint ships no mtp.* weights — speculation off")
            return nil
        }
        let pairing = await container.perform { context in
            MTPDrafterSupport.drafterPairing(for: context.model)
        }
        guard let pairing else {
            Log.agent.info(
                "MTP drafter: no drafter pairs with the loaded target class — speculation off")
            return nil
        }
        do {
            RequestMemoryTelemetry.recordAllocation(phase: "modelMTPLoadBegin", facts: [:])
            let context = try await MTPDrafterSupport.loadDrafter(
                directory: directory, pairing: pairing)
            // The head shard's pre-quantization arrays are dead once the
            // draft holds its parameters; return them before the next load.
            Memory.clearCache()
            RequestMemoryTelemetry.recordAllocation(phase: "modelMTPLoaded", facts: [:])
            Log.agent.notice(
                "MTP drafter loaded — pairing=\(pairing.rawValue) "
                    + "blockSize=\(MTPDrafterSupport.blockSize)")
            return context.model
        } catch {
            Log.agent.warning(
                "MTP drafter load failed — continuing without speculation: \(error)")
            return nil
        }
    }

    /// The DFlash2 draft, loaded from its own folder when it was downloaded,
    /// the checkpoint accepts it, the loaded target class pairs with it and
    /// the draft was distilled for the target's depth.
    private static func loadDFlash2Drafter(
        container: ModelContainer, identity: ModelIdentity, storageRoot: URL
    ) async -> (any DFlash2DrafterModel)? {
        guard let directory = DFlash2Support.draftDirectory(storageRoot: storageRoot) else {
            return nil  // draft not downloaded — the common case, stay silent
        }
        guard !DFlash2Support.checkpointRefusesDraft(identity) else {
            Log.agent.notice(
                "DFlash2 draft: the target is a Rotated Ternary Checkpoint — the draft was "
                    + "distilled for the full-precision target and decodes slower on it — off")
            return nil
        }
        let targetLayers = await container.perform { context in
            DFlash2Support.targetLayerCount(context.model)
        }
        guard let targetLayers else {
            Log.agent.info(
                "DFlash2 draft: loaded target class pairs with no DFlash2 draft — off")
            return nil
        }
        do {
            // Geometry comes from config.json, so a mismatched draft is
            // refused before its weights are read.
            let draftConfig = try DFlash2Support.draftConfiguration(directory: directory)
            guard
                DFlash2Support.geometryMatches(
                    targetLayerCount: targetLayers,
                    draftNumTargetLayers: draftConfig.numTargetLayers,
                    draftTargetLayerIds: draftConfig.dflash.targetLayerIds)
            else {
                Log.agent.notice(
                    "DFlash2 draft: distilled for a \(draftConfig.numTargetLayers)-layer target, "
                        + "loaded target has \(targetLayers) layers — off")
                return nil
            }
            RequestMemoryTelemetry.recordAllocation(phase: "modelDFlash2LoadBegin", facts: [:])
            let draft = try DFlash2Support.loadDrafter(directory: directory)
            RequestMemoryTelemetry.recordAllocation(phase: "modelDFlash2Loaded", facts: [:])
            // The target's projections were stacked at load; the draft's
            // fold here (same pass, bitwise-exact).
            Memory.clearCache()
            RequestMemoryTelemetry.recordAllocation(phase: "modelDraftStackingBegin", facts: [:])
            let stackedDraft = stackSameInputProjections(in: draft)
            if stackedDraft > 0 { Memory.clearCache() }
            Log.agent.notice("DFlash2 same-input stacking: draft=\(stackedDraft) blocks")
            RequestMemoryTelemetry.recordAllocation(phase: "modelProjectionStackingEnd", facts: [:])
            Log.agent.notice(
                "DFlash2 draft loaded (4-bit) — blockSize=\(DFlash2Support.blockSize)")
            return draft
        } catch {
            Log.agent.warning(
                "DFlash2 draft load failed — continuing without it: \(error)")
            return nil
        }
    }
}
