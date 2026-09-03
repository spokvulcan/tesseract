import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXVLM

/// The anchored vision `prepare` verb: `(input, cache, state, windowSize)` →
/// `PrepareResult`. Present on a session only when the loaded family anchors
/// warm continuations — M-RoPE positions seeded from `state`, image-bearing
/// forwards windowed to `[heads, window, L]`.
typealias AnchoredVisionPrepare =
    (LMInput, [any KVCache], LMOutput.State?, Int?) throws -> PrepareResult

/// **Model Session** (CONTEXT.md → Server completion; ADR-0016): the scoped,
/// Metal-affine model handle the **Server Completion** enters for one batch of
/// model verbs. One session is one Metal-affine batch — the port's single
/// entry mirrors `ModelContainer.perform`, so ADR-0015's "decide before
/// entering" affinity discipline lives at the seam, and verbs are synchronous
/// inside it (`prepare` excepted: the vendor processor API is async).
///
/// The verbs are exactly the model operations the module already performs;
/// MLX value types (`MLXArray` inside `LMInput`, `[any KVCache]`) stay in the
/// port vocabulary deliberately — abstracting them would force the test peer
/// to reimplement decode semantics that then drift (ADR-0016, rejected
/// alternatives). Two adapters make the seam real: the container-backed
/// production provider below, and the test target's toy-model-backed provider
/// that runs these same verb implementations over microscopic tensors.
nonisolated protocol ModelSession {

    /// Load-time model configuration (stop-token set, tool-call format).
    var configuration: ModelConfiguration { get }

    /// The loaded tokenizer. Handed onward to tokenizer-affine helpers
    /// (boundary detection, the generation loop's detokenizer).
    var tokenizer: any Tokenizer { get }

    /// The loaded model's anchored vision `prepare`, when the family has
    /// wired the state-threaded windowed path (`nil` otherwise). The
    /// feature-detect `as?` cast, as a queryable fact.
    var anchoredVisionPrepare: AnchoredVisionPrepare? { get }

    /// Whether this model's processor emits a flat 1-D `[seq]` token list from
    /// a text-only `prepare` — true for the LLM families, false for the vision
    /// containers, whose text-only `prepare` still emits 2D `[batch, seq]`.
    ///
    /// The same feature-detect-as-a-fact shape as `anchoredVisionPrepare`, and
    /// the precondition for the C25 **Render+Token Cache** on the request path:
    /// building `LMInput(tokens:)` from the cache's token list must reproduce
    /// what the processor would have built.
    var producesFlatTextTokens: Bool { get }

    /// The MTP speculative-decoding drafter paired with the loaded model,
    /// when one was loaded beside it (`mtp.*` head weights present and the
    /// setting on). `nil` disables the speculative arm — the common case.
    var mtpDrafter: (any MTPDrafterModel)? { get }

    /// The DFlash2 block-parallel drafter paired with the loaded model, when
    /// the separate draft checkpoint was downloaded and the setting allows
    /// it. Preferred over ``mtpDrafter`` when both are present: deeper blocks
    /// (8 vs 2 effective), and it speculates under sampling presets too.
    var dflash2Drafter: (any DFlash2DrafterModel)? { get }

    /// Run the model's input processor: `UserInput` (messages, images,
    /// tools) → tokenized `LMInput`.
    func prepare(_ input: UserInput) async throws -> LMInput

    /// The chat-template message dicts the processor would render for
    /// `input` — the model's own `messageGenerator` where the family has one
    /// (`LLMModel`), the prompt's own `.messages` otherwise, `nil` when the
    /// session cannot say. The **Render+Token Cache** renders exactly these,
    /// so its token list reproduces the processor's.
    func templateMessages(for input: UserInput) -> [Message]?

    /// Construct the raw arms' decode iterator over the whole prompt from
    /// zero: the **Prefill Strategy** route (ADR-0044) decided and executed
    /// — the chunked arm warms a fresh cache through `PrefillExecutor`, the
    /// single-shot arm lets the vendor iterator's init prefill. The upstream
    /// `TokenIterator` shape the agent chat path decodes through.
    func makeRawDecodeIterator(
        _ input: LMInput,
        parameters: GenerateParameters
    ) throws -> TokenIterator

    /// Create the model-shaped empty KV cache array.
    func newCache(parameters: GenerateParameters) throws -> [any KVCache]

    /// Materialize a captured snapshot back into a live KV cache array.
    func restore(_ snapshot: HybridCacheSnapshot) throws -> [any KVCache]

    /// App-owned chunked prefill (`PrefillExecutor.run`) over `text` into
    /// `cache`, capturing checkpoints at the given absolute offsets.
    // Port vocabulary mirrors PrefillExecutor.run one-to-one by design.
    // swiftlint:disable:next function_parameter_count
    func prefill(
        text: LMInput.Text,
        cache: [any KVCache],
        checkpoints: [Int: HybridCacheSnapshot.CheckpointType],
        checkpointBaseOffset: Int,
        prefillStepSize: Int,
        consumeAll: Bool,
        initialState: LMOutput.State?,
        evalPolicy: PrefillExecutor.EvalPolicy
    ) throws -> PrefillExecutor.Output

    /// Construct the post-prefill decode iterator: the cache already covers
    /// everything but `remainder`; its init runs the real prime forward.
    func makeDecodeIterator(
        remainder: LMInput.Text,
        fullText: LMInput.Text,
        cache: [any KVCache],
        state: LMOutput.State?,
        parameters: GenerateParameters
    ) -> StateThreadedTokenIterator

    /// Construct the whole-prompt decode iterator (the **Unkeyed
    /// Completion**'s form): its init runs the model `prepare` — or the
    /// injected override, e.g. the windowed vision continuation from zero.
    func makePreparingDecodeIterator(
        _ input: LMInput,
        cache: [any KVCache],
        parameters: GenerateParameters,
        prepare: ((LMInput, [any KVCache], Int?) throws -> PrepareResult)?
    ) throws -> StateThreadedTokenIterator

    /// Construct the MTP speculative decode iterator over the whole prompt
    /// (its init runs the vendor `prepare` — whole-prompt and unchunked when
    /// the drafter requires prompt prefill, which is why callers gate on a
    /// scratch-size budget first). Only callable when ``mtpDrafter`` is
    /// non-nil. Penalties are stripped from the iterator's parameters inside
    /// the implementation and re-attached as the app logit processor
    /// (ADR-0053), so the vendor's parameter-built penalty processor never
    /// doubles them.
    func makeMTPDecodeIterator(
        _ input: LMInput,
        cache: [any KVCache],
        parameters: GenerateParameters
    ) throws -> MTPSpeculativeTokenIterator

    /// Construct the DFlash2 speculative decode iterator. Its init runs the
    /// capture-emitting chunked prefill (building the drafter's sliding
    /// hidden-state window) over the prompt positions past
    /// `prefilledPrefixTokens` — the leading positions `cache` already holds
    /// from a warm restore plus the app driver's checkpoint-capturing
    /// prefill. Only callable when ``dflash2Drafter`` is non-nil. Penalties
    /// are stripped and re-attached as the app logit processor, exactly like
    /// the MTP variant.
    func makeDFlash2DecodeIterator(
        _ input: LMInput,
        cache: [any KVCache],
        prefilledPrefixTokens: Int,
        parameters: GenerateParameters
    ) throws -> DFlash2SpeculativeTokenIterator

    /// Quantize the cache in place per the parameters' `kvBits`/`kvGroupSize`
    /// (no-op when unset) — once, before the iterator, so the array the
    /// module retains stays the live final cache.
    func quantizeKVCache(_ cache: inout [any KVCache], parameters: GenerateParameters)

    /// Capture a `HybridCacheSnapshot` of `cache` at `offset`. Returns `nil`
    /// on unsupported layer classes.
    func captureSnapshot(
        cache: [any KVCache],
        offset: Int,
        type: HybridCacheSnapshot.CheckpointType
    ) -> HybridCacheSnapshot?
}

/// Thrown by the default ``ModelSession/makeMTPDecodeIterator(_:cache:parameters:)``
/// when a session has no drafter — reaching it means an engagement-policy bug,
/// since callers gate on ``ModelSession/mtpDrafter`` first.
struct MTPDrafterUnavailableError: Error {}

/// The DFlash2 arm's twin of ``MTPDrafterUnavailableError``.
struct DFlash2DrafterUnavailableError: Error {}

extension ModelSession {
    /// Sessions without speculative decoding (the test peers, and any future
    /// adapter that never loads a drafter) inherit the disabled state.
    ///
    /// `nonisolated` is load-bearing: the protocol is nonisolated, but under
    /// the project's MainActor default isolation an unannotated extension
    /// member becomes a MainActor-isolated witness for a nonisolated
    /// requirement, and the runtime isolation check traps off the main actor.
    nonisolated var mtpDrafter: (any MTPDrafterModel)? { nil }

    nonisolated var dflash2Drafter: (any DFlash2DrafterModel)? { nil }

    /// The agent-edge tokenize verb (ADR-0016 amendment): the **Conversation
    /// Render**'s `agentEdgeFullRender` — C25 **Render+Token Cache** render +
    /// verified suffix encode when the request is eligible (no media, a
    /// flat-token model, a known fingerprint, a rendering tokenizer), fed
    /// lazily by `templateMessages(for:)` — and the processor's `prepare`
    /// otherwise. The **Raw Generation Start** tokenizes through it; the
    /// **Request Keying** phase builds its own `ConversationRender` value at
    /// the edge because later phases carry it. Not a requirement: one
    /// spelling over the port's own verbs, which decorators inherit — a
    /// recording peer's overridden `producesFlatTextTokens` steers the same
    /// code production runs.
    ///
    /// A `nil` fingerprint BYPASSES rather than resolving under a synthetic
    /// key; any render/encode failure falls back too, which reproduces the
    /// processor's own error handling (the missing-template plain-text
    /// fallback stays in the processor).
    nonisolated func prepareText(
        _ input: UserInput, modelFingerprint: String?
    ) async throws -> LMInput {
        if let tokens = ConversationRender.agentEdgeFullRender(
            tokenizer: tokenizer,
            messages: templateMessages(for: input),
            tools: input.tools,
            additionalContext: input.additionalContext,
            hasMedia: !(input.images.isEmpty && input.videos.isEmpty && input.audios.isEmpty),
            producesFlatTextTokens: producesFlatTextTokens,
            modelFingerprint: modelFingerprint
        ) {
            return LMInput(tokens: MLXArray(tokens))
        }
        return try await prepare(input)
    }

    nonisolated func makeMTPDecodeIterator(
        _ input: LMInput,
        cache: [any KVCache],
        parameters: GenerateParameters
    ) throws -> MTPSpeculativeTokenIterator {
        throw MTPDrafterUnavailableError()
    }

    nonisolated func makeDFlash2DecodeIterator(
        _ input: LMInput,
        cache: [any KVCache],
        prefilledPrefixTokens: Int,
        parameters: GenerateParameters
    ) throws -> DFlash2SpeculativeTokenIterator {
        throw DFlash2DrafterUnavailableError()
    }
}

/// The **Model Session** port: how the Server Completion enters a session.
/// Production adapter wraps the model container; the test peer enters a toy
/// model directly. Everything inside `body` runs on the session's isolation —
/// callers must eval any `MLXArray` before returning, exactly as with
/// `ModelContainer.perform`.
nonisolated protocol ModelSessionProviding: Sendable {
    /// The single entry, carrying a payload into the session — the
    /// container's `perform(nonSendable:)` shape, so a non-`Sendable`
    /// value (the agent's `UserInput`: images, tool dicts) crosses without
    /// a box at the caller. Payload-free callers use the extension form.
    func withSession<V, R: Sendable>(
        nonSendable payload: sending V,
        _ body: @Sendable (any ModelSession, V) async throws -> R
    ) async throws -> R
}

extension ModelSessionProviding {
    /// The payload-free form the server path uses: its `UserInput` is built
    /// inside the session, so nothing crosses in.
    nonisolated func withSession<R: Sendable>(
        _ body: @Sendable (any ModelSession) async throws -> R
    ) async throws -> R {
        try await withSession(nonSendable: ()) { session, _ in try await body(session) }
    }
}

/// The shared verb implementations over a live `ModelContext` — used by the
/// production provider below and (over a toy-model context) by the test
/// peer, so only the model varies across the seam.
nonisolated struct ContextBackedModelSession: ModelSession {
    let context: ModelContext
    /// The drafter loaded beside this model, threaded in by the production
    /// provider; `nil` on sessions without speculative decoding.
    var mtpDrafter: (any MTPDrafterModel)?
    /// The DFlash2 draft loaded beside this model (separate checkpoint);
    /// `nil` unless the draft folder exists and the setting is on.
    var dflash2Drafter: (any DFlash2DrafterModel)?

    var configuration: ModelConfiguration { context.configuration }
    var tokenizer: any Tokenizer { context.tokenizer }
    var anchoredVisionPrepare: AnchoredVisionPrepare? {
        // Concrete-class feature detect: since upstream #399 the anchored
        // windowed continuation is the Qwen3.5/3.6 container's own `prepare`
        // (the old `WindowedVisionContinuation` protocol is gone). Other VLM
        // families accept `state:` but ignore it (mlx-swift-lm issue #420),
        // so only the class that anchors qualifies.
        guard let model = context.model as? Qwen35 else { return nil }
        return { input, cache, state, windowSize in
            try model.prepare(
                input, cache: cache, state: state, prefill: .init(stepSize: windowSize))
        }
    }
    var producesFlatTextTokens: Bool {
        // The same marker protocol both installed processors branch on
        // (`LLMUserInputProcessor` in MLXLLM, the app's
        // `ParoQuantInputProcessor`): an `LLMModel` gets the text-only
        // processor that returns `LMInput(tokens: MLXArray(promptTokens))`.
        // `VLMModel` is a disjoint marker, so a vision container answers false.
        context.model is any LLMModel
    }

    func prepare(_ input: UserInput) async throws -> LMInput {
        try await context.processor.prepare(input: input)
    }

    func templateMessages(for input: UserInput) -> [Message]? {
        // The model's own generator is the exact expression the installed
        // processors (`LLMUserInputProcessor` in MLXLLM, the app-side
        // ParoQuant processor) captured at load — so the dicts rendered here
        // are the dicts the processor would render. A `.messages` prompt
        // reaches every processor's `generate(from:)` unchanged.
        if let llmModel = context.model as? any LLMModel {
            return llmModel.messageGenerator(tokenizer: context.tokenizer).generate(from: input)
        }
        // The processor installed on every other path (`ParoQuantLoader`'s
        // and the test peer's) forms messages with the vendor default.
        return DefaultMessageGenerator().generate(from: input)
    }

    func makeRawDecodeIterator(
        _ input: LMInput,
        parameters: GenerateParameters
    ) throws -> TokenIterator {
        try PrefillStrategy.decide(for: input, prefillStepSize: parameters.prefill.stepSize)
            .makeIterator(input: input, model: context.model, parameters: parameters)
    }

    func newCache(parameters: GenerateParameters) throws -> [any KVCache] {
        try context.model.newCache(parameters: parameters)
    }

    func restore(_ snapshot: HybridCacheSnapshot) throws -> [any KVCache] {
        try snapshot.restore()
    }

    // swiftlint:disable:next function_parameter_count
    func prefill(
        text: LMInput.Text,
        cache: [any KVCache],
        checkpoints: [Int: HybridCacheSnapshot.CheckpointType],
        checkpointBaseOffset: Int,
        prefillStepSize: Int,
        consumeAll: Bool,
        initialState: LMOutput.State?,
        evalPolicy: PrefillExecutor.EvalPolicy
    ) throws -> PrefillExecutor.Output {
        try PrefillExecutor.run(
            model: context.model,
            text: text,
            cache: cache,
            checkpoints: checkpoints,
            checkpointBaseOffset: checkpointBaseOffset,
            prefillStepSize: prefillStepSize,
            consumeAll: consumeAll,
            initialState: initialState,
            evalPolicy: evalPolicy
        )
    }

    func makeDecodeIterator(
        remainder: LMInput.Text,
        fullText: LMInput.Text,
        cache: [any KVCache],
        state: LMOutput.State?,
        parameters: GenerateParameters
    ) -> StateThreadedTokenIterator {
        StateThreadedTokenIterator(
            remainder: remainder,
            fullText: fullText,
            model: context.model,
            cache: cache,
            state: state,
            parameters: parameters
        )
    }

    func makePreparingDecodeIterator(
        _ input: LMInput,
        cache: [any KVCache],
        parameters: GenerateParameters,
        prepare: ((LMInput, [any KVCache], Int?) throws -> PrepareResult)?
    ) throws -> StateThreadedTokenIterator {
        try StateThreadedTokenIterator(
            preparing: input,
            model: context.model,
            cache: cache,
            parameters: parameters,
            prepare: prepare
        )
    }

    func makeMTPDecodeIterator(
        _ input: LMInput,
        cache: [any KVCache],
        parameters: GenerateParameters
    ) throws -> MTPSpeculativeTokenIterator {
        guard let drafter = mtpDrafter else {
            throw MTPDrafterUnavailableError()
        }
        // Penalties ride the app processor (ADR-0053), injected through
        // `GenerationComponents`; strip them from the iterator's parameters
        // so `components.logitProcessor(parameters:)` doesn't also build the
        // vendor's parameter-driven penalty processor on top.
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
        return try MTPSpeculativeTokenIterator(
            input: input,
            mainModel: context.model,
            drafter: drafter,
            mainCache: cache,
            parameters: iteratorParams,
            blockSize: MTPDrafterSupport.blockSize,
            components: components
        )
    }

    func makeDFlash2DecodeIterator(
        _ input: LMInput,
        cache: [any KVCache],
        prefilledPrefixTokens: Int,
        parameters: GenerateParameters
    ) throws -> DFlash2SpeculativeTokenIterator {
        guard let drafter = dflash2Drafter else {
            throw DFlash2DrafterUnavailableError()
        }
        return try DFlash2Support.makeIterator(
            input: input,
            model: context.model,
            drafter: drafter,
            cache: cache,
            prefilledPrefixTokens: prefilledPrefixTokens,
            parameters: parameters
        )
    }

    func quantizeKVCache(_ cache: inout [any KVCache], parameters: GenerateParameters) {
        maybeQuantizeKVCache(
            cache: &cache,
            kvBits: parameters.kvBits,
            kvGroupSize: parameters.kvGroupSize,
            quantizedKVStart: parameters.quantizedKVStart
        )
    }

    func captureSnapshot(
        cache: [any KVCache],
        offset: Int,
        type: HybridCacheSnapshot.CheckpointType
    ) -> HybridCacheSnapshot? {
        HybridCacheSnapshot.capture(cache: cache, offset: offset, type: type)
    }
}

/// Production adapter: one session = one `ModelContainer.perform` — the
/// perform hop *is* the adapter, so converted and unconverted call sites can
/// coexist mid-migration with identical Metal-affine batching.
nonisolated struct ContainerModelSessionProvider: ModelSessionProviding {
    let container: ModelContainer
    /// Boxed drafter handed through to every session; see the `LLMActor`
    /// field for the sharing rationale.
    var mtpDrafter: UnsafeSendableBox<any MTPDrafterModel>?
    /// Boxed DFlash2 draft handed through to every session.
    var dflash2Drafter: UnsafeSendableBox<any DFlash2DrafterModel>?

    func withSession<V, R: Sendable>(
        nonSendable payload: sending V,
        _ body: @Sendable (any ModelSession, V) async throws -> R
    ) async throws -> R {
        try await container.perform(nonSendable: payload) { context, payload in
            try await body(
                ContextBackedModelSession(
                    context: context, mtpDrafter: mtpDrafter?.value,
                    dflash2Drafter: dflash2Drafter?.value),
                payload)
        }
    }
}
