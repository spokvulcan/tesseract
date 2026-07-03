import Foundation
import MLX
import MLXLMCommon

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

    /// The loaded model's windowed vision continuation, when the family has
    /// wired the chunked image path (`nil` otherwise). The feature-detect
    /// `as?` cast, as a queryable fact.
    var windowedVisionContinuation: (any WindowedVisionContinuation)? { get }

    /// Run the model's input processor: `UserInput` (messages, images,
    /// tools) → tokenized `LMInput`.
    func prepare(_ input: UserInput) async throws -> LMInput

    /// Create the model-shaped empty KV cache array.
    func newCache(parameters: GenerateParameters) -> [any KVCache]

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
        prepare: ((LMInput, [any KVCache], Int) throws -> PrepareResult)?
    ) throws -> StateThreadedTokenIterator

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

/// The **Model Session** port: how the Server Completion enters a session.
/// Production adapter wraps the model container; the test peer enters a toy
/// model directly. Everything inside `body` runs on the session's isolation —
/// callers must eval any `MLXArray` before returning, exactly as with
/// `ModelContainer.perform`.
nonisolated protocol ModelSessionProviding: Sendable {
    func withSession<R: Sendable>(
        _ body: @Sendable (any ModelSession) async throws -> R
    ) async throws -> R
}

/// The shared verb implementations over a live `ModelContext` — used by the
/// production provider below and (over a toy-model context) by the test
/// peer, so only the model varies across the seam.
nonisolated struct ContextBackedModelSession: ModelSession {
    let context: ModelContext

    var configuration: ModelConfiguration { context.configuration }
    var tokenizer: any Tokenizer { context.tokenizer }
    var windowedVisionContinuation: (any WindowedVisionContinuation)? {
        context.model as? WindowedVisionContinuation
    }

    func prepare(_ input: UserInput) async throws -> LMInput {
        try await context.processor.prepare(input: input)
    }

    func newCache(parameters: GenerateParameters) -> [any KVCache] {
        context.model.newCache(parameters: parameters)
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
        prepare: ((LMInput, [any KVCache], Int) throws -> PrepareResult)?
    ) throws -> StateThreadedTokenIterator {
        try StateThreadedTokenIterator(
            preparing: input,
            model: context.model,
            cache: cache,
            parameters: parameters,
            prepare: prepare
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

    func withSession<R: Sendable>(
        _ body: @Sendable (any ModelSession) async throws -> R
    ) async throws -> R {
        try await container.perform { context in
            try await body(ContextBackedModelSession(context: context))
        }
    }
}
