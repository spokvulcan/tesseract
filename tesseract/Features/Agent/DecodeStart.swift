import Foundation
import MLX
import MLXLMCommon

/// A **Decode Handover** (CONTEXT.md → Prefill orchestration): where one
/// generation's prefill hands its prompt to the decode iterator. The case
/// says who prefills what is left; the **Decode Start** reads from it the
/// Model Session verb that builds the iterator and where the **KV Scheme**
/// converts.
nonisolated enum DecodeHandover {
    /// The app's chunked prefill ran every prompt token but the last, which
    /// `remainder` holds, and `state` is its last chunk's model state (or a
    /// seeded **Position Anchor**). The standard iterator's prime forward
    /// runs the last token over the converted cache, and its penalties are
    /// seeded with `fullText`, the text the prefill covered.
    case standard(remainder: LMInput.Text, fullText: LMInput.Text, state: LMOutput.State?)
    /// The **Speculation Plan**'s iterator prefills `input` from
    /// `prefilledPrefixTokens` on: zero is the whole prompt from an empty
    /// cache; past zero is a split after the app's capturing prefill, past
    /// any images, whose rope delta `positionDelta` carries. The iterator
    /// converts the cache before the final prompt position.
    case speculative(
        SpeculationPlan, input: LMInput, prefilledPrefixTokens: Int, positionDelta: Int)
    /// The app's state-threaded iterator prefills the whole prompt into an
    /// empty cache, through `prepare` (the anchored vision `prepare` from
    /// zero) or the model's own, and converts the cache once it holds the
    /// whole prompt.
    case preparing(LMInput, prepare: ((LMInput, [any KVCache], Int?) throws -> PrepareResult)?)
}

/// **Decode Start** (CONTEXT.md → Prefill orchestration; ADR-0093): the one
/// module that starts decode for every generation path whose cache the app
/// holds — the keyed turn (standard, or speculative at a split), the keyed
/// whole-prompt arm, the Unkeyed Completion and the Raw Generation Start's
/// speculative arm. From the **Decode Handover** it builds the iterator
/// through the Model Session's existing verbs, converts the KV cache once
/// where that handover needs it, hands the array decode advances to its
/// owner, badges a speculative arm and starts the token loop. The Raw
/// Generation Start's other arm keeps no cache of the app's and starts
/// through `PrefillStrategyDecodeStart`.
///
/// The conversion rule lives here and nowhere else (ADR-0083): a standard
/// handover converts before its iterator, so the iterator converts nothing;
/// an iterator that prefills (speculative, preparing) converts at the end
/// of its own prefill, and the array it advances is read back. Affine
/// `kvBits` applies only where the app quantizes before the iterator: no
/// iterator past the app's prefill applies it (ADR-0006, ADR-0007).
///
/// Built and started inside one Model Session. Starts its loop once.
nonisolated final class DecodeStart {

    /// The speculative arm the request decodes with; `nil` decodes
    /// ordinarily.
    let arm: SpeculativeArm?

    /// The array decode advances, after any conversion.
    private let decoded: [any KVCache]

    private var iterator: Iterator?

    private enum Iterator {
        case stateThreaded(StateThreadedTokenIterator)
        case speculative(SpeculativeDecodeIterator)
    }

    private init(arm: SpeculativeArm?, decoded: [any KVCache], iterator: Iterator) {
        self.arm = arm
        self.decoded = decoded
        self.iterator = iterator
    }

    /// Build the iterator `handover` names over `cache`, converting the
    /// cache once where the handover needs it. `parameters` are the
    /// request's own, KV plan included; each verb gets the part that is
    /// still its to apply. On return `cache` is the array decode advances:
    /// the caller's reference never keeps the layers a conversion replaced.
    static func make(
        _ handover: DecodeHandover,
        cache: inout [any KVCache],
        session: any ModelSession,
        parameters: GenerateParameters
    ) throws -> DecodeStart {
        var iteratorParameters = parameters
        iteratorParameters.kvBits = nil
        switch handover {
        case .standard(let remainder, let fullText, let state):
            session.quantizeKVCache(&cache, parameters: parameters)
            // Converted: nothing is left for the iterator to apply.
            iteratorParameters.kvScheme = nil
            let iterator = session.makeDecodeIterator(
                remainder: remainder, fullText: fullText, cache: cache, state: state,
                parameters: iteratorParameters)
            return DecodeStart(arm: nil, decoded: cache, iterator: .stateThreaded(iterator))
        case .speculative(let plan, let input, let prefilledPrefixTokens, let positionDelta):
            let iterator = try session.makeSpeculativeDecodeIterator(
                input, cache: cache, prefilledPrefixTokens: prefilledPrefixTokens,
                positionDelta: positionDelta, plan: plan, parameters: iteratorParameters)
            if let converted = iterator.cache { cache = converted }
            return DecodeStart(arm: plan.arm, decoded: cache, iterator: .speculative(iterator))
        case .preparing(let input, let prepare):
            let iterator = try session.makePreparingDecodeIterator(
                input, cache: cache, parameters: iteratorParameters, prepare: prepare)
            cache = iterator.cache
            return DecodeStart(arm: nil, decoded: cache, iterator: .stateThreaded(iterator))
        }
    }

    /// Hand the array decode advances to its owner: a leased leaf's cache,
    /// which adopts it (a conversion may have replaced its attention
    /// entries), or a new owner for a cache the request holds outright.
    func cacheOwner(lease: FinalGenerationCache?) -> FinalGenerationCache {
        guard let lease else { return FinalGenerationCache(decoded) }
        lease.adopt(decoded)
        return lease
    }

    /// Report the prefill done: the speculation badge first when the request
    /// speculates, so the activity surfaces name the arm live, then the
    /// prefill's numbers.
    func reportPrefillFinished(
        _ prefill: ServerInferenceProgressEvent.PrefillInfo,
        to progressHandler: ServerInferenceProgressHandler?
    ) async {
        if let arm {
            await progressHandler?(.speculationEngaged(arm))
        }
        await progressHandler?(.prefillFinished(prefill))
    }

    /// Start the app-owned token loop over the iterator, badged with the
    /// speculative arm it decodes with.
    func startLoop(
        promptTokenCount: Int,
        session: any ModelSession,
        tools: [ToolSpec]?,
        generatedTokens: GeneratedTokenRecorder?
    ) -> (AsyncStream<RawGeneration>, Task<Void, Never>) {
        guard let iterator else { preconditionFailure("a Decode Start starts its loop once") }
        self.iterator = nil
        let arm = arm
        func loop(
            _ iterator: consuming some TokenIteratorProtocol & SendableMetatype
        ) -> (AsyncStream<RawGeneration>, Task<Void, Never>) {
            TokenGenerationLoop.start(
                promptTokenCount: promptTokenCount, modelConfiguration: session.configuration,
                tokenizer: session.tokenizer, iterator: iterator, tools: tools,
                generatedTokens: generatedTokens, speculativeArm: arm)
        }
        switch iterator {
        case .stateThreaded(let iterator): return loop(iterator)
        case .speculative(.mtp(let iterator)): return loop(iterator)
        case .speculative(.dflash2(let iterator)): return loop(iterator)
        }
    }
}

/// The Raw Generation Start's arm without a Speculation Plan: the **Prefill
/// Strategy** route (ADR-0044). Its vendor iterator prefills the whole prompt
/// into a cache of its own, chunked or single-shot, and applies the
/// parameters' KV plan itself, so this start hands no array to an owner,
/// converts nothing and records no generated ids; upstream's loop runs the
/// iterator. What the **Decode Start** does that this route must not is
/// therefore not in its interface at all.
nonisolated final class PrefillStrategyDecodeStart {
    private var iterator: TokenIterator?

    /// Build the vendor iterator over `input` with the request's own
    /// parameters, KV plan included.
    init(_ input: LMInput, session: any ModelSession, parameters: GenerateParameters) throws {
        iterator = try session.makeRawDecodeIterator(input, parameters: parameters)
    }

    /// Start upstream's token loop over the iterator, once.
    func startLoop(
        promptTokenCount: Int, session: any ModelSession, tools: [ToolSpec]?
    ) -> (AsyncStream<RawGeneration>, Task<Void, Never>) {
        guard let iterator else { preconditionFailure("a decode start starts its loop once") }
        self.iterator = nil
        return TokenGenerationLoop.start(
            promptTokenCount: promptTokenCount, modelConfiguration: session.configuration,
            tokenizer: session.tokenizer, iterator: iterator, tools: tools)
    }
}
