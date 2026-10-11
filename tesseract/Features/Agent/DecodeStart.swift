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
    /// The **Prefill Strategy** route (ADR-0044) prefills the whole prompt
    /// into a cache of its own, chunked or single-shot, and builds the
    /// vendor iterator, which applies the parameters' KV plan itself.
    case prefillStrategy(LMInput)
}

/// **Decode Start** (CONTEXT.md → Prefill orchestration; ADR-0093): the one
/// module that starts decode for every generation path — the keyed turn
/// (standard, or speculative at a split), the keyed whole-prompt arm, the
/// Unkeyed Completion and both arms of the Raw Generation Start. From the
/// **Decode Handover** it builds the iterator through the Model Session's
/// existing verbs, converts the KV cache once where that handover needs it,
/// hands the array decode advances to its owner, badges a speculative arm
/// and starts the token loop.
///
/// The conversion rule lives here and nowhere else (ADR-0083): a standard
/// handover converts before its iterator, so the iterator converts nothing;
/// an iterator that prefills (speculative, preparing, the Prefill Strategy
/// route) converts at the end of its own prefill, and the array it advances
/// is read back. Affine `kvBits` applies only where the app quantizes
/// before the iterator or the route's own plan does: no iterator past the
/// app's prefill applies it (ADR-0006, ADR-0007).
///
/// Built and started inside one Model Session. Starts its loop once.
nonisolated final class DecodeStart {

    /// The speculative arm the request decodes with; `nil` decodes
    /// ordinarily.
    let arm: SpeculativeArm?

    /// The array decode advances, after any conversion; `nil` on the
    /// Prefill Strategy route, whose vendor iterator keeps its own.
    private let decoded: [any KVCache]?

    private var iterator: Iterator?

    private enum Iterator {
        case stateThreaded(StateThreadedTokenIterator)
        case speculative(SpeculativeDecodeIterator)
        case vendor(TokenIterator)
    }

    private init(arm: SpeculativeArm?, decoded: [any KVCache]?, iterator: Iterator) {
        self.arm = arm
        self.decoded = decoded
        self.iterator = iterator
    }

    /// Build the iterator `handover` names over `cache`, converting the
    /// cache once where the handover needs it. `parameters` are the
    /// request's own, KV plan included; each verb gets the part that is
    /// still its to apply. On return `cache` is the array decode advances:
    /// the caller's reference never keeps the layers a conversion replaced.
    /// The Prefill Strategy route allocates its cache itself, so `cache`
    /// stays empty there.
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
        case .prefillStrategy(let input):
            precondition(cache.isEmpty, "the Prefill Strategy route allocates its own cache")
            let iterator = try session.makeRawDecodeIterator(input, parameters: parameters)
            return DecodeStart(arm: nil, decoded: nil, iterator: .vendor(iterator))
        }
    }

    /// Hand the array decode advances to its owner: a leased leaf's cache,
    /// which adopts it (a conversion may have replaced its attention
    /// entries), or a new owner for a cache the request holds outright.
    func cacheOwner(lease: FinalGenerationCache?) -> FinalGenerationCache {
        guard let decoded else {
            preconditionFailure("the Prefill Strategy route keeps its cache to itself")
        }
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

    /// Start the app-owned token loop over the iterator. The vendor
    /// iterator keeps upstream's loop, which records no generated ids.
    func startLoop(
        promptTokenCount: Int,
        session: any ModelSession,
        tools: [ToolSpec]?,
        generatedTokens: GeneratedTokenRecorder?
    ) -> (AsyncStream<RawGeneration>, Task<Void, Never>) {
        guard let iterator else { preconditionFailure("a Decode Start starts its loop once") }
        self.iterator = nil
        let configuration = session.configuration
        let tokenizer = session.tokenizer
        switch iterator {
        case .stateThreaded(let iterator):
            return TokenGenerationLoop.start(
                promptTokenCount: promptTokenCount, modelConfiguration: configuration,
                tokenizer: tokenizer, iterator: iterator, tools: tools,
                generatedTokens: generatedTokens)
        case .speculative(.mtp(let iterator)):
            return TokenGenerationLoop.start(
                promptTokenCount: promptTokenCount, modelConfiguration: configuration,
                tokenizer: tokenizer, iterator: iterator, tools: tools,
                generatedTokens: generatedTokens, speculativeArm: .mtp)
        case .speculative(.dflash2(let iterator)):
            return TokenGenerationLoop.start(
                promptTokenCount: promptTokenCount, modelConfiguration: configuration,
                tokenizer: tokenizer, iterator: iterator, tools: tools,
                generatedTokens: generatedTokens, speculativeArm: .dflash2)
        case .vendor(let iterator):
            precondition(generatedTokens == nil, "upstream's loop records no generated ids")
            return TokenGenerationLoop.start(
                promptTokenCount: promptTokenCount, modelConfiguration: configuration,
                tokenizer: tokenizer, iterator: iterator, tools: tools)
        }
    }
}
