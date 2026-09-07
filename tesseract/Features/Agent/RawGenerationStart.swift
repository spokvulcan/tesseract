import Foundation
import MLX
import MLXLMCommon

/// A whole prompt tokenized from zero for a raw generation.
nonisolated enum RawGenerationPrompt {
    case fresh(UserInput)

    var lookupReason: String { "standardGenerationNoPrefixCache" }
}

/// **Raw Generation Start** (CONTEXT.md → Server completion; ADR-0016
/// amendment): the one script that starts a whole-prompt-from-zero
/// generation over a **Model Session** — tokenize through the session's
/// agent-edge verb, emit the lookup and prefill progress events, engage the
/// DFlash2 raw arm when the session pairs a drafter and the prompt is
/// text-only, else run the **Prefill Strategy** route, start the token-event
/// loop, and wrap the handles. Every raw start runs the same script, so the
/// speculation badge, tokenize step, and handle wrap exist once.
///
/// **Metal-affinity contract:** must run inside a session (`withSession`),
/// exactly as the arms it replaced ran inside `container.perform`. The actor
/// keeps the lifecycle around it: container guard, memory cap, speculative
/// prefill preemption, parameter conversion, tool canonicalisation.
nonisolated enum RawGenerationStart {

    static func start(
        session: any ModelSession,
        prompt: RawGenerationPrompt,
        tools: [ToolSpec]?,
        parameters: GenerateParameters,
        modelFingerprint: String?,
        progressHandler: ServerInferenceProgressHandler?
    ) async throws -> HTTPServerRawGenerationStart {
        await progressHandler?(.cacheLookupStarted)
        let lookupStarted = Date.timeIntervalSinceReferenceDate
        let prepared = try await tokenize(
            prompt, session: session, modelFingerprint: modelFingerprint)
        let lookupMs = (Date.timeIntervalSinceReferenceDate - lookupStarted) * 1000
        // Sequence length is always the LAST dim: `[seq]` on the LLM
        // families, `[batch, seq]` on the vision containers.
        let promptTokenCount = prepared.text.tokens.dim(-1)
        await progressHandler?(
            .cacheLookupFinished(
                .init(
                    reason: prompt.lookupReason,
                    cachedTokens: 0,
                    sharedPrefixLength: 0,
                    promptTokens: promptTokenCount,
                    newTokensToPrefill: promptTokenCount,
                    lookupMs: lookupMs,
                    restoreMs: 0
                )))
        var prefill = ServerInferenceProgressEvent.PrefillInfo(
            promptTokens: promptTokenCount,
            cachedTokens: 0,
            newTokensToPrefill: promptTokenCount,
            prefillMs: nil
        )
        await progressHandler?(.prefillStarted(prefill))
        let prefillStarted = Date.timeIntervalSinceReferenceDate

        // DFlash2 speculative arm: a text-only prompt on a session whose
        // drafter pairs with the loaded target (the drafter is only loaded
        // when it pairs) decodes through the block-parallel speculative
        // iterator — its init runs the capture-emitting chunked prefill
        // itself. Sampling presets speculate identically (the draft carries
        // a selector for rejection sampling).
        //
        // `prefillMs` is stamped right after the iterator build, before the
        // MainActor round trips of the badge and the loop start — the number
        // measures the model, not the renderer.
        let loop: (AsyncStream<RawGeneration>, Task<Void, Never>)
        var engagedArm: SpeculativeArm?
        if DFlash2Support.shouldEngageRawArm(
            hasDrafter: session.dflash2Drafter != nil, input: prepared)
        {
            let specParams = DFlash2Support.rawArmParameters(parameters)
            let cache = try session.newCache(parameters: specParams)
            let iterator = try session.makeDFlash2DecodeIterator(
                prepared, cache: cache, prefilledPrefixTokens: 0, parameters: specParams)
            prefill.prefillMs = (Date.timeIntervalSinceReferenceDate - prefillStarted) * 1000
            engagedArm = .dflash2
            loop = TokenGenerationLoop.start(
                promptTokenCount: promptTokenCount,
                modelConfiguration: session.configuration,
                tokenizer: session.tokenizer,
                iterator: iterator,
                tools: tools
            )
        } else {
            let iterator = try session.makeRawDecodeIterator(prepared, parameters: parameters)
            prefill.prefillMs = (Date.timeIntervalSinceReferenceDate - prefillStarted) * 1000
            loop = TokenGenerationLoop.start(
                promptTokenCount: promptTokenCount,
                modelConfiguration: session.configuration,
                tokenizer: session.tokenizer,
                iterator: iterator,
                tools: tools
            )
        }
        if let engagedArm {
            await progressHandler?(.speculationEngaged(engagedArm))
        }
        await progressHandler?(.prefillFinished(prefill))
        let (stream, completion) = loop
        return HTTPServerRawGenerationStart(
            stream: stream,
            cancel: { completion.cancel() },
            waitForCompletion: { await completion.value }
        )
    }

    /// Prepare the prompt through the session’s agent-edge verb.
    private static func tokenize(
        _ prompt: RawGenerationPrompt,
        session: any ModelSession,
        modelFingerprint: String?
    ) async throws -> LMInput {
        switch prompt {
        case .fresh(let input):
            return try await session.prepareText(input, modelFingerprint: modelFingerprint)
        }
    }
}
