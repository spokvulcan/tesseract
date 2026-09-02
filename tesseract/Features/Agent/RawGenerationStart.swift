import Foundation
import MLX
import MLXLMCommon

/// The prompt shape one **Raw Generation Start** begins from: a fresh
/// prompt, or a thinking-safeguard continuation extending an in-progress
/// assistant turn with a hand-off suffix.
///
/// The three arms `LLMActor` used to carry — agent chat, the server's
/// token-continuation, the agent's input-continuation — are the three
/// cases of one value; the module's script is the same for all of them.
nonisolated enum RawGenerationPrompt {
    /// A whole prompt tokenized from zero (the agent chat turn).
    case fresh(UserInput)
    /// The safeguard continuation: `base` re-tokenized (or already
    /// tokenized), extended with `handoff` encoded as plain text
    /// (`addSpecialTokens: false`) — the original chat-template prompt
    /// already ends inside the assistant turn, so the appended tokens
    /// extend it rather than open a new one.
    case continuation(base: Base, handoff: String)

    nonisolated enum Base {
        /// The fully-tokenized original prompt, captured during the server
        /// path's prefill — no re-tokenize needed.
        case tokens([Int], ndim: Int)
        /// The original `UserInput`; tokenized here through the session's
        /// one tokenize authority.
        case input(UserInput)
    }
}

/// **Raw Generation Start** (CONTEXT.md → Server completion; ADR-0016
/// amendment): the one script that starts a whole-prompt-from-zero
/// generation over a **Model Session** — tokenize through the session's one
/// authority, emit the lookup and prefill progress events, engage the DFlash2
/// raw arm when the session pairs a drafter and the prompt is text-only, else
/// run the **Prefill Strategy** route, start the token-event loop, and wrap
/// the handles. Every prompt shape — fresh or continuation — runs the same
/// script, so the speculation badge, the tokenize authority, and the handle
/// wrap exist once.
///
/// **Metal-affinity contract:** must run inside a session (`withSession`),
/// exactly as the arms it replaced ran inside `container.perform`. The actor
/// keeps the lifecycle around it: container guard, memory cap, speculative
/// prefill preemption, parameter conversion, tool canonicalisation.
nonisolated enum RawGenerationStart {

    /// The cache-lookup reason the progress event reports: raw starts never
    /// consult the prefix cache.
    static let freshLookupReason = "standardGenerationNoPrefixCache"
    static let continuationLookupReason = "thinkingContinuationNoPrefixCache"

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
        let prepared: LMInput
        let lookupReason: String
        switch prompt {
        case .fresh(let input):
            prepared = try await session.prepareText(input, modelFingerprint: modelFingerprint)
            lookupReason = freshLookupReason
        case .continuation(let base, let handoff):
            let originalTokens: [Int]
            let tokenNDim: Int
            switch base {
            case .tokens(let tokens, let ndim):
                originalTokens = tokens
                tokenNDim = ndim
            case .input(let input):
                let basePrepared = try await session.prepareText(
                    input, modelFingerprint: modelFingerprint)
                originalTokens = LLMActor.extractTokenSequence(basePrepared.text.tokens)
                tokenNDim = basePrepared.text.tokens.ndim
            }
            let appendedIDs = try session.tokenizer.encode(text: handoff, addSpecialTokens: false)
            let flat = MLXArray((originalTokens + appendedIDs).map { Int32($0) })
            let tokenArr: MLXArray = tokenNDim >= 2 ? flat.expandedDimensions(axis: 0) : flat
            prepared = LMInput(text: LMInput.Text(tokens: tokenArr, mask: nil))
            lookupReason = continuationLookupReason
        }
        let lookupMs = (Date.timeIntervalSinceReferenceDate - lookupStarted) * 1000
        let promptTokenCount = prepared.text.tokens.dim(-1)
        await progressHandler?(
            .cacheLookupFinished(
                .init(
                    reason: lookupReason,
                    cachedTokens: 0,
                    sharedPrefixLength: 0,
                    promptTokens: promptTokenCount,
                    newTokensToPrefill: promptTokenCount,
                    lookupMs: lookupMs,
                    restoreMs: 0
                )))
        let prefillInfo = ServerInferenceProgressEvent.PrefillInfo(
            promptTokens: promptTokenCount,
            cachedTokens: 0,
            newTokensToPrefill: promptTokenCount,
            prefillMs: nil
        )
        await progressHandler?(.prefillStarted(prefillInfo))
        let prefillStarted = Date.timeIntervalSinceReferenceDate

        // DFlash2 speculative arm: a text-only prompt on a session whose
        // drafter pairs with the loaded target (the drafter is only loaded
        // when it pairs) decodes through the block-parallel speculative
        // iterator — its init runs the capture-emitting chunked prefill
        // itself. Sampling presets speculate identically (the draft carries
        // a selector for rejection sampling). `kvBits` is cleared because
        // speculation rewinds verify rows in place and needs trimmable
        // caches. Without this arm an intervened turn's continuation would
        // decode autoregressively while the drafter sits loaded (~19 vs
        // ~33 tok/s observed on the 27B).
        let start: HTTPServerRawGenerationStart
        if DFlash2Support.shouldEngageRawArm(
            hasDrafter: session.dflash2Drafter != nil, input: prepared)
        {
            var specParams = parameters
            specParams.kvBits = nil
            let cache = try session.newCache(parameters: specParams)
            let iterator = try session.makeDFlash2DecodeIterator(
                prepared, cache: cache, prefilledPrefixTokens: 0, parameters: specParams)
            await progressHandler?(.speculationEngaged(.dflash2))
            start = wrap(
                TokenGenerationLoop.start(
                    promptTokenCount: promptTokenCount,
                    modelConfiguration: session.configuration,
                    tokenizer: session.tokenizer,
                    iterator: iterator,
                    tools: tools
                ))
        } else {
            let iterator = try session.makeRawDecodeIterator(prepared, parameters: parameters)
            start = wrap(
                TokenGenerationLoop.start(
                    promptTokenCount: promptTokenCount,
                    modelConfiguration: session.configuration,
                    tokenizer: session.tokenizer,
                    iterator: iterator,
                    tools: tools
                ))
        }
        let prefillMs = (Date.timeIntervalSinceReferenceDate - prefillStarted) * 1000
        await progressHandler?(
            .prefillFinished(
                .init(
                    promptTokens: promptTokenCount,
                    cachedTokens: 0,
                    newTokensToPrefill: promptTokenCount,
                    prefillMs: prefillMs
                )))
        return start
    }

    /// The shared tail: the loop's stream and mapping task as the start
    /// value every caller receives.
    private static func wrap(
        _ loop: (AsyncStream<RawGeneration>, Task<Void, Never>)
    ) -> HTTPServerRawGenerationStart {
        let (stream, completion) = loop
        return HTTPServerRawGenerationStart(
            stream: stream,
            cancel: { completion.cancel() },
            waitForCompletion: { await completion.value }
        )
    }
}
