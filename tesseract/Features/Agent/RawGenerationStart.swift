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
        /// agent-edge verb.
        case input(UserInput)
    }

    /// The cache-lookup reason the progress event reports: raw starts never
    /// consult the prefix cache, and the reason names which shape skipped it.
    var lookupReason: String {
        switch self {
        case .fresh: "standardGenerationNoPrefixCache"
        case .continuation: "thinkingContinuationNoPrefixCache"
        }
    }
}

/// **Raw Generation Start** (CONTEXT.md → Server completion; ADR-0016
/// amendment): the one script that starts a whole-prompt-from-zero
/// generation over a **Model Session** — tokenize through the session's
/// agent-edge verb, emit the lookup and prefill progress events, engage the
/// DFlash2 raw arm when the session pairs a drafter and the prompt is
/// text-only, else run the **Prefill Strategy** route, start the token-event
/// loop, and wrap the handles. Every prompt shape — fresh or continuation —
/// runs the same script, so the speculation badge, the tokenize step, and
/// the handle wrap exist once.
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
        // a selector for rejection sampling). Without this arm an intervened
        // turn's continuation would decode autoregressively while the
        // drafter sits loaded (~19 vs ~33 tok/s observed on the 27B).
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

    /// The prompt as the model input it prefills from: a fresh prompt
    /// through the session's agent-edge verb; a continuation as the base
    /// token list (captured, or re-tokenized through the same verb)
    /// extended with the hand-off suffix.
    private static func tokenize(
        _ prompt: RawGenerationPrompt,
        session: any ModelSession,
        modelFingerprint: String?
    ) async throws -> LMInput {
        switch prompt {
        case .fresh(let input):
            return try await session.prepareText(input, modelFingerprint: modelFingerprint)
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
            // Rebuild at the original rank: the vision containers index
            // `inputIds.dim(1)` (Qwen3.5's rope index) and need `[batch,
            // seq]`, while the LLM families add the batch axis themselves —
            // the same rule the **Prefill Strategy** routes on.
            let tokenArr: MLXArray = tokenNDim >= 2 ? flat.expandedDimensions(axis: 0) : flat
            return LMInput(text: LMInput.Text(tokens: tokenArr, mask: nil))
        }
    }
}
