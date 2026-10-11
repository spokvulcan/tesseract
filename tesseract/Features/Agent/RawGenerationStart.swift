import Foundation
import MLX
import MLXLMCommon

/// A whole prompt tokenized from zero for a raw generation, and the render
/// context its template renders under: the input's `additionalContext`, as
/// the typed value the **Generation Prompt** probe is remembered by.
nonisolated enum RawGenerationPrompt {
    case fresh(UserInput, renderContext: TemplateRenderContext)

    var lookupReason: String { "standardGenerationNoPrefixCache" }

    var renderContext: TemplateRenderContext {
        switch self {
        case .fresh(_, let renderContext): renderContext
        }
    }
}

/// **Raw Generation Start** (CONTEXT.md → Server completion; ADR-0016
/// amendment): the one script that starts a whole-prompt-from-zero
/// generation over a **Model Session** — tokenize through the session's
/// agent-edge verb, emit the lookup and prefill progress events, decode
/// through the session's **Speculation Plan** when there is one, else run
/// the **Prefill Strategy** route, start the token-event loop, and wrap the
/// handles. Every raw start runs the same script, so the speculation badge,
/// tokenize step, and handle wrap exist once.
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
        // The start that tokenized the prompt reports its Generation Prompt
        // (ADR-0070): the template's probe for this render context, checked
        // against the tokens this prompt fed.
        let generationPrompt = ConversationRender.checkedGenerationPrompt(
            tokenizer: session.tokenizer, renderContext: prompt.renderContext,
            modelFingerprint: modelFingerprint,
            fed: LLMActor.extractTokenSequence(prepared.text.tokens), diagnostics: nil)
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

        // The Speculation Plan (ADR-0079): the whole prompt from zero, so
        // nothing is restored and no leaf is stored. A plan's iterator runs
        // the whole prompt's prefill in its init — DFlash2's own chunked
        // capture prefill — over a fresh cache. A prompt with images (a tool
        // result's screenshot) has the vision target prefill through its last
        // image first, and the iterator takes the text after it (ADR-0089).
        // Without a plan the Prefill Strategy route prefills into a cache of
        // its own.
        //
        // `prefillMs` is stamped right after the iterator build, before the
        // MainActor round trips of the badge and the loop start — the number
        // measures the model, not the renderer.
        let speculation = session.speculation.plan(
            for: SpeculationRequest(
                isTextOnly: prepared.image == nil && prepared.video == nil
                    && prepared.audio == nil,
                kvBits: parameters.kvBits,
                kvScheme: KVScheme(parameters),
                temperature: parameters.temperature,
                promptTokens: promptTokenCount,
                restoresPrefix: false,
                storedLeaf: nil
            ))
        // No reference outlives the Decode Start's conversion: it hands back
        // the array the iterator advances.
        var cache: [any KVCache] = []
        let handover: DecodeHandover
        if let speculation {
            cache = try session.newCache(parameters: parameters)
            for layer in cache { layer.reserveCapacity(promptTokenCount) }
            handover = .speculative(
                speculation, input: prepared, prefilledPrefixTokens: 0, positionDelta: 0)
        } else {
            handover = .prefillStrategy(prepared)
        }
        let decode = try DecodeStart.make(
            handover, cache: &cache, session: session, parameters: parameters)
        prefill.prefillMs = (Date.timeIntervalSinceReferenceDate - prefillStarted) * 1000
        let (stream, completion) = decode.startLoop(
            promptTokenCount: promptTokenCount, session: session, tools: tools,
            generatedTokens: nil)
        await decode.reportPrefillFinished(prefill, to: progressHandler)
        return HTTPServerRawGenerationStart(
            stream: stream,
            generationPrompt: generationPrompt,
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
        case .fresh(let input, _):
            return try await session.prepareText(input, modelFingerprint: modelFingerprint)
        }
    }
}
