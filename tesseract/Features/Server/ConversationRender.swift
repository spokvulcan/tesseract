//
//  ConversationRender.swift
//  tesseract
//
//  The **Conversation Render** contract (CONTEXT.md) as a module: the one
//  home for token-only rendering — family message-forming plus chat-template
//  application, no pixel work — shared by the request edge, the planner's
//  last-user re-render, the leaf store's stored-conversation measure, and the
//  admission builder's continuation probes. Each verb owns the whole
//  choreography its call sites used to repeat by hand: cache-eligibility,
//  the **Render+Token Cache** resolve, the `applyChatTemplate` fallback, and
//  the `add_generation_prompt: false` merged-context derivation.
//
//  Eligibility is decided at construction, from instance truth, once:
//  a `nil` `cacheFingerprint` means "always render+encode in full" — the
//  pre-C25 behavior. The cache never sees a synthetic key: an unknown model
//  fingerprint bypasses rather than sharing a bucket, because the repeat
//  path is the one resolve whose exactness rests on the key alone (a
//  byte-identical render under the same fingerprint returns the cached
//  tokens outright, with no empirical arbiter behind it). This absorbs
//  `RenderTokenSource`, whose predicate had once been five different
//  spellings across the C25–C31 seams (two disagreeing on the
//  unknown-fingerprint case) — and whose *choreography* still was.
//  Issue #439 / PR #449 was exactly that shape failing: one call site wiring
//  the wrong ingredient (the request images instead of the instance-filtered
//  list) into the shared predicate. Construction at the one place instance
//  truth lives makes that class of defect unrepresentable.
//
//  The value is built at the **Request Keying** edge (pre-key-space, from
//  instance facts), *sealed* with the request's **Cache Key Space** once
//  keying settles (image-bearing key spaces need the real token list for
//  their placeholder runs, so a non-identity space clears the fingerprint),
//  and enriched once more by the leaf store with the C31 base render. It
//  rides `HTTPPrefixCacheGeneration` to the post-generation phases.
//
//  Concurrency: holds the request's tokenizer, so it is `@unchecked
//  Sendable` under the same discipline as `HTTPPrefixCacheGeneration`, which
//  carries it — built and consumed on the LLM actor's isolation. One
//  scheduling change from the pre-module call sites: the leaf store's
//  stored-conversation render used to run inside a **Model Session** lease
//  and now runs off it (a CPU-only render no longer serializes against the
//  session), relying on the tokenizer's own thread-safety exactly as the
//  planner's and builder's renders always have.
//

import Foundation
import MLXLMCommon

nonisolated struct ConversationRender: @unchecked Sendable {

    /// The request's tokenizer — exposed because sibling tokenizer-affine
    /// work (`StablePrefixDetector`, the generation-prompt suffix encode)
    /// legitimately shares it; the *render choreography* is what callers must
    /// not re-open.
    let tokenizer: any Tokenizer

    /// The request's canonicalized tool specs — one value for every render
    /// of this request, so a probe render cannot drift from prepare's.
    let toolSpecs: [ToolSpec]?

    /// The request's template render context; base and merged
    /// `additionalContext` both derive from it, inside the verbs.
    let renderContext: TemplateRenderContext

    /// The fingerprint every cache resolve keys under, or `nil` to bypass
    /// the cache and render in full (see the header). Cleared only by
    /// `sealed(for:)`.
    private(set) var cacheFingerprint: String?

    /// The stored (base) conversation's render-space token list, when the
    /// leaf store already computed the identical render this request (C31).
    /// Only ever the base render — a continuation render is a different
    /// conversation and is always computed. Set exclusively by
    /// `carryingBaseRender(_:)`.
    private(set) var baseRenderTokens: [Int]?

    // The `private` cache member keeps the synthesized memberwise init
    // private too: outside construction goes through the eligibility
    // constructors below, never field-by-field.
    private let cache: RenderTokenCache

    // MARK: - Construction (the eligibility decision)

    /// The one spelling of the eligibility predicate, shared by the request
    /// edge and the agent edge: engage the cache only for a media-free
    /// request on a model whose processor emits a flat 1-D token list, and
    /// only under a known fingerprint.
    private static func eligibleFingerprint(
        hasMedia: Bool,
        producesFlatTextTokens: Bool,
        modelFingerprint: String?
    ) -> String? {
        (!hasMedia && producesFlatTextTokens) ? modelFingerprint : nil
    }

    /// The request-edge constructor: engage the cache only for a media-free
    /// request on a model whose processor emits a flat 1-D token list.
    ///
    /// `producesFlatTextTokens` is the DIRECT property the cache path needs —
    /// `LMInput(tokens:)` must reproduce what the processor would build, and
    /// a vision container's text-only `prepare` emits 2D `[batch, seq]`. It
    /// replaced the old `imageKeying == nil` proxy, which asked whether the
    /// app RECOGNIZES a vision container — true of the then-only VLM family
    /// by coincidence, and silently wrong for any VLM family added without
    /// an image-keying rule.
    ///
    /// `hasMedia` must key on the INSTANCE-FILTERED image list (issue #439):
    /// a dropped-image request is text-only by construction.
    static func forTextOnlyRequest(
        tokenizer: any Tokenizer,
        toolSpecs: [ToolSpec]?,
        renderContext: TemplateRenderContext,
        hasMedia: Bool,
        producesFlatTextTokens: Bool,
        modelFingerprint: String?,
        cache: RenderTokenCache = .shared
    ) -> ConversationRender {
        ConversationRender(
            tokenizer: tokenizer,
            toolSpecs: toolSpecs,
            renderContext: renderContext,
            cacheFingerprint: eligibleFingerprint(
                hasMedia: hasMedia,
                producesFlatTextTokens: producesFlatTextTokens,
                modelFingerprint: modelFingerprint
            ),
            baseRenderTokens: nil,
            cache: cache
        )
    }

    /// A render that always runs in full — no cache participation, by
    /// construction rather than by answering the eligibility questions with
    /// literals. For sites with no request instance behind them: replay
    /// telemetry probes, benchmarks, tests.
    static func uncached(
        tokenizer: any Tokenizer,
        toolSpecs: [ToolSpec]? = nil,
        renderContext: TemplateRenderContext = .canonical
    ) -> ConversationRender {
        ConversationRender(
            tokenizer: tokenizer,
            toolSpecs: toolSpecs,
            renderContext: renderContext,
            cacheFingerprint: nil,
            baseRenderTokens: nil,
            cache: .shared
        )
    }

    /// Seal the edge-constructed value with the settled **Cache Key Space**:
    /// cache resolves stay engaged only on an identity (text-only) space.
    /// Image-bearing key spaces need the real token list for their
    /// placeholder runs, so they always render in full.
    func sealed(for keySpace: CacheKeySpace) -> ConversationRender {
        guard !keySpace.isIdentity else { return self }
        var copy = self
        copy.cacheFingerprint = nil
        return copy
    }

    /// The C31 enrichment: carry the stored conversation's just-computed
    /// render so the admission builder's base probe never re-runs the
    /// identical computation. The single sanctioned way to plumb a render
    /// between phases — only ever the base render.
    func carryingBaseRender(_ tokens: [Int]) -> ConversationRender {
        var copy = self
        copy.baseRenderTokens = tokens
        return copy
    }

    // MARK: - Render verbs

    /// The request-edge full render (C25): resolve the whole conversation
    /// through the cache, or `nil` — bypass, cold cache, non-rendering
    /// tokenizer, or any render/encode failure — in which case the caller
    /// falls back to its processor's `prepare`, which reproduces today's
    /// error handling (the missing-template plain-text fallback stays in
    /// the processor). The one verb whose fallback is model-affine and so
    /// stays with the caller. `messages` is an autoclosure so a bypassing
    /// render never pays the message-forming pass.
    func fullRender(messages: @autoclosure () -> [[String: any Sendable]]) -> [Int]? {
        guard let cacheFingerprint else { return nil }
        return Self.resolveFull(
            cache: cache,
            tokenizer: tokenizer,
            messages: messages(),
            tools: toolSpecs,
            additionalContext: renderContext.additionalContext(),
            fingerprint: cacheFingerprint
        )
    }

    /// The planner's last-user boundary render (C27): the conversation
    /// truncated at a message index, rendered without a generation prompt.
    /// Recovered as a verified trim of the entry the request-edge resolve
    /// cached for this same conversation — `messages` are a prompt-message
    /// prefix of it, so the entry's cumulative digest-chain head IS the
    /// truncated chain (a bypassed or fallen-back edge leaves an older
    /// entry, which the render arbiters reject; the assertion is a cost
    /// hint, never a correctness input). Any inexactness falls back to the
    /// full `applyChatTemplate`. Throws only what the fallback render
    /// throws.
    func lastUserPrefixRender(messages: [[String: any Sendable]]) throws -> [Int] {
        try trimRecoveredRender(messages: messages, messagesAreEntryPrefix: true)
    }

    /// A continuation of the cached conversation, rendered without a
    /// generation prompt (C28): the stored conversation the leaf store
    /// measures, and the admission builder's continuation probes — one verb,
    /// because they are the identical computation. Recovered as a verified
    /// trim+extension of the request-edge entry; any inexactness falls back
    /// to the full `applyChatTemplate`. Throws only what the fallback render
    /// throws.
    func continuationRender(messages: [[String: any Sendable]]) throws -> [Int] {
        try trimRecoveredRender(messages: messages, messagesAreEntryPrefix: false)
    }

    /// The base (stored) conversation's render: the C31 plumbed tokens when
    /// the leaf store already computed them this request, else the same
    /// continuation render the plumbing short-circuits.
    func baseRender(messages: [[String: any Sendable]]) throws -> [Int] {
        try baseRenderTokens ?? continuationRender(messages: messages)
    }

    /// The shared no-generation-prompt ladder behind the C27 and C28 verbs:
    /// merged-context derivation, the trim-recovery resolve (against the
    /// request-edge entry; `messagesAreEntryPrefix` selects the C27
    /// truncated form over the C28 tail-replacement), and the
    /// `applyChatTemplate` fallback — one spelling, so the two verbs cannot
    /// drift apart.
    private func trimRecoveredRender(
        messages: [[String: any Sendable]],
        messagesAreEntryPrefix: Bool
    ) throws -> [Int] {
        let merged = renderContext.additionalContext(
            merging: ["add_generation_prompt": false]
        )
        if let cacheFingerprint {
            let resolved: [Int]? =
                messagesAreEntryPrefix
                ? try? cache.resolveTruncated(
                    tokenizer: tokenizer,
                    messages: messages,
                    tools: toolSpecs,
                    baseAdditionalContext: renderContext.additionalContext(),
                    mergedAdditionalContext: merged,
                    modelFingerprint: cacheFingerprint,
                    messagesAreEntryPrefix: true
                )
                : try? cache.resolveReplacingTail(
                    tokenizer: tokenizer,
                    messages: messages,
                    tools: toolSpecs,
                    baseAdditionalContext: renderContext.additionalContext(),
                    mergedAdditionalContext: merged,
                    modelFingerprint: cacheFingerprint
                )
            if let resolved {
                return resolved
            }
        }
        return try tokenizer.applyChatTemplate(
            messages: messages,
            tools: toolSpecs,
            additionalContext: merged
        )
    }

    /// The one spelling of the C25 resolve body, shared by `fullRender` and
    /// the agent edge (whose `additionalContext` is raw, not a
    /// `TemplateRenderContext`). Absorbs every failure into `nil` — the
    /// caller's processor fallback.
    private static func resolveFull(
        cache: RenderTokenCache,
        tokenizer: any Tokenizer,
        messages: [[String: any Sendable]],
        tools: [ToolSpec]?,
        additionalContext: [String: any Sendable]?,
        fingerprint: String
    ) -> [Int]? {
        (try? cache.resolve(
            tokenizer: tokenizer,
            messages: messages,
            tools: tools,
            additionalContext: additionalContext,
            modelFingerprint: fingerprint
        ))?.tokens
    }

    // MARK: - The agent edge

    /// The agent raw-generation edge (`LLMActor`), which has no
    /// conversation value, no key space, and a raw `additionalContext`
    /// instead of a `TemplateRenderContext` — the narrow static entry over
    /// the same eligibility and resolve, so the fifth spelling shares this
    /// home without pretending the contexts match. `nil` sends the caller
    /// to its processor's `prepare`. `messages` is an autoclosure so an
    /// ineligible request (media, 2D tokens, unknown fingerprint) never pays
    /// the message-forming pass; a caller that cannot form messages at all
    /// yields `nil` from it and falls back the same way.
    static func agentEdgeFullRender(
        tokenizer: any Tokenizer,
        messages: @autoclosure () -> [[String: any Sendable]]?,
        tools: [ToolSpec]?,
        additionalContext: [String: any Sendable]?,
        hasMedia: Bool,
        producesFlatTextTokens: Bool,
        modelFingerprint: String?,
        cache: RenderTokenCache = .shared
    ) -> [Int]? {
        guard
            let fingerprint = eligibleFingerprint(
                hasMedia: hasMedia,
                producesFlatTextTokens: producesFlatTextTokens,
                modelFingerprint: modelFingerprint
            )
        else { return nil }
        guard let messages = messages() else { return nil }
        return resolveFull(
            cache: cache,
            tokenizer: tokenizer,
            messages: messages,
            tools: tools,
            additionalContext: additionalContext,
            fingerprint: fingerprint
        )
    }
}
