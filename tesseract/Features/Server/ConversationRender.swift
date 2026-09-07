//
//  ConversationRender.swift
//  tesseract
//
//  The **Conversation Render** contract (CONTEXT.md) as a module: the one
//  home for token-only rendering — family message-forming plus chat-template
//  application, no pixel work — shared by the request edge, the planner's
//  last-user re-render, the leaf store's stored-conversation measure, the
//  admission builder's continuation probes, its cache-free future-shared-
//  prefix probe pair, and the stable-prefix detector's probe pair. Each verb
//  owns the whole choreography its call sites used to repeat by hand:
//  cache-eligibility, the **Render+Token Cache** resolve, the
//  `applyChatTemplate` fallback, and the `add_generation_prompt: false`
//  merged-context derivation. Since ticket #473 (issue #471) no server
//  source outside the module — this file and its **Render+Token Cache**,
//  the resolve arm below it — applies the chat template; a source-shape
//  test pins it, so the **Emitted Path Resolve** (ADR-0063) is added here
//  once and reaches every server spelling. What stays outside, by design:
//  the processor `prepare` a bypassing request falls back to (the vendor
//  and in-tree PARO input processors, which ADR-0063 decision 5 narrows in
//  #475), the planner's generation-prompt measure, and the agent hand-off
//  suffix — the last two plain-text encodes past the last end-of-turn marker.
//
//  Emitted Path Resolve (tickets #475/#476): every render that produces
//  bytes — the cache's resolves and the split render+encode the fallbacks
//  run — hands those bytes to `EmittedPathResolve.compose` against the
//  **Emitted Path Index**, and the verb serves the composition on a hit:
//  the fed ids for every registered turn, the canonical encode only for the
//  bytes after the deepest hit. A miss serves the canonical tokens. Every
//  spelling of one history serves the same ids, so a planner boundary
//  measured on one is an offset into another. The index is consulted only
//  under an engaged fingerprint, so an image-bearing (sealed non-identity)
//  render, an unknown fingerprint, and the uncached replay renders never
//  touch it; the request edge logs that skip once per request. The leaf
//  store's fast path (`storedRenderBytes`) renders to bytes only and never
//  resolves: the live leaf is stored under the ids that were fed.
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

    /// The **Emitted Path Index** every byte-producing render resolves
    /// against, or `nil` for renders that never consult it (the uncached
    /// replay/benchmark/test renders).
    private let emittedPathIndex: EmittedPathIndex?

    /// The fingerprint the index is scoped to. Equal to `cacheFingerprint`
    /// at the request edge; independent for an uncached render that learns
    /// a harness-local index. Cleared with the cache fingerprint by
    /// `sealed(for:)`.
    private(set) var emittedPathFingerprint: String?

    /// The request's account of every resolve — a reference, so the copies
    /// the phases make all write to one record.
    let emittedPathTelemetry: EmittedPathRequestTelemetry?

    /// Why the render bypasses the cache and the index, when it does.
    private(set) var ineligibility: Ineligibility?

    enum Ineligibility: String, Sendable {
        case media
        case nonFlatTokens
        case unknownFingerprint
        case nonIdentityKeySpace
        case uncached
    }

    /// Tokens plus the bytes they encode, when the render produced bytes
    /// (a `ChatTemplateRendering` tokenizer; `nil` under the fused fallback).
    struct Rendered: Sendable {
        let tokens: [Int]
        let bytes: [UInt8]?
    }

    // MARK: - Construction (the eligibility decision)

    /// The one spelling of the eligibility predicate, shared by the request
    /// edge and the agent edge: engage the cache only for a media-free
    /// request on a model whose processor emits a flat 1-D token list, and
    /// only under a known fingerprint.
    private static func eligibility(
        hasMedia: Bool,
        producesFlatTextTokens: Bool,
        modelFingerprint: String?
    ) -> (fingerprint: String?, ineligibility: Ineligibility?) {
        if hasMedia { return (nil, .media) }
        if !producesFlatTextTokens { return (nil, .nonFlatTokens) }
        guard let modelFingerprint else { return (nil, .unknownFingerprint) }
        return (modelFingerprint, nil)
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
    ///
    /// `diagnostics` is the request's diagnostics net for the Emitted Path
    /// events; `nil` sends them to the server log.
    static func forTextOnlyRequest(
        tokenizer: any Tokenizer,
        toolSpecs: [ToolSpec]?,
        renderContext: TemplateRenderContext,
        hasMedia: Bool,
        producesFlatTextTokens: Bool,
        modelFingerprint: String?,
        cache: RenderTokenCache = .shared,
        emittedPathIndex: EmittedPathIndex? = .shared,
        diagnostics: PrefixCacheDiagnostics.Context? = nil
    ) -> ConversationRender {
        let eligibility = eligibility(
            hasMedia: hasMedia,
            producesFlatTextTokens: producesFlatTextTokens,
            modelFingerprint: modelFingerprint
        )
        return ConversationRender(
            tokenizer: tokenizer,
            toolSpecs: toolSpecs,
            renderContext: renderContext,
            cacheFingerprint: eligibility.fingerprint,
            baseRenderTokens: nil,
            cache: cache,
            emittedPathIndex: emittedPathIndex,
            emittedPathFingerprint: emittedPathIndex == nil ? nil : eligibility.fingerprint,
            emittedPathTelemetry: emittedPathIndex == nil
                ? nil : EmittedPathRequestTelemetry(diagnostics: diagnostics),
            ineligibility: eligibility.ineligibility
        )
    }

    /// A render that always runs in full — no cache participation, by
    /// construction rather than by answering the eligibility questions with
    /// literals. For sites with no request instance behind them: replay
    /// telemetry probes, benchmarks, tests. An `emittedPathIndex` with its
    /// `fingerprint` lets an offline harness learn and resolve a private
    /// index through the same verbs; the default consults none.
    static func uncached(
        tokenizer: any Tokenizer,
        toolSpecs: [ToolSpec]? = nil,
        renderContext: TemplateRenderContext = .canonical,
        emittedPathIndex: EmittedPathIndex? = nil,
        emittedPathFingerprint: String? = nil,
        emittedPathTelemetry: EmittedPathRequestTelemetry? = nil
    ) -> ConversationRender {
        ConversationRender(
            tokenizer: tokenizer,
            toolSpecs: toolSpecs,
            renderContext: renderContext,
            cacheFingerprint: nil,
            baseRenderTokens: nil,
            cache: .shared,
            emittedPathIndex: emittedPathIndex,
            emittedPathFingerprint: emittedPathIndex == nil ? nil : emittedPathFingerprint,
            emittedPathTelemetry: emittedPathTelemetry,
            ineligibility: .uncached
        )
    }

    /// Seal the edge-constructed value with the settled **Cache Key Space**:
    /// cache resolves stay engaged only on an identity (text-only) space.
    /// Image-bearing key spaces need the real token list for their
    /// placeholder runs, so they always render in full — and never consult
    /// the Emitted Path Index, whose paths are fed ids.
    func sealed(for keySpace: CacheKeySpace) -> ConversationRender {
        guard !keySpace.isIdentity else { return self }
        var copy = self
        copy.cacheFingerprint = nil
        copy.emittedPathFingerprint = nil
        copy.ineligibility = .nonIdentityKeySpace
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
        guard let cacheFingerprint else {
            emittedPathTelemetry?.recordSkip(
                spelling: .request, reason: ineligibility?.rawValue ?? "bypass")
            return nil
        }
        guard
            let resolution = Self.resolveFull(
                cache: cache,
                tokenizer: tokenizer,
                messages: messages(),
                tools: toolSpecs,
                additionalContext: renderContext.additionalContext(),
                fingerprint: cacheFingerprint
            )
        else {
            emittedPathTelemetry?.recordSkip(spelling: .request, reason: "renderFallback")
            return nil
        }
        return serve(
            spelling: .request,
            rendered: Rendered(tokens: resolution.tokens, bytes: resolution.renderedBytes)
        ).tokens
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
        try trimRecoveredRender(
            messages: messages, messagesAreEntryPrefix: true, spelling: .lastUserPrefix
        ).tokens
    }

    /// A continuation of the cached conversation, rendered without a
    /// generation prompt (C28): the stored conversation the leaf store
    /// measures, and the admission builder's continuation probes — one verb,
    /// because they are the identical computation. Recovered as a verified
    /// trim+extension of the request-edge entry; any inexactness falls back
    /// to the full `applyChatTemplate`. Throws only what the fallback render
    /// throws.
    func continuationRender(messages: [[String: any Sendable]]) throws -> [Int] {
        try storedRender(messages: messages).tokens
    }

    /// The leaf store's boundary-path spelling of `continuationRender`: the
    /// same ladder, returning the render bytes too (the tokens served, the
    /// bytes as rendered).
    func storedRender(messages: [[String: any Sendable]]) throws -> Rendered {
        try trimRecoveredRender(
            messages: messages, messagesAreEntryPrefix: false, spelling: .continuation)
    }

    /// The leaf store's fast-path render (ADR-0063 decision 10): the stored
    /// conversation to bytes, without a generation prompt — one Jinja
    /// render, no tokenization, no cache, no index resolve. The Emitted
    /// Path registration keys on these bytes through their last end-of-turn
    /// marker; nothing else on the fast path needs the render. `nil` when
    /// the tokenizer cannot render to bytes (the fused fallback), which the
    /// registration reports as `renderUnavailable`. Throws only what the
    /// template render throws.
    func storedRenderBytes(messages: [[String: any Sendable]]) throws -> [UInt8]? {
        try Self.renderText(
            tokenizer: tokenizer,
            messages: messages,
            tools: toolSpecs,
            additionalContext: renderContext.additionalContext(
                merging: ["add_generation_prompt": false]
            )
        ).map { Array($0.utf8) }
    }

    /// The base (stored) conversation's render: the C31 plumbed tokens when
    /// the leaf store already computed them this request, else the same
    /// continuation render the plumbing short-circuits.
    func baseRender(messages: [[String: any Sendable]]) throws -> [Int] {
        if let baseRenderTokens { return baseRenderTokens }
        return try trimRecoveredRender(
            messages: messages, messagesAreEntryPrefix: false, spelling: .base
        ).tokens
    }

    /// The shared no-generation-prompt ladder behind the C27 and C28 verbs:
    /// merged-context derivation, the trim-recovery resolve (against the
    /// request-edge entry; `messagesAreEntryPrefix` selects the C27
    /// truncated form over the C28 tail-replacement), and the
    /// `applyChatTemplate` fallback — one spelling, so the two verbs cannot
    /// drift apart. Every rung produces the render bytes the Emitted Path
    /// Resolve serves from.
    private func trimRecoveredRender(
        messages: [[String: any Sendable]],
        messagesAreEntryPrefix: Bool,
        spelling: EmittedPathRequestTelemetry.Spelling
    ) throws -> Rendered {
        let merged = renderContext.additionalContext(
            merging: ["add_generation_prompt": false]
        )
        var recovered: Rendered?
        if let cacheFingerprint {
            let resolved: RenderTokenCache.Recovered? =
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
            recovered = resolved.map { Rendered(tokens: $0.tokens, bytes: $0.renderedBytes) }
        }
        let rendered =
            try recovered
            ?? Self.applyTemplate(
                tokenizer: tokenizer,
                messages: messages,
                tools: toolSpecs,
                additionalContext: merged
            )
        return serve(spelling: spelling, rendered: rendered)
    }

    // MARK: - Cache-free probe renders

    /// The admission builder's future-shared-prefix probe render: the stored
    /// conversation plus one synthetic continuation, rendered without a
    /// generation prompt under the request's ingredients — the same
    /// computation as `continuationRender`, deliberately CACHE-FREE. The
    /// probe runs detached from the speculative pass and is cancelled on
    /// preemption; its caller checks cancellation between the two renders,
    /// bounding abandoned work to one render, and a resolve against the live
    /// entry could neither observe those checks nor be abandoned mid-render.
    /// Never counts against the Render+Token Cache's telemetry. The Emitted
    /// Path Resolve still serves from its bytes (it is a lookup, not a
    /// store), so the probe's tokens are offsets into the same ids the
    /// request edge fed.
    func uncachedContinuationRender(messages: [[String: any Sendable]]) throws -> [Int] {
        let rendered = try Self.applyTemplate(
            tokenizer: tokenizer,
            messages: messages,
            tools: toolSpecs,
            additionalContext: renderContext.additionalContext(
                merging: ["add_generation_prompt": false]
            )
        )
        return serve(spelling: .admissionProbe, rendered: rendered).tokens
    }

    /// The stable-prefix detector's probe render, from raw ingredients: the
    /// detector has no request value (it is memoized on exactly these
    /// ingredients and benchmarked standalone), so this is the narrow static
    /// entry over the module's one template application, parallel to
    /// `agentEdgeFullRender`. `additionalContext` passes through verbatim —
    /// the detector renders its system+user probes under the request's base
    /// context, generation prompt included, exactly as before. No index
    /// resolve: the probes carry no assistant turn to find.
    static func stablePrefixProbeRender(
        tokenizer: any Tokenizer,
        messages: [[String: any Sendable]],
        tools: [ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        try applyTemplate(
            tokenizer: tokenizer,
            messages: messages,
            tools: tools,
            additionalContext: additionalContext
        ).tokens
    }

    // MARK: - The one template application

    /// The single spelling of the chat-template application inside the
    /// module. Every render that runs in full bottoms out here — the C27/C28
    /// fallback and both probe verbs; the cached paths' miss render lives in
    /// `RenderTokenCache`, the module's resolve arm. Split into render-to-
    /// bytes and encode when the tokenizer renders (byte-exact with the
    /// fused call — the cache's miss path has served that split since C25),
    /// so the bytes reach the Emitted Path Resolve; the fused call remains
    /// for tokenizers that cannot render.
    private static func applyTemplate(
        tokenizer: any Tokenizer,
        messages: [[String: any Sendable]],
        tools: [ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> Rendered {
        guard
            let rendered = try renderText(
                tokenizer: tokenizer,
                messages: messages,
                tools: tools,
                additionalContext: additionalContext
            )
        else {
            return Rendered(
                tokens: try tokenizer.applyChatTemplate(
                    messages: messages,
                    tools: tools,
                    additionalContext: additionalContext
                ),
                bytes: nil)
        }
        return Rendered(
            tokens: tokenizer.encode(text: rendered, addSpecialTokens: false),
            bytes: Array(rendered.utf8))
    }

    /// The module's render-to-text rung: the template applied by a rendering
    /// tokenizer, without encoding — `nil` for a tokenizer that cannot render
    /// (the fused fallback), which `applyTemplate` sends to the fused call.
    private static func renderText(
        tokenizer: any Tokenizer,
        messages: [[String: any Sendable]],
        tools: [ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> String? {
        guard let rendering = tokenizer as? any ChatTemplateRendering else { return nil }
        return try rendering.renderChatTemplate(
            messages: messages,
            tools: tools,
            additionalContext: additionalContext
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
    ) -> RenderTokenCache.Resolution? {
        try? cache.resolve(
            tokenizer: tokenizer,
            messages: messages,
            tools: tools,
            additionalContext: additionalContext,
            modelFingerprint: fingerprint
        )
    }

    // MARK: - The Emitted Path Resolve

    /// The end-of-turn marker the index keys on for this render's model,
    /// derived once per fingerprint from the module's probe renders and
    /// remembered until the fingerprint changes. `nil` when the marker is
    /// unavailable — see `endOfTurnMarkerStatus` for the reason.
    static func endOfTurnMarker(
        index: EmittedPathIndex,
        fingerprint: String,
        tokenizer: any Tokenizer
    ) -> EndOfTurnMarker? {
        endOfTurnMarkerStatus(index: index, fingerprint: fingerprint, tokenizer: tokenizer).marker
    }

    /// The marker's status for the fingerprint. Derivation renders one user
    /// message and one assistant message with a sentinel content, without a
    /// generation prompt, so the bytes after the sentinel are the template's
    /// assistant-turn tail (`EndOfTurnMarker.derive`); then a three-message
    /// probe — a user turn after the assistant's — checks that the marker is
    /// a hard encoding boundary at every occurrence
    /// (`EndOfTurnMarker.splitsEncoding`), the equality the served
    /// composition rests on. A tokenizer that fails it is refused for good:
    /// no path registers, no resolve runs, and the registration reports
    /// `suffixEncodeUnstable`.
    static func endOfTurnMarkerStatus(
        index: EmittedPathIndex,
        fingerprint: String,
        tokenizer: any Tokenizer
    ) -> EndOfTurnMarkerStatus {
        index.endOfTurnMarker(fingerprint: fingerprint) {
            guard let rendering = tokenizer as? any ChatTemplateRendering else {
                return .unavailable(.noEndOfTurnMarker)
            }
            let user: [String: any Sendable] = ["role": "user", "content": "probe"]
            let assistant: [String: any Sendable] = [
                "role": "assistant", "content": EndOfTurnMarker.probeContent,
            ]
            let noGenerationPrompt: [String: any Sendable] = ["add_generation_prompt": false]
            guard
                let probe = try? rendering.renderChatTemplate(
                    messages: [user, assistant], tools: nil,
                    additionalContext: noGenerationPrompt),
                let marker = EndOfTurnMarker.derive(probeRender: probe, tokenizer: tokenizer)
            else { return .unavailable(.noEndOfTurnMarker) }
            guard
                let splitProbe = try? rendering.renderChatTemplate(
                    messages: [user, assistant, user], tools: nil,
                    additionalContext: noGenerationPrompt),
                EndOfTurnMarker.splitsEncoding(
                    of: Array(splitProbe.utf8), marker: marker.bytes, tokenizer: tokenizer)
            else { return .unavailable(.suffixEncodeUnstable) }
            return .available(marker)
        }
    }

    /// How the registration side reaches the index: the engaged index, its
    /// fingerprint and the model's marker — or why this render never
    /// consults the index.
    enum EmittedPathEligibility {
        case eligible(index: EmittedPathIndex, fingerprint: String, marker: EndOfTurnMarker)
        case ineligible(reason: String)
    }

    func emittedPathEligibility() -> EmittedPathEligibility {
        guard let index = emittedPathIndex, let fingerprint = emittedPathFingerprint else {
            return .ineligible(reason: ineligibility?.rawValue ?? "noIndex")
        }
        switch Self.endOfTurnMarkerStatus(
            index: index, fingerprint: fingerprint, tokenizer: tokenizer)
        {
        case .available(let marker):
            return .eligible(index: index, fingerprint: fingerprint, marker: marker)
        case .unavailable(let reason):
            return .ineligible(reason: reason.rawValue)
        }
    }

    /// Serve a render: the composition against the index on a hit, the
    /// canonical tokens otherwise — when the render produced no bytes, no
    /// index is engaged, the marker is unavailable, or nothing registered
    /// for the history.
    private func serve(
        spelling: EmittedPathRequestTelemetry.Spelling,
        rendered: Rendered
    ) -> Rendered {
        guard let bytes = rendered.bytes, let emittedPathIndex, let emittedPathFingerprint else {
            return rendered
        }
        return Rendered(
            tokens: Self.serve(
                index: emittedPathIndex, fingerprint: emittedPathFingerprint,
                tokenizer: tokenizer, telemetry: emittedPathTelemetry, spelling: spelling,
                renderedBytes: bytes, canonical: rendered.tokens),
            bytes: bytes)
    }

    private static func serve(
        index: EmittedPathIndex,
        fingerprint: String,
        tokenizer: any Tokenizer,
        telemetry: EmittedPathRequestTelemetry?,
        spelling: EmittedPathRequestTelemetry.Spelling,
        renderedBytes: [UInt8],
        canonical: [Int]
    ) -> [Int] {
        guard
            let marker = endOfTurnMarker(
                index: index, fingerprint: fingerprint, tokenizer: tokenizer)
        else { return canonical }
        let start = Date.timeIntervalSinceReferenceDate
        let composition = EmittedPathResolve.compose(
            index: index, fingerprint: fingerprint, marker: marker,
            renderedBytes: renderedBytes, tokenizer: tokenizer)
        let served: [Int]
        switch composition {
        case .indexed(let tokens, _, _, _, _): served = tokens
        case .miss: served = canonical
        }
        (telemetry ?? EmittedPathRequestTelemetry(diagnostics: nil)).record(
            spelling: spelling,
            composition: composition,
            tokens: served.count,
            resolveSeconds: Date.timeIntervalSinceReferenceDate - start)
        return served
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
        cache: RenderTokenCache = .shared,
        emittedPathIndex: EmittedPathIndex? = .shared
    ) -> [Int]? {
        guard
            let fingerprint = eligibility(
                hasMedia: hasMedia,
                producesFlatTextTokens: producesFlatTextTokens,
                modelFingerprint: modelFingerprint
            ).fingerprint
        else { return nil }
        guard let messages = messages() else { return nil }
        guard
            let resolution = resolveFull(
                cache: cache,
                tokenizer: tokenizer,
                messages: messages,
                tools: tools,
                additionalContext: additionalContext,
                fingerprint: fingerprint
            )
        else { return nil }
        guard let emittedPathIndex else { return resolution.tokens }
        return serve(
            index: emittedPathIndex, fingerprint: fingerprint, tokenizer: tokenizer,
            telemetry: nil, spelling: .agentEdge, renderedBytes: resolution.renderedBytes,
            canonical: resolution.tokens)
    }
}
