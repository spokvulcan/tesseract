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
//  #475), and the agent hand-off suffix, a plain-text encode past the last
//  end-of-turn marker. The **Generation Prompt** the planner once spelled
//  and encoded by hand is measured here (ADR-0070).
//
//  Emitted Path Resolve (tickets #475/#476): every render that produces
//  bytes — the cache's resolves and the split render+encode the fallbacks
//  run — hands those bytes to `EmittedPathResolve.compose` against the
//  **Emitted Path Index**, and the verb serves the composition on a hit:
//  the fed ids for every registered turn, the canonical encode only for the
//  bytes after the deepest hit. A miss serves the canonical tokens. Every
//  spelling of one history serves the same ids, so a planner boundary
//  measured on one is an offset into another. The index is consulted only
//  under an engaged fingerprint, so an unknown fingerprint and the uncached
//  replay renders never touch it; the request edge logs that skip once per
//  request. Images are not an input (ADR-0063, amended 2026-09-18): the
//  template renders each as its single placeholder, so every render — and
//  every registered path — is in render space, one pad per image, and an
//  image's identity and run live in the **Cache Key Space**, never in the
//  index; the **Request Keying** edge expands the pads into the processor's
//  runs after the resolve. The leaf store's fast path (`storedRenderBytes`)
//  renders to bytes only and never resolves: the live leaf is stored under
//  the ids that were fed.
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
//  The value is built at the **Request Keying** edge, from instance facts,
//  and enriched once by the leaf store with the C31 base render. It rides
//  `HTTPPrefixCacheGeneration` to the post-generation phases.
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
    /// work (`StablePrefixDetector`) legitimately shares it; the *render
    /// choreography* is what callers must not re-open.
    let tokenizer: any Tokenizer

    /// The request's canonicalized tool specs — one value for every render
    /// of this request, so a probe render cannot drift from prepare's.
    let toolSpecs: [ToolSpec]?

    /// The request's template render context; base and merged
    /// `additionalContext` both derive from it, inside the verbs.
    let renderContext: TemplateRenderContext

    /// The fingerprint every cache resolve keys under, or `nil` to bypass
    /// the cache and render in full (see the header). Fixed at construction.
    private(set) var cacheFingerprint: String?

    /// The loaded model's fingerprint as the render was built with it, kept
    /// when `bypassing(_:)` drops the cache: the key the **Generation
    /// Prompt** probe is remembered under. `nil` for an unknown fingerprint
    /// and for uncached renders, whose probe is remembered per tokenizer
    /// instance instead.
    private let modelFingerprint: String?

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
    /// a harness-local index.
    private(set) var emittedPathFingerprint: String?

    /// The request's account of every resolve — a reference, so the copies
    /// the phases make all write to one record.
    let emittedPathTelemetry: EmittedPathRequestTelemetry?

    /// Why the render bypasses the cache and the index, when it does.
    private(set) var ineligibility: Ineligibility?

    enum Ineligibility: String, Sendable {
        case unknownFingerprint
        case uncached
        /// The keying edge could not expand this image-bearing request's
        /// render at the processor's placeholder runs, so the model was fed
        /// the processor's own tokens: no render of this request may serve
        /// or register an emitted path (`bypassing(_:)`).
        case placeholderStructureMismatch
    }

    /// Tokens plus the bytes they encode, when the render produced bytes
    /// (a `ChatTemplateRendering` tokenizer; `nil` under the fused fallback).
    struct Rendered: Sendable {
        let tokens: [Int]
        let bytes: [UInt8]?
    }

    // MARK: - Construction (the eligibility decision)

    /// The one spelling of the eligibility predicate, shared by the request
    /// edge and the agent edge: engage the cache under a known fingerprint.
    ///
    /// Neither the model's class nor the request's images is an input. The
    /// token list this render produces is shape-agnostic; the **Model
    /// Session** shapes it at the processor's own rank (`textOnlyInput`
    /// — 1D on the LLM-class text processor, 2D `[1, seq]` on a vision
    /// container). Until 2026-09-18 a `producesFlatTextTokens` leg here
    /// excluded every vision container, so Bonsai 2 27B and the PARO
    /// Qwen3.5 pack served text-only coding sessions without the
    /// Render+Token Cache OR the Emitted Path Index, and one 18,939-token
    /// `write` turn re-prefilled in 95 s. A `media` leg then excluded every
    /// request after an image entered a session: the same Pi session read a
    /// PNG, its next request re-encoded a 59 KB `write` turn canonically,
    /// diverged inside it, and re-prefilled 29,715 tokens — and every later
    /// request carried that image. The render is image-agnostic (one pad
    /// per image, the header above); the keying edge expands the pads.
    private static func eligibility(
        modelFingerprint: String?
    ) -> (fingerprint: String?, ineligibility: Ineligibility?) {
        guard let modelFingerprint else { return (nil, .unknownFingerprint) }
        return (modelFingerprint, nil)
    }

    /// The request-edge constructor: engage the cache under a known
    /// fingerprint. Whether the instance processes the request's images is
    /// the keying phase's `producesFlatTextTokens` reading; it decides
    /// which images the processor sees, never the render's eligibility.
    ///
    /// `diagnostics` is the request's diagnostics net for the Emitted Path
    /// events; `nil` sends them to the server log.
    static func forRequest(
        tokenizer: any Tokenizer,
        toolSpecs: [ToolSpec]?,
        renderContext: TemplateRenderContext,
        modelFingerprint: String?,
        cache: RenderTokenCache = .shared,
        emittedPathIndex: EmittedPathIndex? = .shared,
        diagnostics: PrefixCacheDiagnostics.Context? = nil
    ) -> ConversationRender {
        let eligibility = eligibility(modelFingerprint: modelFingerprint)
        return ConversationRender(
            tokenizer: tokenizer,
            toolSpecs: toolSpecs,
            renderContext: renderContext,
            cacheFingerprint: eligibility.fingerprint,
            modelFingerprint: modelFingerprint,
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
            modelFingerprint: nil,
            baseRenderTokens: nil,
            cache: .shared,
            emittedPathIndex: emittedPathIndex,
            emittedPathFingerprint: emittedPathIndex == nil ? nil : emittedPathFingerprint,
            emittedPathTelemetry: emittedPathTelemetry,
            ineligibility: .uncached
        )
    }

    /// A copy that renders in full for the rest of this request — no cache,
    /// no index — because what the model was fed is not what this render
    /// produces. The keying edge takes it when an image-bearing request's
    /// render could not be expanded at the processor's placeholder runs and
    /// the processor's own tokens were fed instead: a registered path would
    /// then carry a placeholder its rendered-byte key does not, and a later
    /// request that hashes to the same key — an image-free one, say — would
    /// be served that placeholder. The registration reports `reason` as its
    /// cause; the later renders serve canonical tokens, which is what the
    /// key path was built from.
    func bypassing(_ reason: Ineligibility) -> ConversationRender {
        var copy = self
        copy.cacheFingerprint = nil
        copy.emittedPathFingerprint = nil
        copy.ineligibility = reason
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

    /// The marker's status for the fingerprint: the marker layer's
    /// derivation (`EndOfTurnMarkerStatus.derive`, two probe renders and
    /// the split check) over this module's render-to-text rung, memoized in
    /// the index until the fingerprint changes. A tokenizer it refuses
    /// registers no path and resolves nothing; the registration reports
    /// the reason.
    static func endOfTurnMarkerStatus(
        index: EmittedPathIndex,
        fingerprint: String,
        tokenizer: any Tokenizer
    ) -> EndOfTurnMarkerStatus {
        index.endOfTurnMarker(fingerprint: fingerprint) {
            EndOfTurnMarkerStatus.derive(tokenizer: tokenizer) { messages, additionalContext in
                try renderText(
                    tokenizer: tokenizer, messages: messages, tools: nil,
                    additionalContext: additionalContext)
            }
        }
    }

    /// How the registration side reaches the index: the engaged index, its
    /// fingerprint and the model's marker — or the registration skip this
    /// render reports instead: `ineligibleRender` with the render's own
    /// reason as its cause, or the marker's unavailability.
    enum EmittedPathEligibility {
        case eligible(index: EmittedPathIndex, fingerprint: String, marker: EndOfTurnMarker)
        case ineligible(EmittedPathRegistration.SkipReason, cause: String?)
    }

    func emittedPathEligibility() -> EmittedPathEligibility {
        guard let index = emittedPathIndex, let fingerprint = emittedPathFingerprint else {
            return .ineligible(.ineligibleRender, cause: ineligibility?.rawValue ?? "noIndex")
        }
        switch Self.endOfTurnMarkerStatus(
            index: index, fingerprint: fingerprint, tokenizer: tokenizer)
        {
        case .available(let marker):
            return .eligible(index: index, fingerprint: fingerprint, marker: marker)
        case .unavailable(let reason):
            return .ineligible(reason.skipReason, cause: nil)
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
    /// to its processor's `prepare`: for an unknown fingerprint, and for
    /// any media — this edge has no **Cache Key Space** and no grid step to
    /// expand a render's placeholders with, so a media-bearing agent input
    /// is the processor's alone. `messages` is an autoclosure so such a
    /// request never pays the message-forming pass; a caller that cannot
    /// form messages at all yields `nil` from it and falls back the same
    /// way. The caller shapes the returned list at its processor's rank
    /// (`textOnlyInput(tokens:)`).
    static func agentEdgeFullRender(
        tokenizer: any Tokenizer,
        messages: @autoclosure () -> [[String: any Sendable]]?,
        tools: [ToolSpec]?,
        additionalContext: [String: any Sendable]?,
        hasMedia: Bool,
        modelFingerprint: String?,
        cache: RenderTokenCache = .shared,
        emittedPathIndex: EmittedPathIndex? = .shared
    ) -> [Int]? {
        guard !hasMedia,
            let fingerprint = eligibility(modelFingerprint: modelFingerprint).fingerprint
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

// MARK: - The Generation Prompt (ADR-0070)

nonisolated extension ConversationRender {

    /// The template-level **Generation Prompt** under this render's context:
    /// two one-message probe renders, remembered per model fingerprint and
    /// render context in the **Render+Token Cache** (per tokenizer instance
    /// when the fingerprint is unknown). Never rendered with tools. A
    /// request reads only what `checkedGenerationPrompt(fed:diagnostics:)`
    /// returns, the probe checked against what that request fed.
    var generationPromptProbe: GenerationPrompt.Probe {
        Self.generationPromptProbe(
            tokenizer: tokenizer, renderContext: renderContext,
            modelFingerprint: modelFingerprint, cache: cache)
    }

    /// The probe from raw ingredients, for the agent's Raw Generation Start
    /// and the load-time measure, which have no request render. The same
    /// memo as the instance property: one pair of renders per model and
    /// render context, whoever asks first.
    static func generationPromptProbe(
        tokenizer: any Tokenizer,
        renderContext: TemplateRenderContext,
        modelFingerprint: String?,
        cache: RenderTokenCache = .shared
    ) -> GenerationPrompt.Probe {
        cache.generationPromptProbes.probe(
            modelFingerprint: modelFingerprint, tokenizer: tokenizer,
            contextDigest: renderContext.digest
        ) {
            measureGenerationPrompt(tokenizer: tokenizer, renderContext: renderContext)
        }
    }

    /// The request's **Generation Prompt**: this render's probe checked
    /// against the tokens the request fed (its **Cache Key Path**, or the
    /// prompt tokens of an Unkeyed Completion). An unknown result logs its
    /// reason on the request's diagnostics every time and a warning once
    /// per model and render context.
    func checkedGenerationPrompt(
        fed fedTokens: [Int], diagnostics: PrefixCacheDiagnostics.Context?
    ) -> GenerationPrompt {
        Self.checkedGenerationPrompt(
            tokenizer: tokenizer, renderContext: renderContext,
            modelFingerprint: modelFingerprint, fed: fedTokens, cache: cache,
            diagnostics: diagnostics)
    }

    /// `checkedGenerationPrompt(fed:diagnostics:)` from raw ingredients.
    static func checkedGenerationPrompt(
        tokenizer: any Tokenizer,
        renderContext: TemplateRenderContext,
        modelFingerprint: String?,
        fed fedTokens: [Int],
        cache: RenderTokenCache = .shared,
        diagnostics: PrefixCacheDiagnostics.Context?
    ) -> GenerationPrompt {
        let prompt = generationPromptProbe(
            tokenizer: tokenizer, renderContext: renderContext,
            modelFingerprint: modelFingerprint, cache: cache
        ).checked(against: fedTokens)
        guard let reason = prompt.unknownReason else { return prompt }
        let digest = String(renderContext.digest.prefix(12))
        diagnostics?.logSkip(
            stage: "generationPrompt", reason: reason.rawValue,
            extraFields: [("contextDigest", digest)])
        if cache.generationPromptProbes.firstUnknown(
            modelFingerprint: modelFingerprint, tokenizer: tokenizer,
            contextDigest: renderContext.digest, reason: reason)
        {
            Log.server.warning(
                "generation prompt unknown (\(reason.rawValue)) under render context \(digest): "
                    + "the stream parser starts outside a think block, a stop turn takes the "
                    + "canonical user leaf, MTP stays off and no last-message boundary is placed")
        }
        return prompt
    }

    /// Stage one of the measurement: one user message rendered with and
    /// without the generation prompt under `renderContext`, through the
    /// module's one template application. The prompt measures only when the
    /// render without it is a byte prefix of the render with it and encoding
    /// the whole equals encoding the prefix followed by the suffix, so no
    /// token merges across the append point (the hard-boundary check the
    /// end-of-turn marker makes). A tokenizer that cannot render text is
    /// checked as a token prefix, and the text is the suffix's decode. An
    /// empty difference is measured: the template appends nothing.
    private static func measureGenerationPrompt(
        tokenizer: any Tokenizer, renderContext: TemplateRenderContext
    ) -> GenerationPrompt.Probe {
        let probe: [[String: any Sendable]] = [
            ["role": "user", "content": GenerationPrompt.probeContent]
        ]
        let withPrompt: Rendered
        let withoutPrompt: Rendered
        do {
            withPrompt = try applyTemplate(
                tokenizer: tokenizer, messages: probe, tools: nil,
                additionalContext: renderContext.additionalContext())
            withoutPrompt = try applyTemplate(
                tokenizer: tokenizer, messages: probe, tools: nil,
                additionalContext: renderContext.additionalContext(
                    merging: ["add_generation_prompt": false]))
        } catch {
            return GenerationPrompt.Probe(failed: .renderFailed)
        }
        guard let withBytes = withPrompt.bytes, let withoutBytes = withoutPrompt.bytes else {
            guard withPrompt.tokens.starts(with: withoutPrompt.tokens) else {
                return GenerationPrompt.Probe(failed: .notAnAppend)
            }
            let tokens = Array(withPrompt.tokens[withoutPrompt.tokens.count...])
            return GenerationPrompt.Probe(
                tokens: tokens, text: tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false))
        }
        guard withBytes.starts(with: withoutBytes),
            let text = String(bytes: withBytes[withoutBytes.count...], encoding: .utf8)
        else { return GenerationPrompt.Probe(failed: .notAnAppend) }
        let tokens = tokenizer.encode(text: text, addSpecialTokens: false)
        guard withoutPrompt.tokens + tokens == withPrompt.tokens else {
            return GenerationPrompt.Probe(failed: .unstableSplit)
        }
        return GenerationPrompt.Probe(tokens: tokens, text: text)
    }
}

/// The **Generation Prompt** (CONTEXT.md, ADR-0070): what the chat template
/// appends after the last message to open the assistant turn under one
/// render context. Measured from the template in two stages, never spelled
/// by hand: the probe (`ConversationRender.generationPromptProbe`), then
/// the check against what one request fed (`Probe.checked(against:)`).
/// Whether generation starts inside a think block, whether the turn
/// carries one a think-stripping template will drop from history, and
/// where the last-message boundary sits are all read from the checked
/// value. Constructible only by that derivation, so no caller can pair a
/// template with a flag it does not produce.
nonisolated struct GenerationPrompt: Sendable, Equatable {

    /// The think block the prompt leaves generation in.
    enum ThinkBlock: Sendable, Equatable {
        /// The prompt's last think-open tag has no close tag after it:
        /// generation starts inside a think block.
        case opens
        /// The prompt carries a think block it also closes, as
        /// `enable_thinking: false` does on the Qwen3.5 and Qwen3.8
        /// templates.
        case closed
        /// A measured prompt with no think tag, an empty one included.
        case none
        /// The prompt could not be measured, or this request did not feed
        /// what was measured. Every consumer has its own answer for it.
        case unknown(Unknown)
    }

    /// Why a request's Generation Prompt is unknown.
    enum Unknown: String, Sendable, Equatable, Hashable {
        /// A probe render threw.
        case renderFailed
        /// The render without the prompt is not a prefix of the render
        /// with it: adding the prompt rewrote what came before.
        case notAnAppend
        /// A token merges across the append point, so the prompt's tokens
        /// depend on what precedes them.
        case unstableSplit
        /// The request's fed tokens do not end with the probe's: its
        /// template's prompt depends on the conversation or the tools.
        case notFed
    }

    /// The prompt's tokens: `nil` when unknown, empty when the template
    /// appends nothing.
    let tokens: [Int]?
    let thinkBlock: ThinkBlock

    /// Whether the stream parser starts inside a think block.
    var startsInsideThinkBlock: Bool { thinkBlock == .opens }

    /// Why the prompt is unknown, or `nil` when it was measured.
    var unknownReason: Unknown? {
        guard case .unknown(let reason) = thinkBlock else { return nil }
        return reason
    }

    /// The trace spelling: `opens`, `closed`, `none`, or `unknown(<reason>)`.
    var traceValue: String {
        switch thinkBlock {
        case .opens: "opens"
        case .closed: "closed"
        case .none: "none"
        case .unknown(let reason): "unknown(\(reason.rawValue))"
        }
    }

    fileprivate init(tokens: [Int]?, thinkBlock: ThinkBlock) {
        self.tokens = tokens
        self.thinkBlock = thinkBlock
    }

    /// The probe conversation's one user message: no think tag, so every
    /// tag in the difference is the template's own.
    fileprivate static let probeContent = "generation-prompt probe"

    /// Stage one: the template-level measurement, remembered per model and
    /// render context. Not a fact a consumer may read; `checked(against:)`
    /// turns it into one for a request.
    struct Probe: Sendable, Equatable {
        fileprivate enum Measurement: Sendable, Equatable {
            case measured(tokens: [Int], thinkBlock: ThinkBlock)
            case failed(Unknown)
        }

        fileprivate let measurement: Measurement

        fileprivate init(failed reason: Unknown) {
            measurement = .failed(reason)
        }

        fileprivate init(tokens: [Int], text: String) {
            measurement = .measured(tokens: tokens, thinkBlock: Self.thinkBlock(in: text))
        }

        /// Stage two: what one request may read. The probe's tokens when
        /// `fedTokens` ends with them, `.unknown(.notFed)` when it does not,
        /// the probe's own failure when it failed.
        func checked(against fedTokens: [Int]) -> GenerationPrompt {
            switch measurement {
            case .failed(let reason):
                return GenerationPrompt(tokens: nil, thinkBlock: .unknown(reason))
            case .measured(let tokens, let thinkBlock):
                guard fedTokens.count >= tokens.count,
                    fedTokens.suffix(tokens.count).elementsEqual(tokens)
                else { return GenerationPrompt(tokens: nil, thinkBlock: .unknown(.notFed)) }
                return GenerationPrompt(tokens: tokens, thinkBlock: thinkBlock)
            }
        }

        /// The measured prompt's token count, `nil` when the probe failed.
        /// Only for the replay harness's prefill arithmetic about a request
        /// it never renders with a prompt; a consumer reads a checked value.
        var measuredTokenCount: Int? {
            guard case .measured(let tokens, _) = measurement else { return nil }
            return tokens.count
        }

        /// The think block in a measured prompt's text, by the stream
        /// parser's own tags.
        private static func thinkBlock(in text: String) -> ThinkBlock {
            let open = ToolCallParser.thinkStartTag
            let close = ToolCallParser.thinkEndTag
            guard let lastOpen = text.range(of: open, options: .backwards) else {
                return text.contains(close) ? .closed : .none
            }
            return text[lastOpen.upperBound...].contains(close) ? .closed : .opens
        }
    }
}
