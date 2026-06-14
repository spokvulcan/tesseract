import Foundation
import MLXLMCommon

/// The two leaf-store modes that capture from a **restored boundary** — as
/// opposed to `directLeaf`, which snapshots the live final KV cache actor-side
/// and never enters the builder. The actor maps the selected `HTTPLeafStoreMode`
/// down to this narrower vocabulary (returning `nil` for `directLeaf`), so
/// `plan`, `leafStages`, and the skip-log mapping are each *total* over exactly
/// the cases they handle — no dead `directLeaf` arm kept only for exhaustiveness.
nonisolated enum BoundaryLeafMode: Sendable {
    /// Tool-call turn, reused by the immediate tool-result continuation.
    case directTool
    /// Non-latest assistant turn re-rendered under a thinking template.
    case canonical
}

/// The GPU-free decision the **Leaf Admission Builder** emits for one **boundary**
/// leaf **Snapshot Admission**: which token path to capture, and from where. A
/// *total* value — every `BoundaryLeafMode` maps to exactly one of these — so the
/// actor's post-generation `switch` is exhaustive and a future boundary leaf mode
/// surfaces as a compile error rather than a silently missed branch. (`directLeaf`
/// is decided before the builder; it captures from the live final cache.)
nonisolated enum LeafCapturePlan: Sendable {
    /// Restore `boundary`, reprefill the residual `storedTokens[boundary.tokenOffset...]`,
    /// capture a leaf at `storedTokens.count`.
    case fromBoundary(boundary: HybridCacheSnapshot, storedTokens: [Int])
    /// A decidable skip — the routing ruled out a capture before any Metal work.
    case skip(reason: LeafSkipReason)
}

/// The typed vocabulary of **decidable** leaf-store skips: the failures the
/// GPU-free builder rules on from the tokenizer and the resolved boundary
/// alone. Each case carries the diagnostic payload the actor reproduces in
/// today's `logSkip` fields, so the decidable-skip vocabulary lives in one
/// typed place instead of string literals scattered across the capture
/// helpers. The live-`finalCache` skips (`no-final-cache`, `no-reusable-cache-state`,
/// `normalization-trim`, `unsupported-cache-type`, `invalid-path`,
/// `capturedThenEvicted`) and the `intervention` guard stay actor-side and are
/// not represented here.
nonisolated enum LeafSkipReason: Sendable, Equatable {
    /// The reusable-prefix probe's chat-template render threw.
    case tokenizationFailed(error: String)
    /// The probe found no usable common prefix — no distinct continuation boundary.
    case probeDivergence
    /// *directTool*: no request-local transient boundary snapshot to restore from.
    case noTransientBoundary
    /// *canonical*: neither the transient boundary nor **Snapshot Resolution**
    /// yielded a usable restore boundary.
    case noResolvedBoundary(canonicalLen: Int)
    /// *directTool*: the residual is empty — the stored path ends at or before
    /// the boundary.
    case storedAtOrBeforeBoundary(storedLen: Int, boundaryOffset: Int)
    /// *canonical*: the canonical prefix is longer than the stored path — the
    /// render disagreement the canonical leaf exists to sidestep.
    case canonicalLongerThanStored(canonicalLen: Int, storedLen: Int)
    /// The probe's render-space token path could not be translated into key
    /// space (image placeholder arithmetic disagreed with the request's image
    /// table). Only this leaf capture skips; the request keeps serving.
    case renderTranslationFailed(failure: CacheKeySpace.TranslationFailure)
    /// The restore boundary sits inside the request's image prefix: the
    /// residual would contain an image run (pseudo-tokens, no pixels), which
    /// cannot be reprefilled (ADR-0007 spike results).
    case boundaryInsideImagePrefix(boundaryOffset: Int, minimumWarmOffset: Int)
}

/// The GPU-free routing decisions for capturing a **leaf** Snapshot Admission:
/// which token path a future continuation can hydrate. Tokenizer-affine, no
/// live KV cache — the actor executes the Metal capture/`admit` from the
/// decision this produces.
///
/// Today this owns the **reusable-prefix probes** shared by the canonical-user
/// and tool-continuation leaf modes. Leaf-mode selection still lives on
/// `ServerCompletion.selectHTTPLeafStoreMode` (already a tested pure function), and the
/// Metal capture stays in the actor.
nonisolated enum LeafAdmissionBuilder {

    /// The synthetic continuation a reusable-prefix probe appends to discover
    /// the shared token path. A stop turn re-renders as a user continuation; a
    /// tool-call turn re-renders as a tool-result continuation.
    enum Continuation: Sendable {
        case userTurn
        case toolResult

        var probeMessage: [String: any Sendable] {
            switch self {
            case .userTurn:
                ["role": Chat.Message.Role.user.rawValue, "content": "Aqkz_strip_probe"]
            case .toolResult:
                ["role": Chat.Message.Role.tool.rawValue, "content": "Aqkz_tool_probe"]
            }
        }
    }

    /// The reusable token prefix for the stored assistant turn: the shared
    /// prefix between the turn's isolated render (no generation prompt) and the
    /// render that appends a single synthetic continuation. That shared prefix
    /// is the exact token path the immediate continuation can hydrate.
    ///
    /// Both renders are *render space* (one placeholder token per image,
    /// unexpanded); the shared prefix is translated through the **Cache Key
    /// Space** before it can serve as an admission path or compare against
    /// boundary offsets. Identity for text-only requests.
    ///
    /// Returns `nil` when the probe diverges — no common prefix, or the prefix
    /// is not strictly shorter than the continuation (the appended turn added
    /// no tokens, so there is no distinct boundary to align to).
    static func reusablePrefix(
        continuation: Continuation,
        storedConversation: HTTPPrefixCacheConversation,
        toolSpecs: [ToolSpec]?,
        tokenizer: any Tokenizer,
        keySpace: CacheKeySpace,
        renderContext: TemplateRenderContext = .canonical
    ) throws -> Result<[Int], CacheKeySpace.TranslationFailure>? {
        let baseMessages = storedConversation.promptMessages
        let probeContext = renderContext.additionalContext(
            merging: ["add_generation_prompt": false]
        )
        let storedTokens = try tokenizer.applyChatTemplate(
            messages: baseMessages,
            tools: toolSpecs,
            additionalContext: probeContext
        )
        let continuationTokens = try tokenizer.applyChatTemplate(
            messages: baseMessages + [continuation.probeMessage],
            tools: toolSpecs,
            additionalContext: probeContext
        )

        let common = zip(storedTokens, continuationTokens).prefix { $0 == $1 }.count
        guard common > 0, common <= storedTokens.count, common < continuationTokens.count else {
            return nil
        }
        return keySpace.translate(renderTokens: Array(continuationTokens[0..<common]))
    }

    /// The second user probe for `futureSharedPrefix`. Diverges from
    /// `Continuation.userTurn`'s content at the *first* character: the common
    /// prefix of the two probe renders must stop inside the user-turn header,
    /// before any content token. A shared leading character would leak probe
    /// content into the "shared" path and admit a leaf no real user message
    /// could ever walk to.
    static let divergentUserProbeMessage: [String: any Sendable] = [
        "role": Chat.Message.Role.user.rawValue,
        "content": "Zqxv_strip_probe",
    ]

    /// The token path the *next* request will share regardless of what the
    /// user types: the longest common prefix between two single-user-turn
    /// continuation renders whose probe contents diverge at the first
    /// character. A thinking template strips the post-last-user `<think>`
    /// span identically in both renders — exactly as it will for the real
    /// next user message — so the shared prefix runs through the whole
    /// think-stripped conversation and the next user turn's header, ending
    /// where that message's content would begin.
    ///
    /// This is the **Speculative Canonical Prefill** target path
    /// (CONTEXT.md): everything in it past the canonical leaf is the
    /// **Think-Strip Rewind** span a background pass can re-prefill while the
    /// GPU is idle. Same render/key-space contract and failure channels as
    /// `reusablePrefix`: render space in, key space out; `nil` when the two
    /// renders never converge or never diverge (no distinct boundary).
    ///
    /// Cooperatively cancellable: the speculative pass cancels its probe on
    /// preemption (and discarded seeds cancel it on the floor), and the
    /// checks here bound the abandoned work to one render+tokenize — without
    /// them a cancel would be a no-op against this synchronous body, and the
    /// preempting request would wait out the full remaining probe.
    static func futureSharedPrefix(
        storedConversation: HTTPPrefixCacheConversation,
        toolSpecs: [ToolSpec]?,
        tokenizer: any Tokenizer,
        keySpace: CacheKeySpace,
        renderContext: TemplateRenderContext = .canonical
    ) throws -> Result<[Int], CacheKeySpace.TranslationFailure>? {
        try Task.checkCancellation()
        let baseMessages = storedConversation.promptMessages
        // Render under the request's flags, exactly as `reusablePrefix`
        // does. The seed is keyed under a render-context-dependent
        // partition, so its probe path must render under the same flags —
        // otherwise the spine lands in a partition the real next turn never
        // walks. (`preserve_thinking` disables speculation today, so the
        // canonical default is what currently runs; this keeps the two in
        // step for the next flag that does not.)
        let probeContext = renderContext.additionalContext(
            merging: ["add_generation_prompt": false]
        )
        let firstRender = try tokenizer.applyChatTemplate(
            messages: baseMessages + [Continuation.userTurn.probeMessage],
            tools: toolSpecs,
            additionalContext: probeContext
        )
        try Task.checkCancellation()
        let secondRender = try tokenizer.applyChatTemplate(
            messages: baseMessages + [divergentUserProbeMessage],
            tools: toolSpecs,
            additionalContext: probeContext
        )

        let common = zip(firstRender, secondRender).prefix { $0 == $1 }.count
        guard common > 0, common < firstRender.count, common < secondRender.count else {
            return nil
        }
        return keySpace.translate(renderTokens: Array(firstRender[0..<common]))
    }

    /// `reusablePrefix` with its three failure channels — tokenization throw,
    /// probe divergence (`nil`), translation failure — folded into the one
    /// `LeafSkipReason` vocabulary both boundary modes speak.
    private enum Probe {
        case tokens([Int])
        case skip(LeafSkipReason)
    }

    private static func probeTokens(
        continuation: Continuation,
        storedConversation: HTTPPrefixCacheConversation,
        toolSpecs: [ToolSpec]?,
        tokenizer: any Tokenizer,
        keySpace: CacheKeySpace,
        renderContext: TemplateRenderContext
    ) -> Probe {
        let probe: Result<[Int], CacheKeySpace.TranslationFailure>?
        do {
            probe = try reusablePrefix(
                continuation: continuation,
                storedConversation: storedConversation,
                toolSpecs: toolSpecs,
                tokenizer: tokenizer,
                keySpace: keySpace,
                renderContext: renderContext
            )
        } catch {
            return .skip(.tokenizationFailed(error: error.localizedDescription))
        }
        switch probe {
        case .none:
            return .skip(.probeDivergence)
        case .some(.failure(let failure)):
            return .skip(.renderTranslationFailed(failure: failure))
        case .some(.success(let translated)):
            return .tokens(translated)
        }
    }

    /// The whole GPU-free leaf-capture routing decision for one boundary mode.
    ///
    /// Takes the boundary `mode` (the actor maps `selectHTTPLeafStoreMode`'s
    /// output down to the two boundary modes; `directLeaf` captures from the live
    /// cache and never reaches here), the re-tokenized stored token path, the
    /// mode-relevant transient boundary snapshot, a tokenizer, and a
    /// `resolveBoundary` closure-**peer** for the canonical fallback. Runs the
    /// reusable-prefix probe, chooses the boundary source, applies the
    /// offset-guard arithmetic, and emits a `LeafCapturePlan`. No live KV cache,
    /// no Metal — the actor executes the capture/`admit` from the decision.
    static func plan(
        mode: BoundaryLeafMode,
        storedConversation: HTTPPrefixCacheConversation,
        storedTokens: [Int],
        toolSpecs: [ToolSpec]?,
        transientBoundary: HybridCacheSnapshot?,
        tokenizer: any Tokenizer,
        keySpace: CacheKeySpace,
        renderContext: TemplateRenderContext = .canonical,
        resolveBoundary: @Sendable ([Int]) async -> HybridCacheSnapshot?
    ) async -> LeafCapturePlan {
        switch mode {
        case .directTool:
            // Tool-call turns are reused by the immediate tool-result
            // continuation: restore the request-local boundary and reprefill
            // the tool-continuation render.
            guard let transientBoundary else {
                return .skip(reason: .noTransientBoundary)
            }
            let toolTokens: [Int]
            switch probeTokens(
                continuation: .toolResult,
                storedConversation: storedConversation,
                toolSpecs: toolSpecs,
                tokenizer: tokenizer,
                keySpace: keySpace,
                renderContext: renderContext
            ) {
            case .tokens(let translated): toolTokens = translated
            case .skip(let reason): return .skip(reason: reason)
            }
            guard toolTokens.count > transientBoundary.tokenOffset else {
                return .skip(
                    reason: .storedAtOrBeforeBoundary(
                        storedLen: toolTokens.count,
                        boundaryOffset: transientBoundary.tokenOffset
                    ))
            }
            // The residual `toolTokens[boundary...]` doubles as the reprefill
            // input, so it must be image-free — past the last image run, key
            // space and real tokens agree token-for-token.
            guard transientBoundary.tokenOffset >= keySpace.minimumWarmOffset else {
                return .skip(
                    reason: .boundaryInsideImagePrefix(
                        boundaryOffset: transientBoundary.tokenOffset,
                        minimumWarmOffset: keySpace.minimumWarmOffset
                    ))
            }
            return .fromBoundary(boundary: transientBoundary, storedTokens: toolTokens)

        case .canonical:
            // Thinking templates may re-render non-latest assistant turns
            // differently from the just-generated form: capture the canonical
            // render under the token path a future non-latest render will see.
            let canonicalTokens: [Int]
            switch probeTokens(
                continuation: .userTurn,
                storedConversation: storedConversation,
                toolSpecs: toolSpecs,
                tokenizer: tokenizer,
                keySpace: keySpace,
                renderContext: renderContext
            ) {
            case .tokens(let translated): canonicalTokens = translated
            case .skip(let reason): return .skip(reason: reason)
            }

            // Prefer the request-local transient boundary when it sits strictly
            // before the canonical path; otherwise fall back through the injected
            // **Snapshot Resolution** closure-peer. Both arms require
            // `tokenOffset < canonicalTokens.count`, so the chosen boundary always
            // leaves a non-empty residual to reprefill — no separate
            // at-or-before-boundary guard is needed. Both arms also require the
            // boundary to sit past the image prefix (`≥ minimumWarmOffset`):
            // the residual doubles as the reprefill input, and an image run in
            // it cannot be reprefilled (a resolved snapshot from an older,
            // pre-image turn can land there).
            let minimumWarmOffset = keySpace.minimumWarmOffset
            let boundary: HybridCacheSnapshot
            if let transientBoundary,
                transientBoundary.tokenOffset < canonicalTokens.count,
                transientBoundary.tokenOffset >= minimumWarmOffset
            {
                boundary = transientBoundary
            } else if let resolved = await resolveBoundary(canonicalTokens),
                resolved.tokenOffset > 0,
                resolved.tokenOffset < canonicalTokens.count,
                resolved.tokenOffset >= minimumWarmOffset
            {
                boundary = resolved
            } else {
                return .skip(reason: .noResolvedBoundary(canonicalLen: canonicalTokens.count))
            }

            guard canonicalTokens.count <= storedTokens.count else {
                return .skip(
                    reason: .canonicalLongerThanStored(
                        canonicalLen: canonicalTokens.count,
                        storedLen: storedTokens.count
                    ))
            }
            return .fromBoundary(boundary: boundary, storedTokens: canonicalTokens)
        }
    }
}
