//
//  LeafStorePhase.swift
//  tesseract
//
//  The **Leaf Store** phase of a cache-aware **Server Completion**: after the
//  stream drive finishes, decide how (or whether) the finished turn's KV
//  state is admitted as a leaf — the mode selection (direct vs boundary), the
//  stored-conversation re-tokenization and key-space translation, the
//  boundary routing (reusable-prefix probe → **Live Leaf Capture** decision →
//  boundary plan only on its fallback), the **Speculative Canonical Prefill**
//  seeding, and the dispatch to the model-affine executors
//  (`LeafStorePhase+Executors.swift`). Previously the ~380-line `leafBlock`
//  inside the completion drive; now a named phase whose skip ladder and
//  decision rules are the module's interface.
//
//  Canonical leaf policy:
//  - thinking templates store one template-canonical leaf: from the live
//    final cache when the fed path proves canonical (ADR-0062), otherwise
//    synthesized from the transient boundary snapshot
//  - non-thinking templates store the direct post-response leaf captured
//    from the final cache
//
//  Isolation matches the drive that calls it (nonisolated, off-actor); every
//  model-affine step hops through the **Model Session** (ADR-0016) and cache
//  admissions hop to the MainActor-confined Prefix Cache, exactly as before.
//

import Foundation
import MLX
import MLXLMCommon

nonisolated enum LeafStorePhase {

    /// What the phase concluded: the alpha tuner's leaf-store record if one
    /// landed, the Speculative Canonical Prefill seed if this turn armed
    /// one (handed to the post-finish hook by the drive), and the report of
    /// which path ran and where its time went.
    struct Result: Sendable {
        var leafStore: AlphaTuner.LeafStore?
        var speculativeSeed: SpeculativeCanonicalPrefill.Seed?
        var report = Report()
    }

    /// The request facts every step of the phase reads: the generation the
    /// drive ran, the session provider, and the admission context.
    struct Inputs: Sendable {
        let mlxStartBox: UnsafeSendableBox<HTTPPrefixCacheGeneration>
        let sessions: any ModelSessionProviding
        let requestID: UUID
        let prefixCache: PrefixCacheManager
        let diagnosticsContext: PrefixCacheDiagnostics.Context
        /// Whether a thinking-safeguard continuation swapped the raw
        /// generation — the registered final cache is then the cancelled
        /// phase's, so the live path is refused.
        let intervened: Bool

        var mlxStart: HTTPPrefixCacheGeneration { mlxStartBox.value }
    }

    // Evolving MVP mid-refactor (see CLAUDE.md); the phase keeps the drive's
    // lenient structural limits — splitting further is deferred. The wide
    // parameter list is the phase's honest input set (the drive's request
    // context); `Inputs` carries its request-constant subset to the helpers.
    // swiftlint:disable function_body_length function_parameter_count
    /// `assistantReasoning` must be the wire-truth reasoning — what THIS
    /// client will echo back (the drive passes the streamed form for
    /// streaming clients). Intervened turns store like any other: the
    /// boundary capture never reuses the raw continuation's live KV — it
    /// restores the boundary snapshot and re-prefills the canonical render.
    static func run(
        mlxStartBox: UnsafeSendableBox<HTTPPrefixCacheGeneration>,
        conversation: HTTPPrefixCacheConversation,
        sessions: any ModelSessionProviding,
        requestID: UUID,
        prefixCache: PrefixCacheManager,
        promptStartsThinking: Bool,
        assistantText: String,
        assistantReasoning: String?,
        toolCalls: [HTTPPrefixCacheToolCall],
        intervened: Bool,
        diagnosticsContext: PrefixCacheDiagnostics.Context,
        trace: inout CompletionTraceAccumulator
    ) async -> Result {
        // swiftlint:enable function_body_length function_parameter_count
        let inputs = Inputs(
            mlxStartBox: mlxStartBox, sessions: sessions, requestID: requestID,
            prefixCache: prefixCache, diagnosticsContext: diagnosticsContext,
            intervened: intervened)
        let mlxStart = inputs.mlxStart
        var result = Result()

        // An Unkeyed Completion never touches the radix tree — construction
        // failed, so no token path of this request can be trusted as a key.
        if let unkeyedReason = mlxStart.unkeyedReason {
            result.report.recordSkip(
                LeafSkipLog(
                    stage: "leafStore", reason: "unkeyed-completion", level: .info,
                    extraFields: [("unkeyedReason", unkeyedReason.rawValue)]),
                in: diagnosticsContext)
            return result
        }

        // 1. Build stored conversation (prompt + generated assistant turn).
        let storedConversation = conversation.appendingAssistant(
            .assistant(
                content: assistantText,
                reasoning: assistantReasoning ?? "",
                toolCalls: toolCalls
            ))

        // 2. Re-tokenize stored conversation → flat render sequence, then
        // translate into key space (identity for text-only). The translated
        // path is what every capture offset and admission below keys on —
        // length-equal to the prepared sequence, so key index == KV offset
        // holds. The **Conversation Render** owns the whole C28 ladder
        // (cache-eligibility, tail-replacement resolve, template fallback);
        // the guard/nil diagnostics behavior is unchanged. Raw prompt
        // messages, so assistant `reasoning_content` and `tool_calls`
        // survive template rendering.
        guard let render = mlxStart.render else {
            // The render is nil exactly for an Unkeyed Completion, which the
            // guard above already returned on.
            return result
        }
        let renderStart = Date.timeIntervalSinceReferenceDate
        let storedRenderTokens: [Int]
        do {
            storedRenderTokens = try render.continuationRender(
                messages: storedConversation.promptMessages
            )
        } catch {
            Log.agent.warning(
                "Stored token sequence measurement failed — error=\(error.localizedDescription)"
            )
            result.report.recordSkip(
                LeafSkipLog(
                    stage: "leafStore", reason: "tokenization-failed", level: .warning,
                    extraFields: []),
                in: diagnosticsContext)
            return result
        }
        let storedTokens: [Int]
        switch mlxStart.keySpace.translate(renderTokens: storedRenderTokens) {
        case .success(let translated):
            storedTokens = translated
        case .failure(let failure):
            result.report.recordSkip(
                LeafSkipLog(
                    stage: "leafStore", reason: "render-translation-failed", level: .warning,
                    extraFields: [("failure", "\(failure)")]),
                in: diagnosticsContext)
            return result
        }
        result.report.renderSeconds = secondsSince(renderStart)

        let leafStoreMode = Self.selectHTTPLeafStoreMode(
            promptStartsThinking: promptStartsThinking,
            emittedToolCalls: !toolCalls.isEmpty
        )
        result.report.mode = leafStoreMode.rawValue
        diagnosticsContext.log(
            PrefixCacheDiagnostics.LeafModeEvent(
                mode: leafStoreMode.rawValue,
                continuation: toolCalls.isEmpty
                    ? HTTPLeafContinuationKind.userTurn.rawValue
                    : HTTPLeafContinuationKind.toolResult.rawValue
            ))

        // directLeaf snapshots the live final KV cache and needs none of the
        // builder's probe/boundary/tokenizer work; only the boundary modes
        // route through the GPU-free plan. This mapping is the one place
        // that knows directLeaf is the render-trusting live path, so a
        // future `HTTPLeafStoreMode` surfaces as a compile error here rather
        // than a silently missed branch.
        let boundaryMode: BoundaryLeafMode? =
            switch leafStoreMode {
            case .directToolLeaf: .directTool
            case .canonicalUserLeaf: .canonical
            case .directLeaf: nil
            }

        let capture: LeafCapture
        let path: Report.Path
        var pendingSeed: SpeculativeCanonicalPrefill.Seed?
        if let boundaryMode {
            let planStart = Date.timeIntervalSinceReferenceDate
            guard
                let route = await routeBoundaryMode(
                    boundaryMode,
                    inputs: inputs,
                    storedConversation: storedConversation,
                    storedTokens: storedTokens,
                    // C31: the stored render just computed above is the
                    // identical computation the builder's base probe would
                    // re-run (verified in C28) — carry it so the base render
                    // runs once per request.
                    probeRender: render.carryingBaseRender(storedRenderTokens),
                    preservesThinking: render.renderContext.preservesThinking,
                    report: &result.report
                )
            else { return result }
            result.report.planSeconds = secondsSince(planStart)

            // Seed the **Speculative Canonical Prefill** before the GPU-side
            // store: the seed spawns the future-path probe immediately, so
            // its CPU render+tokenize overlaps the store (#76's earlier
            // start). Kept only if the leaf store below succeeds. The
            // worth-it floor differs by trigger: a canonical leaf IS the
            // strip floor; a tool stretch measures its rewind span from the
            // last-user boundary.
            pendingSeed = Self.speculativeSeedPlan(
                boundaryMode: boundaryMode,
                renderContext: render.renderContext
            ).map { plan in
                SpeculativeCanonicalPrefill.makeSeed(
                    storedConversation: storedConversation,
                    render: render,
                    keySpace: mlxStart.keySpace,
                    partitionKey: mlxStart.partitionKey,
                    prefillStepSize: mlxStart.prefillStepSize,
                    ssdEnabled: mlxStart.ssdEnabled,
                    seedsPositionAnchor: mlxStart.seedsPositionAnchor,
                    canonicalLeafOffset: boundaryMode == .canonical
                        ? route.leafOffset
                        : mlxStart.transientLastUserBoundarySnapshot?.tokenOffset ?? 0,
                    idleDelay: plan.idleDelay,
                    ramOnlySpine: plan.ramOnlySpine,
                    diagnostics: diagnosticsContext
                )
            }

            let context = LeafAdmissionContext(
                storedTokens: route.storedTokens, inputs: inputs,
                stages: Self.leafStages(for: boundaryMode))
            path = route.path
            switch route {
            case .live:
                capture = await captureLiveLeaf(
                    sessions: sessions, mlxStartBox: mlxStartBox, context: context)
            case .boundary(let boundarySnapshot, let positionAnchorRopeDelta, _):
                capture = await captureStructuredLeafFromBoundary(
                    sessions: sessions,
                    boundarySnapshot: boundarySnapshot,
                    positionAnchorRopeDelta: positionAnchorRopeDelta,
                    prefillStepSize: mlxStart.prefillStepSize,
                    tokenNDim: mlxStart.tokenNDim,
                    context: context
                )
            }
        } else {
            // Non-thinking templates: the pre-existing live path. The direct
            // executor snapshots the live final cache under the stored path.
            path = .direct
            capture = await captureDirectLeaf(
                sessions: sessions, mlxStartBox: mlxStartBox,
                context: LeafAdmissionContext(
                    storedTokens: storedTokens, inputs: inputs, stages: .direct))
        }

        result.report.absorb(capture, path: path)
        if let admission = capture.admission {
            trace.ingest(evictions: admission.evictions, diagnostics: diagnosticsContext)
            trace.logSupersessions(admission.supersededLeaves, diagnostics: diagnosticsContext)
        }
        // A stored canonical leaf still ends at the think-strip divergence;
        // everything past it would re-prefill interactively on the next user
        // message — hand the seed to the post-finish hook so the pass can
        // extend the leaf while the GPU is idle (#76).
        result.leafStore = capture.leafStore
        if capture.leafStore != nil {
            result.speculativeSeed = pendingSeed
        } else {
            pendingSeed?.discard()
        }
        return result
    }

    // MARK: - Boundary routing

    /// Where a boundary mode's leaf comes from, once the reusable-prefix
    /// probe, the **Live Leaf Capture** decision and — only on its fallback —
    /// the boundary plan have run.
    enum BoundaryRoute {
        /// **Live Leaf Capture**: the live final cache at the cache's own
        /// offset, admitted under the probed path's prefix of that length.
        case live(storedTokens: [Int])
        /// Restore the boundary, re-prefill `storedTokens[boundary.tokenOffset...]`
        /// (seeded with the position-anchor delta on the vision container),
        /// capture at `storedTokens.count`.
        case boundary(HybridCacheSnapshot, positionAnchorRopeDelta: Int?, storedTokens: [Int])

        var storedTokens: [Int] {
            switch self {
            case .live(let tokens), .boundary(_, _, let tokens): tokens
            }
        }

        /// The leaf's offset — what the speculative seed measures from.
        var leafOffset: Int { storedTokens.count }

        var path: Report.Path {
            switch self {
            case .live: .live
            case .boundary: .boundary
            }
        }
    }

    /// Route one boundary mode: probe the shared token path, decide the live
    /// capture on it, and only when the live path is refused choose a restore
    /// boundary (**Snapshot Resolution** may hydrate from SSD, and is never
    /// paid for a leaf the live path already holds). Returns `nil` after
    /// recording the decidable skip in `report`.
    static func routeBoundaryMode(
        _ mode: BoundaryLeafMode,
        inputs: Inputs,
        storedConversation: HTTPPrefixCacheConversation,
        storedTokens: [Int],
        probeRender: ConversationRender,
        preservesThinking: Bool,
        report: inout Report
    ) async -> BoundaryRoute? {
        let mlxStart = inputs.mlxStart
        let diagnostics = inputs.diagnosticsContext

        let probedTokens: [Int]
        switch LeafAdmissionBuilder.probe(
            mode: mode,
            storedConversation: storedConversation,
            storedTokens: storedTokens,
            keySpace: mlxStart.keySpace,
            render: probeRender
        ) {
        case .tokens(let tokens):
            probedTokens = tokens
        case .skip(let reason):
            report.recordSkip(leafSkipLog(for: reason, mode: mode), in: diagnostics)
            return nil
        }

        // **Live Leaf Capture** (GPU-free): when the fed token path is a
        // prefix of the canonical stored path, the live final cache already
        // holds this leaf's state and the boundary restore + residual
        // re-prefill would only recompute it — take the live cache instead.
        // Any disagreement keeps the boundary executor, logged so an
        // unexpected divergence on an append-stable render is visible.
        switch LiveLeafCapture.decide(
            promptKeyPath: mlxStart.keySpace.keyPath,
            generatedTokens: mlxStart.generatedTokens.snapshot,
            cacheOffset: httpPrefixCacheReportedTokenCount(mlxStart.finalCache),
            storedTokens: probedTokens,
            intervened: inputs.intervened,
            keySpaceIsIdentity: mlxStart.keySpace.isIdentity
        ) {
        case .live(let offset):
            return .live(storedTokens: Array(probedTokens.prefix(offset)))
        case .boundary(let reason):
            let record = liveFallbackLog(
                for: reason, mode: mode, preservesThinking: preservesThinking)
            record.emit(in: diagnostics)
            report.liveFallbackReason = record.reason
        }

        let transientBoundary: HybridCacheSnapshot? =
            switch mode {
            case .directTool: mlxStart.transientLastMessageBoundarySnapshot
            case .canonical: mlxStart.transientLastUserBoundarySnapshot
            }
        let plan = await LeafAdmissionBuilder.plan(
            mode: mode,
            probedTokens: probedTokens,
            transientBoundary: transientBoundary,
            keySpace: mlxStart.keySpace,
            resolveBoundary: { tokens in
                // Drive Snapshot Resolution inside the Model Session so the
                // SSD `loadSync` stays off-MainActor (ADR-0001). Session
                // entry cannot fail with a non-throwing body; the
                // hypothetical failure degrades to "no boundary snapshot".
                let resolved = try? await inputs.sessions.withSession { _ in
                    await inputs.prefixCache.resolve(
                        tokens: tokens,
                        promptTokenCount: tokens.count,
                        partitionKey: mlxStart.partitionKey,
                        modelFingerprint: mlxStart.partitionKey.modelFingerprint,
                        diagnostics: diagnostics,
                        pinningRestorePathFor: diagnostics.requestID
                    ).lookup.snapshot
                }
                return resolved.flatMap { $0 }
            }
        )
        switch plan {
        case .skip(let reason):
            report.recordSkip(leafSkipLog(for: reason, mode: mode), in: diagnostics)
            return nil
        case .fromBoundary(let boundary, let tokens):
            // The boundary sits past the image prefix (builder guard), so the
            // residual is real tokens in both spaces and the anchor delta is
            // always defined; on the vision container the residual reprefill
            // must resume with it seeded.
            var positionAnchorRopeDelta: Int?
            if mlxStart.seedsPositionAnchor {
                guard
                    let delta = mlxStart.keySpace.positionAnchorDelta(upTo: boundary.tokenOffset)
                else {
                    report.recordSkip(
                        LeafSkipLog(
                            stage: leafStages(for: mode).store,
                            reason: "boundary-splits-image-run", level: .warning,
                            extraFields: [("offset", "\(boundary.tokenOffset)")]),
                        in: diagnostics)
                    return nil
                }
                positionAnchorRopeDelta = delta
            }
            return .boundary(
                boundary, positionAnchorRopeDelta: positionAnchorRopeDelta, storedTokens: tokens)
        }
    }

    // MARK: - Mode selection

    static func selectHTTPLeafStoreMode(
        promptStartsThinking: Bool,
        emittedToolCalls: Bool
    ) -> HTTPLeafStoreMode {
        if emittedToolCalls {
            return .directToolLeaf
        }
        if promptStartsThinking {
            return .canonicalUserLeaf
        }
        return .directLeaf
    }

    /// The diagnostics stage labels of a boundary leaf mode — the exact
    /// strings the dissolved `captureDirectToolLeaf` /
    /// `captureCanonicalTemplateLeaf` helpers passed to the shared executor.
    static func leafStages(for mode: BoundaryLeafMode) -> LeafStages {
        switch mode {
        case .directTool:
            LeafStages(
                store: "directToolLeafStore", capture: "directToolLeafCapture",
                admission: "directToolLeafAdmission", source: "directToolLeaf")
        case .canonical:
            LeafStages(
                store: "canonicalLeafStore", capture: "canonicalLeafCapture",
                admission: "canonicalLeafAdmission", source: "canonicalLeaf")
        }
    }

    // MARK: - Skip wire format

    /// The exact `logSkip` record a decidable skip reproduces — the
    /// stage/reason/level/fields the dissolved capture helpers logged.
    struct LeafSkipLog: Sendable {
        let stage: String
        let reason: String
        let level: PrefixCacheDiagnostics.Level
        let extraFields: [(String, String)]

        /// Emit through the request's diagnostics net.
        func emit(in diagnostics: PrefixCacheDiagnostics.Context) {
            diagnostics.logSkip(
                stage: stage, reason: reason, level: level, extraFields: extraFields)
        }
    }

    /// Map a decidable skip to its wire record. The reason carries the payload
    /// (offsets, lengths); the stage prefix follows the boundary mode, exactly as
    /// the dissolved `captureDirectToolLeaf` / `captureCanonicalTemplateLeaf`
    /// helpers did. `.info` is the `logSkip` default those untyped helpers relied
    /// on, made explicit so the level is pinned too. A pure value (no `Context`,
    /// no side effect) so `ServerCompletionLeafSkipLogTests` pins the
    /// byte-for-byte wire format — mirroring `ssdDropReasonString` — and any
    /// future drift (a renamed stage, a flipped level) fails a test rather than
    /// silently shifting dashboards and the diagnostics net.
    static func leafSkipLog(
        for reason: LeafSkipReason,
        mode: BoundaryLeafMode
    ) -> LeafSkipLog {
        let stage = leafStages(for: mode).store
        switch reason {
        case .tokenizationFailed(let error):
            // The probe's chat-template render threw — today's helpers catch this
            // in the same `do/catch` as the prefill, logged as `prefill-threw`.
            return LeafSkipLog(
                stage: stage, reason: "prefill-threw", level: .warning,
                extraFields: [("error", error)]
            )
        case .probeDivergence:
            return LeafSkipLog(
                stage: stage, reason: "probe-divergence-failed", level: .info, extraFields: []
            )
        case .noTransientBoundary:
            return LeafSkipLog(
                stage: stage, reason: "no-transient-boundary-snapshot", level: .info,
                extraFields: []
            )
        case .noResolvedBoundary(let canonicalLen):
            return LeafSkipLog(
                stage: stage, reason: "no-canonical-restore-boundary", level: .info,
                extraFields: [("canonicalLen", "\(canonicalLen)")]
            )
        case .storedAtOrBeforeBoundary(let storedLen, let boundaryOffset):
            return LeafSkipLog(
                stage: stage, reason: "stored-at-or-before-boundary", level: .info,
                extraFields: [
                    ("storedLen", "\(storedLen)"), ("boundaryOffset", "\(boundaryOffset)"),
                ]
            )
        case .canonicalLongerThanStored(let canonicalLen, let storedLen):
            return LeafSkipLog(
                stage: stage, reason: "canonical-longer-than-stored", level: .warning,
                extraFields: [("canonicalLen", "\(canonicalLen)"), ("storedLen", "\(storedLen)")]
            )
        case .renderTranslationFailed(let failure):
            // The probe render's image-placeholder arithmetic disagreed with
            // the request's image table — feature-level skip, request unharmed.
            return LeafSkipLog(
                stage: stage, reason: "render-translation-failed", level: .warning,
                extraFields: [("failure", "\(failure)")]
            )
        case .boundaryInsideImagePrefix(let boundaryOffset, let minimumWarmOffset):
            // The residual would contain an image run, which cannot be
            // reprefilled — expected on image-add turns, hence `.info`.
            return LeafSkipLog(
                stage: stage, reason: "boundary-inside-image-prefix", level: .info,
                extraFields: [
                    ("boundaryOffset", "\(boundaryOffset)"),
                    ("minimumWarmOffset", "\(minimumWarmOffset)"),
                ]
            )
        }
    }

    /// The wire record of a **Live Leaf Capture** refusal (stage
    /// `liveLeafCapture`). Eligibility-only reasons — an intervened turn, an
    /// image key space, no fed ids — are `.info` like every other expected
    /// skip. A render that ended before or disagreed with the emission is
    /// `.warning` when the render is expected to be append-stable (the
    /// preserve-thinking render, and every tool stretch) and `.info` under
    /// the strip-by-default canonical render, which drops the emitted
    /// thinking by design. A cache offset outside the live path is the loop
    /// and the cache disagreeing about what was fed, and is always
    /// `.warning`. Pure, so `ServerCompletionLeafSkipLogTests` pins it beside
    /// `leafSkipLog`.
    static func liveFallbackLog(
        for reason: LiveLeafCapture.FallbackReason,
        mode: BoundaryLeafMode,
        preservesThinking: Bool
    ) -> LeafSkipLog {
        let stripExpected = mode == .canonical && !preservesThinking
        let disagreement: PrefixCacheDiagnostics.Level = stripExpected ? .info : .warning
        func record(
            _ token: String, _ level: PrefixCacheDiagnostics.Level,
            _ fields: [(String, String)] = []
        ) -> LeafSkipLog {
            LeafSkipLog(
                stage: "liveLeafCapture", reason: token, level: level,
                extraFields: fields + [
                    ("mode", leafStages(for: mode).source),
                    ("preservesThinking", "\(preservesThinking)"),
                ])
        }
        switch reason {
        case .intervened:
            return record("intervened", .info)
        case .nonIdentityKeySpace:
            return record("non-identity-key-space", .info)
        case .noGeneratedTokens:
            return record("no-generated-tokens", .info)
        case .cacheOffsetOutsideLivePath(let cacheOffset, let promptCount, let liveCount):
            return record(
                "cache-offset-outside-live-path", .warning,
                [
                    ("cacheOffset", "\(cacheOffset)"), ("promptCount", "\(promptCount)"),
                    ("liveCount", "\(liveCount)"),
                ])
        case .liveLongerThanStored(let cacheOffset, let storedLen):
            return record(
                "live-longer-than-stored", disagreement,
                [("cacheOffset", "\(cacheOffset)"), ("storedLen", "\(storedLen)")])
        case .divergence(let offset, let liveToken, let storedToken, let live, let stored):
            return record(
                "divergence", disagreement,
                [
                    ("offset", "\(offset)"), ("liveToken", "\(liveToken)"),
                    ("storedToken", "\(storedToken)"),
                    ("liveContext", "\(live)"), ("storedContext", "\(stored)"),
                ])
        }
    }

    // MARK: - Speculative seeding

    /// How a finished turn seeds the **Speculative Canonical Prefill** —
    /// the trigger table (issues #76, #100):
    /// - A canonical-user boundary (stop-finish answer) seeds immediately,
    ///   durable — the original #76 trigger.
    /// - A tool-call boundary arms **Stretch Abandonment**'s timer: the
    ///   pass starts only if no follow-up request lands inside the idle
    ///   window, and its spine admits RAM-only so a false alarm (the tool
    ///   result arrives) costs zero SSD writes (ADR-0009).
    /// - Under the **Preserve-Thinking Render** (issue #98) nothing seeds:
    ///   the render is append-stable, so the canonical future path equals
    ///   the live path and there is no Think-Strip Rewind span to
    ///   pre-prefill.
    struct SpeculativeSeedPlan: Equatable {
        let idleDelay: Duration
        let ramOnlySpine: Bool
    }

    static func speculativeSeedPlan(
        boundaryMode: BoundaryLeafMode,
        renderContext: TemplateRenderContext
    ) -> SpeculativeSeedPlan? {
        guard !renderContext.preservesThinking else { return nil }
        switch boundaryMode {
        case .canonical:
            return SpeculativeSeedPlan(idleDelay: .zero, ramOnlySpine: false)
        case .directTool:
            return SpeculativeSeedPlan(
                idleDelay: SpeculativeCanonicalPrefill.stretchAbandonmentIdleWindow,
                ramOnlySpine: true
            )
        }
    }
}
