//
//  LeafStorePhase.swift
//  tesseract
//
//  The **Leaf Store** phase of a cache-aware **Server Completion**: after the
//  stream drive finishes, decide how (or whether) the finished turn's KV
//  state is admitted as a leaf — the mode selection (direct vs boundary), the
//  stored-conversation re-tokenization and key-space translation, the
//  boundary plan and its restore→reprefill→capture executor, the direct
//  final-cache capture with its normalization-trim guard, and the
//  **Speculative Canonical Prefill** seeding. Previously the ~380-line
//  `leafBlock` inside the completion drive; now a named phase whose skip
//  ladder and decision rules are the module's interface.
//
//  Canonical leaf policy:
//  - thinking templates store one template-canonical leaf synthesized from
//    the transient boundary snapshot
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

    // Evolving MVP mid-refactor (see CLAUDE.md); the phase keeps the drive's
    // lenient structural limits — splitting further is deferred. The wide
    // parameter list is the phase's honest input set (the drive's request
    // context); bundling it into a struct would just rename the coupling.
    // swiftlint:disable function_body_length function_parameter_count
    /// `assistantReasoning` must be the wire-truth reasoning — what THIS
    /// client will echo back (the drive passes the streamed form for
    /// streaming clients). Intervened turns store like any other: the
    /// boundary capture never reuses the raw continuation's live KV — it
    /// restores the boundary snapshot and re-prefills the canonical render.
    ///
    /// `generatedTokens` is the loop's record of every id fed past the prompt
    /// (`GeneratedTokenRecorder.snapshot`, read after the generation task
    /// finished) and `intervened` whether a safeguard continuation swapped the
    /// raw generation — the two inputs the **Live Leaf Capture** decision
    /// needs beside the canonical stored path.
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
        generatedTokens: [Int],
        intervened: Bool,
        diagnosticsContext: PrefixCacheDiagnostics.Context,
        trace: inout CompletionTraceAccumulator
    ) async throws -> Result {
        // swiftlint:enable function_body_length function_parameter_count
        let mlxStart = mlxStartBox.value
        var result = Result()

        // An Unkeyed Completion never touches the radix tree — construction
        // failed, so no token path of this request can be trusted as a key.
        if let unkeyedReason = mlxStart.unkeyedReason {
            diagnosticsContext.logSkip(
                stage: "leafStore",
                reason: "unkeyed-completion",
                extraFields: [("unkeyedReason", unkeyedReason.rawValue)]
            )
            result.report.skipReason = "unkeyed-completion"
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
            diagnosticsContext.logSkip(
                stage: "leafStore",
                reason: "tokenization-failed",
                level: .warning
            )
            result.report.skipReason = "tokenization-failed"
            return result
        }
        let storedTokens: [Int]
        switch mlxStart.keySpace.translate(renderTokens: storedRenderTokens) {
        case .success(let translated):
            storedTokens = translated
        case .failure(let failure):
            diagnosticsContext.logSkip(
                stage: "leafStore",
                reason: "render-translation-failed",
                level: .warning,
                extraFields: [("failure", "\(failure)")]
            )
            result.report.skipReason = "render-translation-failed"
            return result
        }

        result.report.timings.renderMs = Self.millisecondsSince(renderStart)

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

        // directLeaf snapshots the live final KV cache (below) and needs none
        // of the builder's probe/boundary/tokenizer work; only the boundary
        // modes route through the GPU-free plan. This mapping is the one
        // place that knows directLeaf is the live-cache path, so a future
        // `HTTPLeafStoreMode` surfaces as a compile error here rather than a
        // silently missed branch.
        let boundaryMode: BoundaryLeafMode? =
            switch leafStoreMode {
            case .directToolLeaf: .directTool
            case .canonicalUserLeaf: .canonical
            case .directLeaf: nil
            }
        if let boundaryMode {
            let transientBoundary: HybridCacheSnapshot? =
                switch boundaryMode {
                case .directTool: mlxStart.transientLastMessageBoundarySnapshot
                case .canonical: mlxStart.transientLastUserBoundarySnapshot
                }
            let leafPlan = await LeafAdmissionBuilder.plan(
                mode: boundaryMode,
                storedConversation: storedConversation,
                storedTokens: storedTokens,
                transientBoundary: transientBoundary,
                keySpace: mlxStart.keySpace,
                // C31: the stored render just computed above is the identical
                // computation the builder's base probe would re-run (verified
                // in C28) — carry it so the base render runs once per request.
                render: render.carryingBaseRender(storedRenderTokens),
                resolveBoundary: { tokens in
                    // Drive Snapshot Resolution inside the Model Session so
                    // the SSD `loadSync` stays off-MainActor (ADR-0001).
                    // Session entry cannot fail with a non-throwing body; the
                    // hypothetical failure degrades to "no boundary snapshot".
                    let resolved = try? await sessions.withSession { _ in
                        await prefixCache.resolve(
                            tokens: tokens,
                            promptTokenCount: tokens.count,
                            partitionKey: mlxStart.partitionKey,
                            modelFingerprint: mlxStart.partitionKey.modelFingerprint,
                            diagnostics: diagnosticsContext,
                            pinningRestorePathFor: diagnosticsContext.requestID
                        ).lookup.snapshot
                    }
                    return resolved.flatMap { $0 }
                }
            )

            // One exhaustive switch over the boundary plan: `.skip` logs the
            // decidable reason; `.fromBoundary` runs the shared
            // restore→reprefill→capture executor. Only directLeaf reaches the
            // live final-cache capture below.
            switch leafPlan {
            case .skip(let reason):
                logLeafSkip(reason, mode: boundaryMode, diagnosticsContext: diagnosticsContext)
                result.report.skipReason = leafSkipLog(for: reason, mode: boundaryMode).reason
                return result
            case .fromBoundary(let boundarySnapshot, let boundaryStoredTokens):
                // The boundary sits past the image prefix (builder guard), so
                // the residual is real tokens in both spaces and the anchor
                // delta is always defined; on the vision container the
                // residual reprefill must resume with it seeded.
                var positionAnchorRopeDelta: Int?
                if mlxStart.seedsPositionAnchor {
                    guard
                        let delta = mlxStart.keySpace.positionAnchorDelta(
                            upTo: boundarySnapshot.tokenOffset
                        )
                    else {
                        diagnosticsContext.logSkip(
                            stage: Self.leafStages(for: boundaryMode).store,
                            reason: "boundary-splits-image-run",
                            level: .warning,
                            extraFields: [("offset", "\(boundarySnapshot.tokenOffset)")]
                        )
                        result.report.skipReason = "boundary-splits-image-run"
                        return result
                    }
                    positionAnchorRopeDelta = delta
                }
                let stages = Self.leafStages(for: boundaryMode)

                // **Live Leaf Capture** (GPU-free decision): when the fed
                // token path is a prefix of the canonical stored path, the
                // live final cache already holds this leaf's state and the
                // boundary restore + residual re-prefill would only recompute
                // it — take the live cache instead. Any disagreement keeps
                // the boundary executor, logged so an unexpected divergence
                // on an append-stable render is visible.
                let liveDecision = LiveLeafCapture.decide(
                    promptKeyPath: mlxStart.keySpace.keyPath,
                    generatedTokens: generatedTokens,
                    cacheOffset: httpPrefixCacheReportedTokenCount(mlxStart.finalCache),
                    storedTokens: boundaryStoredTokens,
                    intervened: intervened,
                    keySpaceIsIdentity: mlxStart.keySpace.isIdentity
                )
                let leafOffset: Int
                switch liveDecision {
                case .live(let offset):
                    leafOffset = offset
                case .boundary(let reason):
                    leafOffset = boundaryStoredTokens.count
                    logLiveFallback(
                        reason,
                        mode: boundaryMode,
                        renderContext: render.renderContext,
                        diagnosticsContext: diagnosticsContext
                    )
                    result.report.liveFallbackReason = reason.wireReason
                }

                // Seed the **Speculative Canonical Prefill** before the
                // GPU-side store: the seed spawns the future-path probe
                // immediately, so its CPU render+tokenize overlaps the store
                // (#76's earlier start). Kept only if the leaf store below
                // succeeds. The worth-it floor differs by trigger: a
                // canonical leaf IS the strip floor; a tool stretch measures
                // its rewind span from the last-user boundary.
                let seedPlan = Self.speculativeSeedPlan(
                    boundaryMode: boundaryMode,
                    renderContext: render.renderContext
                )
                let pendingSeed: SpeculativeCanonicalPrefill.Seed? =
                    seedPlan.map { plan in
                        SpeculativeCanonicalPrefill.makeSeed(
                            storedConversation: storedConversation,
                            render: render,
                            keySpace: mlxStart.keySpace,
                            partitionKey: mlxStart.partitionKey,
                            prefillStepSize: mlxStart.prefillStepSize,
                            ssdEnabled: mlxStart.ssdEnabled,
                            seedsPositionAnchor: mlxStart.seedsPositionAnchor,
                            canonicalLeafOffset: boundaryMode == .canonical
                                ? leafOffset
                                : mlxStart.transientLastUserBoundarySnapshot?
                                    .tokenOffset ?? 0,
                            idleDelay: plan.idleDelay,
                            ramOnlySpine: plan.ramOnlySpine,
                            diagnostics: diagnosticsContext
                        )
                    }
                let capture: LeafCapture
                switch liveDecision {
                case .live(let offset):
                    result.report.path = .live
                    capture = await captureLiveLeaf(
                        sessions: sessions,
                        mlxStartBox: mlxStartBox,
                        storedTokens: Array(boundaryStoredTokens.prefix(offset)),
                        requestID: requestID,
                        prefixCache: prefixCache,
                        diagnosticsContext: diagnosticsContext,
                        captureStage: stages.capture,
                        admissionStage: stages.admission,
                        captureSource: stages.source
                    )
                case .boundary:
                    result.report.path = .boundary
                    capture = await captureStructuredLeafFromBoundary(
                        sessions: sessions,
                        storedTokens: boundaryStoredTokens,
                        boundarySnapshot: boundarySnapshot,
                        positionAnchorRopeDelta: positionAnchorRopeDelta,
                        partitionKey: mlxStart.partitionKey,
                        prefillStepSize: mlxStart.prefillStepSize,
                        tokenNDim: mlxStart.tokenNDim,
                        requestID: requestID,
                        prefixCache: prefixCache,
                        diagnosticsContext: diagnosticsContext,
                        ssdEnabled: mlxStart.ssdEnabled,
                        storeStage: stages.store,
                        captureStage: stages.capture,
                        admissionStage: stages.admission,
                        captureSource: stages.source
                    )
                }
                result.report.leafOffset = capture.leafOffset
                result.report.residualTokens = capture.residualTokens
                var timings = capture.timings
                timings.renderMs = result.report.timings.renderMs
                result.report.timings = timings
                if let admission = capture.admission {
                    trace.ingest(
                        evictions: admission.evictions, diagnostics: diagnosticsContext)
                    trace.logSupersessions(
                        admission.supersededLeaves, diagnostics: diagnosticsContext)
                }
                // A stored canonical leaf still ends at the think-strip
                // divergence; everything past it would re-prefill
                // interactively on the next user message — hand the seed to
                // the post-finish hook so the pass can extend the leaf while
                // the GPU is idle (#76).
                result.leafStore = capture.leafStore
                if capture.leafStore != nil {
                    result.speculativeSeed = pendingSeed
                } else {
                    pendingSeed?.discard()
                }
                return result
            }
        }

        // Non-thinking templates: the pre-existing live path. The direct
        // executor snapshots the live final cache under the stored path.
        let capture = try await captureDirectLeaf(
            mlxStartBox: mlxStartBox,
            sessions: sessions,
            storedTokens: storedTokens,
            requestID: requestID,
            prefixCache: prefixCache,
            diagnosticsContext: diagnosticsContext
        )
        result.report.skipReason = capture.skipReason
        result.report.path = capture.leafOffset == nil ? .skipped : .direct
        result.report.leafOffset = capture.leafOffset
        var directTimings = capture.timings
        directTimings.renderMs = result.report.timings.renderMs
        result.report.timings = directTimings
        if let admission = capture.admission {
            trace.ingest(evictions: admission.evictions, diagnostics: diagnosticsContext)
            trace.logSupersessions(admission.supersededLeaves, diagnostics: diagnosticsContext)
        }
        result.leafStore = capture.leafStore
        return result
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

    /// The diagnostics stage labels for a `.fromBoundary` capture, by boundary
    /// leaf mode — the exact strings the dissolved `captureDirectToolLeaf` /
    /// `captureCanonicalTemplateLeaf` helpers passed to the shared executor.
    static func leafStages(
        for mode: BoundaryLeafMode
            // Evolving MVP mid-refactor (see CLAUDE.md); structural limit kept lenient — splitting deferred.
            // swiftlint:disable:next large_tuple
    ) -> (store: String, capture: String, admission: String, source: String) {
        switch mode {
        case .directTool:
            (
                "directToolLeafStore", "directToolLeafCapture", "directToolLeafAdmission",
                "directToolLeaf"
            )
        case .canonical:
            (
                "canonicalLeafStore", "canonicalLeafCapture", "canonicalLeafAdmission",
                "canonicalLeaf"
            )
        }
    }

    // MARK: - Skip wire format

    /// The exact `logSkip` record a decidable `LeafSkipReason` reproduces — the
    /// stage/reason/level/fields the dissolved capture helpers logged.
    struct LeafSkipLog: Sendable {
        let stage: String
        let reason: String
        let level: PrefixCacheDiagnostics.Level
        let extraFields: [(String, String)]
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

    /// Emit the mapped wire record for a decidable `LeafSkipReason` the **Leaf
    /// Admission Builder** returned, so existing dashboards and the diagnostics
    /// net keep working byte-for-byte.
    private static func logLeafSkip(
        _ reason: LeafSkipReason,
        mode: BoundaryLeafMode,
        diagnosticsContext: PrefixCacheDiagnostics.Context
    ) {
        let record = leafSkipLog(for: reason, mode: mode)
        diagnosticsContext.logSkip(
            stage: record.stage,
            reason: record.reason,
            level: record.level,
            extraFields: record.extraFields
        )
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
