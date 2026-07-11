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
    /// landed, and the Speculative Canonical Prefill seed if this turn armed
    /// one (handed to the post-finish hook by the drive).
    struct Result: Sendable {
        var leafStore: AlphaTuner.LeafStore?
        var speculativeSeed: SpeculativeCanonicalPrefill.Seed?
    }

    // Evolving MVP mid-refactor (see CLAUDE.md); the phase keeps the drive's
    // lenient structural limits — splitting further is deferred. The wide
    // parameter list is the phase's honest input set (the drive's request
    // context); bundling it into a struct would just rename the coupling.
    // swiftlint:disable function_body_length function_parameter_count
    static func run(
        mlxStartBox: UnsafeSendableBox<HTTPPrefixCacheGeneration>,
        conversation: HTTPPrefixCacheConversation,
        sessions: any ModelSessionProviding,
        canonicalTools: [ToolSpec]?,
        requestID: UUID,
        prefixCache: PrefixCacheManager,
        renderContext: TemplateRenderContext,
        promptStartsThinking: Bool,
        intervened: Bool,
        assistantText: String,
        assistantReasoning: String?,
        toolCalls: [HTTPPrefixCacheToolCall],
        diagnosticsContext: PrefixCacheDiagnostics.Context,
        trace: inout CompletionTraceAccumulator
    ) async throws -> Result {
        // swiftlint:enable function_body_length function_parameter_count
        let mlxStart = mlxStartBox.value
        var result = Result()

        // Skip leaf-store when a thinking-safeguard intervention fired: the
        // continuation ran through the raw path, so the on-device KV cache no
        // longer matches the radix-tree logical snapshot we'd compute from
        // `textContent + thinkingContent + toolCalls`. Storing anything here
        // would corrupt future prefix-cache hits for requests sharing this
        // prefix. The stable-prefix snapshot captured pre-generation is still
        // stored unconditionally by the drive, so future requests still
        // benefit from partial cache reuse; only the leaf is lost for this
        // one turn.
        if intervened {
            diagnosticsContext.logSkip(
                stage: "leafStore",
                reason: "thinking-safeguard-intervention"
            )
            return result
        }

        // An Unkeyed Completion never touches the radix tree — construction
        // failed, so no token path of this request can be trusted as a key.
        if let unkeyedReason = mlxStart.unkeyedReason {
            diagnosticsContext.logSkip(
                stage: "leafStore",
                reason: "unkeyed-completion",
                extraFields: [("unkeyedReason", unkeyedReason.rawValue)]
            )
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
        // holds.
        guard
            let storedRenderTokens = await measureStoredTokenSequence(
                sessions: sessions,
                conversation: storedConversation,
                toolSpecs: canonicalTools,
                renderContext: renderContext
            )
        else {
            diagnosticsContext.logSkip(
                stage: "leafStore",
                reason: "tokenization-failed",
                level: .warning
            )
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
            return result
        }

        let leafStoreMode = Self.selectHTTPLeafStoreMode(
            promptStartsThinking: promptStartsThinking,
            emittedToolCalls: !toolCalls.isEmpty
        )
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
            let leafTokenizer = try await sessions.withSession { $0.tokenizer }
            let leafPlan = await LeafAdmissionBuilder.plan(
                mode: boundaryMode,
                storedConversation: storedConversation,
                storedTokens: storedTokens,
                toolSpecs: canonicalTools,
                transientBoundary: transientBoundary,
                tokenizer: leafTokenizer,
                keySpace: mlxStart.keySpace,
                renderContext: renderContext,
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
                        return result
                    }
                    positionAnchorRopeDelta = delta
                }
                let stages = Self.leafStages(for: boundaryMode)
                // Seed the **Speculative Canonical Prefill** before the
                // GPU-side boundary store: the seed spawns the future-path
                // probe immediately, so its CPU render+tokenize overlaps the
                // store (#76's earlier start). Kept only if the leaf store
                // below succeeds. The worth-it floor differs by trigger: a
                // canonical leaf IS the strip floor; a tool stretch measures
                // its rewind span from the last-user boundary.
                let seedPlan = Self.speculativeSeedPlan(
                    boundaryMode: boundaryMode,
                    renderContext: renderContext
                )
                let pendingSeed: SpeculativeCanonicalPrefill.Seed? =
                    seedPlan.map { plan in
                        SpeculativeCanonicalPrefill.makeSeed(
                            storedConversation: storedConversation,
                            toolSpecs: canonicalTools,
                            tokenizer: leafTokenizer,
                            keySpace: mlxStart.keySpace,
                            partitionKey: mlxStart.partitionKey,
                            prefillStepSize: mlxStart.prefillStepSize,
                            ssdEnabled: mlxStart.ssdEnabled,
                            seedsPositionAnchor: mlxStart.seedsPositionAnchor,
                            canonicalLeafOffset: boundaryMode == .canonical
                                ? boundaryStoredTokens.count
                                : mlxStart.transientLastUserBoundarySnapshot?
                                    .tokenOffset ?? 0,
                            renderContext: renderContext,
                            idleDelay: plan.idleDelay,
                            ramOnlySpine: plan.ramOnlySpine,
                            diagnostics: diagnosticsContext
                        )
                    }
                let capture = await captureStructuredLeafFromBoundary(
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

        // The module owns the cache array the generation ran on; the loop's
        // completion task has been awaited by the drive, so the array is no
        // longer being mutated (ADR-0006 — this read replaced the fork's
        // FinalizedKVCacheHandle hand-off).
        guard !Task.isCancelled else {
            return result
        }
        let finalCache = mlxStart.finalCache

        let cacheOffsets = httpPrefixCacheOffsets(finalCache)
        guard httpPrefixCacheHasReusableState(finalCache) else {
            diagnosticsContext.logSkip(
                stage: "store",
                reason: "no-reusable-cache-state",
                extraFields: [("cacheOffsets", "\(cacheOffsets)")]
            )
            return result
        }

        // 3. Offset-alignment guard: if normalization shortened the stored
        //    conversation (whitespace-only assistant content → ""), we can
        //    only trim attention K/V — Mamba's recurrent state can't be
        //    unwound (`canTrimPromptCache` returns `false`). Trimming the
        //    cache and capturing it as a leaf produces a snapshot whose
        //    attention is aligned to `storedTokens.count` but whose Mamba
        //    state is from the full pre-trim offset. On Qwen3.5 the resulting
        //    leaf hit perturbs raw logits by ~10 even at trim=1: argmax stays
        //    stable (greedy decoding survives), but the rest of the
        //    distribution drifts in a way that affects sampled decoding.
        //    Since the HTTP server propagates the request's
        //    `temperature`/`top_p` and we can't predict future request
        //    sampling params at store time, the safe choice is to skip the
        //    leaf store entirely when normalization would require any trim.
        //    Lost cache hits on whitespace-normalized conversations are the
        //    trade-off for sampler-agnostic correctness. Verified by
        //    `HybridCacheCorrectnessRunner` test 9 — see the
        //    `leafHitWithNormalizationDivergence...` diagnostics for the
        //    empirical drift measurements.
        let actualCacheOffset = httpPrefixCacheReportedTokenCount(finalCache)
        if actualCacheOffset > storedTokens.count {
            let trimAmount = actualCacheOffset - storedTokens.count
            diagnosticsContext.logSkip(
                stage: "leafStore",
                reason: "normalization-trim",
                extraFields: [
                    ("trimAmount", "\(trimAmount)"),
                    ("offsetBefore", "\(actualCacheOffset)"),
                    ("canonicalCount", "\(storedTokens.count)"),
                ]
            )
            return result
        }

        // 4. Capture the leaf snapshot and derive its admission storage
        //    inside a Metal-affine Model Session so any per-array `asData()`
        //    calls run on the inference thread. `finalCache` is non-`Sendable`
        //    `[any KVCache]` — reached through the boxed `mlxStart` instead
        //    of a direct capture. The offset guard above ensures no per-layer
        //    trimming is needed before capture.
        let ssdEnabled = mlxStart.ssdEnabled
        let extensionBase = await ServerCompletion.resolveExtensionBase(
            ssdEnabled: ssdEnabled,
            tokens: storedTokens,
            partitionKey: mlxStart.partitionKey,
            prefixCache: prefixCache
        )
        let (maybeLeaf, maybeStorage): (HybridCacheSnapshot?, SnapshotAdmission.Storage?) =
            try await sessions.withSession { session in
                let cache = mlxStartBox.value.finalCache
                guard
                    let snap = session.captureSnapshot(
                        cache: cache,
                        offset: storedTokens.count,
                        type: .leaf
                    )
                else {
                    return (nil, nil)
                }
                let storage = ServerCompletion.snapshotAdmissionStorage(
                    for: snap,
                    ssdEnabled: ssdEnabled,
                    extending: extensionBase
                )
                return (snap, storage)
            }
        guard let leafSnapshot = maybeLeaf, let leafStorage = maybeStorage else {
            diagnosticsContext.logSkip(
                stage: "leafCapture",
                reason: "unsupported-cache-type",
                extraFields: [("cacheOffsets", "\(cacheOffsets)")]
            )
            return result
        }

        // Admission + eviction/supersession classification through the one
        // shared admit (the same path the boundary executor and the
        // speculative pass use), tallied into the per-request trace.
        let admission = await ServerCompletion.admitStructuredLeaf(
            leafSnapshot,
            storedTokens: storedTokens,
            storage: leafStorage,
            partitionKey: mlxStart.partitionKey,
            requestID: requestID,
            prefixCache: prefixCache,
            diagnostics: diagnosticsContext,
            admissionStage: "leafAdmission",
            captureSource: "leaf"
        )
        if let store = admission.store {
            trace.ingest(evictions: store.evictions, diagnostics: diagnosticsContext)
            trace.logSupersessions(store.supersededLeaves, diagnostics: diagnosticsContext)
        }
        if admission.survived {
            result.leafStore = AlphaTuner.LeafStore(
                storedTokens: storedTokens,
                bytes: leafSnapshot.memoryBytes
            )
        }

        // Release the MLX free buffer pool back to the OS so it doesn't
        // accumulate transient prefill intermediates across requests.
        Memory.clearCache()
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
    private static func leafStages(
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

    // MARK: - Stored-token measurement

    /// Re-tokenize the stored conversation (prompt + generated response) and return
    /// the flat token sequence. The HTTP prefix cache uses raw prompt messages here
    /// so assistant `reasoning_content` and `tool_calls` survive template rendering.
    /// Used for storing the leaf snapshot under the correct radix path.
    /// Returns `nil` on tokenization failure.
    private static func measureStoredTokenSequence(
        sessions: any ModelSessionProviding,
        conversation: HTTPPrefixCacheConversation,
        toolSpecs: [ToolSpec]?,
        renderContext: TemplateRenderContext = .canonical
    ) async -> [Int]? {
        do {
            return try await sessions.withSession { session in
                try session.tokenizer.applyChatTemplate(
                    messages: conversation.promptMessages,
                    tools: toolSpecs,
                    additionalContext: renderContext.additionalContext(
                        merging: ["add_generation_prompt": false]
                    )
                )
            }
        } catch {
            Log.agent.warning(
                "Stored token sequence measurement failed — error=\(error.localizedDescription)"
            )
            return nil
        }
    }

    // MARK: - Boundary executor

    /// What the boundary executor produced: the tuner record when the leaf
    /// survived, and the admission's store diagnostics for the phase to tally
    /// into the per-request trace (nil when no admission was attempted).
    struct BoundaryCapture: Sendable {
        let leafStore: AlphaTuner.LeafStore?
        let admission: PrefixCacheManager.StoreDiagnostics?
    }

    // Evolving MVP mid-refactor (see CLAUDE.md); structural limit kept lenient — splitting deferred.
    // swiftlint:disable function_parameter_count
    /// Restore the boundary snapshot, prefill the residual stored-token suffix,
    /// capture a `.leaf`, and admit it under the given token path. The pure
    /// model-affine executor for a `.fromBoundary` **Leaf Capture Plan**, shared
    /// by the direct-tool and canonical-user modes so both align to the
    /// structured template render, not the raw generated bytes.
    ///
    /// The **Leaf Admission Builder** only emits `.fromBoundary` when
    /// `storedTokens.count > boundary.tokenOffset`, so the residual is non-empty
    /// and no caller-side trim is required. The leaf is captured at
    /// `storedTokens.count` after a clean extension prefill, which works because
    /// each cache type's `update(...)` extends its own state at the absolute
    /// offset.
    private static func captureStructuredLeafFromBoundary(
        sessions: any ModelSessionProviding,
        storedTokens: [Int],
        boundarySnapshot: HybridCacheSnapshot,
        positionAnchorRopeDelta: Int?,
        partitionKey: CachePartitionKey,
        prefillStepSize: Int,
        tokenNDim: Int,
        requestID: UUID,
        prefixCache: PrefixCacheManager,
        diagnosticsContext: PrefixCacheDiagnostics.Context,
        ssdEnabled: Bool,
        storeStage: String,
        captureStage: String,
        admissionStage: String,
        captureSource: String
    ) async -> BoundaryCapture {
        // swiftlint:enable function_parameter_count
        // The residual is guaranteed non-empty by the builder's offset guard
        // (it only emits `.fromBoundary` when `storedTokens.count > tokenOffset`).
        let boundaryOffset = boundarySnapshot.tokenOffset

        let extensionBase = await ServerCompletion.resolveExtensionBase(
            ssdEnabled: ssdEnabled,
            tokens: storedTokens,
            partitionKey: partitionKey,
            prefixCache: prefixCache
        )

        do {
            return try await sessions.withSession { session in
                let restoredCache = try session.restore(boundarySnapshot)

                let residual = Array(storedTokens[boundaryOffset...])
                let prefillStart = Date.timeIntervalSinceReferenceDate
                // Qwen3.5 is a `Qwen3_5ForConditionalGeneration` (VLM)
                // whose `prepare` indexes tokens with two axes
                // (`y[0..., ..<step]`) — 1D crashes in `getRopeIndex` on
                // `inputIds.dim(1)`. Pure LLMs use the default
                // `LLMModel.prepare`, which adds the batch dim itself via
                // `.newAxis` and would promote a pre-batched 2D chunk to
                // 3D. Match the processor's original rank.
                let flatInput = MLXArray(residual.map { Int32($0) })
                let inputArr =
                    tokenNDim >= 2
                    ? flatInput.expandedDimensions(axis: 0)
                    : flatInput
                _ = try prefixCache.storageActivityGate.withPrefillMarked {
                    try session.prefill(
                        text: .init(tokens: inputArr, mask: nil),
                        cache: restoredCache,
                        checkpoints: [:],
                        checkpointBaseOffset: boundaryOffset,
                        prefillStepSize: prefillStepSize,
                        consumeAll: true,
                        initialState: positionAnchorRopeDelta.map(PositionAnchor.seededState),
                        evalPolicy: .pipelined
                    )
                }
                let prefillMs = Date.timeIntervalSinceReferenceDate - prefillStart

                guard
                    let leaf = session.captureSnapshot(
                        cache: restoredCache,
                        offset: storedTokens.count,
                        type: .leaf
                    )
                else {
                    diagnosticsContext.logSkip(
                        stage: captureStage,
                        reason: "unsupported-cache-type"
                    )
                    return BoundaryCapture(leafStore: nil, admission: nil)
                }
                Log.agent.info(
                    "\(captureSource) captured — offset=\(leaf.tokenOffset) "
                        + "residualTokens=\(residual.count) "
                        + "prefillMs=\(String(format: "%.3f", prefillMs * 1000)) "
                        + "storedLen=\(storedTokens.count)"
                )

                let admission = await ServerCompletion.admitStructuredLeaf(
                    leaf,
                    storedTokens: storedTokens,
                    storage: ServerCompletion.snapshotAdmissionStorage(
                        for: leaf,
                        ssdEnabled: ssdEnabled,
                        extending: extensionBase
                    ),
                    partitionKey: partitionKey,
                    requestID: requestID,
                    prefixCache: prefixCache,
                    diagnostics: diagnosticsContext,
                    admissionStage: admissionStage,
                    captureSource: captureSource
                )
                Memory.clearCache()
                guard admission.survived else {
                    return BoundaryCapture(leafStore: nil, admission: admission.store)
                }
                return BoundaryCapture(
                    leafStore: AlphaTuner.LeafStore(
                        storedTokens: storedTokens,
                        bytes: leaf.memoryBytes
                    ),
                    admission: admission.store
                )
            }
        } catch {
            diagnosticsContext.logSkip(
                stage: storeStage,
                reason: "prefill-threw",
                level: .warning,
                extraFields: [("error", error.localizedDescription)]
            )
            return BoundaryCapture(leafStore: nil, admission: nil)
        }
    }
}
