//
//  LeafStorePhase+Executors.swift
//  tesseract
//
//  The model-affine executors of the **Leaf Store** phase — `LeafStorePhase.run`
//  decides, these bring the cache to the state its leaf is stored at and hand
//  it to the leaf's **Leaf Admission** (ADR-0078). Three ways to bring the
//  cache to snapshot, one shared tail (`admitLeaf`):
//  - live: the finished turn's final cache at its own offset, under the
//    fed path, no prefill (**Live Leaf Capture** — the fast path)
//  - direct: the same live cache under a non-thinking template's canonical
//    stored path, behind the reusable-state and normalization-trim guards of
//    the render-trusting path (the boundary route's non-thinking arm)
//  - boundary: restore the boundary snapshot and re-prefill the canonical
//    residual
//

import Foundation
import MLX
import MLXLMCommon

nonisolated extension LeafStorePhase {
    /// The diagnostics labels one leaf path logs under: the `logSkip` stages
    /// of its store, capture and admission steps, and the capture source the
    /// admission's `CaptureEvent` names (also the `mode` field of the live
    /// fallback record).
    struct LeafStages: Sendable {
        let store: String
        let capture: String
        let admission: String
        let source: String

        /// `directLeaf`: the pre-existing labels of the non-thinking live path.
        static let direct = LeafStages(
            store: "leafStore", capture: "leafCapture", admission: "leafAdmission",
            source: "leaf")

        /// The labels the path's **Leaf Admission** logs under.
        var admissionLabels: LeafAdmission.Labels {
            LeafAdmission.Labels(capture: capture, admission: admission, source: source)
        }
    }

    /// What every executor needs beside the cache it brings to snapshot: the
    /// token path the leaf is admitted under and the request's admission
    /// context.
    struct LeafAdmissionContext: Sendable {
        let storedTokens: [Int]
        let partitionKey: CachePartitionKey
        let ssdEnabled: Bool
        let requestID: UUID
        let prefixCache: PrefixCacheManager
        let diagnosticsContext: PrefixCacheDiagnostics.Context
        let stages: LeafStages
        let copyReason: Report.CopyReason?
        let memory: RequestMemoryTelemetry?
        /// The request's **Cache Claim**: a leaf captured by move from the
        /// leased cache checks in through it.
        let claim: CacheClaim
        /// Whether no image reached the model (instance truth, ADR-0070): one
        /// of the facts a finished turn's leaf moves on.
        let isTextOnly: Bool
        /// The request's restore mode (`cold`, `copy`, `failedCopy`,
        /// `handoff`), as its check-out decided it — one input to the stored
        /// leaf's source.
        let restoreMode: String
        /// The turn's maximum advance, for the **Active-Inference
        /// Reserve**'s growth allowance (#522).
        let maximumAdvance: Int

        init(storedTokens: [Int], inputs: Inputs, stages: LeafStages, ssdEnabled: Bool? = nil) {
            let mlxStart = inputs.mlxStart
            self.storedTokens = storedTokens
            partitionKey = inputs.request.facts.partitionKey
            self.ssdEnabled = ssdEnabled ?? inputs.request.facts.ssdEnabled
            requestID = inputs.requestID
            prefixCache = inputs.prefixCache
            diagnosticsContext = inputs.diagnosticsContext
            self.stages = stages
            memory = inputs.memory
            claim = inputs.claim
            isTextOnly = inputs.request.facts.isTextOnly
            restoreMode = mlxStart.restoreMode
            maximumAdvance = mlxStart.maximumAdvance
            copyReason =
                !inputs.request.facts.isTextOnly
                ? .imageKeySpace
                : inputs.request.facts.partitionKey.kvBits != nil ? .quantized : nil
        }

        /// Prepare this leaf's **Leaf Admission** before the session is
        /// entered: an SSD-bound one resolves its extension base here (a
        /// MainActor hop).
        func prepareAdmission() async -> LeafAdmission {
            await LeafAdmission.prepare(
                storedTokens: storedTokens,
                partitionKey: partitionKey,
                reachesSSD: ssdEnabled,
                requestID: requestID,
                prefixCache: prefixCache,
                diagnostics: diagnosticsContext
            )
        }
    }

    /// What an executor produced: the tuner record when the leaf survived,
    /// the admission's eviction tally for the phase to fold into the
    /// per-request trace (nil when no admission was attempted), and the
    /// report fields the drive prints.
    struct LeafCapture: Sendable {
        var leafStore: AlphaTuner.LeafStore?
        /// Already logged by the admission; the phase merges it without
        /// logging again.
        var evictionTally: CompletionTraceAccumulator?
        /// Why no leaf was captured, when the executor decided that itself.
        var skipReason: String?
        /// The captured leaf's token offset (nil when capture never ran).
        var leafOffset: Int?
        /// Tokens re-prefilled from the boundary; `0` on the live and
        /// direct paths.
        var residualTokens = 0
        var handedOff = false
        var copyReason: Report.CopyReason?
        /// Bytes the #534 compaction freed before the capture (`0` below threshold).
        var compactedBytes = 0
        var timings = Timings()
    }

    // MARK: - Live executor

    /// Capture the live final cache at the length of `context.storedTokens`
    /// and admit it under those ids — the fed path cut at the cache's
    /// offset, the fast path under every mode (**Live Leaf Capture**). No
    /// restore, no prefill: the generation loop has been
    /// awaited by the drive, so the array is quiescent (ADR-0006), and the
    /// eligible text leaf takes ownership inside a Metal-affine Model Session.
    /// `move: false` lends the cache instead, so the leaf is always copied.
    /// `path` is the report path the caller runs under (`.live`, or
    /// `.direct` through the direct executor); it names the stored leaf's
    /// source for the reserve.
    static func captureLiveLeaf(
        sessions: any ModelSessionProviding,
        mlxStartBox: UnsafeSendableBox<HTTPPrefixCacheGeneration>,
        context: LeafAdmissionContext,
        move: Bool = true,
        path: Report.Path
    ) async -> LeafCapture {
        let admission = await context.prepareAdmission()
        return await sessions.withSession { session in
            // `finalCache` is non-`Sendable` `[any KVCache]` — reached
            // through the boxed generation instead of a direct capture.
            let generation = mlxStartBox.value
            let cache: LeafAdmission.Cache =
                move
                ? .finishedTurn(
                    generation.finalCacheOwner, claim: context.claim,
                    textOnly: context.isTextOnly, arm: generation.speculativeArm)
                : .lent(generation.finalCache)
            return await admitLeaf(
                cache,
                preparing: generation.finalCache,
                admission: admission,
                path: path,
                session: session,
                residualTokens: 0,
                context: context,
                timings: Timings()
            )
        }
    }

    // MARK: - Direct executor

    /// The `directLeaf` executor (non-thinking templates): the live capture
    /// behind the two guards the render-trusting path needs — the cache must
    /// hold reusable state, and normalization must not have shortened the
    /// stored conversation.
    static func captureDirectLeaf(
        sessions: any ModelSessionProviding,
        mlxStartBox: UnsafeSendableBox<HTTPPrefixCacheGeneration>,
        context: LeafAdmissionContext
    ) async -> LeafCapture {
        // The module owns the cache array the generation ran on; the loop's
        // completion task has been awaited by the drive, so the array is no
        // longer being mutated (ADR-0006 — this read replaced the fork's
        // FinalizedKVCacheHandle hand-off).
        guard !Task.isCancelled else {
            return LeafCapture(skipReason: "cancelled")
        }
        let finalCache = mlxStartBox.value.finalCache
        let diagnostics = context.diagnosticsContext

        guard httpPrefixCacheHasReusableState(finalCache) else {
            diagnostics.logSkip(
                stage: "store",
                reason: "no-reusable-cache-state",
                extraFields: [("cacheOffsets", "\(httpPrefixCacheOffsets(finalCache))")]
            )
            return LeafCapture(skipReason: "no-reusable-cache-state")
        }

        // Offset-alignment guard: if normalization shortened the stored
        // conversation (whitespace-only assistant content → ""), we can only
        // trim attention K/V — Mamba's recurrent state can't be unwound
        // (`canTrimPromptCache` returns `false`). Trimming the cache and
        // capturing it as a leaf produces a snapshot whose attention is
        // aligned to `storedTokens.count` but whose Mamba state is from the
        // full pre-trim offset. On Qwen3.5 the resulting leaf hit perturbs
        // raw logits by ~10 even at trim=1: argmax stays stable (greedy
        // decoding survives), but the rest of the distribution drifts in a
        // way that affects sampled decoding. Since the HTTP server propagates
        // the request's `temperature`/`top_p` and we can't predict future
        // request sampling params at store time, the safe choice is to skip
        // the leaf store entirely when normalization would require any trim.
        // Lost cache hits on whitespace-normalized conversations are the
        // trade-off for sampler-agnostic correctness. Verified by
        // `HybridCacheCorrectnessRunner` test 9 — see the
        // `leafHitWithNormalizationDivergence...` diagnostics for the
        // empirical drift measurements.
        let actualCacheOffset = httpPrefixCacheReportedTokenCount(finalCache)
        let canonicalCount = context.storedTokens.count
        if actualCacheOffset > canonicalCount {
            diagnostics.logSkip(
                stage: "leafStore",
                reason: "normalization-trim",
                extraFields: [
                    ("trimAmount", "\(actualCacheOffset - canonicalCount)"),
                    ("offsetBefore", "\(actualCacheOffset)"),
                    ("canonicalCount", "\(canonicalCount)"),
                ]
            )
            return LeafCapture(skipReason: "normalization-trim")
        }

        return await captureLiveLeaf(
            sessions: sessions, mlxStartBox: mlxStartBox, context: context, move: false,
            path: .direct)
    }

    // MARK: - Boundary executor

    /// Restore the boundary snapshot, prefill the residual stored-token
    /// suffix, and hand the restored cache to the leaf's admission under the
    /// stored path. The model-affine executor for a `.fromBoundary` **Leaf
    /// Capture Plan**, shared by the direct-tool and canonical-user modes so
    /// both align to the structured template render, not the raw generated
    /// bytes.
    ///
    /// The **Leaf Admission Builder** only emits `.fromBoundary` when
    /// `storedTokens.count > boundary.tokenOffset`, so the residual is
    /// non-empty and no caller-side trim is required. The leaf is captured
    /// at `storedTokens.count` after a clean extension prefill, which works
    /// because each cache type's `update(...)` extends its own state at the
    /// absolute offset.
    static func captureStructuredLeafFromBoundary(
        sessions: any ModelSessionProviding,
        boundarySnapshot: HybridCacheSnapshot,
        backingLeaf: HybridCacheSnapshot?,
        positionAnchorRopeDelta: Int?,
        prefillStepSize: Int,
        tokenNDim: Int,
        context: LeafAdmissionContext
    ) async -> LeafCapture {
        let boundaryOffset = boundarySnapshot.tokenOffset
        let storedTokens = context.storedTokens
        let admission = await context.prepareAdmission()

        do {
            return try await sessions.withSession { session in
                var timings = Timings()
                let restoreStart = Date.timeIntervalSinceReferenceDate
                let restoredCache = try session.restore(boundarySnapshot, backingLeaf: backingLeaf)
                // Exactly the stored length: the leaf below moves these
                // buffers in and keeps them, so `KVCacheGrowth`'s rounding
                // slack is resident tree memory now, not prefill scratch.
                for layer in restoredCache { layer.reserveCapacity(storedTokens.count) }
                timings.restoreSeconds = secondsSince(restoreStart)

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
                _ = try context.prefixCache.storageActivityGate.withPrefillMarked {
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
                timings.prefillSeconds = secondsSince(prefillStart)

                // The restored cache is this request's own: `session.restore`
                // deep-copies every layer out of the tree (a Prefix-View
                // Checkpoint's slices of its Backing Leaf included) and the
                // residual prefill wrote only into it. Nothing else can reach
                // these objects, so it is handed over as owned and the leaf
                // takes them when every layer can move, instead of paying a
                // second full-KV deep copy — the boundary turn's memory peak,
                // and the `.copied` body that made the next turn restore by
                // copy. ADR-0064's one-owner rule holds: after the move the
                // leaf holds the only reference.
                return await admitLeaf(
                    .owned(restoredCache),
                    preparing: restoredCache,
                    admission: admission,
                    path: .boundary,
                    session: session,
                    residualTokens: residual.count,
                    context: context,
                    timings: timings
                )
            }
        } catch {
            context.diagnosticsContext.logSkip(
                stage: context.stages.store,
                reason: "prefill-threw",
                level: .warning,
                extraFields: [("error", error.localizedDescription)]
            )
            return LeafCapture(skipReason: "prefill-threw")
        }
    }

    // MARK: - Shared tail

    /// Inside a Model Session: compact the cache the leaf is taken from
    /// (#534), hand it to the leaf's **Leaf Admission** — which captures it,
    /// checks a moved leaf in through the request's claim before anything is
    /// extracted, admits it and classifies what that evicted — and release the
    /// MLX buffer pool so it doesn't accumulate transient prefill intermediates
    /// across requests. `timings` carries the stages the caller already ran.
    private static func admitLeaf(
        _ cache: LeafAdmission.Cache,
        preparing target: [any KVCache],
        admission: LeafAdmission,
        path: Report.Path,
        session: any ModelSession,
        residualTokens: Int,
        context: LeafAdmissionContext,
        timings: Timings
    ) async -> LeafCapture {
        var timings = timings
        // #534: a check-in keeps whatever capacity the growth granule rounded
        // up, and a returned leaf carries a rewound generation's rows; compact
        // above the threshold before the capture takes the arrays.
        let compaction = AttentionCapacityCompaction.compactIfNeeded(target)
        context.memory?.mark(
            .capturingLeaf,
            facts: RequestMemoryTelemetry.cacheFacts(target).merging(
                ["leafCompactedBytes": "\(compaction.freedBytes)"], uniquingKeysWith: { $1 }))
        let outcome = await admission.admit(
            cache,
            in: session,
            labels: context.stages.admissionLabels,
            turn: LeafAdmission.Turn(
                path: path, restoreMode: context.restoreMode,
                maximumAdvance: context.maximumAdvance),
            memory: context.memory
        )
        switch outcome {
        case .notCaptured(let reason):
            return LeafCapture(skipReason: reason, residualTokens: residualTokens, timings: timings)
        case .returned(let cause, let capture):
            logCaptured(capture, residualTokens: residualTokens, context: context)
            return LeafCapture(
                skipReason: cause == .cancelled ? "cancelled" : "lease-return-refused")
        case .admitted(let admitted):
            timings.captureSeconds = admitted.capture.seconds
            timings.payloadSeconds = admitted.payloadSeconds
            timings.admitSeconds = admitted.admitSeconds
            logCaptured(admitted.capture, residualTokens: residualTokens, context: context)
            Memory.clearCache()
            return LeafCapture(
                leafStore: admitted.survived
                    ? AlphaTuner.LeafStore(
                        storedTokens: context.storedTokens, bytes: admitted.capture.bytes)
                    : nil,
                evictionTally: admitted.tally,
                leafOffset: admitted.capture.offset,
                residualTokens: residualTokens,
                handedOff: admitted.capture.handedOff,
                copyReason: context.copyReason,
                compactedBytes: compaction.freedBytes,
                timings: timings
            )
        }
    }

    private static func logCaptured(
        _ capture: LeafAdmission.Capture, residualTokens: Int, context: LeafAdmissionContext
    ) {
        Log.agent.info(
            "\(context.stages.source) captured — offset=\(capture.offset) "
                + "residualTokens=\(residualTokens) "
                + "captureMs=\(PrefixCacheDiagnostics.milliseconds(capture.seconds)) "
                + "storedLen=\(context.storedTokens.count)"
        )
    }
}
