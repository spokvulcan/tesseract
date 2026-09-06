//
//  LeafStorePhase+Executors.swift
//  tesseract
//
//  The model-affine executors of the **Leaf Store** phase: the direct
//  (non-thinking template) capture of the live final cache, and the boundary
//  restore + canonical residual re-prefill. `LeafStorePhase.run` decides;
//  these perform the Metal capture and admit. The **Live Leaf Capture**
//  executor lives beside its decision in `LeafStorePhase+LiveLeaf.swift`.
//

import Foundation
import MLX
import MLXLMCommon

nonisolated extension LeafStorePhase {
    // MARK: - Direct executor

    /// The `directLeaf` executor (non-thinking templates): snapshot the live
    /// final cache at the stored path's length and admit it. The cache array
    /// is quiescent — the drive awaited the loop's completion task (ADR-0006).
    static func captureDirectLeaf(
        mlxStartBox: UnsafeSendableBox<HTTPPrefixCacheGeneration>,
        sessions: any ModelSessionProviding,
        storedTokens: [Int],
        requestID: UUID,
        prefixCache: PrefixCacheManager,
        diagnosticsContext: PrefixCacheDiagnostics.Context
    ) async throws -> LeafCapture {
        // The module owns the cache array the generation ran on; the loop's
        // completion task has been awaited by the drive, so the array is no
        // longer being mutated (ADR-0006 — this read replaced the fork's
        // FinalizedKVCacheHandle hand-off).
        guard !Task.isCancelled else {
            return LeafCapture(skipReason: "cancelled")
        }
        let mlxStart = mlxStartBox.value
        let finalCache = mlxStart.finalCache

        let cacheOffsets = httpPrefixCacheOffsets(finalCache)
        guard httpPrefixCacheHasReusableState(finalCache) else {
            diagnosticsContext.logSkip(
                stage: "store",
                reason: "no-reusable-cache-state",
                extraFields: [("cacheOffsets", "\(cacheOffsets)")]
            )
            return LeafCapture(skipReason: "no-reusable-cache-state")
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
            return LeafCapture(skipReason: "normalization-trim")
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
        var timings = Timings()
        let (maybeLeaf, maybeStorage, sessionTimings):
            (HybridCacheSnapshot?, SnapshotAdmission.Storage?, Timings) =
                try await sessions.withSession { session in
                    var timings = Timings()
                    let cache = mlxStartBox.value.finalCache
                    let captureStart = Date.timeIntervalSinceReferenceDate
                    guard
                        let snap = session.captureSnapshot(
                            cache: cache,
                            offset: storedTokens.count,
                            type: .leaf
                        )
                    else {
                        return (nil, nil, timings)
                    }
                    timings.captureMs = millisecondsSince(captureStart)
                    let payloadStart = Date.timeIntervalSinceReferenceDate
                    let storage = ServerCompletion.snapshotAdmissionStorage(
                        for: snap,
                        ssdEnabled: ssdEnabled,
                        extending: extensionBase
                    )
                    timings.payloadMs = millisecondsSince(payloadStart)
                    return (snap, storage, timings)
                }
        timings = sessionTimings
        guard let leafSnapshot = maybeLeaf, let leafStorage = maybeStorage else {
            diagnosticsContext.logSkip(
                stage: "leafCapture",
                reason: "unsupported-cache-type",
                extraFields: [("cacheOffsets", "\(cacheOffsets)")]
            )
            return LeafCapture(skipReason: "unsupported-cache-type", timings: timings)
        }

        // Admission + eviction/supersession classification through the one
        // shared admit (the same path the boundary executor and the
        // speculative pass use), tallied into the per-request trace.
        let admitStart = Date.timeIntervalSinceReferenceDate
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
        timings.admitMs = millisecondsSince(admitStart)

        // Release the MLX free buffer pool back to the OS so it doesn't
        // accumulate transient prefill intermediates across requests.
        Memory.clearCache()
        return LeafCapture(
            leafStore: admission.survived
                ? AlphaTuner.LeafStore(storedTokens: storedTokens, bytes: leafSnapshot.memoryBytes)
                : nil,
            admission: admission.store,
            leafOffset: leafSnapshot.tokenOffset,
            timings: timings
        )
    }

    // MARK: - Boundary executor

    /// What an executor produced: the tuner record when the leaf survived,
    /// the admission's store diagnostics for the phase to tally into the
    /// per-request trace (nil when no admission was attempted), and the
    /// report fields the drive prints.
    struct LeafCapture: Sendable {
        var leafStore: AlphaTuner.LeafStore?
        var admission: PrefixCacheManager.StoreDiagnostics?
        /// Why no leaf was captured, when the executor decided that itself.
        var skipReason: String?
        /// The captured leaf's token offset (nil when capture never ran).
        var leafOffset: Int?
        /// Tokens re-prefilled from the boundary; `0` on the live and
        /// direct paths.
        var residualTokens = 0
        var timings = Timings()
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
    static func captureStructuredLeafFromBoundary(
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
    ) async -> LeafCapture {
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
                var timings = Timings()
                let restoreStart = Date.timeIntervalSinceReferenceDate
                let restoredCache = try session.restore(boundarySnapshot)
                timings.restoreMs = millisecondsSince(restoreStart)

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
                timings.prefillMs = millisecondsSince(prefillStart)

                let captureStart = Date.timeIntervalSinceReferenceDate
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
                    return LeafCapture(
                        leafStore: nil, admission: nil,
                        residualTokens: residual.count, timings: timings)
                }
                timings.captureMs = millisecondsSince(captureStart)
                Log.agent.info(
                    "\(captureSource) captured — offset=\(leaf.tokenOffset) "
                        + "residualTokens=\(residual.count) "
                        + "prefillMs=\(String(format: "%.3f", timings.prefillMs)) "
                        + "storedLen=\(storedTokens.count)"
                )

                let payloadStart = Date.timeIntervalSinceReferenceDate
                let storage = ServerCompletion.snapshotAdmissionStorage(
                    for: leaf,
                    ssdEnabled: ssdEnabled,
                    extending: extensionBase
                )
                timings.payloadMs = millisecondsSince(payloadStart)
                let admitStart = Date.timeIntervalSinceReferenceDate
                let admission = await ServerCompletion.admitStructuredLeaf(
                    leaf,
                    storedTokens: storedTokens,
                    storage: storage,
                    partitionKey: partitionKey,
                    requestID: requestID,
                    prefixCache: prefixCache,
                    diagnostics: diagnosticsContext,
                    admissionStage: admissionStage,
                    captureSource: captureSource
                )
                timings.admitMs = millisecondsSince(admitStart)
                Memory.clearCache()
                guard admission.survived else {
                    return LeafCapture(
                        leafStore: nil, admission: admission.store,
                        leafOffset: leaf.tokenOffset, residualTokens: residual.count,
                        timings: timings)
                }
                return LeafCapture(
                    leafStore: AlphaTuner.LeafStore(
                        storedTokens: storedTokens,
                        bytes: leaf.memoryBytes
                    ),
                    admission: admission.store,
                    leafOffset: leaf.tokenOffset,
                    residualTokens: residual.count,
                    timings: timings
                )
            }
        } catch {
            diagnosticsContext.logSkip(
                stage: storeStage,
                reason: "prefill-threw",
                level: .warning,
                extraFields: [("error", error.localizedDescription)]
            )
            return LeafCapture(leafStore: nil, admission: nil)
        }
    }
}
