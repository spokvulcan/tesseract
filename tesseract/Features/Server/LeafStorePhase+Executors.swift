//
//  LeafStorePhase+Executors.swift
//  tesseract
//
//  The model-affine executors of the **Leaf Store** phase — `LeafStorePhase.run`
//  decides, these perform the Metal capture and admit. Three ways to bring
//  the cache to snapshot, one shared tail (`admitLeaf`):
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

        init(storedTokens: [Int], inputs: Inputs, stages: LeafStages) {
            let mlxStart = inputs.mlxStart
            self.storedTokens = storedTokens
            partitionKey = mlxStart.partitionKey
            ssdEnabled = mlxStart.ssdEnabled
            requestID = inputs.requestID
            prefixCache = inputs.prefixCache
            diagnosticsContext = inputs.diagnosticsContext
            self.stages = stages
            memory = inputs.memory
            copyReason =
                inputs.containsImages || !mlxStart.keySpace.isIdentity
                ? .imageKeySpace
                : mlxStart.partitionKey.kvBits != nil ? .quantized : nil
        }

        /// The SSD extension base for `storedTokens`, resolved before the
        /// session is entered (a MainActor hop).
        func resolveExtensionBase() async -> SnapshotExtension? {
            await ServerCompletion.resolveExtensionBase(
                ssdEnabled: ssdEnabled,
                tokens: storedTokens,
                partitionKey: partitionKey,
                prefixCache: prefixCache
            )
        }
    }

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
        var handedOff = false
        var copyReason: Report.CopyReason?
        var timings = Timings()
    }

    // MARK: - Live executor

    /// Capture the live final cache at the length of `context.storedTokens`
    /// and admit it under those ids — the fed path cut at the cache's
    /// offset, the fast path under every mode (**Live Leaf Capture**). No
    /// restore, no prefill: the generation loop has been
    /// awaited by the drive, so the array is quiescent (ADR-0006), and the
    /// eligible text leaf takes ownership inside a Metal-affine Model Session.
    static func captureLiveLeaf(
        sessions: any ModelSessionProviding,
        mlxStartBox: UnsafeSendableBox<HTTPPrefixCacheGeneration>,
        context: LeafAdmissionContext,
        move: Bool = true
    ) async -> LeafCapture {
        let extensionBase = await context.resolveExtensionBase()
        do {
            return try await sessions.withSession { session in
                // `finalCache` is non-`Sendable` `[any KVCache]` — reached
                // through the boxed generation instead of a direct capture.
                let generation = mlxStartBox.value
                let moving =
                    move && context.copyReason == nil && session.mtpDrafter == nil
                    ? generation.finalCacheOwner : nil
                return await admitLeaf(
                    cache: moving == nil ? generation.finalCache : [],
                    moving: moving,
                    session: session,
                    residualTokens: 0,
                    extensionBase: extensionBase,
                    context: context,
                    timings: Timings()
                )
            }
        } catch {
            context.diagnosticsContext.logSkip(
                stage: context.stages.capture,
                reason: "live-capture-threw",
                level: .warning,
                extraFields: [("error", error.localizedDescription)]
            )
            return LeafCapture(skipReason: "live-capture-threw")
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
            sessions: sessions, mlxStartBox: mlxStartBox, context: context, move: false)
    }

    // MARK: - Boundary executor

    /// Restore the boundary snapshot, prefill the residual stored-token
    /// suffix, capture a `.leaf`, and admit it under the stored path. The
    /// model-affine executor for a `.fromBoundary` **Leaf Capture Plan**,
    /// shared by the direct-tool and canonical-user modes so both align to
    /// the structured template render, not the raw generated bytes.
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
        positionAnchorRopeDelta: Int?,
        prefillStepSize: Int,
        tokenNDim: Int,
        context: LeafAdmissionContext
    ) async -> LeafCapture {
        let boundaryOffset = boundarySnapshot.tokenOffset
        let storedTokens = context.storedTokens
        let extensionBase = await context.resolveExtensionBase()

        do {
            return try await sessions.withSession { session in
                var timings = Timings()
                let restoreStart = Date.timeIntervalSinceReferenceDate
                let restoredCache = try session.restore(boundarySnapshot)
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

                return await admitLeaf(
                    cache: restoredCache,
                    session: session,
                    residualTokens: residual.count,
                    extensionBase: extensionBase,
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

    /// Inside a Model Session: snapshot `cache` at the stored path's length,
    /// derive its admission storage (**Deferred Payload Extraction**: an
    /// extension payload's arrays — the attention suffix slices and the
    /// whole recurrent state — are detached and evaluated here, the host
    /// copy waits for the SSD writer), admit through the one shared owner — the same path
    /// the speculative pass uses — and release the MLX buffer pool so it
    /// doesn't accumulate transient prefill intermediates across requests.
    /// `timings` carries the stages the caller already ran.
    private static func admitLeaf(
        cache: [any KVCache],
        moving: FinalGenerationCache? = nil,
        session: any ModelSession,
        residualTokens: Int,
        extensionBase: SnapshotExtension?,
        context: LeafAdmissionContext,
        timings: Timings
    ) async -> LeafCapture {
        var timings = timings
        let storedTokens = context.storedTokens
        context.memory?.mark(
            .capturingLeaf, facts: RequestMemoryTelemetry.cacheFacts(moving?.cache ?? cache))
        let captureStart = Date.timeIntervalSinceReferenceDate
        guard
            let leaf = moving != nil
                ? moving?.moveSnapshot(offset: storedTokens.count)
                : session.captureSnapshot(
                    cache: cache,
                    offset: storedTokens.count,
                    type: .leaf
                )
        else {
            context.diagnosticsContext.logSkip(
                stage: context.stages.capture,
                reason: "unsupported-cache-type"
            )
            return LeafCapture(
                skipReason: "unsupported-cache-type",
                residualTokens: residualTokens, timings: timings)
        }
        timings.captureSeconds = secondsSince(captureStart)
        Log.agent.info(
            "\(context.stages.source) captured — offset=\(leaf.tokenOffset) "
                + "residualTokens=\(residualTokens) "
                + "captureMs=\(PrefixCacheDiagnostics.milliseconds(timings.captureSeconds)) "
                + "storedLen=\(storedTokens.count)"
        )

        context.memory?.mark(
            .preparingPayload,
            facts: [
                "leafSnapshotArrayBytes": "\(leaf.memoryBytes)",
                "leafCaptureMode": moving == nil ? "copy" : "handoff",
                "requestCacheLayerCountAfterCapture": "\(moving?.cache.count ?? cache.count)",
            ])
        let payloadStart = Date.timeIntervalSinceReferenceDate
        let storage = ServerCompletion.snapshotAdmissionStorage(
            for: leaf,
            ssdEnabled: context.ssdEnabled,
            extending: extensionBase
        )
        timings.payloadSeconds = secondsSince(payloadStart)

        var payloadFacts = ["ssdPayloadMode": "none", "ssdPayloadArrayBytes": "0"]
        if case .ramAndSSD(let payload) = storage {
            payloadFacts = [
                "ssdPayloadMode": payload.extending == nil ? "full" : "extension",
                "ssdPayloadArrayBytes": "\(payload.totalBytes)",
            ]
        }
        context.memory?.mark(.admittingLeaf, facts: payloadFacts)
        let admitStart = Date.timeIntervalSinceReferenceDate
        let admission = await ServerCompletion.admitStructuredLeaf(
            leaf,
            storedTokens: storedTokens,
            storage: storage,
            partitionKey: context.partitionKey,
            requestID: context.requestID,
            prefixCache: context.prefixCache,
            diagnostics: context.diagnosticsContext,
            admissionStage: context.stages.admission,
            captureSource: context.stages.source
        )
        timings.admitSeconds = secondsSince(admitStart)
        Memory.clearCache()
        return LeafCapture(
            leafStore: admission.survived
                ? AlphaTuner.LeafStore(storedTokens: storedTokens, bytes: leaf.memoryBytes)
                : nil,
            admission: admission.store,
            leafOffset: leaf.tokenOffset,
            residualTokens: residualTokens,
            handedOff: moving != nil,
            copyReason: context.copyReason,
            timings: timings
        )
    }
}
