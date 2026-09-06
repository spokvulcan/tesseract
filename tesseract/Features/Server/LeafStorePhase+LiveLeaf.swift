//
//  LeafStorePhase+LiveLeaf.swift
//  tesseract
//
//  The **Live Leaf Capture** executor and its fallback log: the model-affine
//  half of the decision `LiveLeafCapture.decide` makes GPU-free.
//

import Foundation
import MLX
import MLXLMCommon

nonisolated extension LeafStorePhase {
    // MARK: - Live executor

    /// Emit the **Live Leaf Capture** refusal. Eligibility-only reasons
    /// (intervened, image key space, empty turn) are `.info` like every other
    /// expected skip; a render that disagreed with the emission is `.warning`
    /// when the render is expected to be append-stable — the preserve-thinking
    /// render, and every tool stretch — and `.info` when the strip-by-default
    /// canonical render legitimately diverges at the think strip.
    static func logLiveFallback(
        _ reason: LiveLeafCapture.FallbackReason,
        mode: BoundaryLeafMode,
        renderContext: TemplateRenderContext,
        diagnosticsContext: PrefixCacheDiagnostics.Context
    ) {
        let expectedDivergence = mode == .canonical && !renderContext.preservesThinking
        let level: PrefixCacheDiagnostics.Level =
            reason.isEligibilityOnly || expectedDivergence ? .info : .warning
        diagnosticsContext.logSkip(
            stage: "liveLeafCapture",
            reason: reason.wireReason,
            level: level,
            extraFields: reason.logFields + [
                ("mode", leafStages(for: mode).source),
                ("preservesThinking", "\(renderContext.preservesThinking)"),
            ]
        )
    }

    // Evolving MVP mid-refactor (see CLAUDE.md); structural limit kept lenient — splitting deferred.
    // swiftlint:disable function_parameter_count
    /// Capture the live final cache at the decided offset and admit it under
    /// `storedTokens` (the canonical stored path's prefix of that length,
    /// already proven equal to the fed path by `LiveLeafCapture.decide`).
    /// No restore, no prefill: the generation loop has been awaited by the
    /// drive, so the array is quiescent (ADR-0006), and the snapshot is a
    /// deep copy inside a Metal-affine Model Session like the direct path's.
    static func captureLiveLeaf(
        sessions: any ModelSessionProviding,
        mlxStartBox: UnsafeSendableBox<HTTPPrefixCacheGeneration>,
        storedTokens: [Int],
        requestID: UUID,
        prefixCache: PrefixCacheManager,
        diagnosticsContext: PrefixCacheDiagnostics.Context,
        captureStage: String,
        admissionStage: String,
        captureSource: String
    ) async -> LeafCapture {
        // swiftlint:enable function_parameter_count
        let mlxStart = mlxStartBox.value
        let ssdEnabled = mlxStart.ssdEnabled
        let extensionBase = await ServerCompletion.resolveExtensionBase(
            ssdEnabled: ssdEnabled,
            tokens: storedTokens,
            partitionKey: mlxStart.partitionKey,
            prefixCache: prefixCache
        )
        do {
            return try await sessions.withSession { session in
                var timings = Timings()
                let captureStart = Date.timeIntervalSinceReferenceDate
                guard
                    let leaf = session.captureSnapshot(
                        cache: mlxStartBox.value.finalCache,
                        offset: storedTokens.count,
                        type: .leaf
                    )
                else {
                    diagnosticsContext.logSkip(
                        stage: captureStage,
                        reason: "unsupported-cache-type"
                    )
                    return LeafCapture(leafStore: nil, admission: nil, timings: timings)
                }
                timings.captureMs = millisecondsSince(captureStart)
                Log.agent.info(
                    "\(captureSource) captured from live cache — offset=\(leaf.tokenOffset) "
                        + "captureMs=\(String(format: "%.3f", timings.captureMs)) "
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
                    partitionKey: mlxStart.partitionKey,
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
                        leafOffset: leaf.tokenOffset, timings: timings)
                }
                return LeafCapture(
                    leafStore: AlphaTuner.LeafStore(
                        storedTokens: storedTokens,
                        bytes: leaf.memoryBytes
                    ),
                    admission: admission.store,
                    leafOffset: leaf.tokenOffset,
                    timings: timings
                )
            }
        } catch {
            diagnosticsContext.logSkip(
                stage: captureStage,
                reason: "live-capture-threw",
                level: .warning,
                extraFields: [("error", error.localizedDescription)]
            )
            return LeafCapture(leafStore: nil, admission: nil)
        }
    }
}
