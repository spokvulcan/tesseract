import Foundation
import MLX
import MLXLMCommon

/// **Speculative Canonical Prefill** (CONTEXT.md, ADR-0009): after a
/// stop-finish answer under a thinking template, the canonical user leaf ends
/// at the **Think-Strip Rewind** divergence — everything the template will
/// strip from the post-last-user span is invalidated, and without help the
/// *next* user request re-prefills it interactively (the felt "cache miss
/// after tool use", issue #76). This module re-prefills that span in the
/// background instead, while the next event is human-paced and the GPU
/// otherwise idle: restore the just-admitted canonical leaf, extend it along
/// the future shared token path (`LeafAdmissionBuilder.futureSharedPrefix`),
/// and admit a deeper leaf the next request hits directly.
///
/// Lifecycle is owned by `ServerCompletion`: scheduled only when the module
/// is quiescent, preempted (task cancellation, observed between prefill
/// chunks) by every new generation entry, and drained on unload. Preempting
/// entries cancel-and-await the pass: a preempted pass settles before
/// yielding — progress at or above `minimumPreemptCaptureTokens` is admitted
/// as a RAM-only partial leaf (capture-on-preempt, so the preempting request
/// restores it instead of re-prefilling the same span), anything less is
/// dropped, leaving the cache exactly as the canonical leaf left it.
nonisolated enum SpeculativeCanonicalPrefill {

    /// Stop the admitted path this many tokens short of the probe LCP. The
    /// LCP's final tokens sit at the template→user-content seam, where BPE
    /// can merge template text with the first content character into a token
    /// the real next render may not contain. Undershooting costs the next
    /// request a few header tokens of prefill; overshooting orphans the leaf
    /// entirely (radix lookups walk exact token paths) — and the admission
    /// supersedes the canonical leaf, so there is no shallower fallback.
    static let futureBoundarySafetyMarginTokens = 2

    /// Skip spans whose background pass would save fewer prefill tokens than
    /// this. Below it the next request's re-prefill is already sub-second —
    /// not worth a GPU wake-up plus a leaf admission (and its SSD write).
    static let minimumResidualTokens = 512

    /// A preempted pass keeps its progress only at or above this many
    /// prefilled tokens. The partial capture delays the preempting request
    /// by roughly one RAM-only leaf capture (~0.5 s); below this the delay
    /// rivals the re-prefill it would save, above it the trade is a
    /// guaranteed net win (2,048 tokens re-prefill interactively in ~2.5 s).
    static let minimumPreemptCaptureTokens = 2_048

    /// **Stretch Abandonment**'s timer trigger (issue #100): a tool-calls
    /// finish with no follow-up request inside this window looks abandoned
    /// and seeds the canonical pass. Long enough that an agent loop's
    /// next tool-result request lands well inside it (those arrive in
    /// milliseconds-to-seconds), short enough to leave most of a human
    /// typing pause as speculation runway.
    static let stretchAbandonmentIdleWindow: Duration = .seconds(5)

    /// Everything the background pass needs, captured from the originating
    /// request's post-generation context. Built by the drive task only after
    /// the canonical leaf was admitted — that leaf is the restore base this
    /// pass extends.
    struct Seed: Sendable {
        let keySpace: CacheKeySpace
        let partitionKey: CachePartitionKey
        let prefillStepSize: Int
        let ssdEnabled: Bool
        let seedsPositionAnchor: Bool
        /// Offset of the canonical leaf this pass extends. Drives the
        /// worth-it threshold and the diagnostics' rewind-span field; the
        /// actual restore boundary is re-resolved against the live tree.
        let canonicalLeafOffset: Int
        /// How long the scheduled pass waits before touching the GPU.
        /// `.zero` for the stop-finish and abort triggers; the
        /// **Stretch Abandonment** idle window for a tool-calls finish —
        /// a follow-up request inside the window preempts the sleeping
        /// pass before it does any work (issue #100).
        let idleDelay: Duration
        /// `true` for abandonment-triggered passes (issue #100): the
        /// speculated spine admits RAM-only (ADR-0009 durability), so a
        /// false alarm — the tool result arrives after all — costs zero
        /// SSD writes. The rewind landing persists the branch via the
        /// existing self-heal full write.
        let ramOnlySpine: Bool
        /// The originating request's diagnostics context, reused so the
        /// speculative events correlate with the turn that scheduled them.
        let diagnostics: PrefixCacheDiagnostics.Context
        /// The future-shared-path probe (`LeafAdmissionBuilder
        /// .futureSharedPrefix`), spawned by `makeSeed` — the probe task is
        /// the only holder of the conversation inputs, so the seed cannot
        /// drift into a second source of truth for the path.
        let futureSharedPrefixProbe:
            Task<Result<[Int], CacheKeySpace.TranslationFailure>?, any Error>

        /// Cancel the probe when this seed will never be scheduled (failed
        /// leaf store, not-idle schedule, unloaded model) — the builder's
        /// cooperative checks stop its remaining render work.
        func discard() {
            futureSharedPrefixProbe.cancel()
        }
    }

    // Evolving MVP mid-refactor (see CLAUDE.md); structural limit kept lenient — splitting deferred.
    // swiftlint:disable function_parameter_count
    /// Build a seed for the turn that just planned a canonical leaf,
    /// spawning the future-shared-path probe immediately. Call *before* the
    /// GPU-side leaf store begins so the CPU render+tokenize overlaps it
    /// (#76's earlier start) — the pass then spends none of its human-paced
    /// window on its own stage 1. A seed that will never be scheduled should
    /// be `discard()`ed so the probe stops at its next cooperative check.
    static func makeSeed(
        storedConversation: HTTPPrefixCacheConversation,
        toolSpecs: [ToolSpec]?,
        tokenizer: any Tokenizer,
        keySpace: CacheKeySpace,
        partitionKey: CachePartitionKey,
        prefillStepSize: Int,
        ssdEnabled: Bool,
        seedsPositionAnchor: Bool,
        canonicalLeafOffset: Int,
        renderContext: TemplateRenderContext = .canonical,
        idleDelay: Duration = .zero,
        ramOnlySpine: Bool = false,
        diagnostics: PrefixCacheDiagnostics.Context
    ) -> Seed {
        // swiftlint:enable function_parameter_count
        let probe = Task.detached {
            try LeafAdmissionBuilder.futureSharedPrefix(
                storedConversation: storedConversation,
                toolSpecs: toolSpecs,
                tokenizer: tokenizer,
                keySpace: keySpace,
                renderContext: renderContext
            )
        }
        return Seed(
            keySpace: keySpace,
            partitionKey: partitionKey,
            prefillStepSize: prefillStepSize,
            ssdEnabled: ssdEnabled,
            seedsPositionAnchor: seedsPositionAnchor,
            canonicalLeafOffset: canonicalLeafOffset,
            idleDelay: idleDelay,
            ramOnlySpine: ramOnlySpine,
            diagnostics: diagnostics,
            futureSharedPrefixProbe: probe
        )
    }

    /// The admitted token path for a future shared prefix: the margin-trimmed
    /// probe LCP, or `nil` when the trimmed span past the canonical leaf is
    /// below the worth-it threshold. Pure — unit-tested directly.
    static func admitPath(
        futureSharedPrefix: [Int],
        canonicalLeafOffset: Int
    ) -> [Int]? {
        let end = futureSharedPrefix.count - futureBoundarySafetyMarginTokens
        guard end - canonicalLeafOffset >= minimumResidualTokens else { return nil }
        return Array(futureSharedPrefix[0..<end])
    }

    /// The offset a preempted pass may admit its partial progress at: the
    /// restore boundary plus the consumed prefix of the residual, or `nil`
    /// when the progress is below the capture threshold or not deeper than
    /// the canonical leaf it set out to extend. Pure — unit-tested directly.
    static func preemptCaptureOffset(
        boundaryOffset: Int,
        consumedTokens: Int,
        canonicalLeafOffset: Int
    ) -> Int? {
        guard consumedTokens >= minimumPreemptCaptureTokens else { return nil }
        let offset = boundaryOffset + consumedTokens
        guard offset > canonicalLeafOffset else { return nil }
        return offset
    }

    // Evolving MVP mid-refactor (see CLAUDE.md); structural limit kept lenient — splitting deferred.
    // swiftlint:disable function_body_length
    /// Run one speculative pass to completion or preemption. Cancellation is
    /// observed before every stage and between prefill chunks; a preempted
    /// pass settles (admitting partial progress past the capture threshold,
    /// dropping anything less) and releases the container actor within ~one
    /// chunk plus at most one RAM-only capture. Runs off-actor (like the
    /// drive task); every model-affine step hops through `container.perform`.
    static func run(
        seed: Seed,
        container: ModelContainer,
        prefixCache: PrefixCacheManager
    ) async {
        // swiftlint:enable function_body_length
        let diagnostics = seed.diagnostics

        // 1. GPU-free: the future shared token path (probe-pair LCP),
        //    computed by the drive task concurrently with the canonical
        //    leaf's GPU-side store — usually finished before this pass is
        //    even scheduled. `Task.value` is not a cancellation point for
        //    the waiter, and the preempting entry awaits this pass, so a
        //    preemption must be forwarded to the probe explicitly — its
        //    cooperative checks then end the wait within one render.
        let futurePrefix: [Int]
        do {
            let probed = try await withTaskCancellationHandler {
                try await seed.futureSharedPrefixProbe.value
            } onCancel: {
                seed.futureSharedPrefixProbe.cancel()
            }
            switch probed {
            case .none:
                diagnostics.logSkip(stage: "speculativePrefill", reason: "probe-divergence")
                return
            case .some(.failure(let failure)):
                diagnostics.logSkip(
                    stage: "speculativePrefill",
                    reason: "render-translation-failed",
                    extraFields: [("failure", "\(failure)")]
                )
                return
            case .some(.success(let translated)):
                futurePrefix = translated
            }
        } catch is CancellationError {
            logPreempted(diagnostics, prefilledTokens: 0, residualTokens: 0)
            return
        } catch {
            diagnostics.logSkip(
                stage: "speculativePrefill",
                reason: "tokenization-failed",
                level: .warning,
                extraFields: [("error", error.localizedDescription)]
            )
            return
        }

        // 2. Pure: safety margin + worth-it threshold.
        guard
            let admitPath = admitPath(
                futureSharedPrefix: futurePrefix,
                canonicalLeafOffset: seed.canonicalLeafOffset
            )
        else {
            diagnostics.logSkip(
                stage: "speculativePrefill",
                reason: "below-residual-threshold",
                extraFields: [
                    ("futurePrefixLen", "\(futurePrefix.count)"),
                    ("canonicalLeafOffset", "\(seed.canonicalLeafOffset)"),
                ]
            )
            return
        }

        // 3. Resolve the restore boundary against the live tree — usually
        // the canonical leaf admitted moments ago, but whatever Snapshot
        // Resolution surfaces works (a shallower hit just re-prefills more).
        // Driven inside `container.perform` so an SSD `loadSync` stays
        // off-MainActor (ADR-0001).
        if Task.isCancelled {
            logPreempted(diagnostics, prefilledTokens: 0, residualTokens: 0)
            return
        }
        let resolved = await container.perform { _ in
            await prefixCache.resolve(
                tokens: admitPath,
                promptTokenCount: admitPath.count,
                partitionKey: seed.partitionKey,
                modelFingerprint: seed.partitionKey.modelFingerprint,
                diagnostics: diagnostics
            ).lookup.snapshot
        }
        // The boundary guards mirror the canonical-leaf arm: a usable
        // boundary is inside the path and past the image prefix (the
        // residual doubles as the reprefill input, so it must be image-free).
        guard let boundary = resolved,
            boundary.tokenOffset > 0,
            boundary.tokenOffset < admitPath.count,
            boundary.tokenOffset >= seed.keySpace.minimumWarmOffset
        else {
            diagnostics.logSkip(
                stage: "speculativePrefill",
                reason: "no-resolved-boundary",
                extraFields: [("admitLen", "\(admitPath.count)")]
            )
            return
        }
        let anchorDelta: Int?
        if seed.seedsPositionAnchor {
            guard let delta = seed.keySpace.positionAnchorDelta(upTo: boundary.tokenOffset) else {
                diagnostics.logSkip(
                    stage: "speculativePrefill",
                    reason: "boundary-splits-image-run",
                    extraFields: [("offset", "\(boundary.tokenOffset)")]
                )
                return
            }
            anchorDelta = delta
        } else {
            anchorDelta = nil
        }

        // 4. Restore, then chunked extension prefill. One `container.perform`
        // per chunk with a cancellation check between chunks, so a preempting
        // generation acquires the container actor within ~one chunk; the
        // synchronous per-chunk `eval` keeps that bound real (no pipelined
        // work outliving the perform that enqueued it).
        let residual = Array(admitPath[boundary.tokenOffset...])
        let warm = WarmState()
        let prefillStart = Date.timeIntervalSinceReferenceDate

        let restoreOK = await container.perform { _ in
            do {
                warm.cache = try boundary.restore()
                return true
            } catch {
                warm.failure = error.localizedDescription
                return false
            }
        }
        guard restoreOK else {
            diagnostics.logSkip(
                stage: "speculativePrefill",
                reason: "restore-failed",
                level: .warning,
                extraFields: [("error", warm.failure ?? "unknown")]
            )
            return
        }

        let stepSize = max(seed.prefillStepSize, 1)
        while warm.consumed < residual.count, !Task.isCancelled {
            let range = warm.consumed..<min(warm.consumed + stepSize, residual.count)
            await container.perform { context in
                // Both model classes take a batched 2D chunk (PrefillExecutor
                // adds the batch axis the same way for 1D-prepare LLMs).
                let input = LMInput.Text(
                    tokens: MLXArray(residual[range].map { Int32($0) })
                        .expandedDimensions(axis: 0),
                    mask: nil
                )
                let initialState =
                    warm.consumed == 0
                    ? anchorDelta.map { PositionAnchor.seededState(ropeDelta: $0) }
                    : warm.chunkState
                let output = context.model(
                    input,
                    cache: warm.cache,
                    state: initialState
                )
                warm.chunkState = output.state
                eval(warm.cache)
            }
            warm.consumed += range.count
        }
        let prefillSeconds = Date.timeIntervalSinceReferenceDate - prefillStart

        if Task.isCancelled {
            // Mid-span or just after the last chunk — the capture economics
            // are identical, and the RAM-only admission keeps the preempting
            // entry's wait bounded.
            await settlePreemption(
                seed: seed,
                container: container,
                prefixCache: prefixCache,
                admitPath: admitPath,
                boundaryOffset: boundary.tokenOffset,
                warm: warm,
                prefillStart: prefillStart
            )
            return
        }

        // 5. Capture the deeper leaf and admit it. The admission supersedes
        // the canonical leaf on the same path — the speculative leaf covers
        // everything it covered, plus the rewind span. The trailing trim
        // returns the pass's working buffers to the OS for the human-paced
        // idle window (captureAndAdmit released the warm cache into the pool).
        await captureAndAdmit(
            storedTokens: admitPath,
            warm: warm,
            seed: seed,
            container: container,
            prefixCache: prefixCache,
            boundaryOffset: boundary.tokenOffset,
            prefillSeconds: prefillSeconds,
            preempted: false
        )
        Memory.clearCache()
    }

    /// Settle a preempted pass before it yields the container: progress at
    /// or above `minimumPreemptCaptureTokens` is admitted as a partial leaf
    /// (the chunk loop keeps the warm cache exactly chunk-aligned to the
    /// admit path, so no trim is needed), anything less is dropped. The
    /// preempting entry awaits this — the whole settle is bounded by one
    /// RAM-only capture. Deliberately no `Memory.clearCache()` here: the
    /// pass's buffers go back to the MLX pool when the warm state dies, and
    /// the preempting generation reuses them immediately — trimming the pool
    /// inside its awaited window would be free-then-reallocate.
    private static func settlePreemption(
        seed: Seed,
        container: ModelContainer,
        prefixCache: PrefixCacheManager,
        admitPath: [Int],
        boundaryOffset: Int,
        warm: WarmState,
        prefillStart: TimeInterval
    ) async {
        guard
            let offset = preemptCaptureOffset(
                boundaryOffset: boundaryOffset,
                consumedTokens: warm.consumed,
                canonicalLeafOffset: seed.canonicalLeafOffset
            )
        else {
            logPreempted(
                seed.diagnostics,
                prefilledTokens: warm.consumed,
                residualTokens: admitPath.count - boundaryOffset
            )
            return
        }
        await captureAndAdmit(
            storedTokens: Array(admitPath[0..<offset]),
            warm: warm,
            seed: seed,
            container: container,
            prefixCache: prefixCache,
            boundaryOffset: boundaryOffset,
            prefillSeconds: Date.timeIntervalSinceReferenceDate - prefillStart,
            preempted: true
        )
    }

    /// Shared capture tail for both exits: snapshot the warm cache at
    /// `storedTokens.count` and admit it via the structured-leaf admission
    /// (`ServerCompletion.admitStructuredLeaf`). Completed passes persist
    /// RAM+SSD like any leaf; preempted partial leaves are **RAM-only** —
    /// their sole purpose is the imminent preempting request, which
    /// supersedes them with its own SSD-backed leaf moments later, so the
    /// SSD payload extraction (the dominant capture cost) and the disk
    /// churn are both skipped. Releases the warm cache on every path: the
    /// leaf deep-copied what it needed, and holding the working buffers any
    /// longer would keep their pool memory from the next user.
    private static func captureAndAdmit(
        storedTokens: [Int],
        warm: WarmState,
        seed: Seed,
        container: ModelContainer,
        prefixCache: PrefixCacheManager,
        boundaryOffset: Int,
        prefillSeconds: TimeInterval,
        preempted: Bool
    ) async {
        let diagnostics = seed.diagnostics
        // Preempted partial leaves and abandonment spines are RAM-only
        // and never consult the extension base.
        let extensionBase = await ServerCompletion.resolveExtensionBase(
            ssdEnabled: seed.ssdEnabled && !preempted && !seed.ramOnlySpine,
            tokens: storedTokens,
            partitionKey: seed.partitionKey,
            prefixCache: prefixCache
        )
        await container.perform { _ in
            defer {
                warm.cache = []
                warm.chunkState = nil
            }
            guard
                let leaf = HybridCacheSnapshot.capture(
                    cache: warm.cache,
                    offset: storedTokens.count,
                    type: .leaf
                )
            else {
                diagnostics.logSkip(
                    stage: "speculativePrefill",
                    reason: "unsupported-cache-type"
                )
                return
            }
            let storage: SnapshotAdmission.Storage =
                preempted || seed.ramOnlySpine
                ? .ramOnly
                : ServerCompletion.snapshotAdmissionStorage(
                    for: leaf,
                    ssdEnabled: seed.ssdEnabled,
                    extending: extensionBase
                )
            let survived = await ServerCompletion.admitStructuredLeaf(
                leaf,
                storedTokens: storedTokens,
                storage: storage,
                partitionKey: seed.partitionKey,
                requestID: diagnostics.requestID,
                prefixCache: prefixCache,
                diagnostics: diagnostics,
                admissionStage: "speculativePrefill",
                captureSource: preempted ? "speculativePartialLeaf" : "speculativeLeaf"
            )
            guard survived else { return }

            diagnostics.log(
                PrefixCacheDiagnostics.SpeculativePrefillEvent(
                    targetOffset: storedTokens.count,
                    boundaryOffset: boundaryOffset,
                    prefilledTokens: warm.consumed,
                    prefillSeconds: prefillSeconds,
                    rewindSpanTokens: storedTokens.count - seed.canonicalLeafOffset,
                    preempted: preempted
                ))
        }
    }

    /// Mutable model-affine state carried across the per-chunk
    /// `container.perform` hops. Only the one speculative task touches it,
    /// and every mutation happens inside a `perform` (Metal-affine scope) —
    /// the box launders the actor hop, never shares.
    private final class WarmState: @unchecked Sendable {
        var cache: [any KVCache] = []
        var chunkState: LMOutput.State?
        var consumed = 0
        var failure: String?
    }

    private static func logPreempted(
        _ diagnostics: PrefixCacheDiagnostics.Context,
        prefilledTokens: Int,
        residualTokens: Int
    ) {
        diagnostics.logSkip(
            stage: "speculativePrefill",
            reason: "preempted",
            extraFields: [
                ("prefilledTokens", "\(prefilledTokens)"),
                ("residualTokens", "\(residualTokens)"),
            ]
        )
    }
}
