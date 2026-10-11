//
//  ResidualPrefill.swift
//  tesseract
//
//  The **Residual Prefill** (ADR-0094): the one way a leaf producer restores
//  a resolved prefix and extends it as text — the Leaf Store's boundary route
//  and the Speculative Canonical Prefill. The usable-restore rule
//  (`PrefillPlanner.usableRestore`) decides whether a snapshot can seed the
//  prefill; this module does everything after that check, inside the
//  producer's **Model Session**, and the producer hands the resulting cache to
//  its **Leaf Admission** (ADR-0078 keeps re-prefill out of the admission).
//

import Foundation
import MLX
import MLXLMCommon

nonisolated enum ResidualPrefill {

    /// A restore the usable-restore rule took: the snapshot, the Backing Leaf
    /// a **Prefix-View Checkpoint** restores through, where the rule placed
    /// it, and the key path the leaf is admitted under — the snapshot holds
    /// its prefix, and the residual past it is text.
    struct Base: Sendable {
        let snapshot: HybridCacheSnapshot
        let backingLeaf: HybridCacheSnapshot?
        let point: PrefillPlanner.RestorePoint
        let path: [Int]
    }

    /// A request-local **Prefix-View Checkpoint** (a transient boundary) as
    /// a restore candidate for `path`: its whole-state layers belong to the
    /// key path of the request that captured it, so another path may use it
    /// only through that prefix, and only past the image prefix. `nil` when
    /// it may not serve.
    static func matchingView(
        _ view: HybridCacheSnapshot?, for path: [Int], keySpace: CacheKeySpace
    ) -> HybridCacheSnapshot? {
        guard let view, view.isPrefixView,
            view.tokenOffset >= keySpace.minimumWarmOffset,
            keySpace.keyPath.count >= view.tokenOffset,
            path.starts(with: keySpace.keyPath.prefix(view.tokenOffset))
        else { return nil }
        return view
    }

    /// Why a restore produced no cache.
    enum Failure: Error {
        /// The snapshot's restore threw, and the failed-restore policy
        /// dropped it.
        case unrestorable(any Error)

        /// What the restore threw.
        var underlying: any Error {
            switch self {
            case .unrestorable(let error): error
            }
        }
    }

    /// Restore `base`'s snapshot (through its Backing Leaf) inside `session`
    /// and reserve the whole path's rows, so neither the residual prefill nor
    /// the leaf's capture grows the cache. A restore that throws runs the
    /// failed-restore policy, which drops the snapshot.
    static func restore(
        _ base: Base,
        partitionKey: CachePartitionKey,
        prefixCache: PrefixCacheManager,
        diagnostics: PrefixCacheDiagnostics.Context,
        in session: any ModelSession
    ) async throws(Failure) -> [any KVCache] {
        let cache: [any KVCache]
        do {
            cache = try session.restore(base.snapshot, backingLeaf: base.backingLeaf)
        } catch {
            await dropUnrestorable(
                base.snapshot, error: error, tokens: base.path, partitionKey: partitionKey,
                prefixCache: prefixCache, diagnostics: diagnostics)
            throw .unrestorable(error)
        }
        for layer in cache { layer.reserveCapacity(base.path.count) }
        return cache
    }

    /// Prefill `path[range]` into `cache` as text through the session's
    /// `prefill` verb: every token consumed, the tokens at the processor's
    /// rank (`tokenNDim`), forwarded from `state` (the base's initial state
    /// for the residual's first tokens). Returns the state the next range
    /// continues from. Cancellation is checked before every chunk. Under
    /// `.checkedSynchronous` a chunk's MLX failure throws; under `.pipelined`
    /// only the closing flush's does (`PrefillExecutor.EvalPolicy`).
    // Mirrors the prefill verb's own parameters.
    // swiftlint:disable:next function_parameter_count
    static func prefill(
        _ cache: [any KVCache],
        path: [Int],
        range: Range<Int>,
        state: LMOutput.State?,
        prefillStepSize: Int,
        tokenNDim: Int,
        evalPolicy: PrefillExecutor.EvalPolicy,
        storedForm: KVScheme?,
        in session: any ModelSession
    ) throws -> LMOutput.State? {
        // Qwen3.5 is a `Qwen3_5ForConditionalGeneration` (VLM) whose
        // `prepare` indexes tokens with two axes (`y[0..., ..<step]`) — 1D
        // crashes in `getRopeIndex` on `inputIds.dim(1)`. Pure LLMs take the
        // flat list, and the prefill adds the batch axis per chunk. Match the
        // processor's rank.
        let flat = MLXArray(path[range].map { Int32($0) })
        let tokens = tokenNDim >= 2 ? flat.expandedDimensions(axis: 0) : flat
        return try session.prefill(
            text: .init(tokens: tokens, mask: nil),
            cache: cache,
            checkpoints: [:],
            checkpointBaseOffset: range.lowerBound,
            prefillStepSize: prefillStepSize,
            consumeAll: true,
            initialState: state,
            evalPolicy: evalPolicy,
            storedForm: storedForm
        ).state
    }

    /// The one failed-restore policy (ADR-0069 amendment): a snapshot whose
    /// restore threw is dropped through the manager, in one MainActor hop, so
    /// no later request resolves to it and pays the same failed restore; and
    /// one skip line says what went. A `HybridCacheSnapshot.RestoreError`
    /// means the layers themselves are corrupt, so the SSD copy goes too. A
    /// keyed turn passes its `PlannedResolution`: when a body went, its
    /// checkpoint plan is made again against the settled tree in the same
    /// hop and returned (`nil` otherwise).
    @discardableResult
    static func dropUnrestorable(
        _ snapshot: HybridCacheSnapshot,
        error: any Error,
        tokens: [Int],
        partitionKey: CachePartitionKey,
        replanning planned: PrefixCacheManager.PlannedResolution? = nil,
        prefixCache: PrefixCacheManager,
        diagnostics: PrefixCacheDiagnostics.Context
    ) async -> PrefixCacheManager.CheckpointPlan? {
        let corruptLayers = error is HybridCacheSnapshot.RestoreError
        let outcome = await MainActor.run {
            prefixCache.dropUnrestorableSnapshot(
                snapshot, tokens: tokens, partitionKey: partitionKey,
                corruptLayers: corruptLayers, replanning: planned)
        }
        let (drop, replanned) = (outcome.drop, outcome.replanned)
        diagnostics.logSkip(
            stage: "restore",
            reason: "unrestorable-snapshot",
            level: .warning,
            extraFields: [
                ("offset", "\(snapshot.tokenOffset)"),
                ("checkpointType", snapshot.checkpointType.wireString),
                ("drop", drop.rawValue),
                ("error", String(describing: error)),
            ]
        )
        return replanned
    }
}
