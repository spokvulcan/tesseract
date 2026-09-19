import Foundation
import MLX
import MLXLMCommon

/// A request's ownership of a resident leaf. GPU state is touched only in
/// the Model Session; its claim returns the body on the tree's MainActor.
nonisolated final class LeafCheckout: @unchecked Sendable {
    struct Claim: Sendable {
        let tree: TokenRadixTree
        let node: RadixTreeNode
        let lease: LeafLease

        @MainActor
        func returnBody(
            _ body: HybridCacheSnapshot, tokens: [Int], reason: LeafLease.ReleaseReason
        ) -> Bool {
            tree.endLeafLease(lease, on: node, returning: body, tokens: tokens, reason: reason)
        }
    }

    enum ClaimResult: Sendable {
        case claimed(Claim)
        case copy(LeafStorePhase.Report.CopyReason)
    }

    struct Attempt: Sendable {
        var owner: FinalGenerationCache?
        var copyReason: LeafStorePhase.Report.CopyReason?
        /// Seconds this attempt spent waiting for a pending full payload
        /// to stop aliasing the leaf's body (#523). `0` on every attempt
        /// that never waited — including the ones refused for a *queued*
        /// payload, which is deliberately not waited for. The copy reason
        /// keeps its name, `pendingFullPayload`; this is what it cost.
        var pendingPayloadWaitSeconds: TimeInterval = 0
    }

    /// A whole-state layer's independent copy, saved at check-out. Every
    /// whole-state layer of an eligible leaf is recurrent (see
    /// `HybridCacheSnapshot.checkoutCopyReason`), so the rewind rebuilds
    /// it as an `ArraysCache`.
    private struct RecurrentState {
        let index: Int
        let layer: HybridCacheSnapshot.LayerState
    }

    let claim: Claim
    let originalTokens: [Int]
    /// Each cache object's kind, in cache order, as the moved body
    /// recorded it: what **Leaf Rewind** trims and what it rebuilds.
    private let kinds: [HybridCacheSnapshot.LayerState.Kind]
    private let recurrent: [RecurrentState]
    var rewindStateBytes: Int {
        recurrent.reduce(0) { $0 + $1.layer.state.reduce(0) { $0 + $1.nbytes } }
    }

    init(
        claim: Claim, tokens: [Int], cache: [any KVCache],
        kinds: [HybridCacheSnapshot.LayerState.Kind]
    ) {
        self.claim = claim
        self.kinds = kinds
        originalTokens = Array(tokens.prefix(claim.lease.offset))
        recurrent = zip(cache, kinds).enumerated().compactMap { index, entry in
            let (layer, kind) = entry
            guard kind == .wholeState, let className = HybridCacheSnapshot.classNameForCache(layer)
            else { return nil }
            precondition(
                layer is ArraysCache,
                "an eligible leaf's whole-state layers are recurrent (checkoutCopyReason)")
            return RecurrentState(
                index: index,
                layer: .init(
                    className: className,
                    state: layer.state.map { HybridCacheSnapshot.deepCopyState($0) },
                    metaState: layer.metaState, offset: layer.offset))
        }
        eval(recurrent.flatMap { $0.layer.state })
    }

    /// Sliceable attention preserves its prefix even across growth, so it
    /// is trimmed back to the leaf offset. Whole-state layers are rebuilt
    /// from the independently owned state, including empty slots.
    func rewind(cache: inout [any KVCache]) {
        eval(cache)
        for (layer, kind) in zip(cache, kinds) where kind == .sliceableAttention {
            let advance = layer.offset - claim.lease.offset
            precondition(advance >= 0)
            let trimmed = layer.trim(advance)
            precondition(trimmed == advance)
        }
        for saved in recurrent {
            var arrays: [MLXArray] = []
            cache[saved.index] = HybridCacheSnapshot.makeArraysCache(
                mamba: saved.layer.className == "MambaCache", state: saved.layer.state,
                metaState: saved.layer.metaState, offset: saved.layer.offset,
                copyStrategy: nil, copiedArrays: &arrays)
        }
        eval(cache)
    }

    static func maximumAdvance(
        newPromptTokens: Int, outputCeiling: Int?, speculativeAllowance: Int
    ) -> Int {
        guard let outputCeiling else { return Int.max }
        let (withPrompt, overflow) = max(0, outputCeiling).addingReportingOverflow(
            max(0, newPromptTokens))
        let (total, draftOverflow) = withPrompt.addingReportingOverflow(
            max(0, speculativeAllowance))
        return overflow || draftOverflow ? Int.max : total
    }

    static func attempt(
        resolved: PrefixCacheManager.Resolved, tokens: [Int], maximumAdvance: Int,
        identityKeySpace: Bool, prefixCache: PrefixCacheManager,
        context: PrefixCacheDiagnostics.Context
    ) async -> Attempt {
        guard let snapshot = resolved.lookup.snapshot,
            let key = resolved.lookup.partitionKey
        else { return Attempt() }
        guard identityKeySpace else { return Attempt(copyReason: .imageKeySpace) }
        guard key.kvBits == nil else { return Attempt(copyReason: .quantized) }
        guard !resolved.wasChainPrefixRestore,
            resolved.lookup.snapshotTokenOffset == snapshot.tokenOffset,
            tokens.count > snapshot.tokenOffset
        else { return Attempt(copyReason: .checkpoint) }
        let bodyCopyReason = snapshot.checkoutCopyReason(maximumAdvance: maximumAdvance)
        let claim: Claim
        var waitedSeconds: TimeInterval = 0
        var result = await prefixCache.claimLeaf(
            snapshot: snapshot, tokens: tokens, partitionKey: key,
            bodyCopyReason: bodyCopyReason, context: context)
        // The one refusal that clears itself: a full payload aliases the
        // body only until the SSD writer materializes it (ADR-0064
        // decision 5, ADR-0019's Deferred Payload Extraction amendment).
        if case .copy(.pendingFullPayload) = result {
            (result, waitedSeconds) = await awaitPendingFullPayload(
                snapshot: snapshot, tokens: tokens, partitionKey: key,
                bodyCopyReason: bodyCopyReason, prefixCache: prefixCache, context: context)
        }
        switch result {
        case .claimed(let acquired): claim = acquired
        case .copy(let reason):
            return Attempt(copyReason: reason, pendingPayloadWaitSeconds: waitedSeconds)
        }
        guard let (cache, kinds) = snapshot.takeMovingCache() else {
            preconditionFailure("an eligible claimed leaf must own cache objects")
        }
        let owner = FinalGenerationCache(cache)
        owner.restoreMode = "handoff"
        owner.checkout = LeafCheckout(claim: claim, tokens: tokens, cache: cache, kinds: kinds)
        return Attempt(owner: owner, pendingPayloadWaitSeconds: waitedSeconds)
    }

    /// How often the wait re-reads the writer's answer. Small enough that
    /// a materialization is turned into a handoff promptly, large enough
    /// that the longest bound costs a bounded number of MainActor hops.
    static let pendingFullPayloadPoll: Duration = .milliseconds(5)

    /// The bounded wait of #523. **Leaf Checkout** was refused only
    /// because the leaf's full payload is still pending; while the SSD
    /// writer reports that payload `.inProgress`, the request waits up to
    /// the **Eviction Configuration** bound and then re-attempts the
    /// check-out once. A payload still `.queued` behind other writes has
    /// no bounded completion time and is not waited for — it copies
    /// immediately, as it does today.
    ///
    /// The wait is an `await`, never a blocking sleep: no Metal work runs
    /// and no thread is held, and it happens before any Model Session verb
    /// touches the cache. Cancellation settles it as a copy at once.
    private static func awaitPendingFullPayload(
        snapshot: HybridCacheSnapshot, tokens: [Int], partitionKey: CachePartitionKey,
        bodyCopyReason: LeafStorePhase.Report.CopyReason?,
        prefixCache: PrefixCacheManager, context: PrefixCacheDiagnostics.Context
    ) async -> (result: ClaimResult, waitedSeconds: TimeInterval) {
        func progress() async -> PendingPayloadProgress {
            await prefixCache.pendingFullPayloadProgress(
                snapshot: snapshot, tokens: tokens, partitionKey: partitionKey)
        }
        func reattempt() async -> ClaimResult {
            await prefixCache.claimLeaf(
                snapshot: snapshot, tokens: tokens, partitionKey: partitionKey,
                bodyCopyReason: bodyCopyReason, context: context)
        }
        let bound = await prefixCache.pendingFullPayloadWait
        guard bound > .zero else { return (.copy(.pendingFullPayload), 0) }
        switch await progress() {
        case .absent:
            // The writer let go between the refusal and this read, so the
            // refusal is already stale. Nothing to wait for — but nothing
            // to copy for either, so re-attempt and take the leaf.
            return (await reattempt(), 0)
        case .queued:
            // Queued behind other writes, with no bounded completion time
            // of its own. This request copies now, exactly as before #523.
            return (.copy(.pendingFullPayload), 0)
        case .inProgress:
            break
        }
        let started = ContinuousClock.now
        var elapsed = Duration.zero
        while elapsed < bound {
            do {
                try await Task.sleep(for: min(pendingFullPayloadPoll, bound - elapsed))
            } catch {
                break  // cancelled: settle as a copy rather than hold the request
            }
            elapsed = ContinuousClock.now - started
            if await progress() != .inProgress { break }
        }
        let waitedSeconds = elapsed.seconds
        guard !Task.isCancelled else { return (.copy(.pendingFullPayload), waitedSeconds) }
        return (await reattempt(), waitedSeconds)
    }
}
