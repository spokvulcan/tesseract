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
        precondition(kinds.count == cache.count, "one kind per moved cache object")
        self.claim = claim
        self.kinds = kinds
        originalTokens = Array(tokens.prefix(claim.lease.offset))
        recurrent = zip(cache, kinds).enumerated().compactMap { index, entry in
            let (layer, kind) = entry
            guard kind == .wholeState, let className = HybridCacheSnapshot.classNameForCache(layer)
            else { return nil }
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
        switch await prefixCache.claimLeaf(
            snapshot: snapshot, tokens: tokens, partitionKey: key,
            bodyCopyReason: bodyCopyReason, context: context)
        {
        case .claimed(let acquired): claim = acquired
        case .copy(let reason): return Attempt(copyReason: reason)
        }
        guard let kinds = snapshot.movingLayerKinds, let cache = snapshot.takeMovingCache() else {
            preconditionFailure("an eligible claimed leaf must own cache objects")
        }
        let owner = FinalGenerationCache(cache)
        owner.restoreMode = "handoff"
        owner.checkout = LeafCheckout(claim: claim, tokens: tokens, cache: cache, kinds: kinds)
        return Attempt(owner: owner)
    }
}
