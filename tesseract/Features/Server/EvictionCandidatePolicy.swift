//
//  EvictionCandidatePolicy.swift
//  tesseract
//
//  The **Eviction Candidate Policy** (ADR-0049): "who is the next victim"
//  as a pure decision, promoted out of both tier managers. The RAM tier's
//  four-strategy ladder and the SSD tier's terminal-loss ordering are the
//  same shape — a tested pure scorer (`EvictionPolicy`) composed by a
//  candidate selection that used to be sealed private inside its manager,
//  reachable only by replaying admissions. Here the selection has its own
//  seam; the effects (body drops, demotion, the ledger lock, manifest
//  mutation) stay in the managers.
//

import Foundation

/// Pure candidate selection for both cache tiers. Holds no state: every
/// call takes the current tier contents and the **Eviction Configuration**
/// by value and returns a decision — the RAM tier's next victim, or the
/// SSD tier's worst-victim-first ordering. Stays `@MainActor` because the
/// RAM tier's candidates are MainActor-isolated `RadixTreeNode`s; the SSD
/// ordering is `nonisolated` because the **Snapshot Ledger** decides on
/// the writer thread, under its own lock.
@MainActor
enum EvictionCandidatePolicy {

    /// The RAM tier's chosen victim: everything `evictToFitBudget` needs
    /// to demote, drop, and account for it — the owning partition and
    /// tree, the node itself, which ladder strategy named it, and the
    /// utility score when scoring (not the fallback) decided.
    struct Candidate {
        let partitionKey: CachePartitionKey
        let tree: TokenRadixTree
        let node: RadixTreeNode
        let strategy: PrefixCacheManager.EvictionEvent.Strategy
        let score: EvictionScore?
    }

    // MARK: - RAM tier

    /// Pick one snapshot to evict from the supplied (already-sorted) trees.
    ///
    /// Strategy (in order):
    /// 1. **Preferred utility**: if a `preferred` tree is supplied, score
    ///    its eligible nodes (every body-bearing node — uniform eviction,
    ///    ADR-0019) and return the lowest-utility one. This is the
    ///    writing-partition-first rule.
    /// 2. **Global utility**: if the preferred tree has no eligible
    ///    candidates (or none was supplied), score eligible nodes across
    ///    all partitions and return the lowest-utility one. Preserves
    ///    Marconi's "global utility" semantics for single-partition
    ///    configurations and for the spill-over case when the writing
    ///    partition is already drained.
    /// 3. **Preferred fallback**: if both utility paths are empty but
    ///    the preferred tree still has any unprotected snapshot, drop
    ///    the oldest one from the preferred tree. With uniform
    ///    eligibility this is a residual safety net (the eligible set
    ///    equals the snapshot set), kept so the hard budget invariant
    ///    never depends on scoring returning a victim.
    /// 4. **Global fallback**: drop the oldest snapshot from any tree,
    ///    so the hard budget invariant holds in degenerate cases like a
    ///    zero-budget drain.
    ///
    /// `protected` is the **Budget Floor** membership — never victims, on
    /// every strategy. Reads the trees live (no mutation): the drain loop
    /// calls once per drop, and each call re-derives the eligible sets
    /// from what the previous drop left behind.
    static func candidate(
        now: ContinuousClock.Instant,
        orderedPartitions: [(key: CachePartitionKey, tree: TokenRadixTree)],
        preferred: (key: CachePartitionKey, tree: TokenRadixTree)? = nil,
        protected: Set<ObjectIdentifier> = [],
        config: EvictionConfiguration
    ) -> Candidate? {
        func unprotected(_ nodes: [RadixTreeNode]) -> [RadixTreeNode] {
            protected.isEmpty
                ? nodes
                : nodes.filter { !protected.contains(ObjectIdentifier($0)) }
        }

        // 1. Preferred utility — writing-partition-first.
        if let preferred {
            let preferredCandidates = unprotected(preferred.tree.eligibleEvictionNodes())
            if let victim = EvictionPolicy.selectVictim(
                candidates: preferredCandidates, now: now, config: config
            ) {
                return Candidate(
                    partitionKey: preferred.key,
                    tree: preferred.tree,
                    node: victim.node,
                    strategy: .utility,
                    score: victim.score
                )
            }
        }

        // 2. Global utility — spill to other partitions.
        var partitionByNode: [ObjectIdentifier: (key: CachePartitionKey, tree: TokenRadixTree)] =
            [:]
        var candidates: [RadixTreeNode] = []
        for partition in orderedPartitions where partition.tree !== preferred?.tree {
            for node in unprotected(partition.tree.eligibleEvictionNodes()) {
                partitionByNode[ObjectIdentifier(node)] = partition
                candidates.append(node)
            }
        }
        if let victim = EvictionPolicy.selectVictim(
            candidates: candidates, now: now, config: config
        ),
            let partition = partitionByNode[ObjectIdentifier(victim.node)]
        {
            return Candidate(
                partitionKey: partition.key,
                tree: partition.tree,
                node: victim.node,
                strategy: .utility,
                score: victim.score
            )
        }

        // 3. Preferred fallback — oldest snapshot in the writing partition.
        if let preferred,
            let oldest = unprotected(preferred.tree.allSnapshotNodes()).min(
                by: { $0.lastAccessTime < $1.lastAccessTime }
            )
        {
            return Candidate(
                partitionKey: preferred.key,
                tree: preferred.tree,
                node: oldest,
                strategy: .fallback,
                score: nil
            )
        }

        // 4. Global fallback — oldest snapshot anywhere.
        return orderedPartitions
            .lazy
            .flatMap { partition in
                unprotected(partition.tree.allSnapshotNodes())
                    .lazy.map { (partition: partition, node: $0) }
            }
            .min(by: { $0.node.lastAccessTime < $1.node.lastAccessTime })
            .map { candidate in
                Candidate(
                    partitionKey: candidate.partition.key,
                    tree: candidate.partition.tree,
                    node: candidate.node,
                    strategy: .fallback,
                    score: nil
                )
            }
    }

    // MARK: - SSD tier

    /// Order descriptors worst-victim-first under terminal-loss
    /// utility: ascending `norm(1/age) + α · norm(re-prefill seconds /
    /// chain bytes)`. Re-prefill spans the *whole* chain — an SSD loss
    /// re-prefills `[0, tokenOffset]` from scratch — and bytes are the
    /// chain total (`totalBytes`), never per-segment values; both
    /// inputs are already persisted (schema v8 carries `tokenOffset`,
    /// `bytes`, and the inherited segments), so the cut needs no
    /// manifest schema bump. The `α = 0` fast path returns the plain
    /// LRU order — byte-identical to the pre-ADR-0011 cut. Pure
    /// derivation over its inputs (`now` included); shared by the
    /// **Snapshot Ledger**'s admission cut and the **Survival Gate**
    /// simulation, both under the ledger lock on the writer thread.
    nonisolated static func terminalLossOrder(
        _ candidates: [PersistedSnapshotDescriptor],
        config: EvictionConfiguration,
        now: TimeInterval
    ) -> [PersistedSnapshotDescriptor] {
        let lru = candidates.sorted {
            ($0.lastAccessAt, $0.snapshotID) < ($1.lastAccessAt, $1.snapshotID)
        }
        guard config.alpha != 0, lru.count > 1 else { return lru }

        let rawRecencies = lru.map {
            EvictionPolicy.recencyWeight(ageSeconds: now - $0.lastAccessAt)
        }
        let terms = EvictionPolicy.blendedTerms(
            rawRecencies: rawRecencies, alpha: config.alpha
        ) {
            lru.map { resident -> Double in
                guard resident.totalBytes > 0 else { return 0 }
                let rePrefillSeconds =
                    EvictionPolicy.parentRelativeFlops(
                        nodeOffset: resident.tokenOffset,
                        parentOffset: 0,
                        profile: config.flopProfile
                    ) / config.estimates.prefillFlopsPerSecond
                return rePrefillSeconds / Double(resident.totalBytes)
            }
        }

        return zip(lru, terms)
            .map { resident, terms in
                (resident: resident, utility: terms.utility)
            }
            .sorted {
                ($0.utility, $0.resident.lastAccessAt, $0.resident.snapshotID)
                    < ($1.utility, $1.resident.lastAccessAt, $1.resident.snapshotID)
            }
            .map(\.resident)
    }
}
