//
//  SnapshotStore.swift
//  tesseract
//
//  Abstraction over the per-partition `TokenRadixTree` collection that
//  `PrefixCacheManager` operates on, so a RAM+SSD tiered implementation
//  can slot in alongside the pure-RAM one without touching the manager.
//
//  MainActor-isolated because every RAM-resident node op in the prefix
//  cache already runs on MainActor; a nonisolated protocol would force
//  a pointless hop in the conformance.
//

import Foundation
import MLXLMCommon

// MARK: - Protocol

/// Per-partition snapshot storage surface consumed by
/// `PrefixCacheManager`. Conforming types own the set of live
/// `TokenRadixTree` instances — the manager does not construct or
/// discard trees directly.
@MainActor
protocol SnapshotStore: AnyObject {
    /// Lookup-only; returns `nil` for partitions that have no tree.
    func tree(for key: CachePartitionKey) -> TokenRadixTree?

    /// Lookup-or-allocate. Used on the write path before the first
    /// insertion into a freshly seen partition.
    func getOrCreateTree(for key: CachePartitionKey) -> TokenRadixTree

    /// Deterministic key-sorted iteration. The eviction drain pins a
    /// single ordered list across all rounds of `findEvictionCandidate`
    /// so tie-breaks stay stable; callers rely on the ordering being
    /// consistent across calls within one drain.
    func orderedPartitions() -> [(key: CachePartitionKey, tree: TokenRadixTree)]

    /// Live partition count.
    var partitionCount: Int { get }

    /// Sum of every partition's RAM-resident snapshot bytes. Invoked
    /// inside the eviction drain loop condition — must stay
    /// O(partitions) or cheaper.
    var totalSnapshotBytes: Int { get }
}

// MARK: - InMemorySnapshotTier

/// Pure-RAM `SnapshotStore`, backed by a plain dictionary.
///
/// Kept deliberately dumb: partitions spring into existence on first
/// write and live for the process lifetime. Explicit discard is a
/// tiered-store concern, not a RAM-tier one.
@MainActor
final class InMemorySnapshotTier: SnapshotStore {
    private var trees: [CachePartitionKey: TokenRadixTree] = [:]

    init() {}

    func tree(for key: CachePartitionKey) -> TokenRadixTree? {
        trees[key]
    }

    func getOrCreateTree(for key: CachePartitionKey) -> TokenRadixTree {
        if let existing = trees[key] { return existing }
        let fresh = TokenRadixTree()
        trees[key] = fresh
        return fresh
    }

    func orderedPartitions() -> [(key: CachePartitionKey, tree: TokenRadixTree)] {
        trees
            .sorted { $0.key < $1.key }
            .map { (key: $0.key, tree: $0.value) }
    }

    var partitionCount: Int { trees.count }

    var totalSnapshotBytes: Int {
        trees.values.reduce(0) { $0 + $1.totalSnapshotBytes }
    }
}
