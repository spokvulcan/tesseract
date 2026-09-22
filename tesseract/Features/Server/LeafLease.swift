import Foundation
import MLX
import MLXLMCommon

/// Scalar identity for a tree-side Leaf Lease. Holding this value retains
/// neither the tree, the node, nor any cache object or array.
nonisolated struct LeafLease: Sendable {
    let id = UUID()
    let context: PrefixCacheDiagnostics.Context
    let offset: Int
    let bytes: Int

    enum ReleaseReason: String, Sendable {
        case checkIn, rewind
    }
}

/// The small cross-thread exclusion boundary between the tree's owner and
/// the SSD writer. It retains scalar lease state and weak payload probes,
/// never cache objects or payloads. The lock is never held while copying
/// arrays or performing I/O.
nonisolated final class LeafBodyAccess: @unchecked Sendable {
    private let lock = NSLock()
    private var current: LeafLease?
    private var readers = 0
    private var reportedDeferral: UUID?
    private var payloadMaterialized: [@Sendable () -> Bool] = []

    func observeFullPayload(_ payload: SnapshotPayload) {
        lock.withLock { payloadMaterialized.append(payload.materializationProbe) }
    }

    var lease: LeafLease? { lock.withLock { current } }

    /// Whether a lease here would be refused on the SSD writer's account:
    /// a full payload that still aliases the body, or the writer's own
    /// body read while it prepares and writes one. False means the next
    /// `begin(requireDetachedPayload: true)` is not refused by the writer
    /// — which is what makes the bounded pending-payload wait (#523)
    /// race-free rather than hopeful.
    ///
    /// Read only: unlike `begin` it prunes no probe and logs no refusal,
    /// so the wait can poll it without stamping a `leafLeaseRefused`
    /// event per poll.
    var blockedByWriter: Bool {
        lock.withLock { readers > 0 || payloadMaterialized.contains { !$0() } }
    }

    func begin(_ lease: LeafLease, requireDetachedPayload: Bool = false) -> Bool {
        lock.withLock {
            guard current == nil, readers == 0 else { return false }
            if requireDetachedPayload {
                payloadMaterialized.removeAll { $0() }
                guard payloadMaterialized.isEmpty else { return false }
            }
            current = lease
            return true
        }
    }

    func end(_ lease: LeafLease) -> Bool {
        lock.withLock {
            guard current?.id == lease.id else { return false }
            current = nil
            reportedDeferral = nil
            return true
        }
    }

    /// Claim before removing a full payload from the writer queue. A lease
    /// either wins this race or waits until the last borrowed byte is written.
    func beginRead(snapshotID: String) -> Bool {
        let result: (allowed: Bool, report: LeafLease?) = lock.withLock {
            if let current {
                let report = reportedDeferral != current.id ? current : nil
                reportedDeferral = current.id
                return (false, report)
            }
            readers += 1
            return (true, nil)
        }
        if let lease = result.report {
            lease.context.log(
                LeafLeaseDeferredEvent(lease: lease, snapshotID: snapshotID), level: .notice)
        }
        return result.allowed
    }

    func endRead() {
        lock.withLock {
            precondition(readers > 0)
            readers -= 1
        }
    }

    @discardableResult
    func blocks(_ operation: LeafLeaseRefusedEvent.Reason) -> Bool {
        guard let lease else { return false }
        lease.context.log(
            LeafLeaseRefusedEvent(
                reason: operation, offset: lease.offset, bytes: lease.bytes, leaseID: lease.id),
            level: .notice)
        return true
    }
}

/// Typed scalar telemetry; a refused acquisition has no new lease identity.
nonisolated struct LeafLeaseAccounting: Sendable {
    let treeSnapshotBytes: Int
    let leasedBytes: Int
    let leaseCount: Int

    var fields: [(String, String)] {
        [
            ("treeSnapshotBytes", "\(treeSnapshotBytes)"), ("leasedBytes", "\(leasedBytes)"),
            ("leaseCount", "\(leaseCount)"),
        ]
    }
}

extension LeafLease {
    nonisolated var fields: [(String, String)] {
        [("leaseID", id.uuidString), ("offset", "\(offset)"), ("bytes", "\(bytes)")]
    }
}

nonisolated struct LeafLeaseBeginEvent: PrefixCacheDiagnostics.Payload {
    let lease: LeafLease
    let accounting: LeafLeaseAccounting
    let eventName = "leafLeaseBegin"
    var fields: [(String, String)] { lease.fields + accounting.fields }
}

nonisolated struct LeafLeaseEndEvent: PrefixCacheDiagnostics.Payload {
    let lease: LeafLease
    let reason: LeafLease.ReleaseReason
    let returnedOffset: Int
    let returnedBytes: Int
    let accounting: LeafLeaseAccounting
    let eventName = "leafLeaseEnd"
    var fields: [(String, String)] {
        lease.fields + [
            ("reason", reason.rawValue), ("returnedOffset", "\(returnedOffset)"),
            ("returnedBytes", "\(returnedBytes)"),
            ("growthBytes", "\(returnedBytes - lease.bytes)"),
        ] + accounting.fields
    }
}

nonisolated struct LeafLeaseRefusedEvent: PrefixCacheDiagnostics.Payload {
    /// Why the tree refused. An `Error`, so a lease attempt can answer
    /// `Result<LeafLease, Reason>`.
    enum Reason: String, Sendable, Error {
        case bodyReplacement, ssdAdmission, supersession, ramClear, demotion, writePromotion,
            dropBody
        case wrongTree, notLeaf, alreadyLeased, writerReading, pendingFullPayload, staleLease
        case invalidBody, invalidPath, invalidRewind, occupiedDestination, destinationBusy
    }

    let reason: Reason
    let offset: Int
    let bytes: Int
    var leaseID: UUID?
    var activeLeaseID: UUID?
    var activeRequestID: UUID?
    let eventName = "leafLeaseRefused"
    var fields: [(String, String)] {
        var fields = [("reason", reason.rawValue), ("offset", "\(offset)"), ("bytes", "\(bytes)")]
        if let leaseID { fields.append(("leaseID", leaseID.uuidString)) }
        if let activeLeaseID { fields.append(("activeLeaseID", activeLeaseID.uuidString)) }
        if let activeRequestID { fields.append(("activeRequestID", activeRequestID.uuidString)) }
        return fields
    }
}

nonisolated struct LeafLeaseDeferredEvent: PrefixCacheDiagnostics.Payload {
    let lease: LeafLease
    let snapshotID: String
    let eventName = "leafLeaseDeferred"
    var fields: [(String, String)] {
        lease.fields + [("reason", "writerMaterialization"), ("snapshotID", snapshotID)]
    }
}

nonisolated struct LeafRewindEvent: PrefixCacheDiagnostics.Payload {
    let lease: LeafLease
    let recurrentBytes: Int
    /// The rewound cache's full-attention array extents versus the rows the
    /// lease offset addresses: their difference is the growth capacity the
    /// aborted generation left in the leaf (#501 / #534).
    var fullAttentionArrayBytes = 0
    var fullAttentionLogicalBytes = 0
    /// Bytes the #534 compaction freed before the move (`0` below threshold).
    var compactedBytes = 0
    let eventName = "leafRewind"
    var fields: [(String, String)] {
        lease.fields + [
            ("recurrentRewindStateBytes", "\(recurrentBytes)"),
            ("fullAttentionArrayBytes", "\(fullAttentionArrayBytes)"),
            ("fullAttentionLogicalBytes", "\(fullAttentionLogicalBytes)"),
            (
                "fullAttentionUnusedArrayBytes",
                "\(fullAttentionArrayBytes - fullAttentionLogicalBytes)"
            ),
            ("compactedBytes", "\(compactedBytes)"),
        ]
    }
}

/// What exact **Leaf Rewind** needs, saved when a leaf is checked out: each
/// cache object's kind, and an independent copy of every recurrent layer.
/// Sliceable attention preserves its prefix even across growth, so the
/// rewind trims it back to `offset`; whole-state layers are rebuilt from
/// their copy, including empty slots. The copy is the only array a
/// check-out allocates.
nonisolated struct LeafRewind: @unchecked Sendable {
    /// A whole-state layer's independent copy. Every whole-state layer of
    /// an eligible leaf is recurrent (`HybridCacheSnapshot.checkoutRefusal`),
    /// so the rewind rebuilds it as an `ArraysCache`.
    private struct RecurrentState {
        let index: Int
        let layer: HybridCacheSnapshot.LayerState
    }

    /// The leased leaf's token offset: where the rewind trims attention to.
    let offset: Int
    /// Each cache object's kind, in cache order, as the moved body recorded
    /// it: what the rewind trims and what it rebuilds.
    private let kinds: [HybridCacheSnapshot.LayerState.Kind]
    private let recurrent: [RecurrentState]

    var stateBytes: Int {
        recurrent.reduce(0) { $0 + $1.layer.state.reduce(0) { $0 + $1.nbytes } }
    }

    /// Model Session only, on the objects just taken from the leaf.
    init(cache: [any KVCache], kinds: [HybridCacheSnapshot.LayerState.Kind], offset: Int) {
        self.offset = offset
        self.kinds = kinds
        recurrent = zip(cache, kinds).enumerated().compactMap { index, entry in
            let (layer, kind) = entry
            guard kind == .wholeState, let className = HybridCacheSnapshot.classNameForCache(layer)
            else { return nil }
            precondition(
                layer is ArraysCache,
                "an eligible leaf's whole-state layers are recurrent (checkoutRefusal)")
            return RecurrentState(
                index: index,
                layer: .init(
                    className: className,
                    state: layer.state.map { HybridCacheSnapshot.deepCopyState($0) },
                    metaState: layer.metaState, offset: layer.offset))
        }
        eval(recurrent.flatMap { $0.layer.state })
    }

    /// Model Session only, once generation has quiesced.
    func rewind(_ cache: inout [any KVCache]) {
        eval(cache)
        for (layer, kind) in zip(cache, kinds) where kind == .sliceableAttention {
            let advance = layer.offset - offset
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
}
