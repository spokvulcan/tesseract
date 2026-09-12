import Foundation

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
/// the SSD writer. No cache references live here. The lock protects scalars
/// only and is never held while copying arrays or performing I/O.
nonisolated final class LeafBodyAccess: @unchecked Sendable {
    private let lock = NSLock()
    private var current: LeafLease?
    private var readers = 0
    private var reportedDeferral: UUID?

    var lease: LeafLease? { lock.withLock { current } }

    func begin(_ lease: LeafLease) -> Bool {
        lock.withLock {
            guard current == nil, readers == 0 else { return false }
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
    /// either wins this race or refuses acquisition until the reader releases.
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
    enum Reason: String, Sendable {
        case bodyReplacement, ssdAdmission, supersession, ramClear, demotion, writePromotion,
            dropBody
        case wrongTree, notLeaf, alreadyLeased, writerReading, staleLease
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
