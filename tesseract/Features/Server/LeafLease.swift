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
                LeafLeaseEvent(
                    name: "leafLeaseDeferred", lease: lease,
                    facts: [
                        "reason": "writerMaterialization", "snapshotID": snapshotID,
                    ]), level: .notice)
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
    func refuse(_ operation: String) -> Bool {
        guard let lease else { return false }
        lease.context.log(
            LeafLeaseEvent(name: "leafLeaseRefused", lease: lease, facts: ["reason": operation]),
            level: .notice)
        return true
    }
}

nonisolated struct LeafLeaseEvent: PrefixCacheDiagnostics.Payload {
    let name: String
    let lease: LeafLease
    var facts: [String: String] = [:]

    var eventName: String { name }
    var fields: [(String, String)] {
        var values = facts
        values.merge([
            "leaseID": lease.id.uuidString, "offset": "\(lease.offset)", "bytes": "\(lease.bytes)",
        ]) { _, value in value }
        return values.sorted { $0.key < $1.key }.map { ($0.key, $0.value) }
    }
}
