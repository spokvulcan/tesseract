import Foundation
import MLXLMCommon

nonisolated struct NonEmpty<Element: Sendable>: Sendable {
    let first: Element
    let rest: [Element]

    nonisolated init(first: Element, rest: [Element] = []) {
        self.first = first
        self.rest = rest
    }

    nonisolated var count: Int { 1 + rest.count }

    nonisolated subscript(index: Int) -> Element {
        precondition(index >= 0 && index < count, "NonEmpty index out of bounds")
        return index == 0 ? first : rest[index - 1]
    }
}

/// A validated prefix path into a Snapshot Admission's shared full prompt token
/// sequence. Stores only the offset so checkpoint entries do not duplicate
/// token arrays.
nonisolated struct SnapshotAdmissionPath: Equatable, Sendable {
    let offset: Int

    private init(offset: Int) {
        self.offset = offset
    }

    nonisolated static func validating(
        offset: Int,
        fullPromptTokenCount: Int
    ) -> SnapshotAdmissionPath? {
        guard offset > 0, offset <= fullPromptTokenCount else { return nil }
        return SnapshotAdmissionPath(offset: offset)
    }

    nonisolated static func validatingLeaf(
        offset: Int,
        storedTokenCount: Int
    ) -> SnapshotAdmissionPath? {
        guard offset == storedTokenCount else { return nil }
        return SnapshotAdmissionPath(offset: offset)
    }
}

/// Already-valid write-side admission into Prefix Cache. Carries the shared
/// prompt token sequence once plus per-entry validated paths and storage intent.
nonisolated struct SnapshotAdmission: Sendable {
    enum Kind: Equatable, Sendable {
        case checkpoints
        case leaf
    }

    enum Storage: Sendable {
        case ramOnly
        case ramAndSSD(SnapshotPayload)
    }

    struct CheckpointCandidate: Sendable {
        let snapshot: HybridCacheSnapshot
        let storage: Storage

        nonisolated init(snapshot: HybridCacheSnapshot, storage: Storage) {
            self.snapshot = snapshot
            self.storage = storage
        }
    }

    struct Entry: Sendable {
        let path: SnapshotAdmissionPath
        let snapshot: HybridCacheSnapshot
        let storage: Storage
    }

    let fullPromptTokens: [Int]
    let entries: NonEmpty<Entry>
    let kind: Kind
    let partitionKey: CachePartitionKey
    let requestID: UUID?

    nonisolated var snapshots: [HybridCacheSnapshot] {
        var result: [HybridCacheSnapshot] = []
        result.reserveCapacity(entries.count)
        for index in 0..<entries.count {
            result.append(entries[index].snapshot)
        }
        return result
    }

    private init(
        fullPromptTokens: [Int],
        entries: NonEmpty<Entry>,
        kind: Kind,
        partitionKey: CachePartitionKey,
        requestID: UUID?
    ) {
        self.fullPromptTokens = fullPromptTokens
        self.entries = entries
        self.kind = kind
        self.partitionKey = partitionKey
        self.requestID = requestID
    }

    nonisolated static func checkpoints(
        fullPromptTokens: [Int],
        candidates: [CheckpointCandidate],
        partitionKey: CachePartitionKey,
        requestID: UUID?
    ) -> SnapshotAdmission? {
        var entries: [Entry] = []
        entries.reserveCapacity(candidates.count)

        for candidate in candidates {
            guard let path = SnapshotAdmissionPath.validating(
                offset: candidate.snapshot.tokenOffset,
                fullPromptTokenCount: fullPromptTokens.count
            ) else { continue }
            entries.append(Entry(
                path: path,
                snapshot: candidate.snapshot,
                storage: candidate.storage
            ))
        }

        guard let first = entries.first else { return nil }
        return SnapshotAdmission(
            fullPromptTokens: fullPromptTokens,
            entries: NonEmpty(first: first, rest: Array(entries.dropFirst())),
            kind: .checkpoints,
            partitionKey: partitionKey,
            requestID: requestID
        )
    }

    nonisolated static func leaf(
        storedTokens: [Int],
        snapshot: HybridCacheSnapshot,
        storage: Storage,
        partitionKey: CachePartitionKey,
        requestID: UUID?
    ) -> SnapshotAdmission? {
        guard let path = SnapshotAdmissionPath.validatingLeaf(
            offset: snapshot.tokenOffset,
            storedTokenCount: storedTokens.count
        ) else { return nil }

        let entry = Entry(
            path: path,
            snapshot: snapshot,
            storage: storage
        )
        return SnapshotAdmission(
            fullPromptTokens: storedTokens,
            entries: NonEmpty(first: entry),
            kind: .leaf,
            partitionKey: partitionKey,
            requestID: requestID
        )
    }
}
