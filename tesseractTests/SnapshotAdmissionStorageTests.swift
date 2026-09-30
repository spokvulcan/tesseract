//
//  SnapshotAdmissionStorageTests.swift
//  tesseractTests
//
//  Storage intent at the extraction edge (ADR-0078): each captured snapshot
//  is paired with RAM-only, view, or RAM-and-SSD storage before its Snapshot
//  Admission crosses to the MainActor, and the manager admits either shape.
//  The payload a RAM-and-SSD entry carries is pinned by `SnapshotPayloadTests`.
//

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

struct SnapshotAdmissionStorageTests {

    @Test func prefixViewAdmissionRetainsSSDIntentUntilTheLeafChecksIn() throws {
        let attention = KVCacheSimple()
        attention.state = [MLXArray.ones([1, 1, 4, 64]), MLXArray.ones([1, 1, 4, 64])]
        let view = try #require(
            HybridCacheSnapshot.capture(
                cache: [attention], offset: 4, type: .branchPoint, prefixView: true))
        let candidates = SnapshotAdmission.checkpointCandidates(
            [view], ssdEnabled: true)
        let candidate = try #require(candidates.first)
        guard case .viewSSD = candidate.storage else {
            Issue.record("a view must retain SSD intent without extracting incomplete arrays")
            return
        }
        #expect(candidate.snapshot.memoryBytes == 0)
    }

    // MARK: - Fixture builders

    private func checkpointCandidates(
        for snapshots: [HybridCacheSnapshot],
        ssdEnabled: Bool = true
    ) -> [SnapshotAdmission.CheckpointCandidate] {
        SnapshotAdmission.checkpointCandidates(
            snapshots,
            ssdEnabled: ssdEnabled
        )
    }

    private func checkpointPayload(
        for snapshot: HybridCacheSnapshot
    ) throws -> SnapshotPayload {
        let payloads = checkpointPayloads(for: [snapshot])
        try #require(payloads.count == 1)
        return payloads[0]
    }

    private func checkpointPayloads(
        for snapshots: [HybridCacheSnapshot],
        ssdEnabled: Bool = true
    ) -> [SnapshotPayload] {
        checkpointCandidates(
            for: snapshots,
            ssdEnabled: ssdEnabled
        ).compactMap { candidate in
            if case .ramAndSSD(let payload) = candidate.storage {
                return payload
            }
            return nil
        }
    }

    private func expectRAMOnly(
        _ candidates: [SnapshotAdmission.CheckpointCandidate]
    ) {
        for candidate in candidates {
            if case .ramOnly = candidate.storage {
                // expected
            } else {
                #expect(Bool(false), "Expected candidate to be RAM-only")
            }
        }
    }

    // MARK: - SSD gate

    @Test
    func candidatesAreRAMOnlyWhenSSDDisabled() {
        let snapshot = PrefixCacheTestFixtures.makeSimpleKVSnapshot()
        let candidates = checkpointCandidates(for: [snapshot], ssdEnabled: false)

        #expect(candidates.count == 1)
        #expect(candidates.first?.snapshot.tokenOffset == snapshot.tokenOffset)
        expectRAMOnly(candidates)
    }

    @Test
    func candidatesAreRAMOnlyWhenSSDDisabledForMultipleSnapshots() {
        // The gate must preserve every snapshot while marking each
        // candidate RAM-only. SSD-disabled is no longer represented as
        // an empty payload array.
        let snapshots = [
            PrefixCacheTestFixtures.makeSimpleKVSnapshot(tokenOffset: 5),
            PrefixCacheTestFixtures.makeSimpleKVSnapshot(tokenOffset: 9),
            PrefixCacheTestFixtures.makeMixedSnapshot(tokenOffset: 40),
        ]
        let candidates = checkpointCandidates(for: snapshots, ssdEnabled: false)

        #expect(candidates.map(\.snapshot.tokenOffset) == snapshots.map(\.tokenOffset))
        expectRAMOnly(candidates)
    }

    @Test
    func returnsNoCandidatesForEmptyInputEvenWhenEnabled() {
        let candidates = checkpointCandidates(for: [], ssdEnabled: true)

        #expect(candidates.isEmpty)
    }

    @Test
    func checkpointCandidatesCarryPerEntryStorageAtExtractionEdge() throws {
        let snapshots = [
            PrefixCacheTestFixtures.makeSimpleKVSnapshot(tokenOffset: 5),
            PrefixCacheTestFixtures.makeSimpleKVSnapshot(tokenOffset: 9, type: .branchPoint),
        ]

        let ssdCandidates = checkpointCandidates(for: snapshots)

        #expect(ssdCandidates.count == snapshots.count)
        for index in snapshots.indices {
            #expect(ssdCandidates[index].snapshot.tokenOffset == snapshots[index].tokenOffset)
            if case .ramAndSSD(let payload) = ssdCandidates[index].storage {
                #expect(payload.tokenOffset == snapshots[index].tokenOffset)
                #expect(payload.checkpointType == snapshots[index].checkpointType)
            } else {
                #expect(Bool(false), "Expected SSD-enabled candidate to carry its payload")
            }
        }

        let ramOnlyCandidates = checkpointCandidates(for: snapshots, ssdEnabled: false)

        #expect(ramOnlyCandidates.count == snapshots.count)
        expectRAMOnly(ramOnlyCandidates)
    }

    // MARK: - Structural equivalence

    @Test
    func payloadIsPositionallyAlignedWithInput() throws {
        let snapshots = [
            PrefixCacheTestFixtures.makeSimpleKVSnapshot(tokenOffset: 5),
            PrefixCacheTestFixtures.makeSimpleKVSnapshot(tokenOffset: 9),
            PrefixCacheTestFixtures.makeSimpleKVSnapshot(tokenOffset: 13),
        ]
        let payloads = checkpointPayloads(for: snapshots)
        try #require(payloads.count == snapshots.count)
        for (i, payload) in payloads.enumerated() {
            #expect(payload.tokenOffset == snapshots[i].tokenOffset)
            #expect(payload.checkpointType == snapshots[i].checkpointType)
        }
    }

    // MARK: - PrefixCacheManager admission plumbing

    @MainActor
    @Test
    func admitAcceptsSnapshotAdmissionPayloads() throws {
        // Snapshot Admission must carry extracted payloads into the
        // cache manager without breaking RAM insertion or diagnostics.
        let manager = PrefixCacheManager(
            memoryBudgetBytes: 16 * 1024 * 1024
        )
        let snapshot = PrefixCacheTestFixtures.makeSimpleKVSnapshot(tokenOffset: 4)
        let payload = try checkpointPayload(for: snapshot)
        let partitionKey = CachePartitionKey(
            modelID: "test-model", kvBits: nil, kvGroupSize: 64
        )
        let admission = try #require(
            SnapshotAdmission.checkpoints(
                fullPromptTokens: [1, 2, 3, 4],
                candidates: [
                    SnapshotAdmission.CheckpointCandidate(
                        snapshot: snapshot,
                        storage: .ramAndSSD(payload)
                    )
                ],
                partitionKey: partitionKey,
                requestID: UUID()
            ))
        let diagnostics = manager.admit(admission)

        #expect(diagnostics.evictions.isEmpty)
        #expect(manager.stats.snapshotCount == 1)
    }

    @MainActor
    @Test
    func snapshotAdmissionSupportsRAMOnlyCheckpointEntries() throws {
        // When `ssdConfig?.enabled` is off, the Server Completion helper returns
        // `[]` for payload extraction. The extraction edge represents
        // that as a RAM-only admission entry, not as "no snapshots".
        let manager = PrefixCacheManager(
            memoryBudgetBytes: 16 * 1024 * 1024
        )
        let snapshot = PrefixCacheTestFixtures.makeSimpleKVSnapshot(tokenOffset: 5)
        let partitionKey = CachePartitionKey(
            modelID: "test-model", kvBits: nil, kvGroupSize: 64
        )
        let admission = try #require(
            SnapshotAdmission.checkpoints(
                fullPromptTokens: [7, 8, 9, 10, 11],
                candidates: [
                    SnapshotAdmission.CheckpointCandidate(
                        snapshot: snapshot,
                        storage: .ramOnly
                    )
                ],
                partitionKey: partitionKey,
                requestID: UUID()
            ))
        let diagnostics = manager.admit(admission)

        #expect(diagnostics.evictions.isEmpty)
        #expect(manager.stats.snapshotCount == 1)
    }
}
