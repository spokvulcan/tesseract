import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// Slice #83 (PRD #82): the manager's lifetime cumulative counters —
/// hit tokens, recovered-vs-terminal eviction outcomes, hydrations —
/// and their surfacing on the telemetry snapshot. RAM-only managers,
/// no shared state.
@MainActor
struct PromptCacheCountersTests {

    private let key = CachePartitionKey(modelID: "counters-test", kvBits: nil, kvGroupSize: 64)

    @discardableResult
    private func admitLeaf(
        _ manager: PrefixCacheManager,
        tokens: [Int]
    ) -> PrefixCacheManager.StoreDiagnostics {
        PrefixCacheTestFixtures.admitUniformLeaf(manager, tokens: tokens, partitionKey: key)
    }

    @Test func ramHitsAccumulateHitTokens() {
        let manager = PrefixCacheManager(memoryBudgetBytes: 100 * 1024 * 1024)
        let tokens = Array(1...40)
        admitLeaf(manager, tokens: tokens)

        #expect(manager.cumulativeCounters.hitTokens == 0)
        _ = manager.lookup(tokens: tokens + [99], partitionKey: key)
        _ = manager.lookup(tokens: tokens + [99], partitionKey: key)
        #expect(manager.cumulativeCounters.hitTokens == 80)
        // Misses contribute nothing.
        _ = manager.lookup(tokens: [7, 8, 9], partitionKey: key)
        #expect(manager.cumulativeCounters.hitTokens == 80)
    }

    @Test func terminalAndRecoveredEvictionsAreClassifiedByRefSurvival() {
        let snapBytes = PrefixCacheTestFixtures.makeUniformSnapshot(
            offset: 10, type: .leaf
        ).memoryBytes
        let tieredStore = TieredSnapshotStore(ssdConfig: nil)
        let manager = PrefixCacheManager(
            memoryBudgetBytes: snapBytes,
            tieredStore: tieredStore
        )

        // First leaf gets a committed ref: its eviction is recovered
        // (the body drops, the node stays SSD-hittable).
        let firstTokens = Array(1...10)
        admitLeaf(manager, tokens: firstTokens)
        let tree = tieredStore.tree(for: key)!
        let (firstNode, _) = tree.findBestSnapshot(
            tokens: firstTokens, updateAccess: false
        )!
        let ref = PrefixCacheTestFixtures.makeRef(tokenOffset: firstTokens.count)
        tree.admit(node: firstNode, ref: ref)
        tree.commitRef(node: firstNode, expectedID: ref.snapshotID)

        let secondTokens = Array(20...29)
        admitLeaf(manager, tokens: secondTokens)
        #expect(manager.cumulativeCounters.recoveredEvictions == 1)
        #expect(manager.cumulativeCounters.terminalEvictions == 0)

        // Second leaf is ramOnly: its eviction is a terminal loss.
        let thirdTokens = Array(40...49)
        admitLeaf(manager, tokens: thirdTokens)
        #expect(manager.cumulativeCounters.recoveredEvictions == 1)
        #expect(manager.cumulativeCounters.terminalEvictions == 1)
    }

    @Test func promoteCountsHydrationAndHitTokens() {
        let tieredStore = TieredSnapshotStore(ssdConfig: nil)
        let manager = PrefixCacheManager(
            memoryBudgetBytes: 100 * 1024 * 1024,
            tieredStore: tieredStore
        )

        // Drive a node to state 5 (committed ref, no body), then promote
        // a freshly "hydrated" body onto it.
        let tokens = Array(1...12)
        admitLeaf(manager, tokens: tokens)
        let tree = tieredStore.tree(for: key)!
        let (node, _) = tree.findBestSnapshot(tokens: tokens, updateAccess: false)!
        let ref = PrefixCacheTestFixtures.makeRef(tokenOffset: tokens.count)
        tree.admit(node: node, ref: ref)
        tree.commitRef(node: node, expectedID: ref.snapshotID)
        tree.dropBody(node: node)
        manager.cumulativeCountersResetForTesting()

        let body = PrefixCacheTestFixtures.makeUniformSnapshot(
            offset: tokens.count, type: .leaf
        )
        manager.promote(node: node, snapshot: body, partitionKey: key)
        #expect(manager.cumulativeCounters.hydrations == 1)
        #expect(manager.cumulativeCounters.hitTokens == tokens.count)

        // A promote that misses its window (node no longer ssdOnly) is
        // ignored and must not count.
        manager.promote(node: node, snapshot: body, partitionKey: key)
        #expect(manager.cumulativeCounters.hydrations == 1)
    }

    @Test func telemetrySnapshotSurfacesCountersAndEstimates() {
        let manager = PrefixCacheManager(memoryBudgetBytes: 100 * 1024 * 1024)
        let tokens = Array(1...25)
        admitLeaf(manager, tokens: tokens)
        _ = manager.lookup(tokens: tokens, partitionKey: key)
        manager.recordPrefillMeasurement(flops: 3.0e12, seconds: 1.5)
        manager.recordHydrationMeasurement(bytes: 4_000_000_000, seconds: 2.0)

        let snapshot = manager.makeTelemetrySnapshot()
        #expect(snapshot.counters.hitTokens == 25)
        #expect(snapshot.estimates.prefillFlopsPerSecond == 2.0e12)
        #expect(snapshot.estimates.hydrationBytesPerSecond == 2.0e9)
        #expect(snapshot.estimates.prefillSampleCount == 1)
        #expect(snapshot.estimates.hydrationSampleCount == 1)
    }
}
