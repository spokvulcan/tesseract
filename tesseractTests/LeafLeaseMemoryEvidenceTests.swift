import Darwin
import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// Bounded ownership/accounting evidence for #479. Run this suite alone for
/// process-memory observations; no model weights or private prompts are loaded.
@MainActor
struct LeafLeaseMemoryEvidenceTests {
    private struct SimulatedFailure: Error {}

    @Test func repeatedReturnsKeepOnlyTheExpectedSurvivalSet() async throws {
        let key = CachePartitionKey(modelID: "leaf-lease-evidence", kvBits: nil, kvGroupSize: 64)
        let store = TieredSnapshotStore(ssdConfig: nil)
        let manager = PrefixCacheManager(memoryBudgetBytes: 1_000_000, tieredStore: store)
        let tree = store.getOrCreateTree(for: key)
        let tokens = Array(1...8)
        var retiredLeases: [LeafLease] = []
        var rows: [[String: String]] = []
        var accessAllocationBytes = 0
        var lastMemory: RequestMemoryTelemetry?

        func record(_ stage: String, iteration: Int, requestID: UUID) {
            guard iteration < 3 || iteration == 23 else { return }
            let sample = RequestMemoryTelemetry.Sample.current()
            var facts = manager.memoryTelemetryFacts()
            facts["stage"] = stage
            facts["iteration"] = "\(iteration)"
            facts["requestID"] = requestID.uuidString
            facts["activeMlxBytes"] = "\(sample.activeBytes)"
            facts["cachedMlxBytes"] = "\(sample.cacheBytes)"
            facts["processFootprintBytes"] = sample.footprintBytes.map(String.init)
            facts["systemSwapUsedBytes"] = sample.systemSwapUsedBytes.map(String.init)
            rows.append(facts)
        }

        for iteration in 0..<24 {
            let requestID = UUID()
            let context = PrefixCacheDiagnostics.Context(
                requestID: requestID, modelID: key.modelID, kvBits: nil, kvGroupSize: 64)
            let memory =
                iteration < 3 || iteration == 23 ? RequestMemoryTelemetry(context: context) : nil
            let outcome = ["completed", "cancelled", "failed"][iteration % 3]
            weak var attention: KVCacheSimple?
            weak var recurrent: MambaCache?
            weak var array: MLXArray?
            do {
                let kv = KVCacheSimple()
                kv.state = [MLXArray.ones([1, 1, 8, 64]), MLXArray.ones([1, 1, 8, 64])]
                let state = MambaCache()
                state.state = [MLXArray.ones([8]), MLXArray.ones([8])]
                attention = kv
                recurrent = state
                var live: [any KVCache] = [kv, state]
                eval(live)
                let body = try #require(HybridCacheSnapshot.captureMoving(cache: &live, offset: 8))
                #expect(live.isEmpty)
                array = body.layers[0].state[0]
                let addresses = body.layers.flatMap(\.state).map(backingAddress)
                manager.admit(
                    try #require(
                        SnapshotAdmission.leaf(
                            storedTokens: tokens, snapshot: body, storage: .ramOnly,
                            partitionKey: key)))
                let node = try #require(
                    tree.findBestSnapshot(tokens: tokens, updateAccess: false)?.node)
                accessAllocationBytes = malloc_size(
                    Unmanaged.passUnretained(node.bodyAccess).toOpaque())
                memory?.mark(.restored, facts: manager.memoryTelemetryFacts())
                record("beforeLease", iteration: iteration, requestID: requestID)
                let lease = try #require(tree.beginLeafLease(on: node, context: context))
                retiredLeases.append(lease)
                memory?.mark(.decoding, facts: manager.memoryTelemetryFacts())
                record("leased", iteration: iteration, requestID: requestID)
                do {
                    defer {
                        #expect(
                            tree.endLeafLease(
                                lease, on: node, returning: body, tokens: tokens,
                                reason: outcome == "completed" ? .checkIn : .rewind))
                    }
                    manager.pinRestorePath(node: node, requestID: requestID)
                    manager.setMemoryBudget(0)
                    #expect(manager.clearRAMTier() == 0)
                    #expect(tree.totalSnapshotBytes == 4_160)
                    #expect(tree.dropBody(node: node).effect == .ignored(.leased))
                    #expect(
                        node.state.body?.layers.flatMap(\.state).map(backingAddress) == addresses)
                    if outcome == "cancelled" { throw CancellationError() }
                    if outcome == "failed" { throw SimulatedFailure() }
                } catch is CancellationError {
                    memory?.recordCancellationSignal(origin: "caller")
                } catch is SimulatedFailure {}
                manager.completeRequest(requestID: requestID)
                #expect(tree.leaseCount == 0)
                #expect(tree.leasedBytes == 0)
                #expect(
                    manager.budgetFloorBytes() == 4_160,
                    "the newest leaf remains a legitimate floor member")
                memory?.mark(.releasingRequest, facts: manager.memoryTelemetryFacts())
                record("returned", iteration: iteration, requestID: requestID)
            }
            #expect(manager.clearRAMTier() == 4_160)
            #expect(attention == nil)
            #expect(recurrent == nil)
            #expect(array == nil)
            #expect(tree.totalSnapshotBytes == 0)
            #expect(manager.budgetFloorBytes() == 0)
            record("cleared", iteration: iteration, requestID: requestID)
            memory?.mark(.releasingRequest, facts: manager.memoryTelemetryFacts())
            memory?.finish(outcome: outcome)
            lastMemory = memory
        }
        // Recorders and all retired lease tokens remain alive during the
        // weak-reference assertions above; neither can keep cache objects alive.
        #expect(retiredLeases.count == 24)
        await lastMemory?.sampleAfterRelease()
        let report: [String: Any] = [
            "iterations": 24, "rows": rows,
            "leaseTokenStrideBytes": MemoryLayout<LeafLease>.stride,
            "accessObjectAllocationBytes": accessAllocationBytes,
            "cacheObjectAndArrayReferencesReleased": true,
        ]
        let data = try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys])
        print("LEAF_LEASE_EVIDENCE=" + (try #require(String(data: data, encoding: .utf8))))
    }
}
