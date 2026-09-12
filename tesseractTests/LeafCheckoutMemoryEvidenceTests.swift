import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// Run alone for process measurements. The attention body is about 2 MiB;
/// no model weights, private prompts, or real long-context replay are loaded.
@MainActor
struct LeafCheckoutMemoryEvidenceTests {
    private struct SimulatedFailure: Error {}

    @Test func repeatedTurnsKeepOneAttentionOwnerAndReleaseRewindState() async throws {
        let key = CachePartitionKey(modelID: "checkout-evidence", kvBits: nil, kvGroupSize: 64)
        let store = TieredSnapshotStore(ssdConfig: nil)
        let manager = PrefixCacheManager(memoryBudgetBytes: 8_000_000, tieredStore: store)
        var tokens = Array(0..<4096)
        var retiredRequests: [FinalGenerationCache] = []
        var rows: [[String: String]] = []
        weak var attention: KVCacheSimple?
        var copiedRestoreAllocation = 0
        do {
            let kv = KVCacheSimple()
            kv.state = [MLXArray.ones([1, 1, 4096, 64]), MLXArray.ones([1, 1, 4096, 64])]
            let recurrent = MambaCache()
            recurrent.state = [MLXArray.ones([8]), MLXArray.ones([8])]
            let seed = FinalGenerationCache([kv, recurrent])
            let body = try #require(seed.moveSnapshot(offset: tokens.count))
            attention = kv
            manager.admit(
                try #require(
                    SnapshotAdmission.leaf(
                        storedTokens: tokens, snapshot: body, storage: .ramOnly, partitionKey: key))
            )
            let beforeCopy = Memory.activeMemory
            let copy = try body.restore()
            copiedRestoreAllocation = Memory.activeMemory - beforeCopy
            #expect(backingAddress(copy[0].state[0]) != backingAddress(kv.state[0]))
            #expect(copiedRestoreAllocation >= 2_097_152)
        }

        func record(_ stage: String, iteration: Int, requestID: UUID, rewindBytes: Int) {
            let sample = RequestMemoryTelemetry.Sample.current()
            var facts = manager.memoryTelemetryFacts()
            facts.merge([
                "stage": stage, "iteration": "\(iteration)", "requestID": requestID.uuidString,
                "activeMlxBytes": "\(sample.activeBytes)", "cachedMlxBytes": "\(sample.cacheBytes)",
                "recurrentRewindStateBytes": "\(rewindBytes)",
            ]) { _, new in new }
            facts["processFootprintBytes"] = sample.footprintBytes.map(String.init)
            facts["systemSwapUsedBytes"] = sample.systemSwapUsedBytes.map(String.init)
            rows.append(facts)
        }

        for iteration in 0..<24 {
            let requestID = UUID()
            let context = PrefixCacheDiagnostics.Context(
                requestID: requestID, modelID: key.modelID, kvBits: nil, kvGroupSize: 64)
            let memory = RequestMemoryTelemetry(context: context)
            let outcome = ["completed", "cancelled", "failed"][iteration % 3]
            let requested = tokens + [5000 + iteration, 6000 + iteration]
            record("beforeCheckout", iteration: iteration, requestID: requestID, rewindBytes: 0)
            memory.mark(.restoring, facts: manager.memoryTelemetryFacts())
            let beforeMove = Memory.activeMemory
            let request = try #require(
                await LeafCheckout.attempt(
                    resolved: .init(
                        lookup: manager.lookup(tokens: requested, partitionKey: key),
                        hydratedFromSSD: false, hydrationSeconds: 0),
                    tokens: requested, maximumAdvance: 10, identityKeySpace: true,
                    prefixCache: manager, context: context
                ).owner)
            #expect(request.cache[0] as AnyObject === attention)
            #expect(request.rewindStateBytes == 64)
            #expect(
                Memory.activeMemory - beforeMove < 4096,
                "checkout allocates only recurrent rewind state")
            memory.mark(
                .restored, facts: ["restoreMode": "handoff", "recurrentRewindStateBytes": "64"])
            record("checkedOut", iteration: iteration, requestID: requestID, rewindBytes: 64)
            do {
                _ = request.cache[0].update(
                    keys: MLXArray.ones([1, 1, 2, 64]), values: MLXArray.ones([1, 1, 2, 64]))
                let recurrent = try #require(request.cache[1] as? MambaCache)
                recurrent.state = [MLXArray.ones([8]) * 9, MLXArray.ones([8]) * 10]
                eval(request.cache)
                if outcome == "cancelled" { throw CancellationError() }
                if outcome == "failed" { throw SimulatedFailure() }
                let body = try #require(request.moveSnapshot(offset: requested.count))
                #expect(await request.checkIn(body, tokens: requested))
                tokens = requested
            } catch is CancellationError {
                memory.recordCancellationSignal(origin: "caller")
                await request.rewindIfNeeded(memory: memory)
            } catch is SimulatedFailure {
                await request.rewindIfNeeded(memory: memory)
            }
            #expect(request.cache.isEmpty)
            #expect(request.rewindStateBytes == 0)
            #expect(manager.memoryTelemetryFacts()["treeLeaseCount"] == "0")
            #expect(manager.totalSnapshotBytes == tokens.count * 512 + 64)
            retiredRequests.append(request)
            memory.mark(.releasingRequest, facts: manager.memoryTelemetryFacts())
            memory.finish(outcome: outcome)
            record("returned", iteration: iteration, requestID: requestID, rewindBytes: 0)
        }
        let retainedBytes = manager.totalSnapshotBytes
        #expect(manager.clearRAMTier() == retainedBytes)
        #expect(attention == nil, "retired requests must not retain the attention body")
        #expect(retiredRequests.allSatisfy { $0.cache.isEmpty && $0.checkout == nil })
        let report: [String: Any] = [
            "iterations": 24, "rows": rows, "copiedRestoreAllocationBytes": copiedRestoreAllocation,
            "initialAttentionBytes": 2_097_152, "recurrentRewindStateBytes": 64,
            "retainedLeafBytesBeforeClear": retainedBytes,
            "retiredRequestsRetainNoCache": true,
        ]
        let data = try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys])
        print("LEAF_CHECKOUT_EVIDENCE=" + (try #require(String(data: data, encoding: .utf8))))
    }
}
